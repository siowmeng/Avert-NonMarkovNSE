#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:53:55 2022

@author: siowmeng
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma

epsilon = 1e-8

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RolloutBuffer:
    
    def __init__(self, obs_dim = 2, act_dim = 2, max_horizon = 20, num_envs = 100, gae_lambda = 1.0, gamma = 0.99):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = max_horizon
        self.n_envs = num_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        # self.ptr, self.full, self.generator_ready = 0, False, False        
        self.reset()
    
    def swap_and_flatten(self, arr):
        
        shape = arr.shape
        
        if len(shape) < 3:
            shape = shape + (1,)
        
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def reset(self):
        
        self.observations = np.zeros((self.horizon, self.n_envs, self.obs_dim), dtype=np.float32)
        self.norm_observations = np.zeros((self.horizon, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.horizon, self.n_envs, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.horizon, self.n_envs), dtype=np.float32)
        
        self.ptr, self.full, self.generator_ready = 0, False, False
    
    def compute_returns_and_advantage(self, last_values, dones):
        
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.horizon)):
            if step == self.horizon - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values

    def add(self, obs, norm_obs, action, reward, episode_start, value, log_prob):
        
        self.observations[self.ptr] = np.array(obs).copy()
        self.norm_observations[self.ptr] = np.array(norm_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.episode_starts[self.ptr] = np.array(episode_start).copy()
        self.values[self.ptr] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.ptr] = log_prob.clone().cpu().numpy()
        
        self.ptr += 1
        if self.ptr == self.horizon:
            self.full = True
    
    def get(self, batch_size = None):
        
        assert self.full, ""
        
        indices = np.random.permutation(self.horizon * self.n_envs)
        
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "norm_observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.horizon * self.n_envs

        start_idx = 0
        while start_idx < self.horizon * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        
        batch_data = dict(observations=self.observations[batch_inds], 
                          norm_observations=self.norm_observations[batch_inds], 
                          actions=self.actions[batch_inds], 
                          values=self.values[batch_inds].flatten(), 
                          log_probs=self.log_probs[batch_inds].flatten(), 
                          advantages=self.advantages[batch_inds].flatten(), 
                          returns=self.returns[batch_inds].flatten())
        
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch_data.items()}

    # def sample_batch(self, batch_size=64):
    #     idxs = np.random.randint(0, self.size, size=batch_size)
    #     batch = dict(states=self.state_buf[idxs], 
    #                  norm_states=self.norm_state_buf[idxs], 
    #                  next_states=self.next_state_buf[idxs], 
    #                  norm_next_states=self.norm_next_state_buf[idxs], 
    #                  actions=self.action_buf[idxs], 
    #                  rewards=self.reward_buf[idxs], 
    #                  dones=self.done_buf[idxs])
    #     return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

class ReplayBuffer:

    def __init__(self, obs_dim = 2, act_dim = 2, size = 500):
        self.state_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.norm_state_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.norm_next_state_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, states, norm_states, actions, rewards, next_states, norm_next_states, dones):
        self.state_buf[self.ptr] = states
        self.norm_state_buf[self.ptr] = norm_states
        self.next_state_buf[self.ptr] = next_states
        self.norm_next_state_buf[self.ptr] = norm_next_states
        self.action_buf[self.ptr] = actions
        self.reward_buf[self.ptr] = rewards
        self.done_buf[self.ptr] = dones
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(states=self.state_buf[idxs], 
                     norm_states=self.norm_state_buf[idxs], 
                     next_states=self.next_state_buf[idxs], 
                     norm_next_states=self.norm_next_state_buf[idxs], 
                     actions=self.action_buf[idxs], 
                     rewards=self.reward_buf[idxs], 
                     dones=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

#polyak_update
def polyak_update(nw, targ_nw, polyak):
    with torch.no_grad():
        for p, p_targ in zip(nw.parameters(), targ_nw.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

def sample_init_states(init_states, n_samples):
    
    #return torch.FloatTensor(np.random.uniform([0.0, 20.0], [0.0, 20.0], (n_samples, 2))).to(device)
    return init_states.repeat(n_samples, 1)

def construct_rddl_domain_dict(domain, gamma, rddl_env):
    
    domain_dict = dict()
    domain_dict['domain_name'] = domain
    domain_dict['gamma'] = gamma
    domain_dict['horizon'] = rddl_env.horizon
    
    if domain.startswith('NAV'):
        
        domain_dict['action_label'] = 'move/1'
        domain_dict['state_label'] = 'location/1'
        domain_dict['num_states'] = rddl_env.observation_space[domain_dict['state_label']].shape[0]
        domain_dict['num_actions'] = rddl_env.action_space[domain_dict['action_label']].shape[0]
        
        domain_dict['dec_decays'] = rddl_env.non_fluents['DECELERATION_ZONE_DECAY/1']
        domain_dict['dec_centers'] = rddl_env.non_fluents['DECELERATION_ZONE_CENTER/2']
        domain_dict['move_var'] = rddl_env.non_fluents['MOVE_VARIANCE_MULT/1']
        domain_dict['goal'] = rddl_env.non_fluents['GOAL/1']
        
        domain_dict['min_actions'] = torch.tensor([-1.] * domain_dict['num_actions']).to(device)
        domain_dict['max_actions'] = torch.tensor([1.] * domain_dict['num_actions']).to(device)
        
        domain_dict['txn_distr'] = MultivariateNormal(torch.zeros(len(domain_dict['move_var'])).to(device), 
                                                      torch.diag(torch.FloatTensor(domain_dict['move_var'])).to(device))
        
        domain_dict['last_actv'] = nn.Tanh()
        
    elif domain.startswith('HVAC'):
        domain_dict['action_label'] = 'air/1'
        domain_dict['state_label'] = 'temp/1'
        domain_dict['num_states'] = rddl_env.observation_space[domain_dict['state_label']].shape[0]
        domain_dict['num_actions'] = rddl_env.action_space[domain_dict['action_label']].shape[0]
        domain_dict['air_max'] = torch.tensor(rddl_env.non_fluents['AIR_MAX/1']).to(device)
        domain_dict['is_room'] = torch.tensor(rddl_env.non_fluents['IS_ROOM/1']).to(device).float()
        domain_dict['cost_air'] = rddl_env.non_fluents['COST_AIR/0']
        domain_dict['temp_up'] = torch.tensor(rddl_env.non_fluents['TEMP_UP/1']).to(device)
        domain_dict['temp_low'] = torch.tensor(rddl_env.non_fluents['TEMP_LOW/1']).to(device)
        domain_dict['penalty'] = rddl_env.non_fluents['PENALTY/0']
        domain_dict['time_delta'] = rddl_env.non_fluents['TIME_DELTA/0']
        domain_dict['cap'] = torch.tensor(rddl_env.non_fluents['CAP/1']).to(device)
        domain_dict['cap_air'] = rddl_env.non_fluents['CAP_AIR/0']
        domain_dict['temp_air'] = rddl_env.non_fluents['TEMP_AIR/0']
        domain_dict['r_outside'] = torch.tensor(rddl_env.non_fluents['R_OUTSIDE/1']).to(device)
        domain_dict['r_hall'] = torch.tensor(rddl_env.non_fluents['R_HALL/1']).to(device)
        
        adj_matrix = rddl_env.non_fluents['ADJ/2']
        for i in range(adj_matrix.shape[0]):
            for j in range(i + 1, adj_matrix.shape[0]):
                adj_matrix[j, i] = adj_matrix[i, j]
        
        domain_dict['adj_matrix'] = torch.tensor(adj_matrix).to(device).float()
        domain_dict['r_wall_matrix'] = torch.tensor(rddl_env.non_fluents['R_WALL/2']).to(device)
        
        domain_dict['adj_out'] = torch.tensor(rddl_env.non_fluents['ADJ_OUTSIDE/1']).to(device).float()
        domain_dict['adj_hall'] = torch.tensor(rddl_env.non_fluents['ADJ_HALL/1']).to(device).float()
        
        domain_dict['temp_out_mean'] = torch.tensor(rddl_env.non_fluents['TEMP_OUTSIDE_MEAN/1']).to(device)
        domain_dict['temp_out_var'] = torch.tensor(rddl_env.non_fluents['TEMP_OUTSIDE_VARIANCE/1']).to(device)
        domain_dict['temp_hall_mean'] = torch.tensor(rddl_env.non_fluents['TEMP_HALL_MEAN/1']).to(device)
        domain_dict['temp_hall_var'] = torch.tensor(rddl_env.non_fluents['TEMP_HALL_VARIANCE/1']).to(device)
        
        domain_dict['min_actions'] = torch.tensor([0.] * domain_dict['num_actions']).to(device)
        domain_dict['max_actions'] = domain_dict['air_max']
        
        domain_dict['temp_out_distr'] = MultivariateNormal(domain_dict['temp_out_mean'], 
                                                           torch.diag(domain_dict['temp_out_var']))
        domain_dict['temp_hall_distr'] = MultivariateNormal(domain_dict['temp_hall_mean'], 
                                                           torch.diag(domain_dict['temp_hall_var']))
        
        diag_vars = domain_dict['adj_out'] * domain_dict['temp_out_var'] / domain_dict['r_outside']**2 + domain_dict['adj_hall'] * domain_dict['temp_hall_var'] / domain_dict['r_hall']**2
        domain_dict['txn_distr'] = MultivariateNormal(torch.zeros(len(domain_dict['temp_out_mean'])).to(device), 
                                                      torch.diag(diag_vars))
        
        domain_dict['last_actv'] = nn.Sigmoid()
        
    elif domain.startswith('RES'):
        domain_dict['action_label'] = 'outflow/1'
        domain_dict['state_label'] = 'rlevel/1'
        domain_dict['num_states'] = rddl_env.observation_space[domain_dict['state_label']].shape[0]
        domain_dict['num_actions'] = rddl_env.action_space[domain_dict['action_label']].shape[0]
        
        domain_dict['evap_frac'] = rddl_env.non_fluents['MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0']
        domain_dict['res_cap'] = torch.tensor(rddl_env.non_fluents['MAX_RES_CAP/1']).to(device)
        domain_dict['downstream_matrix'] = torch.tensor(rddl_env.non_fluents['DOWNSTREAM/2']).to(device)
        
        domain_dict['lower_bound'] = torch.tensor(rddl_env.non_fluents['LOWER_BOUND/1']).to(device)
        domain_dict['upper_bound'] = torch.tensor(rddl_env.non_fluents['UPPER_BOUND/1']).to(device)
        domain_dict['low_penalty'] = torch.tensor(rddl_env.non_fluents['LOW_PENALTY/1']).to(device)
        domain_dict['high_penalty'] = torch.tensor(rddl_env.non_fluents['HIGH_PENALTY/1']).to(device)
        domain_dict['rain_shape'] = torch.tensor(rddl_env.non_fluents['RAIN_SHAPE/1']).to(device)
        domain_dict['rain_rate'] = 1. / torch.tensor(rddl_env.non_fluents['RAIN_SCALE/1']).to(device)
        
        domain_dict['min_actions'] = torch.tensor([0.] * domain_dict['num_actions']).to(device)
        domain_dict['max_actions'] = None
                
        domain_dict['txn_distr'] = Gamma(domain_dict['rain_shape'], domain_dict['rain_rate'])
        
        domain_dict['last_actv'] = nn.Sigmoid()
    
    return domain_dict

def sample_policy_traj_noise(X0, domain_dict, pi_nw=None, s_rms=None, ret_rms=None, replay_buffer=None, pi_replay_buffer=None, action_noise_prop=1.0, min_obs=400, mode='train', explore=False, horizon_offset=0):
    
    if mode != 'pretrain':
        assert (pi_nw is not None) and (s_rms is not None) and (ret_rms is not None)
    if mode == 'train':
        assert replay_buffer is not None
    
    domain = domain_dict['domain_name']
    gamma = domain_dict['gamma']
    horizon = domain_dict['horizon'] + horizon_offset
    action_dim = domain_dict['num_actions']
    min_actions, max_actions = domain_dict['min_actions'], domain_dict['max_actions']
    
    if domain.startswith('NAV'):
        noise_std = 0.01**0.5 # Action Noise: 0.1 standard deviation
    elif domain.startswith('HVAC'):
        noise_std = 0.01**0.5 # Action Noise: 0.1 standard deviation
    elif domain.startswith('RES'):
        noise_std = 0.01**0.5 # Action Noise: 0.1 standard deviation
    
    X = torch.clone(X0)
    ts_s, ts_s_norm, ts_a, ts_r, ts_r_norm = [], [], [], [], []
    J, J_norm = torch.zeros(X.size(dim = 0)).to(device), torch.zeros(X.size(dim = 0)).to(device)
    a, r, r_norm = None, None, None
    prev_X_np, prev_Xnorm_np = None, None

    for h in range(horizon):
        
        if mode == 'pretrain':
            if domain.startswith('RES') and max_actions == None:
                a = (X - min_actions) * torch.rand(X.shape[0], action_dim).to(device) + min_actions
            else:
                a = (max_actions - min_actions) * torch.rand(X.shape[0], action_dim).to(device) + min_actions
        else:        
            # Training Samples: (1) Update S, J rolling means, (2) Persist into replay buffer, (3) exploratory action noise
            if mode == 'train':
                
                s_rms.update(X.detach().cpu().numpy())
                X_norm = (X - torch.tensor(s_rms.mean).to(device)) / torch.sqrt(torch.tensor(s_rms.var + epsilon).to(device))
                
                if h > 0:
                    prev_a_np = a.detach().cpu().numpy()
                    prev_r_np, prev_rnorm_np = r.detach().cpu().numpy(), r_norm.detach().cpu().numpy()
                    curr_X_np, curr_Xnorm_np = X.detach().cpu().numpy(), X_norm.detach().cpu().numpy()
                    if s_rms.count >= min_obs:
                        for i in range(prev_a_np.shape[0]):
                            replay_buffer.store(prev_X_np[i], prev_Xnorm_np[i], prev_a_np[i], prev_rnorm_np[i], curr_X_np[i], curr_Xnorm_np[i], False)
                            pi_replay_buffer.store(prev_X_np[i], prev_Xnorm_np[i], prev_a_np[i], prev_rnorm_np[i], curr_X_np[i], curr_Xnorm_np[i], False)
                
                if explore:
                    if domain.startswith('RES') and max_actions == None:
                        a = (X - min_actions) * torch.rand(X.shape[0], action_dim).to(device) + min_actions
                    else:
                        a = (max_actions - min_actions) * torch.rand(X.shape[0], action_dim).to(device) + min_actions
                else:
                    if domain.startswith('RES') and max_actions == None:
                        a = torch.clamp(pi_nw(X_norm, X) + torch.normal(0., noise_std * action_noise_prop, 
                                                                         size = (X.shape[0], action_dim)).to(device) * (X - min_actions), 
                                        min = min_actions, 
                                        max = X)
                    else:
                        a = torch.clamp(pi_nw(X_norm) + torch.normal(0., noise_std * action_noise_prop, 
                                                                     size = (X.shape[0], action_dim)).to(device) * (max_actions - min_actions), 
                                        min = min_actions, 
                                        max = max_actions)
            
            elif mode == 'test':
                X_norm = (X - torch.tensor(s_rms.mean).to(device)) / torch.sqrt(torch.tensor(s_rms.var + epsilon).to(device))
                if domain.startswith('RES') and max_actions == None:
                    a = pi_nw(X_norm, X)
                else:
                    a = pi_nw(X_norm)
            
            ts_s_norm.append(X_norm)
            prev_Xnorm_np = X_norm.detach().cpu().numpy()
        
        ts_s.append(X)        
        ts_a.append(a)                
        prev_X_np = X.detach().cpu().numpy()
        
        X, r = rddl_nextstate_reward_txn(domain_dict, X, a, ilbo_obj = False)
        
        ts_r.append(r)
        J += (r * (gamma**h))
        
        if mode != 'pretrain':
            
            if mode == 'train':
                ret_rms.update(J.detach().cpu().numpy())
            
            r_norm = r / torch.sqrt(torch.tensor(ret_rms.var + epsilon).to(device))
            J_norm += (r_norm * (gamma**h))
            ts_r_norm.append(r_norm)
            
    if mode == 'train':
        
        s_rms.update(X.detach().cpu().numpy())
        X_norm = (X - torch.tensor(s_rms.mean).to(device)) / torch.sqrt(torch.tensor(s_rms.var + epsilon).to(device))
        
        prev_a_np = a.detach().cpu().numpy()
        prev_r_np, prev_rnorm_np = r.detach().cpu().numpy(), r_norm.detach().cpu().numpy()
        curr_X_np, curr_Xnorm_np = X.detach().cpu().numpy(), X_norm.detach().cpu().numpy()
        if s_rms.count >= min_obs:
            for i in range(prev_a_np.shape[0]):
                replay_buffer.store(prev_X_np[i], prev_Xnorm_np[i], prev_a_np[i], prev_rnorm_np[i], curr_X_np[i], curr_Xnorm_np[i], False)
                pi_replay_buffer.store(prev_X_np[i], prev_Xnorm_np[i], prev_a_np[i], prev_rnorm_np[i], curr_X_np[i], curr_Xnorm_np[i], False)
        
    elif mode == 'test':
        X_norm = (X - torch.tensor(s_rms.mean).to(device)) / torch.sqrt(torch.tensor(s_rms.var + epsilon).to(device))
    
    ts_s = torch.stack(ts_s, dim = 1)
    ts_a = torch.stack(ts_a, dim = 1)
    ts_r = torch.stack(ts_r, dim = 1)
    ts_lengths = torch.FloatTensor([horizon] * X.shape[0]).to(device)
    
    if mode == 'pretrain':
        return ts_s, ts_a, ts_r, J, ts_lengths
    else:
        ts_s_norm = torch.stack(ts_s_norm, dim = 1)
        ts_r_norm = torch.stack(ts_r_norm, dim = 1)
        return ts_s, ts_s_norm, ts_a, ts_r, ts_r_norm, J, J_norm, ts_lengths

def sample_probpolicy_traj(X0, domain_dict, pi_nw, s_rms, ret_rms, v_nw, rollout_buffer=None, train=False, horizon_offset=0):
    
    if train:
        assert v_nw is not None and rollout_buffer is not None
    
    rollout_buffer.reset()
    
    domain = domain_dict['domain_name']
    gamma = domain_dict['gamma']
    horizon = domain_dict['horizon'] + horizon_offset
    min_actions, max_actions = domain_dict['min_actions'], domain_dict['max_actions']
    
    X = torch.clone(X0)
    ts_s, ts_s_norm, ts_a, ts_r, ts_r_norm = [], [], [], [], []
    J, Jnorm = torch.zeros(X.size(dim = 0)).to(device), torch.zeros(X.size(dim = 0)).to(device)
    last_episode_starts = np.ones((X.size(dim = 0), ), dtype = bool)

    for h in range(horizon):
        
        # Training Samples: (1) Update S, J rolling means, (2) Persist into rollout buffer
            
        last_X_np = X.detach().cpu().numpy()
        
        if train:
            s_rms.update(last_X_np)
            
        Xnorm = (X - torch.tensor(s_rms.mean).to(device)) / torch.sqrt(torch.tensor(s_rms.var + epsilon).to(device))
        last_Xnorm_np = Xnorm.detach().cpu().numpy()
        
        ts_s.append(X)
        ts_s_norm.append(Xnorm)
        
        actions, log_probs = pi_nw.get_action(Xnorm)
        
        with torch.no_grad():
            if train:
                values = v_nw(Xnorm)
        
        if domain.startswith('RES') and max_actions == None:
            clipped_actions = torch.clamp(actions, min = min_actions, max = X)
        else:
            clipped_actions = torch.clamp(actions, min = min_actions, max = max_actions)
        
        X, r = rddl_nextstate_reward_txn(domain_dict, X, clipped_actions, ilbo_obj = False)
        
        J += (r * (gamma**h))
        
        if train:
            ret_rms.update(J.detach().cpu().numpy())
        
        rnorm = r / torch.sqrt(torch.tensor(ret_rms.var + epsilon).to(device))
        Jnorm += (rnorm * (gamma**h))
        
        actions_np = actions.detach().cpu().numpy()
        # r_np = r.detach().cpu().numpy()
        rnorm_np = rnorm.detach().cpu().numpy()
        
        if train:
            rollout_buffer.add(last_X_np, last_Xnorm_np, actions_np, rnorm_np, last_episode_starts, values, log_probs.detach())
        
        if h == horizon - 1:
            last_episode_starts = np.ones((X.size(dim = 0), ), dtype = bool)
        else:
            last_episode_starts = np.zeros((X.size(dim = 0), ), dtype = bool)
        
        ts_a.append(clipped_actions)
        ts_r.append(r)
        ts_r_norm.append(rnorm)    
    
    if train:
        
        with torch.no_grad():
            # Compute value for the last timestep
            values = v_nw(Xnorm)
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=last_episode_starts)
        
    ts_s = torch.stack(ts_s, dim = 1)
    ts_s_norm = torch.stack(ts_s_norm, dim = 1)
    ts_a = torch.stack(ts_a, dim = 1)
    ts_r = torch.stack(ts_r, dim = 1)
    ts_r_norm = torch.stack(ts_r_norm, dim = 1)    
    ts_lengths = torch.FloatTensor([horizon] * X.shape[0]).to(device)
    
    return ts_s, ts_s_norm, ts_a, ts_r, ts_r_norm, J, Jnorm, ts_lengths

def rddl_nextstate_reward_txn(domain_dict, X, a, ilbo_obj=False):
    
    domain = domain_dict['domain_name']
    txn_noise_distr = domain_dict['txn_distr']
    
    if domain.startswith('NAV'):
        
        dec_decays = domain_dict['dec_decays']
        dec_centers = domain_dict['dec_centers']
        goal = domain_dict['goal']
        
        r = -(torch.norm(X - torch.tensor(goal).to(device), p = 2, dim = 1))
        
        total_dec = 1.0
        for decay_coef, decay_loc in zip(dec_decays, dec_centers):
            dist = torch.norm(X - torch.FloatTensor(decay_loc).to(device), dim = 1)
            total_dec *= (2.0 / (1.0 + torch.exp(-decay_coef * dist)) - 1.0)

        total_dec = total_dec[:, None]
        
        if ilbo_obj: # Mean prediction for ILBO objective
            X_txn_noise = 0
        else:
            X_txn_noise = txn_noise_distr.sample((X.shape[0],))
            
        X = X + total_dec * a + X_txn_noise
        
    elif domain.startswith('HVAC'):
        
        is_room = domain_dict['is_room']
        cost_air = domain_dict['cost_air']
        temp_up = domain_dict['temp_up']
        temp_low = domain_dict['temp_low']
        penalty = domain_dict['penalty']
        time_delta = domain_dict['time_delta']
        cap = domain_dict['cap']
        cap_air = domain_dict['cap_air']
        temp_air = domain_dict['temp_air']
        adj_matrix = domain_dict['adj_matrix']
        adj_out = domain_dict['adj_out']
        adj_hall = domain_dict['adj_hall']
        r_outside = domain_dict['r_outside']
        r_hall = domain_dict['r_hall']
        r_wall_matrix = domain_dict['r_wall_matrix']
        temp_out_distr = domain_dict['temp_out_distr']
        temp_hall_distr = domain_dict['temp_hall_distr']
        temp_out_mean = domain_dict['temp_out_mean']
        temp_hall_mean = domain_dict['temp_hall_mean']
        
        out_of_range = (X < temp_low) | (X > temp_up)
        r = -torch.sum(is_room * (a * cost_air + out_of_range * penalty) + 10.0 * torch.abs((temp_up + temp_low) / 2.0 - X), 
                       dim = 1)
        
        if ilbo_obj: # Mean prediction for ILBO objective
            temp_out = temp_out_mean
            temp_hall = temp_hall_mean
        else:
            temp_out = temp_out_distr.sample((X.shape[0],))
            temp_hall = temp_hall_distr.sample((X.shape[0],))
        
        X = X + time_delta / cap * \
                (a * cap_air * (temp_air - X) * is_room \
                     + torch.matmul(X, adj_matrix / r_wall_matrix) \
                     - X * torch.sum(adj_matrix / r_wall_matrix, dim = 1) \
                     + adj_out * (temp_out - X) / r_outside \
                     + adj_hall * (temp_hall - X)  / r_hall)
    
    elif domain.startswith('RES'):
        
        evap_frac = domain_dict['evap_frac']
        res_cap = domain_dict['res_cap']
        downstream_matrix = domain_dict['downstream_matrix']
        lower_bound = domain_dict['lower_bound']
        upper_bound = domain_dict['upper_bound']
        low_penalty = domain_dict['low_penalty']
        high_penalty = domain_dict['high_penalty']
        rainfall_distr = domain_dict['txn_distr']

        r = torch.sum((X < lower_bound) * (lower_bound - X) * low_penalty + (X > upper_bound) * (X - upper_bound) * high_penalty, 
                      dim = 1)
        
        evaporated = evap_frac * (X / res_cap)**2 * X
        overflow = nn.ReLU()(X - a - res_cap)
        inflow = torch.matmul(a + overflow, downstream_matrix.float())
        
        level_bef_rainfall = X - evaporated - a - overflow + inflow
        
        if ilbo_obj:
            X = level_bef_rainfall
        else:
            rainfall = rainfall_distr.sample((X.shape[0],))
            X = nn.ReLU()(level_bef_rainfall + rainfall)
    
    return X, r