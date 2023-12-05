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

grid_maxsteps = {'box': 500, 
                 'nav': 500}

class RolloutBuffer:
    
    def __init__(self, obs_dim = 2, act_dim = 2, max_horizon = 20, num_envs = 100, gae_lambda = 1.0, gamma = 0.99):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = max_horizon        
        self.n_envs = num_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma  
        self.reset()
    
    def swap_and_flatten(self, arr):
        
        shape = arr.shape
        
        if len(shape) < 3:
            shape = shape + (1,)
        
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    
    def reset(self):
        
        self.observations = np.zeros((self.horizon, self.n_envs, self.obs_dim), dtype=np.float32)
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

    def add(self, obs, action, reward, episode_start, value, log_prob):
                
        self.observations[self.ptr] = np.array(obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.episode_starts[self.ptr] = np.array(episode_start).copy()
        self.values[self.ptr] = np.array(value).copy().flatten()
        self.log_probs[self.ptr] = np.array(log_prob).copy()
        
        self.ptr += 1
        if self.ptr == self.horizon:
            self.full = True
    
    # Shrink the numpy matrices and clear out those at absorbing state
    def shrink(self, tensor_names):
        
        bool_indices = (self.observations[:, -2] == 0) # Not in absorbing states
        
        for tensor in tensor_names:
            self.__dict__[tensor] = self.__dict__[tensor][bool_indices]
    
    def get(self, batch_size = None):
        
        assert self.full, ""
                
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values", 
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            
            self.shrink(_tensor_names)            
            self.generator_ready = True
        
        num_samples = len(self.observations)
        indices = np.random.permutation(num_samples)

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = num_samples

        start_idx = 0
        while start_idx < num_samples:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
                
        batch_data = dict(observations=self.observations[batch_inds], 
                          actions=self.actions[batch_inds], 
                          values=self.values[batch_inds].flatten(), 
                          log_probs=self.log_probs[batch_inds].flatten(), 
                          advantages=self.advantages[batch_inds].flatten(), 
                          returns=self.returns[batch_inds].flatten())
        
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch_data.items()}

def sample_batch_trajs(env, pi_nw, v_nw=None, classifier_nw=None, rollout_buffer=None, batchsize = 256, train=False):
    
    if train:
        assert v_nw is not None and rollout_buffer is not None
        rollout_buffer.reset()
    
    gamma = env.gamma
    horizon = env.max_horizon
    X = env.init_state.clone().repeat(batchsize, 1)
    
    batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, batch_ts_c_hat = [], [], [], [], []
    batch_J = torch.zeros(batchsize).to(device)
    last_episode_starts = np.ones(batchsize, dtype=bool)
    
    for h in range(horizon):
        
        actions, log_probs = pi_nw.get_action(X, reparam = True)
        
        if train:
            if classifier_nw is not None:
                c_hat = classifier_nw(X, actions)
            
            with torch.no_grad():
                values = v_nw(X)
            
            last_X_np = X.detach().cpu().numpy()
        
        batch_ts_s.append(X.clone())        
                
        X, r, c = env.step(X, actions, reparam = True)
        
        batch_J += (r * (gamma**h))
        batch_ts_a.append(actions)
        batch_ts_r.append(r)
        batch_ts_c.append(c)
        
        if train:
            if classifier_nw is not None:
                batch_ts_c_hat.append(c_hat)
            
            actions_np = actions.detach().cpu().numpy()
            r_np = r.detach().cpu().numpy()
            values_np = values.detach().cpu().numpy()
            log_probs_np = log_probs.detach().cpu().numpy()            
            
            rollout_buffer.add(last_X_np, actions_np, r_np, last_episode_starts, values_np, log_probs_np)
        
        if h == horizon - 1:
            last_episode_starts = np.ones(batchsize, dtype = bool)
        else:
            last_episode_starts = np.zeros(batchsize, dtype = bool)
    
    # Batch, Timestep, Features
    batch_ts_s = torch.stack(batch_ts_s, dim = 1)
    batch_ts_a = torch.stack(batch_ts_a, dim = 1)
    batch_ts_r = torch.stack(batch_ts_r, dim = 1)
    batch_ts_c = torch.stack(batch_ts_c, dim = 1)
    
    if train:
        with torch.no_grad():
            # Compute value for the last timestep
            values = v_nw(X)
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=last_episode_starts)
        if classifier_nw is not None:
            batch_ts_c_hat = torch.cat(batch_ts_c_hat, dim = 1)
            return batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, batch_ts_c_hat, batch_J
        else:
            ts_lengths = torch.FloatTensor([horizon] * X.shape[0]).to(device)
            return batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, batch_J, ts_lengths
    else:
        return batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, batch_J
