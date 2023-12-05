#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import torch
import torch.nn as nn
import numpy as np
import time
import rddlgym
#import matplotlib.pyplot as plt
import utils, models, losses, nse, log
from copy import deepcopy
#from collections import OrderedDict
from itertools import chain
from decimal import Decimal
from rddl_NSEClassify import TSGRU, TSLSTM
from rddl_GenNSESamples import rddl_dict

ts_model_dict = {'GRU' : TSGRU, 
                 'LSTM': TSLSTM}

def test_policy(rddl_env, pi_network, num_test_trajs, domain_dict, s_rms, ret_rms, v_network = None, rollout_buffer=None):
    
    dict_discRets = {'min': 0.0, 
                     'mean': 0.0, 
                     'median': 0.0, 
                     'max': 0.0}
    dict_undiscRets = {'min': 0.0, 
                     'mean': 0.0, 
                     'median': 0.0, 
                     'max': 0.0}
        
    state, t = rddl_env.reset()
    X0_test = utils.sample_init_states(torch.tensor(state[domain_dict['state_label']]).to(utils.device), num_test_trajs)
    
    if isinstance(pi_network, models.PolicyNW):
        batch_ts_s, batch_ts_s_norm, batch_ts_a, batch_ts_r, batch_ts_r_norm, J_batch, J_norm_batch, batch_ts_lengths = utils.sample_policy_traj_noise(X0_test, domain_dict, pi_network, s_rms, ret_rms, mode='test')
    elif isinstance(pi_network, models.DiagGaussPolicyNW):
        batch_ts_s, batch_ts_s_norm, batch_ts_a, batch_ts_r, batch_ts_r_norm, J_batch, J_norm_batch, batch_ts_lengths = utils.sample_probpolicy_traj(X0_test, domain_dict, pi_network, s_rms, ret_rms, v_network, rollout_buffer=rollout_buffer, train=False)
    
    dict_discRets['min'] = torch.min(J_batch).item()
    dict_discRets['mean'] = torch.mean(J_batch).item()
    dict_discRets['median'] = torch.median(J_batch).item()
    dict_discRets['max'] = torch.max(J_batch).item()
    
    J_undisc_batch = torch.sum(batch_ts_r, dim = 1)
    
    dict_undiscRets['min'] = torch.min(J_undisc_batch).item()
    dict_undiscRets['mean'] = torch.mean(J_undisc_batch).item()
    dict_undiscRets['median'] = torch.median(J_undisc_batch).item()
    dict_undiscRets['max'] = torch.max(J_undisc_batch).item()
    
    batch_ts_danger_dict = nse.batch_ts_danger(domain_dict, batch_ts_s, batch_ts_a)
    dict_TrajSets, prop_NSEs = nse.batch_trajDF_by_nse(domain_dict, batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_danger_dict)
    
    return dict_TrajSets, prop_NSEs, dict_discRets, dict_undiscRets, batch_ts_s, batch_ts_a, batch_ts_r

if __name__ == "__main__":
    
    try:
        ptFilePath = sys.argv[1]
        domain, modelType, hidden_units, batch_size = ptFilePath.split("/")[-1].split("_")
        hidden_units, batch_size = int(hidden_units), int(batch_size.split(".pt")[0])
        logFilePath = sys.argv[2]
        if logFilePath[-1] != '/':
            logFilePath += '/'
        actorObj = sys.argv[3] # TD3 or ILBO
        assert actorObj == 'TD3' or actorObj == 'PPO' or actorObj == 'ILBO'
        constrOptMode = sys.argv[4] # RCPO or NIL
        assert constrOptMode == 'RCPO' or constrOptMode == 'RCPO-MF' or constrOptMode == 'NIL'
        init_lambda = float(sys.argv[5]) # Initial value of lambda for all constraints
        lambda_lr = float(sys.argv[6])
        seed = int(sys.argv[7])
        writeTrajBool = True if (len(sys.argv) > 8) else False
        if writeTrajBool:
            writeTrajPath = sys.argv[8]
            if writeTrajPath[-1] != '/':
                writeTrajPath += '/'
    except:
        print("Invalid parameters specified")
        sys.exit(1)
       
    if domain in rddl_dict:
        if domain.startswith('NAV'):
            feature_dim = 4
        else:
            feature_dim = int(re.findall(r'\d+', domain)[-1]) * 2
    else:
        print("Domain specified in filepath not recognized")
        sys.exit(1)
    
    # (index of class - 0 to 2, GE or LE, threshold value)
    constr_dict = {'NAV3': [(0, 'GE', 0.95)], 
                   'HVAC6': [(0, 'GE', 0.95)], 
                   'RES20': [(0, 'GE', 0.8)]}
    
    tau = 0.995
    tau_lambda = 1.0 - lambda_lr
    # tau_lambda = 0.995
    batchsize = 100 #128
    num_epoch = {'NAV3': 5000, 
                 'HVAC6': 5000, 
                 'RES20': 10000}
    test_interval = 25
    num_test_trajs = 100 #128
    gamma = 0.99
    n_critics = 2
    lambdas_reset_interval = 100
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_printoptions(precision = 4)
    
    base_logfile_path = log.init_log_folder(logFilePath, domain, ptFilePath, actorObj, constrOptMode, init_lambda, 
                                            seed, constr_dict[domain], lambda_lr)
    
    # create RDDLGYM environment
    rddl_id = rddl_dict[domain]
    env = rddlgym.make(rddl_id, mode=rddlgym.GYM)
    
    # Load RDDL Domain Info
    domain_dict = utils.construct_rddl_domain_dict(domain, gamma, env)    
    
    # Load Corresponding Classifier
    classifier = ts_model_dict[modelType](feature_dim=feature_dim, nb_gru_units=hidden_units, batch_size=batch_size).to(utils.device)
    classifier.load_state_dict(torch.load(ptFilePath, map_location=utils.device))
    
    # Freeze classifier param
    for param in classifier.parameters():
        param.requires_grad_(False)
    
    classifier.eval()
    
    if actorObj == 'PPO':
        
        # Hyperparameters
        ppo_clip_range = 0.2
        
        # Value Net and Policy Net
        value_net = models.ValueNW(input_dim=domain_dict['num_states'], output_dim=1, hidden_layers=[64, 64]).to(utils.device)
        policy_net = models.DiagGaussPolicyNW(input_dim=domain_dict['num_states'], output_dim=domain_dict['num_actions'], hidden_layers=[64, 64]).to(utils.device)
        
        # Optimizer
        value_net_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
        policy_net_optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)

        # Rollout buffer
        rollout_buffer = utils.RolloutBuffer(obs_dim=domain_dict['num_states'], act_dim=domain_dict['num_actions'], max_horizon = domain_dict['horizon'], num_envs = batchsize, gae_lambda = 1.0, gamma = gamma)
        
        l_Mean_VLoss, l_Mean_PgLoss, l_Mean_EntLoss, l_Mean_CLoss = [], [], [], []
        
    else:        
        qvalue_nets, qvalue_nets_targ = [], []
        for i in range(n_critics):
            critics_model = models.QValueNW(input_dim=domain_dict['num_states'], output_dim=1, hidden_layers=[64, 64]).to(utils.device)            
            qvalue_nets.append(critics_model)
            qvalue_nets_targ.append(deepcopy(critics_model))
    
        policy_net = models.PolicyNW(input_dim=domain_dict['num_states'], output_dim=domain_dict['num_actions'], hidden_layers=[64, 64], last_actv=domain_dict['last_actv'], action_bound = domain_dict['max_actions']).to(utils.device)
        policy_net_targ = deepcopy(policy_net)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for net in qvalue_nets_targ:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
        for p in policy_net_targ.parameters():
            p.requires_grad = False
        policy_net_targ.eval()
        
        # Optimizer
        q_chain_optimizer = torch.optim.Adam(chain(*[net.parameters() for net in qvalue_nets]), lr=2e-4)
        pi_constr_optimizer = torch.optim.Adam(policy_net.parameters(), lr=2e-4)

        # Experience buffer
        replay_buffer_constr = utils.ReplayBuffer(obs_dim=domain_dict['num_states'], act_dim=domain_dict['num_actions'], size=1048576)
        pi_replay_buffer_constr = utils.ReplayBuffer(obs_dim=domain_dict['num_states'], act_dim=domain_dict['num_actions'], size=16384)
        
        l_Mean_QLoss, l_Mean_JLoss, l_Mean_LaLoss, l_Mean_RegLoss = [], [], [], []
        l_Mean_J, l_Median_J, l_Min_J, l_Max_J = [], [], [], []
        l_Mean_QGrad, l_Median_QGrad, l_Min_QGrad, l_Max_QGrad = [], [], [], []
        l_Mean_PiGrad, l_Median_PiGrad, l_Min_PiGrad, l_Max_PiGrad = [], [], [], []
    
    # Start learning after ten batches of sample trajectories collected
    min_buffer_samples = batchsize * domain_dict['horizon'] * 10
    # Running mean and std for state and return
    state_rms_constr = utils.RunningMeanStd(shape = domain_dict['num_states'])
    return_rms_constr = utils.RunningMeanStd(shape = ())
    
    if constrOptMode == 'RCPO' or constrOptMode == 'RCPO-MF':
        log_lambda_vec = torch.tensor([np.log(init_lambda)] * len(constr_dict[domain_dict['domain_name']]), 
                                      requires_grad=True, 
                                      device=utils.device)
        lambda_vec = torch.exp(log_lambda_vec)
        
        log_lambda_constr_optimizer = torch.optim.Adam([log_lambda_vec], lr=lambda_lr)        
    else:
        lambda_vec = torch.tensor([0.] * len(constr_dict[domain_dict['domain_name']]), 
                                  requires_grad=False, 
                                  device=utils.device)
    
    # cudnn does not support backward operations during classifier eval, thus has to be turned off
    torch.backends.cudnn.enabled = False

    start = time.time()
    
    l_lambdas = []
    l_Mean_J, l_Median_J, l_Min_J, l_Max_J = [], [], [], []
    lTest_Prop_NSE0, lTest_Prop_NSE1, lTest_Prop_NSE2 = [], [], []
    lTest_Min_DiscRet, lTest_Mean_DiscRet, lTest_Median_DiscRet, lTest_Max_DiscRet = [], [], [], []
    lTest_Min_UndiscRet, lTest_Mean_UndiscRet, lTest_Median_UndiscRet, lTest_Max_UndiscRet = [], [], [], []
    lTest_T_Min_DiscRet, lTest_T_Mean_DiscRet, lTest_T_Median_DiscRet, lTest_T_Max_DiscRet = [], [], [], []
    trajDFs_by_nse = {0: [], 
                      1: [], 
                      2: []}

    for lambda_epoch in range(num_epoch[domain]):
        
        l_lambdas.append(lambda_vec.tolist())
        print("Lambdas = ", lambda_vec)
        
        state, t = env.reset()
        
        X0_tensor = utils.sample_init_states(torch.tensor(state[domain_dict['state_label']]).to(utils.device), batchsize)
        
        if actorObj == 'PPO':
            
            learnBool = False if lambda_epoch < 10 else True # Start learning
            
            value_net.eval()
            policy_net.eval()
            
            batch_ts_s, batch_ts_s_norm, batch_ts_a, batch_ts_r, batch_ts_r_norm, J_batch, J_norm_batch, batch_ts_lengths = utils.sample_probpolicy_traj(X0_tensor, domain_dict, policy_net, state_rms_constr, return_rms_constr, value_net, rollout_buffer=rollout_buffer, train=True)
                        
            batch_ts_sa = torch.concat((batch_ts_s, batch_ts_a), dim=2)
            class_outputs = classifier(batch_ts_sa, batch_ts_lengths)
            
            PgLosses, VLosses, EntLosses, CLosses = [], [], [], []
            ppo_clip_fractions = []
            
            if learnBool:
                
                value_net.train()
                policy_net.train()
                
                entropy_strength = max(0, 1 - lambda_epoch / num_epoch[domain] * 4) * 0.001
                
                PiGrads, LambdaPiGrads = 0, 0
                
                with torch.no_grad():
                    lambda_sum = lambda_vec.sum().item()
                
                if (len(constr_dict[domain_dict['domain_name']]) > 0):
                    
                    if constrOptMode == 'RCPO':
                        
                        # RCPO-style of gradient update of lambda
                        LaLoss, means_satifs = losses.pi_rcpo_lambda_loss(class_outputs, lambda_vec, constr_dict[domain_dict['domain_name']])
                        LaLoss = -LaLoss / (1 + lambda_sum)
                        policy_net_optimizer.zero_grad()
                        log_lambda_constr_optimizer.zero_grad()
                        LaLoss.backward()
                        CLosses.append(LaLoss.item())
                        
                        if log_lambda_vec.grad is not None:
                            log_lambda_vec.grad.mul_(-(1 + lambda_sum))
                        
                        torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()), 0.5)
                        torch.nn.utils.clip_grad_norm_(list(log_lambda_vec), 0.001)
                        
                        LambdaPiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                        
                        policy_net_optimizer.step()
                        log_lambda_constr_optimizer.step()
                        
                        lambda_vec = torch.exp(log_lambda_vec)
                    
                    elif constrOptMode == 'RCPO-MF':
                        
                        sum_log_prob = torch.zeros((batch_ts_s_norm.shape[0])).to(utils.device)
                        for i in range(batch_ts_s_norm.shape[1]):
                            log_prob, _ = policy_net.eval_action(batch_ts_s_norm[:, i, :], batch_ts_a[:, i, :])
                            sum_log_prob += log_prob
                        
                        class_outputs_nograd = class_outputs.detach()
                        lambda_vec_nograd = lambda_vec.detach()
                        
                        LaLoss = losses.pi_rcpo_mf_lambda_loss(sum_log_prob, class_outputs_nograd, lambda_vec_nograd, constr_dict[domain_dict['domain_name']])
                        
                        LaLoss = -LaLoss / (1 + lambda_sum)
                        policy_net_optimizer.zero_grad()
                        LaLoss.backward()
                        CLosses.append(LaLoss.item())
                        
                        torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()), 0.5)                        
                        policy_net_optimizer.step()
                        
                        LambdaPiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                        
                        # No need to reverse sign of LamLoss since we are minimizing wrt lambda
                        LamLoss, means_satifs = losses.pi_rcpo_lambda_loss(class_outputs_nograd, lambda_vec, constr_dict[domain_dict['domain_name']])
                        
                        log_lambda_constr_optimizer.zero_grad()
                        LamLoss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(list(log_lambda_vec), 0.001)
                        log_lambda_constr_optimizer.step()                        
                        
                        lambda_vec = torch.exp(log_lambda_vec)
                
                num_rollout_batches = len(list(rollout_buffer.get(batchsize)))
                
                for rollout_data in rollout_buffer.get(batch_size):
                    
                    actions = rollout_data['actions']
                    
                    log_prob, entropy = policy_net.eval_action(rollout_data['norm_observations'], actions)
                    values = value_net(rollout_data['norm_observations'])
                    
                    advantages = rollout_data['advantages']
                    
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + utils.epsilon)
                    
                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data['log_probs'])
    
                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - ppo_clip_range, 1 + ppo_clip_range)
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
                    # Logging
                    PgLosses.append(policy_loss.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > ppo_clip_range).float()).item()
                    ppo_clip_fractions.append(clip_fraction)
                    
                    values_pred = values
                    
                    # Value loss using the TD(gae_lambda) target
                    value_loss = torch.nn.MSELoss()(rollout_data['returns'], values_pred.flatten())
                    VLosses.append(value_loss.item())
                    
                    entropy_loss = -torch.mean(entropy)
                    EntLosses.append(entropy_loss.item())
                    
                    loss = ((policy_loss + entropy_strength * entropy_loss) / (1 + lambda_sum) + 0.5 * value_loss) / num_rollout_batches
    
                    # Optimization step
                    policy_net_optimizer.zero_grad()
                    value_net_optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()) + list(value_net.parameters()), 0.5)
                    
                    PiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                    
                    policy_net_optimizer.step()
                    value_net_optimizer.step()
                
                # print statistics
                print('[%d] Average V loss: %.6f' % (lambda_epoch + 1, np.mean(VLosses)))
                print('       Average Pg loss: %.6f' % np.mean(PgLosses))
                print('       Average Entropy loss: %.6f' % np.mean(EntLosses))
                print('       Average Constraints loss: %.6f' % np.mean(CLosses))
                print('       Average Clip Fraction: %.4f' % np.mean(ppo_clip_fractions))
                print('       Average Lambda Pi Grad: %.4f' % LambdaPiGrads)
                print('       Average Pi Grad: %.4f' % PiGrads)
                print()
            
            l_Mean_VLoss.append(np.mean(VLosses))
            l_Mean_PgLoss.append(np.mean(PgLosses))
            l_Mean_EntLoss.append(np.mean(EntLosses))
            l_Mean_CLoss.append(np.mean(CLosses))
            l_Mean_J.append(J_batch.mean().item())
            l_Median_J.append(J_batch.median().item())
            l_Min_J.append(J_batch.min().item())
            l_Max_J.append(J_batch.max().item())
                    
        else:
        
            exploreBool = True if lambda_epoch < 10 else False # Random action exploration for the first 10 learning epochs
            
            for net in qvalue_nets:
                net.eval()
            policy_net.eval()
            
            batch_ts_s, batch_ts_s_norm, batch_ts_a, batch_ts_r, batch_ts_r_norm, J_batch, J_norm_batch, batch_ts_lengths = utils.sample_policy_traj_noise(X0_tensor, domain_dict, policy_net, state_rms_constr, return_rms_constr, replay_buffer_constr, pi_replay_buffer_constr, action_noise_prop = max(0.0, 1.0 - lambda_epoch / num_epoch[domain]), mode='train', explore = exploreBool)
        
            batch_ts_sa = torch.concat((batch_ts_s, batch_ts_a), dim=2)
            class_outputs = classifier(batch_ts_sa, batch_ts_lengths)
                    
            total_QLoss, total_PiJLoss, total_PiLaLoss, total_RegLoss = 0, 0, 0, 0
            Mean_QGrad, Median_QGrad, Min_QGrad, Max_QGrad = 0, 0, 0, 0
            Mean_PiGrad, Median_PiGrad, Min_PiGrad, Max_PiGrad = 0, 0, 0, 0
        
            if replay_buffer_constr.size >= min_buffer_samples:
                
                for net in qvalue_nets:
                    net.train()
                policy_net.train()
                
                q_chain_optimizer.zero_grad()
                
                l_batch_pi_samples = []
                
                # Number of updates = horizon length of the batch trajectories collected
                for _ in range(env.horizon):
                
                    batch_q_samples = replay_buffer_constr.sample_batch(batchsize)
                    l_batch_pi_samples.append(batch_q_samples)
                    
                    QLoss = losses.q_loss_TD3(batch_q_samples['norm_states'], batch_q_samples['actions'], 
                                              batch_q_samples['rewards'], batch_q_samples['norm_next_states'], 
                                              batch_q_samples['dones'], domain_dict, qvalue_nets, qvalue_nets_targ, 
                                              policy_net_targ, state_rms_constr) / domain_dict['horizon']
                    
                    QLoss.backward()
                    total_QLoss += QLoss.item()
                
                QGrads = np.concatenate([p.grad.abs().cpu().numpy().flatten() for nw in qvalue_nets for p in nw.parameters()], axis=0)
                Mean_QGrad, Median_QGrad, Min_QGrad, Max_QGrad = np.mean(QGrads), np.median(QGrads), np.min(QGrads), np.max(QGrads)
                
                q_chain_optimizer.step()
                
                if lambda_epoch % 2 == 1:
                
                    for net in qvalue_nets:
                        for p in net.parameters():
                            p.requires_grad = False
                        net.eval()
                    
                    pi_constr_optimizer.zero_grad()
                    
                    # =========================================
                    with torch.no_grad():
                        lambda_sum = lambda_vec.sum().item()
                    # =========================================
                    
                    for batch_pi_samples in l_batch_pi_samples:
                                            
                        if actorObj == 'TD3':
                            PiLoss = losses.pi_TD3_loss(batch_pi_samples['norm_states'], domain_dict, qvalue_nets, policy_net, state_rms_constr) / domain_dict['horizon']
                        elif actorObj == 'ILBO':
                            PiLoss = losses.pi_ILBO_loss(batch_pi_samples['norm_states'], batch_pi_samples['rewards'], 
                                                         batch_pi_samples['norm_next_states'], domain_dict, 
                                                         qvalue_nets[0], policy_net, 
                                                         state_rms_constr, return_rms_constr) / domain_dict['horizon']
                        
                        PiLoss = PiLoss / (1 + lambda_sum)
                        PiLoss.backward()
                        total_PiJLoss += PiLoss.item()
                    
                    if (len(constr_dict[domain_dict['domain_name']]) > 0):
                        
                        if constrOptMode == 'RCPO':
                            
                            log_lambda_constr_optimizer.zero_grad()
                            
                            LaLoss, means_satifs = losses.pi_rcpo_lambda_loss(class_outputs, lambda_vec, constr_dict[domain_dict['domain_name']])
                            # =========================================
                            LaLoss = LaLoss / (1 + lambda_sum)
                            # =========================================
                            LaLoss.backward()
                            total_PiLaLoss += LaLoss.item()
                            
                            if log_lambda_vec.grad is not None:
                                log_lambda_vec.grad.mul_(1 + lambda_sum)
                            
                            log_lambda_constr_optimizer.step()
                            
                            lambda_vec = torch.exp(log_lambda_vec)
                    
                    # Maximize the lagrangian w.r.t theta
                    for p in policy_net.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(-1)
                    
                    PiGrads = np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0)
                    Mean_PiGrad, Median_PiGrad, Min_PiGrad, Max_PiGrad = np.mean(PiGrads), np.median(PiGrads), np.min(PiGrads), np.max(PiGrads)
                    
                    pi_constr_optimizer.step()
                    
                    for net in qvalue_nets:
                        for p in net.parameters():
                            p.requires_grad = True
                        net.train()
                    
                    for net, net_targ in zip(qvalue_nets, qvalue_nets_targ):
                        utils.polyak_update(net, net_targ, tau)
                    utils.polyak_update(policy_net, policy_net_targ, tau)
                
                else:
                    total_PiJLoss, total_PiLaLoss, Mean_PiGrad, Median_PiGrad, Min_PiGrad, Max_PiGrad = np.float32(float('nan')), np.float32(float('nan')), np.float32(float('nan')), np.float32(float('nan')), np.float32(float('nan')), np.float32(float('nan'))
                
                # print statistics
                print('[%d] Average Q loss: %.6f' % (lambda_epoch + 1, total_QLoss / 1))
                print('       Average Q Gradient: %.2E' % Decimal(Mean_QGrad.item()))
                print()
                print('       Average Pi J loss: %.6f' % (total_PiJLoss / 1))
                print('       Average Pi Lambda loss: %.6f' % (total_PiLaLoss / 1))
                print('       Average Pi Gradient: %.2E' % Decimal(Mean_PiGrad.item()))
                print()
        
            l_Mean_QLoss.append(total_QLoss)
            l_Mean_JLoss.append(total_PiJLoss)
            l_Mean_LaLoss.append(total_PiLaLoss)
            l_Mean_RegLoss.append(total_RegLoss)
            l_Mean_J.append(J_batch.mean().item())
            l_Median_J.append(J_batch.median().item())
            l_Min_J.append(J_batch.min().item())
            l_Max_J.append(J_batch.max().item())
            l_Mean_QGrad.append(Mean_QGrad)
            l_Median_QGrad.append(Median_QGrad)
            l_Min_QGrad.append(Min_QGrad)
            l_Max_QGrad.append(Max_QGrad)
            l_Mean_PiGrad.append(Mean_PiGrad)
            l_Median_PiGrad.append(Median_PiGrad)
            l_Min_PiGrad.append(Min_PiGrad)
            l_Max_PiGrad.append(Max_PiGrad)
        
        # Test the Policy Network
        if (lambda_epoch == 0) or ((lambda_epoch + 1) % test_interval == 0):
            
            policy_net.eval()
            
            dict_nse_traj, dict_prop_nse, dict_disc_rets, dict_undisc_rets, \
                tensor_s, tensor_a, tensor_r = test_policy(env, policy_net, num_test_trajs, 
                                                           domain_dict, state_rms_constr, return_rms_constr, 
                                                           v_network = value_net if actorObj == 'PPO' else None, 
                                                           rollout_buffer = rollout_buffer if actorObj == 'PPO' else None)
            
            log.log_test_trajs(base_logfile_path, lambda_epoch, domain_dict, 
                               dict_nse_traj, dict_prop_nse, dict_disc_rets, dict_undisc_rets)
            
            lTest_Prop_NSE0.append(dict_prop_nse[0])
            lTest_Prop_NSE1.append(dict_prop_nse[1])
            lTest_Prop_NSE2.append(dict_prop_nse[2])
            
            lTest_Min_DiscRet.append(dict_disc_rets['min'])
            lTest_Mean_DiscRet.append(dict_disc_rets['mean'])
            lTest_Median_DiscRet.append(dict_disc_rets['median'])
            lTest_Max_DiscRet.append(dict_disc_rets['max'])
            
            lTest_Min_UndiscRet.append(dict_undisc_rets['min'])
            lTest_Mean_UndiscRet.append(dict_undisc_rets['mean'])
            lTest_Median_UndiscRet.append(dict_undisc_rets['median'])
            lTest_Max_UndiscRet.append(dict_undisc_rets['max'])
        
        # Add trajectory to NSE Dict
        if writeTrajBool and (lambda_epoch < num_epoch[domain] // 2): # Unconstrained policy converges much earlier, samples after convergence (e.g. after 50% of learning epochs) are not so meaningful for NSE classifier
            
            ts_danger_dict = nse.batch_ts_danger(domain_dict, batch_ts_s, batch_ts_a)
            dict_TrajSets, prop_NSEs = nse.batch_trajDF_by_nse(domain_dict, batch_ts_s, batch_ts_a, batch_ts_r, ts_danger_dict)
            
            for nse_class in dict_TrajSets:
                trajDFs_by_nse[nse_class] += dict_TrajSets[nse_class]
    
    if writeTrajBool:
        log.log_nse_trajs(writeTrajPath, domain_dict, trajDFs_by_nse, prop_NSEs)
    
    print("Finished Training")
    torch.backends.cudnn.enabled = True
    
    end = time.time()
    print("Elapsed Time: %.2f secs" % (end - start))
        
    if actorObj == 'PPO':
        trainDict = dict(VLoss = l_Mean_VLoss, JLoss = l_Mean_PgLoss, EntLoss = l_Mean_EntLoss, CLoss = l_Mean_CLoss, 
                         MinRet = l_Min_J, MeanRet = l_Mean_J, MedianRet = l_Median_J, MaxRet = l_Max_J)
    elif actorObj == 'TD3':
        trainDict = dict(QLoss = l_Mean_QLoss, JObj = l_Mean_JLoss, LaObj = l_Mean_LaLoss, RegObj = l_Mean_RegLoss, 
                         MinRet = l_Min_J, MeanRet = l_Mean_J, MedianRet = l_Median_J, MaxRet = l_Max_J, 
                         MinQGrad = l_Min_QGrad, MeanQGrad = l_Mean_QGrad, MedianQGrad = l_Median_QGrad, 
                         MaxQGrad = l_Max_QGrad, MinPiGrad = l_Min_PiGrad, MeanPiGrad = l_Mean_PiGrad, 
                         MedianPiGrad = l_Median_PiGrad, MaxPiGrad = l_Max_PiGrad)
    
    log.log_summary(base_logfile_path, train = True, n_test_interval = test_interval, 
                    lambdas = l_lambdas, **trainDict)
    
    log.log_summary(base_logfile_path, train = False, n_test_interval = test_interval, 
                    lambdas = l_lambdas, Perc_NSE0 = lTest_Prop_NSE0, Perc_NSE1 = lTest_Prop_NSE1, Perc_NSE2 = lTest_Prop_NSE2, 
                    MinDiscRet = lTest_Min_DiscRet, MeanDiscRet = lTest_Mean_DiscRet, MedianDiscRet = lTest_Median_DiscRet, MaxDiscRet = lTest_Max_DiscRet, 
                    MinUndiscRet = lTest_Min_UndiscRet, MeanUndiscRet = lTest_Mean_UndiscRet, MedianDUndscRet = lTest_Median_UndiscRet, MaxUndiscRet = lTest_Max_UndiscRet)
    
