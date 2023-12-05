#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from itertools import chain
from decimal import Decimal
from boxpushing import boxpushing
from navigation import navigation
from grid_NSEClassify import domain_dict, SAClassifier, TSGRU
import utils, models, grid_losses, grid_log, grid_nse

def test_policy(env, pi_network, num_test_trajs, gamma, traj_bool=False):
    
    pi_network.eval()
    
    dict_discRets = {'min': 0.0, 
                     'mean': 0.0, 
                     'median': 0.0, 
                     'max': 0.0, 
                     'std': 0.0}
    dict_undiscRets = {'min': 0.0, 
                       'mean': 0.0, 
                       'median': 0.0, 
                       'max': 0.0, 
                       'std': 0.0}
    dict_discNSEs = {'min': 0.0, 
                     'mean': 0.0, 
                     'median': 0.0, 
                     'max': 0.0, 
                     'std': 0.0}
    dict_undiscNSEs = {'min': 0.0, 
                       'mean': 0.0, 
                       'median': 0.0, 
                       'max': 0.0, 
                       'std': 0.0}
    
    batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, J_batch = utils.sample_batch_trajs(env, policy_net, batchsize = num_test_trajs, train=False)
    
    dict_discRets['min'] = torch.min(J_batch).item()
    dict_discRets['mean'] = torch.mean(J_batch).item()
    dict_discRets['median'] = torch.median(J_batch).item()
    dict_discRets['max'] = torch.max(J_batch).item()
    dict_discRets['std'] = torch.std(J_batch).item()
    
    J_undisc_batch = torch.sum(batch_ts_r, dim = 1)
    
    dict_undiscRets['min'] = torch.min(J_undisc_batch).item()
    dict_undiscRets['mean'] = torch.mean(J_undisc_batch).item()
    dict_undiscRets['median'] = torch.median(J_undisc_batch).item()
    dict_undiscRets['max'] = torch.max(J_undisc_batch).item()
    dict_undiscRets['std'] = torch.std(J_undisc_batch).item()
    
    batch_mean_c = batch_ts_c.mean(dim = 1)
    batch_disc_sum_c = (batch_ts_c * torch.tensor([gamma**i for i in range(batch_ts_c.shape[1])]).to(utils.device)).sum(dim = 1)
    
    dict_discNSEs['min'] = torch.min(batch_disc_sum_c).item()
    dict_discNSEs['mean'] = torch.mean(batch_disc_sum_c).item()
    dict_discNSEs['median'] = torch.median(batch_disc_sum_c).item()
    dict_discNSEs['max'] = torch.max(batch_disc_sum_c).item()
    dict_discNSEs['std'] = torch.std(batch_disc_sum_c).item()
    
    batch_undisc_sum_c = batch_ts_c.sum(dim = 1)
    
    dict_undiscNSEs['min'] = torch.min(batch_undisc_sum_c).item()
    dict_undiscNSEs['mean'] = torch.mean(batch_undisc_sum_c).item()
    dict_undiscNSEs['median'] = torch.median(batch_undisc_sum_c).item()
    dict_undiscNSEs['max'] = torch.max(batch_undisc_sum_c).item()
    dict_undiscNSEs['std'] = torch.std(batch_undisc_sum_c).item()
    
    dict_TrajSets = {0: [], 
                     1: []}
    prop_NSEs = {0: 0.0, 
                 1: 0.0}
    
    if traj_bool and isinstance(env, boxpushing):
        dict_TrajSets[2], dict_TrajSets[3] = [], []
        prop_NSEs[2], prop_NSEs[3] = 0.0, 0.0
    
    num_trajs = batch_ts_s.shape[0]
    
    for traj_idx in range(num_trajs):
        
        ts_s_np = batch_ts_s[traj_idx].detach().cpu().numpy()
        action_np = batch_ts_a[traj_idx].detach().cpu().numpy()
        reward_np = batch_ts_r[traj_idx].detach().cpu().numpy().reshape((-1, 1))
        nse_np = batch_ts_c[traj_idx].detach().cpu().numpy().reshape((-1, 1))
        
        if isinstance(env, boxpushing):
            
            agent_locs_np = ts_s_np[:, :-2].argmax(axis = -1)            
            absorb_np = (ts_s_np[:, -2] == 1).reshape((-1, 1))
            
            final_colnames = ['Cell' + str(i) for i in range(env.rows * env.columns)] + ['DummyCell', 'Loaded', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'X', 'Y', 'Absorbing', 'Action', 'reward', 'nse']
                    
        else:
            
            agent_locs_np = ts_s_np[:, :-1].argmax(axis = -1)
            absorb_np = (ts_s_np[:, -1] == 1).reshape((-1, 1))
            
            final_colnames = ['Cell' + str(i) for i in range(env.rows * env.columns)] + ['DummyCell', 'LEFT_SLOW', 'LEFT_FAST', 'RIGHT_SLOW', 'RIGHT_FAST', 'UP_SLOW', 'UP_FAST', 'DOWN_SLOW', 'DOWN_FAST', 'X', 'Y', 'Absorbing', 'Action', 'reward', 'nse']            
        
        agent_x_np = (agent_locs_np % env.columns).reshape((-1, 1))
        agent_y_np = (agent_locs_np // env.columns).reshape((-1, 1))
        action_idx_np = action_np.argmax(axis = -1).reshape((-1, 1))
        
        final_np = np.concatenate((ts_s_np, action_np, agent_x_np, agent_y_np, absorb_np, action_idx_np, reward_np, nse_np), axis = 1)
        
        traj_df = pd.DataFrame(final_np, columns = final_colnames)
                
        if traj_bool:
            traj_danger = (traj_df['nse'] > 0)
            consec_danger = traj_danger.groupby((traj_danger != traj_danger.shift()).cumsum()).transform('size') * traj_danger
            max_consec_danger = consec_danger.max()
            if isinstance(env, boxpushing):
                traj_length = (traj_df['DummyCell'] == 0.).sum()
                danger_ratio = traj_danger.sum() / traj_length
                if danger_ratio < 0.1:
                    if traj_length < env.max_horizon:
                        dict_TrajSets[0].append(traj_df)
                    else:
                        dict_TrajSets[2].append(traj_df)
                else:
                    if traj_length < env.max_horizon:
                        dict_TrajSets[1].append(traj_df)
                    else:
                        dict_TrajSets[3].append(traj_df)
                
            else:
                danger_ratio = traj_danger.sum() / (traj_df['DummyCell'] == 0.).sum()
                if danger_ratio < 0.1:
                    dict_TrajSets[0].append(traj_df)
                else:
                    dict_TrajSets[1].append(traj_df)
        else:
            if batch_mean_c[traj_idx] > 0.0:
                dict_TrajSets[1].append(traj_df)
            else:
                dict_TrajSets[0].append(traj_df)
    
    for k in dict_TrajSets.keys():
        prop_NSEs[k] = len(dict_TrajSets[k]) / num_trajs
        
    return dict_TrajSets, prop_NSEs, dict_discRets, dict_undiscRets, dict_discNSEs, dict_undiscNSEs, batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, J_batch

if __name__ == "__main__":
    
    try:
        domain = sys.argv[1]
        ptFileDir = sys.argv[2]
        if ptFileDir[-1] != '/':
            ptFileDir += '/'        
        logFilePath = sys.argv[3]
        if logFilePath[-1] != '/':
            logFilePath += '/'
        actorObj = sys.argv[4] # PPO
        assert actorObj == 'PPO' or actorObj == 'PPO_Traj'
        constrOptMode = sys.argv[5] # RCPO or KKT or NIL
        if actorObj == 'PPO':
            assert constrOptMode == 'RCPO' or constrOptMode == 'RCPO-MF' or constrOptMode == 'NIL'
        else:
            assert constrOptMode == 'RCPO-MK' or constrOptMode == 'RCPO-NonMK' or constrOptMode == 'NIL'
        init_lambda = float(sys.argv[6]) # Initial value of lambda for all constraints
        lambda_lr = float(sys.argv[7])
        seed = int(sys.argv[8])
        strict = sys.argv[9]
        assert strict.lower() == 'y' or strict.lower() == 'n', ""
        if strict.lower() == 'y':
            strict = True
        else:
            strict = False
        writeTrajBool = True if (len(sys.argv) > 10) else False
        if writeTrajBool:
            writeTrajPath = sys.argv[10]
            if writeTrajPath[-1] != '/':
                writeTrajPath += '/'
        if (len(sys.argv) > 11):
            budget_str = sys.argv[11]
        else:
            budget_str = str('inf')
    except:
        print("Invalid parameters specified")
        sys.exit(1)
    
    if domain in domain_dict:
        state_dim, action_dim, env_files = domain_dict[domain]
        # env_files = ['grid-3','grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7'] # box
        # env_files = ['grid-3','grid-3-t6','grid-3-t4','grid-3-t3','grid-3-t5','grid-3-t7'] # nav
    else:
        print("Domain specified in filepath not recognized")
        sys.exit(1)
        
    feature_dim = state_dim + action_dim
    
    # (index of class - 0 to 2, GE or LE, threshold value)
    constr_dict = {'box': [('LE', 0.0)], 
                   'nav': [('LE', 0.0)]}
    
    if actorObj == 'PPO_Traj':
        if domain == 'box':
            env_files = ['grid-3-t4']
        else:
            env_files = ['grid-3-t1']
        if constrOptMode == 'RCPO-MK':
            constr_dict = {'box': [('LE', 0.05)], 
                           'nav': [('LE', 0.05)]}
        else:
            constr_dict = {'box': [(0, 'GE', 0.95)], 
                           'nav': [(0, 'GE', 0.95)]}
    
    batchsize = 100
    num_epoch = {'box': 1000, 
                 'nav': 1000}
    test_interval = 50
    num_test_trajs = 100
    gamma = 0.99
    # Hidden Units of Classifier
    hidden_units = [32, 32]
    # Hyperparameters
    ppo_clip_range = 0.2
    classifier_treshold = 0.5
    learn_constr_aft_epoch = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_printoptions(precision = 4)
    # torch.autograd.set_detect_anomaly(True)
    
    # cudnn does not support backward operations during classifier eval, thus has to be turned off
    torch.backends.cudnn.enabled = False
        
    for envfile in env_files:
        
        print("Env: " + envfile)
        
        base_logfile_path = grid_log.init_log_folder(logFilePath, domain, envfile, ptFileDir, actorObj, 
                                                     constrOptMode, init_lambda, seed, constr_dict[domain], lambda_lr)
        
        if (actorObj == 'PPO_Traj'):
            
            if (constrOptMode == 'RCPO-NonMK'):                
                classifier = TSGRU(feature_dim=feature_dim, num_classes = 4, gru_layers = 2, nb_gru_units=hidden_units[0], batch_size=batchsize).to(utils.device)
                ptFilePath = ptFileDir + domain + '/' + envfile + '_GRU_' + str(hidden_units[0]) + '_' + str(batchsize) + '.pt'
                
            else:                
                classifier = SAClassifier(state_dim=state_dim, action_dim=action_dim, hidden_layers=hidden_units).to(utils.device)                
                ptFilePath = ptFileDir + domain + '/' + envfile + '_MarkovClassifier_'                
                for unit in hidden_units:
                    ptFilePath += (str(unit) + '_')
                ptFilePath += (str(batchsize) + '.pt')
        
        else:
            # Load Corresponding Classifier        
            classifier = SAClassifier(state_dim=state_dim, action_dim=action_dim, hidden_layers=hidden_units).to(utils.device)
            
            ptFilePath = ptFileDir + domain + '/' + envfile + '_' + budget_str + '_' + ('HA-S' if strict else 'HA-L') + '_32_32_Classifier.pt'
        
        classifier.load_state_dict(torch.load(ptFilePath, map_location=utils.device))
        
        # Freeze classifier param
        for param in classifier.parameters():
            param.requires_grad_(False)
        
        classifier.eval()
        
        if domain == 'box':
            env = boxpushing(ptFileDir + domain + '/' + envfile + '.bp', gamma)
        else:
            env = navigation(ptFileDir + domain + '/' + envfile + '.nav', gamma)
        
        
        if actorObj.startswith('PPO'):
            
            # Value Net and Policy Net
            value_net = models.ValueNW(input_dim=state_dim, output_dim=1, hidden_layers=[32, 32]).to(utils.device)
            policy_net = models.CategoricalPolicyNW(input_dim=state_dim, output_dim=action_dim, hidden_layers=[32, 32]).to(utils.device)
            
            # Optimizer
            value_net_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-4)
            policy_net_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    
            # Rollout buffer
            rollout_buffer = utils.RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, max_horizon = env.max_horizon, num_envs = batchsize, gae_lambda = 1.0, gamma = gamma)
            
            l_Mean_VLoss, l_Mean_PgLoss, l_Mean_EntLoss, l_Mean_CLoss = [], [], [], []
        
        lambda_vec = torch.tensor([0.] * len(constr_dict[domain]), 
                                  requires_grad=False, 
                                  device=utils.device)
        
        start = time.time()
        
        l_lambdas = []
        l_Mean_J, l_Median_J, l_Min_J, l_Max_J = [], [], [], []
        lTest_Prop_NSE0, lTest_Prop_NSE1, lTest_Pred_NSE0, lTest_Pred_NSE1 = [], [], [], []#, lTest_Prop_NSE2 = [], [], []
        lTest_Prop_NSE2, lTest_Prop_NSE3, lTest_Pred_NSE2, lTest_Pred_NSE3 = [], [], [], []
        lTest_Min_DiscRet, lTest_Mean_DiscRet, lTest_Median_DiscRet, lTest_Max_DiscRet, lTest_Std_DiscRet = [], [], [], [], []
        lTest_Min_UndiscRet, lTest_Mean_UndiscRet, lTest_Median_UndiscRet, lTest_Max_UndiscRet, lTest_Std_UndiscRet = [], [], [], [], []
        lTest_Min_DiscNSE, lTest_Mean_DiscNSE, lTest_Median_DiscNSE, lTest_Max_DiscNSE, lTest_Std_DiscNSE = [], [], [], [], []
        lTest_Min_UndiscNSE, lTest_Mean_UndiscNSE, lTest_Median_UndiscNSE, lTest_Max_UndiscNSE, lTest_Std_UndiscNSE = [], [], [], [], []
        trajDFs_by_nse = {0: [], 
                          1: []}
        
        if (actorObj == 'PPO_Traj') and isinstance(env, boxpushing):
            trajDFs_by_nse[2], trajDFs_by_nse[3] = [], []
        
        for lambda_epoch in range(num_epoch[domain]):
            
            if (constrOptMode.startswith('RCPO')) and (lambda_epoch == learn_constr_aft_epoch):
                log_lambda_vec = torch.tensor([np.log(init_lambda)] * len(constr_dict[domain]), 
                                              requires_grad=True, 
                                              device=utils.device)
                
                log_lambda_constr_optimizer = torch.optim.Adam([log_lambda_vec], lr=lambda_lr)
            
            with torch.no_grad():
                if (constrOptMode.startswith('RCPO')) and (lambda_epoch >= learn_constr_aft_epoch):                
                    lambda_vec = log_lambda_vec.exp()
                
                lambda_sum = lambda_vec.sum().item()
                l_lambdas.append(lambda_vec.tolist())
                print("Lambdas = ", lambda_vec)
            
            if actorObj.startswith('PPO'):
                
                value_net.eval()
                policy_net.eval()
                
                if constrOptMode == 'RCPO-NonMK':
                    batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, J_batch, batch_ts_lengths = utils.sample_batch_trajs(env, policy_net, v_nw=value_net, rollout_buffer=rollout_buffer, batchsize = batchsize, train=True)
                    dummyIdx = -2 if domain == 'box' else -1
                    batch_ts_lengths = (batch_ts_s[:, :, dummyIdx] == 0).sum(dim = -1)
                else:
                    batch_ts_s, batch_ts_a, batch_ts_r, batch_ts_c, batch_ts_c_hat, J_batch = utils.sample_batch_trajs(env, policy_net, v_nw=value_net, classifier_nw=classifier, rollout_buffer=rollout_buffer, batchsize = batchsize, train=True)
                
                PgLosses, VLosses, EntLosses, CLoss = [], [], [], 0
                ppo_clip_fractions = []
                
                value_net.train()
                policy_net.train()
                
                # Entropy in the first 100 epochs
                entropy_strength = max(0, 1 - lambda_epoch / num_epoch[domain] * 2) * 0.005
                
                PiGrads, LambdaPiGrads = 0, 0
                
                # Ignore constraint in the first 200 epochs
                if (len(constr_dict[domain]) > 0) and (lambda_epoch >= learn_constr_aft_epoch):
                    
                    if (constrOptMode == 'RCPO') or (constrOptMode == 'RCPO-MK'):
                                                
                        # traj_mean_nse_probs = batch_ts_c_hat.mean(dim = 1)
                        bool_high_c_hat = (batch_ts_c_hat > classifier_treshold)
                        if constrOptMode == 'RCPO-MK':
                            dummyIdx = -2 if domain == 'box' else -1
                            bool_high_c_hat = bool_high_c_hat * (batch_ts_s[:, :, dummyIdx] == 0)
                        traj_sum_nse_probs = (batch_ts_c_hat * bool_high_c_hat).sum(dim = 1)
                        
                        LaLoss, means_satifs = grid_losses.rcpo_lambda_loss(traj_sum_nse_probs, log_lambda_vec.exp(), constr_dict[domain])
                        # =========================================
                        LaLoss = -LaLoss / (1 + lambda_sum)
                        # =========================================
                        policy_net_optimizer.zero_grad()
                        log_lambda_constr_optimizer.zero_grad()
                        LaLoss.backward()
                        CLoss += LaLoss.item()
                        
                        if log_lambda_vec.grad is not None:
                            log_lambda_vec.grad.mul_(-(1 + lambda_sum))
                        
                        torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()), 0.5)
                        torch.nn.utils.clip_grad_norm_(list(log_lambda_vec), 0.005)
                        
                        LambdaPiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                        
                        policy_net_optimizer.step()
                        log_lambda_constr_optimizer.step()
                    
                    elif constrOptMode == 'RCPO-MF':
                        
                        weighted_sum_log_prob = torch.zeros((batch_ts_c_hat.shape[0])).to(utils.device)
                        dummyIdx = -2 if domain == 'box' else -1
                        for i in range(batch_ts_c_hat.shape[1]):
                            log_prob, _ = policy_net.eval_action(batch_ts_s[:, i, :], batch_ts_a[:, i, :])
                            weighted_sum_log_prob += (log_prob * (batch_ts_s[:, i, dummyIdx] == 0) * batch_ts_c_hat[:, i:].sum(dim = 1) / 100.)
                        
                        batch_ts_c_hat_noDummy = batch_ts_c_hat * (batch_ts_s[:, :, dummyIdx] == 0)
                        sum_outputs_nograd = batch_ts_c_hat_noDummy.sum(dim = 1).detach()
                        log_lambda_vec_nograd = log_lambda_vec.detach()
                        
                        LaLoss = grid_losses.rcpo_mf_lambda_loss_markov(weighted_sum_log_prob, log_lambda_vec_nograd.exp(), constr_dict[domain])
                        
                        LaLoss = -LaLoss / (1 + lambda_sum)
                        policy_net_optimizer.zero_grad()
                        LaLoss.backward()
                        CLoss += LaLoss.item()
                        
                        torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()), 0.5)                        
                        policy_net_optimizer.step()
                        
                        LambdaPiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                        
                        # No need to reverse sign of LamLoss since we are minimizing wrt lambda
                        LamLoss, means_satifs = grid_losses.rcpo_lambda_loss(sum_outputs_nograd, log_lambda_vec.exp(), constr_dict[domain])
                        
                        log_lambda_constr_optimizer.zero_grad()
                        LamLoss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(list(log_lambda_vec), 0.005)
                        log_lambda_constr_optimizer.step()
                    
                    elif constrOptMode == 'RCPO-NonMK':
                        batch_ts_sa = torch.concat((batch_ts_s, batch_ts_a), dim = -1)
                        class_outputs = classifier(batch_ts_sa, batch_ts_lengths)
                        
                        # RCPO-style of gradient update of lambda
                        LaLoss, means_satifs = grid_losses.pi_rcpo_lambda_loss(class_outputs, log_lambda_vec.exp(), constr_dict[domain])
                        # =========================================
                        LaLoss = -LaLoss / (1 + lambda_sum)
                        # =========================================
                        policy_net_optimizer.zero_grad()
                        log_lambda_constr_optimizer.zero_grad()
                        LaLoss.backward()
                        CLoss += LaLoss.item()
                        
                        if log_lambda_vec.grad is not None:
                            log_lambda_vec.grad.mul_(-(1 + lambda_sum))
                        
                        torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()), 0.5)
                        torch.nn.utils.clip_grad_norm_(list(log_lambda_vec), 0.005)
                        
                        LambdaPiGrads += np.concatenate([p.grad.abs().cpu().numpy().flatten() for p in policy_net.parameters()], axis=0).sum()
                        
                        policy_net_optimizer.step()
                        log_lambda_constr_optimizer.step()
                                    
                num_rollout_batches = len(list(rollout_buffer.get(batchsize)))
                
                for rollout_data in rollout_buffer.get(batchsize):
                    
                    actions = rollout_data['actions']
                    
                    log_prob, entropy = policy_net.eval_action(rollout_data['observations'], actions)
                    values = value_net(rollout_data['observations'])
                    
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
                print('       Average Constraints loss: %.6f' % CLoss)
                print('       Average Clip Fraction: %.4f' % np.mean(ppo_clip_fractions))
                print('       Average Lambda Pi Grad: %.4f' % LambdaPiGrads)
                print('       Average Pi Grad: %.4f' % PiGrads)
                print()
                
                l_Mean_VLoss.append(np.mean(VLosses))
                l_Mean_PgLoss.append(np.mean(PgLosses))
                l_Mean_EntLoss.append(np.mean(EntLosses))
                l_Mean_CLoss.append(CLoss)
                l_Mean_J.append(J_batch.mean().item())
                l_Median_J.append(J_batch.median().item())
                l_Min_J.append(J_batch.min().item())
                l_Max_J.append(J_batch.max().item())
                    
            # Test the Policy Network
            if (lambda_epoch == 0) or ((lambda_epoch + 1) % test_interval == 0):
                
                policy_net.eval()
                
                dict_nse_traj, dict_prop_nse, dict_disc_rets, dict_undisc_rets, dict_disc_NSEs, dict_undisc_NSEs, \
                    tensor_s, tensor_a, tensor_r, tensor_c, tensor_J = test_policy(env, policy_net, num_test_trajs, gamma, 
                                                                                   traj_bool = (actorObj == 'PPO_Traj'))
                
                grid_log.log_test_trajs(base_logfile_path, lambda_epoch, 
                                        dict_nse_traj, dict_prop_nse, dict_disc_rets, dict_undisc_rets, dict_disc_NSEs, dict_undisc_NSEs)
                
                lTest_Prop_NSE0.append(dict_prop_nse[0])
                lTest_Prop_NSE1.append(dict_prop_nse[1])
                
                if (actorObj == 'PPO_Traj') and (constrOptMode == 'RCPO-NonMK'):
                    dummyIdx = -2 if domain == 'box' else -1
                    tensor_lengths = (tensor_s[:, :, dummyIdx] == 0).sum(dim = -1)
                    tensor_sa = torch.concat((tensor_s, tensor_a), dim = -1)
                    test_class_outputs = classifier(tensor_sa, tensor_lengths)
                    
                    test_softmax_outputs = nn.Softmax(dim = 1)(test_class_outputs)
                    
                    lTest_Pred_NSE0.append((test_softmax_outputs[:, 0] > classifier_treshold).sum().item() / num_test_trajs)
                    lTest_Pred_NSE1.append((test_softmax_outputs[:, 1] > classifier_treshold).sum().item() / num_test_trajs)
                    if domain == 'box':
                        lTest_Prop_NSE2.append(dict_prop_nse[2])
                        lTest_Prop_NSE3.append(dict_prop_nse[3])
                        lTest_Pred_NSE2.append((test_softmax_outputs[:, 2] > classifier_treshold).sum().item() / num_test_trajs)
                        lTest_Pred_NSE3.append((test_softmax_outputs[:, 3] > classifier_treshold).sum().item() / num_test_trajs)
                else:
                    lTest_Pred_NSE0.append(0.)
                    lTest_Pred_NSE1.append(0.)
                
                lTest_Min_DiscRet.append(dict_disc_rets['min'])
                lTest_Mean_DiscRet.append(dict_disc_rets['mean'])
                lTest_Median_DiscRet.append(dict_disc_rets['median'])
                lTest_Max_DiscRet.append(dict_disc_rets['max'])
                lTest_Std_DiscRet.append(dict_disc_rets['std'])
                
                lTest_Min_UndiscRet.append(dict_undisc_rets['min'])
                lTest_Mean_UndiscRet.append(dict_undisc_rets['mean'])
                lTest_Median_UndiscRet.append(dict_undisc_rets['median'])
                lTest_Max_UndiscRet.append(dict_undisc_rets['max'])
                lTest_Std_UndiscRet.append(dict_undisc_rets['std'])
                
                lTest_Min_DiscNSE.append(dict_disc_NSEs['min'])
                lTest_Mean_DiscNSE.append(dict_disc_NSEs['mean'])
                lTest_Median_DiscNSE.append(dict_disc_NSEs['median'])
                lTest_Max_DiscNSE.append(dict_disc_NSEs['max'])
                lTest_Std_DiscNSE.append(dict_disc_NSEs['std'])
                
                lTest_Min_UndiscNSE.append(dict_undisc_NSEs['min'])
                lTest_Mean_UndiscNSE.append(dict_undisc_NSEs['mean'])
                lTest_Median_UndiscNSE.append(dict_undisc_NSEs['median'])
                lTest_Max_UndiscNSE.append(dict_undisc_NSEs['max'])
                lTest_Std_UndiscNSE.append(dict_undisc_NSEs['std'])
        
            # Add trajectory to NSE Dict
            if writeTrajBool and (lambda_epoch < num_epoch[domain] // 4): # Unconstrained policy converges much earlier, samples after convergence (e.g. after 25% of learning epochs) are not so meaningful for NSE classifier
                
                ts_danger_dict = grid_nse.batch_ts_danger(domain, batch_ts_s, batch_ts_a, batch_ts_c)
                dict_TrajSets, prop_NSEs = grid_nse.batch_trajDF_by_nse(domain, env.max_horizon, batch_ts_s, batch_ts_a, batch_ts_r, ts_danger_dict)
                
                for nse_class in dict_TrajSets:
                    trajDFs_by_nse[nse_class] += dict_TrajSets[nse_class]
    
        if writeTrajBool:
            grid_log.log_nse_trajs(writeTrajPath, domain, trajDFs_by_nse, prop_NSEs)
                
        print("Finished Training")
        torch.backends.cudnn.enabled = True
        
        end = time.time()
        print("Elapsed Time: %.2f secs" % (end - start))
                    
        if actorObj == 'PPO' or actorObj == 'PPO_Traj':
            trainDict = dict(VLoss = l_Mean_VLoss, JLoss = l_Mean_PgLoss, EntLoss = l_Mean_EntLoss, CLoss = l_Mean_CLoss, 
                             MinRet = l_Min_J, MeanRet = l_Mean_J, MedianRet = l_Median_J, MaxRet = l_Max_J)
        
        grid_log.log_summary(base_logfile_path, train = True, n_test_interval = test_interval, 
                             lambdas = l_lambdas, **trainDict)
        
        if (actorObj == 'PPO_Traj') and (constrOptMode == 'RCPO-NonMK'):
            grid_log.log_summary(base_logfile_path, train = False, n_test_interval = test_interval, 
                                 lambdas = l_lambdas, Perc_NSE0 = lTest_Prop_NSE0, Perc_NSE1 = lTest_Prop_NSE1, Pred_NSE0 = lTest_Pred_NSE0, Pred_NSE1 = lTest_Pred_NSE1, #Perc_NSE2 = lTest_Prop_NSE2, 
                                 Perc_NSE2 = lTest_Prop_NSE2, Perc_NSE3 = lTest_Prop_NSE3, Pred_NSE2 = lTest_Pred_NSE2, Pred_NSE3 = lTest_Pred_NSE3, #Perc_NSE2 = lTest_Prop_NSE2, 
                                 MinDiscRet = lTest_Min_DiscRet, MeanDiscRet = lTest_Mean_DiscRet, MedianDiscRet = lTest_Median_DiscRet, MaxDiscRet = lTest_Max_DiscRet, StdDiscRet = lTest_Std_DiscRet, 
                                 MinUndiscRet = lTest_Min_UndiscRet, MeanUndiscRet = lTest_Mean_UndiscRet, MedianDUndscRet = lTest_Median_UndiscRet, MaxUndiscRet = lTest_Max_UndiscRet, StdUndiscRet = lTest_Std_UndiscRet, 
                                 MinDiscNSE = lTest_Min_DiscNSE, MeanDiscNSE = lTest_Mean_DiscNSE, MedianDiscNSE = lTest_Median_DiscNSE, MaxDiscNSE = lTest_Max_DiscNSE, StdDiscNSE = lTest_Std_DiscNSE, 
                                 MinUndiscNSE = lTest_Min_UndiscNSE, MeanUndiscNSE = lTest_Mean_UndiscNSE, MedianDUndscNSE = lTest_Median_UndiscNSE, MaxUndiscNSE = lTest_Max_UndiscNSE, StdUndiscNSE = lTest_Std_UndiscNSE)
        else:            
            grid_log.log_summary(base_logfile_path, train = False, n_test_interval = test_interval, 
                                 lambdas = l_lambdas, Perc_NSE0 = lTest_Prop_NSE0, Perc_NSE1 = lTest_Prop_NSE1, Pred_NSE0 = lTest_Pred_NSE0, Pred_NSE1 = lTest_Pred_NSE1, #Perc_NSE2 = lTest_Prop_NSE2, 
                                 MinDiscRet = lTest_Min_DiscRet, MeanDiscRet = lTest_Mean_DiscRet, MedianDiscRet = lTest_Median_DiscRet, MaxDiscRet = lTest_Max_DiscRet, StdDiscRet = lTest_Std_DiscRet, 
                                 MinUndiscRet = lTest_Min_UndiscRet, MeanUndiscRet = lTest_Mean_UndiscRet, MedianDUndscRet = lTest_Median_UndiscRet, MaxUndiscRet = lTest_Max_UndiscRet, StdUndiscRet = lTest_Std_UndiscRet, 
                                 MinDiscNSE = lTest_Min_DiscNSE, MeanDiscNSE = lTest_Mean_DiscNSE, MedianDiscNSE = lTest_Median_DiscNSE, MaxDiscNSE = lTest_Max_DiscNSE, StdDiscNSE = lTest_Std_DiscNSE, 
                                 MinUndiscNSE = lTest_Min_UndiscNSE, MeanUndiscNSE = lTest_Mean_UndiscNSE, MedianDUndscNSE = lTest_Median_UndiscNSE, MaxUndiscNSE = lTest_Max_UndiscNSE, StdUndiscNSE = lTest_Std_UndiscNSE)
            
