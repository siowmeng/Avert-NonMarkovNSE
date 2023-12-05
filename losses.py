#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import utils

def q_loss_TD3(batch_states_norm, batch_actions, batch_rewards, batch_next_states_norm, batch_dones, domain_dict, q_nws, q_tar_nws, pi_tar_nw, s_rms):
    
    # target_policy_noise_std = 0.05
    # target_noise_clip = 0.125
    target_policy_noise_std = 0.1
    target_noise_clip = 0.25
    domain = domain_dict['domain_name']
    gamma = domain_dict['gamma']
    action_dim = domain_dict['num_actions']
    min_actions, max_actions = domain_dict['min_actions'], domain_dict['max_actions']
    
    curr_qs = []
    for nw in q_nws:
        curr_qs.append(nw(batch_states_norm, batch_actions))
    
    with torch.no_grad():
        
        batch_next_action_noise = torch.normal(0., target_policy_noise_std, 
                                               size = (batch_next_states_norm.shape[0], action_dim)).to(utils.device)
        
        batch_next_action_noise = batch_next_action_noise.clamp(-target_noise_clip, target_noise_clip)
        
        if domain.startswith('RES') and max_actions is None:
            # Use unnormalized states for action clamping
            batch_states = batch_states_norm * torch.sqrt(torch.tensor(s_rms.var + utils.epsilon).to(utils.device)) + torch.tensor(s_rms.mean).to(utils.device)
            
            batch_next_action_noise = batch_next_action_noise * (batch_states - min_actions)
            next_action = torch.clamp(pi_tar_nw(batch_next_states_norm, batch_states) + batch_next_action_noise, 
                                      min = min_actions, 
                                      max = batch_states)
        else:
            batch_next_action_noise = batch_next_action_noise * (max_actions - min_actions)
            next_action = torch.clamp(pi_tar_nw(batch_next_states_norm) + batch_next_action_noise, 
                                      min = min_actions, 
                                      max = max_actions)
        
        tar_next_qs = []
        for tar_nw in q_tar_nws:
            tar_next_qs.append(tar_nw(batch_next_states_norm, next_action))
        
        tar_next_qs = torch.cat(tar_next_qs, dim = 1)
        
        tar_next_q, _ = torch.min(tar_next_qs, dim = 1, keepdim = True)
        
        backup = batch_rewards + gamma * (1 - batch_dones) * tar_next_q
    
    loss_q = 0
    for curr_q in curr_qs:
        loss_q += ((curr_q - backup)**2).mean()
    
    return loss_q#, loss_info

def q_loss(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, gamma, q_nw, q_tar_nw, pi_tar_nw):
    
    #q = q_nw(torch.hstack((torch.tensor(batch_states), torch.tensor(batch_actions))))
    #q = q_nw(torch.hstack((batch_states, batch_actions)))
    q = q_nw(batch_states, batch_actions)
    
    with torch.no_grad():
        
        #q_pi_tar = q_tar_nw(torch.hstack((torch.tensor(batch_next_states), pi_tar_nw(torch.tensor(batch_next_states)))))
        #q_pi_tar = q_tar_nw(torch.hstack((batch_next_states, pi_tar_nw(batch_next_states))))
        q_pi_tar = q_tar_nw(batch_next_states, pi_tar_nw(batch_next_states))
        #backup = torch.tensor(batch_rewards) + gamma * (1 - torch.tensor(batch_dones)) * q_pi_tar
        backup = batch_rewards + gamma * (1 - batch_dones) * q_pi_tar
        #backup = torch.tensor(batch_rewards) + gamma * q_pi_tar        
#         backup = batch_rewards + gamma * q_pi_tar
    
    loss_q = ((q - backup)**2).mean()
    #loss_q = torch.abs(q - backup).mean()
    
    #loss_info = dict(QVals=q.detach().numpy())
    
    return loss_q#, loss_info

def pi_Q_loss(batch_states, q_nw, pi_nw):
    
    loss = q_nw(batch_states, pi_nw(batch_states)).mean()
    
    return loss

def pi_TD3_loss(batch_states_norm, domain_dict, q_nws, pi_nw, s_rms=None):
    
    domain = domain_dict['domain_name']
    max_actions = domain_dict['max_actions']
    
    if domain.startswith('RES') and max_actions is None:
        assert s_rms is not None
        # Use unnormalized states for action clamping
        batch_states = batch_states_norm * torch.sqrt(torch.tensor(s_rms.var + utils.epsilon).to(utils.device)) + torch.tensor(s_rms.mean).to(utils.device)
        pi_action = pi_nw(batch_states_norm, batch_states)
    else:
        pi_action = pi_nw(batch_states_norm)
    
    loss = q_nws[0](batch_states_norm, pi_action).mean()    
    return loss

def pi_ILBO_loss(norm_batch_states, batch_rewards, norm_batch_next_states, domain_dict, q_nw, pi_nw, s_rms, ret_rms):
    
    domain = domain_dict['domain_name']
    gamma = domain_dict['gamma']
    max_actions = domain_dict['max_actions']
    txn_distr = domain_dict['txn_distr']
    
    # When calculating the log prob, we need to use unnormalized states
    batch_states = norm_batch_states * torch.sqrt(torch.tensor(s_rms.var + utils.epsilon).to(utils.device)) + torch.tensor(s_rms.mean).to(utils.device)
    batch_next_states = norm_batch_next_states * torch.sqrt(torch.tensor(s_rms.var + utils.epsilon).to(utils.device)) + torch.tensor(s_rms.mean).to(utils.device)    
    
    if domain.startswith('RES') and max_actions is None:
        
        actions = pi_nw(norm_batch_states, batch_states)
        
        level_bef_rainfall, reward_sa = utils.rddl_nextstate_reward_txn(domain_dict, batch_states, actions, ilbo_obj = True)
        
        rainfall = batch_next_states - level_bef_rainfall
        rainfall_nonneg = nn.ReLU()(rainfall) + utils.epsilon
        log_prob_txn = torch.sum(txn_distr.log_prob(rainfall_nonneg), dim = 1)
        
        with torch.no_grad():
            V_currState = q_nw(norm_batch_states, actions)
            V_nextState = q_nw(norm_batch_next_states, pi_nw(norm_batch_next_states, batch_next_states))
        
    else: # for NAV3 and HVAC6
        
        actions = pi_nw(norm_batch_states)
        
        batch_next_states_mean, reward_sa = utils.rddl_nextstate_reward_txn(domain_dict, batch_states, actions, ilbo_obj = True)
        
        batch_diff_next_states = batch_next_states - batch_next_states_mean
        # Normalized using the mode of the distribution
        log_prob_txn = txn_distr.log_prob(batch_diff_next_states)# - txn_distr.log_prob(torch.tensor([0, 0]).to(utils.device))
        
        with torch.no_grad():
            V_currState = q_nw(norm_batch_states, actions)
            V_nextState = q_nw(norm_batch_next_states, pi_nw(norm_batch_next_states))
    
    loss = (reward_sa + gamma * (V_nextState - V_currState) * log_prob_txn[:, None]).mean()
    
    return loss

def pi_rcpo_lambda_loss(class_scores, lambdas, domain_constr_list):
    
    softmax_probs = nn.Softmax(dim = 1)
    #constr_sat_list = []
    
    class_idx_list = [tup[0] for tup in domain_constr_list]
    rel_sign_np = np.array([tup[1] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    threshold_tensor = torch.tensor([tup[2] for tup in domain_constr_list]).to(utils.device)
    
    constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * (softmax_probs(class_scores)[:, class_idx_list] - threshold_tensor)
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
        
    return loss, constr_satifs_mean

def pi_rcpo_mf_lambda_loss(sum_policy_log_prob, class_scores_nograd, lambdas, domain_constr_list):
    
    softmax_probs = nn.Softmax(dim = 1)
    #constr_sat_list = []
    
    class_idx_list = [tup[0] for tup in domain_constr_list]
    rel_sign_np = np.array([tup[1] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    threshold_tensor = torch.tensor([tup[2] for tup in domain_constr_list]).to(utils.device)
    
    if len(domain_constr_list) > 1:
        constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * sum_policy_log_prob[:, None] * (softmax_probs(class_scores_nograd)[:, class_idx_list] - threshold_tensor)
    else:
        constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * sum_policy_log_prob * (softmax_probs(class_scores_nograd)[:, class_idx_list] - threshold_tensor)
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
        
    return loss#, constr_satifs_mean

def pi_kkt_lambda_loss(class_scores, single_lambda, domain_constr):
    
    softmax_probs = nn.Softmax(dim = 1)
    
    class_idx, rel_sign, threshold = domain_constr
    
    sign = 1 if rel_sign == 'GE' else -1
    
    constr_satif = sign * (softmax_probs(class_scores)[:, class_idx] - threshold)
    constr_satif_mean = constr_satif.mean(dim = 0)
    
    loss = single_lambda * constr_satif_mean
        
    return loss, constr_satif_mean
