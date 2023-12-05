#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import utils

def rcpo_lambda_loss(nse_probs, lambdas, domain_constr_list):
    
    rel_sign_np = np.array([tup[0] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    threshold_tensor = torch.tensor([tup[1] for tup in domain_constr_list]).to(utils.device)
    
    constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * (nse_probs - threshold_tensor)
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
    
    return loss, constr_satifs_mean

def rcpo_mf_lambda_loss(sum_policy_log_prob, sum_scores_nograd, lambdas, domain_constr_list):
        
    rel_sign_np = np.array([tup[0] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    threshold_tensor = torch.tensor([tup[1] for tup in domain_constr_list]).to(utils.device)
    
    constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * sum_policy_log_prob * (sum_scores_nograd - threshold_tensor)
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
        
    return loss

def rcpo_mf_lambda_loss_markov(weighted_sum_policy_log_prob, lambdas, domain_constr_list):
        
    rel_sign_np = np.array([tup[0] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    
    # Threshold set to be 0
    constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * weighted_sum_policy_log_prob
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
        
    return loss

def pi_rcpo_lambda_loss(class_scores, lambdas, domain_constr_list):
    
    softmax_probs = nn.Softmax(dim = 1)
    
    class_idx_list = [tup[0] for tup in domain_constr_list]
    rel_sign_np = np.array([tup[1] for tup in domain_constr_list])
    rel_GE_IntTensor = torch.tensor(rel_sign_np == 'GE').int().to(utils.device)
    rel_LE_IntTensor = torch.tensor(rel_sign_np == 'LE').int().to(utils.device)
    threshold_tensor = torch.tensor([tup[2] for tup in domain_constr_list]).to(utils.device)
    
    constr_satifs = (rel_GE_IntTensor - rel_LE_IntTensor) * (softmax_probs(class_scores)[:, class_idx_list] - threshold_tensor)
    constr_satifs_mean = constr_satifs.mean(dim = 0)
    loss = (lambdas * constr_satifs_mean).sum()
        
    return loss, constr_satifs_mean

