#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
#import torch.nn as nn
import numpy as np
import pandas as pd
import utils

def batch_ts_danger(domain_dict, ts_s_batch, ts_a_batch):
    
    danger_dict = dict()
    
    if domain_dict['domain_name'].startswith('NAV'):
        
        # Rectangle 1
        danger1 = torch.all((ts_s_batch >= torch.tensor([2., 0.]).to(utils.device)) & (ts_s_batch <= torch.tensor([4.5, 10.]).to(utils.device)), dim = 2)
        # Logical OR for all 3 ponds, landing in ANY of the 3 ponds constitutes NSE
        danger_dict['danger'] = danger1
        
    elif domain_dict['domain_name'].startswith('HVAC'):
        danger_dict['danger'] = (ts_s_batch[:, :, 1] >= 21)
        
    elif domain_dict['domain_name'].startswith('RES'):
        state_max = domain_dict['res_cap']
        ratio_rLevel = ts_s_batch / state_max
        danger_dict['danger'] = ((ratio_rLevel.max(dim = 2).values - ratio_rLevel.min(dim = 2).values) >= 0.5)
    
    return danger_dict

def batch_trajDF_by_nse(domain_dict, ts_s_batch, ts_a_batch, ts_r_batch, danger_dict):
    
    # Idx 0 = None, Idx 1 = Mild, Idx 2 = Severe
    dict_TrajSets = {0: [], 
                     1: [], 
                     2: []}
    prop_NSEs = {0: 0.0, 
                 1: 0.0, 
                 2: 0.0}
    
    num_trajs = ts_s_batch.shape[0]
    
    for traj_idx in range(num_trajs):
        traj_df = pd.DataFrame()
        state_prefix = domain_dict['state_label'].split('/')[0]
        action_prefix = domain_dict['action_label'].split('/')[0]
        traj_df[[state_prefix + str(i + 1) for i in range(ts_s_batch.shape[-1])]] = ts_s_batch[traj_idx].detach().cpu().numpy()
        traj_df[[action_prefix + str(i + 1) for i in range(ts_a_batch.shape[-1])]] = ts_a_batch[traj_idx].detach().cpu().numpy()
        traj_df['reward'] = ts_r_batch[traj_idx].detach().cpu().numpy()
        
        for key in danger_dict:
            traj_df[key] = danger_dict[key][traj_idx].detach().cpu().numpy()
        
        if domain_dict['domain_name'].startswith('NAV'):
            if np.sum(traj_df['danger'].values) > 3:
                dict_TrajSets[2].append(traj_df)
            elif np.sum(traj_df['danger'].values) > 1:
                dict_TrajSets[1].append(traj_df)
            else:
                dict_TrajSets[0].append(traj_df)
            
        else: # For HVAC and RES
            consec_danger = traj_df.danger.groupby((traj_df.danger != traj_df.danger.shift()).cumsum()).transform('size') * traj_df.danger
            max_consec_danger = consec_danger.max()
            
            if max_consec_danger > 3:
                dict_TrajSets[2].append(traj_df)
            elif max_consec_danger > 1:
                dict_TrajSets[1].append(traj_df)
            else:
                dict_TrajSets[0].append(traj_df)
        
    total_trajs = sum([len(dict_TrajSets[k]) for k in dict_TrajSets.keys()])
    for k in dict_TrajSets.keys():
        prop_NSEs[k] = len(dict_TrajSets[k]) / total_trajs
    
    return dict_TrajSets, prop_NSEs

