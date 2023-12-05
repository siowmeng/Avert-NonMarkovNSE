#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import utils

def batch_ts_danger(domain, ts_s_batch, ts_a_batch, ts_c_batch):
    
    danger_dict = dict()
    
    danger_dict['danger'] = (ts_c_batch > 0)
    
    return danger_dict

def batch_trajDF_by_nse(domain, domain_horizon, ts_s_batch, ts_a_batch, ts_r_batch, danger_dict):
    
    # Idx 0 = No NSE, Idx 1 = NSE
    dict_TrajSets = {0: [], 
                     1: []}
    prop_NSEs = {0: 0.0, 
                 1: 0.0}
    
    if domain == 'box':
        dict_TrajSets[2], dict_TrajSets[3] = [], []
        prop_NSEs[2], prop_NSEs[3] = 0.0, 0.0
    
    num_trajs = ts_s_batch.shape[0]
    
    for traj_idx in range(num_trajs):
        traj_df = pd.DataFrame()
        if domain == 'box':
            state_headings = ['Cell' + str(i) for i in range(ts_s_batch.shape[-1] - 1)] + ['Loaded']
            dummyState_heading = 'Cell' + str(ts_s_batch.shape[-1] - 2)
        else:
            state_headings = ['Cell' + str(i) for i in range(ts_s_batch.shape[-1])]
            dummyState_heading = 'Cell' + str(ts_s_batch.shape[-1] - 1)
        action_headings = ['Act' + str(i) for i in range(ts_a_batch.shape[-1])]
        
        traj_df[state_headings] = ts_s_batch[traj_idx].detach().cpu().numpy()
        traj_df[action_headings] = ts_a_batch[traj_idx].detach().cpu().numpy()
        traj_df['reward'] = ts_r_batch[traj_idx].detach().cpu().numpy()
        
        for key in danger_dict:
            traj_df[key] = danger_dict[key][traj_idx].detach().cpu().numpy()
        
        consec_danger = traj_df.danger.groupby((traj_df.danger != traj_df.danger.shift()).cumsum()).transform('size') * traj_df.danger
        max_consec_danger = consec_danger.max()
        
        if domain == 'box':
            traj_length = (traj_df[dummyState_heading] == 0.).sum()
            danger_ratio =  traj_df.danger.sum() / traj_length
            if danger_ratio < 0.1:
                if traj_length < domain_horizon:
                    dict_TrajSets[0].append(traj_df)
                else:
                    dict_TrajSets[2].append(traj_df)
            else:
                if traj_length < domain_horizon:
                    dict_TrajSets[1].append(traj_df)
                else:
                    dict_TrajSets[3].append(traj_df)
        else:
            danger_ratio = traj_df.danger.sum() / (traj_df[dummyState_heading] == 0.).sum()
            if danger_ratio < 0.1:
                dict_TrajSets[0].append(traj_df)
            else:
                dict_TrajSets[1].append(traj_df)
        
    total_trajs = sum([len(dict_TrajSets[k]) for k in dict_TrajSets.keys()])
    
    for k in dict_TrajSets.keys():
        prop_NSEs[k] = len(dict_TrajSets[k]) / total_trajs
    
    return dict_TrajSets, prop_NSEs

