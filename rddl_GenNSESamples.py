#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
#import torch.nn as nn
import sys
import pathlib
import rddlgym
#import numpy as np
#from collections import OrderedDict
import nse, utils

rddl_dict = {'NAV3' : 'Navigation-v3', 
             'HVAC6': 'HVAC-6_Corrected.rddl', 
             'RES20': 'Reservoir-20_Corrected.rddl', 
             'RES10': 'Reservoir-10'}

# Number of Batches, Batchsize, offsets
num_ep = {'NAV3': {'train': (130, 256, (-5, 0, 5)), 'test': (13, 256, (-5, 0, 5))}, 
          'HVAC6': {'train': (130, 256, (-5, 0, 5)), 'test': (13, 256, (-5, 0, 5))}, 
          'RES20': {'train': (260, 256, (-5, 0, 5)), 'test': (26, 256, (-5, 0, 5))}}

if __name__ == "__main__":
    
    try:
        domain = sys.argv[1]
    except:
        print("No domain specified")
        sys.exit(1)
    
    if domain in rddl_dict:
        rddl_id = rddl_dict[domain]
    else:
        print("Given domain not recognized")
        sys.exit(1)
    
    env = rddlgym.make(rddl_id, mode=rddlgym.GYM)
    
    gamma = 0.99
    domain_dict = utils.construct_rddl_domain_dict(domain, gamma, env)
    
    if domain.startswith('NAV'):
        state_min = torch.tensor([0.] * domain_dict['num_states'])
        state_max = torch.tensor([10.] * domain_dict['num_states'])
        
    elif domain.startswith('HVAC'):
        state_min = torch.tensor([0.] * domain_dict['num_states'])
        state_max = torch.tensor([30.] * domain_dict['num_states'])
        
    elif domain.startswith('RES'):
        state_min = torch.tensor([0.] * domain_dict['num_states'])
        state_max = torch.tensor(env.non_fluents['MAX_RES_CAP/1'])
    
    for phase in num_ep[domain].keys():
        
        print("Start generating " + phase + " trajectory samples for " + domain + "...")
        
        num_batches, batchsize, offsets = num_ep[domain][phase]
        trajDFs_by_nse = {0: [], 
                          1: [], 
                          2: []}
        
        for i in range(num_batches):
            
            print("Batch " + str(i + 1))
            
            state, t = env.reset()
            X0_pretrain = utils.sample_init_states(torch.tensor(state[domain_dict['state_label']]).to(utils.device), batchsize)            
            
            for offset in offsets:
                ts_s, ts_a, ts_r, Js, ts_lengths = utils.sample_policy_traj_noise(X0_pretrain, domain_dict, mode='pretrain', horizon_offset=offset)
                ts_danger_dict = nse.batch_ts_danger(domain_dict, ts_s, ts_a)
                dict_TrajSets, prop_NSEs = nse.batch_trajDF_by_nse(domain_dict, ts_s, ts_a, ts_r, ts_danger_dict)
                
                for nse_class in dict_TrajSets:
                    trajDFs_by_nse[nse_class] += dict_TrajSets[nse_class]
        
        print("Persisting " + domain + " " + phase + " data to respective directories...")
        
        for label, dfs_dict_key in zip(['safe', 'middle', 'unsafe'], sorted(trajDFs_by_nse.keys())):
            
            file_path = './' + domain + '/' + phase + '/' + label + '/'
            pathlib.Path(file_path).mkdir(parents = True, exist_ok = True)
            
            for idx, df in enumerate(trajDFs_by_nse[dfs_dict_key]):
                df.to_csv(file_path + str(idx) + '.csv')