#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import datetime
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nse_label_map = {0: 'safe', 1: 'unsafe', 2: 'safe-incomplete', 3: 'unsafe-incomplete'}

def init_log_folder(logfile_path, domain, env_file, ptPath, actorObjMode, constrMode, init_lambda, seed, list_constrs, lambda_lr):
    
    rundate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lambda_lr_str = "%0.4f" % lambda_lr
    file_path = logfile_path + domain + '/' + env_file + '/' + actorObjMode + '/' + constrMode + '/' + str(init_lambda) + '/' + lambda_lr_str + '/' + str(seed) + '/'
    pathlib.Path(file_path).mkdir(parents = True, exist_ok = True)
    
    with open(file_path + 'mode.log', 'w') as f:
        f.write("Experiment Domain: " + domain)
        f.write("\n")
        f.write("Environment File: " + env_file)
        f.write("\n")
        f.write("Experiment Run Date: " + rundate)
        f.write("\n")
        f.write("PyTorch StateDict File: " + ptPath)
        f.write("\n")
        f.write("Type of Actor Objective: " + actorObjMode)
        f.write("\n")
        f.write("Mode Used to Optimize Lagrange Multipliers: " + constrMode)
        f.write("\n")
        f.write("Initial Value Used for All Lambdas: " + str(init_lambda))
        f.write("\n")
        f.write("Lambda Learning Rate: " + str(lambda_lr))
        f.write("\n")
        f.write("Seed Value: " + str(seed))
        f.write("\n")
        f.write("List of Constraints: ")
        f.write("\n")
        for constr_tup in list_constrs:
            if len(constr_tup) > 2:
                class_idx, sign, threshold = constr_tup
                sign_str = ">=" if (sign == 'GE') else "<="
                f.write("Class " + str(class_idx) + " " + sign_str + " " + str(threshold))
            else:
                sign, threshold = constr_tup
                sign_str = ">=" if (sign == 'GE') else "<="
                f.write("NSE " + sign_str + " " + str(threshold))
            f.write("\n")
    
    return file_path

def log_test_trajs(base_path, epoch, d_traj_nse, d_prop_nse, d_discRet_stats, d_undiscRet_stats, d_discNSE_stats, d_undiscNSE_stats):
    
    file_path = base_path + 'test_trajs/E' + str(epoch + 1) + '/'
    pathlib.Path(file_path).mkdir(parents = True, exist_ok = True)

    with open(file_path + 'trajectories_summary.log', 'w') as f:
        
        f.write("==================== NSE Proportions ====================")
        f.write("\n")
        
        for class_idx in sorted(d_prop_nse.keys()):
            f.write("Class " + str(class_idx) + ": " + str(d_prop_nse[class_idx]))
            f.write("\n")
        
        f.write("=================== Discounted Returns ===================")
        f.write("\n")
        
        for stat in ['Min', 'Mean', 'Median', 'Max', 'Std']:
            f.write(stat + ": " + str(d_discRet_stats[stat.lower()]))
            f.write("\n")
        
        f.write("=================== Discounted NSEs ======================")
        f.write("\n")
        
        for stat in ['Min', 'Mean', 'Median', 'Max', 'Std']:
            f.write(stat + ": " + str(d_discNSE_stats[stat.lower()]))
            f.write("\n")
        
        f.write("================== Undiscounted Returns ==================")
        f.write("\n")
        
        for stat in ['Min', 'Mean', 'Median', 'Max', 'Std']:
            f.write(stat + ": " + str(d_undiscRet_stats[stat.lower()]))
            f.write("\n")
        
        f.write("================== Undiscounted NSEs =====================")
        f.write("\n")
        
        for stat in ['Min', 'Mean', 'Median', 'Max', 'Std']:
            f.write(stat + ": " + str(d_undiscNSE_stats[stat.lower()]))
            f.write("\n")
    
    for class_idx in d_traj_nse.keys():
        
        sub_path = file_path + 'class' + str(class_idx) +'/'
        pathlib.Path(sub_path).mkdir(parents = True, exist_ok = True)
        
        for idx, traj in enumerate(d_traj_nse[class_idx]):
            full_path_prefix = sub_path + str(idx + 1)
            traj.to_csv(full_path_prefix + '.csv', index = False)

def log_summary(base_path, train=True, n_test_interval=None, **kwargs):
    
    df_dict = kwargs.copy()
    
    if train:
        file_prefix = base_path + 'train_'
        epoch_range = range(1, len(df_dict['lambdas']) + 1)
    else:
        file_prefix = base_path + 'test_'
        assert n_test_interval is not None
        df_dict['lambdas'] = [df_dict['lambdas'][0]] + df_dict['lambdas'][(n_test_interval - 1)::n_test_interval] # Only require lambdas at testing epoch
        epoch_range = [1] + [x for x in range(n_test_interval, n_test_interval * (len(df_dict['lambdas']) - 1) + 1, n_test_interval)]
    
    if isinstance(df_dict['lambdas'][0], list):
        for i in range(len(df_dict['lambdas'][0])):
            df_dict['lambda' + str(i + 1)] = [x[i] for x in df_dict['lambdas']]        
        df_dict.pop('lambdas')
    else: # Impossible clause
        # Persist to CSV and plot
        df_dict['lambda1'] = df_dict.pop('lambdas')
        
    pd.DataFrame(df_dict).to_csv(file_prefix + 'summary.csv', index = False)
    
    for k in df_dict.keys():
        if not ((k.startswith('Min')) or (k.startswith('Median')) or (k.startswith('Max'))):
            if isinstance(df_dict[k], list):
                data = np.array(df_dict[k])
            maskNaN = np.isfinite(data)
            plt.figure()
            plt.plot(np.array(epoch_range)[maskNaN], data[maskNaN])
            plt.xlabel("Epoch")
            plt.ylabel(k)
            plt.savefig(file_prefix + k + '.png')
            plt.close()

def log_nse_trajs(base_path, domain, d_traj_nse, d_prop_nse):
    
    file_path = base_path + domain + '/'
    pathlib.Path(file_path).mkdir(parents = True, exist_ok = True)
        
    for class_idx in d_traj_nse.keys():
        
        traj_list = d_traj_nse[class_idx].copy()
        random.shuffle(traj_list)
        train_test_idx = int(0.1 * len(traj_list))
        train_traj_list, test_traj_list = traj_list[train_test_idx:], traj_list[:train_test_idx]
        
        for phase, traj_vector in zip(['train', 'test'], [train_traj_list, test_traj_list]):
            sub_path = file_path + phase + '/' + nse_label_map[class_idx] +'/'
            pathlib.Path(sub_path).mkdir(parents = True, exist_ok = True)
            
            path_numFiles_txt = sub_path + 'numfiles.txt'
            
            if os.path.exists(path_numFiles_txt):
                with open(path_numFiles_txt, 'r+') as f:
                    numFiles = int(f.read())
                    f.seek(0)
                    f.write(str(numFiles + len(traj_vector)))
                    f.truncate()
            else:
                numFiles = 0
                with open(path_numFiles_txt, 'w') as f:
                    f.write(str(len(traj_vector)))
            
            for idx, traj in enumerate(traj_vector, numFiles):
                full_path_prefix = sub_path + str(idx)
                traj.to_csv(full_path_prefix + '.csv', index = False)
