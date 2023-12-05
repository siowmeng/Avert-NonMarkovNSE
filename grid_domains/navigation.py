#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:55:50 2022

@author: siowmeng
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel
import utils

cell_map = {'.': 0, # Floor
            'X': 1, # Wall
            '@': 2, # Water
            'H': 3, # Human
            'G': 4, # Goal
            'S': 5} # Start

class navigation:
    
    def __init__(self, bpfile, gamma):
        # self.testprob = ['grid-3','grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7']
        self.gamma = gamma
        self.rows, self.columns = 15, 15
        self.max_horizon = 500
        self.grid = self.parseBPFile(bpfile)
        self.reinit_state()
    
    def reinit_state(self):
        
        # First nrows * ncols = One-Hot Encoding of Agent Coordinates, -1: Absorbing State
        self.init_state = torch.zeros(self.rows * self.columns + 1, dtype = torch.float32).to(utils.device) # Agent Unloaded at initial state
        # Agent Starting Location
        self.init_state[torch.where(self.grid == cell_map['S'])[0].item()] = 1
    
    def parseBPFile(self, filename):
        
        grid = torch.zeros(self.rows * self.columns, dtype = torch.int32).to(utils.device)
        
        with open(filename, 'rt') as bpdata:
            for i, line in enumerate(bpdata.readlines()):
                for j, cell in enumerate(line.rstrip()):
                    if cell.upper() in cell_map:
                        grid[i * self.columns + j] = cell_map[cell.upper()]
                    else:
                        grid[i * self.columns + j] = cell_map['.'] # Assumed to be floor
        
        return grid
    
    def checkMoveGoal(self, batch_state):
        
        # Agent Loaded & Goal Position
        return (batch_state[:, :-1] * (self.grid == cell_map['G'])).sum(dim = -1)
    
    def checkAtGoal(self, batch_state):
        
        # Agent at Absorbing State
        return (batch_state[:, -1] == 1)
        
    def highNSE(self, batch_state, batch_action):
        
        # Fast Actions & Human Position
        return (batch_action[:, torch.tensor([1, 3, 5, 7])].sum(dim = -1)) * (batch_state[:, :-1] * (self.grid == cell_map['H'])).sum(dim = -1)
    
    def lowNSE(self, batch_state, batch_action):
        
        # Fast Actions & Water Position
        return (batch_action[:, torch.tensor([1, 3, 5, 7])].sum(dim = -1)) * (batch_state[:, :-1] * (self.grid == cell_map['@'])).sum(dim = -1)
    
    def calcNSE(self, batch_state, batch_action):
        
        return (10.0 * self.highNSE(batch_state, batch_action) + 
                5.0 * self.lowNSE(batch_state, batch_action))
    
    def step(self, batch_state, batch_action, reparam=False):
        
        # Action One-Hot Encoding
        # 0: LEFT SLOW, 1: LEFT FAST, 2: RIGHT SLOW, 3: RIGHT FAST, 4: UP SLOW, 5: UP FAST, 6: DOWN SLOW, 7: DOWN FAST
        batch_prev_state = batch_state.clone()
        batch_prev_coord = batch_prev_state[:, :-1] # Excl dummy absorbing state
        
        # Movement
        prob_success = 0.9
        prob_slide = 1.0 - prob_success
        
        # Successful Move One-Hot Encoding
        # 0: LEFT, 1: RIGHT, 2: UP, 3: DOWN
        batch_move_dir = batch_action[:, torch.tensor([0, 2, 4, 6])] + batch_action[:, torch.tensor([1, 3, 5, 7])]
        batch_PMove = prob_success * batch_move_dir
        batch_PMove[:, 3] += (prob_slide * batch_move_dir[:, torch.tensor([0, 1])].sum(dim = -1)) # Slide Down
        batch_PMove[:, 1] += (prob_slide * batch_move_dir[:, torch.tensor([2, 3])].sum(dim = -1)) # Slide Right
        
        # Sample the move (One-Hot Encoding)
        if reparam:
            tau = 1.0            
            
            # Sample Gumbel(0, 1)
            gumbels_dist = Gumbel(torch.zeros_like(batch_PMove), torch.ones_like(batch_PMove))
            gumbels = gumbels_dist.sample()
            # Softmax
            gumbels = ((batch_PMove + utils.epsilon).log() + gumbels) / tau # Add epsilon to prevent log(0)
            # gumbels_exp = gumbels.exp()
            # y_soft = gumbels_exp / gumbels_exp.sum(dim = -1, keepdim = True)
            y_soft = gumbels.softmax(dim = -1)
            
            y_hard_index = y_soft.argmax(dim = -1)
            y_hard = F.one_hot(y_hard_index, num_classes = batch_PMove.shape[-1])
            
            batch_sampled_move_onehot = (y_hard - y_soft).detach() + y_soft
            
            # batch_sampled_move = F.gumbel_softmax(batch_PMove.log(), tau = 1, hard = True)
        else:
            move_dist = Categorical(probs = batch_PMove)
            batch_sampled_move = move_dist.sample()
            batch_sampled_move_onehot = F.one_hot(batch_sampled_move, num_classes = 4)
        
        # Goal or absorbing state reached
        batch_moveGoal_bool = self.checkMoveGoal(batch_state)
        batch_notMoveGoal_bool = batch_moveGoal_bool * -1 + 1
        batch_atGoal_bool = self.checkAtGoal(batch_state)
        batch_notAtGoal_bool = batch_atGoal_bool * -1 + 1
        # Not Loaded and Pick Box and Box Location
        batch_moveLeft_bool = batch_sampled_move_onehot[:, 0] * (batch_state[:, [i * self.columns for i in range(self.rows)]].sum(dim = -1) <= 0) * ((batch_state[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]] * (self.grid[[i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] == cell_map['X'])).sum(dim = -1) <= 0)
        batch_moveRight_bool = batch_sampled_move_onehot[:, 1] * (batch_state[:, [(i + 1) * self.columns - 1 for i in range(self.rows)]].sum(dim = -1) <= 0) * ((batch_state[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] * (self.grid[[i for i in range(self.rows * self.columns) if i % self.columns != 0]] == cell_map['X'])).sum(dim = -1) <= 0)
        batch_moveUp_bool = batch_sampled_move_onehot[:, 2] * (batch_state[:, :self.columns].sum(dim = -1) <= 0) * ((batch_state[:, self.columns:-1] * (self.grid[:-self.columns] == cell_map['X'])).sum(dim = -1) <= 0)
        batch_moveDown_bool = batch_sampled_move_onehot[:, 3] * (batch_state[:, -(self.columns + 1):-1].sum(dim = -1) <= 0) * ((batch_state[:, :-(self.columns + 1)] * (self.grid[self.columns:] == cell_map['X'])).sum(dim = -1) <= 0)
        
        batch_movement_bool = batch_moveUp_bool + batch_moveDown_bool + batch_moveLeft_bool + batch_moveRight_bool
        
        # Reward
        batch_atWall_bool = (batch_state[:, :-1] * (self.grid == cell_map['X'])).sum(dim = -1)
        batch_notAtWall_bool = batch_atWall_bool * -1 + 1
        batch_r = -1.0 + (batch_moveGoal_bool + batch_atGoal_bool) * 1.0 - batch_atWall_bool * 4.0
        batch_r -= batch_notMoveGoal_bool * batch_notAtGoal_bool * batch_notAtWall_bool * (batch_action[:, torch.tensor([0, 2, 4, 6])].sum(dim = -1)) * 1.0
        # NSE
        batch_c = self.calcNSE(batch_state, batch_action)
        
        # Reset State if there is movement and not at absorbing state
        batch_state[:, :-1] -= (batch_notAtGoal_bool * (batch_moveGoal_bool + batch_notMoveGoal_bool * batch_movement_bool))[:, None] * batch_prev_coord
        # Move to Absorbing State
        batch_state[:, -1] += batch_moveGoal_bool * 1 # AtGoal and MoveGoal are mutually exclusive
        # Move Up
        batch_state[:, :-(self.columns + 1)] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveUp_bool)[:, None] * batch_prev_coord[:, self.columns:]
        # Move Down
        batch_state[:, self.columns:-1] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveDown_bool)[:, None] * batch_prev_coord[:, :-self.columns]        
        # Move Left
        batch_state[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveLeft_bool)[:, None] * batch_prev_coord[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]]
        # Move Right
        batch_state[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveRight_bool)[:, None] * batch_prev_coord[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]]
                
        return batch_state, batch_r, batch_c
