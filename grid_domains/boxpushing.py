#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:55:50 2022

@author: siowmeng
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel
# import numpy as np
import utils

cell_map = {'.': 0, # Floor
            'x': 1, # Wall
            '@': 2, # Holes
            'F': 3, # Fragiles
            'G': 4, # Goal
            'B': 5, # Box
            'S': 6} # Start

class boxpushing:
    
    def __init__(self, bpfile, gamma):
        # self.testprob = ['grid-3','grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7']
        self.gamma = gamma
        # self.rows, self.columns = 3, 3
        self.rows, self.columns = 15, 15
        self.max_horizon = 500
        self.grid = self.parseBPFile(bpfile)
        self.reinit_state()
    
    def reinit_state(self):
        
        # First nrows * ncols = One-Hot Encoding of Agent Coordinates, -2: Absorbing State, -1: Loaded Bool
        self.init_state = torch.zeros(self.rows * self.columns + 2, dtype = torch.float32).to(utils.device) # Agent Unloaded at initial state
        # Agent Starting Location
        self.init_state[torch.where(self.grid == cell_map['S'])[0].item()] = 1
                
        # agent_x, agent_y = torch.where(self.grid == cell_map['S'])
        # box_x, box_y = torch.where(self.grid == cell_map['B'])
        # goal_x, goal_y = torch.where(self.grid == cell_map['G'])
        
        # self.init_state = torch.cat((agent_x, agent_y, torch.tensor([False]).to(utils.device))).type(torch.float32).to(utils.device)
        # self.init_box_loc = (box_x.item(), box_y.item())
        # self.goal_loc = (goal_x.item(), goal_y.item())
    
    def parseBPFile(self, filename):
        
        grid = torch.zeros(self.rows * self.columns, dtype = torch.int32).to(utils.device)
        
        with open(filename, 'rt') as bpdata:
            for i, line in enumerate(bpdata.readlines()):
                for j, cell in enumerate(line.rstrip()):
                    if cell in cell_map:
                        grid[i * self.columns + j] = cell_map[cell]
                    else:
                        grid[i * self.columns + j] = cell_map['.'] # Assumed to be floor
        
        return grid
    
    def checkMoveGoal(self, batch_state):
        
        # Agent Loaded & Goal Position
        return (batch_state[:, -1] == 1) * (batch_state[:, :-2] * (self.grid == cell_map['G'])).sum(dim = -1)
        # No need to be loaded (Testing Purpose)
        # return (batch_state[:, :-2] * (self.grid == cell_map['G'])).sum(dim = -1)
    
    def checkAtGoal(self, batch_state):
        
        # Agent at Absorbing State
        return (batch_state[:, -2] == 1)
    
    def checkLoad(self, batch_state, batch_action):
        
        # Pick Action
        actLoad = batch_action[:, -1]#.type(torch.bool)
        # Agent Not Loaded & Box Position
        canLoad = (batch_state[:, -1] == 0) * (batch_state[:, :-2] * (self.grid == cell_map['B'])).sum(dim = -1)
        
        return actLoad * canLoad
        
    def highNSE(self, batch_state):
        
        # Agent Loaded & Hole Position
        return (batch_state[:, -1] == 1) * (batch_state[:, :-2] * (self.grid == cell_map['@'])).sum(dim = -1)
    
    def lowNSE(self, batch_state):
        
        # Agent Loaded & Fragile Position
        return (batch_state[:, -1] == 1) * (batch_state[:, :-2] * (self.grid == cell_map['F'])).sum(dim = -1)
    
    def calcNSE(self, batch_state, batch_action):
        
        return (10.0 * self.highNSE(batch_state) + 
                5.0 * self.lowNSE(batch_state))
    
    def step(self, batch_state, batch_action, reparam=False):
        
        # Action One-Hot Encoding
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: PICK
        batch_prev_state = batch_state.clone()
        batch_prev_coord = batch_prev_state[:, :-2] # Excl dummy absorbing state
        
        # Movement
        prob_success = 0.95
        prob_slide = (1.0 - prob_success) / 2.0
        
        batch_move_act = batch_action[:, :4]#.clone()
        
        # Successful Move One-Hot Encoding
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        batch_PMove = prob_success * batch_move_act + prob_slide * batch_move_act.logical_not() - prob_slide * batch_move_act[:, torch.tensor([1, 0, 3, 2])]
        
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
        
        # Zero out the sampled move if agent is performing pick action
        batch_notPick = (batch_action[:, [-1]] <= 0)
        batch_sampled_move_onehot *= batch_notPick
        
        # Goal or absorbing state reached
        batch_moveGoal_bool = self.checkMoveGoal(batch_state)
        batch_notMoveGoal_bool = batch_moveGoal_bool * -1 + 1
        batch_atGoal_bool = self.checkAtGoal(batch_state)
        batch_notAtGoal_bool = batch_atGoal_bool * -1 + 1
        # Not Loaded and Pick Box and Box Location
        batch_moveUp_bool = batch_sampled_move_onehot[:, 0] * (batch_state[:, :self.columns].sum(dim = -1) <= 0) * ((batch_state[:, self.columns:-2] * (self.grid[:-self.columns] == cell_map['x'])).sum(dim = -1) <= 0)
        batch_moveDown_bool = batch_sampled_move_onehot[:, 1] * (batch_state[:, -(self.columns + 2):-2].sum(dim = -1) <= 0) * ((batch_state[:, :-(self.columns + 2)] * (self.grid[self.columns:] == cell_map['x'])).sum(dim = -1) <= 0)
        batch_moveLeft_bool = batch_sampled_move_onehot[:, 2] * (batch_state[:, [i * self.columns for i in range(self.rows)]].sum(dim = -1) <= 0) * ((batch_state[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]] * (self.grid[[i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] == cell_map['x'])).sum(dim = -1) <= 0)
        batch_moveRight_bool = batch_sampled_move_onehot[:, 3] * (batch_state[:, [(i + 1) * self.columns - 1 for i in range(self.rows)]].sum(dim = -1) <= 0) * ((batch_state[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] * (self.grid[[i for i in range(self.rows * self.columns) if i % self.columns != 0]] == cell_map['x'])).sum(dim = -1) <= 0)
        batch_movement_bool = batch_moveUp_bool + batch_moveDown_bool + batch_moveLeft_bool + batch_moveRight_bool
        batch_load = self.checkLoad(batch_state, batch_action)
        # batch_pick_bool = batch_action[:, 4].type(torch.bool)
        
        # Reward
        batch_r = -1.0 + (batch_moveGoal_bool + batch_atGoal_bool) * 1.0
        # NSE
        batch_c = self.calcNSE(batch_state, batch_action)
        
        # Reset State if there is movement and not at absorbing state
        batch_state[:, :-2] -= (batch_notAtGoal_bool * (batch_moveGoal_bool + batch_notMoveGoal_bool * batch_movement_bool))[:, None] * batch_prev_coord
        # Move to Absorbing State
        batch_state[:, -2] += batch_moveGoal_bool * 1 # AtGoal and MoveGoal are mutually exclusive
        # Move Up
        batch_state[:, :-(self.columns + 2)] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveUp_bool)[:, None] * batch_prev_coord[:, self.columns:]
        # Move Down
        batch_state[:, self.columns:-2] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveDown_bool)[:, None] * batch_prev_coord[:, :-self.columns]        
        # Move Left
        batch_state[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveLeft_bool)[:, None] * batch_prev_coord[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]]
        # Move Right
        batch_state[:, [i for i in range(self.rows * self.columns) if i % self.columns != 0]] += (batch_notAtGoal_bool * batch_notMoveGoal_bool * batch_moveRight_bool)[:, None] * batch_prev_coord[:, [i for i in range(self.rows * self.columns) if (i + 1) % self.columns != 0]]        
        # Pick Up
        batch_state[:, -1] += batch_load
                
        return batch_state, batch_r, batch_c # Obs, Reward, NSE
