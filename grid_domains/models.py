#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Gumbel

class ValueNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, hidden_layers=[32, 32]):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # build actual NN
        self.__build_model()

    def __build_model(self):
        
        self.mlp = nn.ModuleList([])
        
        prev_size = self.input_dim
        for num_hidden in self.hidden_layers:
            
            self.mlp.append(nn.Linear(prev_size, num_hidden))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.LayerNorm(num_hidden))
            
            prev_size = num_hidden
        
        last_layer = nn.Linear(prev_size, self.output_dim)
        
        self.mlp.append(last_layer)
        self.mlp.append(nn.ReLU())
        
    def forward(self, X):
        
        for i, l in enumerate(self.mlp):
            X = l(X)
        
        return X * -1.

class DiagGaussPolicyNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, hidden_layers=[256, 256], log_std_init=0.0):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.log_std_init = log_std_init
        
        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design LSTM
        self.mlp = nn.ModuleList([])
        
        prev_size = self.input_dim
        for num_hidden in self.hidden_layers:
            
            self.mlp.append(nn.Linear(prev_size, num_hidden))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.LayerNorm(num_hidden))
            
            prev_size = num_hidden
        
        last_layer = nn.Linear(prev_size, self.output_dim)
        
        self.mlp.append(last_layer)
        self.log_std = nn.Parameter(torch.ones(self.output_dim) * self.log_std_init, requires_grad=True)
        
    def forward(self, X):
        
        for i, l in enumerate(self.mlp):
            X = l(X)
        
        return Normal(X, self.log_std.exp())
    
    def get_action(self, X, deterministic = False):
        
        action_dist = self(X)
        
        if deterministic:
            actions = action_dist.mean
        else:
            actions = action_dist.rsample()
        
        log_prob = action_dist.log_prob(actions)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim = 1)
        else:
            log_prob = log_prob.sum()
        
        return actions, log_prob
    
    def eval_action(self, X, A):
        
        action_dist = self(X)
        log_prob = action_dist.log_prob(A)
        log_prob = log_prob.sum(dim = 1)
        entropy = action_dist.entropy()
        
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim = 1)            
        else:
            entropy = entropy.sum()
        
        return log_prob, entropy
        
class CategoricalPolicyNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, hidden_layers=[32, 32]):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design LSTM
        self.mlp = nn.ModuleList([])
        
        prev_size = self.input_dim
        for num_hidden in self.hidden_layers:
            
            self.mlp.append(nn.Linear(prev_size, num_hidden))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.LayerNorm(num_hidden))
            
            prev_size = num_hidden
        
        last_layer = nn.Linear(prev_size, self.output_dim)
        
        self.mlp.append(last_layer)
        
    def forward(self, X):
        
        for i, l in enumerate(self.mlp):
            X = l(X.clone()) # Categorical seems to have inplace operation issue if clone() is not used
        
        return Categorical(logits = X)
    
    def get_action(self, X, deterministic = False, reparam = False):
        
        action_dist = self(X)
        
        if reparam:
            
            tau = 1.0
            
            # Sample Gumbel(0, 1)
            if deterministic:
                gumbels = 0.0
            else:             
                gumbels_dist = Gumbel(torch.zeros_like(action_dist.logits), torch.ones_like(action_dist.logits))
                gumbels = gumbels_dist.sample()
            
            # Softmax
            gumbels = (action_dist.logits + gumbels) / tau
            # gumbels_exp = gumbels.exp()
            # y_soft = gumbels_exp / gumbels_exp.sum(dim = -1, keepdim = True)
            y_soft = gumbels.softmax(dim = -1)
            
            y_hard_index = y_soft.argmax(dim = -1)
            y_hard = F.one_hot(y_hard_index, num_classes = self.output_dim)
            
            actions_onehot = (y_hard - y_soft).detach() + y_soft
            # actions_onehot = y_soft
            actions = actions_onehot.argmax(dim=-1)
            
            # actions_onehot = action_dist.logits.softmax(dim = -1)
            # actions = actions_onehot.argmax(dim=-1)
            
        else:
            if deterministic:
                actions = torch.argmax(action_dist.probs, dim = -1)
            else:
                actions = action_dist.sample()
            actions_onehot = F.one_hot(actions, num_classes = self.output_dim)
        
        log_prob = action_dist.log_prob(actions) # Log Prob requires index value, so argmax converts onehot to index
        
        return actions_onehot, log_prob
    
    def eval_action(self, X, A):
        
        action_dist = self(X)
        log_prob = action_dist.log_prob(A.argmax(dim = 1))
        entropy = action_dist.entropy()
        
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim = 1)
        else:
            entropy = entropy.sum()
        
        return log_prob, entropy
