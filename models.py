#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class QValueNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, x_encode_layers = [16, 32], a_encode_layers = [32], hidden_layers=[256, 256]):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.x_encode_layers = x_encode_layers
        self.a_encode_layers = a_encode_layers
        self.hidden_layers = hidden_layers
        #self.epsilon = 1e-8
        
        # build actual NN
        self.__build_model()

    def __build_model(self):
        
        self.x_encode = nn.ModuleList([])
        
        prev_x_size = self.input_dim
        for num_x_encode in self.x_encode_layers:
            self.x_encode.append(nn.Linear(prev_x_size, num_x_encode))
            self.x_encode.append(nn.ReLU())
            self.x_encode.append(nn.LayerNorm(num_x_encode))
            
            prev_x_size = num_x_encode
          
        self.a_encode = nn.ModuleList([])
        
        prev_a_size = self.input_dim
        for num_a_encode in self.a_encode_layers:
            self.a_encode.append(nn.Linear(prev_a_size, num_a_encode))
            self.a_encode.append(nn.ReLU())
            self.a_encode.append(nn.LayerNorm(num_a_encode))
            
            prev_a_size = num_a_encode
        
        self.mlp = nn.ModuleList([])
        
        prev_size = prev_x_size + prev_a_size
        for num_hidden in self.hidden_layers:
            
            self.mlp.append(nn.Linear(prev_size, num_hidden))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.LayerNorm(num_hidden))
            
            prev_size = num_hidden
        
        last_layer = nn.Linear(prev_size, self.output_dim)
        #torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.1)
        #torch.nn.init.uniform_(last_layer.weight, a=-3e-3, b=3e-3)
        self.mlp.append(last_layer)
        self.mlp.append(nn.ReLU())
        
    def forward(self, X, A):
        
        for i, l in enumerate(self.x_encode):
            X = l(X)
        
        for i, l in enumerate(self.a_encode):
            A = l(A)
        
        XA = torch.cat((X, A), 1)
        for i, l in enumerate(self.mlp):
            XA = l(XA)
        
        return XA * -1.

class PolicyNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, hidden_layers=[256, 256], last_actv=nn.Tanh(), action_bound=1.0):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.last_actv = last_actv
        self.action_bound = action_bound
        #self.epsilon = 1e-8
        
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
        #torch.nn.init.uniform_(last_layer.weight, a=-3e-3, b=3e-3)
        #torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
        #nn.init.normal_(last_layer.weight)
        
        self.mlp.append(last_layer)
        self.mlp.append(self.last_actv)
        
    def forward(self, X, X_orig = None):
                
        for i, l in enumerate(self.mlp):
            X = l(X)
        
        if self.action_bound is None:
            assert X_orig is not None
            return X * X_orig
        else:
            return X * self.action_bound.to(X.device)

class ValueNW(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, hidden_layers=[256, 256]):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        #self.epsilon = 1e-8
        
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
        #torch.nn.init.uniform_(last_layer.weight, a=-3e-3, b=3e-3)
        #torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
        #nn.init.normal_(last_layer.weight)
        
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
        # self.action_bounds = action_bounds
        #self.epsilon = 1e-8
        
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
        #torch.nn.init.uniform_(last_layer.weight, a=-3e-3, b=3e-3)
        #torch.nn.init.normal_(last_layer.weight, mean=0.0, std=0.01)
        #nn.init.normal_(last_layer.weight)
        
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
        
        # actions = actions.reshape((-1,) + self.action_space.shape)
        
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
        
        
