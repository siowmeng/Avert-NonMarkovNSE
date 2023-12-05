#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os, sys, time, re
import pandas as pd
import numpy as np
import utils
from rddl_GenNSESamples import rddl_dict

rddl_classify_epoch = {'NAV3' : 20, 
                       'HVAC6': 20, 
                       'RES20': 20}

def rddl_collate(batch):
    
    batch_size = len(batch)
    batch_feat_dim = batch[0][0].shape[1]
    batch_lengths = [item[0].shape[0] for item in batch]
    batch_max_length = max(batch_lengths)
    
    X_tensor = Variable(torch.zeros((batch_size, batch_max_length, batch_feat_dim)).to(utils.device)).float()
    
    y_tensor = []
    
    for idx, (item, target) in enumerate(batch):
        X_tensor[idx, :item.shape[0], :] = item
        y_tensor.append(target)
    
    y_tensor = torch.LongTensor(y_tensor).to(utils.device)
    batch_lengths = torch.tensor(batch_lengths).to(utils.device)
    
    return X_tensor, y_tensor, batch_lengths

class RDDLDataset(Dataset):
    def __init__(self, root, rddl_domain, train=True, transform=None, target_transform=None):
        
        self.target_map = {'safe': 0, 'middle': 1, 'unsafe': 2}
        self.init_ts_db(root, rddl_domain, train)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y
    
    def _getColNames(self, rddl_domain, df_colnames):
        if rddl_domain.startswith('NAV'):
            return ['location1', 'location2'], ['move1', 'move2']        
        elif rddl_domain.startswith('HVAC'):
            num_cols = len([col for col in df_colnames if col.startswith('air')])
            return ['temp' + str(i + 1) for i in range(num_cols)], ['air' + str(i + 1) for i in range(num_cols)]
        elif rddl_domain.startswith('RES'):
            num_cols = len([col for col in df_colnames if col.startswith('outflow')])
            return ['rlevel' + str(i + 1) for i in range(num_cols)], ['outflow' + str(i + 1) for i in range(num_cols)]
    
    def init_ts_db(self, root_path, rddl_domain, train=True):
        
        self.X, self.y = [], []
        phase = 'train' if train else 'test'
        
        for label in self.target_map.keys():
            
            file_path = root_path + '/' + rddl_domain + '/' + phase + '/' + label + '/'
            num_files = len([name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name)) and name.endswith('.csv')])
            
            for file_no in range(num_files): # Only use 20% trajectories for classifier training
                df = pd.read_csv(file_path + str(file_no) + '.csv')
                
                s_cols, a_cols = self._getColNames(rddl_domain, df.columns.values)
                
                data = np.column_stack((df[s_cols].values, df[a_cols].values))
                
                target = self.target_map[label]
                
                self.X.append(torch.FloatTensor(data).to(utils.device))
                self.y.append(target)

class TSGRU(nn.Module):
    def __init__(self, feature_dim = 4, num_classes = 3, nb_gru_units=100, batch_size=256):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.nb_gru_units = nb_gru_units
        self.gru_layers = 4
        self.batch_size = batch_size
        
        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design LSTM
        self.gru = nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=self.nb_gru_units, 
            num_layers=self.gru_layers, 
            batch_first=True, 
            dropout=0.5
        )#.to(utils.device)

        # output layer which projects back to tag space
        self.class_output = nn.Linear(self.nb_gru_units, self.num_classes)#.to(utils.device)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_h = torch.zeros(self.gru_layers, self.batch_size, self.nb_gru_units).to(utils.device)

        hidden_h = Variable(hidden_h)

        return hidden_h
    
    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()
        
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths.cpu(), enforce_sorted=False, batch_first=True)
        
        # now run through LSTM
        X, self.hidden = self.gru(X, self.hidden)
        
        X_unpack = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=False, padding_value=0.0)
        meanH = X_unpack[0].sum(dim=0) / X_unpack[1][:, None].to(utils.device)
        
        #y_hat = self.class_output(self.hidden[-1])
        y_hat = self.class_output(meanH)
        
        return y_hat
    
    def loss(self, y_hat, y):
        
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)

        return loss


class TSLSTM(nn.Module):
    def __init__(self, feature_dim = 4, num_classes = 3, nb_lstm_units=100, batch_size=256):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.nb_lstm_units = nb_lstm_units
        self.lstm_layers = 4
        self.batch_size = batch_size
        
        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, 
            hidden_size=self.nb_lstm_units, 
            num_layers=self.lstm_layers, 
            batch_first=True, 
            dropout=0.5
        )#.to(utils.device)

        # output layer which projects back to tag space
        self.class_output = nn.Linear(self.nb_lstm_units, self.num_classes)#.to(utils.device)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.lstm_layers, self.batch_size, self.nb_lstm_units).to(utils.device)
        hidden_b = torch.zeros(self.lstm_layers, self.batch_size, self.nb_lstm_units).to(utils.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
    
    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = X.size()
        
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths.cpu(), enforce_sorted=False, batch_first=True)
        
        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)
        
        X_unpack = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=False, padding_value=0.0)
        meanH = X_unpack[0].sum(dim=0) / X_unpack[1][:, None].to(utils.device)
        
        #y_hat = self.class_output(self.hidden[0][-1])
        y_hat = self.class_output(meanH)
        
        return y_hat
    
    def loss(self, y_hat, y):
        
        loss = nn.CrossEntropyLoss()
        loss = loss(y_hat, y)

        return loss

if __name__ == "__main__":
    
    try:
        domain = sys.argv[1]
        rootPath = sys.argv[2]
        if rootPath[-1] != '/':
            rootPath += '/'
    except:
        print("No domain specified")
        sys.exit(1)
    
    if domain in rddl_dict:
        if domain.startswith('NAV'):
            feature_dim = 4
        else:
            feature_dim = int(re.findall(r'\d+', domain)[-1]) * 2
    else:
        print("Given domain not recognized")
        sys.exit(1)

    seed = 999
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    batch_size = 128
    
    load_start = time.time()
    train_dataset = RDDLDataset(root=rootPath,#'.', 
                                rddl_domain=domain, 
                                train=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=rddl_collate, shuffle=True)
    load_end = time.time()
    print("Data Loading Elapsed Time: %.2f secs" % (load_end - load_start))
    

    load_test_start = time.time()
    test_dataset = RDDLDataset(root=rootPath,#'.', 
                               rddl_domain=domain, 
                               train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=rddl_collate, shuffle=True)
    
    load_test_end = time.time()
    print("Data Loading (Test) Elapsed Time: %.2f secs" % (load_test_end - load_test_start))
    
    net = TSGRU(feature_dim=feature_dim, nb_gru_units=64, batch_size=batch_size).to(utils.device) #HVAC6 - 4 layers of 256, 0.0008 #RES20 (200k) - 4 layers of 256, 0.0001 (79%) 200 epochs
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)#, momentum=0.9)
    
    train_start = time.time()
    
    for epoch in range(rddl_classify_epoch[domain]):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, input_lengths = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs, input_lengths)
            loss = net.loss(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        
        net.eval()

        running_valid_loss = 0.0
        
        for j, valid_data in enumerate(test_dataloader, 0):
            inputs_valid, labels_valid, input_lengths_valid = valid_data
            valid_outputs = net(inputs_valid, input_lengths_valid)
            valid_loss = net.loss(valid_outputs, labels_valid)
            
            running_valid_loss += valid_loss.item()
        
        print('[%d] Validation loss: %.3f' % 
              (epoch + 1, running_valid_loss / len(test_dataloader)))

    print('Finished Training')
    
    train_end = time.time()
    print("Training Elapsed Time: %.2f secs" % (train_end - train_start))
    
    torch.save(net.state_dict(), domain + '_classifier_statedict.pt')
    
    test_size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    
    tp = {0: 0, 1: 0, 2: 0}
    tpfp = {0: 0, 1: 0, 2: 0}
    tpfn = {0: 0, 1: 0, 2: 0}
    
    net.eval()
    
    test_start = time.time()
    
    with torch.no_grad():
        for X, y, X_lengths in test_dataloader:
            test_outputs = net(X, X_lengths)
            test_loss += net.loss(test_outputs, y).item()
            correct += (test_outputs.argmax(1) == y).type(torch.float).sum().item()
            
            for y_label in [0, 1, 2]:
                tp[y_label] += ((test_outputs.argmax(1) == y) & (y == y_label)).type(torch.float).sum().item()
                tpfp[y_label] += (test_outputs.argmax(1) == y_label).type(torch.float).sum().item()
                tpfn[y_label] += (y == y_label).type(torch.float).sum().item()
    
    test_end = time.time()
    print("Testing Elapsed Time: %.2f secs" % (test_end - test_start))
    
    test_loss /= num_batches
    correct /= test_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    for y_label in [0, 1, 2]:
        print(f"Class {y_label}: \n Precision: {(100 * tp[y_label] / tpfp[y_label]):>8f}%, Recall: {(100 * tp[y_label] / tpfn[y_label]):>8f}% \n")
