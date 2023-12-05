#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os, sys, time
import pandas as pd
import numpy as np
import utils

grid_nrows, grid_ncols = 15, 15

domain_dict = {'box': [grid_nrows * grid_ncols + 2, 5, ['grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7']], 
               'nav': [grid_nrows * grid_ncols + 1, 8, ['grid-3-t6','grid-3-t4','grid-3-t3','grid-3-t5','grid-3-t7']]}
classify_epoch = {'box': 20, 
                  'nav': 10}

class GridDataset(Dataset):
    def __init__(self, root, domain, filename, XY_transform=None, L_transform=None, A_transform=None, target_transform=None):
        
        self.init_db(root, domain, filename)
        self.XY_transform = XY_transform
        self.L_transform = L_transform
        self.A_transform = A_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        
        XY = self.XY[idx]
        A = self.A[idx]
        y = self.y[idx]
        if self.XY_transform:
            XY = self.XY_transform(XY)
        if self.A_transform:
            A = self.A_transform(A)
        if self.target_transform:
            y = self.target_transform(y)
        
        if hasattr(self, 'L'):
            L = self.L[idx]
            if self.L_transform:
                L = self.L_transform(L)
            return XY, L, A, y
        
        return XY, A, y
    
    def init_db(self, root_path, domain, filename):
        
        self.XY, self.A, self.y = [], [], []
        
        stateArr, actArr, nseArr = [], [], []
        
        if domain == 'box':
            self.L, loadedArr = [], []
        
        with open(root_path + domain + '/' + filename + '_Testing.txt', 'rt') as nsedata:            
            for i, line in enumerate(nsedata.readlines()):                
                stateFeatsStr, actionStr = line.strip().split('(')[1].split(') ')
                stateFeats = [int(x) for x in stateFeatsStr.split(', ')]
                coord_x = stateFeats[0]
                coord_y = stateFeats[1]
                                
                actionEnc = actionStr.split(' ')
                actionIx = int(actionEnc[-2])
                nse_val = int(actionEnc[-1])
                if (coord_x == -1) and (coord_y == -1):
                    stateArr.append(grid_nrows * grid_ncols)
                else:
                    stateArr.append(coord_x + coord_y * grid_ncols)
                
                if domain == 'box':
                    loaded = stateFeats[2]
                    loadedArr.append(loaded)
                
                actArr.append(actionIx)
                nseArr.append(nse_val)
        
        stateArr = np.array(stateArr) # Need One Hot Encoding later on
        actArr = np.array(actArr) # Need One Hot Encoding later on
        nseArr = np.array(nseArr) # Need to classify 5 & 10 as binary
        p = np.random.permutation(len(stateArr))
        self.XY = stateArr[p]
        if domain == 'box':
            loadedArr = np.array(loadedArr) # Need to concat with stateArr later on
            self.L = loadedArr[p]
        self.A = actArr[p]
        self.y = nseArr[p]

class SAClassifier(nn.Module):
    def __init__(self, state_dim = 5, action_dim = 5, hidden_layers=[32, 32]):
        
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        
        # build actual NN
        self.__build_model()

    def __build_model(self):
        
        self.mlp = nn.ModuleList([])
        
        prev_size = self.state_dim + self.action_dim
        for num_hidden in self.hidden_layers:
            
            self.mlp.append(nn.Linear(prev_size, num_hidden))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=0.5))
            self.mlp.append(nn.BatchNorm1d(num_hidden))
            
            prev_size = num_hidden
        
        self.mlp.append(nn.Dropout(p = 0.5))
        last_layer = nn.Linear(prev_size, 1)        
        self.mlp.append(last_layer)        
        self.mlp.append(nn.Sigmoid())
    
    def forward(self, X, A):
        
        XA = torch.cat((X, A), 1)
        for i, l in enumerate(self.mlp):
            XA = l(XA)
        
        return XA
    
    def loss(self, y_hat, y):
        
        loss = nn.BCELoss()
        loss = loss(y_hat, y)

        return loss

class GridTrajSADataset(Dataset):
    def __init__(self, root, domain, train=True, X_transform=None, L_transform=None, A_transform=None, target_transform=None):
        
        self.target_map = {'safe': 0, 'unsafe': 1, 'safe-incomplete': 2, 'unsafe-incomplete': 3}
        self.init_db(root, domain, train)
        self.X_transform = X_transform
        self.A_transform = A_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        
        X = self.X[idx]
        A = self.A[idx]
        y = self.y[idx]
        if self.X_transform:
            X = self.X_transform(X)
        if self.A_transform:
            A = self.A_transform(A)
        if self.target_transform:
            y = self.target_transform(y)
        
        return X, A, y
    
    def _getColNames(self, domain, df_colnames):
        
        if domain == 'box':
            return ['Cell' + str(i) for i in range(domain_dict[domain][0] - 1)] + ['Loaded'], ['Act' + str(i) for i in range(domain_dict[domain][1])]
        else:
            return ['Cell' + str(i) for i in range(domain_dict[domain][0])], ['Act' + str(i) for i in range(domain_dict[domain][1])]
    
    def init_db(self, root_path, domain, train=True):
        
        self.X, self.A, self.y = [], [], []
        stateArr, actArr, nseArr = [], [], []
        
        phase = 'train' if train else 'test'
        
        for label in self.target_map.keys():
            
            file_path = root_path + '/' + domain + '/' + phase + '/' + label + '/'
            num_files = len([name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name)) and name.endswith('.csv')])
            
            for file_no in range(num_files): # Only use 20% trajectories for classifier training
                df = pd.read_csv(file_path + str(file_no) + '.csv')
                
                s_cols, a_cols = self._getColNames(domain, df.columns.values)
                
                dummyCell = s_cols[-2] if domain == 'box' else s_cols[-1]
                
                df = df[df[dummyCell] == 0]
                
                x_data = df[s_cols].values
                a_data = df[a_cols].values
                
                target = (self.target_map[label] > 0) * np.ones((df.shape[0]))
                
                stateArr.append(x_data)
                actArr.append(a_data)
                nseArr.append(target)
        
        stateArr = np.concatenate(stateArr, axis = 0)
        actArr = np.concatenate(actArr, axis = 0)
        nseArr = np.concatenate(nseArr, axis = 0)
        p = np.random.permutation(len(stateArr))
        self.X = stateArr[p]
        self.A = actArr[p]
        self.y = nseArr[p]          

def grid_collate(batch):
    
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

class GridTrajDataset(Dataset):
    def __init__(self, root, domain, train=True, transform=None, target_transform=None):
        
        self.target_map = {'safe': 0, 'unsafe': 1, 'safe-incomplete': 2, 'unsafe-incomplete': 3}
        self.init_ts_db(root, domain, train)
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
    
    def _getColNames(self, domain, df_colnames):
        
        if domain == 'box':
            return ['Cell' + str(i) for i in range(domain_dict[domain][0] - 1)] + ['Loaded'], ['Act' + str(i) for i in range(domain_dict[domain][1])]
        else:
            return ['Cell' + str(i) for i in range(domain_dict[domain][0])], ['Act' + str(i) for i in range(domain_dict[domain][1])]
    
    def init_ts_db(self, root_path, domain, train=True):
        
        self.X, self.y = [], []
        phase = 'train' if train else 'test'
        
        for label in self.target_map.keys():
            
            file_path = root_path + '/' + domain + '/' + phase + '/' + label + '/'
            num_files = len([name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name)) and name.endswith('.csv')])
            
            for file_no in range(num_files): # Only use 20% trajectories for classifier training
                df = pd.read_csv(file_path + str(file_no) + '.csv')
                
                s_cols, a_cols = self._getColNames(domain, df.columns.values)
                
                dummyCell = s_cols[-2] if domain == 'box' else s_cols[-1]
                
                df = df[df[dummyCell] == 0]
                
                data = np.column_stack((df[s_cols].values, df[a_cols].values))
                
                target = self.target_map[label]
                
                self.X.append(torch.FloatTensor(data))
                self.y.append(target)

class TSGRU(nn.Module):
    def __init__(self, feature_dim = 4, num_classes = 3, gru_layers = 2, nb_gru_units=100, batch_size=256):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.nb_gru_units = nb_gru_units
        self.gru_layers = gru_layers
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
        )

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

if __name__ == "__main__":
    
    try:
        domain = sys.argv[1]
        rootPath = sys.argv[2]
        if rootPath[-1] != '/':
            rootPath += '/'
        strict = sys.argv[3]
        assert strict.lower() == 'y' or strict.lower() == 'n', ""
        if strict.lower() == 'y':
            strict = True
        else:
            strict = False
        trajExpt = sys.argv[4]
        assert trajExpt.lower() == 'y' or trajExpt.lower() == 'n', ""
        if trajExpt.lower() == 'y':
            trajExpt = True
        else:
            trajExpt = False
        if (len(sys.argv) > 5):
            budget = int(sys.argv[5])
        else:
            budget = float('Inf')
    except:
        print("Errors Parsing Input Arguments")
        sys.exit(1)
    
    if domain in domain_dict:
        state_dim, action_dim, env_files = domain_dict[domain]
        # env_files = ['grid-3','grid-3-t1','grid-3-t2','grid-3-t3','grid-3-t6','grid-3-t7'] # box
        # env_files = ['grid-3','grid-3-t6','grid-3-t4','grid-3-t3','grid-3-t5','grid-3-t7'] # nav
    else:
        print("Given domain not recognized")
        sys.exit(1)
        
    feature_dim = state_dim + action_dim

    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_layers = [32, 32]
    batch_size = 100
    classify_epochs = classify_epoch[domain]
    
    if trajExpt:
        if domain == 'box':
            env_files = ['grid-3-t4']
        else:
            env_files = ['grid-3-t1']
        gru_units = 32
        classify_epochs *= 2
    
    for file in env_files:
        
        print("Domain: " + domain)
                
        if trajExpt:
            
            print("Env: " + file)
            
            # Train Trajectory Classifier first
            print("Traj Mode: Yes")
            
            load_start = time.time()
            train_dataset = GridTrajDataset(root=rootPath, 
                                            domain=domain, 
                                            train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=grid_collate, shuffle=True)            
            load_end = time.time()
            print("Data Loading Elapsed Time: %.2f secs" % (load_end - load_start))
            
            load_test_start = time.time()
            test_dataset = GridTrajDataset(root=rootPath, 
                                           domain=domain, 
                                           train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=grid_collate, shuffle=True)            
            load_test_end = time.time()
            print("Data Loading (Test) Elapsed Time: %.2f secs" % (load_test_end - load_test_start))
            
            net = TSGRU(feature_dim=feature_dim, num_classes = 4, gru_layers = 2, nb_gru_units=gru_units, batch_size=batch_size).to(utils.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)
            
            train_start = time.time()
            
            for epoch in range(classify_epochs):  # loop over the dataset multiple times
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
            
            print('Finished Training Trajectory Model')
            
            train_end = time.time()
            print("Training Elapsed Time: %.2f secs" % (train_end - train_start))
            
            save_path = rootPath + domain + '/' + file + '_GRU_' + str(gru_units) + '_' + str(batch_size) + '.pt'
            torch.save(net.state_dict(), save_path)
            
            test_size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            test_loss, correct = 0, 0
            
            tp = {0: 0, 1: 0, 2: 0, 3: 0}
            tpfp = {0: 0, 1: 0, 2: 0, 3: 0}
            tpfn = {0: 0, 1: 0, 2: 0, 3: 0}
            
            net.eval()
            
            test_start = time.time()
            
            with torch.no_grad():
                for X, y, X_lengths in test_dataloader:
                    test_outputs = net(X, X_lengths)
                    test_loss += net.loss(test_outputs, y).item()
                    correct += (test_outputs.argmax(1) == y).type(torch.float).sum().item()
                    
                    for y_label in [0, 1, 2, 3]:
                        tp[y_label] += ((test_outputs.argmax(1) == y) & (y == y_label)).type(torch.float).sum().item()
                        tpfp[y_label] += (test_outputs.argmax(1) == y_label).type(torch.float).sum().item()
                        tpfn[y_label] += (y == y_label).type(torch.float).sum().item()
            
            test_end = time.time()
            print("Testing Elapsed Time: %.2f secs" % (test_end - test_start))
            
            test_loss /= num_batches
            correct /= test_size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            for y_label in [0, 1, 2, 3]:
                print(f"Class {y_label}: \n Precision: {(100 * tp[y_label] / tpfp[y_label]):>8f}%, Recall: {(100 * tp[y_label] / tpfn[y_label]):>8f}% \n")
            
            # Train Binary Classifier Next
            print("Traj Mode: No")
            
            load_start = time.time()
            train_dataset = GridTrajSADataset(root=rootPath, 
                                              domain=domain, 
                                              train=True)            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)         
            load_end = time.time()
            print("Data Loading Elapsed Time: %.2f secs" % (load_end - load_start))
            
            load_test_start = time.time()
            test_dataset = GridTrajSADataset(root=rootPath, 
                                             domain=domain, 
                                             train=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)         
            load_test_end = time.time()
            print("Data Loading (Test) Elapsed Time: %.2f secs" % (load_test_end - load_test_start))
            
            net = SAClassifier(state_dim = state_dim, action_dim = action_dim, hidden_layers=hidden_layers).to(utils.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
            
            train_start = time.time()
                        
            # for epoch in range(classify_epochs):  # loop over the dataset multiple times
            for epoch in range(10):  # loop over the dataset multiple times
                net.train()
                running_loss = 0.0
                for i, data in enumerate(train_dataloader, 0):
                    
                    X_train, A_train, y_train = data
                    X_train = X_train.type(torch.float).to(utils.device)
                    A_train = A_train.type(torch.float).to(utils.device)
                    y_train = y_train[:, None].type(torch.float).to(utils.device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(X_train, A_train)
                    loss = net.loss(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    # print statistics
                    running_loss += loss.item()
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1))
                    running_loss = 0.0
                    
                    if (i + 1) * batch_size >= budget:
                        break
                
                net.eval()
        
                running_valid_loss = 0.0
                
                for j, valid_data in enumerate(test_dataloader, 0):
                    
                    X_valid, A_valid, y_valid = valid_data
                    X_valid = X_valid.type(torch.float).to(utils.device)
                    A_valid = A_valid.type(torch.float).to(utils.device)
                    y_valid = y_valid[:, None].type(torch.float).to(utils.device)
                    
                    valid_outputs = net(X_valid, A_valid)
                    valid_loss = net.loss(valid_outputs, y_valid)
                    
                    running_valid_loss += valid_loss.item()
                
                print('[%d] Validation loss: %.3f' % 
                      (epoch + 1, running_valid_loss / len(test_dataloader)))
        
            print('Finished Training')
            
            train_end = time.time()
            print("Training Elapsed Time: %.2f secs" % (train_end - train_start))
            
            save_path = rootPath + domain + '/' + file + '_MarkovClassifier_' 
            
            for num_units in hidden_layers:
                save_path += (str(num_units) + '_')
            
            save_path += (str(batch_size) + '.pt')
            
            torch.save(net.state_dict(), save_path)
            
            test_size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            test_loss, correct = 0, 0
            
            tp = 0
            tpfp = 0
            tpfn = 0
            
            net.eval()
            
            test_start = time.time()
            
            with torch.no_grad():
                for k, test_data in enumerate(test_dataloader, 0):
                    
                    X_test, A_test, y_test = valid_data
                    X_test = X_test.type(torch.float).to(utils.device)
                    A_test = A_test.type(torch.float).to(utils.device)
                    y_test = y_test[:, None].type(torch.float).to(utils.device)
                    
                    test_outputs = net(X_test, A_test)
                    
                    test_loss += net.loss(test_outputs, y_test).item()
                    correct += ((test_outputs >= 0.5) == y_test).type(torch.float).sum().item()
                    
                    tp += ((test_outputs >= 0.5) & (y_test == 1)).type(torch.float).sum().item()
                    tpfp += (test_outputs >= 0.5).type(torch.float).sum().item()
                    tpfn += (y_test == 1).type(torch.float).sum().item()                
            
            test_end = time.time()
            print("Testing Elapsed Time: %.2f secs" % (test_end - test_start))
            
            test_loss /= num_batches
            correct /= test_size
            print(tp)
            print(tpfp)
            print(tpfn)
            precision = 100 * tp / tpfp if tpfp > 0 else float('nan')
            recall = 100 * tp / tpfn if tpfn > 0 else float('nan')
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            print(f"Precision: {precision:>8f}%, Recall: {recall:>8f}% \n")
            
        else:
            
            print("Env File: " + file + "_Testing.txt")
            
            whole_dataset = GridDataset(root=rootPath,
                                        domain=domain, 
                                        filename=file)
            
            whole_dataloader = DataLoader(whole_dataset, batch_size=batch_size, shuffle=False)
            
            load_end = time.time()
            print("Data Loading Elapsed Time: %.2f secs" % (load_end - load_start))
            
            net = SAClassifier(state_dim = state_dim, action_dim = action_dim, hidden_layers=hidden_layers).to(utils.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
            train_start = time.time()
        
            for epoch in range(classify_epochs):  # loop over the dataset multiple times
                net.train()
                running_loss = 0.0
                for i, data in enumerate(whole_dataloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    if domain == 'box':
                        XY_train, L_train, A_train, y_train = data
                        XY_train = F.one_hot(XY_train, state_dim - 1).type(torch.float).to(utils.device)
                        L_train = L_train[:, None].type(torch.float).to(utils.device)
                        X_train = torch.cat((XY_train, L_train), dim = -1)
                        A_train = F.one_hot(A_train, action_dim).type(torch.float).to(utils.device)
                        y_train = y_train[:, None].type(torch.float).to(utils.device)
                    else:
                        XY_train, A_train, y_train = data
                        XY_train = F.one_hot(XY_train, state_dim).type(torch.float).to(utils.device)
                        X_train = XY_train
                        A_train = F.one_hot(A_train, action_dim).type(torch.float).to(utils.device)
                        y_train = y_train[:, None].type(torch.float).to(utils.device)
                    
                    if strict:
                        y_train[y_train <= 0] = False
                        y_train[y_train > 0] = True
                    else:
                        y_train[y_train <= 5] = False
                        y_train[y_train > 5] = True                    
            
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(X_train, A_train)
                    loss = net.loss(outputs, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    # print statistics
                    running_loss += loss.item()
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1))
                    running_loss = 0.0
                    
                    if (i + 1) * batch_size >= budget:
                        break
                
                net.eval()
        
                running_valid_loss = 0.0
                
                for j, valid_data in enumerate(whole_dataloader, 0):
                    
                    if domain == 'box':
                        XY_test, L_test, A_test, y_test = valid_data
                        XY_test = F.one_hot(XY_test, state_dim - 1).type(torch.float).to(utils.device)
                        L_test = L_test[:, None].type(torch.float).to(utils.device)
                        X_test = torch.cat((XY_test, L_test), dim = -1)
                        A_test = F.one_hot(A_test, action_dim).type(torch.float).to(utils.device)
                        y_test = y_test[:, None].type(torch.float).to(utils.device)
                    else:
                        XY_test, A_test, y_test = valid_data
                        XY_test = F.one_hot(XY_test, state_dim).type(torch.float).to(utils.device)
                        X_test = XY_test
                        A_test = F.one_hot(A_test, action_dim).type(torch.float).to(utils.device)
                        y_test = y_test[:, None].type(torch.float).to(utils.device)
                    
                    if strict:
                        y_test[y_test <= 0] = False
                        y_test[y_test > 0] = True                    
                    else:
                        y_test[y_test <= 5] = False
                        y_test[y_test > 5] = True
                    
                    valid_outputs = net(X_test, A_test)
                    valid_loss = net.loss(valid_outputs, y_test)
                    
                    print((j + 1) * batch_size)
                    
                    running_valid_loss += valid_loss.item()
                
                print('[%d] Validation loss: %.3f' % 
                      (epoch + 1, running_valid_loss / len(whole_dataloader)))
        
            print('Finished Training')
            
            train_end = time.time()
            print("Training Elapsed Time: %.2f secs" % (train_end - train_start))
            
            save_path = rootPath + domain + '/' + file + '_' + str(budget) + '_' + ('HA-S' if strict else 'HA-L')
            
            for num_units in hidden_layers:
                save_path += ('_' + str(num_units))
            
            save_path += '_Classifier.pt'
            torch.save(net.state_dict(), save_path)
            
            test_size = len(whole_dataloader.dataset)
            num_batches = len(whole_dataloader)
            test_loss, correct = 0, 0
            
            tp = 0
            tpfp = 0
            tpfn = 0
            
            net.eval()
            
            test_start = time.time()
            
            with torch.no_grad():
                for k, test_data in enumerate(whole_dataloader, 0):
                    
                    if domain == 'box':
                        XY_test, L_test, A_test, y_test = test_data
                        XY_test = F.one_hot(XY_test, state_dim - 1).type(torch.float).to(utils.device)
                        L_test = L_test[:, None].type(torch.float).to(utils.device)
                        X_test = torch.cat((XY_test, L_test), dim = -1)
                        A_test = F.one_hot(A_test, action_dim).type(torch.float).to(utils.device)
                        y_test = y_test[:, None].type(torch.float).to(utils.device)
                    else:
                        XY_test, A_test, y_test = test_data
                        XY_test = F.one_hot(XY_test, state_dim).type(torch.float).to(utils.device)
                        X_test = XY_test
                        A_test = F.one_hot(A_test, action_dim).type(torch.float).to(utils.device)
                        y_test = y_test[:, None].type(torch.float).to(utils.device)
                    
                    if strict:
                        y_test[y_test <= 0] = False
                        y_test[y_test > 0] = True
                    else:
                        y_test[y_test <= 5] = False
                        y_test[y_test > 5] = True
                    test_outputs = net(X_test, A_test)
                    
                    test_loss += net.loss(test_outputs, y_test).item()
                    correct += ((test_outputs >= 0.5) == y_test).type(torch.float).sum().item()
                    
                    tp += ((test_outputs >= 0.5) & (y_test == 1)).type(torch.float).sum().item()
                    tpfp += (test_outputs >= 0.5).type(torch.float).sum().item()
                    tpfn += (y_test == 1).type(torch.float).sum().item()                
            
            test_end = time.time()
            print("Testing Elapsed Time: %.2f secs" % (test_end - test_start))
            
            test_loss /= num_batches
            correct /= test_size
            print(tp)
            print(tpfp)
            print(tpfn)
            precision = 100 * tp / tpfp if tpfp > 0 else float('nan')
            recall = 100 * tp / tpfn if tpfn > 0 else float('nan')
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            print(f"Precision: {precision:>8f}%, Recall: {recall:>8f}% \n")
