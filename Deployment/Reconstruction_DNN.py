#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[19]:


import torch, pytz
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, csv
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, xlabel
import streamlit as st
from PIL import Image


# ## Model Architecture

# In[20]:


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class DNN(nn.Module):
  def __init__(self, config):
    super(DNN, self).__init__()
    
    self.config = config

    dropout_rate = self.config['dropout_rate']
    num_features = self.config['num_features']
    num_layers = self.config['num_layers']
    hidden_size = self.config['hidden_size']
    
    self.linear1 = nn.Linear(num_features-1, hidden_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p = dropout_rate)
    self.linear2 = []
    for i in range(self.config['num_layers']):
        self.linear2.append(nn.Linear(hidden_size, hidden_size))
        self.linear2.append(self.dropout)
        self.linear2.append(self.relu)
    self.net = nn.Sequential(*self.linear2)
    self.linear3 = nn.Linear(hidden_size, 1)
    
  def forward(self, x):
    
    # -> batch, num_features-1
    
    x = self.linear1(x) # -> batch, hidden_size
    x = self.dropout(x)
    x = self.relu(x)
    
    x = self.net(x)
    
    outputs = self.linear3(x) # -> batch, 1
    outputs = torch.squeeze(outputs)

    return outputs


  def cal_loss(self, pred, labels):
    if self.config['loss_function'] == 'MSE':
        self.criterion = nn.MSELoss()
    elif self.config['loss_function'] == 'RMSE':
        self.criterion = RMSELoss()
    else:
        print('This loss function doesn\'t exist!')
        raise Exception('WrongLossFunctionError')
    
    return self.criterion(pred, labels)


# ## Data Preprocessing

# In[21]:


class BicepCurlDataset(Dataset):
  def __init__(self, file, mode, config):
    '''num_aug = number of augmentation for each trial
      len_seg = length of segmentation for each sample
      num_feature = number of features including data and label'''
    
    self.mode = mode
    num_feature = config['num_features']
    self.avg = config['mean']
    self.std = config['std']

    #Check if path is correct
    if self.mode not in ['Test','Val', 'Train']:
        print('This mode doesn\'t exist, try \'Train\', \'Val\', or \'Test\'')
        raise Exception('WrongModeError')
      

    if self.mode in ['Train', 'Val']:

        xy = file[:, 1:]

        for j in range(num_feature-1):
                xy[:, j] = (xy[:,j]-config['mean'][0, j])/config['std'][0, j]

        self.xdata = xy[:, :num_feature-1]
        self.ydata = xy[:, 3]
            
        
        self.xdata = torch.from_numpy(self.xdata).float()
        self.ydata = torch.from_numpy(self.ydata).float()
        self.dim = self.xdata.shape[1]
        self.length = self.xdata.shape[0]
        #Here, dim does not include label

    else:
        #Allocate data and label
        self.xdata = []
        self.ydata = []

        xy = file[:, 1:]

        for j in range(num_feature-1):
            xy[:, j] = (xy[:,j]-config['mean'][0, j])/config['std'][0, j]
        
        times_plot = np.linspace(0, xy.shape[0]/150, xy.shape[0])

        
        plt.plot(times_plot, xy[:, 0])
        plt.title('Voltage Signals from Elbow')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.ioff()
        plt.savefig('elbow_data.png')
        plt.close()

        
        plt.plot(times_plot, xy[:, 1])
        plt.title('Voltage Signals from Bicep')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.ioff()
        plt.savefig('bicep_data.png')
        plt.close()

        
        plt.plot(times_plot, xy[:, 3])
        plt.title('Elbow Angle from Mocap')
        plt.xlabel('Time [s]')
        plt.ylabel('Elbow Angle [$^\circ$]')
        plt.ioff()
        plt.savefig('angle_data.png')
        plt.close()

        self.xdata.append(torch.from_numpy(xy[:, :num_feature-1]).float())
        self.ydata.append(xy[:, 3])
      
        self.dim = self.xdata[0].shape[1]
        #Here, dim does not include label

        self.length = len(self.xdata)

    print('Finished reading the {} set of BicepCurl Dataset ({} samples found, each dim = {})'
                .format(mode, self.length, self.dim))


  def __getitem__(self, index):
    #if self.mode in ['Train', 'Val']:
        # For training
    #    return self.xdata[index], self.ydata[index]
    #else:
        # For testing (no target)
    #    return self.xdata[index]
    return self.xdata[index], self.ydata[index]

  def __len__(self):
    # Returns the size of the dataset
    return self.length


# ## Dataset and DataLoader

# In[22]:

@st.cache(suppress_st_warning=True)
def prep_dataloader(file, mode, batch_size, n_jobs, config):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = BicepCurlDataset(file, mode=mode, config = config)  # Construct dataset

    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'Train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# ## Load Model

# In[23]:
@st.cache(suppress_st_warning=True)
def Reconstruct(file):
    print('Reconstruct...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'n_epochs': 100,                # maximum number of epochs
        'batch_size': 32,               # mini-batch size for dataloader
        'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
        'optim_hparas': {                # hyper-parameters for the optimizer
            'lr': 0.001, 
            'weight_decay': 10**(-4)                           
        },
        'early_stop': 20,               # early stopping epochs (the number epochs since your model's last improvement)
        'num_features': 3,
        'loss_function': 'RMSE',
        'lr_scheduler': 'ExponentialLR',
        'lr_scheduler_hparas':{
            'gamma': 0.9,
        },
        'hidden_size': 256,
        'num_layers': 3,
        'dropout_rate': 0,
        'mean': np.array([[0.30896, 0.3175, 1.2814]]),
        'std': np.array([[0.14742, 0.089428, 0.85991]])
    }


    test_set = prep_dataloader(file, mode = 'Test', batch_size=1, n_jobs = 0, config=config)
    model = DNN(config).to(device)
    ckpt = torch.load('/app/human_digital_twin/Deployment/model.pth', map_location='cpu')  # Load the best model
    model.load_state_dict(ckpt)

    preds, targets = test(test_set, model, device)  
    save_pred(preds, targets)         # save prediction file to pred.csv


# ## Testing

# In[24]:

@st.cache(suppress_st_warning=True)
def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    targets = []
    for x, y in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu().numpy())   # collect prediction
            targets.append(y)
    #preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds, targets

@st.cache(suppress_st_warning=True)
def save_pred(preds, targets):
    
    print('Saving results...')
    

    for index, i in enumerate(preds):
        with open('results.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['TimeId', 'Elbow Angle (preds)', 'Elbow Angle (targets)'])
            for j in range(i.shape[0]):
                writer.writerow([j, i[j], targets[index][0, j].detach().cpu().item()])
                
    for index, i in enumerate(preds):
        with open('results.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            preds_plot = []
            targets_plot = []
    
            length = 0
    
            for index2, row in enumerate(reader):
                if index2 == 0:
                    continue
            
                preds_plot.append(float(row[1]))
                targets_plot.append(float(row[2]))
                length+=1
                
            times_plot = np.linspace(0, length/150, length)
                
            plt.plot(times_plot, preds_plot, c='tab:red', label='preds')
            plt.plot(times_plot, targets_plot, c='tab:cyan', label='targets')
            plt.xlabel('Time [s]')
            plt.ylabel('Elbow Angle [$^\circ$]')
            plt.title('Reconstruction of Elbow Angle')
            plt.legend()
            
            plt.ioff()
            plt.savefig('plot.png')
            plt.close()


