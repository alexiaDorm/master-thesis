from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pickle
import numpy as np
import copy
from functools import partial
import time
import os

from models.pytorch_datasets import BiasDataset
from models.models import BPNet
from models.eval_metrics import ATACloss, counts_metrics, profile_metrics

""" #Create subset of data to check model on
with open('../results/background_GC_matched.pkl', 'rb') as file:
    sequences = pickle.load(file)   
sequences.index = sequences.chr + ":" + sequences.start.astype("str") + "-" + sequences.end.astype('str')

with open('../results/ATAC_backgroundt.pkl', 'rb') as file:
    tracks = pickle.load(file)

sequences = sequences.sample(50000, replace=False)
tracks = tracks.loc[sequences.index]

with open('../results/background_GC_matchedt.pkl', 'wb') as file:
    pickle.dump(sequences, file)

with open('../results/ATAC_backgroundtest.pkl', 'wb') as file:
    pickle.dump(tracks, file)

del sequences
del tracks """

#Define training loop
data_dir = "../results/"

def train():

    #Define chromosome split 
    chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']

    #Load the data
    batch_size = 32

    train_dataset = BiasDataset(data_dir + 'background_GC_matchedt.pkl', data_dir + 'ATAC_backgroundtest.pkl', chr_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 6

    biasModel = BPNet(nb_conv=nb_conv, nb_filters=2**nb_filters)
    biasModel.to(device)

    weight_MSE = 2
    criterion = ATACloss(weight_MSE= weight_MSE)

    lr = 0.001

    optimizer = torch.optim.Adam(biasModel.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    train_loss, train_MNLLL, train_MSE = [], [], []

    nb_epoch=150
    biasModel.train() 
    for epoch in range(0, nb_epoch):
        
        running_loss, epoch_steps = 0.0, 0
        running_MNLLL, running_MSE = 0.0, 0.0
        for i, data in enumerate(train_dataloader):
            
            inputs, tracks = data 
            inputs = inputs.to(device)
            tracks = tracks.to(device)

            optimizer.zero_grad()

            _, profile, count = biasModel(inputs)
            
            loss, MNLLL, MSE = criterion(tracks, profile, count)

            loss.backward() 
            #print(biasModel.profile_conv.weight.grad) 
            #print(biasModel.linear.weight.grad) 

            optimizer.step()

            running_loss += loss.item()
            running_MNLLL += MNLLL.item()
            running_MSE += MSE.item()

            #print every 2000 batch the loss
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
        
        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_MNLL = running_MNLLL / len(train_dataloader)
        epoch_MSE = running_MSE / len(train_dataloader)

        train_loss.append(epoch_loss)
        train_MNLLL.append(epoch_MNLL)
        train_MSE.append(epoch_MSE)

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, MNLLL: {running_MNLLL/len(train_dataloader):.4f}, MSE: {running_MSE/len(train_dataloader):.4f}')

    print('Finished Training')

    return biasModel, train_loss, train_MNLLL, train_MSE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  

biasModel, train_loss, train_MNLLL, train_MSE = train()

with open('../results/scale_train_loss_1e-3.pkl', 'wb') as file:
        pickle.dump(train_loss, file)

with open('../results/scale_train_MNLL_1e-3.pkl', 'wb') as file:
        pickle.dump(train_MNLLL, file)

with open('../results/scale_train_MSE_1e-3.pkl', 'wb') as file:
        pickle.dump(train_MSE, file)

