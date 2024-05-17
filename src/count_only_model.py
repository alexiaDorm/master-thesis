from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pickle
import numpy as np
import copy
from functools import partial
import time
import os

from models.pytorch_datasets import BiasDataset
from models.models import TotCountOnly

#Define training loop
data_dir = "../results/"

def train():

    #Define chromosome split 
    chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']

    #Load the data
    batch_size = 64

    train_dataset = BiasDataset(data_dir + 'background_GC_matchedt.pkl', data_dir + 'ATAC_backgroundtest.pkl', chr_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 6

    biasModel = TotCountOnly(nb_conv=nb_conv, nb_filters=2**nb_filters)
    biasModel.to(device)

    criterion = nn.MSELoss(reduction='mean')

    lr = 0.0001

    optimizer = torch.optim.Adam(biasModel.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_loss = []

    nb_epoch=50
    biasModel.train() 
    for epoch in range(0, nb_epoch):
        
        running_loss, epoch_steps = 0.0, 0
        for i, data in enumerate(train_dataloader):
            
            inputs, tracks = data 
            inputs = inputs.to(device)
            tracks = tracks.to(device)

            optimizer.zero_grad()

            _, count = biasModel(inputs)
            
            tracks = tracks[:,:-1]
            counts_per_example = torch.sum(tracks, dim=1)

            loss = criterion(torch.log(counts_per_example + 1), count.squeeze())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #print every 2000 batch the loss
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
        
        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader)
        train_loss.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}')

    print('Finished Training')

    return biasModel, train_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  

biasModel, train_loss = train()

with open('../results/count_train_loss_1e-4.pkl', 'wb') as file:
        pickle.dump(train_loss, file)

