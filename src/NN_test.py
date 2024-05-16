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

from pytorch_datasets import BiasDataset
from models import BPNet
from eval_metrics import ATACloss, counts_metrics, profile_metrics

import optuna
from optuna.trial import TrialState

#import torcheck

data_dir = "../results/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Define chromosome split 
chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chr_test = ['6','13','22']

#Load the data
batch_size = 5

train_dataset = BiasDataset(data_dir + 'background_GC_matched.pkl', data_dir + 'ATAC_backgroundt.pkl', chr_train)
train_dataloader = DataLoader(train_dataset, batch_size=2**batch_size,
                    shuffle=True, num_workers=4)

#Initialize model, loss, and optimizer
nb_conv=4
nb_filters=6

biasModel = BPNet(nb_conv=nb_conv, nb_filters=2**nb_filters)
biasModel.to(device)

weight_MSE = 2
criterion = ATACloss(weight_MSE= weight_MSE)

lr = 0.001
optimizer = torch.optim.Adam(biasModel.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

def train(data, epoch):
        
    biasModel.train() 
    running_loss, epoch_steps = 0.0, 0
    running_MNLLL, running_MSE = 0.0, 0.0
    for i, d in tqdm(enumerate(data)):
            
        inputs, tracks = d 
        inputs = torch.reshape(inputs, (-1,4,train_dataset.len_seq)).to(device)
        tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

        optimizer.zero_grad()

        _, profile, count = biasModel(inputs)
            
        loss, MNLLL, MSE = criterion(tracks, profile, count)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_MNLLL += MNLLL.item()
        running_MSE += MSE.item()

        #print every 2000 batch the loss
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "loss: %.3f"
                % (running_loss / epoch_steps)
            )
        
    scheduler.step()

    epoch_loss = running_loss / len(train_dataloader)

    print(f'Epoch [{epoch + 1}/{10}], Loss: {epoch_loss:.4f}, MNLLL: {running_MNLLL/len(train_dataloader):.4f}, MSE: {running_MSE/len(train_dataloader):.4f}')

    print('Finished Training')

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/BiasModel'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        prof.step() 
        if step >= 1 + 1 + 3:
            break
        train(batch_data, step)