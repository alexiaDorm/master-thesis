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
from models.eval_metrics import ATACloss_KLD, counts_metrics, profile_metrics

#Create subset of data to check model on
with open('../results/background_GC_matched.pkl', 'rb') as file:
    sequences = pickle.load(file)   
sequences.index = sequences.chr + ":" + sequences.start.astype("str") + "-" + sequences.end.astype('str')

with open('../results/ATAC_background1.pkl', 'rb') as file:
    tracks = pickle.load(file)

sequences = sequences.sample(50000, replace=False)
tracks = tracks.loc[sequences.index]

with open('../results/background_GC_matchedt.pkl', 'wb') as file:
    pickle.dump(sequences, file)

with open('../results/ATAC_backgroundtest.pkl', 'wb') as file:
    pickle.dump(tracks, file)

del sequences
del tracks

#Define training loop
data_dir = "../results/"

def train():

    #Define chromosome split 
    chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
    chr_test = ['6','13','22']

    #Load the data
    batch_size = 32

    train_dataset = BiasDataset(data_dir + 'background_GC_matchedt.pkl', data_dir + 'ATAC_backgroundtest.pkl', chr_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    test_dataset = BiasDataset(data_dir + 'background_GC_matchedt.pkl', data_dir + 'ATAC_backgroundtest.pkl', chr_test)
    test_dataloader = DataLoader(test_dataset, batch_size=108,
                        shuffle=True, num_workers=0)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 6

    nb_epoch_profile = 0

    biasModel = BPNet(nb_conv=nb_conv, nb_filters=2**nb_filters)
    biasModel.to(device)

    weight_MSE, weight_KLD = 0, 1
    criterion = ATACloss_KLD(weight_MSE= weight_MSE, weight_KLD = weight_KLD)

    lr = 0.001

    optimizer = torch.optim.Adam(biasModel.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_loss, train_KLD, train_MSE = [], [], []
    test_loss, test_KLD, test_MSE = [], [], []
    corr_test, jsd_test = [], []

    best_loss = float('inf')

    nb_epoch = 100
    biasModel.train() 
    for epoch in range(0, nb_epoch):

        if epoch == nb_epoch_profile:
            for group in optimizer.param_groups:
                group['lr'] = lr

        if epoch > (nb_epoch_profile - 1)  and epoch < (nb_epoch_profile + 50):
            criterion = ATACloss_KLD(weight_MSE = (epoch - nb_epoch_profile)/50 * 1)
        
        running_loss, epoch_steps = 0.0, 0
        running_KLD, running_MSE = 0.0, 0.0
        for i, data in enumerate(train_dataloader):

            inputs, tracks = data 
            inputs = inputs.to(device)
            tracks = tracks.to(device)

            optimizer.zero_grad()

            _, profile, count = biasModel(inputs)
            
            loss, KLD, MSE = criterion(tracks, profile, count)

            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            running_KLD += KLD.item()
            running_MSE += MSE.item()
        
        scheduler.step()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_KLD = running_KLD / len(train_dataloader)
        epoch_MSE = running_MSE / len(train_dataloader)

        train_loss.append(epoch_loss)
        train_KLD.append(epoch_KLD)
        train_MSE.append(epoch_MSE)

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, KLD: {running_KLD/len(train_dataloader):.4f}, MSE: {running_MSE/len(train_dataloader):.4f}')

        #Evaluate the model on test set after each epoch, save best performing model weights
        val_loss, spear_corr, jsd = 0.0, 0.0, 0.0
        running_KLD, running_MSE = 0.0, 0.0
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                inputs, tracks = data 
                inputs = inputs.to(device)
                tracks = tracks.to(device)

                _, profile, count = biasModel(inputs)

                #Compute loss
                loss, KLD, MSE = criterion(tracks, profile, count)

                val_loss += loss.item()
                running_KLD += KLD.item()
                running_MSE += MSE.item()

                #Compute evaluation metrics: pearson correlation
                corr = counts_metrics(tracks, count)
                spear_corr += corr

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = np.nanmean(profile_metrics(tracks, profile))
                jsd += j

        test_loss.append(val_loss /len(test_dataloader))
        test_KLD.append(running_KLD/len(test_dataloader))
        test_MSE.append(running_MSE/len(test_dataloader))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Test loss: {val_loss /len(test_dataloader):.4f}, KLD: {running_KLD/len(test_dataloader):.4f}, MSE: {running_MSE/len(test_dataloader):.4f}, Spear corr: {spear_corr/len(test_dataloader):.4f}, JSD: {jsd/len(test_dataloader):.4f}')

        #Save model with best val loss
        #if test_loss < best_loss: 
            

    print('Finished Training')

    return biasModel, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  

biasModel, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test = train()

with open('../results/train_loss_1e-3.pkl', 'wb') as file:
        pickle.dump(train_loss, file)

with open('../results/train_KLD_1e-3.pkl', 'wb') as file:
        pickle.dump(train_KLD, file)

with open('../results/train_MSE_1e-3.pkl', 'wb') as file:
        pickle.dump(train_MSE, file)

with open('../results/test_KLD_1e-3.pkl', 'wb') as file:
        pickle.dump(test_KLD, file)

with open('../results/test_MSE_1e-3.pkl', 'wb') as file:
        pickle.dump(test_MSE, file)

with open('../results/corr_1e-3.pkl', 'wb') as file:
        pickle.dump(corr_test, file)

with open('../results/jsd_1e-3.pkl', 'wb') as file:
        pickle.dump(jsd_test, file)