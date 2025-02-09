#Paths assume run from src folder

#Train the model without the bias correction, the model and all performance metric are saved every 5 epochs

#Note you may want to change the prefix appended to the saved results and any hyperparameters. Important to note that "KLD" is still used throught the code and saved loss even if MNLL is used.
#--------------------------------------------

import torch
import random 
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pickle
import numpy as np
import pandas as pd

from models.pytorch_datasets import PeaksDataset_wo_bias
from models.models import CATAC_wo_bias
from models.eval_metrics import ATACloss_KLD, ATACloss_MNLLL, counts_metrics, profile_metrics

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

#Define training loop
data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

save_prefix = "128_wobias"

def train():

    batch_size = 128

    #Load the data
    with open('../results/train_dataset_wobias.pkl', 'rb') as file:
        train_dataset = pickle.load(file)

    train_dataloader = DataLoader(train_dataset, batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    with open('../results/test_dataset_wobias.pkl', 'rb') as file:
        test_dataset = pickle.load(file)
     
    test_dataloader = DataLoader(test_dataset, 128,
                        shuffle=True, num_workers=4, pin_memory=True)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 128
    nb_pred = len(time_order)

    size_final_conv = 4096 - (21 - 1)
    cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))

    for c in cropped:
        size_final_conv -= c
    
    #Initialize model, loss, and optimizer
    model = CATAC_wo_bias(nb_conv=nb_conv, nb_filters=nb_filters, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=nb_pred, size_final_conv=size_final_conv)
        
    model = model.to(device)

    weight_MSE, weight_KLD = 1, 1
    #criterion = ATACloss_KLD(weight_MSE= weight_MSE, weight_KLD = weight_KLD)
    criterion = ATACloss_MNLLL(weight_MSE= weight_MSE)

    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_loss, train_KLD, train_MSE = [], [], []
    test_loss, test_KLD, test_MSE = [], [], []
    corr_test, jsd_test = [], []

    nb_epoch = 40
    model.train() 

    for epoch in range(0, nb_epoch):

        running_loss = torch.zeros((1), device=device)
        running_KLD, running_MSE = torch.zeros((4), device=device), torch.zeros((4), device=device)
        for i, data in enumerate(train_dataloader):

            inputs, tracks, idx_skip = data 
            inputs = inputs.to(device, dtype=torch.float32)
            tracks = tracks.to(device, dtype=torch.float32)
            idx_skip = idx_skip.to(device)

            optimizer.zero_grad()

            _, profile, count = model(inputs)

            #Compute loss for each head
            loss, KLD, MSE  = criterion(tracks, profile, count, idx_skip)

            loss.backward() 
            optimizer.step()

            running_loss += loss.detach()
            running_KLD += KLD.detach()
            running_MSE += MSE.detach()
        
        scheduler.step()

        epoch_loss = running_loss.cpu() / len(train_dataset)
        epoch_KLD = running_KLD.cpu() / len(train_dataset)
        epoch_MSE = running_MSE.cpu() / len(train_dataset)

        train_loss.append(epoch_loss)
        train_KLD.append(epoch_KLD)
        train_MSE.append(epoch_MSE)

        #Evaluate the model on test set after each epoch, save best performing model weights
        val_loss, spear_corr, jsd = torch.zeros((1), device=device), [], []
        running_KLD, running_MSE = torch.zeros((4), device=device), torch.zeros((4), device=device)
        for i, data in enumerate(test_dataloader):
            
           
            with torch.no_grad():
                inputs, tracks, idx_skip = data 
                inputs = inputs.to(device, dtype=torch.float32)
                tracks = tracks.to(device, dtype=torch.float32)
                idx_skip = idx_skip.to(device)

                _, profile, count = model(inputs)

                #Compute loss
                loss, KLD, MSE  = criterion(tracks, profile, count, idx_skip)

                val_loss += loss
                running_KLD += KLD
                running_MSE += MSE

                #Compute evaluation metrics: pearson correlation
                corr =  [counts_metrics(tracks[:,:,j], count[:,j], idx_skip[:,j]) for j in range(0,profile.size(-1))]
                corr = torch.tensor(corr)
                spear_corr.append(corr)

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = [profile_metrics(tracks[:,:,j], profile[:,:,j], idx_skip[:,j]) for j in range(0,profile.size(-1))]
                j = torch.tensor(j)
                jsd.append(j)

        spear_corr = torch.stack(spear_corr)
        spear_corr = torch.nansum(spear_corr, dim=0)
        jsd = torch.stack(jsd)
        jsd = torch.nansum(jsd, dim=0)

        test_loss.append(val_loss.cpu() /len(test_dataset))
        test_KLD.append(running_KLD.cpu()/len(test_dataset))
        test_MSE.append(running_MSE.cpu()/len(test_dataset))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        #Save every three epoch
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), '../results/' + save_prefix + '_model.pkl')

            with open('../results/' + save_prefix + '_train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)

            with open('../results/' + save_prefix + '_train_KLD.pkl', 'wb') as file:
                pickle.dump(train_KLD, file)

            with open('../results/' + save_prefix + '_train_MSE.pkl', 'wb') as file:
                pickle.dump(train_MSE, file)

            with open('../results/' + save_prefix + '_test_KLD.pkl', 'wb') as file:
                pickle.dump(test_KLD, file)

            with open('../results/' + save_prefix + '_test_MSE.pkl', 'wb') as file:
                pickle.dump(test_MSE, file)

            with open('../results/' + save_prefix + '_corr.pkl', 'wb') as file:
                pickle.dump(corr_test, file)

            with open('../results/' + save_prefix + '_jsd.pkl', 'wb') as file:
                pickle.dump(jsd_test, file)
    
    
    print('Finished Training')

    return model, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  

model, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test = train()

