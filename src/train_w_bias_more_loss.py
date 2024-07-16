from tqdm import tqdm
import torch
import random 
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pickle
import numpy as np
import pandas as pd

from models.pytorch_datasets import PeaksDataset_w_bias
from models.models import CATAC_w_bias
from models.eval_metrics import ATACloss_KLD, ATACloss_MNLLL, counts_metrics, profile_metrics

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#Define training loop
data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

save_prefix = "512_MNLL"

def train():

    batch_size = 64

    #Load the data
    with open('../results/train_dataset_bias.pkl', 'rb') as file:
        train_dataset = pickle.load(file)

    train_dataloader = DataLoader(train_dataset, batch_size,
                        shuffle=True, num_workers=4)

    with open('../results/test_dataset_bias.pkl', 'rb') as file:
        test_dataset = pickle.load(file)
    
    test_dataloader = DataLoader(test_dataset, 108,
                        shuffle=True, num_workers=4)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 512
    nb_pred = len(time_order)

    size_final_conv = 4096 - (21 - 1)
    cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))

    for c in cropped:
        size_final_conv -= c
    
    #Initialize model, loss, and optimizer
    model = CATAC_w_bias(nb_conv=nb_conv, nb_filters=nb_filters, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=nb_pred, size_final_conv=size_final_conv)
        
    model = model.to(device)

    weight_MSE, weight_KLD = 2, 1
    #criterion = ATACloss_KLD(weight_MSE= weight_MSE, weight_KLD = weight_KLD)
    criterion = ATACloss_MNLLL(weight_MSE= weight_MSE)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_loss, train_KLD, train_MSE = [], [], []
    test_loss, test_KLD, test_MSE = [], [], []
    corr_test, jsd_test = [], []

    nb_epoch = 25
    model.train() 

    for epoch in range(0, nb_epoch):

        running_loss, epoch_steps = 0.0, 0
        running_KLD, running_MSE = [], []

        for i, data in enumerate(train_dataloader):
                        
                inputs, tracks, idx_skip, tn5_bias = data 
                inputs = inputs.float().to(device)
                tracks = tracks.float().to(device)
                tn5_bias = tn5_bias.float().to(device)

                optimizer.zero_grad()

                _, profile, count = model(inputs, tn5_bias)

                #Compute loss for each head
                losses = [criterion(tracks[:,:,j], profile[j], count[j], idx_skip[:,j]) for j in range(0,len(profile))]
                KLD = torch.stack([loss[1] for loss in losses]).detach();  MSE = torch.stack([loss[2] for loss in losses]).detach()
                loss = torch.stack([loss[0] for loss in losses]).nansum()

                loss.backward() 
                optimizer.step()

                running_loss += loss.item()
                running_KLD.append(KLD)
                running_MSE.append(MSE)

                """ #print every 2000 batch the loss
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                        print(
                        "[%d, %5d] loss: %.3f"
                        % (epoch + 1, i + 1, running_loss / epoch_steps)
                        ) """
        scheduler.step()

        running_KLD = torch.stack(running_KLD)
        running_KLD = torch.nansum(running_KLD, dim=0)
        running_MSE = torch.stack(running_MSE)
        running_MSE = torch.nansum(running_MSE, dim=0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_KLD = running_KLD / len(train_dataset)
        epoch_MSE = running_MSE / len(train_dataset)

        train_loss.append(epoch_loss)
        train_KLD.append(epoch_KLD)
        train_MSE.append(epoch_MSE)

        #print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, KLD: {torch.nansum(running_KLD)/len(train_dataloader):.4f}, MSE: {torch.nansum(running_MSE)/len(train_dataloader):.4f}')

        #Evaluate the model on test set after each epoch, save best performing model weights
        val_loss, spear_corr, jsd = 0.0, [], []
        running_KLD, running_MSE = [], []
        for i, data in enumerate(test_dataloader):

            with torch.no_grad():
                inputs, tracks, idx_skip, tn5_bias = data 
                inputs = inputs.float().to(device)
                tracks = tracks.float().to(device)
                tn5_bias = tn5_bias.float().to(device)

                _, profile, count = model(inputs, tn5_bias)

                #Compute loss
                losses = [criterion(tracks[:,:,j], profile[j], count[j], idx_skip[:,j]) for j in range(0,len(profile))]
                KLD = torch.stack([loss[1] for loss in losses]).detach();  MSE = torch.stack([loss[2] for loss in losses]).detach()
                loss = torch.stack([loss[0] for loss in losses]).nansum()

                val_loss += loss.item()
                running_KLD.append(KLD)
                running_MSE.append(MSE)

                #Compute evaluation metrics: pearson correlation
                corr =  [counts_metrics(tracks[:,:,j], count[j], idx_skip[:,j]) for j in range(0,len(profile))]
                corr = torch.tensor(corr)
                spear_corr.append(corr)

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = [np.nanmean(profile_metrics(tracks[:,:,j], profile[j], idx_skip[:,j])) for j in range(0,len(profile))]
                j = torch.tensor(j)
                jsd.append(j)

        running_KLD = torch.stack(running_KLD)
        running_KLD = torch.nansum(running_KLD, dim=0)
        running_MSE = torch.stack(running_MSE)
        running_MSE = torch.nansum(running_MSE, dim=0)
        spear_corr = torch.stack(spear_corr)
        spear_corr = torch.nansum(spear_corr, dim=0)
        jsd = torch.stack(jsd)
        jsd = torch.nansum(jsd, dim=0)

        test_loss.append(val_loss /len(test_dataset))
        test_KLD.append(running_KLD/len(test_dataset))
        test_MSE.append(running_MSE/len(test_dataset))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        #print(f'Epoch [{epoch + 1}/{nb_epoch}], Test loss: {val_loss /len(test_dataloader):.4f}, KLD: {running_KLD.sum()/len(test_dataloader):.4f}, MSE: {running_MSE.sum()/len(test_dataloader):.4f}, Spear corr: {spear_corr.sum()/len(test_dataloader):.4f}, JSD: {jsd.sum()/len(test_dataloader):.4f}')

        #Save every five epoch
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

