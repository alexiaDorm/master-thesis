from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pickle
import numpy as np

from models.pytorch_datasets import PeaksDataset2
from models.models import CATAC2
from models.eval_metrics import ATACloss_KLD, counts_metrics, profile_metrics

""" #Create subset of data to check model on
with open('../results/peaks_seq.pkl', 'rb') as file:
    sequences = pickle.load(file)   

sequences.index = sequences.chr.astype("str") + ":" + sequences.start.astype("str") + "-" + sequences.end.astype('str')

with open('../results/ATAC_peaks1.pkl', 'rb') as file:
    tracks = pickle.load(file)

sequences = sequences.sample(5000, replace=False)
tracks = tracks.loc[sequences.index]

with open('../results/peaks_seqtest.pkl', 'wb') as file:
    pickle.dump(sequences, file)

with open('../results/ATAC_peakstest.pkl', 'wb') as file:
    pickle.dump(tracks, file)

del sequences
del tracks """

#Define training loop
data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22']

def train():

    #Define chromosome split 
    chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
    chr_test = ['6','13','22']

    batch_size = 32

    #Load the data
    train_dataset = PeaksDataset2(data_dir + 'peaks_seqtest.pkl', data_dir + 'background_GC_matchedt.pkl',
                                 data_dir + 'ATAC_peakstest.pkl', data_dir + 'ATAC_backgroundtest.pkl', 
                                 chr_train, time_order, 0)
    train_dataloader = DataLoader(train_dataset, batch_size,
                        shuffle=True, num_workers=4)

    test_dataset = PeaksDataset2(data_dir + 'peaks_seqtest.pkl', data_dir + 'background_GC_matchedt.pkl',
                                 data_dir + 'ATAC_peakstest.pkl', data_dir + 'ATAC_backgroundtest.pkl', 
                                 chr_test, time_order, 0)
    test_dataloader = DataLoader(test_dataset, 108,
                        shuffle=True, num_workers=4)

    #Initialize model, loss, and optimizer
    nb_conv = 8
    nb_filters = 6
    nb_pred = len(time_order)
    
    #Initialize model, loss, and optimizer
    model = CATAC2(nb_conv=nb_conv, nb_filters=2**nb_filters, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=nb_pred)
        
    model = model.to(device)

    weight_MSE, weight_KLD = 1, 1
    criterion = ATACloss_KLD(weight_MSE= weight_MSE, weight_KLD = weight_KLD)
    #criterion = ATACloss_MNLLL(weight_MSE= weight_MSE)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    train_loss, train_KLD, train_MSE = [], [], []
    test_loss, test_KLD, test_MSE = [], [], []
    corr_test, jsd_test = [], []

    nb_epoch = 1
    model.train() 
    for epoch in range(0, nb_epoch):

        running_loss, epoch_steps = 0.0, 0
        running_KLD, running_MSE = [], []
        for i, data in enumerate(train_dataloader):

            if i ==2:
                break

            inputs, tracks, idx_skip = data 
            inputs = inputs.to(device)
            tracks = tracks.to(device)
            
            idx_skip = torch.stack(idx_skip)
            idx_skip = idx_skip != -1

            optimizer.zero_grad()

            _, profile, count = model(inputs)

            #Compute loss for each head
            losses = [criterion(tracks[:,j,:], profile[j], count[j], idx_skip[j,:]) for j in range(0,len(profile))]
            KLD = torch.stack([loss[1] for loss in losses]).detach();  MSE = torch.stack([loss[2] for loss in losses]).detach()
            loss = torch.stack([loss[0] for loss in losses]).nansum()

            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            running_KLD.append(KLD)
            running_MSE.append(MSE)

            #print every 2000 batch the loss
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
            
        scheduler.step()

        running_KLD = torch.stack(running_KLD)
        running_KLD = torch.nansum(running_KLD, dim=0)
        print(running_KLD.shape)

        running_MSE = np.array(running_MSE)
        running_MSE = np.nansum(running_MSE, axis=0)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_KLD = running_KLD / len(train_dataloader)
        epoch_MSE = running_MSE / len(train_dataloader)

        train_loss.append(epoch_loss)
        train_KLD.append(epoch_KLD)
        train_MSE.append(epoch_MSE)

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, KLD: {np.nansum(running_KLD)/len(train_dataloader):.4f}, MSE: {np.nansum(running_MSE)/len(train_dataloader):.4f}')

        #Evaluate the model on test set after each epoch, save best performing model weights
        val_loss, spear_corr, jsd = 0.0, [], []
        running_KLD, running_MSE = [], []
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                inputs, tracks, idx_skip = data 
                inputs = inputs.to(device)
                tracks = tracks.to(device)
                
                idx_skip = torch.stack(idx_skip)
                idx_skip = idx_skip != -1

                _, profile, count = model(inputs)

                #Compute loss
                losses = [criterion(tracks[:,j,:], profile[j], count[j], idx_skip[j,:]) for j in range(0,len(profile))]
                KLD = torch.stack([loss[1] for loss in losses]).detach();  MSE = torch.stack([loss[2] for loss in losses]).detach()
                loss = torch.stack([loss[0] for loss in losses]).nansum()

                val_loss += loss.item()
                running_KLD.append(KLD)
                running_MSE.append(MSE)

                #Compute evaluation metrics: pearson correlation
                corr =  [counts_metrics(tracks[:,j,:], count[j], idx_skip[j,:]) for j in range(0,len(profile))]
                spear_corr.append(corr)

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = [np.nanmean(profile_metrics(tracks[:,j,:], profile[j], idx_skip[j,:])) for j in range(0,len(profile))]
                jsd.append(j)

        running_KLD = np.array(running_KLD)
        running_KLD = np.nansum(running_KLD, axis=0)
        running_MSE = np.array(running_MSE)
        running_MSE = np.nansum(running_MSE, axis=0)
        spear_corr = np.array(spear_corr)
        spear_corr = np.nansum(spear_corr, axis=0)
        jsd = np.array(jsd)
        jsd = np.nansum(jsd, axis=0)

        test_loss.append(val_loss /len(test_dataloader))
        test_KLD.append(running_KLD/len(test_dataloader))
        test_MSE.append(running_MSE/len(test_dataloader))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Test loss: {val_loss /len(test_dataloader):.4f}, KLD: {running_KLD.sum()/len(test_dataloader):.4f}, MSE: {running_MSE.sum()/len(test_dataloader):.4f}, Spear corr: {spear_corr.sum()/len(test_dataloader):.4f}, JSD: {jsd.sum()/len(test_dataloader):.4f}')


    print('Finished Training')

    return model, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)  

model, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test = train()

torch.save(model.state_dict(), '../results/model_1e-3.pkl')

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

