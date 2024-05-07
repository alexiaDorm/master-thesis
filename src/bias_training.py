from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import copy
from functools import partial
import time
import os

from pytorch_datasets import BiasDataset
from models import BPNet
from eval_metrics import ATACloss, counts_metrics, profile_metrics

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = "/data/gpfs-1/users/adorman_m/work/master-thesis/results/"

def train(config, chr_train, chr_test):

    #Load the data
    train_dataset = BiasDataset(data_dir + 'background_GC_matched.pkl', data_dir + 'ATAC_backgroundt.pkl', chr_train)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"],
                        shuffle=True, num_workers=2)
        
    test_dataset = BiasDataset( data_dir + 'background_GC_matched.pkl', data_dir + 'ATAC_backgroundt.pkl', chr_test)
    test_dataloader = DataLoader(test_dataset, batch_size=128,
                        shuffle=True, num_workers=2)

    #Initialize model, loss, and optimizer
    biasModel = BPNet()
    biasModel.to(device)

    criterion = ATACloss(weight_MSE=config["weight_MSE"])
    optimizer = torch.optim.Adam(biasModel.parameters(), lr=config["lr"])


    best_loss, best_model_weight, patience = float('inf'), None, 5
    
    train_loss, test_loss = [], []
    corr_test, jsd_test = [], []
    for epoch in range(0, config["nb_epoch"]):
        
        biasModel.train() 
        running_loss, epoch_steps = 0.0, 0

        for i, data in tqdm(enumerate(train_dataloader)):
            prof.step()
            
            inputs, tracks = data 
            inputs = torch.reshape(inputs, (-1,4,train_dataset.len_seq)).to(device)
            tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

            optimizer.zero_grad()

            _, profile, count = biasModel.forward_train(inputs)
            
            loss = criterion(tracks, profile, count)
            writer.add_scalar("Loss/train", loss, epoch)

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
            
        epoch_loss = running_loss / len(train_dataloader)
        train_loss.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{config["nb_epoch"]}], Loss: {epoch_loss:.4f}')

        #Evaluate the model after each epoch on the test set
        val_loss, spear_corr, jsd = 0.0, 0.0, 0.0
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                inputs, tracks = data 
                inputs = torch.reshape(inputs, (-1,4,train_dataset.len_seq)).to(device)
                tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

                _, profile, count = biasModel.forward_train(inputs)

                #Compute loss
                loss = criterion(tracks, profile, count)
                writer.add_scalar("Loss/test", loss, epoch)

                val_loss += loss.cpu().item()

                #Compute evaluation metrics: pearson correlation
                corr = counts_metrics(tracks, count)
                spear_corr += corr

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = np.sum(profile_metrics(tracks, profile))
                jsd += j

        test_loss.append(val_loss /len(test_dataloader))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        writer.add_scalar("corr/test", spear_corr/len(test_dataloader), epoch)
        writer.add_scalar("jsd/test", jsd/len(test_dataloader), epoch)


        #Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weight = copy.deepcopy(biasModel.state_dict())
            patience = 5
        
        else:
            patience -= 1
        
        if patience == 0:
            break
    
    #Load best model weights
    biasModel.load_state_dict(best_model_weight)

    #Save model
    torch.save(biasModel, '../results/best_biasModel.pt')

    print('Finished Training')

    return best_model_weight, train_loss, test_loss, corr_test, jsd_test


config = {
    "weight_MSE": 1,
    "nb_epoch": 20,
    "lr": 0.005,
    "batch_size": 32
}

#Define chromosome split 
chrom_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chrom_test = ['6','13','22']

prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)
prof.start()

best_model_weight, train_loss, test_loss, corr_test, jsd_test = train(config, chrom_train, chrom_test)
writer.flush()

writer.close()

with open('../results/best_model_weight.pkl', 'wb') as file:
    pickle.dump(best_model_weight, file)
with open('../results/train_loss.pkl', 'wb') as file:
    pickle.dump(train_loss, file)
with open('../results/test_loss.pkl', 'wb') as file:
    pickle.dump(test_loss, file)
with open('../results/corr_test.pkl', 'wb') as file:
    pickle.dump(corr_test, file)
with open('../results/jsd_test.pkl', 'wb') as file:
    pickle.dump(jsd_test, file)
