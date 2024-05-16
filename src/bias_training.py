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

import optuna
from optuna.trial import TrialState

#import torcheck

data_dir = "../results/"

def train(trial):

    #Define chromosome split 
    chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
    chr_test = ['6','13','22']

    #Load the data
    batch_size = trial.suggest_int("batch_size", 4, 7)

    train_dataset = BiasDataset(data_dir + 'background_GC_matched.pkl', data_dir + 'ATAC_background1.pkl', chr_train)
    train_dataloader = DataLoader(train_dataset, batch_size=2**batch_size,
                        shuffle=True, num_workers=3)
        
    test_dataset = BiasDataset( data_dir + 'background_GC_matched.pkl', data_dir + 'ATAC_background1.pkl', chr_test)
    test_dataloader = DataLoader(test_dataset, batch_size=128,
                        shuffle=True, num_workers=3)

    #Initialize model, loss, and optimizer
    nb_conv= trial.suggest_int("nb_conv", 4, 8)
    nb_filters= trial.suggest_int("nb_filters", 4,7)

    biasModel = BPNet(nb_conv=nb_conv, nb_filters=2**nb_filters)
    biasModel.to(device)

    weight_MSE = 2
    criterion = ATACloss(weight_MSE= weight_MSE)

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    optimizer = torch.optim.Adam(biasModel.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    best_loss, best_model_weight, patience = float('inf'), None, 5
    
    train_loss, test_loss = [], []
    corr_test, jsd_test = [], []

    """ torcheck.register(optimizer)
    torcheck.add_module_changing_check(biasModel, module_name="my_model")
    torcheck.add_module_nan_check(biasModel)
    torcheck.add_module_inf_check(biasModel) """
    
    nb_epoch=25
    for epoch in range(0, nb_epoch):
        
        biasModel.train() 
        running_loss, epoch_steps = 0.0, 0
        running_MNLLL, running_MSE = 0.0, 0.0
        for i, data in tqdm(enumerate(train_dataloader)):
            
            #prof.step()
            inputs, tracks = data 
            inputs = torch.reshape(inputs, (-1,4,train_dataset.len_seq)).to(device)
            tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

            optimizer.zero_grad()

            _, profile, count = biasModel(inputs)
            
            loss, MNLLL, MSE = criterion(tracks, profile, count)
            #writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
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
        train_loss.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, MNLLL: {running_MNLLL/len(train_dataloader):.4f}, MSE: {running_MSE/len(train_dataloader):.4f}')

        #Evaluate the model after each epoch on the test set
        val_loss, spear_corr, jsd = 0.0, 0.0, 0.0
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                inputs, tracks = data 
                inputs = torch.reshape(inputs, (-1,4,train_dataset.len_seq)).to(device)
                tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

                _, profile, count = biasModel(inputs)

                #Compute loss
                loss, MNLLL, MSE = criterion(tracks, profile, count)
                #writer.add_scalar("Loss/test", loss, epoch)

                val_loss += loss.cpu().item()

                #Compute evaluation metrics: pearson correlation
                corr = counts_metrics(tracks, count)
                spear_corr += corr

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = np.nanmean(profile_metrics(tracks, profile))
                jsd += j

        test_loss.append(val_loss /len(test_dataloader))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        #writer.add_scalar("corr/test", spear_corr/len(test_dataloader), epoch)
        #writer.add_scalar("jsd/test", jsd/len(test_dataloader), epoch)


        """ #Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weight = copy.deepcopy(biasModel.state_dict())
            patience = 5
        
        else:
            patience -= 1
        
        if patience == 0:
            break """
    
    #Load best model weights
    #biasModel.load_state_dict(best_model_weight)

    print('Finished Training')

    return biasModel, best_model_weight, train_loss, test_loss, corr_test, jsd_test

def objective(trial):
    
    model, best_model_weight, train_loss, test_loss, corr_test, jsd_test = train(trial)

    #Save model
    #torch.save(model, '../results/best_biasModel.pt')

    with open('../results/train_loss.pkl', 'wb') as file:
        pickle.dump(train_loss, file)
    with open('../results/test_loss.pkl', 'wb') as file:
        pickle.dump(test_loss, file)
    with open('../results/corr_test.pkl', 'wb') as file:
        pickle.dump(corr_test, file)
    with open('../results/jsd_test.pkl', 'wb') as file:
        pickle.dump(jsd_test, file)


    return min(test_loss)

OPTUNA_EARLY_STOPING = 10

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    if EarlyStoppingExceeded.best_score == None:
      EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value < EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
      if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            best_score = None
            raise EarlyStoppingExceeded()
      else:
            EarlyStoppingExceeded.early_stop_count=EarlyStoppingExceeded.early_stop_count+1
    print(f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
    return

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(objective, n_trials=10, timeout=600, callbacks=[early_stopping_opt])
    except EarlyStoppingExceeded:
        print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))