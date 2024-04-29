from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import copy
from functools import partial
import time

from ray import tune
from ray.train import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from pytorch_datasets import PeaksDataset
from models import CATAC
from eval_metrics import ATACloss, counts_metrics, profile_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train(config, chr_train, chr_test):

    #Load the data
    train_dataset = PeaksDataset('../results/peaks_seq1.pkl', '../results/background_GC_matched.pkl',
                                 '../results/ATAC_peaks1.pkl', '../results/ATAC_background1.pkl', 
                                 chr_train, pseudo_bulk_order, nb_back)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"],
                        shuffle=True, num_workers=2)
    
    test_dataset = PeaksDataset('../results/peaks_seq1.pkl', '../results/background_GC_matched.pkl',
                                 '../results/ATAC_peaks1.pkl', '../results/ATAC_background1.pkl', 
                                 chr_test, pseudo_bulk_order, nb_back)    
    test_dataloader = DataLoader(test_dataset, batch_size=128,
                        shuffle=True, num_workers=2)

    #Initialize model, loss, and optimizer
    model = CATAC(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, profile_kernel_size=75, out_pred_len=1024, 
                      nb_pred=1, nb_cell_type_CN = 0):
)    
    model.to(device)

    criterion = ATACloss(weight_MSE=config["weight_MSE"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    """ #Torcheck is used to catched common issues in model class definition: weights not training or become nan or inf
    torcheck.register(optimizer)
    torcheck.add_module_changing_check(biasModel, module_name="my_model")
    torcheck.add_module_nan_check(biasModel)
    torcheck.add_module_inf_check(biasModel) """

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    best_loss, best_model_weight, patience = float('inf'), None, 5
    
    train_loss, test_loss = [], []
    corr_test, jsd_test = [], []
    for epoch in range(start_epoch, config["nb_epoch"]):
        
        model.train() 
        running_loss, epoch_steps = 0.0, 0

        for i, data in tqdm(enumerate(train_dataloader)):
            inputs, _, _, tracks = data 
            inputs = torch.reshape(inputs, (-1,4,2114)).to(device)
            tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

            optimizer.zero_grad()

            _, profile, count = model(inputs)
            
            loss = criterion(tracks, profile, count)
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
                inputs = torch.reshape(inputs, (-1,4,2114)).to(device)
                tracks = torch.stack(tracks, dim=1).type(torch.float32).to(device)

                _, profile, count = model(inputs)

                #Compute loss
                loss = criterion(tracks, profile, count)
                val_loss += loss.cpu().item()

                #Compute evaluation metrics: pearson correlation
                corr = counts_metrics(tracks, count)
                spear_corr += corr

                #Compute the Jensen-Shannon divergence distance between actual read profile and predicted profile 
                j = profile_metrics(tracks, profile)
                jsd += j

        test_loss.append(val_loss /len(test_dataloader))
        corr_test.append(spear_corr/len(test_dataloader))
        jsd_test.append(jsd/len(test_dataloader))

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / len(test_dataloader), 
             "count_correlation": spear_corr/len(test_dataloader), 
             "profile_jsd": jsd/len(test_dataloader)},
            checkpoint=checkpoint,
        )

        #Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weight = copy.deepcopy(model.state_dict())
            patience = 5
        
        else:
            patience -= 1
        
        if patience == 0:
            break
    
    #Load best model weights
    model.load_state_dict(best_model_weight)

    print('Finished Training')

""" config = {
    "weight_MSE": tune.choice([2 ** i for i in range(9)]),
    "nb_epoch": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([8,16,32,64])
}  """

config = {
    "weight_MSE": 1,
    "nb_epoch": 10,
    "lr": 0.005,
    "batch_size": 32
}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=20,
        grace_period=1,
        reduction_factor=2,
    )

#Define chromosome split 
chrom_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chrom_test = ['6','13''22']

pseudo_bulk_order = []
nb_back = 0

#train(config, chrom_train, chrom_test)

result = tune.run(
    partial(train, chr_train=chrom_train, chr_test=chrom_test),
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    checkpoint_at_end=False)

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(f"Best trial final validation correlation for count head: {best_trial.last_result['count_correlation']}")
print(f"Best trial final validation jsd for profile head: {best_trial.last_result['profile_jsd']}")