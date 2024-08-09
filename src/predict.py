import numpy as np
import pandas as pd
import torch
import glob
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random

from models.models import CATAC_wo_bias, CATAC_w_bias, CATAC_w_bias_increase_filter

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_order = ['D8', 'D12', 'D20', 'D22-15']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#Load model and train/test sets
path_model = '../results/train_res/128_MNLL_model.pkl'
model = CATAC_w_bias(nb_conv=8, nb_filters=128, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
model = model.to(device)

with open('../results/train_dataset_bias.pkl', 'rb') as file:
        train_dataset = pickle.load(file)

train_dataloader = DataLoader(train_dataset, 128,
                        shuffle=True, num_workers=4, pin_memory=True)

with open('../results/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)
    
test_dataloader = DataLoader(test_dataset, 128,
                        shuffle=True, num_workers=4)

#Predict for each sequence
pred_track, target_track = [], []
for i, data in enumerate(test_dataloader):
    with torch.no_grad():    
        
        inputs, tracks, idx_skip, tn5_bias = data 
        target_track.append(target_track)
        inputs = inputs.to(device, dtype=torch.float32)
        tracks = tracks.to(device, dtype=torch.float32)
        idx_skip = idx_skip.to(device)
        tn5_bias = tn5_bias.to(device, dtype=torch.float32)

        _, profile, count = model(inputs)

        #Convert profile and total count to track
        p = torch.nn.functional.softmax(profile, dim=1).permute(0,2,1)
        p = p * ((torch.exp(count)-1).unsqueeze(-1))

        pred_track.append(p.cpu())

with open('../results/predictions/test_pred.pkl', 'wb') as file:
    pickle.dump(pred_track, file)

with open('../results/predictions/test_target.pkl', 'wb') as file:
    pickle.dump(target_track, file)

for i, data in enumerate(train_dataloader):
    with torch.no_grad():    
        
        inputs, tracks, idx_skip, tn5_bias = data 
        target_track.append(target_track)
        inputs = inputs.to(device, dtype=torch.float32)
        tracks = tracks.to(device, dtype=torch.float32)
        idx_skip = idx_skip.to(device)
        tn5_bias = tn5_bias.to(device, dtype=torch.float32)

        _, profile, count = model(inputs)

        #Convert profile and total count to track
        p = torch.nn.functional.softmax(profile, dim=1).permute(0,2,1)
        p = p * ((torch.exp(count)-1).unsqueeze(-1))

        pred_track.append(p.cpu())

with open('../results/predictions/train_pred.pkl', 'wb') as file:
    pickle.dump(pred_track, file)

with open('../results/predictions/train_target.pkl', 'wb') as file:
    pickle.dump(target_track, file)
