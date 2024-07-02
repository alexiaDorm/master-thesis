import torch
from torch.utils.data import DataLoader

from models.models import CATAC2
from data_processing.utils_data_preprocessing import one_hot_encode

import pandas as pd
import pickle
import numpy as np

path_seq = "../results/peaks_seq.pkl"

path_model = '../results/train_res/w_model_1e-3.pkl'

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_order = ['D8', 'D12', 'D20', 'D22-15']

#Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CATAC2(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=device))

with open('../results/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)
    
test_dataloader = DataLoader(test_dataset, 108,
                        shuffle=True, num_workers=4)

profile_list, tracks_list = [], []
for i, data in enumerate(test_dataloader):
    with torch.no_grad():    
        inputs, tracks, idx_skip = data 
        inputs = inputs.float().to(device)
        tracks = tracks.float().to(device)

        _, profile, count = model(inputs)

    for j, t in enumerate(time_order):
        p = torch.nn.functional.softmax(profile[j], dim=1)
        p = p * torch.exp(count[j])

        profile_list.append(p)
        tracks_list.append(tracks)

    if i == 5:
        break

with open('../results/pred.pkl', 'wb') as file:
    pickle.dump(profile_list, file)

with open('../results/true.pkl', 'wb') as file:
    pickle.dump(tracks_list, file)
