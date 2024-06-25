import torch
from models.models import CATAC2
from data_processing.utils_data_preprocessing import one_hot_encode
import pandas as pd
import pickle
import re
import numpy as np

path_seq = "../results/peaks_seqtest.pkl"
path_ATAC = "../results/ATAC_peakstest.pkl"

path_model = '../new/model_1e-3.pkl'

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_order = ['D8', 'D12', 'D20', 'D22-15']

#Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CATAC2(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=device))

#Load sequence and ATAC
ATAC = pd.read_pickle(path_ATAC).sample(100)
seq_id = ATAC.index

seq = pd.read_pickle(path_seq).sequence

#ATAC = ATAC.loc[seq_id, :]
seq = seq.loc[seq_id]

#On-hot encode the sequences
seq = seq.apply(lambda x: one_hot_encode(x))
seq = torch.tensor(seq).permute(0,2,1)

#Add cell type encoding
c_type = [re.findall('[A-Z][^A-Z]*', x)[1] for x in ATAC.pseudo_bulk]

c_types = []
mapping = dict(zip(all_c_type, range(len(all_c_type))))    
for c in c_type:
    c = mapping[c]
    c = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c])
    c = c.tile((seq.shape[2],1))
    c_types.append(c)

c_type = torch.stack(c_types).permute(0,2,1)

#Repeat and reshape
seq = torch.cat((seq.squeeze(), c_type), dim=1)

with torch.no_grad():
    x, profile, count = model(seq)

    time = [re.findall('[A-Z][^A-Z]*', x)[0] for x in ATAC.pseudo_bulk]
    time_point = [time_order.index(t) for t in time]

    profile_list = []
    for i, t in enumerate(time_point):
        p = torch.nn.functional.softmax(profile[t][i,:][None,:])
        p = p * torch.exp(count[t][i])

        profile_list.append(p)

    #rofile_list = torch.stack(profile_list)

with open('../results/pred.pkl', 'wb') as file:
    pickle.dump(profile_list, file)

with open('../results/true.pkl', 'wb') as file:
    pickle.dump(ATAC, file)
