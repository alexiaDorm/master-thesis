import torch
from models.models import CATAC2
from data_processing.utils_data_preprocessing import one_hot_encode
import pandas as pd
import pickle
import re
import numpy as np

path_seq = "../results/peaks_seqtest.pkl"
path_ATAC = "../results/ATAC_peakstest.pkl"

path_model = '../results/model_1e-3.pkl'

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
ATAC = pd.read_pickle(path_ATAC)
seq_id = ATAC.sample(50).index

seq = pd.read_pickle(path_seq).sequence

ATAC = ATAC.loc[seq_id, :]
seq = seq.loc[seq_id]

#On-hot encode the sequences
seq = seq.apply(lambda x: one_hot_encode(x))
seq = torch.tensor(seq).permute(0,2,1)

#Add cell type encoding
c_type = [re.findall('[A-Z][^A-Z]*', x)[1] for x in ATAC.pseudo_bulk]

mapping = dict(zip(all_c_type, range(len(all_c_type))))    
c_type = mapping[c_type]
c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
print(c_type.shape)

#Repeat and reshape
c_type = c_type.tile((input.shape[-1],1)).permute(1,0)[:,:]
seq = torch.cat((seq.squeeze(), c_type), dim=0)
print(seq.shape)

with torch.no_grad():
    x, profile, count = model(seq)
    profile = torch.nn.functional.softmax(profile[0])
    profile = profile * torch.exp(count[0])

with open('../results/pred.pkl', 'wb') as file:
    pickle.dump(profile, file)

with open('../results/true.pkl', 'wb') as file:
    pickle.dump(ATAC, file)
