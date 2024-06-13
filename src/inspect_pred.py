import torch
from models.models import CATAC
from data_processing.utils_data_preprocessing import one_hot_encode
import pandas as pd
import pickle

path_seq = "../results/peaks_seqtest.pkl"
path_ATAC = "../results/ATAC_peakstest.pkl"

path_model = '../results/insertion/KLD_model_1e-3.pkl'

#Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CATAC(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, profile_kernel_size=75, out_pred_len=1024, 
                      nb_pred=13, nb_cell_type_CN = 0)
        
model.load_state_dict(torch.load(path_model, map_location=device))

#Load sequence and ATAC
ATAC = pd.read_pickle(path_ATAC)
seq_id = ATAC.sample(5).index

seq = pd.read_pickle(path_seq).sequence

ATAC = ATAC.loc[seq_id, 0]
seq = seq.loc[seq_id]


#On-hot encode the sequences
seq = seq.apply(lambda x: one_hot_encode(x))
seq = torch.tensor(seq).permute(0,2,1)
with torch.no_grad():
    x, profile, count = model(seq)
    profile = torch.nn.functional.softmax(profile[0])
    profile = profile * torch.exp(count[0])

ATAC = pd.DataFrame({"true": ATAC, "pred": profile})
with open('../results/pred.pkl', 'wb') as file:
    pickle.dump(ATAC, file)