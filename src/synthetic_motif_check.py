import numpy as np
import pandas as pd
import torch
import glob
import pickle

from interpretation.synthetic_seq_analysis import generate_motif, generate_seq
from interpretation.interpret import compute_importance_score_c_type, compute_importance_score_bias, visualize_sequence_imp
from models.models import CATAC2, CATAC_w_bias

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create synthetic sequennces
#--------------------------------------------
motif_files, n = glob.glob("../data/TF_motif/*.xlsx"), 5
rng = np.random.default_rng(42)

syn_seq, idx_motif, TF_name = [], [], []
for f in motif_files:
    for _ in range(n):
        motif = generate_motif(f, rng)
        seq, idx = generate_seq(0.41, motif, 4096 - len(motif), 1024)

        syn_seq.append(seq)
        idx_motif.append(idx)
        TF_name.append(f[17:-5])

df = pd.DataFrame({"idx":idx_motif, "TF_name": TF_name})

with open('../results/synthetic_results/synthetic_sequences.pkl', 'wb') as file:
    pickle.dump(syn_seq, file)

with open('../results/synthetic_results/synthetic_sequences_metadata.pkl', 'wb') as file:
    pickle.dump(df, file)

#Compute importance score using shap DeepExplainer
#--------------------------------------------
path_model = '../results/train_res/wbias_model.pkl'
path_seq = '../data/more_synthetic_sequences.pkl'

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_point = ["D8", "D12", "D20", "D22"]

#Load the model
model = CATAC2(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))


with open('../results/synthetic_results/synthetic_sequences_metadata.pkl', 'rb') as file:
    metadata = pickle.open(file)

for i, t in enumerate(time_point):

    if t == "D8":
        defined_c_type = ['Mesenchymal', 'Myogenic', 'Neuronal', 'Somite']
    else:
        defined_c_type = all_c_type

    unique_TF = np.unique(metadata.TF_name)
    
    #For c_type compute score + save full view + zoom attribution map in tmp directory with TF name + c_type in name file 
    for c in defined_c_type:

        #Compute attribution scores
        _, _, proj_score = compute_importance_score_c_type(model, path_seq, device, c, all_c_type, i)

        for TF in unique_TF:
            idx_seq = np.where(metadata.TF_name == TF)
            
            for j,idx in enumerate(idx_seq):
                #Plot sequence overall
                
                
                #Plot zoom on motif
                pos_motif = metadata.idx[idx]
                visualize_sequence_imp(proj_score[[0],:4,:] ,1700, 1800)


            





