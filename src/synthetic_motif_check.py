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

df = pd.DataFrame({"seq":syn_seq, "idx":idx_motif, "TF_name": TF_name})

with open('../results/synthetic_results/synthetic_sequences.pkl', 'wb') as file:
    pickle.dump(df, file)

#Compute importance score using shap DeepExplainer
#--------------------------------------------
path_model = '../results/train_res/w2_model_1e-3.pkl'
path_seq = '../data/more_synthetic_sequences.pkl'

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']

#Load the model
model = CATAC2(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

#Compute attribution scores
seq, shap_score, proj_score = compute_importance_score_c_type(model, path_seq, device, "Somite", all_c_type)