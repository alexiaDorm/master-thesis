import torch
import numpy as np
import pandas as pd
import subprocess
import glob

from interpretation.synthetic_seq_analysis import generate_seq_tn5, generate_motif
from interpretation.interpret import compute_importance_score_wobias, compute_importance_score_bias, visualize_sequence_imp
from models.models import CATAC_wo_bias, CATAC_w_bias

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_point = ["D8", "D12", "D20", "D22"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(42)

#Generate random sequence with TF motifs  
#--------------------------------------------
motif_files, n = glob.glob("../data/TF_motif/*.xlsx"), 3

syn_seq = []
for f in motif_files:
    for _ in range(200):
        motif = generate_motif(f, rng)
        seq = generate_seq_tn5(0.41, 4096, motif)

        syn_seq.append(seq)

syn_seq = pd.Series(syn_seq)

#Load models
#--------------------------------------------
#Load the base model
path_model_base = '../results/train_res/128_MNLL_model.pkl'
model_base = CATAC_w_bias(nb_conv=8, nb_filters=128, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model_base.load_state_dict(torch.load(path_model_base, map_location=torch.device('cpu')))

#Load the model kernel size =4

nb_conv = 8
first_kernel = 4

size_final_conv = 4096 - (first_kernel - 1)
cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))
for c in cropped:
    size_final_conv -= c

path_model_k4 = '../results/train_res/128_k4_model.pkl'
model_k4 = CATAC_w_bias(nb_conv=8, nb_filters=128, first_kernel=4, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4, size_final_conv= size_final_conv)
        
model_k4.load_state_dict(torch.load(path_model_k4, map_location=torch.device('cpu')))

#Load the model kernel size =9

nb_conv = 8
first_kernel = 9

size_final_conv = 4096 - (first_kernel - 1)
cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))
for c in cropped:
    size_final_conv -= c

path_model_k9 = '../results/train_res/128_9_model.pkl'
model_k9 = CATAC_w_bias(nb_conv=8, nb_filters=128, first_kernel=9, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4, size_final_conv= size_final_conv)
        
model_k9.load_state_dict(torch.load(path_model_k9, map_location=torch.device('cpu')))


path_model_bias = "../data/Tn5_NN_model.h5"

#Compute attribution scores
#--------------------------------------------

#Base
seq_base, scores_base, proj_score_base = compute_importance_score_bias(model_base, path_model_bias, syn_seq, device, "Myogenic", all_c_type, 2)
np.savez('../results/encod_seq_base.npz', seq_base[:,:4,:])
np.savez('../results/seq_scores_base.npz', scores_base[:,:4,:], proj_score_base[:,:4,:])

#128_k4
seq_k4, scores_k4, proj_score_k4 = compute_importance_score_bias(model_k4, path_model_bias, syn_seq, device, "Myogenic", all_c_type, 2)
np.savez('../results/encod_seq_k4.npz', seq_k4[:,:4,:])
np.savez('../results/seq_scores_k4.npz', scores_k4[:,:4,:], proj_score_k4[:,:4,:])

#128_k9
seq_k9, scores_k9, proj_score_k9 = compute_importance_score_bias(model_k9, path_model_bias, syn_seq, device, "Myogenic", all_c_type, 2)
np.savez('../results/encod_seq_k9.npz', seq_k9[:,:4,:])
np.savez('../results/seq_scores_k9.npz', scores_k9[:,:4,:], proj_score_k9[:,:4,:])

#Run modisco + TOMTOM
#--------------------------------------------

#Base
cmd_modisco = "modisco motifs -s  ../results/encod_seq_base.npz -a  ../results/seq_scores_base.npz -n 2000 -o modisco_results_base.h5"
subprocess.run(cmd_modisco, shell=True)

cmd_tomtom = "modisco report -i modisco_results_base.h5 -o TOMTOM_base -s ./ -m ../data/JASPAR_motif.txt"
subprocess.run(cmd_tomtom, shell=True)

#128_k4
cmd_modisco = "modisco motifs -s  ../results/encod_seq_k4.npz -a  ../results/seq_scores_k4.npz -n 2000 -o modisco_results_k4.h5"
subprocess.run(cmd_modisco, shell=True)

cmd_tomtom = "modisco report -i modisco_results_k4.h5 -o TOMTOM_k4 -s ./  -m ../data/JASPAR_motif.txt"
subprocess.run(cmd_tomtom, shell=True)

#128_k9
cmd_modisco = "modisco motifs -s  ../results/encod_seq_k9.npz -a  ../results/seq_scores_k9.npz -n 2000 -o modisco_results_k9.h5"
subprocess.run(cmd_modisco, shell=True)

cmd_tomtom = "modisco report -i modisco_results_k9.h5 -o TOMTOM_k9 -s ./  -m ../data/JASPAR_motif.txt"
subprocess.run(cmd_tomtom, shell=True)