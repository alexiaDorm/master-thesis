import numpy as np
import pandas as pd
import torch
import glob
import pickle
import matplotlib.pyplot as plt
from fpdf import FPDF

from interpretation.synthetic_seq_analysis import generate_motif, generate_seq
from interpretation.interpret import compute_importance_score_c_type, compute_importance_score_bias, visualize_sequence_imp
from models.models import CATAC2, CATAC_w_bias

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create subset of sequence to analyzed
#--------------------------------------------
with open('../results/"peaks_seq.pkl', 'rb') as file:
    seq = pickle.load(file)

seq.head()

""" #Compute importance score using shap DeepExplainer
#--------------------------------------------
path_model = '../results/train_res/wbias_model_1e-3.pkl'
path_seq = '../results/synthetic_results/synthetic_sequences.pkl'

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_point = ["D8", "D12", "D20", "D22"]

#Load the model
model = CATAC_w_bias(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

path_model_bias = "../data/Tn5_NN_model.h5"

with open('../results/synthetic_results/synthetic_sequences_metadata.pkl', 'rb') as file:
    metadata = pickle.load(file)

for i, t in enumerate(time_point):

    if t == "D8":
        defined_c_type = ['Mesenchymal', 'Myogenic', 'Neuronal', 'Somite']
    else:
        defined_c_type = all_c_type

    unique_TF = np.unique(metadata.TF_name)

    pdf = FPDF()
    pdf.add_page()
    
    #For c_type compute score + save full view + zoom attribution map in tmp directory with TF name + c_type in name file 
    for c in defined_c_type:

        #Compute attribution scores
        _, _, proj_score = compute_importance_score_bias(model, path_model_bias, path_seq, device, c, all_c_type, i)

        for TF_name in unique_TF:
            pdf.cell(75, 10, TF_name, 0, 2, 'C')
            idx_seq = np.where(metadata.TF_name == TF_name)
            
            for j,idx in enumerate(idx_seq):
                
                #Plot sequence overall
                save_fig = "../results/synthetic_results/tmp.png"
                visualize_sequence_imp(proj_score[[idx],:4,:] ,0, 4096)
                plt.savefig(save_fig); plt.show()
                pdf.image(save_fig, x = None, y = None, w = 0, h = 0, type = '', link = '')

                #Plot zoom on motif
                pos_motif = metadata.idx[idx]
                visualize_sequence_imp(proj_score[[idx],:4,:] , pos_motif, pos_motif+12)
                plt.savefig(save_fig); plt.show()
                pdf.image(save_fig, x = None, y = None, w = 0, h = 0, type = '', link = '')
            
            break

        break
    
    pdf.output('"../results/synthetic_results/test.pdf', 'F')
    break


            



            
 """




