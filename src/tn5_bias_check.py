import torch

from interpretation.synthetic_seq_analysis import generate_seq_tn5
from interpretation.interpret import compute_importance_score_c_type, compute_importance_score_bias, visualize_sequence_imp
from models.models import CATAC_wo_bias, CATAC_w_bias

all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
time_point = ["D8", "D12", "D20", "D22"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define tn5 motifs from https://github.com/kundajelab/chrombpnet/blob/master/chrombpnet/pipelines.py
bias_motifs = [["tn5_1","GCACAGTACAGAGCTG"],["tn5_2","GTGCACAGTTCTAGAGTGTGCAG"],["tn5_3","CCTCTACACTGTGCAGAA"],["tn5_4","GCACAGTTCTAGACTGTGCAG"],["tn5_5","CTGCACAGTGTAGAGTTGTGC"]]

#Generate random sequence with tn5 bias 
seq = []
for i in range(100):
    for j in range(len(bias_motifs)):
        seq.append(generate_seq_tn5(0.41, 4096, bias_motifs[j][1]))

#Load the model - WITH tn5 bias
path_model_bias = '../results/train_res/128_model.pkl'
model_bias = CATAC_w_bias(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model_bias.load_state_dict(torch.load(path_model_bias, map_location=torch.device('cpu')))

path_model_bias = "../data/Tn5_NN_model.h5"

#Load the model - WITHOUT tn5 bias
path_model_wobias = '../results/train_res/126_wobias_model.pkl'
model_wobias = CATAC_wo_bias(nb_conv=8, nb_filters=64, first_kernel=21, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4)
        
model_wobias.load_state_dict(torch.load(path_model_wobias, map_location=torch.device('cpu')))

#Compute contribution scores
