import numpy as np
import random
import pandas as pd
import torch
import scipy
import math
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from interpretation.interpret import compute_tn5_bias
import subprocess
import pyBigWig
import matplotlib.pyplot as plt
from scipy.spatial import distance

from models.models import CATAC_wo_bias, CATAC_w_bias, CATAC_w_bias_increase_filter
from data_processing.utils_data_preprocessing import one_hot_encode, get_continuous_wh_window

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

chr_test = ['6','13','22']  

#Load model
#---------------------------------
path_model = '../results/train_res/128_10_model.pkl'
all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']
TIME_POINT = ["D8", "D12", "D20", "D22-15"]

first_kernel = 21
nb_conv = 10
size_final_conv = 4096 - (first_kernel - 1)
cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))

for c in cropped:
       size_final_conv -= c

model = CATAC_w_bias(nb_conv=nb_conv, nb_filters=128, first_kernel=first_kernel, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4, size_final_conv = size_final_conv)
model.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

path_model_bias = "../data/Tn5_NN_model.h5"
model_bias = load_model(path_model_bias)

""" #---------------------------------

#Predict and evaluate performance in peak test set

#---------------------------------
jsd_list, corr_list = [], []
for c in all_c_type:

    #Get and encode the test sequences
    #---------------------------------
    #Overlap peak and test set
    with open('../results/peaks_seq.pkl', 'rb') as file:
        peaks = pickle.load(file)

    peaks = peaks[peaks.chr.isin(chr_test)]
    seq = peaks.sequence

    #Predict tn5 bias for each sequence
    tn5_bias = seq.apply(lambda x: compute_tn5_bias(model_bias, x))

    #On-hot encode the sequences
    seq_enc = seq.apply(lambda x: one_hot_encode(x))

    #Add cell type encoding
    c_type = c; mapping = dict(zip(all_c_type, range(len(all_c_type)))); c_type = mapping[c_type]
    c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
    c_type = c_type.tile((seq_enc[0].shape[0],1))

    seq_enc = [np.concatenate((s,c_type), axis=1) for s in seq_enc]
    seq_enc = torch.tensor(seq_enc).permute(0,2,1)

    #Predict
    #---------------------------------
    with torch.no_grad():
        x, profile, count = model(seq_enc, torch.tensor(np.vstack(tn5_bias)))

    #Get the ground truth for each peak investigated for cell type
    #---------------------------------
    all_ATAC = []
    for t in TIME_POINT: 
        
        if t == "D8" and (c == "Immature" or c == "Myoblast" or c == "Neuroblast"):
            ATAC_tracks = torch.zeros_like((profile[:,:,0]))
        
        else:
            bw_files = '../results/bam_cell_type/' + t +'/' + c + '_unstranded.bw'

            bw = pyBigWig.open(bw_files)
            ATAC_tracks = peaks.apply(lambda x: get_continuous_wh_window(bw, x, 0, seq_len=1024), axis=1)
            ATAC_tracks = np.stack(ATAC_tracks)
            
        all_ATAC.append(ATAC_tracks)

    all_ATAC = torch.from_numpy(np.stack(all_ATAC, axis=2))

    #Compute performance metric (correlation + jsd)
    #---------------------------------

    corr_c_type=[]
    for i in range(4):
        corr_c_type.append(scipy.stats.spearmanr(count[:,i].detach().numpy(), 
                                            np.log(all_ATAC.sum(dim=1).numpy() +1)[:,i]).statistic)
    
    jsd_c_type = []
    tot_ATAC = all_ATAC.sum(dim=1)
    for i in range (count.size(0)):
        jsd_time = []
        for t in range(4):
            prob_obs = all_ATAC[i,:,t]/tot_ATAC[i,t]
            prob_obs[prob_obs != prob_obs] = 0 #set division by zero to 0 

            prob_pred = torch.nn.functional.softmax(profile[i,:,t])

            jsd_time.append(distance.jensenshannon(prob_pred, prob_obs))
        
        jsd_c_type.append(jsd_c_type)
    
    jsd_c_type = np.array(jsd_c_type).nanmean(axis=0)

    corr_list.append(corr_c_type); jsd_list.append(jsd_c_type)

print(corr_list, jsd_list)
with open('../results/predictions/peak_corr.pkl', 'wb') as file:
        pickle.dump(corr_list, file)

with open('../results/predictions/peak_jsd.pkl', 'wb') as file:
        pickle.dump(jsd_list, file)

#---------------------------------

#Predict and evaluate performance in background test set

#---------------------------------
jsd_list, corr_list = [], []
for c in all_c_type:

    #Get and encode the test sequences
    #---------------------------------
    #Overlap background and test chromosomes
    with open('../results/background_GC_matched.pkl', 'rb') as file:
        peaks = pickle.load(file)

    peaks = peaks[peaks.chr.isin(chr_test)]
    seq = peaks.sequence

    #Predict tn5 bias for each sequence
    tn5_bias = seq.apply(lambda x: compute_tn5_bias(model_bias, x))

    #On-hot encode the sequences
    seq_enc = seq.apply(lambda x: one_hot_encode(x))

    #Add cell type encoding
    c_type = c; mapping = dict(zip(all_c_type, range(len(all_c_type)))); c_type = mapping[c_type]
    c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
    c_type = c_type.tile((seq_enc[0].shape[0],1))

    seq_enc = [np.concatenate((s,c_type), axis=1) for s in seq_enc]
    seq_enc = torch.tensor(seq_enc).permute(0,2,1)

    #Predict
    #---------------------------------
    with torch.no_grad():
        x, profile, count = model(seq_enc, torch.tensor(np.vstack(tn5_bias)))

    #Get the ground truth for each peak investigated for cell type
    #---------------------------------
    all_ATAC = []
    for t in TIME_POINT: 
        if t == "D8" and (c == "Immature" or c == "Myoblast" or c == "Neuroblast"):
            ATAC_tracks = torch.zeros_like((profile[:,:,0]))
        
        else:
            bw_files = '../results/bam_cell_type/' + t +'/' + c + '_unstranded.bw'

            bw = pyBigWig.open(bw_files)
            ATAC_tracks = peaks.apply(lambda x: get_continuous_wh_window(bw, x, 0, seq_len=1024), axis=1)
            ATAC_tracks = np.stack(ATAC_tracks)

    all_ATAC = torch.from_numpy(np.stack(all_ATAC, axis=2))

    #Compute performance metric (correlation + jsd)
    #---------------------------------

    corr_c_type=[]
    for i in range(4):
        corr_c_type.append(scipy.stats.spearmanr(count[:,i].detach().numpy(), 
                                            np.log(all_ATAC.sum(dim=1).numpy() +1)[:,i]).statistic)
    
    jsd_c_type = []
    tot_ATAC = all_ATAC.sum(dim=1)
    for i in range (count.size(0)):
        jsd_time = []
        for t in range(4):
            prob_obs = all_ATAC[i,:,t]/tot_ATAC[i,t]
            prob_obs[prob_obs != prob_obs] = 0 #set division by zero to 0 

            prob_pred = torch.nn.functional.softmax(profile[i,:,t])

            jsd_time.append(distance.jensenshannon(prob_pred, prob_obs))
        
        jsd_c_type.append(jsd_c_type)
    
    jsd_c_type = np.array(jsd_c_type).nanmean(axis=0)

    corr_list.append(corr_c_type); jsd_list.append(jsd_c_type)

print(corr_list, jsd_list)
with open('../results/predictions/back_corr.pkl', 'wb') as file:
        pickle.dump(corr_list, file)

with open('../results/predictions/back_jsd.pkl', 'wb') as file:
        pickle.dump(jsd_list, file) """

#---------------------------------

#Predict and evaluate performance in accessible regulatory regions

#---------------------------------

#Create dataframe of regulatory regions
#---------------------------------
#Overlap between peak (accessible regions) and regulatory regions
with open('../results/peaks_seq.pkl', 'rb') as file:
        peaks = pickle.load(file)

peaks = peaks[peaks.chr.isin(chr_test)]
peaks.to_csv('../results/peaks_set.bed', header=False, index=False, sep='\t')

#Merge regulatory regions bed files
active_enhancer = pd.read_csv("../data/active_enhancers_all_days.bed", header=None, index=None, sep='\t')
rep_enhancer = pd.read_csv("../data/repressed_enhancers_all_days.bed", header=None, index=None, sep='\t')
promoter = pd.read_csv("../data/active_promoters_all_days.bed", header=None, index=None, sep='\t')

reg_regions = pd.concat([active_enhancer, rep_enhancer, promoter])
reg_regions.to_csv('../results/reg_regions.bed', header=False, index=False, sep='\t')

cmd_bed = "bedtools intersect -a ../results/peaks_set.bed -b ../results/reg_regions.bed -wa -wb > ../results/accessible_reg_regions.bed"
subprocess.run(cmd_bed, shell=True)

reg_regions = pd.read_csv("../results/accessible_reg_regions.bed", header=None, index=None, sep='\t', 
                          columns=['chr','start','end'])
reg_regions.index = reg_regions.chr.astype(str) + ":" + reg_regions.start.astype(str) + '-' + reg_regions.end.astype(str)

reg_regions = peaks.loc[reg_regions.index]
with open('../results/reg_regions.pkl', 'wb') as file:
    pickle.dump(reg_regions, file)

jsd_list, corr_list = [], []
for c in all_c_type:

    #Get and encode the test sequences
    #---------------------------------
    #Overlap background and test chromosomes
    with open('../results/reg_regions.pkl', 'rb') as file:
        peaks = pickle.load(file).sample(5000)
    seq = peaks.sequence

    #Predict tn5 bias for each sequence
    tn5_bias = seq.apply(lambda x: compute_tn5_bias(model_bias, x))

    #On-hot encode the sequences
    seq_enc = seq.apply(lambda x: one_hot_encode(x))

    #Add cell type encoding
    c_type = c; mapping = dict(zip(all_c_type, range(len(all_c_type)))); c_type = mapping[c_type]
    c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
    c_type = c_type.tile((seq_enc[0].shape[0],1))

    seq_enc = [np.concatenate((s,c_type), axis=1) for s in seq_enc]
    seq_enc = torch.tensor(seq_enc).permute(0,2,1)

    #Predict
    #---------------------------------
    with torch.no_grad():
        x, profile, count = model(seq_enc, torch.tensor(np.vstack(tn5_bias)))

    #Get the ground truth for each peak investigated for cell type
    #---------------------------------
    all_ATAC = []
    for t in TIME_POINT: 
        if t == "D8" and (c == "Immature" or c == "Myoblast" or c == "Neuroblast"):
            ATAC_tracks = torch.zeros_like((profile[:,:,0]))
        
        else:
            bw_files = '../results/bam_cell_type/' + t +'/' + c + '_unstranded.bw'

            bw = pyBigWig.open(bw_files)
            ATAC_tracks = peaks.apply(lambda x: get_continuous_wh_window(bw, x, 0, seq_len=1024), axis=1)
            ATAC_tracks = np.stack(ATAC_tracks)
            
        all_ATAC.append(ATAC_tracks)

    all_ATAC = torch.from_numpy(np.stack(all_ATAC, axis=2))

    #Compute performance metric (correlation + jsd)
    #---------------------------------

    corr_c_type=[]
    for i in range(4):
        corr_c_type.append(scipy.stats.spearmanr(count[:,i].detach().numpy(), 
                                            np.log(all_ATAC.sum(dim=1).numpy() +1)[:,i]).statistic)
    
    jsd_c_type = []
    tot_ATAC = all_ATAC.sum(dim=1)
    for i in range (count.size(0)):
        jsd_time = []
        for t in range(4):
            prob_obs = all_ATAC[i,:,t]/tot_ATAC[i,t]
            prob_obs[prob_obs != prob_obs] = 0 #set division by zero to 0 

            prob_pred = torch.nn.functional.softmax(profile[i,:,t])

            jsd_time.append(distance.jensenshannon(prob_pred, prob_obs))
        
        jsd_c_type.append(jsd_c_type)
    
    jsd_c_type = np.array(jsd_c_type).nanmean(axis=0)

    corr_list.append(corr_c_type); jsd_list.append(jsd_c_type)

print(corr_list, jsd_list)
with open('../results/predictions/reg_corr.pkl', 'wb') as file:
        pickle.dump(corr_list, file)

with open('../results/predictions/reg_jsd.pkl', 'wb') as file:
        pickle.dump(jsd_list, file)