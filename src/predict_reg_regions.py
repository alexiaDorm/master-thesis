import numpy as np
import pandas as pd
import torch
import math
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from interpretation.interpret import compute_tn5_bias
import subprocess

from models.models import CATAC_wo_bias, CATAC_w_bias, CATAC_w_bias_increase_filter
from data_processing.utils_data_preprocessing import one_hot_encode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load model
#---------------------------------
path_model = '../results/train_res/128_10_model.pkl'
all_c_type = ['Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast',
       'Neuronal', 'Somite']

first_kernel = 21
nb_conv = 10
size_final_conv = 4096 - (first_kernel - 1)
cropped = [2**l for l in range(0,nb_conv-1)] * (2*(3-1))

for c in cropped:
       size_final_conv -= c

model = CATAC_w_bias(nb_conv=nb_conv, nb_filters=128, first_kernel=first_kernel, 
                      rest_kernel=3, out_pred_len=1024, 
                      nb_pred=4, size_final_conv = size_final_conv)
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

path_model_bias = "../data/Tn5_NN_model.h5"
model_bias = load_model(path_model_bias)

#Intersection between variants in active enhancers regions and peak set of dataset
#---------------------------------

#Create peak set bed file
with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

peaks = peaks[peaks.chr.str.isnumeric()].iloc[:,:3]

peaks.to_csv('../results/peaks_set.bed', header=False, index=False, sep='\t')

#Intersect the variants in active enhancer with the peak set 
#Change here the file with variants
cmd_bed = "bedtools intersect -a ../results/peaks_set.bed -b ../data/vars_in_acitive_enchancers.bed -wa -wb > ../results/aevp_pverlap.bed"
subprocess.run(cmd_bed, shell=True)

active_enhancer = pd.read_csv("../results/aevp_pverlap.bed", header=None, sep='\t', names=["chr_peak", "start_peak", "end_peak", "chr_var", "pos_var", "end_var"])

#Add sequence of peak
with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)
active_enhancer.index = active_enhancer.chr_peak.astype(str) + ":" + active_enhancer.start_peak.astype(str) + "-" + active_enhancer.end_peak.astype(str)
active_enhancer["sequence"] = peaks.loc[active_enhancer.index].sequence
active_enhancer["middle"] = peaks.loc[active_enhancer.index].middle_peak

### Get reference sequenece and mutated sequences for overlapping variants
#---------------------------------

variants = pd.DataFrame(pd.read_csv("../data/Muscle_Skeletal.var.txt", sep='\t').variant_id)
variants = variants['variant_id'].str.split('_')

variants = pd.DataFrame(variants.tolist(), index= variants.index, columns=['chr_var','pos_var', 'ref', 'alt', '_']).iloc[:,:-1]
variants.chr_var = [idx[3:] for idx in variants.chr_var]
variants.index = variants.chr_var.astype(str) + ":" + variants.pos_var.astype(str)

#Join both dataset 
active_enhancer.index = active_enhancer.chr_var.astype(str) + ":" + active_enhancer.pos_var.astype(str)

variants = variants.loc[active_enhancer.index]
variants.chr_var = variants.chr_var.astype('int64'); variants.pos_var = variants.pos_var.astype('int64')

active_enhancer = pd.merge(variants, active_enhancer)
active_enhancer.index = active_enhancer.chr_var.astype(str) + ":" + active_enhancer.pos_var.astype(str) + "_" + active_enhancer.ref.astype(str) + "_" + active_enhancer.alt.astype(str)
active_enhancer = active_enhancer[~active_enhancer.index.duplicated(keep='first')]

#Create alt sequences
active_enhancer['dist_var_middle'] = active_enhancer.pos_var - active_enhancer.middle 
active_enhancer['var_pos_seq'] = 2048 + active_enhancer.dist_var_middle -1 

#check ref match to sequence
active_enhancer['nucleotide_var'] = [x[active_enhancer.var_pos_seq[i]] for i,x in enumerate(active_enhancer.sequence)]

#Create alt sequence
active_enhancer['sequence_alt'] = [x[:active_enhancer.var_pos_seq[i]] + active_enhancer.alt[i] + x[active_enhancer.var_pos_seq[i]+1:] for i,x in enumerate(active_enhancer.sequence)]

#if alt more than one nucleotide, crop sequence alt
active_enhancer['sequence_alt'] = [x[math.floor((len(x)-4096)/2):-math.ceil((len(x)-4096)/2)] if len(x) > 4096 else x for x in active_enhancer.sequence_alt]

#Store the ground truth for each peak investigated for Myogenic progenitors
active_enhancer.index = active_enhancer.chr_peak.astype(str) + ":" + active_enhancer.start_peak.astype(str) + "-" + active_enhancer.end_peak.astype(str)

with open("../results/ATAC_peaks_new.pkl", 'rb') as file:
    ATAC_track = pickle.load(file)

with open("../results/c_type_track.pkl", 'rb') as file:
    c_type = pickle.load(file)

seq_id = peaks.index.to_numpy()
seq_id = np.repeat(seq_id, 7)
seq_idx = np.where(np.logical_and(np.isin(seq_id, active_enhancer.index.to_numpy()), c_type == "Myogenic"))[0]
ATAC_track = ATAC_track[seq_idx]

with open('../results/target_active_enhancer.pkl', 'wb') as file:
    pickle.dump(ATAC_track, file)

""" #Predict for variant the ref and alt sequence
#---------------------------------
#Define sequences
ref_seq = active_enhancer.sequence
alt_seq = active_enhancer.sequence_alt

#Predict tn5 bias for each sequence
tn5_bias_ref = ref_seq.apply(lambda x: compute_tn5_bias(model_bias, x))
tn5_bias_alt = alt_seq.apply(lambda x: compute_tn5_bias(model_bias, x))

#On-hot encode the sequences
ref_seq_enc = ref_seq.apply(lambda x: one_hot_encode(x))
alt_seq_enc = alt_seq.apply(lambda x: one_hot_encode(x))

#Add cell type encoding
c_type = "Myogenic"
mapping = dict(zip(all_c_type, range(len(all_c_type))))    
c_type = mapping[c_type]
c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
c_type = c_type.tile((ref_seq_enc[0].shape[0],1))

ref_seq_enc = [np.concatenate((s,c_type), axis=1) for s in ref_seq_enc]
ref_seq_enc = torch.tensor(ref_seq_enc).permute(0,2,1)

alt_seq_enc = [np.concatenate((s,c_type), axis=1) for s in alt_seq_enc]
alt_seq_enc = torch.tensor(alt_seq_enc).permute(0,2,1)

#Predict
x_ref, profile_ref, count_ref = model(ref_seq_enc, torch.tensor(np.vstack(tn5_bias_ref)))
x_alt, profile_alt, count_alt = model(alt_seq_enc, torch.tensor(np.vstack(tn5_bias_alt)))

with open('../results/pred_profile_active_enhancer.pkl', 'wb') as file:
    pickle.dump(profile_ref, file)

with open('../results/pred_count_active_enhancer.pkl', 'wb') as file:
    pickle.dump(count_ref, file)

with open('../results/alt_pred_profile_active_enhancer.pkl', 'wb') as file:
    pickle.dump(profile_alt, file)

with open('../results/alt_pred_count_active_enhancer.pkl', 'wb') as file:
    pickle.dump(count_alt, file) """