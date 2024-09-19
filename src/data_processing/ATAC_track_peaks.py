#Paths assume run from src folder

#Fetch continuous ATAC tracks for each pseudo-bulk and peak regions

#Note:
#All cell type need to be defined here in all_cell_types list
#If due to a insufficient number of cells in a particular pseudo-bulk, the bigwig file of this pseudobulk should be deleated in ../results/bam_cell_type/ or it will be added to the training
#--------------------------------------------

import pickle
import numpy as np
import torch
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_wh_window

TIME_POINT = ["D8", "D12", "D20", "D22-15"]

#Need to define here the name of all cell type 
all_cell_types = ['Neuronal', 'Somite', 'Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast']

with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

peaks['middle'] = np.round((peaks.end - peaks.start)/2 + peaks.start).astype('uint32')
nb_peaks = peaks.shape[0]


all_ATAC, all_is_defined, idx_seq, chr_seq, c_type = [], [], [], [], []
for c in all_cell_types:

    bw_files = ['../results/bam_cell_type/' + t +'/' + c  + '_unstranded.bw' for t in TIME_POINT]
    ATAC_tracks, is_defined = [], []
    for f in bw_files:
        if os.path.isfile(f):
            
            #Get insertion count
            bw = pyBigWig.open(f)
            ATAC = peaks.apply(lambda x: get_continuous_wh_window(bw, x, seq_len=1024), axis=1)
            ATAC = np.stack(ATAC)
            ATAC_tracks.append(ATAC)

            #The tracks are defined for cell type + time point
            is_defined.append([True]*nb_peaks)

        else: 

            #If cell type not defined for time point, fill with zero tracks
            ATAC =  np.zeros((nb_peaks,1024))
            ATAC_tracks.append(ATAC)

            #The tracks are NOT defined for cell type + time point
            is_defined.append([False]*nb_peaks)
    
    #Stack the ATAC tracks and is_defined -> shape:(#peaks, 1024, #time)
    all_ATAC.append(np.transpose(np.stack(ATAC_tracks), (1, 2, 0)))
    all_is_defined.append(np.transpose(np.stack(is_defined), (1, 0)))

    #Keep sequence idx and cell type
    idx_seq.append(np.arange(0,nb_peaks))
    chr_seq.append(peaks.chr.values)
    c_type.append([c]*nb_peaks)
    
all_ATAC = torch.from_numpy(np.concatenate(all_ATAC, axis=0))
all_is_defined = torch.from_numpy(np.concatenate(all_is_defined, axis=0))
idx_seq = torch.from_numpy(np.concatenate(idx_seq, axis=0))
chr_seq = np.concatenate(chr_seq, axis=0)
c_type = np.concatenate(c_type, axis=0)

with open('../results/ATAC_peaks_new.pkl', 'wb') as file:
    pickle.dump(all_ATAC, file)

with open('../results/is_defined.pkl', 'wb') as file:
    pickle.dump(all_is_defined, file)

with open('../results/idx_seq.pkl', 'wb') as file:
    pickle.dump(idx_seq, file)

with open('../results/chr_seq.pkl', 'wb') as file:
    pickle.dump(chr_seq, file)

with open('../results/c_type_track.pkl', 'wb') as file:
    pickle.dump(c_type, file)

print('ok :))')