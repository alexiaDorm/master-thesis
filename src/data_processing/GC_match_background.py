#Paths assume run from src folder

#GC match genomic regions to peaks
#--------------------------------------------

import os
import pickle 
import anndata
import numpy as np
import pandas as pd
import pyfaidx
import matplotlib.pyplot as plt

from utils_data_preprocessing import compute_GC_content

len_seq = 4096
path_genome = '../data/hg38.fa' 

#Get matched GC content background sequence
#--------------------------------------------
with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

with open('../results/background_GC.pkl', 'rb') as file:
    locations = pickle.load(file)

#Shuffle genomic sequence
locations = locations.sample(frac=1)

#Compute GC content
peaks['GC_cont'] = compute_GC_content(peaks.sequence)

#Round GC_content to tolerance for two sequences to be matched
peaks.GC_cont = round(peaks.GC_cont, 2)
locations.GC_cont = round(locations.GC_cont, 2).astype('str')

locations = locations.set_index('GC_cont')

#Iterate over peak and match each peak to a background region in term of GC content
matched_back = locations.iloc[:0,:].copy()
for i,p in peaks.iterrows():
    
    if str(p.GC_cont) in locations.index:
        loc_idx = np.where(locations.index == str(p.GC_cont))[0][0]
        matched_back.loc[i] = locations.iloc[loc_idx]

        #Without replacment, remove the drawn background region
        matched_mask = [True]*locations.shape[0]; matched_mask[loc_idx] = False
        locations = locations[matched_mask]

with open('../results/background_GC_matched.pkl', 'wb') as file:
    pickle.dump(matched_back, file)