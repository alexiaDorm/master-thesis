import os
import pickle 
import numpy as np
import pandas as pd
import pyfaidx

from utils_data_preprocessing import fetch_sequence, compute_GC_content

len_seq = 4096
path_genome = '../data/hg38.fa' 

peaks = pd.read_csv('../results/common_peaks.bed', sep='\t', header=None, 
                    names=['chr','start','end'], dtype={"chr": 'category'})
peaks['peakID'] = peaks.chr.astype(str) + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
peaks = peaks.set_index('peakID')

#Fetch sequence and remove sequences with N nucleotides
peaks['sequence'] = fetch_sequence(peaks, path_genome=path_genome, len_seq=len_seq)
peaks = peaks[np.logical_not(peaks.sequence.str.contains("N"))]

#Compute GC content
peaks['GC_cont'] = compute_GC_content(peaks.sequence)

with open('../results/peaks_seq.pkl', 'wb') as file:
    pickle.dump(peaks, file)
