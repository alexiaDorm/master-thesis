import os
import pickle 
import numpy as np
import pandas as pd
import pyfaidx

from utils_data_preprocessing import fetch_sequence, compute_GC_content

len_seq = 2114
path_genome = '../data/hg38.fa' 

peaks = pd.read_csv('../results/common_peaks.bed', sep='\t', header=None, 
                    names=['chr','start','end'], dtype={"chr": 'category'})
peaks['peakID'] = peaks.chr.astype(str) + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
peaks = peaks.set_index('peakID')

#Fetch sequence and remove sequences with N nucleotides
peaks['sequence'] = fetch_sequence(peaks, path_genome=path_genome, len_seq=len_seq)
peaks = peaks[np.logical_not(peaks.sequence.str.contains("N"))]

with open('../results/peaks_seq.pkl', 'wb') as file:
    pickle.dump(peaks, file)

#Get matched GC content background sequence
#--------------------------------------------
#Compute GC content
peaks['GC_cont'] = compute_GC_content(peaks.sequence)

#Get size of each chromosome, to compute probability of selecting given chromosome, so that truly random
size_chrom = pd.read_csv('../data/size_genome.txt', sep='\t', header=None, index_col=0)
size_chrom = size_chrom.loc[['chr' + c for c in np.unique(peaks.chr)]]

#Generate random start location on chromosome
p_chrom = (size_chrom/np.sum(size_chrom)).iloc[:,0].tolist()
rand_chr = np.random.choice(size_chrom.index, p=p_chrom, size=peaks.shape[0])
rand_start = [np.random.randint(low=1, high=size_chrom.loc[x]-len_seq) for x in rand_chr]
rand_start = [int(x) for x in rand_start]

random_seq = pd.DataFrame({"chr":rand_chr, "start":rand_start})

#Get sequence on genome of random locations
genome = pyfaidx.Fasta(path_genome)
sequences =  random_seq.apply(lambda x: (genome[x['chr']][x['start']:(x['start']+len_seq)]).seq, axis=1)
sequences = sequences.str.upper()
random_seq['sequence'] = sequences

#Remove sequence with N 
random_seq = random_seq[np.logical_not(random_seq.sequence.str.contains("N"))]

#Check if inside peak
def check_whithin_peaks(location, peaks_df):
    within_regions = any(
        (peaks_df['chr'] == location.chr[3:]) & 
        (peaks_df['start'] <= location.start) & 
        (peaks_df['end'] >= location.start))

    return within_regions

random_seq = random_seq[~random_seq.apply(lambda s: check_whithin_peaks(s, peaks), axis=1)]

#Compute GC content
random_seq['GC_cont'] = compute_GC_content(random_seq.sequence)

#Match each peak sequence to its closest GC content (unique match)
peaks = peaks.sort_values('GC_cont')
random_seq = random_seq.sort_values('GC_cont')

merged_df = pd.merge_asof(random_seq, peaks.reset_index(), on='GC_cont', direction='nearest', suffixes=('','_peak'), tolerance=5e-2)
merged_df = peaks.merge(merged_df[['peakID','chr', 'start','sequence']], how='left', left_index=True, right_on='peakID', suffixes=('_x','')).set_index('peakID')
merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

merged_df = merged_df.dropna()
merged_df = merged_df.reset_index()[['chr', 'start', 'sequence']]

with open('../results/match_GC.pkl', 'wb') as file:
    pickle.dump(merged_df, file)
