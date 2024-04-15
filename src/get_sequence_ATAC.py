import os
import pickle 
import anndata
import numpy as np
import pandas as pd
import pyfaidx

from utils_data_preprocessing import concat_data, pseudo_bulk, compute_GC_content, fetch_sequence, encode_sequence

#Concatenate all dataset into single anndata
#--------------------------------------------
if (not os.path.isfile('../results/concat.h5ad')):
    adata = concat_data('../data/initial_10x_outputs/filtered_features', 
                        '../results/cell_types.csv', 
                        '../results/concat.h5ad')
else:
    adata = anndata.read_h5ad('../results/concat.h5ad')

#Create pseudo bulk by time point + cell_type
#--------------------------------------------
adata.obs['cell_type_batch'] = adata.obs.cell_type.astype('str') + adata.obs.batch.astype('str')
adata = pseudo_bulk(adata=adata,col='cell_type_batch')

#adata = adata[:,:500000].copy()

#Get sequence for each of the peaks in count matrix and one-hot encode it
#--------------------------------------------
#Get chromosomal location for each peak
adata.var['chr'] = adata.var_names.to_series().str.split(':', n=1).str.get(0)
adata.var['start'] = adata.var_names.to_series().apply(lambda st: st[st.find(":")+1:st.find("-")]).astype('int')
adata.var['end'] = adata.var_names.to_series().str.split('-', n=1).str.get(1).astype('int')

#Remove scaffold chromosomes
adata = adata[:,np.logical_or(np.logical_or(adata.var.chr.str.isnumeric(), adata.var.chr == 'X'), adata.var.chr == 'Y')]

#Fetch sequence and remove sequences with N nucleotides
adata.var['sequence'] = fetch_sequence(adata, path_genome='../data/hg38.fa')
adata = adata[:, np.logical_not(adata.var.sequence.str.contains("N"))]

#encoded_sequences = encode_sequence(adata)
#encoded_sequences = pd.Series(encoded_sequences, index=adata.var_names)

#with open('../results/encoded_seq_ATAC.pkl', 'wb') as file:
    #pickle.dump(encoded_sequences, file)

with open('../results/peaks_location.pkl', 'wb') as file:
    pickle.dump(adata.var[['chr','start','end', 'sequence']], file)

del adata

#Get matched GC content background sequence
#--------------------------------------------
with open('../results/peaks_location.pkl', 'rb') as file:
    peaks = pickle.load(file)

#Compute GC content
peaks['GC_cont'] = compute_GC_content(peaks.sequence)

len_seq = 2114
path_genome = '../data/hg38.fa'

#Get size of each chromosome, so that truly random
size_chrom = pd.read_csv('../data/size_genome.txt', sep='\t', header=None, index_col=0)
size_chrom = size_chrom.loc[['chr' + c for c in np.unique(peaks.chr)]]

#Generate random start location on chromosome
p_chrom = (size_chrom/np.sum(size_chrom)).iloc[:,0].tolist()
rand_chr = np.random.choice(size_chrom.index, p=p_chrom, size=peaks.shape[0]*5)
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

merged_df = pd.merge_asof(random_seq, peaks.reset_index(), on='GC_cont', direction='nearest', suffixes=('','_peak'), tolerance=3e-2)
merged_df = peaks.merge(merged_df[['peakID','start','sequence']], how='left', left_index=True, right_on='peakID').set_index('peakID')
merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
merged_df['GC_cont_y'] = [sum(x.count(n) for n in ("G", "C"))/len(x) if isinstance(x, str) else 0 for x in merged_df.sequence_y]

with open('../results/match_GC.pkl', 'wb') as file:
    pickle.dump(merged_df, file)