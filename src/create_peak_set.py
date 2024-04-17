import os
import subprocess
import anndata
import numpy as np

from utils_data_preprocessing import concat_data

#Concatenate all dataset into single anndata
#--------------------------------------------
if (not os.path.isfile('../results/concat.h5ad')):
    adata = concat_data('../data/initial_10x_outputs/filtered_features', 
                        '../results/cell_types.csv', 
                        '../results/concat.h5ad')
else:
    adata = anndata.read_h5ad('../results/concat.h5ad')

#Get chromosomal location for each peak
adata.var['chr'] = adata.var_names.to_series().str.split(':', n=1).str.get(0)
adata.var['start'] = adata.var_names.to_series().apply(lambda st: st[st.find(":")+1:st.find("-")]).astype('uint32')
adata.var['end'] = adata.var_names.to_series().str.split('-', n=1).str.get(1).astype('uint32')

#Remove scaffold chromosomes
adata = adata[:,np.logical_or(np.logical_or(adata.var.chr.str.isnumeric(), adata.var.chr == 'X'), adata.var.chr == 'Y')]

peaks = adata.var
peaks.to_csv("../results/all_peaks.bed", sep='\t', index=False, header=None)

#Command line 
#sortBed -i ../results/all_peaks.bed > ../results/all_peaks_sorted.bed
#bedtools merge -i ../results/all_peaks_sorted.bed > ../results/common_peaks.bed