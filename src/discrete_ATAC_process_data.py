import os
import pickle 
import anndata
import numpy as np
import pandas as pd

from utils_data_preprocessing import concat_data, pseudo_bulk, normalize_bulk, fetch_sequence, encode_sequence

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

#Normalize the aggregated count matrix
adata.X = normalize_bulk(adata)

#Split the object, when all at once does not save
#adata = adata[:,1000001:1499999].copy()

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

encoded_sequences = encode_sequence(adata)
encoded_sequences = pd.Series(encoded_sequences, index=adata.var_names)

with open('../results/encoded_seq.pkl', 'wb') as file:
    pickle.dump(encoded_sequences, file)

adata.write('../results/pre_processed_3.h5ad')

