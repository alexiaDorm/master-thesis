import pandas as pd
import numpy as np
import anndata
import gzip

import scanpy as sc
import episcanpy.api as epi

NAME_DATASET = ['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

def preprocess_data(data_path, name_datasets=NAME_DATASET):

    #Load the data + basic filtering
    #----------------------------------------------------------------------------------------------------

    adatasets = []
    for name in name_datasets:

        path_count = data_path + '/filtered_feature_bc_matrix-' + name

        #Read count matrix
        data = anndata.read_mtx(path_count + "/matrix.mtx.gz").transpose()

        #Set features names and cell barcode
        feat_name = pd.read_csv(gzip.open(path_count + '/features.tsv.gz'), sep='\t', header=None, usecols=[0,2,3,4,5], names=['peakID','name2','type','chr','start','end'])
        barcodes = pd.read_csv(gzip.open(path_count + '/barcodes.tsv.gz'), sep='\t', header=None)
        barcodes = list(barcodes[0])

        data.var_names = feat_name['peakID']
        data.obs_names = barcodes

        #Only keep the ATAC features
        peak_name = feat_name[feat_name.type == 'Peaks']
        peak_name = list(peak_name['peakID'])
        data = data[:,peak_name]

        adatasets.append(data)  

    #Concatenate all data into single matrix
    #----------------------------------------------------------------------------------------------------

    adata = adatasets[0]
    adata = adata.concatenate(adatasets[1:], join="outer", index_unique='-')

    #Add cell type info, remove filtered cells
    cell_types = pd.read_csv('../data/cell_types.csv', index_col=0)

    barcodes = cell_types.index.values.tolist()
    mask_barcodes = pd.Series(adata.obs_names.values.tolist()).isin(barcodes)
    adata = adata[mask_barcodes]

    cell_types.index = cell_types.index.values.tolist()
    adata.obs['cell_type'] = cell_types.cell_type[adata.obs_names]

    adata.write('tmp.h5ad')

    return adata

def pseudo_bulk(adata, col):
    
    #Agggregate the ATAC countmatrix by cell type -> pseudo bulk
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    indicator = pd.get_dummies(adata.obs[col])

    return anndata.AnnData(
        indicator.values.T @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=indicator.columns))