import pandas as pd
import numpy as np
import gzip

import anndata
import scanpy as sc
import pyfaidx
import pyBigWig

import torch
from torch.utils.data import Dataset, DataLoader

NAME_DATASET = ['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

def preprocess_data(data_path, cell_type_path, save_path, name_datasets=NAME_DATASET):

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
    cell_types = pd.read_csv(cell_type_path, index_col=0)

    barcodes = cell_types.index.values.tolist()
    mask_barcodes = pd.Series(adata.obs_names.values.tolist()).isin(barcodes)
    adata = adata[mask_barcodes]

    cell_types.index = cell_types.index.values.tolist()
    adata.obs['cell_type'] = cell_types.cell_type[adata.obs_names]

    #Rename batch number to be name dataset
    adata.obs.batch = adata.obs.batch.cat.rename_categories(NAME_DATASET)

    adata.write(save_path)

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

def fetch_sequence(adata, path_genome, bp_around=0):
    
    genome = pyfaidx.Fasta(path_genome)

    adata.var['chr'] = adata.var_names.to_series().str.split(':', n=1).str.get(0)
    adata.var['start'] = adata.var_names.to_series().apply(lambda st: st[st.find(":")+1:st.find("-")]).astype('int')
    adata.var['end'] = adata.var_names.to_series().str.split('-', n=1).str.get(1).astype('int')

    #Remove scaffold chromosomes
    adata = adata[:,np.logical_or(np.logical_or(adata.var.chr.str.isnumeric(), adata.var.chr == 'X'), adata.var.chr == 'Y')]

    #Fetch sequence on reference genome using location of peaks
    adata.var.chr = 'chr' + adata.var.chr
    
    adata.var['region_start'] = adata.var.start - bp_around
    adata.var.loc[adata.var.region_start < 0,'region_start'] = 0
    adata.var['region_end'] = adata.var.end + bp_around
    
    adata.var['sequence'] = adata.var.apply(lambda x: (genome[x['chr']][x['region_start']:x['region_end']]).seq, axis=1)

    return adata

def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def encode_sequence(adata):
    
    #Capitalize all sequence
    adata.var.sequence = adata.var.sequence.str.upper()
    
    #One hot encode the the sequence
    return [one_hot_encode(seq) for seq in adata.var.sequence]

def get_continous_track(peak_metadata):

    for name, group in peak_metadata.groupby(['cell_type', 'dataset']):
        print(name)
        bw_file = '../results/bam_cell_type/' + name[0]
        
        bw = pyBigWig.open('../results/bam_cell_type/D8_1/Somite.bw')
        bw.values("10", 100004794, 100005512)




class SequenceDataset(Dataset):
    """Genomic sequence pytorch dataset with discrete ATAC signal"""

    def __init__(self, h5ad_file, path_ref_genome):
        """
        Arguments:
            h5ad_file (string): Path to the ATAC Anndata object 
            path_ref_genome (string): Path to the reference genome used to fetch sequence
        """
        self.ATAC_count_matrix = anndata.read_h5ad(h5ad_file)
        self.path_ref_genome = path_ref_genome

        #Create pseudo bulk  by cell types + time point
        self.ATAC_count_matrix.obs['cell_type_batch'] = self.ATAC_count_matrix.obs.cell_type.astype('str') + self.ATAC_count_matrix.obs.batch.astype('str')
        self.ATAC_count_matrix = pseudo_bulk(self.ATAC_count_matrix,'cell_type_batch')

        #Normalize the aggregated count matrix


        #Get sequence for each of the peaks in count matrix and one-hot encode it
        self.ATAC_count_matrix = fetch_sequence(self.ATAC_count_matrix, path_genome=self.path_ref_genome)
        #self.ATAC_count_matrix.var.encoded_seq = encode_sequence(self.ATAC_count_matrix)

        #Each data point is peak with time, cell type, and aggregated ATAC signal
        self.sequence_ATAC = pd.DataFrame(self.ATAC_count_matrix.X, index=self.ATAC_count_matrix.obs_names, 
                            columns=self.ATAC_count_matrix.var_names)

        self.sequence_ATAC = self.sequence_ATAC.transpose()

        #Change cell type + time id so that we can use pandas wide_to_long function
        batch_dict = dict(zip(self.sequence_ATAC.columns.values, ['batch' + str(x) for x in np.arange(len(self.sequence_ATAC.columns.values))]))
        convert_back = dict(zip(np.arange(len(self.sequence_ATAC.columns.values)), self.sequence_ATAC.columns.values))

        self.sequence_ATAC = self.sequence_ATAC.rename(columns=batch_dict)
        self.sequence_ATAC["peakID"] = self.sequence_ATAC.index

        self.sequence_ATAC = pd.wide_to_long(self.sequence_ATAC, stubnames='batch', i='peakID', j='ATAC')

        self.sequence_ATAC = self.sequence_ATAC.reset_index()
        self.sequence_ATAC.ATAC = [convert_back[x] for x in self.sequence_ATAC.ATAC]

        self.sequence_ATAC['cell_type'] = [x.split('D')[0] for x in self.sequence_ATAC.ATAC]
        self.sequence_ATAC['dataset'] = ['D' + x.split('D')[1] for x in self.sequence_ATAC.ATAC]

        self.sequence_ATAC = self.sequence_ATAC.drop(columns=['ATAC'])
        self.sequence_ATAC = self.sequence_ATAC.rename(columns={'batch':'ATAC'})

        #Add sequence for each peak

    def __len__(self):
        return len(self.sequence_ATAC.shape[0])

    def __getitem__(self, idx):

        return self.sequence_ATAC[idx]




    