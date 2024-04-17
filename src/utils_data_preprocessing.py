import pandas as pd
import numpy as np
import gzip

import anndata
import pyfaidx
import pyBigWig

import torch
from torch.utils.data import Dataset, DataLoader

NAME_DATASET = ['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

""" Concatenate all time points into a single anndata. Save it at provided save path. """
def concat_data(data_path, cell_type_path, save_path, name_datasets=NAME_DATASET):

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

""" Create pseudo-bulk ATAC data. Pseudo bulk are defined by groupping the data by time point and cell types.  """
def pseudo_bulk(adata, col):
    
    #Agggregate the ATAC count matrix by cell type -> pseudo bulk
    adata.strings_to_categoricals()
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    indicator = pd.get_dummies(adata.obs[col])

    return anndata.AnnData(
        indicator.values.T @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=indicator.columns))

def normalize_bulk(adata, multiply_by=100000):
    total_reads = adata.X.sum(axis=1)
    normalized = (adata.X.T/total_reads * multiply_by).T

    return normalized

""" Fetch the sequence on the reference genome at ATAC peaks. """
def fetch_sequence(adata, path_genome, len_seq = 2114):
    
    genome = pyfaidx.Fasta(path_genome)

    #Fetch sequence on reference genome using location of peaks
    adata.var['middle_peak'] = round((adata.var.end - adata.var.start)/2 + adata.var.start).astype('uint32')
    sequences = adata.var.apply(lambda x: 
                                            (genome[('chr' + x['chr'])][(x['middle_peak']-int(len_seq/2)):(x['middle_peak']+int(len_seq/2))]).seq, axis=1)
    sequences = sequences.str.upper()

    return sequences


def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def encode_sequence(adata):
    
    #One hot encode the the sequence
    return [one_hot_encode(seq) for seq in adata.var.sequence]

def encode_sequence(sequences):
    
    #One hot encode the the sequence
    return [one_hot_encode(seq) for seq in sequences]

def get_continuous_ATAC(bw, seq_loc, total_reads, seq_len=1000, window_size=200):
    bp_around = int(window_size/2 + seq_len/2)
    val = bw.values(seq_loc.chr, seq_loc.middle - bp_around, 
                    seq_loc.middle + bp_around)
    ATAC = [sum(val[i:(i+window_size+1)]) for i in range(0,seq_len+1)]

    return ATAC

def get_continuous_ATAC_background(bw, seq_loc, total_reads, seq_len=1000, window_size=200):
    
    seq_start = int(seq_loc.start + (len(seq_loc.sequence)- seq_len)/2)

    val = bw.values(seq_loc.chr[3:], seq_start, 
                    seq_start + seq_len)
    ATAC = [sum(val[i:(i+window_size+1)]) for i in range(0,seq_len+1)]

    return ATAC

def compute_GC_content(seq):

    return [sum(x.count(n) for n in ("G", "C"))/len(x) for x in seq]


""" Create a sequence ATAC matrix. For each sequence extracts ATAC signal (discrete) at pseudo bulk level (cell_type + time point) """
def get_sequence_ATAC_dicrete(adata):
    
    #Format matrix
    sequence_ATAC = pd.DataFrame(anndata.X, index = anndata.obs_names, 
                            columns = anndata.var_names)

    sequence_ATAC = sequence_ATAC.transpose()
    sequence_ATAC["peakID"] = sequence_ATAC.index

    #wide_to_long need same name of columns name 
    batch_dict = dict(zip(sequence_ATAC.columns.values, ['batch' + str(x) for x in np.arange(len(sequence_ATAC.columns.values))]))
    convert_back = dict(zip(np.arange(len(sequence_ATAC.columns.values)), sequence_ATAC.columns.values))
    sequence_ATAC = sequence_ATAC.rename(columns=batch_dict)

    sequence_ATAC = pd.wide_to_long(sequence_ATAC, stubnames='batch', i='peakID', j='ATAC')

    sequence_ATAC = sequence_ATAC.reset_index()
    sequence_ATAC.ATAC = [convert_back[x] for x in sequence_ATAC.ATAC]

    sequence_ATAC['cell_type'] = [x.split('D')[0] for x in sequence_ATAC.ATAC]
    sequence_ATAC['dataset'] = ['D' + x.split('D')[1] for x in sequence_ATAC.ATAC]

    sequence_ATAC = sequence_ATAC.drop(columns=['ATAC'])
    sequence_ATAC = sequence_ATAC.rename(columns={'batch':'ATAC'})

    #Add sequence for each peak
    ...

    return sequence_ATAC



class SequenceDataset(Dataset):
    """Genomic sequence pytorch dataset with ATAC signal"""

    def __init__(self, h5ad_file, discrete = True):
        """
        Arguments:
            h5ad_file (string): Path to the ATAC Anndata object
            discrete (bool): Whether to get ATAC signal as dicrete or continous values 
        """
        self.ATAC_count_matrix = anndata.read_h5ad(h5ad_file)

        if discrete:
            self.sequence_ATAC = get_sequence_ATAC_dicrete(self.ATAC_count_matrix)
        else:
            self.sequence_ATAC = get_sequence_ATAC_dicrete(self.ATAC_count_matrix)
            #ADD CONTINOUS SIGNAL HERE

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




    