import torch
from torch.utils.data import Dataset

import pickle
import numpy as np
import pandas as pd

from data_processing.utils_data_preprocessing import one_hot_encode

class BiasDataset(Dataset):
    """Background sequences for bias model training"""

    def __init__(self, path_sequences, path_ATAC_signal, chr_include):
        """
        Arguments:
            path_sequences (string): Path to the pickle file with background regions sequences
            path_ATAC_signal (string): Path to the pickle file with ATAC tracks per datasets and time points
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split

        """
        with open(path_sequences, 'rb') as file:
            self.sequences = pickle.load(file)
        
        self.sequences.index = self.sequences.chr.astype('str') + ":" + self.sequences.start.astype('str') + "-" + self.sequences.end.astype('str')

        #Only keep sequences from provided chromosomes
        self.sequences = self.sequences[self.sequences.chr.isin(chr_include)]
        self.sequences = self.sequences.sequence

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        #Store in tensor for faster access
        self.sequences_id = self.sequences.index.to_numpy()
        self.sequences = torch.from_numpy(np.stack(self.sequences.values))
        self.sequences = self.sequences.permute(0,2,1)

        with open(path_ATAC_signal, 'rb') as file:
            self.ATAC_track = pickle.load(file)

        #Only keep track coresponding to given sequences
        self.ATAC_track =  self.ATAC_track.loc[self.sequences_id]
        
        self.ATAC_track_seq = self.ATAC_track.index.to_numpy()
        self.ATAC_track = self.ATAC_track.iloc[:,0]
        self.ATAC_track = torch.from_numpy(np.array(self.ATAC_track.values.tolist())).type(torch.float32)


    def __len__(self):
        return self.ATAC_track.shape[0]

    def __getitem__(self, idx):
        track = self.ATAC_track[idx,:]

        idx_input = np.argwhere(self.sequences_id == self.ATAC_track_seq[idx]).squeeze()
        input = self.sequences[idx_input,:]

        return input, track

#Utils function to order the tracks always in the same order
def order_categories(categories, desired_order):
    category_index_map = {category: index for index, category in enumerate(desired_order)}
    indexes = [category_index_map.get(category) for category in categories]
    return indexes

class PeaksDataset(Dataset):
    """Peaks and background sequences for main model training"""

    def __init__(self, path_sequences_peaks, path_sequences_back, path_ATAC_peaks, path_ATAC_back, chr_include, pseudo_bulk_order, nb_back):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            path_ATAC_peaks (string): Path to the pickle file with ATAC tracks per datasets and time points for peaks regions
            path_ATAC_back (string): Path to the pickle file with ATAC tracks per datasets and time points for background regions
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split
            pseudo_bulk_order (list of string): define order in which the pseudo_bulk should be returned 
            nb_back (int): number of background regions to include in training set

        """
        self.pseudo_bulk_order = pseudo_bulk_order

        #Open sequences files
        with open(path_sequences_peaks, 'rb') as file:
            self.sequences = pickle.load(file)

        with open(path_sequences_back, 'rb') as file:
            self.sequences = pd.concat([self.sequences, pickle.load(file).sample(nb_back)])

        self.sequences.index = self.sequences.chr.astype('str') + ":" + self.sequences.start.astype('str') + "-" + self.sequences.end.astype('str')

        #Only keep sequences from provided chromosomes
        self.sequences = self.sequences[self.sequences.chr.isin(chr_include)]
        self.sequences = self.sequences.sequence

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        #Store in tensor for faster access
        self.sequences_id = self.sequences.index.to_numpy()
        self.sequences = torch.from_numpy(np.stack(self.sequences.values))
        self.sequences = self.sequences.permute(0,2,1)

        #Load the ATAC track
        with open(path_ATAC_peaks, 'rb') as file:
            self.ATAC_track = pickle.load(file)

        with open(path_ATAC_back, 'rb') as file:
            self.ATAC_track = pd.concat([self.ATAC_track, pickle.load(file)]) 

        self.pseudo_bulk = self.ATAC_track.pseudo_bulk.astype('category')

        #Only keep track coresponding to given sequences
        self.ATAC_track =  self.ATAC_track.loc[self.sequences_id]

        self.ATAC_track_seq = self.ATAC_track.index.to_numpy()
        self.ATAC_track = self.ATAC_track.iloc[:,0]
        self.ATAC_track = torch.from_numpy(np.array(self.ATAC_track.values.tolist())).type(torch.float32)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        
        input = self.sequences[idx,:,:]
        
        idx_input = np.argwhere(self.ATAC_track_seq == self.sequences_id[idx]).squeeze()
        tracks = self.ATAC_track[idx_input, :]

        #Order tracks so that always returned in same order
        pseudo_bulk = self.pseudo_bulk[self.sequences_id[idx]].values
        indexes = order_categories(self.pseudo_bulk_order, pseudo_bulk)

        tracks = tracks[indexes,:]

        return input, tracks

