import torch
from torch.utils.data import Dataset, DataLoader

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

        #Only keep sequences from provided chromosomes
        self.chr = '|'.join(chr_include)
        self.sequences = self.sequences[self.sequences.chr.str.contains(self.chr)]
        self.sequences = self.sequences.sequence

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        with open(path_ATAC_signal, 'rb') as file:
            self.ATAC_track = pickle.load(file)

        print(self.ATAC_track.shape)
        print(self.sequences.shape)

        #Only keep track for sequences in chrom test
        self.ATAC_track =  self.ATAC_track[self.sequences.index]

        print(self.ATAC_track.shape)

    def __len__(self):
        return self.ATAC_track.shape[0]

    def __getitem__(self, idx):
        
        track = self.ATAC_track.iloc[idx,0]
        input = torch.from_numpy(self.sequences[self.ATAC_track.index[idx]])

        return input, track

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

        #Only keep sequences from provided chromosomes
        self.chr = '|'.join(chr_include)
        self.sequences = self.sequences[self.sequences.chr.str.contains(self.chr)]
        self.sequences = self.sequences.sequence

        #Load the ATAC track
        with open(path_ATAC_peaks, 'rb') as file:
            self.ATAC_track = pickle.load(file)

        with open(path_ATAC_back, 'rb') as file:
            self.ATAC_track = pd.concat([self.ATAC_track, pickle.load(file)]) 

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        self.ATAC_track['pseudo_bulk'] = (self.ATAC_track.time.astype(str) + self.ATAC_track.cell_type.astype(str)).astype('category')
        self.pseudo_bulk = self.ATAC_track.pseudo_bulk

        self.ATAC_track = self.ATAC_track.iloc[:,0]

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):

        input = self.sequences.iloc[idx].ToTensor()
        tracks = self.ATAC_track[self.sequences.index[idx]]

        #Order tracks so that always returned in same order
        pseudo_bulk = self.pseudo_bulk[self.sequences.index[idx]]
        tracks.index = pseudo_bulk
        tracks = tracks.loc[self.pseudo_bulk_order].ToTensor()

        return input, tracks

