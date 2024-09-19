#Define the pytorch class dataset storing all the sequence and associated accessibility values
#------------------------------------------
import torch
from torch.utils.data import Dataset

import pickle
import numpy as np
import pandas as pd
import h5py

from timeit import default_timer as timer

from data_processing.utils_data_preprocessing import one_hot_encode

#Utils function to order the tracks always in the same order
def order_categories(categories, desired_order):

    if type(desired_order) is str:
        desired_order = [desired_order]
    category_index_map = {category: index for index, category in enumerate(desired_order)}
    
    indexes = [category_index_map.get(category) for category in categories]
    return indexes

class PeaksDataset_w_bias(Dataset):
    """Load Peaks and background sequences for main model training with bias correction"""

    def __init__(self, path_sequences_peaks: str, path_sequences_back: str, 
                 paths_ATAC_peaks: list, paths_ATAC_back: list, chr_include: list, 
                 tn5_bias_file: str):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            paths_ATAC_peaks (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            paths_ATAC_back (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split
            tn5_bias_file (string): path to the precomputed tn5 bias 

        """

        #Load peak ATAC tracks
        self.sequences, self.sequences_id, self.positions = self.load_sequences(path_sequences_peaks, chr_include)
        self.ATAC_track, self.is_defined, self.idx_seq, self.c_type = self.load_ATAC_tracks(paths_ATAC_peaks, chr_include)
        
        #Load the background ATAC tracks
        sequences, sequences_id, positions = self.load_sequences(path_sequences_back, chr_include)
        
        #Add max seq_id of peaks so that it is unique
        max_seq_id = np.max(self.sequences_id) + 1
        sequences_id = sequences_id + max_seq_id

        self.sequences = torch.cat((self.sequences, sequences), 0); self.sequences_id = np.concatenate((self.sequences_id, sequences_id), 0)
        self.positions = np.concatenate((self.positions, positions), 0)
        
        ATAC_track, is_defined, idx_seq, c_type = self.load_ATAC_tracks(paths_ATAC_back, chr_include)

        #Add max seq_id of peaks so that it is unique
        idx_seq = idx_seq + max_seq_id

        self.ATAC_track = torch.cat((self.ATAC_track, ATAC_track), 0); self.is_defined = torch.cat((self.is_defined, is_defined), 0); self.idx_seq = torch.cat((self.idx_seq, idx_seq), 0); self.c_type = np.concatenate((self.c_type, c_type), 0)

        #Define order of c_type for encoding
        self.unique_c_type = np.sort(np.unique(self.c_type))

        #Load the tn5 bias track
        self.tn5_bias = self.load_tn5_bias(tn5_bias_file, chr_include)

    def __len__(self):
        return self.ATAC_track.shape[0]
    
    def load_sequences(self, path_sequence, chr_include):

        #Open sequences files
        with open(path_sequence, 'rb') as file:
            sequences = pickle.load(file)

        #Reset index to be integer
        sequences.reset_index(drop=True, inplace=True)
        
        #Only keep sequences from provided chromosomes
        sequences = sequences[sequences.chr.isin(chr_include)]
        
        #Keep chr + pos dataframe
        sequences.rename(columns={"middle_peak": "middle"}, inplace=True)

        positions =  sequences.loc[:,["chr", "middle"]]
        positions = positions.to_numpy()

        #Encode sequences
        sequences = sequences.sequence
        self.len_seq = len(sequences.iloc[0])
        sequences = sequences.apply(lambda x: one_hot_encode(x))

        #Store in tensor for faster access
        sequences_id = sequences.index.to_numpy()
        sequences = torch.from_numpy(np.stack(sequences.values))
        sequences = sequences.permute(0,2,1)

        return sequences, sequences_id, positions

    def load_ATAC_tracks(self, paths_ATAC_track:list, chr_include:list):
        
        #Define which region are use in split
        with open(paths_ATAC_track[0], 'rb') as file:
            chr_track = pd.Series(pickle.load(file))
        keep_track = chr_track.isin(chr_include)

        with open(paths_ATAC_track[1], 'rb') as file:
           ATAC_track = pickle.load(file)
        ATAC_track = ATAC_track[keep_track,:,:]
        
        with open(paths_ATAC_track[2], 'rb') as file:
            is_defined = pickle.load(file)
        is_defined = is_defined[keep_track,:]

        with open(paths_ATAC_track[3], 'rb') as file:
            idx_seq = pickle.load(file)
        idx_seq = idx_seq[keep_track]

        with open(paths_ATAC_track[4], 'rb') as file:
            c_type = pickle.load(file)
        c_type = c_type[keep_track]

        return ATAC_track, is_defined, idx_seq, c_type

    def load_tn5_bias(self, tn5_bias_file, chr):
        chr = ["chr" + x for x in chr] 

        dictionary = {}
        with h5py.File(tn5_bias_file, "r") as f:
            for key in f.keys():
                if key in chr:
                    ds_arr = f[key][()] 
                    dictionary[key[3:]] = ds_arr

        return dictionary

    def __getitem__(self, idx):

        #Get track and associated encoded sequence input
        tracks = self.ATAC_track[idx,:,:]
        
        seq_idx = self.idx_seq[idx].item()

        seq_idx = np.where(self.sequences_id == seq_idx)[0]
        input = self.sequences[seq_idx,:,:]

        #Add cell type token to input
        #Repeat one-hot encoded cell type so that shape = seq_len x nb_cells
        c_type = self.c_type[idx]
        
        mapping = dict(zip(self.unique_c_type, range(len(self.unique_c_type))))    
        c_type = mapping[c_type]
        c_type = torch.from_numpy(np.eye(len(self.unique_c_type))[c_type])

        c_type = c_type.tile((input.shape[-1],1)).permute(1,0)[:,:]
        input = torch.cat((input.squeeze(), c_type), dim=0)

        #Get which tracks should be omitted for the loss computation
        is_defined = self.is_defined[idx, :]

        #Get tn5 bias to add
        pos =  self.positions[seq_idx,:][0]
        tn5_bias = self.tn5_bias[pos[0]][(pos[1]-512):(pos[1]+512)]
        tn5_bias = torch.from_numpy(tn5_bias)

        return input, tracks, is_defined, tn5_bias

class PeaksDataset_wo_bias(Dataset):
    """Peaks and background sequences for main model training without bias correction"""

    def __init__(self, path_sequences_peaks: str, path_sequences_back: str, 
                 paths_ATAC_peaks: list, paths_ATAC_back: list, chr_include: list):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            paths_ATAC_peaks (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            paths_ATAC_back (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split
        """

        #Load peak ATAC tracks
        self.sequences, self.sequences_id, self.positions = self.load_sequences(path_sequences_peaks, chr_include)
        self.ATAC_track, self.is_defined, self.idx_seq, self.c_type = self.load_ATAC_tracks(paths_ATAC_peaks, chr_include)
        
        #Load the background ATAC tracks
        sequences, sequences_id, positions = self.load_sequences(path_sequences_back, chr_include)
        
        #Add max seq_id of peaks so that it is unique
        max_seq_id = np.max(self.sequences_id) + 1
        sequences_id = sequences_id + max_seq_id

        self.sequences = torch.cat((self.sequences, sequences), 0); self.sequences_id = np.concatenate((self.sequences_id, sequences_id), 0)
        self.positions = np.concatenate((self.positions, positions), 0)
        
        ATAC_track, is_defined, idx_seq, c_type = self.load_ATAC_tracks(paths_ATAC_back, chr_include)

        #Add max seq_id of peaks so that it is unique
        idx_seq = idx_seq + max_seq_id

        self.ATAC_track = torch.cat((self.ATAC_track, ATAC_track), 0); self.is_defined = torch.cat((self.is_defined, is_defined), 0); self.idx_seq = torch.cat((self.idx_seq, idx_seq), 0); self.c_type = np.concatenate((self.c_type, c_type), 0)

        #Define order of c_type for encoding
        self.unique_c_type = np.sort(np.unique(self.c_type))

    def __len__(self):
        return self.ATAC_track.shape[0]
    
    def load_sequences(self, path_sequence, chr_include):

        #Open sequences files
        with open(path_sequence, 'rb') as file:
            sequences = pickle.load(file)

        #Reset index to be integer
        sequences.reset_index(drop=True, inplace=True)
        
        #Only keep sequences from provided chromosomes
        sequences = sequences[sequences.chr.isin(chr_include)]
        
        #Keep chr + pos dataframe
        sequences.rename(columns={"middle_peak": "middle"}, inplace=True)

        positions =  sequences.loc[:,["chr", "middle"]]
        positions = positions.to_numpy()

        #Encode sequences
        sequences = sequences.sequence
        self.len_seq = len(sequences.iloc[0])
        sequences = sequences.apply(lambda x: one_hot_encode(x))

        #Store in tensor for faster access
        sequences_id = sequences.index.to_numpy()
        sequences = torch.from_numpy(np.stack(sequences.values))
        sequences = sequences.permute(0,2,1)

        return sequences, sequences_id, positions

    def load_ATAC_tracks(self, paths_ATAC_track:list, chr_include:list):
        
        #Define which region are use in split
        with open(paths_ATAC_track[0], 'rb') as file:
            chr_track = pd.Series(pickle.load(file))
        keep_track = chr_track.isin(chr_include)

        with open(paths_ATAC_track[1], 'rb') as file:
           ATAC_track = pickle.load(file)
        ATAC_track = ATAC_track[keep_track,:,:]
        
        with open(paths_ATAC_track[2], 'rb') as file:
            is_defined = pickle.load(file)
        is_defined = is_defined[keep_track,:]

        with open(paths_ATAC_track[3], 'rb') as file:
            idx_seq = pickle.load(file)
        idx_seq = idx_seq[keep_track]

        with open(paths_ATAC_track[4], 'rb') as file:
            c_type = pickle.load(file)
        c_type = c_type[keep_track]

        return ATAC_track, is_defined, idx_seq, c_type

    def __getitem__(self, idx):

        #Get track and associated encoded sequence input
        tracks = self.ATAC_track[idx,:,:]
        
        seq_idx = self.idx_seq[idx].item()

        seq_idx = np.where(self.sequences_id == seq_idx)[0]
        input = self.sequences[seq_idx,:,:]

        #Add cell type token to input
        #Repeat one-hot encoded cell type so that shape = seq_len x nb_cells
        c_type = self.c_type[idx]
        
        mapping = dict(zip(self.unique_c_type, range(len(self.unique_c_type))))    
        c_type = mapping[c_type]
        c_type = torch.from_numpy(np.eye(len(self.unique_c_type))[c_type])

        c_type = c_type.tile((input.shape[-1],1)).permute(1,0)[:,:]
        input = torch.cat((input.squeeze(), c_type), dim=0)

        #Get which tracks should be omitted for the loss computation
        is_defined = self.is_defined[idx, :]

        return input, tracks, is_defined



