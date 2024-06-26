import torch
from torch.utils.data import Dataset

import pickle
import numpy as np
import pandas as pd
import re
#import h5py

from timeit import default_timer as timer

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

    if type(desired_order) is str:
        desired_order = [desired_order]
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

        #Only keep track coresponding to given sequences
        self.ATAC_track =  self.ATAC_track.loc[self.sequences_id]

        self.pseudo_bulk = self.ATAC_track.pseudo_bulk.astype('category')
        
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


class PeaksDataset2(Dataset):
    """Peaks and background sequences for main model training"""

    def __init__(self, path_sequences_peaks, path_sequences_back, paths_ATAC_peaks, paths_ATAC_back, chr_include):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            paths_ATAC_peaks (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            paths_ATAC_back (list of strings): Paths to the pickle files of ATAC tracks and corresponding seq_id, c_type, and if track is_defined
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split

        """

        #Load peak ATAC tracks
        self.sequences, self.sequences_id = self.load_sequences(path_sequences_peaks, chr_include)
        self.ATAC_track, self.is_defined, self.idx_seq, self.c_type = self.load_ATAC_tracks(paths_ATAC_peaks, chr_include)
        
        #Load the background ATAC tracks
        sequences, sequences_id = self.load_sequences(path_sequences_back, chr_include)
        
        #Add max seq_id of peaks so that it is unique
        max_seq_id = np.max(self.sequences_id)
        sequences_id = sequences_id + max_seq_id

        self.sequences = torch.cat((self.sequences, sequences), 0); self.sequences_id = np.concatenate((self.sequences_id, sequences_id), 0)

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
        sequences = sequences.sequence

        #Encode sequences
        self.len_seq = len(sequences.iloc[0])
        sequences = sequences.apply(lambda x: one_hot_encode(x))

        #Store in tensor for faster access
        sequences_id = sequences.index.to_numpy()
        sequences = torch.from_numpy(np.stack(sequences.values))
        sequences = sequences.permute(0,2,1)

        return sequences, sequences_id

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
        print(input.squeeze().shape)

        #Add cell type token to input
        #Repeat one-hot encoded cell type so that shape = seq_len x nb_cells
        c_type = self.c_type[idx]
        
        mapping = dict(zip(self.unique_c_type, range(len(self.unique_c_type))))    
        c_type = mapping[c_type]
        c_type = torch.from_numpy(np.eye(len(self.unique_c_type))[c_type])

        c_type = c_type.tile((input.shape[-1],1)).permute(1,0)[:,:]
        print(c_type.shape)
        input = torch.cat((input.squeeze(), c_type), dim=0)

        #Get which tracks should be omitted for the loss computation
        is_defined = self.is_defined[idx, :]

        return input, tracks, is_defined


class PeaksDataset_w_bias(Dataset):
    """Peaks and background sequences for main model training"""

    def __init__(self, path_sequences_peaks, path_sequences_back, path_ATAC_peaks, path_ATAC_back, chr_include, time_order, nb_back, tn5_bias_file):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            path_ATAC_peaks (string): Path to the pickle file with ATAC tracks per datasets and time points for peaks regions
            path_ATAC_back (string): Path to the pickle file with ATAC tracks per datasets and time points for background regions
            chr_include (list of string): only keep the sequences on the provided chromosome, used to define train/split
            time_order (list of string): define order in which the time should be returned 
            nb_back (int): number of background regions to include in training set

        """
        self.time_order = time_order

        #Open sequences files
        with open(path_sequences_peaks, 'rb') as file:
            self.sequences = pickle.load(file)

        with open(path_sequences_back, 'rb') as file:
            self.sequences = pd.concat([self.sequences, pickle.load(file).sample(nb_back)])

        self.sequences.index = self.sequences.chr.astype('str') + ":" + self.sequences.start.astype('str') + "-" + self.sequences.end.astype('str')

        #Only keep sequences from provided chromosomes
        self.sequences = self.sequences[self.sequences.chr.isin(chr_include)]
        
        self.seq_chr =  self.sequences.chr.to_numpy()
        self.seq_pos =  self.sequences.middle_peak.to_numpy()
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

        #Only keep track coresponding to given sequences
        self.ATAC_track = self.ATAC_track.loc[self.sequences_id]

        self.pseudo_bulk = self.ATAC_track.pseudo_bulk.astype('category')
        self.c_type = [re.findall('[A-Z][^A-Z]*', x) for x in self.pseudo_bulk]
        self.time = pd.Series([x[0] for x in self.c_type]); self.c_type = [x[1] for x in self.c_type]
        self.time.index = self.pseudo_bulk.index
        
        self.ATAC_track_seq = self.ATAC_track.index.to_numpy()
        self.ATAC_track = self.ATAC_track.iloc[:,0]
        self.ATAC_track = torch.from_numpy(np.array(self.ATAC_track.values.tolist())).type(torch.float32)

        #Create dataframe with seq id + cell type 
        self.seq_c_type = pd.DataFrame({"seq_id": self.ATAC_track_seq, "c_type": self.c_type})
        self.seq_c_type.drop_duplicates(inplace=True)

        #Store all unique cell type name
        self.unique_c_type = np.sort(np.unique(self.seq_c_type.c_type))

        #Store the tn5 bias 
        self.tn5_bias =  self.load_tn5_bias(tn5_bias_file,chr_include)
        print(self.tn5_bias)


    def __len__(self):
        return self.seq_c_type.shape[0]

    def __getitem__(self, idx):

        seq_c_type = self.seq_c_type.iloc[idx,:]

        seq_idx = np.where(self.sequences_id == seq_c_type["seq_id"])[0]
        input = self.sequences[seq_idx,:,:] 
        
        idx_input = np.argwhere(np.logical_and(self.ATAC_track_seq == seq_c_type["seq_id"], np.array(self.c_type) == seq_c_type["c_type"])).squeeze()
        tracks = self.ATAC_track[idx_input, :]

        if tracks.ndim < 2:
            tracks = tracks[None,:]

        #Order tracks so that always returned in same order
        #Keep which time point not present so skip during loss computation
        time = self.time.iloc[idx_input]
        indexes = order_categories(self.time_order, time)
        indexes = [-1 if i is None else i for i in indexes]
        
        #Add zero tracks for not defined time point
        missing_tracks = torch.zeros((4, tracks.shape[1]))

        #Order
        for idx,i in enumerate(indexes):
            if i != -1:
                missing_tracks[idx] = tracks[i,:]
        
        tracks = missing_tracks

        #Add cell type token to input
        #Repeat one-hot encoded cell type so that shape = seq_len x nb_cells
        c_type = seq_c_type["c_type"]
        
        mapping = dict(zip(self.unique_c_type, range(len(self.unique_c_type))))    
        c_type = mapping[c_type]
        c_type = torch.from_numpy(np.eye(len(self.unique_c_type), dtype=np.float32)[c_type])

        #Repeat and reshape
        c_type = c_type.tile((input.shape[-1],1)).permute(1,0)[:,:]
        input = torch.cat((input.squeeze(), c_type), dim=0)

        #Get bias
        chr, pos = self.seq_chr[seq_idx][0], int(self.seq_pos[seq_idx][0])
        bp_around =  int(tracks.shape[0]/2)

        bias = self.tn5_bias[chr][pos-bp_around: pos+bp_around]

        return input, tracks, indexes, bias
    
    def load_tn5_bias(self, tn5_bias_file, chr):
        chr = ["chr" + x for x in chr] 

        dictionary = {}
        with h5py.File(tn5_bias_file, "r") as f:
            for key in f.keys():
                if key in chr:
                    ds_arr = f[key][()] 
                    dictionary[key[3:]] = ds_arr

        return dictionary


