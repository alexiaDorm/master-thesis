import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

import pickle
import numpy as np
import pandas as pd

from utils_data_preprocessing import one_hot_encode

class BiasDataset(Dataset):
    """Background sequences for bias model training"""

    def __init__(self, path_sequences, path_ATAC_signal):
        """
        Arguments:
            path_sequences (string): Path to the pickle file with background regions sequences
            path_ATAC_signal (string): Path to the pickle file with ATAC tracks per datasets and time points

        """
        with open(path_sequences, 'rb') as file:
            self.sequences = pickle.load(file).sequence

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        with open(path_ATAC_signal, 'rb') as file:
            self.ATAC_track = pickle.load(file)

    def __len__(self):
        return self.ATAC_track.shape[0]

    def __getitem__(self, idx):
        
        track = self.ATAC_track.iloc[idx,0]
        time = self.ATAC_track.iloc[idx,:].time
        cell_type = self.ATAC_track.iloc[idx,:].cell_type

        input = torch.from_numpy(self.sequences[self.ATAC_track.index[idx]])

        return input, time, cell_type, track

class PeaksDataset(Dataset):
    """Peaks and background sequences for model training"""

    def __init__(self, path_sequences_peaks, path_sequences_back, path_ATAC_peaks, path_ATAC_back):
        """
        Arguments:
            path_sequences_peaks (string): Path to the pickle file with peaks regions sequences
            path_sequences_back (string): Path to the pickle file with background regions sequences
            path_ATAC_peaks (string): Path to the pickle file with ATAC tracks per datasets and time points for peaks regions
            path_ATAC_back (string): Path to the pickle file with ATAC tracks per datasets and time points for background regions

        """
        #Open sequences files
        with open(path_sequences_peaks, 'rb') as file:
            self.sequences = pickle.load(file).sequence

        with open(path_sequences_back, 'rb') as file:
            self.sequences = pd.concat([self.sequences, pickle.load(file).sequence])

        #Encode sequences
        self.len_seq = len(self.sequences.iloc[0])
        self.sequences = self.sequences.apply(lambda x: one_hot_encode(x))

        #Load the ATAC track
        with open(path_ATAC_peaks, 'rb') as file:
            self.ATAC_track = pickle.load(file)

        with open(path_ATAC_back, 'rb') as file:
            self.ATAC_track = pd.concat([self.ATAC_track, pickle.load(file)])

        self.ATAC_track.time = self.ATAC_track.time.astype('category')
        self.time = pd.get_dummies(self.ATAC_track.time, dtype=float)

        self.ATAC_track.cell_type = self.ATAC_track.cell_type.astype('category')
        self.cell_type = pd.get_dummies(self.ATAC_track.cell_type, dtype=float)

        self.ATAC_track = self.ATAC_track.iloc[:,0]

    def __len__(self):
        return self.ATAC_track.shape[0]

    def __getitem__(self, idx):
        
        track = self.ATAC_track.iloc[idx]
        time = self.time.iloc[idx]
        cell_type = self.cell_type.iloc[idx]

        input = torch.from_numpy(self.sequences[self.ATAC_track.index[idx]])

        return input, time, cell_type, track

class BPNet(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, profile_kernel_size=75, out_pred_len=1000):
        super().__init__()
        """ BPNet architechture as in paper 
        
        Parameters
        -----------
        nb_conv: int (default 8)
            number of convolutional layers

        nb_filters: int (default 64)
            number of filters in the convolutional layers

        first_kernel: int (default 25)
            size of the kernel in the first convolutional layer

        rest_kernel: int (default 3)
            size of the kernel in all convolutional layers except the first one

        profile_kernel_size: int (default 75)
            size of the kernel in the profile convolution

        out_pred_len: int (default 1000)
            number of bp for which ATAC signal is predicted

        Model Architecture 
        ------------------------

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - Head: 
            > Profile prediction head: a multinomial probability of Tn5 insertion counts at each position 
            in the input sequence, deconvolution layer
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count per strand
        
        The predicted (expected) count at a specific position is a multiplication of the predicted total 
        counts and the multinomial probability at that position.

        -------------------------
        
        Reference: Avsec, Ž., Weilert, M., Shrikumar, A. et al. Base-resolution models of transcription-factor binding 
        reveal soft motif syntax. Nat Genet 53, 354–366 (2021). https://doi.org/10.1038/s41588-021-00782-6

        
        """
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.profile_kernel = profile_kernel_size
        self.out_pred_len = out_pred_len

        #Convolutional layers
        self.convlayers = nn.ModuleList()

        self.convlayers.append(nn.Conv1d(in_channels=4, 
                                         out_channels=self.nb_filters,
                                         kernel_size=self.first_kernel))
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Conv1d(in_channels=self.nb_filters, 
                                         out_channels=self.nb_filters,
                                         kernel_size=self.rest_kernel,
                                         dilation=2**i))
        #Profile prediction head   
        self.profile_conv = nn.ConvTranspose1d(self.nb_filters, 1, kernel_size=self.profile_kernel)
        self.flatten = nn.Flatten()

        #Total count prediction head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.nb_filters,1)

            
    def forward(self,x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = F.relu(self.convlayers[0](x))

        for layer in self.convlayers[1:]:
            
            conv_x = F.relu(layer(x))

            #Crop output previous layer to size of current 
            x_len = x.size(2); conv_x_len = conv_x.size(2)
            cropsize = (x_len - conv_x_len) // 2
            x = x[:, :, cropsize:-cropsize] 

            #Skipped connection
            x = conv_x + x    

        #Profile head
        #-----------------------------------------------
        profile = self.profile_conv(x)
        
        cropsize = int((profile.size(2)/2) - (self.out_pred_len/2))
        profile = profile[:,:, cropsize:-cropsize]
        
        profile = self.flatten(profile)

        #Total count head
        #-----------------------------------------------
        count = self.global_pool(x)  
        count = count.squeeze()
        count = self.linear(count)

        return x, profile, count

#Custom losses functions
class ATACloss(nn.Module):
    def __init__(self, weight_MSE):
        super().__init__()
        self.weight_MSE = weight_MSE
        self.MSE = nn.MSELoss()

    def forward(self, true_counts, logits, tot_pred):
        counts_per_example = torch.sum(true_counts, dim=1)
        dist = [Multinomial(total_count=x.item(), logits=logits[i,:], validate_args=False) 
                    for i,x in enumerate(torch.round(counts_per_example).type(torch.int32))]

        true_counts = true_counts[:,:1000]
        MNLLL = torch.Tensor([x.log_prob(true_counts[i,:]) for i,x in enumerate(dist)])
        MNLLL = ((-torch.sum(MNLLL))/float(true_counts.shape[0]))

        MSE = self.MSE(counts_per_example, tot_pred.squeeze())

        return self.weight_MSE*MSE + MNLLL
