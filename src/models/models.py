import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import pandas as pd

class BPNet(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, profile_kernel_size=75, out_pred_len=1024):
        super().__init__()
        """ BPNet architechture as in paper 
        
        Parameters
        -----------
        nb_conv: int (default 8)
            number of convolutional layers

        nb_filters: int (default 64)
            number of filters in the convolusqueuetional layers

        first_kernel: int (default 25)
            size of the kernel in the first convolutional layer

        rest_kernel: int (default 3)
            size of the kernel in all convolutional layers except the first one

        profile_kernel_size: int (default 75)
            size of the kernel in the profile convolution

        out_pred_len: int (default 1024)
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

        self.convlayers.append(nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.nb_filters, kernel_size=self.first_kernel),
            nn.ReLU()
            ))
        
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Sequential(
                nn.Conv1d(in_channels=self.nb_filters,out_channels=self.nb_filters,kernel_size=self.rest_kernel,dilation=2**i),
                nn.ReLU()
                ))
        
        #Profile prediction head   
        self.profile_conv = nn.Conv1d(self.nb_filters, 1, kernel_size=self.profile_kernel)

        #Total count prediction head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.nb_filters,1)

            
    def forward(self,x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

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
        profile = profile.reshape(-1, self.out_pred_len)

        #Total count head
        #-----------------------------------------------
        count = self.global_pool(x)  
        count = count.squeeze()
        count = self.linear(count)

        return x, profile, count
    
    def forward_t(self,x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

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
        
        profile = profile.reshape(-1, self.out_pred_len)

        #Total count head
        #-----------------------------------------------
        count = self.global_pool(x)  
        count = count.squeeze()
        count = self.linear(count)


        return torch.reshape(count, (-1,1))

class TotCountOnly(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, out_pred_len=1024):
        super().__init__()
        """ BPNet architechture as in paper but only predict the total counts
        
        Parameters
        -----------
        nb_conv: int (default 8)
            number of convolutional layers

        nb_filters: int (default 64)
            number of filters in the convolusqueuetional layers

        first_kernel: int (default 25)
            size of the kernel in the first convolutional layer

        rest_kernel: int (default 3)
            size of the kernel in all convolutional layers except the first one

        out_pred_len: int (default 1024)
            number of bp for which ATAC signal is predicted

        Model Architecture 
        ------------------------

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - Head: 
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count per strand

        -------------------------
        
        Reference: Avsec, Ž., Weilert, M., Shrikumar, A. et al. Base-resolution models of transcription-factor binding 
        reveal soft motif syntax. Nat Genet 53, 354–366 (2021). https://doi.org/10.1038/s41588-021-00782-6

        
        """
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.out_pred_len = out_pred_len

        #Convolutional layers
        self.convlayers = nn.ModuleList()

        self.convlayers.append(nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=self.nb_filters, kernel_size=self.first_kernel),
            nn.ReLU()
            ))
        
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Sequential(
                nn.Conv1d(in_channels=self.nb_filters,out_channels=self.nb_filters,kernel_size=self.rest_kernel,dilation=2**i),
                nn.ReLU()
                ))
        
        
        #Total count prediction head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.nb_filters,1)

            
    def forward(self,x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

            #Crop output previous layer to size of current 
            x_len = x.size(2); conv_x_len = conv_x.size(2)
            cropsize = (x_len - conv_x_len) // 2
            x = x[:, :, cropsize:-cropsize] 

            #Skipped connection
            x = conv_x + x    

        #Total count head
        #-----------------------------------------------
        count = self.global_pool(x)  
        count = count.squeeze()
        count = self.linear(count)

        return x, count
    
    
class CATAC(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, profile_kernel_size=75, out_pred_len=1024, nb_pred=1, nb_cell_type_CN = 0):

        super().__init__()
        """ Model taking genomic sequences and predicting cell type specific ATAC track
        
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

        out_pred_len: int (default 1024)
            number of bp for which ATAC signal is predicted
        
        nb_pred: int (default 1)
            number of ATAC tracks to predict

        nb_cell_type_CN: int (default 0)
            number of cell type specific convolutional layers

        Model Architecture 
        ------------------------

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - Cell-specific conv layers :
            > 

        - # pseudo_bulk x Head : 
            > Profile prediction head: a multinomial probability of Tn5 insertion counts at each position 
            in the input sequence, deconvolution layer
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count per strand
        
        The predicted (expected) count at a specific position is a multiplication of the predicted total 
        counts and the multinomial probability at that position.

        -------------------------
        
        """
        
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.profile_kernel = profile_kernel_size
        self.out_pred_len = out_pred_len
        self.nb_pred = nb_pred
        self.nb_cell_type_CN = nb_cell_type_CN

        #Convolutional layers
        self.convlayers = nn.ModuleList()

        self.convlayers.append(nn.Sequential(nn.Conv1d(in_channels=4, out_channels=self.nb_filters,kernel_size=self.first_kernel),
            nn.ReLU()))
        
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Sequential(
                nn.Conv1d(in_channels=self.nb_filters,out_channels=self.nb_filters,kernel_size=self.rest_kernel,dilation=2**i),
                nn.ReLU()
                ))
        
        #Pseudo bulk specific conv layers
        self.pb_convlayers = nn.ModuleList()

        for i in range(self.nb_pred):
            
            convs = nn.ModuleList()

            for j in range(self.nb_cell_type_CN):
                convs.append(nn.Sequential(nn.Conv1d(in_channels=self.nb_filters, 
                                         out_channels=self.nb_filters,
                                         kernel_size=self.rest_kernel),
                                         nn.ReLU()))

            self.pb_convlayers.append(convs)
        
        #Profile prediction heads
        self.profile_heads = nn.ModuleList() 
        for i in range(self.nb_pred):
            self.profile_heads.append(nn.Conv1d(self.nb_filters, 1, kernel_size=self.profile_kernel))
        
        #Total count prediction heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.count_heads = nn.ModuleList()
        for i in range(self.nb_pred):
            self.count_heads.append(nn.Linear(self.nb_filters,1))
        
    def forward(self,x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

            #Crop output previous layer to size of current 
            x_len = x.size(2); conv_x_len = conv_x.size(2)
            cropsize = (x_len - conv_x_len) // 2
            x = x[:, :, cropsize:-cropsize] 

            #Skipped connection
            x = conv_x + x   

        #Pseudo-bulk specific convolutional layers 
        #-----------------------------------------------
        tmp_x, pred_x = [x]*self.nb_pred, []
        for i in range(self.nb_pred):
            for layer in self.pb_convlayers[i]:
                conv_x = layer(tmp_x[i])

                #Crop output previous layer to size of current 
                x_len = tmp_x[i].size(2); conv_x_len = conv_x.size(2)
                cropsize = (x_len - conv_x_len) // 2
                tmp_x[i] = tmp_x[i][:, :, cropsize:-cropsize]

                #Skipped connection
                tmp_x[i] = conv_x + tmp_x[i] 

            pred_x.append(tmp_x[i])
        
        #Profile head
        #-----------------------------------------------
        pred_profiles = []
        for i, p in enumerate(self.profile_heads):
            
            #Apply conv layer
            profile = p(pred_x[i])

            #Crop and flatten the representation
            cropsize = int((profile.size(2)/2) - (self.out_pred_len/2))
            profile = profile[:,:, cropsize:-cropsize]
            profile = profile.reshape(-1, self.out_pred_len)

            pred_profiles.append(profile)
        
        #Total count head
        #-----------------------------------------------
        pred_counts = []
        for i, c in enumerate(self.count_heads):
            #Apply global average poolling
            count = self.global_pool(pred_x[i])  
            count = count.squeeze()
            
            #Aplly linear layer
            count = c(count)

            pred_counts.append(count)

        return pred_x, pred_profiles, pred_counts

