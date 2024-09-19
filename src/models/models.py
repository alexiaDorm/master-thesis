#Pytorch class definition of models
#----------------------------------------

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
        profile = profile.squeeze()

        #Total count head
        #-----------------------------------------------
        count = self.global_pool(x)  
        count = count.squeeze()
        count = self.linear(count)

        return x, profile, count
    
    
class CATAC_w_bias(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, 
                 out_pred_len=1024, nb_pred=4, size_final_conv=3568):

        super().__init__()
        """ Main model with cell type token
        
        Parameters
        -----------
        nb_conv: int (default 8)
            number of convolutional layers
            
        nb_filters: int (default 64)
            number of filters in the convolutional layers

        first_kernel: int (default 21)
            size of the kernel in the first convolutional layer

        rest_kernel: int (default 3)
            size of the kernel in all convolutional layers except the first one

        out_pred_len: int (default 1024)
            number of bp for which ATAC signal is predicted
        
        nb_pred: int (default 4)
            number of ATAC tracks to predict

        size_final_conv: int (default 3568)
            size of the output of the convolution block

        Model Architecture 
        ------------------------

        - Input: 4096 bp DNA sequence + cell type token

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - 4 (number time point) x Head : 
            > Profile prediction head: a multinomial probability of Tn5 insertion counts at each position 
            in the input sequence
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count 
        
        The predicted (expected) count at a specific position is a multiplication of the predicted total 
        counts and the multinomial probability at that position.

        -------------------------
        
        """
        
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.out_pred_len = out_pred_len
        self.nb_pred = nb_pred
        self.size_final_conv = size_final_conv

        #Convolutional layers
        self.convlayers = nn.ModuleList()

        self.convlayers.append(nn.Sequential(nn.Conv1d(in_channels=11, out_channels=self.nb_filters,kernel_size=self.first_kernel),
            nn.ReLU()))
        
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Sequential(
                nn.Conv1d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.rest_kernel, dilation=2**i),
                nn.ReLU()
                ))
        
        #Profile prediction heads
        self.profile_global_pool = nn.AdaptiveAvgPool1d(1)

        self.profile_heads = nn.ModuleList() 
        for i in range(self.nb_pred):
            self.profile_heads.append(nn.Linear(self.size_final_conv+self.out_pred_len, self.out_pred_len))

        #Total count prediction heads
        self.count_global_pool = nn.AdaptiveAvgPool1d(1)

        self.count_heads = nn.ModuleList()
        for i in range(self.nb_pred):
            self.count_heads.append(nn.Linear(self.nb_filters+1,1))
        
    def forward(self, x, tn5_bias):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

            #Crop output previous layer to size of current 
            cropsize = (x.size(2) - conv_x.size(2)) // 2

            #Skipped connection
            x = conv_x + x[:, :, cropsize:-cropsize]   
    
        #Profile head
        #-----------------------------------------------
        #cropsize = (4096 - self.out_pred_len) //2
        #tn5_bias = tn5_bias[:,cropsize:-cropsize]

        pred_profiles = torch.empty((self.nb_pred, x.size(0), self.out_pred_len), device = x.device, dtype=torch.float32)
        for i, p in enumerate(self.profile_heads):
                
            #Apply global average poolling
            profile = self.profile_global_pool(x.permute(0,2,1)) 
            profile = profile.squeeze()

            #Concatenate total tn5 bias
            profile = torch.cat((profile, tn5_bias), 1)

            #Apply linear layer
            pred_profiles[i] = p(profile)

            #Concatenate the padded tn5 bias, Apply final convolution
            #profile = p(torch.cat((x, tn5_bias),1))

            #Crop and flatten the representation
            #cropsize = int((profile.size(2)/2) - (self.out_pred_len/2))
            #pred_profiles[i] = profile[:,:, cropsize:-cropsize].squeeze()
        
        #Total count head
        #-----------------------------------------------
        cropsize = (tn5_bias.size(1) - self.out_pred_len) // 2
        total_bias = tn5_bias[:,cropsize:-cropsize].squeeze().sum(dim=1).unsqueeze(1) 

        pred_counts = torch.empty((self.nb_pred, x.size(0)), device = x.device, dtype=torch.float32)
        for i, c in enumerate(self.count_heads):
            
            #Apply global average poolling
            count = self.count_global_pool(x).squeeze()

            #Concatenate total tn5 bias, Apply linear layer
            pred_counts[i] = c(torch.cat((count, total_bias), 1)).squeeze()

        return x, pred_profiles.permute(1,2,0), pred_counts.permute(1,0)

    def first_filter_output(self, x):
        return self.convlayers[0](x)

class CATAC_wo_bias(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=64, first_kernel=21, rest_kernel=3, out_pred_len=1024, 
                 nb_pred=4, size_final_conv=3568):

        super().__init__()
        """ Main model with cell type token
        
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

        out_pred_len: int (default 1024)
            number of bp for which ATAC signal is predicted
        
        nb_pred: int (default 4)
            number of ATAC tracks to predict

        size_final_conv: int (default 3568)
            size of the output of the convolution block

        Model Architecture 
        ------------------------

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - 4 (number time point) x Head : 
            > Profile prediction head: a multinomial probability of Tn5 insertion counts at each position 
            in the input sequence
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count
        
        The predicted (expected) count at a specific position is a multiplication of the predicted total 
        counts and the multinomial probability at that position.

        -------------------------
        
        """
        
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.out_pred_len = out_pred_len
        self.nb_pred = nb_pred
        self.size_final_conv = size_final_conv

        #Convolutional layers
        self.convlayers = nn.ModuleList()

        self.convlayers.append(nn.Sequential(nn.Conv1d(in_channels=11, out_channels=self.nb_filters,kernel_size=self.first_kernel),
            nn.ReLU()))
        
        for i in range (1,self.nb_conv):
            self.convlayers.append(nn.Sequential(
                nn.Conv1d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.rest_kernel, dilation=2**i),
                nn.ReLU()
                ))
        
        #Profile prediction heads
        self.profile_global_pool = nn.AdaptiveAvgPool1d(1)

        self.profile_heads = nn.ModuleList() 
        for i in range(self.nb_pred):
            self.profile_heads.append(nn.Linear(self.size_final_conv, self.out_pred_len))
            #self.profile_heads.append(nn.Conv1d(self.nb_filters + 1, 1, kernel_size=75))


        #Total count prediction heads
        self.count_global_pool = nn.AdaptiveAvgPool1d(1)

        self.count_heads = nn.ModuleList()
        for i in range(self.nb_pred):
            self.count_heads.append(nn.Linear(self.nb_filters,1))
        
    def forward(self, x):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for layer in self.convlayers[1:]:
            
            conv_x = layer(x)

            #Crop output previous layer to size of current 
            cropsize = (x.size(2) - conv_x.size(2)) // 2

            #Skipped connection
            x = conv_x + x[:, :, cropsize:-cropsize]   
    
        #Profile head
        #-----------------------------------------------

        pred_profiles = torch.empty((self.nb_pred, x.size(0), self.out_pred_len), device = x.device, dtype=torch.float32)
        for i, p in enumerate(self.profile_heads):
                
            #Apply global average poolling
            profile = self.profile_global_pool(x.permute(0,2,1)) 
            profile = profile.squeeze()

            #Apply linear layer
            pred_profiles[i] = p(profile)
        
        #Total count head
        #-----------------------------------------------
        pred_counts = torch.empty((self.nb_pred, x.size(0)), device = x.device, dtype=torch.float32)
        for i, c in enumerate(self.count_heads):
            
            #Apply global average poolling
            count = self.count_global_pool(x).squeeze()

            #Concatenate total tn5 bias, Apply linear layer
            pred_counts[i] = c(count).squeeze()

        return x, pred_profiles.permute(1,2,0), pred_counts.permute(1,0)
    
    def first_filter_output(self, x):
        return self.convlayers[0](x)
    

class CATAC_w_bias_increase_filter(nn.Module):
    def __init__(self, nb_conv=8, nb_filters=32, first_kernel=21, rest_kernel=3, out_pred_len=1024, 
                 nb_pred=4, size_final_conv=3568, mult_filter=2, max_filters=256):

        super().__init__()
        """ Main model with cell type token and increasing number of filters 
        
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

        out_pred_len: int (default 1024)
            number of bp for which ATAC signal is predicted
        
        nb_pred: int (default 4)
            number of ATAC tracks to predict

        size_final_conv: int (default 3568)
            size of the output of the convolution block
        
        mult_filter: int (default 2)
            the number of filters throught the layers is multipled by this factor until it reaches the maximum number of filters (eg. if first layer has 64 filters, then second has 128 filters and so forth)
        
        max_filters: int (default 256)
            number maximum of filters per layer, the number of filters increases until reaching this value

        Model Architecture 
        ------------------------

        - Body: sequence of convolutional layers with residual skip connections, dilated convolutions, 
        and  ReLU activation functions

        - 4 (number time point) x Head : 
            > Profile prediction head: a multinomial probability of Tn5 insertion counts at each position 
            in the input sequence
            > Total count prediction: the total Tn5 insertion counts over the input region, global average
            poooling and linear layer predicting the total count
        
        The predicted (expected) count at a specific position is a multiplication of the predicted total 
        counts and the multinomial probability at that position.

        -------------------------
        
        """
        
        #Define parameters
        self.nb_conv = nb_conv
        self.nb_filters = nb_filters
        self.first_kernel = first_kernel
        self.rest_kernel = rest_kernel
        self.out_pred_len = out_pred_len
        self.nb_pred = nb_pred
        self.size_final_conv = size_final_conv
        self.mult_filter = mult_filter
        self.max_filters = max_filters

        #Convolutional layers
        self.convlayers = nn.ModuleList()
        self.change_size = nn.ModuleList()

        self.convlayers.append(nn.Sequential(nn.Conv1d(in_channels=11, out_channels=self.nb_filters,kernel_size=self.first_kernel),
            nn.ReLU()))
        
        for i in range (1,self.nb_conv):
            if self.nb_filters < self.max_filters:

                self.convlayers.append(nn.Sequential(
                    nn.Conv1d(in_channels=self.nb_filters, out_channels=self.nb_filters*self.mult_filter, kernel_size=self.rest_kernel, dilation=2**i),
                    nn.ReLU()
                    ))
                
                self.change_size.append(nn.Conv1d(in_channels=self.nb_filters, out_channels=self.nb_filters*self.mult_filter, kernel_size=1))
                
                self.nb_filters = self.nb_filters*self.mult_filter

            else:
                self.convlayers.append(nn.Sequential(
                    nn.Conv1d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=self.rest_kernel, dilation=2**i),
                    nn.ReLU()
                    ))
                self.change_size.append(None)

        
        #Profile prediction heads
        self.profile_global_pool = nn.AdaptiveAvgPool1d(1)

        self.profile_heads = nn.ModuleList() 
        for i in range(self.nb_pred):
            self.profile_heads.append(nn.Linear(self.size_final_conv+self.out_pred_len, self.out_pred_len))
            #self.profile_heads.append(nn.Conv1d(self.nb_filters + 1, 1, kernel_size=75))

        #Total count prediction heads
        self.count_global_pool = nn.AdaptiveAvgPool1d(1)

        self.count_heads = nn.ModuleList()
        for i in range(self.nb_pred):
            self.count_heads.append(nn.Linear(self.nb_filters+1,1))
        
    def forward(self, x, tn5_bias):
        
        #Residual + Dilated convolution layers
        #-----------------------------------------------
        x = self.convlayers[0](x)

        for z, layer in enumerate(self.convlayers[1:]):
            
            conv_x = layer(x)

            #Crop output previous layer to size of current 
            cropsize = (x.size(2) - conv_x.size(2)) // 2

            #Skipped connection
            if conv_x.size(1) != x.size(1):
                x = self.change_size[z](x)
                
            x = conv_x + x[:, :, cropsize:-cropsize]   
    
        #Profile head
        #-----------------------------------------------

        pred_profiles = torch.empty((self.nb_pred, x.size(0), self.out_pred_len), device = x.device, dtype=torch.float32)
        for i, p in enumerate(self.profile_heads):
                
            #Apply global average poolling
            profile = self.profile_global_pool(x.permute(0,2,1)) 
            profile = profile.squeeze()

            #Concatenate total tn5 bias
            profile = torch.cat((profile, tn5_bias), 1)

            #Apply linear layer
            pred_profiles[i] = p(profile)
        
        #Total count head
        #-----------------------------------------------
        cropsize = (tn5_bias.size(1) - self.out_pred_len) // 2
        total_bias = tn5_bias[:,cropsize:-cropsize].squeeze().sum(dim=1).unsqueeze(1) 

        pred_counts = torch.empty((self.nb_pred, x.size(0)), device = x.device, dtype=torch.float32)
        for i, c in enumerate(self.count_heads):
            
            #Apply global average poolling
            count = self.count_global_pool(x).squeeze()

            #Concatenate total tn5 bias, Apply linear layer
            pred_counts[i] = c(torch.cat((count, total_bias), 1)).squeeze()

        return x, pred_profiles.permute(1,2,0), pred_counts.permute(1,0)

    def first_filter_output(self, x):
        return self.convlayers[0](x)