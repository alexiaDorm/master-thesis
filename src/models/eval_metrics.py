import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.multinomial import Multinomial

from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

#Custom losses functions
class ATACloss_MNLLL(nn.Module):
    def __init__(self, weight_MSE=1, weight_NLL=1):
        super().__init__()
        self.weight_NLL = weight_NLL
        self.weight_MSE = weight_MSE
        self.NLL = nn.CrossEntropyLoss(reduction='none')
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, true_counts, logits, tot_pred, idx_skip):
                        
        counts_per_example = torch.sum(true_counts, dim=1)

        true_counts_prob = true_counts/ counts_per_example.unsqueeze(-1)
        true_counts_prob[true_counts_prob != true_counts_prob] = 0 #set division to zero to 0 

        MNLLL = self.NLL(logits, true_counts_prob) * 100
        MSE = self.MSE(torch.log(counts_per_example + 1), tot_pred.squeeze())

        #Skip idx where track was not defined for loss computation
        MNLLL = MNLLL[idx_skip].sum()
        MSE = MSE[idx_skip].sum()

        loss = self.weight_MSE*MSE + self.weight_NLL*MNLLL

        return loss, MNLLL, MSE

class ATACloss_KLD(nn.Module):
    def __init__(self, weight_MSE=1, weight_KLD=1):
        super().__init__()
        self.weight_KLD = weight_KLD
        self.weight_MSE = weight_MSE
        self.KLD = nn.KLDivLoss(reduction="none")
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, true_counts, logits, tot_pred, idx_skip):
                        
        counts_per_example = torch.sum(true_counts, dim=1)

        true_counts_prob = true_counts/ counts_per_example.unsqueeze(-1)
        true_counts_prob[true_counts_prob != true_counts_prob] = 0 #set division to zero to 0 

        KLD = self.KLD(nn.functional.log_softmax(logits, dim=1), true_counts_prob)
        MSE = self.MSE(torch.log(counts_per_example + 1), tot_pred.squeeze())

        #Skip idx where track was not defined for loss computation
        KLD = KLD[idx_skip,:].sum()
        MSE = MSE[idx_skip].sum()

        loss = self.weight_MSE*MSE + self.weight_KLD*KLD

        return loss, KLD, MSE

#Compute spearmann correlations between observed total counts and predictions
def counts_metrics(tracks, counts_pred, idx_skip):
    
    counts_per_seq = torch.sum(tracks, dim=1)[idx_skip].cpu()
    counts_pred = counts_pred.cpu().detach()[idx_skip]

    corr_tot = spearmanr(counts_pred, counts_per_seq)[0]

    return corr_tot    
    


#Compute the Jensen-Shannon divergence between observed and predicted profiles
def profile_metrics(tracks, profile_pred, idx_skip):

    counts_per_example = torch.sum(tracks[idx_skip,:], dim=1)
    true_counts_prob = tracks[idx_skip,:]/ counts_per_example.unsqueeze(-1)
    true_counts_prob[true_counts_prob != true_counts_prob] = 0 #set division to zero to 0 

    profile_pred = F.softmax(profile_pred[idx_skip,:], dim=1)

    M = 0.5*(true_counts_prob + profile_pred)

    jsd = 0.5* (F.kl_div(F.log_softmax(true_counts_prob, dim=1), M, reduction="batchmean") + F.kl_div(F.log_softmax(profile_pred, dim=1), M, reduction="batchmean"))

    return jsd

