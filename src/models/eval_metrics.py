import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.multinomial import Multinomial

from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

#Custom losses functions
class ATACloss_MNLLL(nn.Module):
    def __init__(self, weight_MSE=1):
        super().__init__()
        self.weight_MSE = weight_MSE
        self.NLL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss(reduction='mean')

    def forward(self, true_counts, logits, tot_pred):
                
        counts_per_example = torch.sum(true_counts, dim=1)

        true_counts_prob = true_counts/ counts_per_example.unsqueeze(-1)
        true_counts_prob[true_counts_prob != true_counts_prob] = 0 #set division to zero to 0 

        MNLLL = self.NLL(logits, true_counts_prob) * 1000
        MSE = self.MSE(torch.log(counts_per_example + 1), tot_pred.squeeze())

        loss = self.weight_MSE*MSE + MNLLL

        return loss, MNLLL, MSE

class ATACloss_KLD(nn.Module):
    def __init__(self, weight_MSE=1, weight_KLD=1):
        super().__init__()
        self.weight_KLD = weight_KLD
        self.weight_MSE = weight_MSE
        self.KLD = nn.KLDivLoss(reduction="none")
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, true_counts, logits, tot_pred, idx_skip):

        print(idx_skip)
                
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

#Use to normlaize the JSD
def jsd_min_max_bounds(profile):

    #Uniform distribution
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))
    
    #Tracks as prob 
    profile_prob = profile/profile.sum()

    max_jsd = jensenshannon(profile_prob.cpu().detach(), uniform_profile)

    return 0, max_jsd

def normalized_min_max(value, min, max):
    norm = (value - min) / (max - min)

    return norm 

#Compute the Jensen-Shannon divergence between observed and predicted profiles
def profile_metrics(tracks, profile_pred, idx_skip, pseudocount=0.001):
    
    #Convert logits to prob
    profile_prob = F.softmax(profile_pred, dim=1)

    jsd = []
    for idx,t in enumerate(tracks):

        if idx_skip[idx]:
        
            #Compute Jensen-Shannon divergence + normalize it
            t = t.cpu()
            curr_jsd = jensenshannon(t/(pseudocount+np.nansum(t)), profile_prob[idx,:].cpu().detach())
            min_jsd, max_jsd = jsd_min_max_bounds(t)

            curr_jsd = normalized_min_max(curr_jsd, min_jsd, max_jsd)

            jsd.append(curr_jsd)
        
        else:
            jsd.append(np.nan)

    return jsd

