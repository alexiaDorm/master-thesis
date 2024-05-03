import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.multinomial import Multinomial

from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

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

        true_counts = true_counts[:,:-1]
        MNLLL = torch.Tensor([x.log_prob(true_counts[i,:]) for i,x in enumerate(dist)])
        MNLLL = ((-torch.sum(MNLLL))/float(true_counts.shape[0]))

        MSE = self.MSE(counts_per_example, tot_pred.squeeze())

        return self.weight_MSE*MSE + MNLLL

#Compute spearmann correlations between observed total counts and predictions
def counts_metrics(tracks, counts_pred):
    
    counts_per_seq = torch.sum(tracks, dim=1)
    corr_tot = spearmanr(counts_pred.cpu().detach(), counts_per_seq.cpu().detach())[0]

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
def profile_metrics(tracks, profile_pred, pseudocount=0.001):
    
    #Convert logits to prob
    profile_prob = F.softmax(profile_pred, dim=1)

    jsd = []
    for idx,t in enumerate(tracks):
        
        #Compute Jensen-Shannon divergence + normalize it
        t = t[:-1]
        curr_jsd = jensenshannon(t/(pseudocount+np.nansum(t)), profile_prob[idx,:].cpu().detach())
        min_jsd, max_jsd = jsd_min_max_bounds(t)

        curr_jsd = normalized_min_max(curr_jsd, min_jsd, max_jsd)

        jsd.append(curr_jsd)

    return jsd

