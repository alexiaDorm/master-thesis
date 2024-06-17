from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence

import shap

import torch
import pandas as pd
import numpy as np

from data_processing.utils_data_preprocessing import one_hot_encode


def compute_shap_score(model ,seq, back):
    
    back = torch.reshape(back, (-1,4,len(seq)))
    seq = torch.reshape(torch.from_numpy(seq), (-1,4,len(seq)))
    explainer = shap.DeepExplainer(
        model, back)
    raw_scores = explainer.shap_values(seq)
    
    return raw_scores

def compute_importance_score(model, path_sequence, device):

    #Load the model and sequenece to predict
    model.to(device)
    seq = pd.Series(pd.read_pickle(path_sequence))
    
    #On-hot encode the sequences
    seq = seq.apply(lambda x: one_hot_encode(x))
    
    #Create shuffled sequences for background
    background = [dinuc_shuffle(s, num_shufs=20) for s in seq]

    #Compute importance score for each base of sequences
    shap_scores = [compute_shap_score(model,s,torch.from_numpy(background[i])) for i,s in enumerate(seq)]

    #Reshape the sequeneces and scores
    seq = np.stack(seq.to_numpy())
    seq = torch.from_numpy(seq).permute(0,2,1)
    print(seq.shape)

    shap_scores = torch.from_numpy(np.squeeze(np.stack(shap_scores)))
    
    #Project the scores on the sequence
    proj_score = (shap_scores.sum(axis=1)[:,np.newaxis,:] * seq)

    return seq, shap_scores, proj_score

def visualize_sequence_imp(proj_scores,idx_start, idx_end):
    
    for idx, dinuc_shuff_explanation in enumerate(proj_scores):
        print("Scores for example", idx)

        viz_sequence.plot_weights(
            dinuc_shuff_explanation[:,idx_start:idx_end], subticks_frequency=20,
        )