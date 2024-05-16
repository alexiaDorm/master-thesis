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
    
    return np.squeeze(raw_scores)

def compute_importance_score(path_model, path_sequence, device):

    #Load the model and sequenece to predict
    model = torch.load(path_model)
    model.to(device)
    seq = pd.read_pickle(path_sequence).sequence.iloc[:5]

    #On-hot encode the sequences
    seq = seq.apply(lambda x: one_hot_encode(x))
    
    #Create shuffled sequences for background
    background = [dinuc_shuffle(s, num_shufs=20) for s in seq]

    #Compute importance score for each base of sequences
    shap_scores = [compute_shap_score(model,s,torch.from_numpy(background[i])) for i,s in enumerate(seq)]

    #Reshape the sequeneces an scores
    seq = np.stack(seq.to_numpy())
    seq = torch.reshape(torch.from_numpy(seq),(-1,4,seq.shape[1]))

    shap_scores = np.stack(shap_scores)

    print(seq.shape, shap_scores.shape)
    
    #Project the scores on the sequence
    proj_score = [s * shap_scores[i] for i,s in enumerate(seq)]
    
    return seq, shap_scores, proj_score

def visualize_sequence_imp(proj_scores,idx_start, idx_end):
    
    for idx, dinuc_shuff_explanation in enumerate(proj_scores):
        print("Scores for example", idx)

        viz_sequence.plot_weights(
            dinuc_shuff_explanation[:,idx_start:idx_end], subticks_frequency=20,
        )