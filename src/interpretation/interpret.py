from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence
from captum.attr import IntegratedGradients

import shap

import torch
import pandas as pd
import numpy as np

from data_processing.utils_data_preprocessing import one_hot_encode


def compute_shap_score(model ,seq, back):
    back = back.permute(0,2,1)
    seq = torch.from_numpy(seq).permute(1,0)[None,:,:]
    
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

def compute_importance_score_c_type(model, path_sequence, device, c_type, all_c_type):

    #Load the model and sequenece to predict
    model.to(device)
    seq = pd.Series(pd.read_pickle(path_sequence))
    
    #On-hot encode the sequences
    seq = seq.apply(lambda x: one_hot_encode(x))
    
    #Create shuffled sequences for background
    background = [dinuc_shuffle(s, num_shufs=20) for s in seq]

    #Add cell type encoding
    mapping = dict(zip(all_c_type, range(len(all_c_type))))    
    c_type = mapping[c_type]
    c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])

    #Repeat and reshape
    c_type = c_type.tile((seq[0].shape[0],1))
    seq = [np.concatenate((s,c_type), axis=1) for s in seq]

    c_type = c_type.tile((background[0].shape[0],1,1))
    background = [np.concatenate((b,c_type), axis=2) for b in background]

    #Compute importance score for each base of sequences
    shap_scores = [compute_shap_score(model,s,torch.from_numpy(background[i])) for i,s in enumerate(seq)]

    #Reshape the sequeneces and scores
    seq = torch.from_numpy(np.stack(seq)).permute(0,2,1)

    shap_scores = torch.from_numpy(np.squeeze(np.stack(shap_scores)))
    
    #Project the scores on the sequence
    proj_score = (shap_scores[:,:4,:].sum(axis=1)[:,np.newaxis,:] * seq)

    return seq, shap_scores, proj_score

def compute_integrated_gradient(model, path_sequence, device, c_type, all_c_type):
    
    #Load the model to device and sequences to predict
    model.to(device)
    seq = pd.Series(pd.read_pickle(path_sequence))
    
    #On-hot encode the sequences
    seq = seq.apply(lambda x: one_hot_encode(x))

    #Add cell type encoding
    mapping = dict(zip(all_c_type, range(len(all_c_type))))    
    c_type = mapping[c_type]
    c_type = torch.from_numpy(np.eye(len(all_c_type), dtype=np.float32)[c_type])
    c_type = c_type.tile((seq[0].shape[0],1))
    
    seq = [np.concatenate((s,c_type), axis=1) for s in seq]

    #Compute integrated gradient 
    out = [model(torch.tensor(s)[None,:,:].permute(0,2,1))[2][0] for s in seq]
    integrated_gradients = IntegratedGradients(model)
    attributions = [integrated_gradients.attribute(torch.tensor(s)[None,:,:].permute(0,2,1)) for i,s in enumerate(seq)]

    #Reshape the sequeneces and scores
    seq = torch.from_numpy(np.stack(seq)).permute(0,2,1)

    attributions = torch.from_numpy(np.squeeze(np.stack(attributions)))
    
    #Project the scores on the sequence
    proj_att = (attributions[:,:4,:].sum(axis=1)[:,np.newaxis,:] * seq)

    #Correct integrated gradient
    mean_att = attributions[:,:4,:].mean(axis=1)    
    mean_att = mean_att.tile((4,1,1)).permute(1,0,2)
    attributions = attributions[:,:4,:] - mean_att

    return seq, attributions, proj_att


def visualize_sequence_imp(proj_scores, idx_start, idx_end):
    
    for idx, dinuc_shuff_explanation in enumerate(proj_scores):
        print("Scores for example", idx)

        viz_sequence.plot_weights(
            dinuc_shuff_explanation[:,idx_start:idx_end], subticks_frequency=20,
        )