from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.visualization import viz_sequence

#from captum.attr import IntegratedGradients

import shap
from interpretation.overwrite_shap_explainer import DeepExplainer

import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from keras.models import load_model

from data_processing.utils_data_preprocessing import one_hot_encode


def compute_shap_score(model ,seq, back, idx_time):
    back = back.permute(0,2,1)
    seq = torch.from_numpy(seq).permute(1,0)[None,:,:]
    
    explainer = DeepExplainer(
        model, back, idx_time)
    raw_scores = explainer.shap_values(seq)
    
    return raw_scores

def compute_importance_score(model, path_sequence, idx_time, device):

    #Load the model and sequenece to predict
    model.to(device)
    seq = pd.Series(pd.read_pickle(path_sequence))
    
    #On-hot encode the sequences
    seq = seq.apply(lambda x: one_hot_encode(x))
    
    #Create shuffled sequences for background
    background = [dinuc_shuffle(s, num_shufs=20) for s in seq]

    #Compute importance score for each base of sequences
    shap_scores = [compute_shap_score(model,s,torch.from_numpy(background[i]), idx_time) for i,s in enumerate(seq)]

    #Reshape the sequeneces and scores
    seq = np.stack(seq.to_numpy())
    seq = torch.from_numpy(seq).permute(0,2,1)
    print(seq.shape)

    shap_scores = torch.from_numpy(np.squeeze(np.stack(shap_scores)))
    
    #Project the scores on the sequence
    proj_score = (shap_scores.sum(axis=1)[:,np.newaxis,:] * seq)

    return seq, shap_scores, proj_score

def compute_importance_score_c_type(model, path_sequence, device, c_type, all_c_type, idx_time):

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
    shap_scores = [compute_shap_score(model,s,torch.from_numpy(background[i]), idx_time) for i,s in enumerate(seq)]

    #Reshape the sequeneces and scores
    seq = torch.from_numpy(np.stack(seq)).permute(0,2,1)

    shap_scores = torch.from_numpy(np.squeeze(np.stack(shap_scores)))
    
    #Project the scores on the sequence
    proj_score = (shap_scores[:,:4,:].sum(axis=1)[:,np.newaxis,:] * seq)

    return seq, shap_scores, proj_score

#WITH BIAS 
#-------------------------------------------------
#Code from: https://zenodo.org/records/7121027#.ZCbw4uzMI8N

# One-hot encoding of a DNA sequence.
# Input: 
# (1) seq: a string of length L consisting of A/C/G/T
# Returns: 
# (1) onehot: L-by-4 encoded matrix
def onehot_encode(seq):
    mapping = pd.Series(index = ["A", "C", "G", "T"], data = [0, 1, 2, 3])
    bases = [base for base in seq]
    base_inds = mapping[bases]
    onehot = np.zeros((len(bases), 4))
    onehot[np.arange(len(bases)), base_inds] = 1
    return onehot

# Running one-hot encoding along the sequence of a genomic region
# Starting from the left, encode every consecutive sub-sequence with length = 2 * context_radius + 1
# with step size of 1 until we reach the other end of the sequence.
# Input: 
# (1) region_seq, a string of the DNA sequence in the region of interest. Must be longer than 2 * context_radius + 1
# (2) context_radius, radius of every sub-sequence
# Returns:
# (1) one-hot encoded sequence contexts for each position in the region
def region_onehot_encode(region_seq, context_radius = 50):
    
    # Calculate length of every local sub-sequence
    context_len = 2 * context_radius + 1
    
    # If region width is L, then region_seq should be a string of length L + 2 * context_len
    region_width = len(region_seq) - 2 * context_radius
    
    if "N" in region_seq:
        return np.zeros((region_width, 4))
    else:
        # First encode the whole region sequence 
        # This prevents repetitive computing for overlapping sub-sequences
        region_onehot = np.array(onehot_encode(region_seq))
        
        # Retrieve encoded sub-sequences by subsetting the larger encoded matrix
        region_onehot = np.array([region_onehot[i : (i + context_len), :] for i in range(region_width)])
        
        return region_onehot

def compute_tn5_bias(model, seq, len_pred=1024):
    
    #One-hot encoding of every 101bp sequence context along the region
    region_onehot = region_onehot_encode(seq)

    #Predict bias using pretrained model
    pred_bias = np.transpose(model.predict(region_onehot))[0]
    pred_bias = np.power(10, (pred_bias - 0.5) * 2) - 0.01
    
    #Crop prediction to be correct pred shape
    seq_len = pred_bias.shape[0]
    cropsize = (seq_len - len_pred) // 2
    pred_bias = pred_bias[cropsize:-cropsize] 

    return pred_bias[None,:]

def compute_shap_score_bias(model ,seq, back, tn5_bias, idx_time):
    back = back.permute(0,2,1)
    seq = torch.from_numpy(seq).permute(1,0)[None,:,:]

    explainer = DeepExplainer(
        model, [back, tn5_bias.tile((back.shape[0],1))], idx_time)
    raw_scores = explainer.shap_values([seq, tn5_bias])
    
    return raw_scores[0]

def compute_importance_score_bias(model, model_bias_path, path_sequence, device, c_type, all_c_type, idx_time):

    #Load the model and sequenece to predict
    model.to(device)
    seq = pd.Series(pd.read_pickle(path_sequence))

    #Compute tn5 bias for seq
    model_bias = load_model(model_bias_path)    
    tn5_bias = seq.apply(lambda x: compute_tn5_bias(model_bias, x))

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
    shap_scores = [compute_shap_score_bias(model,s,torch.from_numpy(background[i]), torch.from_numpy(tn5_bias[i]), idx_time) for i,s in enumerate(seq)]

    #Reshape the sequences and scores
    seq = torch.from_numpy(np.stack(seq)).permute(0,2,1)

    shap_scores = torch.from_numpy(np.squeeze(np.stack(shap_scores)))
    
    #Project the scores on the sequence
    proj_score = (shap_scores[:,:4,:].sum(axis=1)[:,np.newaxis,:] * seq)

    return seq, shap_scores, proj_score

#Integrated gradient 
#--------------------------------------------
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

#Visualization
#--------------------------------------------

def visualize_sequence_imp(proj_scores, idx_start, idx_end):
    
    for idx, dinuc_shuff_explanation in enumerate(proj_scores):
        #print("Scores for example", idx)

        viz_sequence.plot_weights(
            dinuc_shuff_explanation[:,idx_start:idx_end], subticks_frequency=20,
        )


