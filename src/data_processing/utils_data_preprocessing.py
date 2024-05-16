import pandas as pd
import numpy as np
import gzip

#import anndata
import pyfaidx
import pyBigWig

import torch
from torch.utils.data import Dataset, DataLoader

NAME_DATASET = ['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

#Fetch the sequence on the reference genome at ATAC peaks
def fetch_sequence(peaks, path_genome, len_seq = 2114):
    
    genome = pyfaidx.Fasta(path_genome)

    #Fetch sequence on reference genome using location of peaks
    peaks['middle_peak'] = round((peaks.end - peaks.start)/2 + peaks.start).astype('uint32')
    sequences = peaks.apply(lambda x: 
                            (genome[('chr' + x['chr'])][(x['middle_peak']-int(len_seq/2)):(x['middle_peak']+int(len_seq/2))]).seq, axis=1)
    sequences = sequences.str.upper()

    return sequences

def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4, dtype=np.float32)[seq2]

def encode_sequence(sequences):
    
    #One hot encode the the sequence
    return [one_hot_encode(seq) for seq in sequences]

def get_continuous_ATAC_peaks(bw, seq_loc, total_reads, norm_mult = 100000, seq_len=1000, window_size=200):
    bp_around = int(window_size/2 + seq_len/2)
    val = bw.values(seq_loc.chr, seq_loc.middle - bp_around, 
                    seq_loc.middle + bp_around)
    ATAC = [sum(val[i:(i+window_size+1)])/total_reads * norm_mult for i in range(0,seq_len+1)]

    return ATAC

def get_continuous_ATAC_background(bw, seq_loc, total_reads, norm_mult = 100000, seq_len=1000, window_size=200):
    middle = int(seq_loc.start + (seq_loc.end - seq_loc.start)/2)
    bp_around = int(window_size/2 + seq_len/2)

    val = bw.values(seq_loc.chr, middle - bp_around, 
                    middle + bp_around)
    ATAC = [sum(val[i:(i+window_size+1)])/total_reads * norm_mult for i in range(0,seq_len+1)]

    return ATAC

def compute_GC_content(seq):

    return [sum(x.count(n) for n in ("G", "C"))/len(x) for x in seq]

