#Utils functions for data processing of ATAC data (Fetch sequence, encode sequence, generation ATAC tracks, GC content)
#--------------------------------------------

import pandas as pd
import numpy as np

#import anndata
import pyfaidx
import pyBigWig

def fetch_sequence(peaks, path_genome, len_seq):
    """
    Fetch the sequence on the reference genome at given location, the sequences are centered in the middle of the regions

    peaks: pd.DataFrame 
        dataframe with start and end of genomic regions as columns
    path_genome: str
        path to the genome used to fetch sequence
    len_seq: int 
        len of the returned sequences
    
    return:  modified peaks dataframe with added middle and sequence columns
    """ 
    genome = pyfaidx.Fasta(path_genome)

    #Fetch sequence on reference genome using location of peaks
    peaks['middle_peak'] = round((peaks.end - peaks.start)/2 + peaks.start).astype('uint32')
    sequences = peaks.apply(lambda x: 
                            (genome[('chr' + x['chr'])][(x['middle_peak']-int(len_seq/2)):(x['middle_peak']+int(len_seq/2))]).seq, axis=1)
    sequences = sequences.str.upper()

    return sequences

def one_hot_encode(seq):
    """
    One hot encode a DNA sequence

    seq: str
        sequence to be encoded
    
    return:  one-hot encoded sequences as array (seq_len x 4)
    """ 
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4, dtype=np.float32)[seq2]

def encode_sequence(sequences):
    """
    One hot encode all provided sequences

    sequences: str
        sequences to be encoded
    
    return:  list of one-hot encoded sequences as array (seq_len x 4)
    """ 
    
    return [one_hot_encode(seq) for seq in sequences]

def get_continuous_wh_window(bw, seq_loc, seq_len=1024):
    """
    Get continuous ATAC track of given pseudobulk data at provided peak genome location.

    bw: object
        bigWig file of a pseudo-bulk ATAC tracks
    seq: pd.Serie
        pd.Serie of genomic region with chr, middle columns
    seq_len: int (1024)
        length of of the ATAC tracks
    
    return:  the ATAC track over the provided genomic region
    """ 
    middle = int(seq_loc.start + (seq_loc.end - seq_loc.start)/2)
    bp_around = int(seq_len/2)
    
    ATAC = bw.values(str(seq_loc.chr), middle - bp_around, middle + bp_around)
    ATAC = np.nan_to_num(ATAC)

    return ATAC

def compute_GC_content(seq):
    """
    Compute the GC content (%GC nucleotides) of given sequence

    seq: str
        sequence to compute the GC content
    
    return:  the GC content of the sequence
    """ 

    return [sum(x.count(n) for n in ("G", "C"))/len(x) for x in seq]

