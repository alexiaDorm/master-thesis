import pandas as pd
import numpy as np
import math

#Code to create synthetic sequences
#--------------------------------------------
def generate_motif(motif_file, rng):
    #Load the pw TF motif
    motif = pd.read_excel(motif_file, index_col=0, header=None)  
    motif = motif/motif.sum(axis=0)

    #Generate seq using the prob at each nuclotide

    df_selections = pd.DataFrame(
        data=rng.multinomial(n=1, pvals=(motif).transpose()),
        columns=motif.index).idxmax(axis=1)

    gen_motif = df_selections.str.cat()
        
    return gen_motif

#https://stackoverflow.com/questions/66622025/python-generate-random-dna-sequencing-with-known-gc-percent
def generate_seq(GC_content, motif, len_seq, pred_len):
    #Randomly generate a DNA sequence with given GC content and length
    n = round(GC_content * len_seq)
    GC_nucl = list(set('GC'))
    AT_nucl = list(set('AT'))

    seq = [np.random.choice(GC_nucl) for _ in range(n)]
    seq += [np.random.choice(AT_nucl) for _ in range(len_seq - n)]
    np.random.shuffle(seq)
    seq = ''.join(seq)

    #Insert randomly motif in sequence
    insert_idx = int(np.random.randint(low=(math.ceil(len_seq - pred_len)/2), 
                                       high = (math.ceil(len_seq - pred_len)/2) + pred_len, size=1))
    seq = seq[:insert_idx] + motif + seq[insert_idx:]

    return seq, insert_idx

def generate_seq_tn5(GC_content, len_seq, motif):
    #Randomly generate a DNA sequence with given GC content and length
    n = round(GC_content * len_seq)
    GC_nucl = list(set('GC'))
    AT_nucl = list(set('AT'))

    seq = [np.random.choice(GC_nucl) for _ in range(n)]
    seq += [np.random.choice(AT_nucl) for _ in range(len_seq - n)]
    np.random.shuffle(seq)
    seq = ''.join(seq)

    #Insert motif in middle sequence
    insert_idx = len_seq//2 - len(motif)//2
    seq = seq[len(motif)//2 :insert_idx] + motif + seq[insert_idx:-len(motif)//2]
    
    return seq