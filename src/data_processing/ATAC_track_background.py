import pickle
import numpy as np
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_ATAC_background

NAME_DATASET = ["D8", "D12", "D20", "D22-15"]

""" with open('../results/background_GC_matched.pkl', 'rb') as file:
    background = pickle.load(file)

for d in NAME_DATASET:

    total_reads = pd.read_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index_col=[0])
    
    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*.bw')
    for f in bw_files:
        bw = pyBigWig.open(f)

        tot = int(total_reads.loc[f.removeprefix('../results/bam_cell_type/').removesuffix('.bw')].values[0][2:-3])
        ATAC = background.apply(lambda x: get_continuous_ATAC_background(bw, x, tot), axis=1)

        if not os.path.exists('../results/background/' + d):
            os.makedirs('../results/background/' + d)
        
        with open(('../results/background/' + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC """

#Merge all datasets into one adding columns: time + cell type 
pkl_files = glob.glob('../results/background/*/*.pkl')

with open(pkl_files[0], 'rb') as file:
    ATAC = pd.DataFrame(pickle.load(file))

ATAC['time'] = [pkl_files[0].split('/')[3]] * ATAC.shape[0]
ATAC['cell_type'] = [pkl_files[0].split('/')[4].removesuffix('.pkl')] * ATAC.shape[0]

for f in pkl_files[1:]:
    with open(f, 'rb') as file:
        tmp = pd.DataFrame(pickle.load(file))
    
    tmp['time'] = [f.split('/')[3]] * tmp.shape[0]
    tmp['cell_type'] = [f.split('/')[4].removesuffix('.pkl')] * tmp.shape[0]

    ATAC = pd.concat([ATAC, tmp])

with open('../results/ATAC_background.pkl', 'wb') as file:
            pickle.dump(ATAC, file)