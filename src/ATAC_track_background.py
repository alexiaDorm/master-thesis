import pickle
import numpy as np
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_ATAC_background

NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

with open('../results/match_GC.pkl', 'rb') as file:
    background = pickle.load(file)

for d in NAME_DATASET:

    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*.bw')
    for f in bw_files:
        bw = pyBigWig.open(f)
    
        ATAC = background.apply(lambda x: get_continuous_ATAC_background(bw, x), axis=1)

        if not os.path.exists('../results/background/' + d):
            os.makedirs('../results/background/' + d)
        
        with open(('../results/background/' + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC