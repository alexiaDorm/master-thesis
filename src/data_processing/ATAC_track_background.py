import pickle
import numpy as np
import glob
import os
import pandas as pd
import pyBigWig


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
        ATAC = background.apply(lambda x: get_continuous_ATAC_background(bw, x, tot, seq_len=1024), axis=1)

        if not os.path.exists('../results/background/' + d):
            os.makedirs('../results/background/' + d)
        
        with open(('../results/background/' + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC """

#Merge all datasets into one adding columns: time + cell type 
pkl_files = glob.glob('../results/background/*/*.pkl')

for f in pkl_files:
    with open(f, 'rb') as file:
        tmp = pd.DataFrame(pickle.load(file))

    tmp['time'] = [f.split('/')[3]] * tmp.shape[0]
    tmp.time = tmp.time.astype('category')
        
    tmp['cell_type'] = ([f.split('/')[4].removesuffix('.pkl')] * tmp.shape[0])
    tmp.cell_type = tmp.cell_type.astype('category')

    tmp['pseudo_bulk'] = tmp.time.astype('str') + tmp.cell_type.astype('str')

    tmp = tmp.drop(['time', 'cell_type'], axis =1)

    with open(f, 'wb') as file:
        pickle.dump(tmp, file)

    del tmp 

""" ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[:13])
with open('../results/ATAC_background1.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

del ATAC

ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[13:])
with open('../results/ATAC_background2.pkl', 'wb') as file:
            pickle.dump(ATAC, file) """

""" ATAC = pd.concat(pd.read_pickle(f) for f in ['../results/ATAC_background1.pkl', '../results/ATAC_background2.pkl'])
with open('../results/ATAC_background.pkl', 'wb') as file:
            pickle.dump(ATAC, file)
 """