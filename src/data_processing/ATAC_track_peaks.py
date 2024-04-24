import pickle
import numpy as np
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_ATAC_peaks

NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

peaks['middle'] = np.round((peaks.end - peaks.start)/2 + peaks.start).astype('uint32')

#Per cell type + dataset createa dataframe with continous track for each peaks
for d in NAME_DATASET:

    total_reads = pd.read_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index_col=[0])

    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*.bw')
    for f in bw_files:
        bw = pyBigWig.open(f)

        tot = int(total_reads.loc[f.removeprefix('../results/bam_cell_type/').removesuffix('.bw')].values[0][2:-3])
        ATAC = peaks.apply(lambda x: get_continuous_ATAC_peaks(bw, x, tot), axis=1)

        if not os.path.exists('../results/ATAC/' + d):
            os.makedirs('../results/ATAC/' + d)
        
        with open(("../results/ATAC/" + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC

#Merge all datasets into one adding columns: time + cell type 
pkl_files = glob.glob('../results/ATAC/*/*.pkl')

with open(pkl_files[0], 'rb') as file:
    ATAC = pd.DataFrame(pickle.load(file))

ATAC['time'] = ([pkl_files[0].split('/')[3]] * ATAC.shape[0]).astype('category')
ATAC['cell_type'] = ([pkl_files[0].split('/')[4].removesuffix('.pkl')] * ATAC.shape[0]).astype('category')

for f in pkl_files[1:]:
    with open(f, 'rb') as file:
        tmp = pd.DataFrame(pickle.load(file))
    
    tmp['time'] = ([f.split('/')[3]] * tmp.shape[0]).astype('category')
    tmp['cell_type'] = ([f.split('/')[4].removesuffix('.pkl')] * tmp.shape[0]).astype('category')

    ATAC = pd.concat([ATAC, tmp])

with open('../results/ATAC_peaks.pkl', 'wb') as file:
            pickle.dump(ATAC, file)