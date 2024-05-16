#Fetch continuous ATAC tracks for each pseudo-bulk and peak regions
#--------------------------------------------

import pickle
import numpy as np
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_ATAC_peaks

TIME_POINT = ["D8", "D12", "D20", "D22-15"]

with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

peaks['middle'] = np.round((peaks.end - peaks.start)/2 + peaks.start).astype('uint32')

#Per cell type + dataset createa dataframe with continous track for each peaks
for d in TIME_POINT:

    total_reads = pd.read_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index_col=[0])

    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*.bw')
    for f in bw_files:
        bw = pyBigWig.open(f)

        tot = int(total_reads.loc[f.removeprefix('../results/bam_cell_type/').removesuffix('.bw')].values[0][2:-3])
        ATAC = peaks.apply(lambda x: get_continuous_ATAC_peaks(bw, x, tot, seq_len=1024), axis=1)

        if not os.path.exists('../results/ATAC/' + d):
            os.makedirs('../results/ATAC/' + d)
        
        with open(("../results/ATAC/" + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC 

#Merge all datasets into one adding columns: time + cell type 
pkl_files = glob.glob('../results/ATAC/*/*.pkl')

for f in pkl_files:
    with open(f, 'rb') as file:
        tmp = pd.DataFrame(pickle.load(file))

    tmp['time'] = [f.split('/')[3]] * tmp.shape[0]
    tmp.time = tmp.time.astype('category')
        
    tmp['cell_type'] = ([f.split('/')[4].removesuffix('.pkl')] * tmp.shape[0])
    tmp.cell_type = tmp.cell_type.astype('category')

    tmp['pseudo_bulk'] = tmp.time.astype('str') + tmp.cell_type.astype('str')

    tmp = tmp.drop(['time', 'cell_type'],axis=1)

    with open(f, 'wb') as file:
            pickle.dump(tmp, file)

    del tmp

ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[:13])
with open('../results/ATAC_peaks1.pkl', 'wb') as file:
            pickle.dump(ATAC, file)
del ATAC

ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[13:])
with open('../results/ATAC_peaks2.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

""" ATAC = pd.concat(pd.read_pickle(f) for f in ['../results/ATAC_peaks1.pkl', '../results/ATAC_peaks2.pkl'])
with open('../results/ATAC_peaks.pkl', 'wb') as file:
            pickle.dump(ATAC, file) """
