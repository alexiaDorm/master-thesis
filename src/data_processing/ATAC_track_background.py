#Fetch continuous ATAC tracks for each pseudo-bulk and background regions
#Sample 10% of the data
#--------------------------------------------
import pickle
import numpy as np
import torch
import glob
import pyBigWig
import os
import pandas as pd

from utils_data_preprocessing import get_continuous_wh_window

TIME_POINT = ["D8", "D12", "D20", "D22-15"]
all_cell_types = ['Neuronal', 'Somite', 'Immature', 'Mesenchymal', 'Myoblast', 'Myogenic', 'Neuroblast']

with open('../results/background_GC_matched.pkl', 'rb') as file:
    back = pickle.load(file)

#Sample about 10% of the regions (30'000)
back = back.sample(30000, random_state=42)

back['middle'] = np.round((back.end - back.start)/2 + back.start).astype('uint32')
nb_reg = back.shape[0]

all_ATAC, all_is_defined, idx_seq, chr_seq, c_type = [], [], [], [], []
for c in all_cell_types:

    bw_files = ['../results/bam_cell_type/' + t +'/' + c  + '_unstranded.bw' for t in TIME_POINT]
    ATAC_tracks, is_defined = [], []
    for f in bw_files:
        if os.path.isfile(f):
            
            #Get insertion count
            bw = pyBigWig.open(f)
            ATAC = nb_reg.apply(lambda x: get_continuous_wh_window(bw, x, 0, seq_len=1024), axis=1)
            ATAC = np.stack(ATAC)
            ATAC_tracks.append(ATAC)

            #The tracks are defined for cell type + time point
            is_defined.append([True]*nb_reg)

        else: 

            #If cell type not defined for time point, fill with zero tracks
            ATAC =  np.zeros((nb_reg,1024))
            ATAC_tracks.append(ATAC)

            #The tracks are NOT defined for cell type + time point
            is_defined.append([False]*nb_reg)
    
    #Stack the ATAC tracks and is_defined -> shape:(#peaks, 1024, #time)
    all_ATAC.append(np.transpose(np.stack(ATAC_tracks), (1, 2, 0)))
    all_is_defined.append(np.transpose(np.stack(is_defined), (1, 0)))

    #Keep sequence idx and cell type
    idx_seq.append(np.arange(0,nb_reg))
    chr_seq.append(nb_reg.chr.values)
    c_type.append([c]*nb_reg)
    
all_ATAC = torch.from_numpy(np.concatenate(all_ATAC, axis=0))
all_is_defined = torch.from_numpy(np.concatenate(all_is_defined, axis=0))
idx_seq = torch.from_numpy(np.concatenate(idx_seq, axis=0))
chr_seq = np.concatenate(chr_seq, axis=0)
c_type = np.concatenate(c_type, axis=0)

with open('../results/ATAC_new_back.pkl', 'wb') as file:
    pickle.dump(all_ATAC, file)

with open('../results/is_defined_back.pkl', 'wb') as file:
    pickle.dump(all_is_defined, file)

with open('../results/idx_seq_back.pkl', 'wb') as file:
    pickle.dump(idx_seq, file)

with open('../results/chr_seq_back.pkl', 'wb') as file:
    pickle.dump(chr_seq, file)

with open('../results/c_type_track_back.pkl', 'wb') as file:
    pickle.dump(c_type, file)

print('ok :)')

#Old way
""" NAME_DATASET = ["D8", "D12", "D20", "D22-15"]

with open('../results/background_GC_matched.pkl', 'rb') as file:
    background = pickle.load(file)

background.index = background.chr + ":" + background.start.astype('str') + "-" + background.end.astype('str')

for d in NAME_DATASET:

    total_reads = pd.read_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index_col=[0])
    
    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*_unstranded.bw')
    for f in bw_files:
        bw = pyBigWig.open(f)

        tot = int(total_reads.loc[f.removeprefix('../results/bam_cell_type/').removesuffix('_unstranded.bw')].values[0][2:-3])
        ATAC = background.apply(lambda x: get_continuous_wh_window(bw, x, tot, seq_len=1024), axis=1)

        if not os.path.exists('../results/background/' + d):
            os.makedirs('../results/background/' + d)
        
        with open(('../results/background/' + f.removeprefix("../results/bam_cell_type/").removesuffix("_unstranded.bw") + ".pkl"), 'wb') as file:
            pickle.dump(ATAC, file)

        del ATAC

#Merge all datasets into one adding columns: time + cell type 
pkl_files = glob.glob('../results/background/*/*.pkl')

for f in pkl_files:
    with open(f, 'rb') as file:
        tmp = pd.DataFrame(pickle.load(file))

    tmp['time'] = [f.split('/')[3]] * tmp.shape[0]        
    tmp['cell_type'] = ([f.split('/')[4].removesuffix('.pkl')] * tmp.shape[0])

    tmp['pseudo_bulk'] = tmp.time.astype('str') + tmp.cell_type.astype('str')

    tmp = tmp.drop(['time', 'cell_type'], axis =1)

    with open(f, 'wb') as file:
        pickle.dump(tmp, file)

    del tmp

ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[:13])
with open('../results/ATAC_background1.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

del ATAC

ATAC = pd.concat(pd.read_pickle(f) for f in pkl_files[13:])
with open('../results/ATAC_background2.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

""" ATAC = pd.concat(pd.read_pickle(f) for f in ['../results/ATAC_background1.pkl', '../results/ATAC_background2.pkl'])
with open('../results/ATAC_background.pkl', 'wb') as file:
            pickle.dump(ATAC, file)
 """ """