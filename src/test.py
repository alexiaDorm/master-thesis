""" import pandas as pd
import pickle

#Create test dataset
ATAC = pd.concat(pd.read_pickle(f) for f in ['../results/ATAC/D8/Somite.pkl', '../results/ATAC/D22-15/Myoblast.pkl'])
with open('../results/ATAC_peakst.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

ATAC = pd.concat(pd.read_pickle(f) for f in ['../results/background/D8/Somite.pkl', '../results/background/D22-15/Myoblast.pkl'])
with open('../results/ATAC_backgroundt.pkl', 'wb') as file:
            pickle.dump(ATAC, file)

del ATAC """

from pytorch_datasets import PeaksDataset
from torch.utils.data import DataLoader

data_dir = '../results/'
chrom_test = ['6','12']

nb_back = 100
pseudo_bulk_order = ["D8Somite","D22-15Myoblast"]

train_dataset = PeaksDataset(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched.pkl',
                                 data_dir + 'ATAC_peakst.pkl', data_dir + 'ATAC_backgroundt.pkl', 
                                 chrom_test, pseudo_bulk_order, nb_back)
train_dataloader = DataLoader(train_dataset, batch_size=32,
                        shuffle=True, num_workers=2)

print(train_dataset.pseudo_bulk)

import tqdm
for i, data in enumerate(train_dataloader):
    inputs, tracks = data 
    print(inputs)
    print(tracks)
    print(len(input), len(tracks))
    print(inputs[0].shape, tracks[0].shape)
    
    break