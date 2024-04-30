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
for i, data in tqdm(enumerate(train_dataloader)):
    inputs, tracks = data 
    print(inputs)
    print(tracks)
    print(len(input), len(tracks))
    print(inputs[0].shape, tracks[0].shape)
    
    break