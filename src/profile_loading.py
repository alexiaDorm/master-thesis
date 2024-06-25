from models.pytorch_datasets import PeaksDataset2
from torch.utils.data import DataLoader
from timeit import default_timer as timer

data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chr_test = ['6','13','22']  

start = timer()

train_dataset = PeaksDataset2(data_dir + 'peaks_seqtest.pkl', data_dir + 'background_GC_matchedt.pkl',
                                 data_dir + 'ATAC_peakstest.pkl', data_dir + 'ATAC_backgroundtest.pkl', 
                                 chr_train, time_order, 20000)
train_dataloader = DataLoader(train_dataset, 64,
                        shuffle=True, num_workers=4)
end = timer()
print(end - start)

import torch
    
for data in train_dataloader:
    input, tracks, indexes = data
    indexes = torch.stack(indexes)

    break