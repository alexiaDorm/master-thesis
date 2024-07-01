from models.pytorch_datasets import PeaksDataset2, PeaksDataset_w_bias
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import sys
import pickle

data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chr_test = ['6','13','22']  

#Load data in train dataset: PeakDataset2
paths_ATAC_tracks = ["chr_seq.pkl", "ATAC_peaks_new.pkl", "is_defined.pkl", "idx_seq.pkl", "c_type_track.pkl"]
paths_ATAC_tracks = [data_dir + x for x in paths_ATAC_tracks]
paths_ATAC_tracks_back = ["chr_seq_back.pkl", "ATAC_new_back.pkl", "is_defined_back.pkl", "idx_seq_back.pkl", "c_type_track_back.pkl"]
paths_ATAC_tracks_back = [data_dir + x for x in paths_ATAC_tracks_back]

""" train_dataset = PeaksDataset2(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched_sample.pkl',
                                 paths_ATAC_tracks, paths_ATAC_tracks_back, 
                                 chr_train)
print("Train", train_dataset.ATAC_track.shape)

with open('../results/train_dataset.pkl', 'wb') as file:
    pickle.dump(train_dataset, file)

del train_dataset

test_dataset = PeaksDataset2(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched_sample.pkl',
                                 paths_ATAC_tracks, paths_ATAC_tracks_back, 
                                 chr_test)
print("Train", test_dataset.ATAC_track.shape)

with open('../results/test_dataset.pkl', 'wb') as file:
    pickle.dump(test_dataset, file)

del test_dataset """

#Load data in train dataset: PeakDataset_w_bias
train_dataset = PeaksDataset_w_bias(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched_sample.pkl',
                                 paths_ATAC_tracks, paths_ATAC_tracks_back, 
                                 chr_train, "../data/hg38Tn5Bias.h5")
print("Train", train_dataset.ATAC_track.shape)

with open('../results/train_dataset_bias.pkl', 'wb') as file:
    pickle.dump(train_dataset, file)

del train_dataset

test_dataset = PeaksDataset_w_bias(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched_sample.pkl',
                                 paths_ATAC_tracks, paths_ATAC_tracks_back, 
                                 chr_test, "../data/hg38Tn5Bias.h5")
print("Test", test_dataset.ATAC_track.shape)

with open('../results/test_dataset_bias.pkl', 'wb') as file:
    pickle.dump(test_dataset, file)

del test_dataset