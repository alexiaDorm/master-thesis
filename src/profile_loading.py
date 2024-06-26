from models.pytorch_datasets import PeaksDataset2
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import sys
import pickle

data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chr_test = ['6','13','22']  

start = timer()

paths_ATAC_tracks = ["chr_seq.pkl", "ATAC_peaks_new.pkl", "is_defined.pkl", "idx_seq.pkl", "c_type_track.pkl"]
paths_ATAC_tracks = [data_dir + x for x in paths_ATAC_tracks]
paths_ATAC_tracks_back = ["chr_seq_back.pkl", "ATAC_new_back.pkl", "is_defined_back.pkl", "idx_seq_back.pkl", "c_type_track_back.pkl"]
paths_ATAC_tracks_back = [data_dir + x for x in paths_ATAC_tracks_back]

train_dataset = PeaksDataset2(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched_sample.pkl',
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

del test_dataset

with open('../results/train_dataset.pkl', 'rb') as file:
    train_dataset = pickle.load(file)
with open('../results/test_dataset.pkl', 'rb') as file:
    test_dataset = pickle.load(file)

""" train_dataloader = DataLoader(train_dataset, 64,
                        shuffle=True, num_workers=4)

print("All loaded")
end = timer()
print(end - start)
    
for data in train_dataloader:
    input, tracks, indexes = data
    print(input.shape)
    print(tracks.shape)
    print(indexes.shape)

    break """