from models.pytorch_datasets import PeaksDataset2
from torch.utils.data import DataLoader
from timeit import default_timer as timer

data_dir = "../results/"
time_order = ['D8', 'D12', 'D20', 'D22-15']

chr_train = ['1','2','3','4','5','7','8','9','10','11','12','14','15','16','17','18','19','20','21','X','Y']
chr_test = ['6','13','22']  

start = timer()

paths_ATAC_tracks = ["chr_seq.pkl", "ATAC_peaks_new.pkl", "is_defined.pkl", "idx_seq.pkl", "c_type_track.pkl"]
paths_ATAC_tracks = [data_dir + x for x in paths_ATAC_tracks]
paths_ATAC_tracks_back = ["chr_seq_back.pkl", "ATAC_peaks_new_back.pkl", "is_defined_back.pkl", "idx_seq_back.pkl", "c_type_track_back.pkl"]
paths_ATAC_tracks_back = [data_dir + x for x in paths_ATAC_tracks_back]

train_dataset = PeaksDataset2(data_dir + 'peaks_seq.pkl', data_dir + 'background_GC_matched.pkl',
                                 paths_ATAC_tracks, paths_ATAC_tracks_back, 
                                 chr_test)
train_dataloader = DataLoader(train_dataset, 64,
                        shuffle=True, num_workers=4)

print("All loaded")
end = timer()
print(end - start)
    
for data in train_dataloader:
    input, tracks, indexes = data
    print(input.shape)
    print(tracks.shape)
    print(indexes.shape)

    break