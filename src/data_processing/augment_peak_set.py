#Paths assume run from src folder

#Augment the peak set, so that we do not only have regions centered on peaks
#--------------------------------------------
import pandas as pd
import numpy as np

shifts = np.array([-300, 0, 300])

peaks = pd.read_csv("../results/common_peaks.bed", header=None, sep='\t', low_memory=False, names= ["chr", "start", "end"])

s = np.tile(np.arange(len(shifts)), peaks.shape[0])
peaks = peaks.loc[peaks.index.repeat(len(shifts))]
peaks['shift_idx'] = s 

peaks.start = peaks.start + shifts[peaks.shift_idx.to_list()]
peaks.end = peaks.end + shifts[peaks.shift_idx.to_list()]

peaks = peaks.iloc[:,:3]

peaks.to_csv("../results/aug_peaks.bed", header=False, index=False, sep='\t')
