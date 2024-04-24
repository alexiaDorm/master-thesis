#Create common peak set across time point and replicates
#Assume that peak in each replicate are non-overlapping
#--------------------------------------------
import os
import subprocess
import numpy as np
import pandas as pd

TIME_POINT = ["D8", "D12", "D20", "D22-15"]
data_path = '../../../../../../projects/schuelke-cubi-muscle-dev/work/BtE_P07_P08_analyses/MULTIOME/outputs/'
data_path = '../../data/test/'

#Load peaks per time point, only keep the one that were in both replicates
peaks_per_rep = []
for t in TIME_POINT:
    peaks_rep1, peaks_rep2 = data_path + t + "_REP1_run1/outs/atac_peaks.bed", data_path + t + "_REP2_run1/outs/atac_peaks.bed"
    peaks_rep1, peaks_rep2 =  pd.read_csv(peaks_rep1 ,header=None, sep='\t', skiprows=range(0, 50)), pd.read_csv(peaks_rep2 ,header=None, sep='\t', skiprows=range(0, 50))
    
    #Remove scaffold chromosomes
    peaks_rep1 = peaks_rep1[np.logical_or(np.logical_or(peaks_rep1[0].str.isnumeric(), peaks_rep1[0] == 'X'), peaks_rep1[0] == 'Y')]
    peaks_rep2 = peaks_rep2[np.logical_or(np.logical_or(peaks_rep2[0].str.isnumeric(), peaks_rep2[0] == 'X'), peaks_rep2[0] == 'Y')]

    #Merge overlapping peaks
    if not os.path.exists('../../results/tmp/'):
        os.makedirs('../../results/tmp/')

    peaks = pd.concat([peaks_rep1, peaks_rep2])
    peaks_path = "../../results/tmp/peaks.bed"
    peaks.to_csv(peaks_path, header=False, index=False, sep='\t')
    
    sort_bed = "sortBed -i " + peaks_path + " > ../../results/tmp/peaks_sorted.bed"
    subprocess.run(sort_bed, shell=True)

    merge_bedtools = "bedtools merge -i ../../results/tmp/peaks_sorted.bed -c 1 -o count > ../../results/tmp/count_peaks.bed"
    subprocess.run(merge_bedtools, shell=True)
    
    #Only keep peaks that were in both replicates
    peaks = pd.read_csv("../../results/tmp/count_peaks.bed" ,header=None, sep='\t')
    peaks = peaks[peaks[3] == 2]

    peaks_per_rep.append(peaks.iloc[:,0:3])

#Take union of peaks of all time points and merge overlapping peaks 
peaks = pd.concat(peaks_per_rep)
peaks.to_csv("../../results/tmp/all_peaks.bed", header=False, index=False, sep='\t')

sort_bed = "sortBed -i ../../results/tmp/all_peaks.bed > ../../results/tmp/all_peaks_sorted.bed"
subprocess.run(sort_bed, shell=True)

merge_bedtools = "bedtools merge -i ../../results/tmp/all_peaks_sorted.bed > ../../results/common_peaks.bed"
subprocess.run(merge_bedtools, shell=True)