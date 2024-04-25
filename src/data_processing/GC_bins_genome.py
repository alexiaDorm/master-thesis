import pickle 
import pyfaidx
import pandas as pd
import numpy as np
import math
import subprocess
from tqdm import tqdm

len_seq = 4096
stride = 2000
path_genome = '../data/hg38.fa' 

chrom = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
chrom = ['chr' + x for x in chrom]

#Bin genome and compute GC content
#--------------------------------------------
genome = pyfaidx.Fasta(path_genome)

locations, GC_content = [], []
for c in tqdm(chrom):
    
    seq = genome[c][:].seq.upper()
    nb_bins = math.floor(len(seq)/stride)

    for i in range(nb_bins):
        
        start = i * stride; sequence = seq[start:(start + len_seq)]
        gc = sum(sequence.count(n) for n in ("G", "C")) / len_seq
        GC_content.append(gc)
        locations.append({"chr": c, "start": start, "end": start+len_seq, "sequence": sequence})

locations = pd.DataFrame(locations)
locations.chr = [x[3:] for x in locations.chr]
locations['GC_cont'] = GC_content
locations['ID'] = locations.chr + ":" + locations.start.astype('str') + "-" + locations.end.astype('str')
locations = locations.set_index('ID')

if not os.path.exists('../results/tmp/'):
        os.makedirs('../results/tmp/')

locations.drop('GC_cont', axis=1).to_csv("../results/tmp/background_regions.bed", sep='\t', header=None, index=False)

with open('../results/background_GC.pkl', 'wb') as file:
    pickle.dump(locations, file)

#Check bins are not inside peaks or blacklisted regions
subprocess.run("bedtools intersect -a ../results/tmp/background_regions.bed -b ../results/common_peaks.bed -v > ..results/tmp/background_regions2.bed", shell=True)
subprocess.run("bedtools intersect -a ../results/tmp/background_regions2.bed -b ../data/h38_blacklist.bed -v > ..results/tmp/back_regions.bed", shell=True)

back_regions = pd.read_csv("../results/tmp/back_regions.bed", sep='\t', header=None)
back_regions = back_regions.iloc[:,0].astype('str') +  ":" + back_regions.iloc[:,1].astype('str') + "-" + back_regions.iloc[:,2].astype('str')

with open('../results/background_GC.pkl', 'rb') as file:
    locations = pickle.load(file)

locations = locations.loc[back_regions]

#Remove sequence with N 
locations = locations[np.logical_not(locations.sequence.str.contains("N"))]

with open('../results/background_GC.pkl', 'wb') as file:
    pickle.dump(locations, file)