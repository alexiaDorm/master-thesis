#Subsample the D8_1 somite cells to have same number of cells as in D12_1 to see influence on coverage
#--------------------------------------------

import pandas as pd
import subprocess
import pyBigWig
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import os

chrom = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']

def compute_cov(start, bw):
    return np.count_nonzero(bw.values(c, start, start + 10000))/10000

    
somite = pd.read_csv('../results/bam_cell_type/D8_1/D8_1_cell_types.tsv', header=None, sep='\t', index_col=0)
somite = somite[somite[1]=='Somite']
somite = somite.sample(2376)
somite[1] = 'Somite_sub'

somite.to_csv('../results/bam_cell_type/D8_1/somite_subsample_cell_types.tsv', header=False, sep='\t')

#Get BAM file file with only sampled barcodes
ATAC_bam = '../../../../../projects/schuelke-cubi-muscle-dev/work/BtE_P07_P08_analyses/MULTIOME/outputs/D8_REP1_run1/outs/atac_possorted_bam.bam'

d = 'D8_1'
sinto_command = ('sinto filterbarcodes -p 8 -b ' + ATAC_bam + 
                 ' -c ../results/bam_cell_type/' + d +
                 '/' +  'somite_subsample_cell_types.tsv --outdir ../results/bam_cell_type/' +
                 d + '/')
    
subprocess.run(sinto_command, shell=True)

#Create index file first
bam_file = '../results/bam_cell_type/D8_1/Somite_sub.bam'
samtools_command = ('samtools index ' + bam_file + ' ' + bam_file + '.bai')
subprocess.run(samtools_command, shell=True)

#Create .bw file
bamCoverage_command = ('bamCoverage -p 8 -b ' + bam_file  + 
                            ' -o ' + bam_file[:-3] + 'bw --binSize 1' )
            
subprocess.run(bamCoverage_command, shell=True)

#Compute coverage across genome
f = '../results/bam_cell_type/D8_1/Somite_sub.bw'
bw = pyBigWig.open(f)
    
all_cov =[]
fig = plt.figure()    
for i, c in enumerate(chrom):
    coverage = [compute_cov(i*10000, bw) for i in range(0,math.floor(bw.chroms(c)/10000))]
    all_cov.append(coverage)

    ax = fig.add_subplot(5, 5, i+1)
    ax.hist(coverage, bins=50)
    ax.set_title(('chr ' + c))

all_cov = list(itertools.chain(*all_cov))
ax = fig.add_subplot(5, 5, i+2)
ax.hist(all_cov, bins=50)
ax.set_title('all chromosomes')

if not os.path.exists('../results/coverage/' + d):
    os.makedirs('../results/coverage/' + d)

fig.savefig("../results/coverage/" + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + "_cov.pdf")
 