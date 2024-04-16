#Compute coverage of each pseudo bulk at each time points

import glob
import subprocess
import pyBigWig
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']
chrom = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']

def compute_cov(start, bw):
    return np.count_nonzero(bw.values(c, start, start + 10000))/10000

for d in NAME_DATASET:

    splitted_files = glob.glob('../results/bam_cell_type/' + d +'/*.bam')
    for f in splitted_files:
        samtools_command = "samtools depth -a " + f +  " |  awk '{sum+=$3} END { print sum/NR}'"
        
        #subprocess.run(samtools_command, shell=True)

    bw_files = glob.glob('../results/bam_cell_type/' + d +'/*.bw')
    for f in bw_files:
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

        fig.savefig("../results/coverage/" + f.removeprefix("../results/bam_cell_type/").removesuffix(".bw") + "_cov.pdf")
 


        

