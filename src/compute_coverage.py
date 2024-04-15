#Compute coverage of each pseudo bulk at each time points

import glob
import subprocess

NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

for d in NAME_DATASET:

    splitted_files = glob.glob('../results/bam_cell_type/' + d +'/*.bam')
    
    for f in splitted_files:
        samtools_command = "samtools depth -a " + f +  " |  awk '{sum+=$3} END { print sum/NR}'"
        
        print(f)
        subprocess.run(samtools_command, shell=True)

