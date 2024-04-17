import glob
import  subprocess
import pandas as pd

NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']

for d in NAME_DATASET:

    bam_files = glob.glob('../results/bam_cell_type/' + d +'/*.bam')

    names, total_reads = [], []
    for f in bam_files:
        
        samtools_command = 'samtools view -c ' + f
        tot = subprocess.check_output(samtools_command, shell=True)
        
        tot.append(total_reads)
        names.append(f.removeprefix('../results/bam_cell_type/').removesuffix('.bam'))

    total_reads = pd.DataFrame({'batch': names, 'total': total_reads})
    total_reads.to_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index=False)