import glob
import  subprocess
import pandas as pd

NAME_DATASET =['D8''D12''D20''D22-15']

for d in NAME_DATASET:

    bam_files = glob.glob('../results/bam_cell_type/' + d +'/*.bam')

    names, total_reads = [], []
    for f in bam_files:
        
        samtools_command = 'samtools view -c ' + f
        tot = subprocess.check_output(samtools_command, shell=True)
        
        total_reads.append(tot)
        names.append(f.removeprefix('../results/bam_cell_type/').removesuffix('.bam'))

    total_reads = pd.DataFrame({'batch': names, 'total': total_reads})
    total_reads.to_csv('../results/bam_cell_type/' + d + '/total_reads.csv', header=None, index=False)