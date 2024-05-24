#Paths assume run from src folder

#Create the pseudo bulk (per time point and cell type) bigwig files , replicates are used together to generate the .bw files
#--------------------------------------------

import subprocess
import glob
import os 
import pandas as pd
import gzip

TIME_POINT = ["D8", "D12", "D20", "D22-15"]
data_path = '../../../../../projects/schuelke-cubi-muscle-dev/work/BtE_P07_P08_analyses/MULTIOME/outputs/'

cell_type = pd.read_csv('../results/cell_types.csv', index_col=0)

#Keep track of dataset number to make barcodes unique
i = 0
for t in TIME_POINT:
 
 """
    if not os.path.exists('../results/bam_cell_type/' + t):
        os.makedirs('../results/bam_cell_type/' + t)

    #Create cell type file for time point
    #Get cell barcodes for each replicates
    barcode_rep1 = data_path +  t + "_REP1_run1/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"
    barcode_rep1 = pd.read_csv(gzip.open(barcode_rep1), header=None)[0]
    barcode_rep1 = [b + "-" + str(i) for b in barcode_rep1]
    i = i + 1

    barcode_rep2 = data_path +  t + "_REP2_run1/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"
    barcode_rep2 = pd.read_csv(gzip.open(barcode_rep2), header=None)[0]
    barcode_rep2 = [b + "-" + str(i) for b in barcode_rep2]
    i = i + 1

    barcodes = barcode_rep1 + barcode_rep2
    
    barcodes_type = cell_type.loc[cell_type.index.intersection(barcodes)].copy()
    barcodes_type.index = barcodes_type.index.str[:-2]
    
    barcodes_type.to_csv('../results/bam_cell_type/' + t + '/' + t + '_cell_types.tsv', header=False, sep='\t')
    
    
    #Merge bam file of replicates
    ATAC_bam_rep1 = data_path + t + '_REP1_run1/outs/atac_possorted_bam.bam'
    ATAC_bam_rep2 = data_path + t + '_REP2_run1/outs/atac_possorted_bam.bam'

    if not os.path.exists('../results/tmp/'):
        os.makedirs('../results/tmp/')

    samtools_merge = "samtools merge -o ../results/tmp/" + t + "_merged.bam " + ATAC_bam_rep1 + " " + ATAC_bam_rep2
    subprocess.run(samtools_merge, shell=True)

    #Create index of merged file 
    samtools_index =  'samtools index ../results/tmp/' + t + '_merged.bam  ../results/tmp/' + t + '_merged.bam.bai'
    subprocess.run(samtools_index, shell=True)
    
    #Split the bam by cell type
    sinto_command = ('sinto filterbarcodes -p 8 -b ../results/tmp/' + t + '_merged.bam -c ../results/bam_cell_type/' + t +
                 '/' + t + '_cell_types.tsv --outdir ../results/bam_cell_type/' +
                 t + '/')

    subprocess.run(sinto_command, shell=True) """

    """ #Create ATAC tracks of read counts using splitted files
    splitted_files = glob.glob('../results/bam_cell_type/' + t + '/*.bam')

    for f in splitted_files:

        #Create index file first
        samtools_command = ('samtools index ' + f + 
                            ' ' + f + '.bai')
            
        subprocess.run(samtools_command, shell=True)

        bamCoverage_command = ('bamCoverage -p 8 -b ' + f  + 
                            ' -o ' + f[:-3] + 'bw --binSize 1' )
            
        subprocess.run(bamCoverage_command, shell=True) """
 
    #Adapted from https://github.com/kundajelab/chrombpnet/tree/master/chrombpnet/helpers/preprocessing

    #Create ATAC tracks of 5' count using splitted files
    splitted_files = glob.glob('../results/bam_cell_type/' + t + '/*.bam')
    plus_shift_delta, minus_shift_delta = 4, -4
    
    chrom_sizes_file =  '../results/bam_cell_type/' + t + '/sizes.genome'
    cmd_size = "samtools idxstats " + splitted_files[0] + " | cut -f1,2"
    subprocess.run(cmd_size, shell=True)
 
    for f in splitted_files:

        #Convert the bam to bed file
        p1 = subprocess.Popen(["bedtools", "bamtobed", "-i", f], stdout=subprocess.PIPE)
        
        #Create the bedgraphfile
        cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | bedtools sort -i stdin """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file)
        subprocess.run(cmd, shell=True)

        with open('../results/bam_cell_type/' + t + '/tmp', 'w') as f:
            p2 = subprocess.Popen([cmd], stdin=p1.stdout, stdout=f, shell=True)
            p1.stdout.close()
            p2.communicate()

        subprocess.run(["bedGraphToBigWig", '../results/bam_cell_type/' + t + '/tmp', chrom_sizes_file, f[:-3] + "_unstranded.bw"])

        break
    break
