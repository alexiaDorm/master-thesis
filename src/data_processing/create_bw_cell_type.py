#Paths assume run from src folder and that there is a results folder at the same level than src

#Create the pseudo bulk (per time point and cell type) bigwig files , replicates are merged to generate the .bw files

#Note:
#The cell type annotation needs to be provided in a csv format with columns barcode and cell_type.
#The barcode is the barcode of each cell in the experiments. To make the barcodes unique across batches the batch ID is added (D8_REP1='-0', D8_REP2='-1', D12_REP1='-2', ..., D22_REP2='-7')

# Example of file:
#------------------
#barcode, cell_type
#AAACAGCCAAACGCGA-1-0,Somite
#AAACAGCCACAGGAAT-1-0,Somite
#AAACAGCCACATGCTA-1-0,Mesenchymal
#AAACAGCCAGGCCATT-1-0,Somite
#...
#GGTCAAGCAGCATGTC-1-5,Neuronal progenitor
#GGTCAAGCAGCCTAAC-1-5,Myogenic progenitor
#GGTCAAGCAGGCTTCG-1-5,Neuronal progenitor
#GGTCAAGCATAAGTCT-1-5,Mesenchymal
#------------------

#--------------------------------------------

import subprocess
import glob
import os 
import pandas as pd
import gzip

#Change here the paths to MULTIOME data and cell type annotation if needed
data_path = '/data/cephfs-1/work/projects/schuelke-cubi-muscle-dev/BtE_P07_P08_analyses/MULTIOME/outputs/'
cell_type = pd.read_csv('../results/cell_types.csv', index_col=0)

TIME_POINT = ["D8", "D12", "D20", "D22-15"]

#Keep track of batch number to make barcodes unique
i = 0
for t in TIME_POINT:
 

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

    subprocess.run(sinto_command, shell=True)
 
    #Adapted from https://github.com/kundajelab/chrombpnet/tree/master/chrombpnet/helpers/preprocessing

    #Create ATAC tracks of 5' count using splitted files

    splitted_files = glob.glob('../results/bam_cell_type/' + t + '/*.bam')
    plus_shift_delta, minus_shift_delta = 4, -4

    #Create size genome file
    chrom_sizes_file =  '../results/bam_cell_type/' + t + '/sizes.genome'
    cmd_size = "samtools idxstats " + splitted_files[0] + " | cut -f1,2 > " + chrom_sizes_file
    subprocess.run(cmd_size, shell=True)

    chrom_size = pd.read_csv(chrom_sizes_file, sep='\t', header=None, index_col=0)
    chrom_size = chrom_size.loc[['1','2','3','4','5','6','7','8','9','10','11','12', '13','14','15','16','17','18','19','20','21','22','X','Y']]
    chrom_size.to_csv(chrom_sizes_file, sep='\t', header=None)

    for f in splitted_files:
        print(f)

        #Remove scaffolds chromosomes
        print("Removing scaffold chromosomes")
        cmd_remove_scaff = "samtools view -b " + f + " $(echo {1..22} X Y)  > ../results/bam_cell_type/output.bam"
        subprocess.run(cmd_remove_scaff, shell=True)

        #Convert the bam to bed file
        print("Convert bam to bed")
        cmd_bam_to_bed = "bedtools bamtobed -i" + " ../results/bam_cell_type/output.bam > ../results/bam_cell_type/output.bed"
        subprocess.run(cmd_bam_to_bed, shell=True)

        #Create the bedgraphfile
        print("Create the bedGraphfile")
        cmd = """awk -v OFS="\\t" '{{if ($6=="+"){{print $1,$2{0:+},$3,$4,$5,$6}} else if ($6=="-") {{print $1,$2,$3{1:+},$4,$5,$6}}}}' ../results/bam_cell_type/output.bed | sort -k1,1 | bedtools genomecov -bg -5 -i stdin -g {2} | bedtools sort -i stdin """.format(plus_shift_delta, minus_shift_delta, chrom_sizes_file)       
        cmd = cmd + "> ../results/bam_cell_type/output"
        subprocess.run(cmd, shell=True)

        #Convert to Bigwig
        print("Convert to bigwig")
        cmd_bw = "bedGraphToBigWig ../results/bam_cell_type/output " + chrom_sizes_file + " "+  f[:-4] + "_unstranded.bw"
        subprocess.run(cmd_bw, shell=True)
