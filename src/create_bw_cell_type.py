import anndata
import subprocess
import glob
import os 

from data_preprocessing import preprocess_data

#NAME_DATASET =['D8_1','D8_2','D12_1','D12_2','D20_1', 'D20_2', 'D22_1', 'D22_2']
NAME_DATASET = ['D20_1']

#Load concatenated data
#adata = preprocess_data('../data/initial_10x_outputs/filtered_features/', '../results/cell_types.csv','../results/concat.h5ad')
adata = anndata.read_h5ad('../results/concat.h5ad')

for d in NAME_DATASET:

    if not os.path.exists('../results/bam_cell_type/' + d):
        os.makedirs('../results/bam_cell_type/' + d)

    #Create file with cell barcodes and group 
    DX =  adata[adata.obs.batch == d]

    barcodes = DX.obs.cell_type
    barcodes = barcodes.cat.rename_categories({'Myoblast/Myocytes': 'Myoblast'})
    barcodes.index = [b[:-2] for b in barcodes.index.values]
    barcodes.to_csv('../results/bam_cell_type/' + d + '/' + d + '_cell_types.tsv', header=False, sep='\t')

    #Split the bam by cell type
    sinto_command = ('sinto filterbarcodes -p 8 -b ../data/initial_10x_outputs/atac_peaks/' + 
                 d +'_ATAC.bam -c ../results/bam_cell_type/' + d +
                 '/' + d + '_cell_types.tsv --outdir ../results/bam_cell_type/' +
                 d + '/')
    
    #subprocess.run(sinto_command, shell=True)

    #Create ATAC tracks using splitted files
    splitted_files = glob.glob('../results/bam_cell_type/' + d +'/*.bam')

    for f in splitted_files:

        #Create index file first
        samtools_command = ('samtools index ' + f + 
                            ' ' + f + '.bai')
            
        subprocess.run(samtools_command, shell=True)

        bamCoverage_command = ('bamCoverage -p 8 -b ' + f  + 
                            ' -o ' + f[:-3] + 'bw')
            
        subprocess.run(bamCoverage_command, shell=True)
