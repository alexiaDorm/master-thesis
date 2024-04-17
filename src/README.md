Quick description of scripts
--------------------------------------------------------------

Exploration/initial-exploration_scvi.ipynb: Integration using scvi of all datasets, cell types definition using scvi latent space 

utils_data_processing.py: Utils function for data processing

create_bw_cell_type.py: Script to generate the ATAC track bigWigs files by pseudo-bulk 

create_total_reads.py: Compute total of reads per pseudo-bulk for normalization

compute_coverage.py: Plot coverage in 10000 bins distribution by chromosomes

get_sequence_peaks.py: Fetch genomic sequence for peaks

GC_match_background.py: Create background sequences GC-matched to peaks and fetch their sequences

ATAC_track_peaks.py: Fetch continuous ATAC tracks for each pseudo-bulk and peak 

ATAC_track_background.py: Fetch continuous ATAC tracks for each pseudo_bulk and background regions

model.py: Pytorch bias and final model class and training loop

datasets.py: Pytorch Dataset/Dataloader class 

Description data processing pipeline
--------------------------------------------------------------

1. Define cell types using gene expression data
2. Create BigWigs files of ATAC signal for each pseudo-bulk
3. Compute the total number of reads per pseudo-bulk for normalization purpose
4. Compute coverage by pseudo bulk to determine for which one we have sufficient cells
5. Get the sequences for each peaks
6. Sample background regions from genome with same GC content distribution as peaks
7. Get the sequences of the background regions
8. Get ATAC continuous track using sliding window for peaks and background regions