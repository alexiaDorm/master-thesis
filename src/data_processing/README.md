Quick description of scripts
--------------------------------------------------------------

utils_data_processing.py: Utils function for data processing

create_bw_cell_type.py: Script to generate the ATAC track bigWigs files by pseudo-bulk 

create_total_reads.py: Compute total of reads per pseudo-bulk for normalization

compute_coverage.py: Plot coverage in 10000 bins distribution by chromosomes

create_peak_set.py: Take all called peaks in datasets and merge overlapping peaks

get_sequence_peaks.py: Fetch genomic sequence for peaks

GC_bins_genome.py: Create potential background regions by binning genome and computing the GC content of each sequence

GC_match_background.py: GC match genomic regions to peaks

ATAC_track_peaks.py: Fetch continuous ATAC tracks for each pseudo-bulk and peak 

ATAC_track_background.py: Fetch continuous ATAC tracks for each pseudo_bulk and background regions

model.py: Pytorch bias and final model class and training loop

datasets.py: Pytorch Dataset/Dataloader class 

Description data processing pipeline
--------------------------------------------------------------
1. Create common peaks by merging all overlapping peaks 

2. Create BigWigs files of ATAC signal for each pseudo-bulk (cell_type + time)

3. Compute coverage by pseudo bulk to determine for which one we have sufficient cells

4. Get the sequences for each peaks

5. Create potential background regions by binning genome and computing the GC content of each sequence

6. GC match each peak to background region 

7. Compute the total number of reads per pseudo-bulk for normalization purpose 

8. Get ATAC continuous track using sliding window for peaks and background regions