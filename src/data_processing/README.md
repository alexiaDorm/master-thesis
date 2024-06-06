Quick description of scripts
--------------------------------------------------------------
0. Divers 
     - utils_data_processing.py: Utils functions for data processing (Fetch sequence, encode, generation ATAC tracks, GC content)
     - subsample_Somite.py: Subsample the D8_1 somite cells to have same number of cells as in D12_1 to see influence on coverage

1. Create Bw files and quality check

    - create_bw_cell_type.py: Script to generate the ATAC track bigWigs files by pseudo-bulk 

    - compute_total_reads.py: Compute total of reads per pseudo-bulk for normalization

    - compute_coverage.py: Plot coverage in 10000 bins distribution by chromosomes and pseudo-bulk

2. Create common peaks set and fetch ATAC tracks

    - create_peak_set.py: Create common peak set across time point and replicates. Only the peaks that are present in both replicates are kept. The intersection of peaks of each time point is taken as the final peak set

    - get_sequence_peaks.py: Fetch genomic sequence for peaks

    - ATAC_track_peaks.py: Fetch continuous ATAC tracks for each pseudo-bulk and peak regions

3. Create GC matched background regions and fetch ATAC tracks

    - GC_bins_genome.py: Create potential background regions by binning genome and computing the GC content of each sequence

    - GC_match_background.py: GC match genomic regions to peaks

    - ATAC_track_background.py: Fetch continuous ATAC tracks for each pseudo_bulk and background regions
