Quick description of scripts
--------------------------------------------------------------
0. Divers 
     - utils_data_processing.py: Utils functions for data processing (Fetch sequence, encode, generation ATAC tracks, GC content)

1. Create pseudobulk Bigwig files

    - create_bw_cell_type.py: Script to generate the ATAC track bigWigs files for each pseudo-bulk 

2. Create common accessible regions set and fetch ATAC tracks in these regions for all pseudo bulk

    - create_peak_set.py: Create common peak set across time point and replicates. Only the peaks that are present in both replicates are kept. The union of peaks of each time point is taken as the final peak set.

    - get_sequence_peaks.py: Fetch genomic sequence for peaks

    - ATAC_track_peaks.py: Fetch ATAC tracks for each pseudo-bulk and peak regions

3. Create GC matched background regions and fetch ATAC tracks

    - GC_bins_genome.py: Create potential background regions by binning genome and computing the GC content of each sequence

    - GC_match_background.py: GC match genomic regions to peaks

    - ATAC_track_background.py: Fetch continuous ATAC tracks for each pseudo_bulk and background regions
