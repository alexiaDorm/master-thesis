# Leveraging Deep Learning and Single-Cell Multiome Data to Identify Key Gene Regulatory Elements in Skeletal Muscle Differentiation

Repository for data preprocessing and training and interpretation of a cell type-aware chromatin accessibility model.

## General organization of src folder
- archive: Old scripts
- exploration: Initial exploration of Multiome data + Integration and cell type annotation
- data_processing: all scripts used to generate the sequences/accessibility pairs
- models: Files defining pytorch classes (dataset, models), losses, evaluation metrics, and training loop compatible with optuna hyperparameters tuning
- interpretation: Definition of functions to compute attribution maps and generate synthetic sequences
- .: Various scripts and notebooks for training model and interpretation

For more details on the scripts/notebook refered to the README.md in each folder
  
## Step by step
Describe all nessesary scripts and notebooks to run to train and use the model

### Download reference genome and Tn5 bias model
   - The reference genome should be called "hg38.fa" and placed in data/.
   - The Tn5 bias model (Tn5_NN_model.h5) can be downloaded from: https://zenodo.org/records/7121027#.ZCbw4uzMI8N It should also be placed into the data folder.

### Generation of accessibility training dataset
1. Generate pseudobulk BigWig files per pseudo-bulk
   - data_processing/create_bw_cell_type.py
      Note that the cell type annotation needs to be a specific csv format and the paths to the data and cell type annotation may need to be changed. See file for more details.
     
2. Generate set of accessible regions and fetch accessibility for each pseudobulk
   - data_processing/create_peak_set.py: Create the accessible regions set used for training.
      Note that if another peak set is used, you can simply run the following two scripts if you save the genomic regions used for training in a bed file (chr, start, end) at ../results/common_peaks.bed
   - data_processing/get_sequence_peaks.py: Fetch sequence in each peak region
   - data_processing/ATAC_track_peaks.py: Get accessibility signal for each pseudo-bulk and peak regions
     Note that all cell types need to be specified in the all_cell_types list. Important point is that all pseudobulk bigwig composed of an insufficient number of cells should be deleted from results/bam_cell_type or the pseudobulk signal with be added to training dataset.

3. Add background regions to the dataset
   - data_processing/GC_bins_genome.py: Create potential negative regions by binning genome of equal length
   - data_processing/GC_match_background.py: GC match the genomic regions to peak regions
   - data_processing/ATAC_track_background.py: Get accessibility signal for each pseudo-bulk and background regions (sample 10% to add to training example)

 ### Training model using pytorch framework
 1.  Load the pytorch dataset with all the training examples and store them for fast access.
    - load_pytorch_dataset.py
      Note that if you are not interested in training a model without bias correction you could comment out the second part of the script
     
 3. Train model
    - train_w_bias.py: Training of model with bias correction.
      Note that the prefix appended to the saved results and hyperparameters can be changed in the script.
    - train_wo_bias.py: Training of model without bias correction
    - compare_models.py: Visualise and compare performance of different models
      
 4. Validation of learnt representation
    - tn5_bias_check.py: Check if model recognized tn5 bias and considers it important for its predictions
    - visu_first_filter.ipynb: Determine the consensus sequence activating each filter and matched them to know TF motifs. The results can be found in results/tomtom_out/
    - TF_check.py: Check if model can recognize the provided TF motifs in syntetic sequences
    - motif_discovery.ipynb: Find TF motifs in attribution maps of test sequences using DeepLift/TF-MOdisco 
   
    - predict.py: Generate predictions for each sequences from the training data
    - inspect_pred.ipynb: Look at first layer cell type encoding weights and compare prediction to observe accessibility.
      
 5. Estimation of variants effect on chromatin accessibility
    - predict_reg_regions.py: Make predictions for accessible regions with and without the variants found in accessible regulatory regions. Identify the variants disturbing chromatin accessibility and compute attribution maps for them.
    - estimate_var_effect.ipynb: Inspect predictions of reference and mutated alleles and prioritised variants. Inspect prioritized attributions maps around varaints
   
## Description of data format for storage 
- The genomic regions and sequences are stored in a pandas DataFrame with columns: chr, start, end, and sequence (peaks_seq.pkl, background_GC_matched.pkl)

|  chr  |   start   |    end     | sequence (4096bp) |  
|-------|-----------|------------|-------------------|
|  1    |   14154   |    15100   | ..CAGGTGTGTGATG.. |
|  1    |   14154   |    15100   | ..ATTAGGTCTCAGC.. |
                     ....
- ATAC tracks are stored in a tensor of shape: (# peaks * # c_type) x 1024 x # time point (ATAC_peaks_new.pkl, ATAC_new_back.pkl)
- is_defined is a tensor of the size: (# peaks * # c_type) x # time point, defining which tracks are defined. For the first time point some c_type are not present, but to make it easier to store the data and manipulate these are defined as zero vector. But this means that the loss should not be optimized or model evaluated on these data points. So is_defined is used to remove this values from loss and evaluation. (is_defined.pkl, is_defined_back.pkl)
- The name of the genomic region is stored in idx_seq.pkl and idx_seq_back.pkl matching the first dimension of the ATAC track. ((# peaks * # c_type) x # time point)
- The chromosome from which track is from  is stored in chr_seq.pkl and chr_seq_back.pkl matching the first dimension of the ATAC track. ((# peaks * # c_type) x # time point)
- The cell type linked to each track is stored in c_type_track.pkl and c_type_track_back.pkl matching to first dimension of the ATAC track. ((# peaks * # c_type) x # time point)
                     
## Programming language and software package

Python (v3.10.14), a programming language, some Bash scripting (v4.4.20), and awk (v4.2.1) were used throughout this work. Conda (v24.5.0) was used to manage all software packages. All computations were done on the HPC for Research cluster of the Berlin Institute of Health.

Python packages

|                 |           |                                        |
|-----------------|-----------|----------------------------------------|
| numpy           | v1.26.4   | Data manipulations                     |
| pandas          | v2.2.2    | Data manipulations                     |
| matplotlib-base | v3.8.4    | Plotting                               |
| seaborn         | v0.13.2   | Plotting                               |
| scanpy          | v1.10.1   | Single-cell data analysis toolkit      |
| scvi-tools      | v1.1.2    | GEX datasets integration               |
| goatools        | v1.4.12   | GO terms enrichment analysis           |
| pyfaidx         | v0.8.1.1  | Quick access to .fasta files in python |
| pyBigWig        | v0.3.22   | Access to bedfiles in python           |
| pytorch (cuda)  | v2.3.0    | Deep learning framework                |
| keras           | v2.11.0   | Tn5 bias model predictions             |
| h5py            | v3.9.0    | Load h5py files                        |
| scipy           | v1.13.0   | Compute spearman correlation           |
| optuna          | v3.6.1    | Hyperparameters optimization           |
| shap            | v0.45.1   | Model interpretation                   |
| deeplift        | v0.6.13.0 | Compute input attribution scores       |
| modisco-lite    | v2.2.1    | Identify TF motifs in deep learning    |

 
Software for high throughput sequencing data analysis

|                        |         |                                   |
|------------------------|---------|-----------------------------------|
| samtools               | v1.18   | BAM files manipulations           |
| bedtools               | v2.31.1 | BED files manipulations           |
| sinto                  | v0.9.0  | Subsetting reads from BAM files   |
| deeptools              | v3.5.5  | Create bigwig files from BAM file |
| ucsc-bedgraphtobigwig  | v445    | Convert bedgraph to bigwig file   |
| meme                   | v5.5.3  | Compare motif to known TF         |
|                        |         | motifs                            |
| JASPAR TFBS extraction | v10     | Annotate motifs in DNA sequences  |

## Architechture note
In the end using a fully connected layer for profile prediction resulted in very large number of parameters (18 millions!). Should probably switch this fully connected layer to a convolution instead to reduce this number.
