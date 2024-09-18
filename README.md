Leveraging Deep Learning and Single-Cell Multiome Data to Identify Key Gene Regulatory Elements in Skeletal Muscle Differentiation

Repository for data preprocessing and training of the cell type-aware chromatin accessibility model.

# Programming language and software package

Python (v3.10.14), a programming language, some Bash scripting (v4.4.20), and awk (v4.2.1) were used throughout this work. Conda (v24.5.0) was used to manage all software packages. All computations were done on the HPC for Research cluster of the Berlin Institute of Health.


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
