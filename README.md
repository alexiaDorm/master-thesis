Leveraging Deep Learning and Single-Cell Multiome Data to Identify Key Gene Regulatory Elements in Skeletal Muscle Differentiation

Repository for data preprocessing and training of the cell type-aware chromatin accessibility model.

\section{Programming language and software package}

Python (v3.10.14), a programming language, some Bash scripting (v4.4.20), and awk (v4.2.1) were used throughout this work. Conda (v24.5.0) was used to manage all software packages. All computations were done on the \textit{HPC for Research cluster of the Berlin Institute of Health}.


\begin{table}[H]
\small
\centering
\caption* {Python packages}
\begin{tabular}{ l c l }
 numpy \cite{numpy} & v1.26.4 & Data manipulations \\
 pandas \cite{pandas} & v2.2.2 & Data manipulations \\
 matplotlib-base \cite{matplotlib} & v3.8.4 & Plotting \\
 seaborn \cite{seaborn} & v0.13.2 & Plotting \\
 scanpy \cite{scanpy} & v1.10.1 & Single-cell data analysis toolkit  \\ 
 scvi-tools \cite{scvi-tools}  & v1.1.2 & GEX datasets integration \\
 goatools \cite{goatools} & v1.4.12 & GO terms enrichment analysis \\
 pyfaidx \cite{pyfaidx} & v0.8.1.1 & Quick access to .fasta files in python \\
 pyBigWig \cite{pyBigWig}  & v0.3.22  & Access to bedfiles in python \\
 pytorch (cuda) \cite{pytorch} & v2.3.0 & Deep learning framework \\
 keras \cite{keras} & v2.11.0 & Tn5 bias model predictions \\
 h5py \cite{h5py}& v3.9.0 & Load h5py files \\
 scipy \cite{scipy}& v1.13.0 & Compute spearman correlation \\
 optuna \cite{optuna}& v3.6.1 & Hyperparameters optimization \\
 shap \cite{shap}& v0.45.1 & Model interpretation \\
 deeplift \cite{deeplift}& v0.6.13.0 & Compute input attribution scores \\
 modisco-lite \cite{TF_modisco} & v2.2.1 & Identify TF motifs in deep learning \\  & & attributions scores \\
 \end{tabular}
 \end{table}
 
 
% \vspace{5mm}

\begin{table}[H]
\small
\centering
\caption* {Software for high throughput sequencing data analysis}
\begin{tabular}{ l c l }
 samtools \cite{samtools} & v1.18 & BAM files manipulations \\ 
 bedtools \cite{bedtools} & v2.31.1 &  BED files manipulations\\
 sinto \cite{sinto} & v0.9.0 & Subsetting reads from BAM files \\
 deeptools \cite{deeptools} & v3.5.5 & Create bigwig files from BAM file \\ 
 ucsc-bedgraphtobigwig \cite{ucsc}& v445 &  Convert bedgraph to bigwig file \\
 meme \cite{meme} & v5.5.3 & Compare motif to known TF \\
 & &  motifs \\
JASPAR TFBS extraction \cite{jaspar} & v10 & Annotate motifs in DNA sequences \\
\end{tabular}
\end{table}
