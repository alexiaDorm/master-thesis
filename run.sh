#!/bin/bash

# Set a name for the job (-J or --job-name).
#SBATCH --job-name=split_BAM

# Set the file to write the stdout and stderr to (if -e is not set; -o or --output).
#SBATCH --output=logs/%x-%j.log

# Set the number of cores (-n or --ntasks).
#SBATCH --ntasks=8

# Force allocation of the two cores on ONE node.
#SBATCH --nodes=1

# Set the total memory. Units can be given in T|G|M|K.
#SBATCH --mem=8G

# Optionally, set the partition to be used (-p or --partition).
#SBATCH --partition=medium

# Set the expected running time of your job (-t or --time).
# Formats are MM:SS, HH:MM:SS, Days-HH, Days-HH:MM, Days-HH:MM:SS
#SBATCH --time=7:30:00

export TMPDIR=/fast/users/${USER}/scratch/tmp
mkdir -p ${TMPDIR}

module load miniconda3
source activate base
python create_bw_cell_type.py