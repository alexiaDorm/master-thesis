#!/bin/bash

# Set a name for the job (-J or --job-name).
#SBATCH --job-name=pre_processing

# Set the file to write the stdout and stderr to (if -e is not set; -o or --output).
#SBATCH --output=logs_pre.log

# Set the number of cores (-n or --ntasks).
#SBATCH --ntasks=8

# Force allocation of the two cores on ONE node.
#SBATCH --nodes=1

# Set the total memory. Units can be given in T|G|M|K.
#SBATCH --mem=50G

# Set the expected running time of your job (-t or --time).
# Formats are MM:SS, HH:MM:SS, Days-HH, Days-HH:MM, Days-HH:MM:SS
#SBATCH --time=24:00:00

export TMPDIR=/fast/users/${USER}/scratch/tmp
mkdir -p ${TMPDIR}

conda run -n masterthesis python process_data.py