import subprocess
import glob
import pandas as pd
import pickle

TIME_POINT = ["D20"]
bias_model_path = "../data/bias_models/ATAC/ENCSR868FGK_bias_fold_0.h5"

#Reformat bed file with peaks to have required 10 colummns
with open('../results/peaks_seq.pkl', 'rb') as file:
    peaks = pickle.load(file)

peaks = peaks[["chr","start","end"]] 
#peaks.loc[:,"chr"] = ["chr" + x for x in peaks.chr]
peaks["name"] = "noname"; peaks["score"] = 0; peaks["strand"] = "."
peaks["thickStart"] = peaks.start; peaks["thickEnd"] = peaks.end
peaks["random1"] = 0; peaks["random2"] = 0; peaks["random3"] = 0
peaks["summit"] = peaks.end - peaks.start

peaks.to_csv('../results/peaks.bed', sep="\t", index=False, header=False)

del peaks

#Reformat bed file with negative regions to have required 10 columns
with open('../results/background_GC_matched.pkl', 'rb') as file:
    back = pickle.load(file)

back = back[["chr","start","end"]] 
#back.loc[:,"chr"] = ["chr" + x for x in back.chr]
back["name"] = "noname"; back["score"] = 0; back["strand"] = "."
back["thickStart"] = back.start; back["thickEnd"] = back.end
back["random1"] = 0; back["random2"] = 0; back["random3"] = 0
back["summit"] = back.end - back.start

back.to_csv('../results/neg_reg.bed', sep="\t", index=False, header=False)

del back

for t in TIME_POINT:

    chrom_sizes_file =  '../results/bam_cell_type/' + t + '/sizes.genome'

    splitted_files = glob.glob('../results/bam_cell_type/' + t + '/*.bam')
    for f in splitted_files:

        print("Removing scaffold chromosomes")
        cmd_remove_scaff = "samtools view -b " + f + " $(echo {1..22} X Y)  > ../results/bam_cell_type/output.bam"
        subprocess.run(cmd_remove_scaff, shell=True)

        cmd_chromBPnet = 'chrombpnet pipeline \
        -ibam ../results/bam_cell_type/output.bam \
        -d "ATAC" \
        -g ../data/hg38_nochr.fa \
        -c ' + chrom_sizes_file + ' \
        -p ../results/peaks.bed \
        -n ../results/neg_reg.bed \
        -fl ../data/fold_0.json \
        -b ../data/bias_model/ENCSR868FGK_bias_fold_0.h5 \
        -o ../results/chrombpnet_model/'

        subprocess.run(cmd_chromBPnet, shell=True)

        break

    