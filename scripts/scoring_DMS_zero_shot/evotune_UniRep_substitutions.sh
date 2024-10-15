#!/bin/bash 

source ../zero_shot_config.sh
source activate protein_fitness_prediction_hsu

export OMP_NUM_THREADS=1

export savedir="folder to save evotuned models to"
export initial_weights_dir="initial weights checkpoint for UniRep model"
export DMS_reference_file_path=$DMS_reference_file_path_subs
export DMS_index="Experiment index to run (E.g. 0,1,2,...,217)"
# uncomment below to run for indels 
# export DMS_reference_file_path=$DMS_reference_file_path_indels
export steps=13000 #Same as Unirep paper

python ../../proteingym/baselines/unirep/unirep_evotune.py \
    --seqs_fasta_path $DMS_MSA_data_folder \
    --save_weights_dir $savedir \
	--initial_weights_dir $initial_weights_dir \
	--num_steps $steps \
    --batch_size 128 \
    --mapping_path $DMS_reference_file_path \
    --DMS_index $DMS_index \
    --max_seq_len 500