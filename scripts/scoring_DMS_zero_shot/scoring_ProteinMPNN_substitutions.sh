#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export output_scores_folder=${DMS_output_score_folder_subs}/ProteinMPNN

export model_checkpoint="Path to ProteinMPNN model checkpoint"
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/protein_mpnn/compute_fitness.py \
    --checkpoint ${model_checkpoint} \
    --structure_folder ${DMS_structure_folder} \
    --DMS_index $DMS_index \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --output_scores_folder ${output_scores_folder}