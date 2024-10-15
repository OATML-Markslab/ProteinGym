#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export ProtGPT2_model_name_or_path="path to ProtGPT2 model checkpoint"
export output_scores_folder="${DMS_output_score_folder_subs}/ProtGPT2"
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/protgpt2/compute_fitness.py \
            --ProtGPT2_model_name_or_path ${ProtGPT2_model_name_or_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder}