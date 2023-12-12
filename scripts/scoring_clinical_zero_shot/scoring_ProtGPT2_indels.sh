#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export ProtGPT2_model_name_or_path="path to ProtGPT2 checpoint"
export output_scores_folder="${clinical_output_score_folder_indels}/ProtGPT2"

export DMS_index="variant index to run (e.g. 1,2,...2525)"

python ../../baselines/protgpt2/compute_fitness.py \
            --ProtGP2_model_name_or_path ${ProtGPT2_model_name_or_path} \
            --DMS_reference_file_path ${clinical_reference_file_path_indels} \
            --DMS_data_folder ${clinical_data_folder_indels} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \
            --indel_mode