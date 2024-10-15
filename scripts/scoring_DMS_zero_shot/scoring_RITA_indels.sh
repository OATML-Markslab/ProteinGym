#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# export RITA_model_path="path to RITA small model"
# export output_scores_folder="${DMS_output_score_folder_indels}/RITA/small"

# export RITA_model_path="path to RITA medium model"
# export output_scores_folder="${DMS_output_score_folder_indels}/RITA/medium"

# export RITA_model_path="path to RITA large model"
# export output_scores_folder="${DMS_output_score_folder_indels}/RITA/large"

export RITA_model_path="path to RITA xlarge model"
export output_scores_folder="${DMS_output_score_folder_indels}/RITA/xlarge"

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/rita/compute_fitness.py \
            --RITA_model_name_or_path ${RITA_model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \
            --indel_mode