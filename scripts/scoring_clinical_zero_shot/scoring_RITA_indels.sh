#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export RITA_model_path="path to RITA small checkpoint"
export output_scores_folder="${clinical_output_score_folder_indels}/RITA/small"

#export RITA_model_path="path to RITA medium checkpoint"
#export output_scores_folder="${clinical_output_score_folder_indels}/RITA/medium"

#export RITA_model_path="path to RITA large checkpoint"
#export output_scores_folder="${clinical_output_score_folder_indels}/RITA/large"

#export RITA_model_path="path to RITA xlarge checkpoint"
#export output_scores_folder="${clinical_output_score_folder_indels}/RITA/xlarge"

export DMS_index="variant index to run (e.g. 1,2,...2525)"

python ../../proteingym/baselines/rita/compute_fitness.py \
            --RITA_model_path ${RITA_model_path} \
            --DMS_reference_file_path ${clinical_reference_file_path_indels} \
            --DMS_data_folder ${clinical_reference_file_path_indels} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \
            --indel_mode