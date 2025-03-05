#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export mutation_type='substitutions'

export DMS_output_score_folder_subs="Path to individual model score files for each CV scheme"
export DMS_supervised_merged_score_folder_subs="Path where merged supervised files are to be stored"

python ../../proteingym/merge_supervised.py \
            --DMS_assays_location ${DMS_data_folder_subs} \
            --model_scores_location ${DMS_output_score_folder_subs} \
            --merged_scores_dir ${DMS_supervised_merged_score_folder_subs} \
            --mutation_type ${mutation_type} \
            --DMS_reference_file ${DMS_reference_file_path_subs} \
            --config_file ../../config.json
