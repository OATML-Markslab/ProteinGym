#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export output_performance_file_folder=../../benchmarks/DMS_zero_shot/substitutions

python ../../proteingym/performance_DMS_benchmarks.py \
--input_scoring_files_folder ${DMS_merged_score_folder_subs} \
--output_performance_file_folder ${output_performance_file_folder} \
--DMS_reference_file_path ${DMS_reference_file_path_subs} \
--DMS_data_folder ${DMS_data_folder_subs} \
--performance_by_depth
