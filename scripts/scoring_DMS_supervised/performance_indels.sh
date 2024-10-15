#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export output_performance_file_folder=../../benchmarks/DMS_supervised/indels
export input_scoring_file=../../benchmarks/raw_score_files/DMS_supervised_indels.csv

python3 ../../proteingym/performance_DMS_supervised_benchmarks.py \
                --input_scoring_file ${input_scoring_file} \
                --output_performance_file_folder ${output_performance_file_folder} \
                --DMS_reference_file_path ${DMS_reference_file_path_indels} \
                --indel_mode