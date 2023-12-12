#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# Ranges of DMSs to score in one run of VESPA 
export DMS_index_range_start="starting index"
export DMS_index_range_end="ending index"
export vespa_cache="cache to place ProtT5 checkpoint"

python ../../proteingym/baselines/vespa/compute_fitness.py \
    --cache_location $vespa_cache \
    --DMS_reference_file_path $DMS_reference_file_path_subs \
    --MSA_data_folder $DMS_MSA_data_folder \
    --DMS_data_folder $DMS_data_folder_subs \
    --DMS_index_range_start $DMS_index_range_start \
    --DMS_index_range_end $DMS_index_range_end 
