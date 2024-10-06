#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export precomputed_embedding_file="path to file with pre-computed ESM2 embeddings"

# set base for relative paths in place so we can cd safely
DMS_output_score_folder_subs=$(readlink -f $DMS_output_score_folder_subs)
DMS_data_folder_subs=$(readlink -f $DMS_data_folder_subs)
DMS_reference_file_path_subs=$(readlink -f $DMS_reference_file_path_subs)

cd ../../proteingym/baselines/vespag # this is necessary to be able to run VespaG as a module

python -m vespag eval proteingym \
    -o $DMS_output_score_folder_subs \
    --dms-directory $DMS_data_folder_subs \
    --reference-file $DMS_reference_file_path_subs \
    -e $precomputed_embedding_file
