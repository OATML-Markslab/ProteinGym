#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env
pip install rsalor

export output_scores_folder=${DMS_output_score_folder_subs}/RSALOR/

python ../../proteingym/baselines/RSALOR/run_rsalor.py \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --MSA_folder ${DMS_MSA_data_folder} \
    --DMS_structure_folder ${DMS_structure_folder} \
    --output_scores_folder ${output_scores_folder}
