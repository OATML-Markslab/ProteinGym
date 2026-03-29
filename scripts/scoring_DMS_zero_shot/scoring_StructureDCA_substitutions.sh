#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env
pip install structuredca

export output_scores_folder=${DMS_output_score_folder_subs}/StructureDCA

python ../../proteingym/baselines/StructureDCA/run_structuredca.py \
    --reference_file_path ${DMS_reference_file_path_subs} \
    --data_folder ${DMS_data_folder_subs} \
    --MSA_folder ${DMS_MSA_data_folder} \
    --structure_folder ${DMS_structure_folder} \
    --output_scores_folder ${output_scores_folder}
