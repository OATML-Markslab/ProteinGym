#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export dms_output_folder=${DMS_output_score_folder_subs}/ESM_C/600M

python ../../proteingym/baselines/evoscale/compute_fitness.py \
    --model_type "esmc_600M" \
    --reference_csv ${DMS_reference_file_path_subs} \
    --dms_dir ${DMS_data_folder_subs} \
    --output_dir ${dms_output_folder}
