#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# MLM 10B model
export dms_output_folder=${DMS_output_score_folder_subs}/xtrimopglm/proteinglm-10b-mlm
python ../../proteingym/baselines/xtrimopglm/compute_fitness.py \
    --model_type "proteinglm-10b-mlm" \
    --eval_mode "mlm" \
    --reference_csv ${DMS_reference_file_path_subs} \
    --dms_dir ${DMS_data_folder_subs} \
    --output_dir ${dms_output_folder}

# CLM 7B model
export dms_output_folder=${DMS_output_score_folder_subs}/xtrimopglm/proteinglm-7b-clm
python ../../proteingym/baselines/xtrimopglm/compute_fitness.py \
    --model_type "proteinglm-7b-clm" \
    --eval_mode "clm" \
    --reference_csv ${DMS_reference_file_path_subs} \
    --dms_dir ${DMS_data_folder_subs} \
    --output_dir ${dms_output_folder}

# 100B-int4 model
export dms_output_folder=${DMS_output_score_folder_subs}/xtrimo/proteinglm-100b-int4
python ../../proteingym/baselines/xtrimopglm/compute_fitness.py \
    --model_type "proteinglm-100b-int4" \
    --eval_mode "both" \
    --reference_csv ${DMS_reference_file_path_subs} \
    --dms_dir ${DMS_data_folder_subs} \
    --output_dir ${dms_output_folder}