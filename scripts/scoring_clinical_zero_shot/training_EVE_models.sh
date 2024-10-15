#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export DMS_index="variant index to run (e.g. 1,2,...2525)"

export model_parameters_location='../../proteingym/baselines/EVE/EVE/default_model_params.json'
export training_logs_location='../../proteingym/baselines/EVE/logs/'
export clinical_reference_file_path=$clinical_reference_file_path_subs

python ../../proteingym/baselines/EVE/train_VAE.py \
    --MSA_data_folder ${clinical_MSA_data_folder_subs} \
    --DMS_reference_file_path ${clinical_reference_file_path} \
    --protein_index "${DMS_index}" \
    --MSA_weights_location ${clinical_MSA_weights_folder_subs} \
    --VAE_checkpoint_location ${clinical_EVE_model_folder} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --threshold_focus_cols_frac_gaps 1 \
    --seed 0 \
    --skip_existing \
    --experimental_stream_data \
    --force_load_weights

