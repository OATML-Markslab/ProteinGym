#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export DMS_index="Experiment index to run (e.g. 1,2,...217)"
# Get seed index as the thousands digit of the array index
export seed="random seed value"

export model_parameters_location='../../proteingym/baselines/EVE/EVE/default_model_params.json'
export training_logs_location='../../proteingym/baselines/EVE/logs/'
export DMS_reference_file_path=$DMS_reference_file_path_subs
# export DMS_reference_file_path=$DMS_reference_file_path_indels

python ../../proteingym/baselines/EVE/train_VAE.py \
    --MSA_data_folder ${DMS_MSA_data_folder} \
    --DMS_reference_file_path ${DMS_reference_file_path} \
    --protein_index "${DMS_index}" \
    --MSA_weights_location ${DMS_MSA_weights_folder} \
    --VAE_checkpoint_location ${DMS_EVE_model_folder} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --threshold_focus_cols_frac_gaps 1 \
    --seed ${seed} \
    --skip_existing \
    --experimental_stream_data \
    --force_load_weights

