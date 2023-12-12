#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export DMS_index="Experiment index to run (e.g. 1,2,...217)"
export model_parameters_location='../../proteingym/baselines/EVE/EVE/default_model_params.json'
export training_logs_location='../../proteingym/baselines/EVE/logs/'
export computation_mode='DMS'
export output_score_folde="${DMS_output_score_folder_subs}/EVE/"
export num_samples_compute_evol_indices=20000
export batch_size=1024  # Pushing batch size to limit of GPU memory
export random_seeds="0 1 2 3 4"

python ../../proteingym/baselines/EVE/compute_evol_indices_DMS.py \
    --MSA_data_folder ${DMS_MSA_data_folder} \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --protein_index ${DMS_index} \
    --VAE_checkpoint_location ${DMS_EVE_model_folder} \
    --model_parameters_location ${model_parameters_location} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --output_evol_indices_location ${output_score_folder} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size} \
    --aggregation_method "full" \
    --threshold_focus_cols_frac_gaps 1 \
    --skip_existing \
    --MSA_weights_location ${DMS_MSA_weights_folder} \
    --random_seeds ${random_seeds} 
