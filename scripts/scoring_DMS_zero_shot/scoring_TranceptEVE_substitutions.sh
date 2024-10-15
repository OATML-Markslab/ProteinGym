#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# export checkpoint="path to Tranception_Small checkpoint"
# export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_S

export checkpoint="path to Tranception_Medium checkpoint"
export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_M

# export checkpoint="path to Tranception_Large checkpoint"
# export output_scores_folder=${DMS_output_score_folder_subs}/TranceptEVE/TranceptEVE_L

# Replace the following paths based on where you store models and data
export DMS_index="Experiment index to run (e.g. 1,2,...217)"
export inference_time_retrieval_type="TranceptEVE"
export EVE_num_samples_log_proba=200000 
export EVE_model_parameters_location="../../proteingym/baselines/trancepteve/trancepteve/utils/eve_model_default_params.json"
export EVE_seeds="0 1 2 3 4"
export scoring_window="optimal" 
python ../../proteingym/baselines/trancepteve/score_trancepteve.py \
                --checkpoint ${checkpoint} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --inference_time_retrieval_type ${inference_time_retrieval_type} \
                --MSA_folder ${DMS_MSA_data_folder} \
                --MSA_weights_folder ${DMS_MSA_weights_folder} \
                --EVE_num_samples_log_proba ${EVE_num_samples_log_proba} \
                --EVE_model_parameters_location ${EVE_model_parameters_location} \
                --EVE_model_folder ${DMS_EVE_model_folder} \
                --scoring_window ${scoring_window} \
                --EVE_seeds ${EVE_seeds} \
                --EVE_recalibrate_probas