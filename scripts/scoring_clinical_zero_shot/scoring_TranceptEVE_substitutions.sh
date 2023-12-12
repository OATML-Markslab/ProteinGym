#!/bin/bash 

source ../zero_shot_config.sh 
source activate proteingym_env

# export checkpoint="path to Tranception small checkpoint"
# export output_scores_folder=${clinical_output_score_folder_subs}/TranceptEVE/TranceptEVE_S

export checkpoint="path to Tranception medium checkpoint"
export output_scores_folder=${clinical_output_score_folder_subs}/TranceptEVE/TranceptEVE_M

# export checkpoint="path to Tranception large checkpoint"
# export output_scores_folder=${clinical_output_score_folder_subs}/TranceptEVE/TranceptEVE_L

export DMS_index="variant index to run (e.g. 1,2,...2525)"
export DMS_reference_file_path="../../reference_files/clinical_substitutions.csv"
export inference_time_retrieval_type="TranceptEVE"
export EVE_num_samples_log_proba=200000 
export EVE_model_parameters_location="../../proteingym/baselines/trancepteve/trancepteve/utils/eve_model_default_params.json"
export EVE_seeds="random seed values for EVE models"
export scoring_window="optimal" 
python ../../proteingym/baselines/trancepteve/score_trancepteve.py \
                --checkpoint ${checkpoint} \
                --DMS_reference_file_path ${clinical_reference_file_path_subs} \
                --DMS_data_folder ${clinical_data_folder_subs} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --inference_time_retrieval_type ${inference_time_retrieval_type} \
                --MSA_folder ${clinical_MSA_data_folder_subs} \
                --MSA_weights_folder ${clinical_MSA_weights_folder_subs} \
                --EVE_num_samples_log_proba ${EVE_num_samples_log_proba} \
                --EVE_model_parameters_location ${EVE_model_parameters_location} \
                --EVE_model_folder ${clinical_EVE_model_folder} \
                --scoring_window ${scoring_window} \
                --EVE_seeds ${EVE_seeds} \
                --EVE_recalibrate_probas \
                --clinvar_scoring