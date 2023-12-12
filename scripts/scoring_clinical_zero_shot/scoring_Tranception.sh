#!/bin/bash 

source ../zero_shot_config.sh
source activate tranception_env

# Replace the following paths based on where you store models and data
export checkpoint="path to Tranception large checkpoint"
export output_scores_folder="${clinical_output_score_folder_subs}/Tranception/Tranception_L"

# export checkpoint="path to Tranception medium checkpoint"
# export output_scores_folder="${clinical_output_score_folder_subs}/Tranception/Tranception_M"

# export checkpoint="path to Tranception small checkpoint"
# export output_scores_folder="${clinical_output_score_folder_subs}/Tranception/Tranception_S"

export DMS_index="variant index to run (e.g. 1,2,...2525)"

export batch_size_inference=10

python ../../proteingym/baselines/tranception/score_tranception_proteingym.py \
                --checkpoint ${checkpoint} \
                --batch_size_inference ${batch_size_inference} \
                --DMS_reference_file_path ${clinical_reference_file_path_subs} \
                --DMS_data_folder ${clinical_data_folder_subs} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --inference_time_retrieval \
                --MSA_folder ${clinical_MSA_data_folder_subs} \
                --MSA_weights_folder ${clinical_MSA_weights_folder_subs} 