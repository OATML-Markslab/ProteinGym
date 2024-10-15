#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export checkpoint="path to Tranception large checkpoint"
export output_scores_folder="${clinical_output_score_folder_indels}/Tranception/Tranception_L"

# export checkpoint="path to Tranception medium checkpoint"
# export output_scores_folder="${clinical_output_score_folder_indels}/Tranception/Tranception_M"

# export checkpoint="path to Tranception small checkpoint"
# export output_scores_folder="${clinical_output_score_folder_indels}/Tranception/Tranception_S"

export DMS_index="variant index to run (e.g. 1,2,...2525)"

# Clustal Omega is required when scoring indels with retrieval (not needed if scoring indels with no retrieval)
export clustal_omega_location="path to clustal omega executable"

# Leveraging retrieval when scoring indels require batch size of 1 (no retrieval can use any batch size fitting in memory)
export batch_size_inference=1 

python ../../proteingym/baselines/tranception/score_tranception_proteingym.py \
                --checkpoint ${checkpoint} \
                --batch_size_inference ${batch_size_inference} \
                --DMS_reference_file_path ${clinical_reference_file_path_indels} \
                --DMS_data_folder ${clinical_data_folder_indels} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --indel_mode \
                --clustal_omega_location ${clustal_omega_location} \
                --inference_time_retrieval \
                --MSA_folder ${clinical_MSA_data_folder_indels} \
                --MSA_weights_folder ${clinical_MSA_weights_folder_indels} 