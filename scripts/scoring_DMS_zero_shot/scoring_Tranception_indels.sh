#!/bin/bash 

source ../zero_shot_config.sh
source activate /n/groups/marks/software/anaconda_o2/envs/proteingym_env

# export checkpoint="path to Tranception Large checkpoint"
# export output_scores_folder=${DMS_output_score_folder_indels}/Tranception/Tranception_L

# export checkpoint="path to Tranception Medium checkpoint"
# export output_scores_folder=${DMS_output_score_folder_indels}/Tranception/Tranception_M

export checkpoint="path to Tranception Small checkpoint"
export output_scores_folder=${DMS_output_score_folder_indels}/Tranception/Tranception_S

export DMS_index="Experiment index to run (e.g. 1,2,...217)"
export clustal_omega_location="path to clustal omega executable"
# Leveraging retrieval when scoring indels require batch size of 1 (no retrieval can use any batch size fitting in memory)
export batch_size_inference=1

python ../../proteingym/baselines/tranception/score_tranception_proteingym.py \
                --checkpoint ${checkpoint} \
                --batch_size_inference ${batch_size_inference} \
                --DMS_reference_file_path ${DMS_reference_file_path_indels} \
                --DMS_data_folder ${DMS_data_folder_indels} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --indel_mode \
                --clustal_omega_location ${clustal_omega_location} \
                --inference_time_retrieval \
                --MSA_folder ${DMS_MSA_data_folder} \
                --MSA_weights_folder ${DMS_MSA_weights_folder} \
                --scoring_window 'optimal'