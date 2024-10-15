#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export checkpoint="path to Tranception Large checkpoint"
export output_scores_folder=${DMS_output_score_folder_indels}/Tranception_no_retrieval/Tranception_L

# export checkpoint="path to Tranception Medium checkpoint"
# export output_scores_folder=${DMS_output_score_folder_indels}/Tranception_no_retrieval/Tranception_M

# export checkpoint="path to Tranception Small checkpoint"
# export output_scores_folder=${DMS_output_score_folder_indels}/Tranception_no_retrieval/Tranception_S

export DMS_data_folder=$DMS_data_folder_indels
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/tranception/score_tranception_proteingym.py \
                --checkpoint ${checkpoint} \
                --DMS_reference_file_path ${DMS_reference_file_path_indels} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index ${DMS_index} \
                --output_scores_folder ${output_scores_folder} \
                --indel_mode 