#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym

#export Progen2_model_name_or_path="path to progen2 small model"
#export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/small"

#export Progen2_model_name_or_path="path to progen2 medium model"
#export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/medium"

export Progen2_model_name_or_path="path to progen2 base model"
export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/base"

#export Progen2_model_name_or_path="path to progen2 large model"
#export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/large"

#export Progen2_model_name_or_path="path to progen2 xlarge model"
#export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/xlarge"

#export Progen2_model_name_or_path="path to progen2 oas model"
#export output_scores_folder="${clinical_output_score_folder_subs}/Progen2/oas"

export DMS_index="variant index to run (e.g. 1,2,...2525)"

python ../../proteingym/baselines/progen2/compute_fitness.py \
            --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
            --DMS_reference_file_path ${clinical_reference_file_path_subs} \
            --DMS_data_folder ${clinical_data_folder_subs} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \