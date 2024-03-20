#!/bin/bash

source ../zero_shot_config.sh

#export SaProt_model_path="path to SaProt model"
export SaProt_model_path="/sujin/Models/SaProt/SaProt_650M_AF2"
export output_scores_folder="${DMS_output_score_folder_subs}/SaProt/SaProt_650M_AF2"
export foldseek_bin="/sujin/bin/foldseek"

#export DMS_index="Experiment index to run (e.g. 0,1,...216)"
export DMS_index=0

python ../../proteingym/baselines/saprot/compute_fitness.py \
            --foldseek_bin ${foldseek_bin} \
            --SaProt_model_name_or_path ${SaProt_model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --structure_data_folder ${DMS_structure_folder} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder}