#!/bin/bash

source ../zero_shot_config.sh

export MULAN_model_path="path to MULAN-small"
export foldseek_path="Path to Mulan foldseek data"
#export DMS_index="Experiment index to run (e.g. 0,1,...216)"
export DMS_index=0

python ../../proteingym/baselines/mulan/compute_fitness.py \
            --MULAN_model_name_or_path ${MULAN_model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --structure_data_folder ${DMS_structure_folder} \
            --DMS_index $DMS_index \
	        --foldseek_path $foldseek_path \
            --output_scores_folder "${DMS_output_score_folder_subs}/MULAN/MULAN_small"
