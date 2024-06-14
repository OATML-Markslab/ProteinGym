#!/bin/bash

source ../zero_shot_config.sh

export MULAN_model_path="DFrolova/MULAN-small"
export use_foldseek=False

#export DMS_index="Experiment index to run (e.g. 0,1,...216)"
export DMS_index=0

python ../../proteingym/baselines/mulan/compute_fitness.py \
            --MULAN_model_name_or_path ${MULAN_model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --structure_data_folder ${DMS_structure_folder} \
            --DMS_index $DMS_index \
            --use_foldseek $use_foldseek \
            # --foldseek_path $foldseek_path \
            --output_scores_folder ${output_scores_folder}
            # --output_dataset_folder
