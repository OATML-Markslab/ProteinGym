#!/bin/bash

source ../zero_shot_config.sh

export mutation_type='indels'
python ../../proteingym/merge.py \
--DMS_assays_location ${DMS_data_folder_indels} \
--model_scores_location ${DMS_output_score_folder_indels} \
--merged_scores_dir ${DMS_merged_score_folder_indels} \
--mutation_type ${mutation_type} \
--DMS_reference_file ${DMS_reference_file_path_indels}
