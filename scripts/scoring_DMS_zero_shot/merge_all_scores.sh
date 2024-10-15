#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export mutation_type='substitutions'

python ../../proteingym/merge.py \
--DMS_assays_location ${DMS_data_folder_subs} \
--model_scores_location ${DMS_output_score_folder_subs} \
--merged_scores_dir ${DMS_merged_score_folder_subs} \
--mutation_type ${mutation_type}
