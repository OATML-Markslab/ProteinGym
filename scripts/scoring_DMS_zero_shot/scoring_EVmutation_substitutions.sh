#!/bin/bash

source ../zero_shot_config.sh
source activate evcouplings_env

export output_score_folder=${DMS_output_score_folder_subs}/EVmutation/
export model_folder="path to folder containing EVCouplings models"
export DMS_index="Experiment index to run (e.g. 1,2,...217)"
python ../../proteingym/baselines/EVmutation/score_mutants.py \
    --DMS_index $DMS_index \
    --DMS_data_folder $DMS_data_folder_subs \
    --model_folder $model_folder \
    --output_scores_folder $output_score_folder \
    --DMS_reference_file_path ${DMS_reference_file_path_subs}