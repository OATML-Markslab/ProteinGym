#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export model_checkpoint="path to ESM1b checkpoint"

export dms_output_folder="${DMS_output_score_folder_subs}/ESM1b/"

export model_type="ESM1b"
export scoring_strategy="wt-marginals"
export scoring_window="overlapping"

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/esm/compute_fitness.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms_index ${DMS_index} \
    --dms_mapping ${DMS_reference_file_path_subs} \
    --dms-input ${DMS_data_folder_subs} \
    --dms-output ${dms_output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --scoring-window ${scoring_window}
