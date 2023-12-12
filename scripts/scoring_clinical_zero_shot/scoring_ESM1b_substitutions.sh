#!/bin/bash

source ../zero_shot_config.sh 
source activate proteingym_env

export DMS_index="variant index to run (e.g. 1,2,...2525)"

# ESM-1v 
# export model_checkpoint1="path to seed 1 model checkpoint"
# export model_checkpoint2="path to seed 2 model checkpoint"
# export model_checkpoint3="path to seed 3 model checkpoint"
# export model_checkpoint4="path to seed 4 model checkpoint"
# export model_checkpoint5="path to seed 5 model checkpoint"
# combine all five into one string 
# export model_checkpoint="${model_checkpoint1} ${model_checkpoint2} ${model_checkpoint3} ${model_checkpoint4} ${model_checkpoint5}"
# export dms_output_folder="${clinical_output_score_folder_subs}/ESM1v/"
# export scoring_window="optimal"
# export model_type="ESM1v"

# ESM1b:
export model_checkpoint=/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/baseline_models/ESM1b/esm1b_t33_650M_UR50S.pt

export dms_output_folder="${clinical_output_score_folder_subs}/ESM1b/"

export scoring_strategy="wt-marginals"
export scoring_window="overlapping"  # For long proteins
export model_type="ESM1b"

python ../../proteingym/baselines/esm/compute_fitness.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms-input ${clinical_data_folder_subs} \
    --dms-output ${dms_output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --scoring-window ${scoring_window} \
    --dms_mapping ${clinical_reference_file_path_subs} \
    --dms_index ${dms_index}
