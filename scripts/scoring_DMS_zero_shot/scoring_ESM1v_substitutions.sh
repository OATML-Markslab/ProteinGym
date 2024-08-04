#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# ESM-1v parameters 
# Five checkpoints for ESM-1v
export model_checkpoint1="path to seed 1 model checkpoint"
export model_checkpoint2="path to seed 2 model checkpoint"
export model_checkpoint3="path to seed 3 model checkpoint"
export model_checkpoint4="path to seed 4 model checkpoint"
export model_checkpoint5="path to seed 5 model checkpoint"
# combine all five into one string 
export model_checkpoint="${model_checkpoint1} ${model_checkpoint2} ${model_checkpoint3} ${model_checkpoint4} ${model_checkpoint5}"

export dms_output_folder="${DMS_output_score_folder_subs}/ESM1v/"

export model_type="ESM1v"
export scoring_strategy="masked-marginals"  # MSATransformer only uses masked-marginals
export scoring_window="optimal"

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
