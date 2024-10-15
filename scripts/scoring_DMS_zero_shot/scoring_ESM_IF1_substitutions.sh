#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

## Regression weights are at: https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50S-contact-regression.pt
#https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50S-contact-regression.pt

export model_checkpoint="path to ESM-IF1 checkpoint"
export DMS_output_score_folder=${DMS_output_score_folder_subs}/ESM-IF1/
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/esm/compute_fitness_esm_if1.py \
    --model_location ${model_checkpoint} \
    --structure_folder ${DMS_structure_folder} \
    --DMS_index $DMS_index \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --output_scores_folder ${DMS_output_score_folder} 