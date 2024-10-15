#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

# MSA transformer checkpoint 
export model_checkpoint="path to MSA transformer checkpoint"
export DMS_index="Experiment index to run (e.g. 1,2,...217)"
export dms_output_folder="${DMS_output_score_folder_subs}/MSA_Transformer/"
export scoring_strategy=masked-marginals # MSA transformer only supports "masked-marginals"
export model_type=MSA_transformer
export scoring_window="optimal"
export random_seeds="1 2 3 4 5"
export DMS_MSA_weights_for_MSA_Transformer_folder="${DMS_MSA_weights_folder}/DMS_msa_weights_for_MSA_Transformer" # Use weights recomputed post MSA filtering used in MSA Transformer

python ../../proteingym/baselines/esm/compute_fitness.py \
    --model-location ${model_checkpoint} \
    --model_type ${model_type} \
    --dms_index ${DMS_index} \
    --dms_mapping ${DMS_reference_file_path_subs} \
    --dms-input ${DMS_data_folder_subs} \
    --dms-output ${dms_output_folder} \
    --scoring-strategy ${scoring_strategy} \
    --scoring-window ${scoring_window} \
    --msa-path ${DMS_MSA_data_folder} \
    --msa-weights-folder ${DMS_MSA_weights_for_MSA_Transformer_folder} \
    --seeds ${random_seeds}
