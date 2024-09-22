#!/bin/bash 

source ../zero_shot_config.sh
conda activate poet

export checkpoint="path to checkpoint"
export output_scores_folder=${DMS_output_score_folder_subs}/PoET

export DMS_index="Experiment index to run (e.g. 0,1,...216)"

python ../../proteingym/baselines/PoET/scripts/score.py \
    --checkpoint ${checkpoint} \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --DMS_index ${DMS_index} \
    --output_scores_folder ${output_scores_folder} \
    --MSA_folder ${DMS_MSA_data_folder} \
    --context_lengths 6144 12288 24576 \
    --batch_size 8
