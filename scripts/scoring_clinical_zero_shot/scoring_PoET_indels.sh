#!/bin/bash 

source ../zero_shot_config.sh
conda activate poet

export checkpoint="path to checkpoint"
export output_scores_folder=${clinical_output_score_folder_indels}/PoET

export DMS_index="variant index to run (e.g. 0,1,...2524)"

python ../../proteingym/baselines/PoET/scripts/score.py \
    --checkpoint ${checkpoint} \
    --DMS_reference_file_path ${clinical_reference_file_path_indels} \
    --DMS_data_folder ${clinical_data_folder_indels} \
    --DMS_index ${DMS_index} \
    --output_scores_folder ${output_scores_folder} \
    --MSA_folder ${clinical_MSA_data_folder_indels} \
    --context_lengths 49152 \
    --batch_size 8 \
    --relative_to_wt
