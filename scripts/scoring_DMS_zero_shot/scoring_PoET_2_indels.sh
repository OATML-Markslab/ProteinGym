#!/bin/bash

source ../zero_shot_config.sh

export checkpoint="${checkpoint:-${PROTEINGYM_CACHE}/baselines/PoET-2/poet-2.ckpt}"
export output_scores_folder=${DMS_output_score_folder_indels}PoET-2
export msa_folder=${PROTEINGYM_CACHE}/baselines/PoET/msas/DMS_indels
export AF2_cache_folder=${PROTEINGYM_CACHE}/baselines/PoET-2/DMS_AF2_structures_cache
export DMS_index="${DMS_index:-"Experiment index to run (e.g. 0,1,...216)"}"
# expand the following env vars to absolute paths instead of paths to relative to
# the working directory since we'll be changing the working directory
export DMS_reference_file_path_indels="$(cd "$(dirname "$DMS_reference_file_path_indels")" && pwd)/$(basename "$DMS_reference_file_path_indels")"
export DMS_data_folder_indels="$(cd "$(dirname "$DMS_data_folder_indels")" && pwd)/$(basename "$DMS_data_folder_indels")"

cd ../../proteingym/baselines/PoET-2 && pixi run --frozen \
    python scripts/score.py \
    --checkpoint $checkpoint \
    --DMS_reference_file_path $DMS_reference_file_path_indels \
    --DMS_data_folder $DMS_data_folder_indels \
    --DMS_structure_folder $DMS_structure_folder \
    --DMS_index $DMS_index \
    --output_scores_folder $output_scores_folder \
    --MSA_folder $msa_folder \
    --AF2_cache_folder $AF2_cache_folder \
    --structure_in_context 0 1 \
    --inverse_folding_query 0 \
    --relative_to_wt
