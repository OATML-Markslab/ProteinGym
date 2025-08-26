#!/bin/bash

source ../zero_shot_config.sh

export checkpoint="${checkpoint:-${PROTEINGYM_CACHE}/baselines/PoET-2/poet-2.ckpt}"
export output_scores_folder=${DMS_output_score_folder_subs}PoET-2
export msa_folder=${PROTEINGYM_CACHE}/baselines/PoET/msas/DMS_substitutions
export AF2_cache_folder=${PROTEINGYM_CACHE}/baselines/PoET-2/DMS_AF2_structures_cache
export DMS_index="${DMS_index:-"Experiment index to run (e.g. 0,1,...216)"}"
# expand the following env vars to absolute paths instead of paths to relative to
# the working directory since we'll be changing the working directory
export DMS_reference_file_path_subs="$(cd "$(dirname "$DMS_reference_file_path_subs")" && pwd)/$(basename "$DMS_reference_file_path_subs")"
export DMS_data_folder_subs="$(cd "$(dirname "$DMS_data_folder_subs")" && pwd)/$(basename "$DMS_data_folder_subs")"

cd ../../proteingym/baselines/PoET-2 && pixi run --frozen \
    python scripts/score.py \
    --checkpoint $checkpoint \
    --DMS_reference_file_path $DMS_reference_file_path_subs \
    --DMS_data_folder $DMS_data_folder_subs \
    --DMS_structure_folder $DMS_structure_folder \
    --DMS_index $DMS_index \
    --output_scores_folder $output_scores_folder \
    --MSA_folder $msa_folder \
    --AF2_cache_folder $AF2_cache_folder \
    --relative_to_wt
