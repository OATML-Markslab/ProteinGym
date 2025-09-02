#!/bin/bash

source ../zero_shot_config.sh

export checkpoint="${checkpoint:-${PROTEINGYM_CACHE}/baselines/PoET-2/poet-2.ckpt}"
export output_folder_override="${output_folder_override:-PoET-2}"
export output_scores_folder="${clinical_output_score_folder_subs}${output_folder_override}"
export msa_folder=${PROTEINGYM_CACHE}/baselines/PoET/msas/clinical_substitutions
export AF2_cache_folder_override="${AF2_cache_folder_override:-clinical_AF2_structures_cache}"
export AF2_cache_folder="${PROTEINGYM_CACHE}/baselines/PoET-2/${AF2_cache_folder_override}"
export DMS_index="${DMS_index:-"variant index to run (e.g. 0,1,...2524)"}"
# expand the following env vars to absolute paths instead of paths to relative to
# the working directory since we'll be changing the working directory
export clinical_reference_file_path_subs="$(cd "$(dirname "$clinical_reference_file_path_subs")" && pwd)/$(basename "$clinical_reference_file_path_subs")"
export clinical_data_folder_subs="$(cd "$(dirname "$clinical_data_folder_subs")" && pwd)/$(basename "$clinical_data_folder_subs")"

cd ../../proteingym/baselines/PoET-2 && pixi run --frozen \
    python scripts/score.py \
    --checkpoint $checkpoint \
    --DMS_reference_file_path $clinical_reference_file_path_subs \
    --DMS_data_folder $clinical_data_folder_subs \
    --DMS_index $DMS_index \
    --output_scores_folder $output_scores_folder \
    --MSA_folder $msa_folder \
    --AF2_cache_folder $AF2_cache_folder \
    --theta 0.2 0.8 \
    --context_length 98304 98304 98304 98304 98304 98304 98304 98304 98304 98304 \
    --max_similarity 0.95 \
    --structure_in_context 0 1 \
    --inverse_folding_query 0 \
    --relative_to_wt \
    --batch_size 32
