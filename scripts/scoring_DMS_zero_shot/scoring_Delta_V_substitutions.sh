#!/bin/bash

source ../zero_shot_config.sh

# Delta V: pure-Python ensemble over five baseline score sets
# (VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME). No checkpoints needed.
#
# One-time setup: build the Delta V database from the official
# zero-shot substitution scores download (see baselines/Delta_V/README.md):
#   python ../../proteingym/baselines/Delta_V/build_delta_v_db.py \
#       --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
#       --model_scores_folder ${DMS_output_score_folder_subs} \
#       --output_db ${PROTEINGYM_CACHE}/Delta_V.db

export Delta_V_db_path="${PROTEINGYM_CACHE}/Delta_V.db"
export output_scores_folder="${DMS_output_score_folder_subs}/Delta_V"

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../../proteingym/baselines/Delta_V/compute_fitness.py \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \
            --MSA_data_folder ${DMS_MSA_data_folder} \
            --Delta_V_db_path ${Delta_V_db_path}
