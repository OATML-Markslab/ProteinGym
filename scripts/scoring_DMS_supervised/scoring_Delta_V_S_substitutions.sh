#!/bin/bash

source ../zero_shot_config.sh

# Delta V-s: supervised ensemble over six supervised baseline score sets
# (Kermut, ProteinNPT, MSA Transformer, Tranception, ESM-1v embeddings,
# DeepSequence one-hot). No checkpoints needed.
#
# One-time setup: build the database from the official supervised scores
# download (see proteingym/baselines/Delta_V_S/setup.sh):
#   bash ../../../proteingym/baselines/Delta_V_S/setup.sh
#
# The output files land under ${DMS_output_score_folder_subs}/Delta_V_S/<fold_scheme>/
# and are drop-in inputs for merge_supervised.py (config entry "Delta-V-s").

export Delta_V_db_path="${PROTEINGYM_CACHE}/Delta_V_S/Delta_V_S.db"
export supervised_scores_folder="${PROTEINGYM_CACHE}/supervised_scores/DMS_supervised_substitutions_scores"
export output_scores_folder="${DMS_output_score_folder_subs}/Delta_V_S"

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

for fold_scheme in fold_random_5 fold_modulo_5 fold_contiguous_5; do
    python ../../../proteingym/baselines/Delta_V_S/compute_fitness.py \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --fold_scheme $fold_scheme \
                --supervised_scores_folder ${supervised_scores_folder} \
                --output_scores_folder ${output_scores_folder} \
                --Delta_V_db_path ${Delta_V_db_path}
done
