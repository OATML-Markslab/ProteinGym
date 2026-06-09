#!/usr/bin/env bash
# Produce ESMC-TabICL score files for the full substitution benchmark.
# Output layout: $OUTPUT_DIR/<cv_scheme>/<DMS_id>.csv  (columns: mutant, y, y_pred, fold)
set -e
DMS_REFERENCE=../../../reference_files/DMS_substitutions.csv
DMS_FOLDER=${DMS_FOLDER:?set DMS_FOLDER to the cv_folds_singles_substitutions directory}
OUTPUT_DIR=${OUTPUT_DIR:-./scores}

python compute_scores.py \
    --dms_reference "$DMS_REFERENCE" \
    --dms_folder    "$DMS_FOLDER" \
    --output_dir    "$OUTPUT_DIR" \
    --device        cuda:0
