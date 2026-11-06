#!/usr/bin/env bash
set -euo pipefail

# Run from the ProteinGym repository root. Inputs should be the paper archives.
: "${DMS_INPUT_DIR:?Set the official substitution assay directory}"
: "${MSA_DIR:?Set the paper benchmark MSA directory}"
: "${MSA_WEIGHTS_DIR:?Set the original sequence-weight directory}"
: "${MODEL_REVISION:?Set the immutable yakiyama/MSA-Pairformer checkpoint revision}"
: "${SCORE_OUTPUT_DIR:?Set the output directory}"

python proteingym/baselines/msa_pairformer/compute_fitness.py \
    --dms_mapping reference_files/DMS_substitutions.csv \
    --dms_index "${DMS_INDEX:-0}" \
    --dms-input "$DMS_INPUT_DIR" \
    --dms-output "$SCORE_OUTPUT_DIR" \
    --msa-path "$MSA_DIR" \
    --msa-weights-folder "$MSA_WEIGHTS_DIR" \
    --model-revision "$MODEL_REVISION" \
    --msa-samples 4096 \
    --seed 0 \
    --numpy-seed 42 \
    --num-cpus "${NUM_CPUS:-1}" \
    --device cuda:0
