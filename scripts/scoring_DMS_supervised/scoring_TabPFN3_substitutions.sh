#!/usr/bin/env bash
set -euo pipefail

# ProteinGym supervised-substitutions scoring example for TabPFN3.
# This scores one assay and one official CV scheme. Run once per assay/scheme,
# or call it from your own array/manifest launcher.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${CSV:?Set CSV to one ProteinGym CV CSV, e.g. /path/to/cv/A0A140D2T1_ZIKV_Sourisseau_2019.csv}"
: "${FEATURE_ROOT:?Set FEATURE_ROOT to the directory containing <assay>/X_float16.dat matrices}"

OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/tabpfn3_scores}"
FOLD_COLUMN="${FOLD_COLUMN:-fold_random_5}"
DEVICE="${DEVICE:-auto}"
N_ESTIMATORS="${N_ESTIMATORS:-8}"
N_PREPROCESSING_JOBS="${N_PREPROCESSING_JOBS:-1}"

args=(
  "${REPO_DIR}/proteingym/baselines/tabpfn3/compute_fitness.py"
  --csv "${CSV}" \
  --out-dir "${OUT_DIR}" \
  --fold-column "${FOLD_COLUMN}" \
  --feature-root "${FEATURE_ROOT}" \
  --feature-source memmap \
  --device "${DEVICE}" \
  --n-estimators "${N_ESTIMATORS}" \
  --fit-mode fit_preprocessors \
  --memory-saving-mode auto \
  --inference-precision auto \
  --n-preprocessing-jobs "${N_PREPROCESSING_JOBS}" \
  --ignore-pretraining-limits
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  args+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${PREDICTION_CHUNK_SIZE:-}" ]]; then
  args+=(--prediction-chunk-size "${PREDICTION_CHUNK_SIZE}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--force)
fi

python "${args[@]}"
