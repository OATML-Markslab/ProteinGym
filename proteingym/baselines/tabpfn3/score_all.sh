#!/usr/bin/env bash
set -euo pipefail

# Example launcher for scoring all available supervised DMS substitution assays
# and fold schemes with TabPFN3. Run from the ProteinGym repository root.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${DMS_FOLDER:?Set DMS_FOLDER to the directory containing ProteinGym supervised CV CSV files}"
: "${FEATURE_ROOT:?Set FEATURE_ROOT to the directory containing <assay>/X_float16.dat matrices}"

OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/tabpfn3_scores}"
SCORING_SCRIPT="${REPO_DIR}/scripts/scoring_DMS_supervised/scoring_TabPFN3_substitutions.sh"

python - "$DMS_FOLDER" <<'PY' | while IFS=$'\t' read -r csv fold_column; do
from pathlib import Path
import sys
import pandas as pd

dms_folder = Path(sys.argv[1])
fold_columns = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")

for csv_path in sorted(dms_folder.glob("*.csv")):
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    for fold_column in fold_columns:
        if fold_column in columns:
            print(f"{csv_path}\t{fold_column}")
PY
  echo "Scoring ${csv} (${fold_column})"
  CSV="${csv}" \
  FOLD_COLUMN="${fold_column}" \
  FEATURE_ROOT="${FEATURE_ROOT}" \
  OUT_DIR="${OUT_DIR}" \
  "${SCORING_SCRIPT}"
done
