#!/bin/bash

set -euo pipefail

assay_name=$1
model_name=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR=$SCRIPT_DIR/data
FASTA_DIR=$DATA_DIR/DMS_ProteinGym_substitutions_fasta
OUTPUTS_DIR=$SCRIPT_DIR/outputs/$model_name

mkdir -p "$FASTA_DIR"
mkdir -p "$OUTPUTS_DIR"

fasta_file_name=$FASTA_DIR/${assay_name}.fasta

if [ "$assay_name" = "all" ]; then
    CSV_PATH="$DATA_DIR/DMS_ProteinGym_substitutions"
else
    CSV_PATH="$DATA_DIR/DMS_ProteinGym_substitutions/$assay_name.csv"
fi

python3 "$SCRIPT_DIR"/csv_to_fasta.py \
    --csv-path "$CSV_PATH" \
    --fasta-path "$fasta_file_name"

if [ ! -f "$OUTPUTS_DIR/$assay_name.csv" ]; then
    torchrun --nproc-per-node=gpu -m progen3.tools.score \
        --model-name "$model_name" \
        --fasta-path "$FASTA_DIR/$assay_name.fasta" \
        --output-path "$OUTPUTS_DIR/$assay_name.csv" \
        --fsdp
else
    echo "Skipping scoring because $OUTPUTS_DIR/$assay_name.csv already exists"
fi

python3 "$SCRIPT_DIR"/score.py \
    --assays-dir "$DATA_DIR/DMS_ProteinGym_substitutions" \
    --outputs-dir "$OUTPUTS_DIR" \
    --index-file "$DATA_DIR/DMS_substitutions.csv" \
    $([ "$assay_name" = "all" ] && echo "--split-all-scores")
