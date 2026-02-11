#!/bin/bash

set -euo pipefail

assay_name=$1
model_name=$2
DATA_DIR=$3
INDEX_FILE=$4
OUTPUTS_DIR=$5

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FASTA_DIR=$DATA_DIR/ProGen3_fasta

mkdir -p "$FASTA_DIR"

fasta_file_name=$FASTA_DIR/${assay_name}.fasta

CSV_PATH="$DATA_DIR/$assay_name.csv"

python3 "$SCRIPT_DIR"/csv_to_fasta.py \
    --csv-path "$CSV_PATH" \
    --fasta-path "$fasta_file_name"

if [ ! -f "$OUTPUTS_DIR/$assay_name.tmp" ]; then
    torchrun --nproc-per-node=gpu -m progen3.tools.score \
        --model-name "$model_name" \
        --fasta-path "$FASTA_DIR/$assay_name.fasta" \
        --output-path "$OUTPUTS_DIR/$assay_name.tmp" \
        --fsdp
else
    echo "Skipping scoring because $OUTPUTS_DIR/$assay_name.tmp already exists"
fi

