#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR=$SCRIPT_DIR/data

if [ ! -f "$DATA_DIR/DMS_ProteinGym_substitutions.zip" ]; then
    wget https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip -P "$DATA_DIR"
fi
if [ ! -f "$DATA_DIR/DMS_substitutions.csv" ]; then
    wget https://marks.hms.harvard.edu/proteingym/DMS_substitutions.csv -P "$DATA_DIR"
fi
if [ ! -d "$DATA_DIR/DMS_ProteinGym_substitutions" ]; then
    unzip "$DATA_DIR/DMS_ProteinGym_substitutions.zip" -d "$DATA_DIR"
fi
