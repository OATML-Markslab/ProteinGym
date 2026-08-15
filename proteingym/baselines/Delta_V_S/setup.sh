#!/usr/bin/env bash
# Delta V-s: Setup script
#
# Downloads ProteinGym data and builds the SQLite database needed
# to run the supervised strategy. Idempotent — re-running skips
# already-downloaded files and rebuilds the DB.
#
# Data downloaded:
#   - DMS_substitutions.csv (small) — reference metadata (from GitHub)
#   - DMS_ProteinGym_substitutions.zip (1 GB) — DMS assay files
#   - DMS_supervised_substitutions_scores.zip (3.3 GB) — supervised model scores
#   - AlphaFold DB structures (optional) — via download_structures.py + compute_asa.py
#
# Output:
#   data/Delta_V_S.db (~2.7 GB)
#
# Usage:
#   bash setup.sh              # full setup
#   bash setup.sh --build-db   # rebuild DB from existing downloads
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
BUILD_ONLY=false
case "${1:-}" in
    --build-db) BUILD_ONLY=true ;;
esac

PG_VERSION="v1.3"
PG_BASE="https://marks.hms.harvard.edu/proteingym/ProteinGym_${PG_VERSION}"

echo "============================================"
echo "   Delta V-s Setup"
echo "============================================"
echo ""
echo "DATA_DIR:  $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

download() {
    local url="$1" dest="$2" name="$3"
    if [[ -f "$dest" ]]; then
        ok "$name (already downloaded, $(du -sh "$dest" | cut -f1))"
        return 0
    fi
    echo "  Downloading $name..."
    if curl -fSL --connect-timeout 30 --retry 3 --retry-delay 5 \
        -o "$dest.part" "$url" && mv "$dest.part" "$dest"; then
        ok "$name ($(du -sh "$dest" | cut -f1))"
    else
        rm -f "$dest.part"
        fail "$name download failed"
        return 1
    fi
}

if [[ "$BUILD_ONLY" == false ]]; then

echo ""
echo "--- Step 1/4: Reference metadata ---"
REF_FILE="$DATA_DIR/DMS_substitutions.csv"
if [[ -f "$REF_FILE" ]]; then
    ok "DMS_substitutions.csv"
else
    download "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/DMS_substitutions.csv" \
        "$REF_FILE" "DMS_substitutions.csv"
fi

echo ""
echo "--- Step 2/4: DMS assay files (1 GB) ---"
DMS_ZIP="$DATA_DIR/DMS_ProteinGym_substitutions.zip"
DMS_DIR="$DATA_DIR/DMS_ProteinGym_substitutions"
download "$PG_BASE/DMS_ProteinGym_substitutions.zip" "$DMS_ZIP" "DMS substitution assays"
if [[ ! -d "$DMS_DIR" ]]; then
    echo "  Extracting..."
    (cd "$DATA_DIR" && unzip -qo "$DMS_ZIP" && rm "$DMS_ZIP") || warn "Extraction may have partially failed — continuing"
    ok "Extracted to DMS_ProteinGym_substitutions/"
else
    ok "DMS_ProteinGym_substitutions/ (already extracted)"
fi

echo ""
echo "--- Step 3/4: Supervised model scores (3.3 GB) ---"
SCORES_ZIP="$DATA_DIR/DMS_supervised_substitutions_scores.zip"
SCORES_DIR="$DATA_DIR/supervised_scores/DMS_supervised_substitutions_scores"
download "$PG_BASE/DMS_supervised_substitutions_scores.zip" "$SCORES_ZIP" "Supervised scores"
if [[ ! -d "$SCORES_DIR/fold_random_5" ]]; then
    echo "  Extracting (this may take a few minutes)..."
    mkdir -p "$DATA_DIR/supervised_scores"
    (cd "$DATA_DIR/supervised_scores" && unzip -qo "$SCORES_ZIP" && rm "$SCORES_ZIP") || warn "Extraction may have partially failed — continuing"
    ok "Extracted to supervised_scores/"
else
    ok "supervised_scores/ (already extracted)"
fi

echo ""
echo "--- Step 3b/4: Structure data (optional, ~1-2 GB from AlphaFold DB) ---"
STRUCT_DB="$DATA_DIR/protein_structures.db"
if [[ -f "$STRUCT_DB" ]]; then
    ok "protein_structures.db (already built)"
else
    pip3 show biopython >/dev/null 2>&1 || pip3 install biopython
    python3 "$SCRIPT_DIR/download_structures.py" \
        --DMS_reference_file_path "$REF_FILE" \
        --pdb_cache "$DATA_DIR/pdb_cache" \
        && python3 "$SCRIPT_DIR/compute_asa.py" \
        --DMS_reference_file_path "$REF_FILE" \
        --pdb_cache "$DATA_DIR/pdb_cache" \
        --output_db "$STRUCT_DB" \
        && ok "protein_structures.db built" \
        || warn "Structure build failed — the strategy works without structure data (MSA-derived proxies)"
fi

fi  # end !BUILD_ONLY

echo ""
echo "--- Step 4/4: Build Delta_V_S.db ---"
DB_PATH="$DATA_DIR/Delta_V_S.db"
[[ -f "$DB_PATH" ]] && rm -f "$DB_PATH"

STRUCT_ARG=""
[[ -f "$DATA_DIR/protein_structures.db" ]] && STRUCT_ARG="--structure_db $DATA_DIR/protein_structures.db"

if python3 "$SCRIPT_DIR/build_supervised_db.py" \
    --DMS_reference_file_path "$REF_FILE" \
    --supervised_scores_folder "$SCORES_DIR" \
    $STRUCT_ARG \
    --output_db "$DB_PATH"; then
    ok "Database built: $DB_PATH ($(du -sh "$DB_PATH" | cut -f1))"
else
    fail "Database build failed"
    exit 1
fi

echo ""
echo "============================================"
echo "   Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  Score:    bash ../../scripts/scoring_DMS_supervised/scoring_Delta_V_S_substitutions.sh"
echo "            (edit DMS_index / fold loop inside, or call compute_fitness.py directly)"
echo "  Merge:    python ../../../proteingym/merge_supervised.py"
echo "            --DMS_assays_location <DMS_ProteinGym_substitutions/>"
echo "            --model_scores_location <output scores parent folder>"
echo "            --merged_scores_dir <merged output folder>"
echo "            --mutation_type substitutions"
echo "            --DMS_reference_file <DMS_substitutions.csv>"
echo "            --config_file ../../config.json"
echo "  Metrics:  scripts/scoring_DMS_supervised/performance_substitutions.sh"
echo "            (see repo README; pass --top_model Kermut)"
