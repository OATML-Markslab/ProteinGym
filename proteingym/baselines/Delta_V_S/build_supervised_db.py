#!/usr/bin/env python3
"""Build proteingym_data.db from ProteinGym supervised score CSVs.

Constructs a single SQLite database giving the strategy script access to
pre-computed supervised model predictions, per-residue structure data, and
MSA statistics — without any ground-truth labels.

Sources:
    - DMS_substitutions.csv (GitHub) -> protein_info (metadata + MSA stats)
    - DMS_supervised_substitutions_scores/<fold>/*.csv -> model_scores
    - ProteinGym_AF2_structures.zip -> residue_structure (computed from PDB)

Output:
    data/proteingym_data.db

Idempotent: safe to re-run. Existing tables are dropped and rebuilt.

NO DMS_score, NO DMS_score_bin, NO mutated_sequence columns anywhere —
label leakage is impossible by construction.

Usage:
    python3 build_supervised_db.py \
        --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
        --supervised_scores_folder <supervised_scores/DMS_supervised_substitutions_scores/> \
        [--structure_folder <ProteinGym_AF2_structures/>] \
        --output_db Delta_V_S.db
"""

import csv
import glob
import os
import sqlite3
import sys
import time

# ── Paths ──────────────────────────────────────────────────────────────────
import argparse

_parser = argparse.ArgumentParser(description="Build the Delta V-s database")
_parser.add_argument("--DMS_reference_file_path", required=True,
                     help="Path to DMS_substitutions.csv (repo reference file)")
_parser.add_argument("--supervised_scores_folder", required=True,
                     help="Path to DMS_supervised_substitutions_scores/ (official download), "
                          "containing one subfolder per CV fold scheme "
                          "(fold_random_5, fold_modulo_5, fold_contiguous_5), each with "
                          "217 per-assay CSVs of supervised model predictions")
_parser.add_argument("--structure_db", default=None,
                     help="Path to protein_structures.db (optional; built by "
                          "download_structures.py + compute_asa.py from AlphaFold DB "
                          "predictions, cropped to assay ranges). Provides per-residue "
                          "RSA features. The strategy falls back to MSA-derived "
                          "proxies when absent.")
_parser.add_argument("--output_db", default=None,
                     help="Output database path (default: Delta_V_S.db next to this script)")
_args = _parser.parse_args()

REFERENCE_FILE = _args.DMS_reference_file_path
SCORES_DIR = _args.supervised_scores_folder
STRUCTURE_DB = _args.structure_db
DB_PATH = _args.output_db or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Delta_V_S.db")

# The three official ProteinGym supervised CV fold schemes.
FOLD_SCHEMES = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")

# Seven supervised model columns extracted from each scores CSV.
# Six are used by the Delta V-s strategy; TranceptEVE is included so the
# data layer exposes the full set available in the official files.
SCORE_COLUMNS = [
    "Kermut_predictions",
    "Embeddings - Augmented - MSA Transformer_predictions",
    "Embeddings - Augmented - Tranception_predictions",
    "Embeddings - Augmented - ESM1v_predictions",
    "OHE - Augmented - DeepSequence_predictions",
    "OHE - Augmented - TranceptEVE_predictions",
    "ProteinNPT_predictions",
]

# DB column names (shorter, no spaces)
DB_COLUMNS = [
    "kermut",
    "msa_transformer_emb",
    "tranception_emb",
    "esm1v_emb",
    "deepsequence_ohe",
    "trancepteve",
    "proteinnpt",
]


# ── Physicochemical features ───────────────────────────────────────────────
AA_PROPS = {
    'A': {'charge': 0, 'volume': 88.6, 'hydro': 1.8},
    'R': {'charge': 1, 'volume': 173.4, 'hydro': -4.5},
    'N': {'charge': 0, 'volume': 114.1, 'hydro': -3.5},
    'D': {'charge': -1, 'volume': 111.1, 'hydro': -3.5},
    'C': {'charge': 0, 'volume': 108.5, 'hydro': 2.5},
    'Q': {'charge': 0, 'volume': 143.8, 'hydro': -3.5},
    'E': {'charge': -1, 'volume': 138.4, 'hydro': -3.5},
    'G': {'charge': 0, 'volume': 60.1, 'hydro': -0.4},
    'H': {'charge': 0.5, 'volume': 153.2, 'hydro': -3.2},
    'I': {'charge': 0, 'volume': 166.7, 'hydro': 4.5},
    'L': {'charge': 0, 'volume': 163.8, 'hydro': 3.8},
    'K': {'charge': 1, 'volume': 168.6, 'hydro': -3.9},
    'M': {'charge': 0, 'volume': 162.9, 'hydro': 1.9},
    'F': {'charge': 0, 'volume': 189.4, 'hydro': 2.8},
    'P': {'charge': 0, 'volume': 112.7, 'hydro': -1.6},
    'S': {'charge': 0, 'volume': 89.0, 'hydro': -0.8},
    'T': {'charge': 0, 'volume': 116.1, 'hydro': -0.7},
    'W': {'charge': 0, 'volume': 226.2, 'hydro': -0.9},
    'Y': {'charge': 0, 'volume': 163.2, 'hydro': -1.3},
    'V': {'charge': 0, 'volume': 140.0, 'hydro': 4.2},
}

_BLOSUM62_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
 A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
 R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
 N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
 D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
 Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
 E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
 H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
 I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
 L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
 K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
 M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
 F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3  0
 P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
 W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2  11  2 -3
 Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1  0 -2 -2  0 -3 -1  4
"""

def _build_blosum62():
    lines = _BLOSUM62_RAW.strip().split('\n')
    aas = lines[0].split()
    mat = {}
    for line in lines[1:]:
        vals = line.split()
        aa1 = vals[0]
        for j, aa2 in enumerate(aas):
            mat[(aa1, aa2)] = int(vals[j + 1])
    return mat

BLOSUM62 = _build_blosum62()


def log(msg):
    print(f"[build_db] {msg}", file=sys.stderr, flush=True)


def _to_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip()
    if s.upper() in ("TRUE", "T", "YES"):
        return 1
    if s.upper() in ("FALSE", "F", "NO", ""):
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def parse_mutation_aa(mutant):
    """Extract WT and mutant amino acids from mutation code like 'A10C'."""
    wt_aa = ''
    mut_aa = ''
    in_pos = False
    for ch in mutant:
        if ch.isalpha() and not in_pos:
            wt_aa += ch
        elif ch.isdigit():
            in_pos = True
        elif ch.isalpha() and in_pos:
            mut_aa += ch
    if len(wt_aa) == 1 and len(mut_aa) == 1:
        return wt_aa, mut_aa
    return None, None


def compute_physicochem(wt_aa, mut_aa):
    if wt_aa not in AA_PROPS or mut_aa not in AA_PROPS:
        return None, None, None, None
    wt_p = AA_PROPS[wt_aa]
    mt_p = AA_PROPS[mut_aa]
    return (mt_p['charge'] - wt_p['charge'],
            mt_p['volume'] - wt_p['volume'],
            mt_p['hydro'] - wt_p['hydro'],
            BLOSUM62.get((wt_aa, mut_aa), 0))


# ── Schema ─────────────────────────────────────────────────────────────────
def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS model_scores;
        DROP TABLE IF EXISTS residue_structure;
        DROP TABLE IF EXISTS protein_info;

        CREATE TABLE protein_info (
            protein_id                 TEXT PRIMARY KEY,
            uniprot_id                 TEXT,
            source_organism            TEXT,
            molecule_name              TEXT,
            selection_type             TEXT,
            coarse_selection_type      TEXT,
            seq_len                    INTEGER,
            taxon                      TEXT,
            includes_multiple_mutants  INTEGER,
            total_mutants              INTEGER,
            single_mutants             INTEGER,
            multiple_mutants           INTEGER,
            msa_num_seqs               INTEGER,
            msa_n_eff                  REAL,
            msa_neff_l                 REAL,
            msa_perc_cov               REAL,
            msa_len                    INTEGER,
            msa_bitscore               REAL,
            msa_theta                  REAL,
            has_structure              INTEGER,
            pdb_file                   TEXT
        );

        CREATE TABLE model_scores (
            protein_id    TEXT,
            fold_scheme   TEXT,
            mutant        TEXT,
            kermut        REAL,
            msa_transformer_emb REAL,
            tranception_emb    REAL,
            esm1v_emb          REAL,
            deepsequence_ohe    REAL,
            trancepteve        REAL,
            proteinnpt         REAL,
            wt_aa         TEXT,
            mut_aa        TEXT,
            delta_charge  REAL,
            delta_volume  REAL,
            delta_hydro   REAL,
            blosum62      INTEGER,
            PRIMARY KEY (protein_id, fold_scheme, mutant),
            FOREIGN KEY (protein_id) REFERENCES protein_info(protein_id)
        );

        CREATE TABLE residue_structure (
            protein_id    TEXT,
            position      INTEGER,
            wt_aa         TEXT,
            asa           REAL,
            rsa           REAL,
            burial_class  TEXT,
            PRIMARY KEY (protein_id, position),
            FOREIGN KEY (protein_id) REFERENCES protein_info(protein_id)
        );
    """)


def create_indexes(conn):
    conn.executescript("""
        CREATE INDEX idx_model_scores_protein ON model_scores(protein_id);
        CREATE INDEX idx_model_scores_fold ON model_scores(fold_scheme);
        CREATE INDEX idx_model_scores_protein_mutant ON model_scores(protein_id, mutant);
        CREATE INDEX idx_residue_protein ON residue_structure(protein_id);
    """)


# ── Loaders ────────────────────────────────────────────────────────────────
def build_protein_info(conn):
    """Populate protein_info from DMS_substitutions.csv."""
    with open(REFERENCE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            pid = r["DMS_id"]
            rows.append((
                pid,
                r.get("UniProt_ID"),
                r.get("source_organism"),
                r.get("molecule_name"),
                r.get("selection_type"),
                r.get("coarse_selection_type"),
                _to_int(r.get("seq_len")),
                r.get("taxon"),
                _to_int(r.get("includes_multiple_mutants")),
                _to_int(r.get("DMS_total_number_mutants")),
                _to_int(r.get("DMS_number_single_mutants")),
                _to_int(r.get("DMS_number_multiple_mutants")),
                _to_int(r.get("MSA_num_seqs")),
                _to_float(r.get("MSA_N_eff")),
                _to_float(r.get("MSA_Neff_L")),
                _to_float(r.get("MSA_perc_cov")),
                _to_int(r.get("MSA_len")),
                _to_float(r.get("MSA_bitscore")),
                _to_float(r.get("MSA_theta")),
                0,  # has_structure — updated below if structures exist
                None,  # pdb_file
            ))
    conn.executemany(
        "INSERT INTO protein_info VALUES (" + ",".join("?" * 21) + ")",
        rows,
    )
    log(f"protein_info: inserted {len(rows)} rows")
    return len(rows)


def update_structure_flags(conn):
    """Copy has_structure / pdb_file flags from protein_structures.db."""
    if STRUCTURE_DB is None or not os.path.exists(STRUCTURE_DB):
        log("No structure DB provided - has_structure/pdb_file stay 0/NULL "
            "(strategy falls back to MSA-derived structural proxies)")
        return
    src = sqlite3.connect(f"file:{STRUCTURE_DB}?mode=ro", uri=True)
    try:
        rows = src.execute(
            "SELECT protein_id, has_structure, pdb_file FROM protein_info"
        ).fetchall()
        cur = conn.cursor()
        n = 0
        for protein_id, has_structure, pdb_file in rows:
            cur.execute(
                "UPDATE protein_info SET has_structure=?, pdb_file=? WHERE protein_id=?",
                (int(has_structure or 0), pdb_file, protein_id))
            n += cur.rowcount
        conn.commit()
        log(f"structure flags updated for {n} proteins")
    finally:
        src.close()

def build_residue_structure(conn):
    """Copy the asa table from protein_structures.db -> residue_structure."""
    if STRUCTURE_DB is None or not os.path.exists(STRUCTURE_DB):
        log("No structure DB provided - residue_structure left empty "
            "(strategy falls back to MSA-derived structural proxies)")
        return 0
    src = sqlite3.connect(f"file:{STRUCTURE_DB}?mode=ro", uri=True)
    try:
        rows = src.execute(
            "SELECT protein_id, position, wt_aa, asa, rsa, burial_class FROM asa"
        ).fetchall()
    finally:
        src.close()
    conn.executemany(
        "INSERT OR REPLACE INTO residue_structure VALUES (?,?,?,?,?,?)",
        rows)
    conn.commit()
    log(f"residue_structure: inserted {len(rows)} rows")
    return len(rows)

def build_model_scores(conn):
    """Load supervised model scores from merged CSVs per fold scheme.

    For each fold directory, reads 217 protein CSVs and extracts the 6 model
    prediction columns. Ground-truth (DMS_score, DMS_score_bin, mutated_sequence)
    is NEVER stored.

    Scores with empty/missing values are stored as NULL.
    """
    total_rows = 0
    total_files = 0
    batch = []
    BATCH_SIZE = 10000

    for fold in FOLD_SCHEMES:
        fold_dir = os.path.join(SCORES_DIR, fold)
        if not os.path.isdir(fold_dir):
            log(f"WARNING: fold directory not found: {fold_dir}")
            continue

        csv_files = sorted(glob.glob(os.path.join(fold_dir, "*.csv")))
        log(f"Processing {fold}: {len(csv_files)} files")

        for path in csv_files:
            protein_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                col_idx = {name.strip(): idx for idx, name in enumerate(header)}

                mutant_idx = col_idx.get("mutant")
                if mutant_idx is None:
                    log(f"  SKIP {protein_id}: no 'mutant' column")
                    continue

                score_indices = []
                for db_col, csv_col in zip(DB_COLUMNS, SCORE_COLUMNS):
                    idx = col_idx.get(csv_col)
                    score_indices.append(idx)

                missing = [csv_col for db_col, csv_col, idx in
                           zip(DB_COLUMNS, SCORE_COLUMNS, score_indices)
                           if idx is None]
                if missing:
                    log(f"  WARN {protein_id}: missing columns {missing}")

                for row in reader:
                    if mutant_idx >= len(row):
                        continue
                    mutant = row[mutant_idx]

                    vals = []
                    for idx in score_indices:
                        if idx is not None and idx < len(row):
                            vals.append(_to_float(row[idx]))
                        else:
                            vals.append(None)

                    # Physicochemical features
                    wt_aa, mut_aa = parse_mutation_aa(mutant)
                    if wt_aa and mut_aa:
                        dc, dv, dh, bl = compute_physicochem(wt_aa, mut_aa)
                    else:
                        wt_aa, mut_aa = None, None
                        dc, dv, dh, bl = None, None, None, None

                    batch.append((
                        protein_id, fold, mutant,
                        *vals,
                        wt_aa, mut_aa, dc, dv, dh, bl,
                    ))
                    if len(batch) >= BATCH_SIZE:
                        conn.executemany(
                            "INSERT OR REPLACE INTO model_scores VALUES ("
                            + ",".join("?" * (3 + len(DB_COLUMNS) + 6)) + ")",
                            batch,
                        )
                        total_rows += len(batch)
                        batch = []

            total_files += 1

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO model_scores VALUES ("
            + ",".join("?" * (3 + len(DB_COLUMNS) + 6)) + ")",
            batch,
        )
        total_rows += len(batch)

    log(f"model_scores: inserted {total_rows} rows across {total_files} files")
    return total_rows


# ── Verification ───────────────────────────────────────────────────────────
def verify(conn):
    checks = []
    n_proteins = conn.execute("SELECT COUNT(*) FROM protein_info").fetchone()[0]
    checks.append(("protein_info has 217 rows", n_proteins == 217, n_proteins))

    n_scores = conn.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
    checks.append(("model_scores has rows", n_scores > 0, n_scores))

    n_proteins_scored = conn.execute(
        "SELECT COUNT(DISTINCT protein_id) FROM model_scores"
    ).fetchone()[0]
    checks.append(("model_scores covers >=200 proteins",
                   n_proteins_scored >= 200, n_proteins_scored))

    # Check fold schemes present
    folds = conn.execute(
        "SELECT DISTINCT fold_scheme FROM model_scores"
    ).fetchall()
    checks.append(("model_scores has 3 fold schemes",
                   len(folds) == 3, [f[0] for f in folds]))

    n_residues = conn.execute("SELECT COUNT(*) FROM residue_structure").fetchone()[0]
    checks.append(("residue_structure has rows (or empty if no biopython)",
                   True, n_residues))

    # No label columns
    for table in ("protein_info", "model_scores", "residue_structure"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        for forbidden in ("DMS_score", "DMS_score_bin", "mutated_sequence"):
            checks.append((f"{table} has no {forbidden}", forbidden not in cols, cols))

    # Check model columns exist
    score_cols = [r[1] for r in conn.execute("PRAGMA table_info(model_scores)").fetchall()]
    for col in DB_COLUMNS:
        checks.append((f"model_scores has column {col}", col in score_cols, score_cols))

    ok = True
    for desc, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        log(f"  verify [{status}] {desc} ({detail})")
        if not passed:
            ok = False
    return ok


def main():
    t0 = time.time()
    log(f"Building {DB_PATH}")
    log(f"  reference:   {REFERENCE_FILE}")
    log(f"  scores dir:  {SCORES_DIR}")
    log(f"  structure:   {STRUCTURE_DB if STRUCTURE_DB else '(none — MSA proxies will be used)'}")

    for label, path in (("reference", REFERENCE_FILE), ("scores dir", SCORES_DIR)):
        if label == "scores dir":
            if not os.path.isdir(path):
                log(f"FATAL: {label} not found: {path}")
                log("Run: python3 scripts/setup.py")
                sys.exit(1)
        elif not os.path.exists(path):
            log(f"FATAL: {label} not found: {path}")
            log("Run: python3 scripts/setup.py")
            sys.exit(1)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")

        create_tables(conn)
        build_protein_info(conn)
        update_structure_flags(conn)
        build_residue_structure(conn)

        log("Loading supervised model scores (this may take a few minutes)...")
        build_model_scores(conn)

        conn.commit()
        create_indexes(conn)
        conn.commit()
    finally:
        conn.close()

    log("Verifying...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        if not verify(conn):
            log("VERIFICATION FAILED")
            sys.exit(1)
    finally:
        conn.close()

    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    log(f"Done in {time.time() - t0:.1f}s ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
