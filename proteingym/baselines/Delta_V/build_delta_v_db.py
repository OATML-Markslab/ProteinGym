#!/usr/bin/env python3
"""Build the Delta V SQLite database from official ProteinGym downloads.

Constructs a single SQLite database giving the Delta V strategy access to
pre-computed baseline model predictions, per-residue structure data, and
protein metadata - without any ground-truth labels.

Sources:
    - DMS_substitutions.csv (repo reference file)      -> protein_info
    - protein_structures.db (optional, see README)     -> residue_structure
    - zero_shot_substitutions_scores/<model>/*.csv     -> model_scores
      (official download; five columns are extracted:
       VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME)

Idempotent: safe to re-run. Existing tables are dropped and rebuilt.

NO DMS_score, NO DMS_score_bin, NO mutated_sequence columns anywhere -
label leakage is impossible by construction.

Usage:
    python3 build_delta_v_db.py \
        --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
        --model_scores_folder ~/.cache/ProteinGym/zero_shot_substitutions_scores/ \
        --structure_db protein_structures.db \
        --output_db Delta_V.db
"""

import csv
import glob
import os
import sqlite3
import sys
import time

# ── Paths ──────────────────────────────────────────────────────────────────
import argparse

_parser = argparse.ArgumentParser(description="Build the Delta V database")
_parser.add_argument("--DMS_reference_file_path", required=True,
                     help="Path to DMS_substitutions.csv (repo reference file)")
_parser.add_argument("--model_scores_folder", required=True,
                     help="Folder of per-assay merged score CSVs (one per DMS assay, each "
                          "containing a 'mutant' column and baseline score columns including "
                          "VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME). This is the "
                          "format of the official zero_shot_substitutions_scores download "
                          "and of the output of scripts/scoring_DMS_zero_shot/merge_all_scores.sh. "
                          "Only the five input columns are extracted; label columns "
                          "(DMS_score, DMS_score_bin) are ignored.")
_parser.add_argument("--structure_db", default=None,
                     help="Path to protein_structures.db (built via download_structures.py "
                          "+ compute_asa.py). Optional - the strategy falls back to "
                          "MSA-derived structural proxies when absent.")
_parser.add_argument("--output_db", default=None,
                     help="Output database path (default: Delta_V.db next to this script)")
_args = _parser.parse_args()

REFERENCE_FILE = _args.DMS_reference_file_path
STRUCTURE_DB = _args.structure_db
SCORES_DIR = _args.model_scores_folder
DB_PATH = _args.output_db or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Delta_V.db")

# The five model columns to extract from each scores CSV. Selected for
# maximum diversity of signal.
SCORE_COLUMNS = ["VenusREM", "S3F_MSA", "ESM2_15B", "ProSST-2048", "GEMME"]

# Amino acid physicochemical properties
AA_PROPS = {
    'A': {'charge': 0,    'volume': 88.6,  'hydro': 1.8},
    'R': {'charge': 1,    'volume': 173.4, 'hydro': -4.5},
    'N': {'charge': 0,    'volume': 114.1, 'hydro': -3.5},
    'D': {'charge': -1,   'volume': 111.1, 'hydro': -3.5},
    'C': {'charge': 0,    'volume': 108.5, 'hydro': 2.5},
    'Q': {'charge': 0,    'volume': 143.8, 'hydro': -3.5},
    'E': {'charge': -1,   'volume': 138.4, 'hydro': -3.5},
    'G': {'charge': 0,    'volume': 60.1,  'hydro': -0.4},
    'H': {'charge': 0.5,  'volume': 153.2, 'hydro': -3.2},
    'I': {'charge': 0,    'volume': 166.7, 'hydro': 4.5},
    'L': {'charge': 0,    'volume': 163.8, 'hydro': 3.8},
    'K': {'charge': 1,    'volume': 168.6, 'hydro': -3.9},
    'M': {'charge': 0,    'volume': 162.9, 'hydro': 1.9},
    'F': {'charge': 0,    'volume': 189.4, 'hydro': 2.8},
    'P': {'charge': 0,    'volume': 112.7, 'hydro': -1.6},
    'S': {'charge': 0,    'volume': 89.0,  'hydro': -0.8},
    'T': {'charge': 0,    'volume': 116.1, 'hydro': -0.7},
    'W': {'charge': 0,    'volume': 226.2, 'hydro': -0.9},
    'Y': {'charge': 0,    'volume': 163.2, 'hydro': -1.3},
    'V': {'charge': 0,    'volume': 140.0, 'hydro': 4.2},
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


def log(msg):
    print(f"[build_db] {msg}", file=sys.stderr, flush=True)


def _to_int(value, default=0):
    """Best-effort int conversion; returns default on failure."""
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
    """Best-effort float conversion; returns default on failure/empty."""
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# ── Schema ─────────────────────────────────────────────────────────────────
def create_tables(conn):
    """Drop and recreate all tables. Ensures idempotent rebuilds."""
    conn.executescript(
        """
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
            mutant        TEXT,
            venus_rem     REAL,
            s3f_msa       REAL,
            esm2_15b      REAL,
            prosst_2048   REAL,
            gemme         REAL,
            wt_aa         TEXT,
            mut_aa        TEXT,
            delta_charge  REAL,
            delta_volume  REAL,
            delta_hydro   REAL,
            blosum62      INTEGER,
            PRIMARY KEY (protein_id, mutant),
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
        """
    )


def create_indexes(conn):
    """Create indexes after data is loaded (faster than pre-load)."""
    conn.executescript(
        """
        CREATE INDEX idx_model_scores_protein ON model_scores(protein_id);
        CREATE INDEX idx_model_scores_mutant  ON model_scores(protein_id, mutant);
        CREATE INDEX idx_residue_protein      ON residue_structure(protein_id);
        """
    )


# ── Loaders ────────────────────────────────────────────────────────────────
def load_structure_lookup():
    """Pull has_structure + pdb_file for every protein from the structure DB.

    Returns dict: protein_id -> {"has_structure": int, "pdb_file": str}
    """
    lookup = {}
    if STRUCTURE_DB is None or not os.path.exists(STRUCTURE_DB):
        if STRUCTURE_DB is not None:
            log(f"WARNING: structure DB not found at {STRUCTURE_DB} — "
                f"has_structure/pdb_file will be 0/NULL for all proteins")
        else:
            log("No structure DB provided (--structure_db) — "
                "has_structure/pdb_file will be 0/NULL for all proteins")
        return lookup
    conn = sqlite3.connect(f"file:{STRUCTURE_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT protein_id, has_structure, pdb_file FROM protein_info"
        ).fetchall()
        for protein_id, has_structure, pdb_file in rows:
            lookup[protein_id] = {
                "has_structure": _to_int(has_structure),
                "pdb_file": pdb_file,
            }
    finally:
        conn.close()
    log(f"Loaded structure metadata for {len(lookup)} proteins")
    return lookup


def build_protein_info(conn, structure_lookup):
    """Populate protein_info from DMS_substitutions.csv + structure DB."""
    with open(REFERENCE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            pid = r["DMS_id"]
            struct = structure_lookup.get(pid, {})
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
                struct.get("has_structure", 0),
                struct.get("pdb_file"),
            ))
    conn.executemany(
        "INSERT INTO protein_info VALUES (" + ",".join("?" * 21) + ")",
        rows,
    )
    log(f"protein_info: inserted {len(rows)} rows")
    return len(rows)


def build_residue_structure(conn):
    """Copy the asa table from protein_structures.db -> residue_structure."""
    if STRUCTURE_DB is None or not os.path.exists(STRUCTURE_DB):
        log("No structure DB provided — residue_structure left empty "
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
        "INSERT INTO residue_structure VALUES (" + ",".join("?" * 6) + ")",
        rows,
    )
    log(f"residue_structure: inserted {len(rows)} rows")
    return len(rows)


def build_model_scores(conn):
    """Load 5 model columns + physicochemical features from each scores CSV.

    Reads mutant, VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME.
    Computes wt_aa, mut_aa, delta_charge, delta_volume, delta_hydro, blosum62
    from the mutation code. DMS_score, DMS_score_bin, mutated_sequence
    are NEVER stored.
    """
    csv_files = sorted(glob.glob(os.path.join(SCORES_DIR, "*.csv")))
    if not csv_files:
        log(f"ERROR: no score CSVs found in {SCORES_DIR}")
        return 0

    total_rows = 0
    batch = []
    BATCH_SIZE = 10000
    db_cols = ("venus_rem", "s3f_msa", "esm2_15b", "prosst_2048", "gemme")

    for path in csv_files:
        protein_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_idx = {name: idx for idx, name in enumerate(header)}
            mutant_idx = col_idx.get("mutant")
            score_indices = {db_col: col_idx.get(csv_col)
                             for db_col, csv_col in zip(db_cols, SCORE_COLUMNS)}
            if mutant_idx is None:
                log(f"WARNING: {protein_id} has no 'mutant' column — skipped")
                continue

            missing = [c for c, i in score_indices.items() if i is None]
            if missing:
                log(f"WARNING: {protein_id} missing score columns {missing} "
                    f"— those will be NULL")

            for row in reader:
                if mutant_idx >= len(row):
                    continue
                mutant = row[mutant_idx]

                vals = []
                for db_col in db_cols:
                    idx = score_indices[db_col]
                    if idx is not None and idx < len(row):
                        vals.append(_to_float(row[idx]))
                    else:
                        vals.append(None)

                # Physicochemical features
                wt_aa, mut_aa = parse_mutation_aa(mutant)
                if wt_aa and mut_aa:
                    dc, dv, dh, bl = compute_physicochem(wt_aa, mut_aa)
                else:
                    wt_aa, mut_aa, dc, dv, dh, bl = None, None, None, None, None, None

                batch.append((protein_id, mutant, vals[0], vals[1], vals[2],
                              vals[3], vals[4], wt_aa, mut_aa, dc, dv, dh, bl))
                if len(batch) >= BATCH_SIZE:
                    conn.executemany(
                        "INSERT OR REPLACE INTO model_scores VALUES ("
                        + ",".join("?" * 13) + ")",
                        batch,
                    )
                    total_rows += len(batch)
                    batch = []

        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO model_scores VALUES ("
                + ",".join("?" * 13) + ")",
                batch,
            )
            total_rows += len(batch)
            batch = []
        log(f"  {protein_id}: done")

    log(f"model_scores: inserted {total_rows} rows across {len(csv_files)} files")
    return total_rows

# ── Verification ───────────────────────────────────────────────────────────
def verify(conn):
    """Sanity-check row counts and absence of label columns."""
    checks = []

    n_proteins = conn.execute("SELECT COUNT(*) FROM protein_info").fetchone()[0]
    checks.append(("protein_info has 217 rows", n_proteins == 217, n_proteins))

    n_scores = conn.execute("SELECT COUNT(*) FROM model_scores").fetchone()[0]
    checks.append(("model_scores has rows", n_scores > 0, n_scores))

    n_proteins_scored = conn.execute(
        "SELECT COUNT(DISTINCT protein_id) FROM model_scores"
    ).fetchone()[0]
    checks.append(("model_scores covers 217 proteins",
                   n_proteins_scored == 217, n_proteins_scored))

    n_residues = conn.execute(
        "SELECT COUNT(*) FROM residue_structure"
    ).fetchone()[0]
    if STRUCTURE_DB is None:
        log("NOTE: no structure DB provided — residue_structure empty by design "
            "(strategy falls back to MSA-derived structural proxies); skipping row check")
    else:
        checks.append(("residue_structure has rows", n_residues > 0, n_residues))

    # No label columns anywhere
    for table in ("protein_info", "model_scores", "residue_structure"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        for forbidden in ("DMS_score", "DMS_score_bin", "mutated_sequence"):
            checks.append((f"{table} has no {forbidden}",
                           forbidden not in cols, cols))

    ok = True
    for desc, passed, detail in checks:
        status = "OK" if passed else "FAIL"
        log(f"  verify [{status}] {desc} ({detail})")
        if not passed:
            ok = False
    return ok


def create_symlink():
    """Deprecated no-op kept for call-site compatibility."""
    return


def main():
    t0 = time.time()
    log(f"Building {DB_PATH}")
    log(f"  reference:   {REFERENCE_FILE}")
    log(f"  structure:   {STRUCTURE_DB if STRUCTURE_DB else '(none — MSA proxies will be used)'}")
    log(f"  scores dir:  {SCORES_DIR}")

    for label, path in (("reference", REFERENCE_FILE),
                        ("structure", STRUCTURE_DB),
                        ("scores dir", SCORES_DIR)):
        if path is None:
            continue  # optional input
        if label == "scores dir":
            if not os.path.isdir(path):
                log(f"FATAL: scores dir not found: {path}")
                sys.exit(1)
        elif not os.path.exists(path):
            log(f"FATAL: {label} not found: {path}")
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
        structure_lookup = load_structure_lookup()
        build_protein_info(conn, structure_lookup)
        build_residue_structure(conn)

        log("Loading model scores (this is the slow step)...")
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

    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
