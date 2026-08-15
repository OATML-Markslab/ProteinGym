"""ProteinGym data access library for strategy scripts.

Provides read-only access to pre-computed SOTA model predictions, protein
structure data, and MSA statistics. The strategy script imports this module
to query the SQLite database without ever touching the DB path or SQLite
directly.

Usage:
    from proteingym_data import (
        get_model_scores,
        get_zeroshot_scores,
        get_residue_structure,
        get_protein_info,
    )

    scores = get_model_scores(protein_id, mutations)
    zs = get_zeroshot_scores(protein_id, mutations)
    structure = get_residue_structure(protein_id)
    info = get_protein_info(protein_id)

Fold schemes:
    The DB stores supervised predictions for three cross-validation schemes
    (fold_random_5, fold_modulo_5, fold_contiguous_5). The active fold is
    selected by passing fold_scheme=... or by setting the PROTEINGYM_FOLD_SCHEME
    environment variable (read at call time, default 'fold_random_5'). The
    eval harness sets the env var per fold so strategy code needs no changes.

Security:
    - All connections are opened read-only (mode=ro URI) so a strategy can
      never lock or mutate the database.
    - The database contains NO ground-truth labels (no DMS_score, no
      DMS_score_bin). Label leakage is impossible by construction.
"""

import os
import sqlite3

_DB_PATH = os.environ.get(
    "PROTEINGYM_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "proteingym_data.db"),
)

# Default fold scheme when neither the fold_scheme argument nor the
# PROTEINGYM_FOLD_SCHEME env var is set.
DEFAULT_FOLD_SCHEME = "fold_random_5"
VALID_FOLD_SCHEMES = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")


def _resolve_fold_scheme(fold_scheme):
    """Pick the effective fold scheme: explicit arg -> env var -> default."""
    if fold_scheme is None:
        fold_scheme = os.environ.get("PROTEINGYM_FOLD_SCHEME", DEFAULT_FOLD_SCHEME)
    if fold_scheme not in VALID_FOLD_SCHEMES:
        raise ValueError(
            f"Unknown fold_scheme {fold_scheme!r}; "
            f"expected one of {VALID_FOLD_SCHEMES}"
        )
    return fold_scheme


def _connect():
    """Open a read-only SQLite connection."""
    return sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)


def get_model_scores(protein_id, mutants=None, fold_scheme=None):
    """Get supervised model predictions for a protein's mutations under a CV fold.

    Args:
        protein_id: e.g. "ARGR_ECOLI_Tsuboyama_2023_1AOY"
        mutants: optional list of mutant strings (e.g. ["A10C", "A10D"]).
                 If None, returns all mutations for this protein.
        fold_scheme: one of 'fold_random_5', 'fold_modulo_5',
                     'fold_contiguous_5'. If None (the default), falls back to
                     the PROTEINGYM_FOLD_SCHEME env var, then to
                     'fold_random_5'. The eval harness sets the env var per
                     fold so strategy code that omits this argument picks up
                     the right fold automatically.

    Returns:
        dict: {mutant_string: {"kermut": float, "msa_transformer_emb": float,
                                "tranception_emb": float, "esm1v_emb": float,
                                "deepsequence_ohe": float, "trancepteve": float,
                                "proteinnpt": float,
                                "wt_aa": str, "mut_aa": str,
                                "delta_charge": float, "delta_volume": float,
                                "delta_hydro": float, "blosum62": int}}
        Returns empty dict if protein_id not found.
    """
    fold_scheme = _resolve_fold_scheme(fold_scheme)
    conn = _connect()
    try:
        cols = ("mutant, kermut, msa_transformer_emb, tranception_emb, esm1v_emb, "
                "deepsequence_ohe, trancepteve, proteinnpt, "
                "wt_aa, mut_aa, delta_charge, delta_volume, delta_hydro, blosum62")
        if mutants:
            SQLITE_MAX_VARS = 900
            mut_list = list(mutants)
            rows = []
            for i in range(0, len(mut_list), SQLITE_MAX_VARS):
                chunk = mut_list[i:i + SQLITE_MAX_VARS]
                placeholders = ",".join("?" * len(chunk))
                rows.extend(conn.execute(
                    f"SELECT {cols} FROM model_scores "
                    f"WHERE protein_id=? AND fold_scheme=? AND mutant IN ({placeholders})",
                    [protein_id, fold_scheme] + chunk,
                ).fetchall())
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM model_scores "
                "WHERE protein_id=? AND fold_scheme=?",
                [protein_id, fold_scheme],
            ).fetchall()
        return {
            r[0]: {"kermut": r[1], "msa_transformer_emb": r[2], "tranception_emb": r[3],
                    "esm1v_emb": r[4], "deepsequence_ohe": r[5],
                    "trancepteve": r[6], "proteinnpt": r[7],
                    "wt_aa": r[8], "mut_aa": r[9],
                    "delta_charge": r[10], "delta_volume": r[11],
                    "delta_hydro": r[12], "blosum62": r[13]}
            for r in rows
        }
    finally:
        conn.close()


def get_residue_structure(protein_id):
    """Get per-residue structure data for a protein.

    Returns:
        dict: {position: {"wt_aa": str, "asa": float, "rsa": float,
                           "burial_class": str}}
        Returns empty dict if protein_id not found or has no structure data.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT position, wt_aa, asa, rsa, burial_class FROM residue_structure "
            "WHERE protein_id=?",
            [protein_id],
        ).fetchall()
        return {
            r[0]: {"wt_aa": r[1], "asa": r[2], "rsa": r[3], "burial_class": r[4]}
            for r in rows
        }
    finally:
        conn.close()


def get_zeroshot_scores(protein_id, mutants=None):
    """Get zero-shot model predictions for a protein's mutations.

    These models were NOT trained on DMS data — they predict from sequence,
    structure, or MSA alone. Their prediction errors are uncorrelated with
    supervised models, providing genuinely orthogonal signal.

    Zero-shot scores are stored once per (protein_id, mutant) and replicated
    across all three fold_scheme rows. We query the default fold for
    deterministic results.

    Args:
        protein_id: e.g. "ARGR_ECOLI_Tsuboyama_2023_1AOY"
        mutants: optional list of mutant strings. If None, returns all.

    Returns:
        dict: {mutant_string: {"venus_rem": float, "s3f_msa": float,
                                "esm2_15b": float, "prosst_2048": float,
                                "gemme": float}}
        Returns empty dict if protein_id not found.
    """
    conn = _connect()
    try:
        cols = "mutant, venus_rem, s3f_msa, esm2_15b, prosst_2048, gemme"
        if mutants:
            SQLITE_MAX_VARS = 900
            mut_list = list(mutants)
            rows = []
            for i in range(0, len(mut_list), SQLITE_MAX_VARS):
                chunk = mut_list[i:i + SQLITE_MAX_VARS]
                placeholders = ",".join("?" * len(chunk))
                rows.extend(conn.execute(
                    f"SELECT {cols} FROM model_scores "
                    f"WHERE protein_id=? AND fold_scheme=? AND mutant IN ({placeholders})",
                    [protein_id, DEFAULT_FOLD_SCHEME] + chunk,
                ).fetchall())
        else:
            rows = conn.execute(
                f"SELECT {cols} FROM model_scores "
                "WHERE protein_id=? AND fold_scheme=?",
                [protein_id, DEFAULT_FOLD_SCHEME],
            ).fetchall()
        return {
            r[0]: {"venus_rem": r[1], "s3f_msa": r[2], "esm2_15b": r[3],
                    "prosst_2048": r[4], "gemme": r[5]}
            for r in rows
        }
    finally:
        conn.close()


def get_protein_info(protein_id):
    """Get metadata for a protein assay.

    Returns:
        dict with keys: uniprot_id, source_organism, molecule_name,
        selection_type, coarse_selection_type, seq_len, taxon,
        includes_multiple_mutants, total_mutants, single_mutants,
        multiple_mutants, msa_num_seqs, msa_n_eff, msa_neff_l,
        msa_perc_cov, msa_len, msa_bitscore, msa_theta,
        has_structure, pdb_file, etc.
        Returns empty dict if not found.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT * FROM protein_info WHERE protein_id=?",
            [protein_id],
        )
        row = cur.fetchone()
        if row:
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        return {}
    finally:
        conn.close()
