import os
import sys
import argparse
import random

import numpy as np
import pandas as pd

"""
Delta V baseline scoring script for ProteinGym DMS substitution benchmarks.

Delta V is a pure-Python ensemble strategy (CPCWE: Constraint-Propagated
Confidence-Weighted Ensemble) discovered by autonomous LLM-driven code
search. It combines the pre-computed predictions of five ProteinGym
baselines (VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME) with MSA
conservation and structural context. No model training, no GPU, no
checkpoints — the only runtime dependency is the SQLite database built
from the official ProteinGym downloads by build_delta_v_db.py.

Setup (once):
    python build_delta_v_db.py \
        --DMS_reference_file_path <DMS_substitutions.csv> \
        --model_scores_folder <zero_shot_substitutions_scores/> \
        --structure_db <protein_structures.db> \
        --output_db <Delta_V.db>

Scoring (per DMS assay, mirrors other ProteinGym baseline scripts):
    python compute_fitness.py \
        --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
        --DMS_data_folder <DMS_ProteinGym_substitutions/> \
        --DMS_index 0 \
        --output_scores_folder <output/Delta_V/> \
        --MSA_data_folder <DMS_msa_files/> \
        --Delta_V_db_path <Delta_V.db>
"""

# ── Deterministic MSA subsampling (matches the strategy's evaluation setup) ──
MSA_SUBSAMPLE_SEED = 42
MSA_MIN_DEPTH = 500
MSA_DEPTH_FACTOR = 10  # sample_n = max(500, min(total, 10 x protein_length))


def compute_msa_sample_n(total_seqs, protein_length):
    """Length-scaled MSA subsampling count."""
    return max(MSA_MIN_DEPTH, min(total_seqs, MSA_DEPTH_FACTOR * protein_length))


def load_msa(msa_path, protein_length=None):
    """Load an MSA fasta, subsampled to a length-scaled cap.

    Reservoir sampling with a fixed seed keeps the subset deterministic
    across runs. Spearman delta vs. full MSA is <0.002 across proteins
    up to 1.9M sequences.
    """
    if not msa_path or not os.path.exists(msa_path):
        return None

    total_seqs = 0
    with open(msa_path) as f:
        for line in f:
            if line.startswith(">"):
                total_seqs += 1
    if total_seqs == 0:
        return None

    sample_n = compute_msa_sample_n(total_seqs, protein_length or 1000)

    if total_seqs <= sample_n:
        seqs, seq = [], ""
        with open(msa_path) as f:
            for line in f:
                if line.startswith(">"):
                    if seq:
                        seqs.append(seq)
                    seq = ""
                else:
                    seq += line.strip()
        if seq:
            seqs.append(seq)
        return seqs if seqs else None

    rng = random.Random(MSA_SUBSAMPLE_SEED)
    keep = set(rng.sample(range(total_seqs), sample_n))

    seqs, seq, idx, in_keep = [], "", 0, False
    with open(msa_path) as f:
        for line in f:
            if line.startswith(">"):
                if seq and in_keep:
                    seqs.append(seq)
                seq = ""
                in_keep = idx in keep
                idx += 1
            else:
                if in_keep:
                    seq += line.strip()
    if seq and in_keep:
        seqs.append(seq)
    return seqs if seqs else None


def main():
    parser = argparse.ArgumentParser(description="Delta V scoring")
    parser.add_argument("--DMS_reference_file_path", type=str,
                        required=True, help="Path to DMS reference file (DMS_substitutions.csv)")
    parser.add_argument("--DMS_data_folder", type=str, required=True,
                        help="Path to folder that contains all DMS datasets")
    parser.add_argument("--DMS_index", type=int, required=True,
                        help="Index of the DMS assay to score (0..216)")
    parser.add_argument("--output_scores_folder", type=str, default=None,
                        help="Folder to write model scores to")
    parser.add_argument("--MSA_data_folder", type=str, default=None,
                        help="Path to folder containing DMS MSA files")
    parser.add_argument("--Delta_V_db_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Delta_V.db"),
                        help="Path to the Delta_V SQLite database (see build_delta_v_db.py)")
    args = parser.parse_args()

    # The strategy module reads the DB location from the environment
    os.environ["PROTEINGYM_DB"] = args.Delta_V_db_path

    # Make sibling modules importable (delta_v_strategy imports proteingym_data)
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)

    from delta_v_strategy import score_mutations  # noqa: E402

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    print("Computing scores for: {}".format(DMS_id))

    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()
    MSA_filename = mapping_protein_seq_DMS["MSA_filename"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]

    DMS_data = pd.read_csv(os.path.join(args.DMS_data_folder, DMS_file_name),
                           low_memory=False)
    mutations = DMS_data["mutant"].astype(str).tolist()

    msa = None
    if args.MSA_data_folder:
        msa_path = os.path.join(args.MSA_data_folder, MSA_filename)
        msa = load_msa(msa_path, protein_length=len(target_seq))

    model_scores = score_mutations(
        sequences={DMS_id: target_seq},
        protein_id=DMS_id,
        wild_type=target_seq,
        mutations=mutations,
        msa=msa,
    )
    model_scores = np.asarray(model_scores, dtype=np.float64)

    DMS_data["Delta_V_score"] = model_scores
    os.makedirs(args.output_scores_folder, exist_ok=True)
    scoring_filename = os.path.join(args.output_scores_folder, DMS_id + ".csv")
    DMS_data[["mutant", "Delta_V_score", "DMS_score"]].to_csv(scoring_filename, index=False)
    print("Wrote {}".format(scoring_filename))


if __name__ == "__main__":
    main()
