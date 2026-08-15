import os
import sys
import argparse

import numpy as np
import pandas as pd

"""
Delta V-s baseline scoring script for the ProteinGym supervised DMS
substitution benchmark.

Delta V-s (MISE-SC: Multi-model Integration with Structural Context and
Conservation) is a pure-Python supervised ensemble strategy. It combines the
cross-validated predictions of six supervised baselines (Kermut, ProteinNPT,
MSA Transformer embeddings, Tranception embeddings, ESM-1v embeddings,
DeepSequence one-hot) using per-mutation adaptive weights derived from
embedding similarity, structural context, and conservation signals.

No model training, no GPU, no checkpoints. The only runtime dependency is
the SQLite database built from the official ProteinGym supervised downloads
by build_supervised_db.py (see setup.sh for the full download-and-build flow).

Output per DMS assay (one CSV per assay, per CV fold scheme):
    <output_scores_folder>/<fold_scheme>/<DMS_id>.csv
with columns: mutant, predictions_fitness, labels_fitness

- predictions_fitness: per-assay z-scored ensemble predictions (label-free
  scale alignment onto the normalized-target scale; Spearman is invariant
  to this monotone transform).
- labels_fitness: the official normalized_targets for the assay (from the
  official supervised score download), matching the convention of other
  supervised baselines.

These files are drop-in inputs for proteingym/merge_supervised.py via the
config.json entry "Delta-V-s" (key: mutant).

Setup (once):
    bash setup.sh                       # downloads + builds Delta_V_S.db
Scoring (per DMS assay; loop via the bash script for all 217):
    python compute_fitness.py \
        --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
        --DMS_data_folder <DMS_ProteinGym_substitutions/> \
        --DMS_index 0 \
        --fold_scheme fold_random_5 \
        --supervised_scores_folder <DMS_supervised_substitutions_scores/> \
        --output_scores_folder <output/Delta_V_S/> \
        --Delta_V_db_path <Delta_V_S.db>
"""


def main():
    parser = argparse.ArgumentParser(description="Delta V-s supervised scoring")
    parser.add_argument("--DMS_reference_file_path", type=str, required=True,
                        help="Path to DMS reference file (DMS_substitutions.csv)")
    parser.add_argument("--DMS_data_folder", type=str, required=True,
                        help="Path to folder that contains all DMS datasets")
    parser.add_argument("--DMS_index", type=int, required=True,
                        help="Index of the DMS assay to score (0..216)")
    parser.add_argument("--fold_scheme", type=str, default="fold_random_5",
                        choices=["fold_random_5", "fold_modulo_5", "fold_contiguous_5"],
                        help="CV fold scheme to score under")
    parser.add_argument("--supervised_scores_folder", type=str, required=True,
                        help="Path to DMS_supervised_substitutions_scores/ "
                             "(official download; source of normalized_targets)")
    parser.add_argument("--output_scores_folder", type=str, default=None,
                        help="Folder to write model scores to "
                             "(files land under <folder>/<fold_scheme>/)")
    parser.add_argument("--Delta_V_db_path", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "Delta_V_S.db"),
                        help="Path to the Delta V-s SQLite database "
                             "(see build_supervised_db.py)")
    args = parser.parse_args()

    os.environ["PROTEINGYM_DB"] = args.Delta_V_db_path

    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)

    from delta_v_s import score_mutations  # noqa: E402
    from proteingym_data import _connect  # noqa: E402

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    print("Computing scores for: {} ({})".format(DMS_id, args.fold_scheme))

    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()

    DMS_data = pd.read_csv(os.path.join(args.DMS_data_folder, DMS_file_name),
                           low_memory=False)

    # Restrict to mutants with supervised predictions in the DB
    # (the official supervised benchmark scores single mutants; multi-mutants
    # have no baseline predictions and are excluded, matching the baselines)
    conn = _connect()
    rows = conn.execute(
        "SELECT DISTINCT mutant FROM model_scores WHERE protein_id=? AND kermut IS NOT NULL",
        [DMS_id]).fetchall()
    conn.close()
    db_mutants = {r[0] for r in rows if r[0]}
    mask = DMS_data["mutant"].astype(str).isin(db_mutants)
    mutations = DMS_data.loc[mask, "mutant"].astype(str).tolist()
    if not mutations:
        raise RuntimeError("no scorable mutants for {}".format(DMS_id))

    # Official normalized targets for this assay + fold (labels convention)
    fold_csv = os.path.join(args.supervised_scores_folder, args.fold_scheme,
                            DMS_id + ".csv")
    nt = {}
    if os.path.exists(fold_csv):
        fold_df = pd.read_csv(fold_csv, usecols=["mutant", "normalized_targets"])
        nt = dict(zip(fold_df["mutant"].astype(str),
                      fold_df["normalized_targets"]))

    # The strategy reads the fold scheme from the environment
    os.environ["PROTEINGYM_FOLD_SCHEME"] = args.fold_scheme

    predicted = score_mutations(
        sequences=None,
        protein_id=DMS_id,
        wild_type=target_seq,
        mutations=mutations,
        msa=None,
    )
    pred = np.asarray(predicted, dtype=np.float64)

    # Label-free per-assay scale alignment onto the normalized-target scale
    z = (pred - pred.mean()) / (pred.std() + 1e-10)

    out = pd.DataFrame({
        "mutant": mutations,
        "predictions_fitness": z,
        "labels_fitness": [nt.get(m, np.nan) for m in mutations],
    })
    out_dir = os.path.join(args.output_scores_folder, args.fold_scheme)
    os.makedirs(out_dir, exist_ok=True)
    scoring_filename = os.path.join(out_dir, DMS_id + ".csv")
    out.to_csv(scoring_filename, index=False)
    print("Wrote {} ({} mutants)".format(scoring_filename, len(out)))


if __name__ == "__main__":
    main()
