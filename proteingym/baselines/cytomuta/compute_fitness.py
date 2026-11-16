"""Score mutants with cytomuta-preview.

The method is a four-expert fusion with fixed weights:

    pred = 0.00 * D + 0.40 * Kermut + 0.40 * AMK + 0.20 * X

`X` is a distilled encoder trained per assay with the ProteinGym fold column, and its predictions
are averaged over eight training seeds. `D` receives weight 0.00 in 14 of 15 (seed x fold)
combinations, so it is not required at scoring time.

Because the supervised benchmark refits every model per assay and per fold, "scoring" here means:
train X on the four training folds of the assay, predict the held-out fold, then combine with the
frozen Kermut and AMK predictions. The training script is `expert/esm_mut_head.py` in
https://github.com/CytoLex/AAKG-M ; this file only assembles the final prediction column.

Inputs required per assay (documented in data/README.md of the method repository):
  - Kermut OOF predictions        column `Kermut_common_std`
  - AMK OOF predictions           column `AMK_common`
  - ESM-650 site and mutant features (built from the wild-type sequence)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {"Kermut": 0.40, "AMK": 0.40, "X": 0.20}
N_SEEDS = 8


def zscore_within_assay(df, cols):
    for c in cols:
        g = df.groupby("assay")[c]
        df[c] = ((df[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_file", required=True, help="ProteinGym DMS csv (mutant, DMS_score, fold columns)")
    ap.add_argument("--expert_table", help="csv with Kermut_common_std and AMK_common (fusion mode)")
    ap.add_argument("--student_preds", help="csv of the 8-seed averaged X predictions (fusion mode)")
    ap.add_argument("--fused_preds",
                    help="csv of already-fused per-mutant predictions (passthrough mode); use this when "
                         "the expert tables are not redistributed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dms = pd.read_csv(a.dms_file)
    if "mutated_sequence" not in dms.columns:
        dms["mutated_sequence"] = dms["mutant"]

    if a.fused_preds:
        fused = pd.read_csv(a.fused_preds)
        df = dms.merge(fused[["mutated_sequence", "predictions_fitness"]],
                       on="mutated_sequence", how="left")
        missing = int(df["predictions_fitness"].isna().sum())
        if missing:
            raise SystemExit(f"{missing} mutants without a fused prediction")
        df[["mutant", "mutated_sequence", "predictions_fitness"]].to_csv(a.out, index=False)
        print(f"wrote {len(df)} predictions to {a.out} (passthrough)")
        return

    if not (a.expert_table and a.student_preds):
        raise SystemExit("fusion mode needs --expert_table and --student_preds; "
                         "or pass --fused_preds for passthrough mode")

    df = dms.merge(pd.read_csv(a.expert_table), on="mutated_sequence", how="left")
    df = df.merge(pd.read_csv(a.student_preds), on="mutated_sequence", how="left")
    df = df.rename(columns={"Kermut_common_std": "Kermut", "AMK_common": "AMK",
                            "pred": "X"})
    df["assay"] = a.dms_file
    df[["Kermut", "AMK", "X"]] = df[["Kermut", "AMK", "X"]].fillna(0.0)
    df = zscore_within_assay(df, ["Kermut", "AMK", "X"])

    df["predictions_fitness"] = sum(w * df[k] for k, w in WEIGHTS.items())
    df[["mutant", "mutated_sequence", "predictions_fitness"]].to_csv(a.out, index=False)
    print(f"wrote {len(df)} predictions to {a.out}")


if __name__ == "__main__":
    main()
