#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compute_fitness.py — PTULyra_ytx baseline for ProteinGym zero-shot DMS substitutions.

This script follows the conventions of the existing baselines in this repository
(cf. ``proteingym/baselines/rita/compute_fitness.py`` and
``proteingym/baselines/venusrem/compute_fitness.py``): it writes **one CSV per assay**,

    <output_scores_folder>/<DMS_id>.csv     columns:  mutant, PTULyra_ytx_score, DMS_score

so that ``scripts/scoring_DMS_zero_shot/merge_all_scores.sh`` can pick the scores up.

External data
-------------
PTULyra_ytx is an ensemble over seven component signals. Six of them are shipped as
precomputed tables (prosst, venusrem, esmc600, esmc6b, gemme, msa1b) plus a
within-benchmark ``transfer`` component; together they are ~630 MB and are archived at

    https://doi.org/10.5281/zenodo.22818848            (Zenodo; see README.md in this folder)

Download and unpack that archive, then pass the ``signals/`` directory as ``--signals_dir``.
Nothing in this script runs a language model: scoring is CPU-only and needs only
numpy/pandas/scipy, so the released scores are reproducible without any GPU and without
downloading any model checkpoint.

Method
------
For a target assay ``T`` (see README.md for the full description):

1. each component is z-scored over the subset of ``T``'s mutants that it covers;
2. ``T``'s ProteinGym functional group selects a weight vector; for each mutant the output
   is a weighted average over the components available **at that mutant**, with the weights
   renormalised over that active subset — so no mutant is dropped when a component covers
   only part of the assay;
3. ``transfer`` is blended in with ``alpha = clip(pearson(z_transfer, z_base), 0, 1)``,
   computed from the two signal vectors alone, with no label fitting.

``transfer`` is **not** a model prediction: it is built from ProteinGym's own experimental
measurements (the ``DMS_score`` column of a sibling assay of the same UniProt). It applies
to 53 of the 217 assays. This is a deliberate, disclosed part of the method; see the
"Ablation" section of README.md for what removing it costs. When reporting PTULyra_ytx
results, please report it as-is rather than as a purely sequence-based zero-shot score.

Dependencies: numpy, pandas, scipy.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

GROUPS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]

# Three assays whose UniProt accession / organism was updated in ProteinGym v1.3. Our
# component tables are keyed by the older names, the benchmark by the newer ones. The
# mutant sets are identical under either name, so this is a pure relabelling.
#   key (ours, older)                     ->  value (ProteinGym v1.3)
ALIAS_OLD_TO_NEW = {
    "F7YBW7_MESOW_Ding_2023": "F7YBW8_MESOW_Ding_2023",
    "PSAE_SYNP2_Tsuboyama_2023_1PSE": "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "Q6WV13_9MAXI_Somermeyer_2022": "Q6WV12_9MAXI_Somermeyer_2022",
}
ALIAS_NEW_TO_OLD = {v: k for k, v in ALIAS_OLD_TO_NEW.items()}


# ────────────────────────────── loading ──────────────────────────────

def load_csv_dir(dirpath):
    """Read every CSV in a directory -> {DMS_id: DataFrame} (only the columns we need)."""
    cols = ("mutant", "DMS_score", "VenusREM")
    out = {}
    for fp in sorted(glob.glob(os.path.join(dirpath, "*.csv"))):
        did = os.path.basename(fp)[:-4]
        try:
            out[did] = pd.read_csv(fp, usecols=lambda c: c in cols,
                                   dtype={"mutant": str, "DMS_score": float, "VenusREM": float})
        except Exception:
            try:
                out[did] = pd.read_csv(fp)
            except Exception:
                continue
    return out


def csv_col_to_dict(df, col):
    """One column of a DataFrame -> {mutant: float}, skipping NaN."""
    if col not in df.columns:
        return {}
    return {m: float(v) for m, v in zip(df["mutant"].tolist(), df[col].tolist()) if v == v}


def load_json_signal(path):
    """Generic {DMS_id: {mutant: score}} JSON; missing file is a warning, not an error."""
    if not os.path.exists(path):
        print("  [WARN] signal table missing: %s (skipped)" % path, flush=True)
        return {}
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    for did, muts in d.items():
        if isinstance(muts, dict):
            s = {str(k): float(v) for k, v in muts.items() if v == v}
            if s:
                out[str(did)] = s
    return out


def zscore(v):
    mu, sd = v.mean(), v.std()
    return (v - mu) / sd if sd > 0 else v - mu


# ──────────────────────────── scoring core ────────────────────────────

def pred_did(did, combo_w, core_sigs, ymap, trans_did):
    """Per-mutant prediction for one assay over its full mutant set.

    Returns (scores, alpha, diagnostics). ``scores`` maps every mutant that has at least
    one active component to a score; ``ymap`` supplies the mutant set, which is the
    benchmark's own set for this assay (so no mutant is filtered out here).
    """
    muts = list(ymap)

    # 1) each component z-scored over the subset of the frame it covers
    zs = {}
    for s, sigd in core_sigs.items():
        if did not in sigd:
            continue
        d = sigd[did]
        sub = [m for m in muts if m in d]
        if len(sub) < 10:
            continue
        zv = zscore(np.array([d[m] for m in sub], dtype=float))
        zs[s] = {m: zv[k] for k, m in enumerate(sub)}
    if not zs:
        return {}, 0.0, {"reason": "no component covers this assay", "frame_n": len(muts)}

    w = {s: combo_w.get(s, 0.0) for s in zs}

    # 2) per-mutant weighted average, weights renormalised over the components active here
    zbase = {}
    for m in muts:
        act = [s for s in zs if m in zs[s]]
        wsum = sum(w[s] for s in act)
        if not act or wsum <= 0:
            continue
        zbase[m] = sum(w[s] / wsum * zs[s][m] for s in act)
    if len(zbase) < 10:
        return {}, 0.0, {"reason": "fewer than 10 mutants scored", "frame_n": len(muts)}

    # 3) alpha-blend with the transfer component
    alpha, blend = 0.0, dict(zbase)
    tset = {m: v for m, v in trans_did.items() if m in zbase}
    if len(tset) >= 10:
        tsub = sorted(tset)
        tz = zscore(np.array([tset[m] for m in tsub], dtype=float))
        zt = {m: tz[k] for k, m in enumerate(tsub)}
        r = pearsonr([zt[m] for m in tsub], [zbase[m] for m in tsub])[0]
        alpha = float(np.clip(0.0 if r != r else r, 0.0, 1.0))
        for m in tsub:
            blend[m] = alpha * zt[m] + (1.0 - alpha) * zbase[m]

    n_scored = len(blend)
    return ({m: float(v) for m, v in blend.items()}, alpha,
            {"frame_n": len(muts), "n_scored": n_scored,
             "coverage": n_scored / len(muts) if muts else 0.0,
             "n_components": len(zs)})


def spearman(y, x):
    if len(y) < 10:
        return float("nan")
    return float(spearmanr(y, x)[0])


# ─────────────────────────── official metric ───────────────────────────

def official_agg(rhos, groups, uniprots):
    """ProteinGym official aggregation: mean over mutants -> mean over assays sharing a
    UniProt -> equal-weight mean over the five functional groups.

    ``uniprots`` maps DMS_id -> UniProt_ID as given by the reference file. It is taken from
    the file rather than parsed out of the DMS_id: the DMS_id is not reliably
    ``<UniProt>_...`` (e.g. ``PSAE_PICP2_Tsuboyama_2023_1PSE`` would split to ``PSAE``).
    """
    g1 = {}
    for did, rho in rhos.items():
        if did not in groups or did not in uniprots or rho is None or rho != rho:
            continue
        g1.setdefault((uniprots[did], groups[did]), []).append(rho)
    g1m = {k: sum(v) / len(v) for k, v in g1.items()}
    g2 = {}
    for (uni, st), mm in g1m.items():
        g2.setdefault(st, []).append(mm)
    gm = {st: sum(v) / len(v) for st, v in g2.items()}
    vals = [gm[g] for g in GROUPS if g in gm]
    return gm, (sum(vals) / len(vals) if vals else float("nan"))


def dms_agg(rhos, groups):
    g = {}
    for did, rho in rhos.items():
        if did not in groups or rho is None or rho != rho:
            continue
        g.setdefault(groups[did], []).append(rho)
    vals = [sum(v) / len(v) for st, v in g.items() if st in GROUPS]
    return sum(vals) / len(vals) if vals else float("nan")


# ─────────────────────────────── driver ───────────────────────────────

def resolve_dms_file(dms_folder, dms_id, dms_filename):
    """Find the benchmark data file for an assay.

    ProteinGym v1.3 renamed three assays. The v1.3 release names the file after the new
    accession; older copies of the data folder (and our component tables) use the older
    name. Both hold the same mutants, so either is accepted.
    """
    cands = [dms_filename, dms_id + ".csv"]
    old = ALIAS_NEW_TO_OLD.get(dms_id)
    if old:
        cands.append(old + ".csv")
    for c in cands:
        if c and os.path.exists(os.path.join(dms_folder, c)):
            return os.path.join(dms_folder, c)
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="PTULyra_ytx scoring on ProteinGym DMS substitutions")
    ap.add_argument("--DMS_reference_file_path",
                    default=os.path.join(here, "..", "..", "..", "reference_files",
                                         "DMS_substitutions.csv"),
                    help="ProteinGym reference file listing the assays (default: reference_files/DMS_substitutions.csv)")
    ap.add_argument("--DMS_data_folder", default=os.environ.get("DMS_data_folder_subs"),
                    help="Folder holding the per-assay benchmark CSVs (DMS_ProteinGym_substitutions)")
    ap.add_argument("--signals_dir", default=os.environ.get("PTULYRA_SIGNALS_DIR"),
                    help="Folder holding PTULyra_ytx's component signal tables (from the Zenodo archive)")
    ap.add_argument("--output_scores_folder", required=True,
                    help="Folder to write one <DMS_id>.csv per assay into")
    ap.add_argument("--config", default=os.path.join(here, "routed_config.json"),
                    help="Per-group arm routing and locked weight vectors")
    ap.add_argument("--DMS_index", type=int, nargs="*", default=None,
                    help="Optional: score only these assays (DMS_index values, 0-based). "
                         "Default: all assays in the reference file.")
    ap.add_argument("--quiet", action="store_true", help="Only print the final summary")
    a = ap.parse_args()

    for name, val in (("--DMS_data_folder", a.DMS_data_folder), ("--signals_dir", a.signals_dir)):
        if not val:
            print("[ERROR] %s is required (or set the matching environment variable)" % name,
                  flush=True)
            return 2
    for name, val in (("--DMS_data_folder", a.DMS_data_folder),
                      ("--signals_dir", a.signals_dir),
                      ("--DMS_reference_file_path", a.DMS_reference_file_path),
                      ("--config", a.config)):
        if not os.path.exists(val):
            print("[ERROR] %s does not exist: %s" % (name, val), flush=True)
            return 2

    cfg = json.load(open(a.config, encoding="utf-8"))
    route, locked = cfg["route"], cfg["locked"]

    # ---- reference file: assay list, functional group, data filename ----
    ref = pd.read_csv(a.DMS_reference_file_path)
    for col in ("DMS_id", "DMS_filename", "coarse_selection_type", "UniProt_ID"):
        if col not in ref.columns:
            print("[ERROR] reference file is missing the '%s' column" % col, flush=True)
            return 2
    assays = []
    for _, r in ref.iterrows():
        assays.append((str(r["DMS_id"]), str(r["DMS_filename"]),
                       str(r["coarse_selection_type"]), str(r["UniProt_ID"])))
    if a.DMS_index:
        keep = set()
        for i in a.DMS_index:
            if not (0 <= i < len(assays)):
                print("[ERROR] --DMS_index %d out of range (0..%d)" % (i, len(assays) - 1), flush=True)
                return 2
            keep.add(i)
        assays = [x for i, x in enumerate(assays) if i in keep]
    print("Reference: %s  (%d assays selected)" % (a.DMS_reference_file_path, len(assays)), flush=True)

    # ---- component signals ----
    S = a.signals_dir
    SIGS = {
        "prosst": {did: csv_col_to_dict(df, "VenusREM")
                   for did, df in load_csv_dir(os.path.join(S, "prosst_scores")).items()},
        "venusrem": {did: csv_col_to_dict(df, "VenusREM")
                     for did, df in load_csv_dir(os.path.join(S, "venusrem_scores")).items()},
    }
    for name in ("gemme", "msa1b", "esmc6b", "esmc600", "transfer"):
        SIGS[name] = load_json_signal(os.path.join(S, "%s.json" % name))
    print("Signals: %s" % {k: len(v) for k, v in SIGS.items()}, flush=True)

    sig_key = lambda did: ALIAS_NEW_TO_OLD.get(did, did)   # noqa: E731

    os.makedirs(a.output_scores_folder, exist_ok=True)
    groups, uniprots, rhos, n_written, problems = {}, {}, {}, 0, []

    for did, dms_filename, group, uniprot in assays:
        path = resolve_dms_file(a.DMS_data_folder, did, dms_filename)
        if path is None:
            problems.append((did, "benchmark data file not found in --DMS_data_folder"))
            continue
        df = pd.read_csv(path, dtype={"mutant": str})
        if "mutant" not in df.columns or "DMS_score" not in df.columns:
            problems.append((did, "benchmark file lacks mutant/DMS_score"))
            continue
        ymap = {m: float(v) for m, v in zip(df["mutant"].tolist(), df["DMS_score"].tolist()) if v == v}

        if group not in route:
            problems.append((did, "functional group '%s' has no arm in routed_config.json" % group))
            continue
        arm = route[group]
        w = locked[group][arm]
        # `transfer` in the locked vectors is inert: it enters only through the step-3 blend.
        core_names = [n for n in w if n != "transfer"]
        core_sigs = {n: SIGS[n] for n in core_names if n in SIGS}

        key = sig_key(did)
        scores, alpha, diag = pred_did(key, w, core_sigs, ymap, SIGS["transfer"].get(key, {}))
        if not scores:
            problems.append((did, diag.get("reason", "no scores produced")))
            continue

        # one row per mutant of the benchmark's own set
        out = df[["mutant"]].copy()
        out["PTULyra_ytx_score"] = out["mutant"].map(scores)
        out["DMS_score"] = df["DMS_score"].values
        out.to_csv(os.path.join(a.output_scores_folder, "%s.csv" % did), index=False)
        n_written += 1

        groups[did] = group
        uniprots[did] = uniprot
        if diag.get("n_scored", 0) < diag.get("frame_n", 0):
            problems.append((did, "scored %d of %d mutants" % (diag["n_scored"], diag["frame_n"])))
        common = [m for m in scores if m in ymap]
        rhos[did] = spearman([ymap[m] for m in common], [scores[m] for m in common]) \
            if len(common) >= 10 else float("nan")
        if not a.quiet:
            print("  %-46s %-18s alpha=%.3f  n=%d/%d  rho=%.4f"
                  % (did, group, alpha, diag.get("n_scored", 0), diag.get("frame_n", 0),
                     rhos[did]), flush=True)

    gm, u = official_agg(rhos, groups, uniprots)
    dm = dms_agg(rhos, groups)
    print("\nWrote %d assay CSVs to %s" % (n_written, a.output_scores_folder), flush=True)
    print("  UniProt-level (official) = %.4f    DMS-level = %.4f" % (u, dm), flush=True)
    for g in GROUPS:
        print("    %-20s %.4f" % (g, gm.get(g, float("nan"))), flush=True)
    if problems:
        print("\n  %d note(s):" % len(problems), flush=True)
        for did, why in problems[:40]:
            print("    %s: %s" % (did, why), flush=True)
    return 0 if n_written else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
