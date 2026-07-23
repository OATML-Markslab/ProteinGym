#!/usr/bin/env python3
"""XSignal zero-shot scorer for ProteinGym substitution assays.

XSignal is the public name for the frozen internal candidate
PureGraphGEContactPair_v1. The submitted score is a 14-channel assay-wise
percentile-rank ensemble over MSA, AlphaFold2-structure, and fixed residue
chemistry evidence.

This script intentionally reads only variant-key columns from DMS assay files:
`mutant` and `mutated_sequence`. It does not read DMS labels, binary labels,
supervised folds, public baseline score packages, or official merged scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from xsignal_core.a2m import load_proteingym_focus_sequences, load_weights, weighted_conservation
from xsignal_core.sensors import (
    contact_pair_ge_score,
    msa_structure_joint_composable_score,
    structure_context_composable_score,
    tm_stability_composable_score,
)
from xsignal_core.structure import parse_af2_structure


LEGACY_SENSORS = {
    "rsalor_like": ("PureRSALORLike_v1", "PureRSALORLike_score"),
    "context": ("PureContextRank_v1", "PureContextRank_score"),
    "family_mlm": ("PureFamilyConvMLM_v1", "PureFamilyConvMLM_score"),
    "family_vae": ("PureFamilyVAE_v1", "PureFamilyVAE_score"),
    "structure_energy": ("PureStructureEnergy_v1", "PureStructureEnergy_score"),
    "potts_pll": ("PurePottsPLL_v1", "PurePottsPLL_score"),
    "tt_physics": ("PureTTPhysics_v1", "PureTTPhysics_score"),
    "contextual_msa": ("PureContextualMSA_v1b", "PureContextualMSA_score"),
    "subfamily_msa": ("PureSubfamilyMSA_v1", "PureSubfamilyMSA_score"),
    "structure_graph": ("PureStructureGraph_v1", "PureStructureGraph_score"),
}
EXPECTED_SENSOR_COUNT = 14
SCORE_NAME = "XSignal_score"


def rank01(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        raise ValueError("rank input contains non-numeric values")
    return numeric.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def parse_start(value: object) -> int:
    text = str(value)
    try:
        return int(text.split("-")[0])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"invalid pdb_range start: {value}") from None


def read_variant_keys(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in ("mutant", "mutated_sequence") if column in header.columns]
    if not usecols:
        raise ValueError(f"{path} is missing mutant/mutated_sequence columns")
    frame = pd.read_csv(path, usecols=usecols)
    if "mutated_sequence" not in frame.columns:
        frame["mutated_sequence"] = frame["mutant"]
    if "mutant" not in frame.columns:
        frame["mutant"] = frame["mutated_sequence"]
    if frame["mutated_sequence"].isna().any() or frame["mutant"].isna().any():
        raise ValueError(f"{path} contains null variant keys")
    return frame[["mutant", "mutated_sequence"]]


def aligned_legacy_sensor(path: Path, assay_frame: pd.DataFrame, score_column: str) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["mutated_sequence", score_column])
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    if frame[score_column].isna().any():
        raise ValueError(f"non-finite legacy sensor values: {path}")
    compact = (
        frame.drop_duplicates("mutated_sequence")
        .groupby("mutated_sequence", as_index=False)
        .mean(numeric_only=True)
    )
    merged = assay_frame[["mutated_sequence"]].merge(compact, on="mutated_sequence", how="left", sort=False)
    if merged[score_column].isna().any():
        raise ValueError(f"legacy sensor does not cover all variants: {path}")
    return rank01(merged[score_column].to_numpy(dtype=np.float64))


def corrected_msa_conservation_with_metadata(
    msa_path: Path,
    weight_path: Path,
    start: int,
    end: int,
) -> tuple[dict[int, float], dict[str, int]]:
    sequences, metadata = load_proteingym_focus_sequences(
        msa_path,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_indeterminate=True,
    )
    expected_length = end - start + 1
    if metadata["msa_focus_length"] != expected_length:
        raise ValueError(
            f"MSA focus length mismatch: {msa_path}: focus={metadata['msa_focus_length']}, expected={expected_length}"
        )
    weights = load_weights(weight_path, len(sequences))
    metadata = dict(metadata)
    metadata["msa_weight_sequences"] = int(len(weights))
    metadata["msa_weight_sum_floor"] = int(np.floor(weights.sum()))
    return weighted_conservation(sequences, weights, start, end), metadata


def score_assay(row: pd.Series, args: argparse.Namespace, structure_cache: dict, msa_cache: dict) -> dict:
    started = time.perf_counter()
    dms_id = str(row["DMS_id"])
    dms_path = args.DMS_data_folder / str(row["DMS_filename"])
    assay = read_variant_keys(dms_path)

    components: list[np.ndarray] = []
    for sensor_name, (folder_name, score_column) in LEGACY_SENSORS.items():
        sensor_path = args.legacy_score_folder / folder_name / f"{dms_id}.csv"
        if not sensor_path.exists():
            raise FileNotFoundError(f"missing legacy XSignal sensor {sensor_name}: {sensor_path}")
        components.append(aligned_legacy_sensor(sensor_path, assay, score_column))

    pdb_path = args.AF2_structures_folder / str(row["pdb_file"])
    structure_key = (str(pdb_path), "A")
    if structure_key not in structure_cache:
        structure_cache[structure_key] = parse_af2_structure(pdb_path, chain_id="A")
    structure = structure_cache[structure_key]
    pdb_start = parse_start(row["pdb_range"])
    mutants = assay["mutant"].astype(str)

    msa_path = args.MSA_folder / str(row["MSA_filename"])
    weight_path = args.MSA_weights_folder / str(row["weight_file_name"])
    msa_key = (str(msa_path), str(weight_path), int(row["MSA_start"]), int(row["MSA_end"]))
    if msa_key not in msa_cache:
        msa_cache[msa_key] = corrected_msa_conservation_with_metadata(
            msa_path,
            weight_path,
            int(row["MSA_start"]),
            int(row["MSA_end"]),
        )
    conservation, msa_metadata = msa_cache[msa_key]

    structure_scores = np.asarray(
        [structure_context_composable_score(value, structure, pdb_start) for value in mutants],
        dtype=np.float64,
    )
    tm_scores = np.asarray(
        [tm_stability_composable_score(value, structure, pdb_start) for value in mutants],
        dtype=np.float64,
    )
    joint_scores = np.asarray(
        [msa_structure_joint_composable_score(value, conservation, structure, pdb_start) for value in mutants],
        dtype=np.float64,
    )
    contact_scores = np.asarray(
        [contact_pair_ge_score(value, structure, pdb_start) for value in mutants],
        dtype=np.float64,
    )
    components.extend([rank01(structure_scores), rank01(tm_scores), rank01(joint_scores), rank01(contact_scores)])
    if len(components) != EXPECTED_SENSOR_COUNT:
        raise ValueError(f"{dms_id}: expected {EXPECTED_SENSOR_COUNT} channels, got {len(components)}")

    score = np.mean(np.column_stack(components), axis=1)
    if not np.isfinite(score).all():
        raise ValueError(f"non-finite XSignal scores: {dms_id}")

    output = assay[["mutant", "mutated_sequence"]].copy()
    output[SCORE_NAME] = score
    args.output_scores_folder.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_scores_folder / f"{dms_id}.csv", index=False)

    mutation_counts = mutants.str.count(":") + 1
    metadata = {
        "DMS_id": dms_id,
        "status": "ok",
        "n_rows": int(len(output)),
        "n_channels": int(len(components)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "structure_residues": int(len(structure.aa)),
        "n_non_single_mutants": int((mutation_counts > 1).sum()),
        "ge_contact_pair_nonzero_frac": float(np.mean(contact_scores != 0.0)),
    }
    metadata.update(msa_metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score ProteinGym substitutions with XSignal.")
    parser.add_argument("--DMS_reference_file_path", type=Path, required=True)
    parser.add_argument("--DMS_data_folder", type=Path, required=True)
    parser.add_argument("--DMS_index", type=int, default=None)
    parser.add_argument("--output_scores_folder", type=Path, required=True)
    parser.add_argument("--MSA_folder", type=Path, required=True)
    parser.add_argument("--MSA_weights_folder", type=Path, required=True)
    parser.add_argument("--AF2_structures_folder", type=Path, required=True)
    parser.add_argument(
        "--legacy_score_folder",
        type=Path,
        required=True,
        help="Folder containing the ten self-generated XSignal legacy sensor score directories.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference = pd.read_csv(args.DMS_reference_file_path)
    if args.DMS_index is not None:
        reference = reference.iloc[[args.DMS_index]]

    manifest = []
    structure_cache: dict = {}
    msa_cache: dict = {}
    for _, row in reference.iterrows():
        metadata = score_assay(row, args, structure_cache, msa_cache)
        manifest.append(metadata)
        (args.output_scores_folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )

    summary = {
        "model": "XSignal",
        "internal_candidate": "PureGraphGEContactPair_v1",
        "score_name": SCORE_NAME,
        "n_assays": sum(item["status"] == "ok" for item in manifest),
        "n_errors": 0,
        "formula": "14-channel assay-wise percentile-rank equal fusion",
        "no_dms_labels_at_inference": True,
        "no_pretrained_protein_language_model_weights": True,
        "uses_precomputed_af2_structures": True,
    }
    (args.output_scores_folder / "complete.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"XSignal scoring failed: {exc}", file=sys.stderr)
        raise
