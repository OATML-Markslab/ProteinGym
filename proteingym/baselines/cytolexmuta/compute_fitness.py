#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / (1.4826 * mad)


def read_reference(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"DMS_id", "DMS_filename"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return rows


def find_score_file(folder: Path, dms_id: str, dms_filename: str) -> Path:
    candidates = [
        folder / f"{dms_id}.csv",
        folder / dms_filename,
        folder / Path(dms_filename).name,
        folder / f"{Path(dms_filename).stem}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No score file found for {dms_id} in {folder}")


def load_scores(path: Path, column: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "mutant" not in fields or column not in fields:
            raise ValueError(f"{path} must contain mutant and {column}")
        for row in reader:
            mutant = row["mutant"]
            if mutant in scores:
                raise ValueError(f"duplicate mutant {mutant} in {path}")
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(f"non-finite score in {path}: {mutant}")
            scores[mutant] = value
    return scores


def load_mutants(dms_data_folder: Path, dms_filename: str) -> list[str]:
    path = dms_data_folder / dms_filename
    if not path.is_file():
        path = dms_data_folder / Path(dms_filename).name
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "mutant" not in set(reader.fieldnames or []):
            raise ValueError(f"{path} missing mutant column")
        return [row["mutant"] for row in reader]


def gather(scores: dict[str, float], mutants: list[str], name: str) -> np.ndarray:
    missing = [mutant for mutant in mutants if mutant not in scores]
    if missing:
        raise ValueError(f"{name} missing {len(missing)} mutants; first={missing[0]}")
    return np.asarray([scores[mutant] for mutant in mutants], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cytolexmuta ProteinGym substitution scores from component score files.")
    parser.add_argument("--reference_csv", required=True, type=Path)
    parser.add_argument("--dms_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--esmc_scores_dir", required=True, type=Path)
    parser.add_argument("--prosst_scores_dir", required=True, type=Path)
    parser.add_argument("--gemme_scores_dir", required=True, type=Path)
    parser.add_argument("--esm_if1_scores_dir", required=True, type=Path)
    parser.add_argument("--esmc_column", default="esmc_600M_score")
    parser.add_argument("--prosst_column", default="ProSST-2048")
    parser.add_argument("--gemme_column", default="GEMME_score")
    parser.add_argument("--esm_if1_column", default="esmif1_ll")
    parser.add_argument("--output_column", default="cytolexmuta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assays = 0
    variants = 0
    for item in read_reference(args.reference_csv):
        dms_id = item["DMS_id"]
        dms_filename = item["DMS_filename"]
        mutants = load_mutants(args.dms_dir, dms_filename)
        esmc = gather(load_scores(find_score_file(args.esmc_scores_dir, dms_id, dms_filename), args.esmc_column), mutants, "ESMC-600M")
        prosst = gather(load_scores(find_score_file(args.prosst_scores_dir, dms_id, dms_filename), args.prosst_column), mutants, "ProSST-2048")
        gemme = gather(load_scores(find_score_file(args.gemme_scores_dir, dms_id, dms_filename), args.gemme_column), mutants, "GEMME")
        esm_if1 = gather(load_scores(find_score_file(args.esm_if1_scores_dir, dms_id, dms_filename), args.esm_if1_column), mutants, "ESM-IF1")
        anchor = 0.5 * esmc + 0.5 * prosst
        fused = 0.5 * robust_z(anchor) + 0.25 * robust_z(gemme) + 0.25 * robust_z(esm_if1)
        output_path = args.output_dir / f"{dms_id}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mutant", args.output_column])
            writer.writeheader()
            for mutant, score in zip(mutants, fused):
                writer.writerow({"mutant": mutant, args.output_column: format(float(score), ".10g")})
        assays += 1
        variants += len(mutants)
    print(f"wrote {assays} assays / {variants} variants to {args.output_dir}")


if __name__ == "__main__":
    main()
