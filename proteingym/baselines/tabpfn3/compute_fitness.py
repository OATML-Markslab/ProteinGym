#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_reproduction import result_to_jsonable, run_proteingym_assay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one ProteinGym supervised DMS substitutions assay with TabPFN3. "
            "This script is a ProteinGym-style baseline wrapper for TabPFN3."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="ProteinGym CV CSV for one DMS assay.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions.")
    parser.add_argument(
        "--fold-column",
        default="fold_random_5",
        choices=["fold_random_5", "fold_modulo_5", "fold_contiguous_5"],
        help="Official ProteinGym supervised split column.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Directory containing <assay>/X_float16.dat feature matrices.",
    )
    parser.add_argument(
        "--feature-source",
        default="memmap",
        choices=["auto", "memmap", "csv_mutation"],
        help="Use stored feature matrices for submission reproduction; csv_mutation is smoke-test only.",
    )
    parser.add_argument("--feature-pca-components", type=int, default=None)
    parser.add_argument("--fold-values", default="", help="Optional comma-separated folds, e.g. 0,1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Optional local TabPFN3 checkpoint path. If omitted, the tabpfn package "
            "uses its official first-use download flow after license acceptance."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--fit-mode", default="fit_preprocessors")
    parser.add_argument("--memory-saving-mode", default="auto")
    parser.add_argument("--inference-precision", default="auto")
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--prediction-chunk-size", type=int, default=None)
    parser.add_argument("--ignore-pretraining-limits", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_values = [int(item) for item in args.fold_values.split(",") if item.strip()] or None
    result = run_proteingym_assay(
        csv_path=args.csv,
        out_dir=args.out_dir,
        model_name="tabpfn3",
        fold_column=args.fold_column,
        feature_source=args.feature_source,
        feature_root=args.feature_root,
        feature_pca_components=args.feature_pca_components,
        fold_values=fold_values,
        seed=args.seed,
        force=args.force,
        tabpfn3_kwargs={
            "model_path": str(args.model_path) if args.model_path is not None else None,
            "device": args.device,
            "n_estimators": args.n_estimators,
            "fit_mode": args.fit_mode,
            "memory_saving_mode": args.memory_saving_mode,
            "inference_precision": args.inference_precision,
            "ignore_pretraining_limits": args.ignore_pretraining_limits,
            "n_preprocessing_jobs": args.n_preprocessing_jobs,
            "prediction_chunk_size": args.prediction_chunk_size,
        },
    )
    print(json.dumps(result_to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
