"""Exercise the real supervised reporting CLI with distinguishable groups."""

import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = ["Low", "Medium", "High"]
TAXA = ["Human", "Eukaryote", "Prokaryote", "Virus"]
FUNCTIONS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]
MODELS = ["ProteinNPT", "Kermut"]


def run_report(tmp_path, *, indels=False, shuffled=False, invalid=None):
    reference, scores = [], []
    schemes = (
        ["fold_random_5"]
        if indels
        else ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    )
    for i, (depth, taxon, function) in enumerate(product(DEPTHS, TAXA, FUNCTIONS)):
        if invalid == "missing" and depth == "High":
            continue
        ref_depth = "Unknown" if invalid == "unknown" and depth == "High" else depth
        reference.append(
            {
                "DMS_id": f"assay_{i}",
                "UniProt_ID": f"protein_{i}",
                "MSA_Neff_L_category": ref_depth.lower(),
                "taxon": taxon,
                "coarse_selection_type": function,
            }
        )
        for model_index, model in enumerate(MODELS):
            for cv_index, scheme in enumerate(schemes):
                value = (
                    0.1
                    + 0.2 * DEPTHS.index(depth)
                    + 0.03 * TAXA.index(taxon)
                    + 0.005 * FUNCTIONS.index(function)
                    + 0.01 * model_index
                    + 0.002 * cv_index
                )
                scores.append(
                    {
                        "DMS_id": f"assay_{i}",
                        "model_name": model,
                        "fold_variable_name": scheme,
                        "Spearman": value,
                        "MSE": 2 - value,
                    }
                )
    reference, scores = pd.DataFrame(reference), pd.DataFrame(scores)
    if shuffled:
        reference = reference.sample(frac=1, random_state=13)
        scores = scores.sample(frac=1, random_state=71)
    reference.to_csv(tmp_path / "reference.csv", index=False)
    scores.to_csv(tmp_path / "scores.csv", index=False)
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(ROOT / "proteingym/performance_DMS_supervised_benchmarks.py"),
        "--input_scoring_file",
        str(tmp_path / "scores.csv"),
        "--DMS_reference_file_path",
        str(tmp_path / "reference.csv"),
        "--output_performance_file_folder",
        str(output),
        "--number_assay_reshuffle",
        "4",
    ]
    if indels:
        command.append("--indel_mode")
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    return result, output, len(schemes)


@pytest.mark.parametrize("indels", [False, True])
@pytest.mark.parametrize("shuffled", [False, True])
def test_summary_preserves_category_identity_and_other_scores(
    tmp_path, indels, shuffled
):
    result, output, scheme_count = run_report(
        tmp_path, indels=indels, shuffled=shuffled
    )
    assert result.returncode == 0, result.stderr
    kind = "indels" if indels else "substitutions"
    cv_offset = 0.002 * (scheme_count - 1) / 2
    for metric in ["Spearman", "MSE"]:
        summary = pd.read_csv(
            output / metric / f"Summary_performance_DMS_{kind}_{metric}.csv"
        ).set_index("Model_name")
        for model_index, model in enumerate(MODELS):

            def expected(value, metric=metric):
                return round(value if metric == "Spearman" else 2 - value, 3)

            base = 0.1 + 0.01 * model_index + cv_offset
            assert summary.loc[model, f"Average_{metric}"] == pytest.approx(
                expected(base + 0.2 + 0.045 + 0.01)
            )
            for function_index, function in enumerate(FUNCTIONS):
                if indels and function == "Binding":
                    continue  # Existing output convention explicitly writes N/A here.
                assert summary.loc[model, f"Function_{function}"] == pytest.approx(
                    expected(base + 0.2 + 0.045 + 0.005 * function_index)
                )
            for index, depth in enumerate(DEPTHS):
                assert summary.loc[model, f"{depth}_MSA_depth"] == pytest.approx(
                    expected(base + 0.2 * index + 0.045 + 0.01)
                )
            for index, name in enumerate(
                ["Taxa_Human", "Taxa_Other_Eukaryote", "Taxa_Prokaryote", "Taxa_Virus"]
            ):
                assert summary.loc[model, name] == pytest.approx(
                    expected(base + 0.2 + 0.03 * index + 0.01)
                )
        per_assay = pd.read_csv(output / metric / f"DMS_{kind}_{metric}_DMS_level.csv")
        assert per_assay.shape == (60, 3)
        assert np.isfinite(per_assay[MODELS].to_numpy()).all()


@pytest.mark.parametrize("indels", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "unknown"])
def test_missing_required_category_cannot_be_silently_relabeled(
    tmp_path, indels, invalid
):
    result, output, _ = run_report(tmp_path, indels=indels, invalid=invalid)
    assert result.returncode != 0
    assert not list(output.rglob("Summary_performance_*.csv"))
