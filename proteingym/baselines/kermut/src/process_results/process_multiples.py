"""Script to compute performance metrics for fold_rand_multiples and domain splits.
Resuls are printed."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main():
    df_ref = pd.read_csv(Path("data", "DMS_substitutions.csv"))
    # Load and filter reference data
    df_ref = df_ref[df_ref["includes_multiple_mutants"]]
    df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
    df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]

    result_dir = Path("results", "predictions")

    model_names = [
        "kermut",
        "kermut_no_m_constant_mean",
        "kermut_constant_mean",
    ]

    # Process results for multiples split
    df = pd.DataFrame()
    for model_name in model_names:
        for dataset in df_ref["DMS_id"].tolist():
            result_path = result_dir / dataset / f"{model_name}_fold_rand_multiples.csv"
            df_ = pd.read_csv(result_path)
            df_["n_mutations"] = df_["mutant"].apply(lambda x: len(x.split(":")))

            # Group by n_mutations and compute MSE and Spearman
            metrics_per_mut = df_.groupby(["n_mutations"], as_index=False).apply(
                lambda x: pd.Series(
                    {
                        "MSE": ((x["y"] - x["y_pred"]) ** 2).mean(),
                        "Spearman": x[["y", "y_pred"]]
                        .corr(method="spearman")
                        .iloc[0, 1],
                    }
                )
            )
            # Overall mean
            corr_tot, _ = spearmanr(df_["y"], df_["y_pred"])
            mse_tot = ((df_["y"] - df_["y_pred"]) ** 2).mean()
            # Combine
            metrics_per_mut = pd.concat(
                [
                    metrics_per_mut,
                    pd.DataFrame(
                        [
                            pd.Series(
                                {
                                    "MSE": mse_tot,
                                    "Spearman": corr_tot,
                                    "n_mutations": "all",
                                }
                            )
                        ]
                    ),
                ]
            )

            metrics_per_mut["model_name"] = model_name
            metrics_per_mut["fold_variable_name"] = "fold_rand_multiples"
            if df.empty:
                df = metrics_per_mut
            else:
                df = pd.concat([df, metrics_per_mut])

            # Handle domain split
            result_path = result_dir / dataset / f"{model_name}_domain.csv"
            df_ = pd.read_csv(result_path)
            df_ = df_[df_["n_mutations"] == 2]

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "model_name": model_name,
                            "fold_variable_name": "domain",
                            "n_mutations": 2,
                            "MSE": ((df_["y"] - df_["y_pred"]) ** 2).mean(),
                            "Spearman": df_[["y", "y_pred"]]
                            .corr(method="spearman")
                            .iloc[0, 1],
                        },
                        index=[0],
                    ),
                ]
            )

    df_agg = df.groupby(
        ["model_name", "fold_variable_name", "n_mutations"], as_index=False
    ).mean()

    df_ = df_agg.pivot(
        index=["fold_variable_name", "n_mutations"],
        columns="model_name",
        values=["Spearman", "MSE"],
    )
    df_ = df_.T
    df_ = df_[df_.columns[1:].tolist() + [df_.columns[0]]]
    print(df_.to_latex(float_format="%.3f"))


if __name__ == "__main__":
    main()
