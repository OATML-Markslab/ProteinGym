from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def compute_calibration_metrics(
    df: pd.DataFrame, method: str, n_bins: int = 5
) -> pd.DataFrame:
    df = df.copy()
    if method == "error-based":
        try:
            df["sq_error"] = (df["y"] - df["y_pred"]) ** 2
            df["bin"] = np.nan
            if "fold" not in df.columns:
                df["fold"] = 0
            # Assign bins for each fold
            for fold in df["fold"].unique():
                df_fold = df[df["fold"] == fold]
                df.loc[df_fold.index, "bin"] = pd.qcut(
                    df_fold["y_var"], n_bins, labels=False, duplicates="drop"
                )
            df_calib = df.groupby(["bin", "fold"], as_index=False).agg(
                {"sq_error": "mean", "y_var": "mean"}
            )
            df_calib["RMSE"] = np.sqrt(df_calib["sq_error"])
            df_calib["RMV"] = np.sqrt(df_calib["y_var"])
            df_calib = df_calib[["bin", "fold", "RMSE", "RMV"]]

            # Compute expected normalized calibration error
            ence = df_calib.groupby(["fold"]).apply(
                lambda x: np.mean(np.abs(x["RMV"] - x["RMSE"]) / x["RMV"])
            )
            # Compute coefficient of variation
            cv = np.zeros(len(df["fold"].unique()))
            df["y_std"] = np.sqrt(df["y_var"])
            for fold in df["fold"].unique():
                df_fold = df[df["fold"] == fold]
                mu_sig = np.mean(df_fold["y_std"])
                cv_ = np.sqrt(
                    np.sum((df_fold["y_std"] - mu_sig) ** 2 / (len(df_fold) - 1))
                    / mu_sig
                )
                cv[fold] = cv_

            df_metrics = pd.DataFrame(
                {
                    "fold": df["fold"].unique(),
                    "ENCE": ence.values,
                    "CV": cv,
                }
            )
        except ValueError:
            print("Error-based calibration metrics could not be computed")

            # Return nan-filled df
            df_metrics = pd.DataFrame(
                {
                    "fold": df["fold"].unique(),
                    "ENCE": np.nan,
                    "CV": np.nan,
                }
            )

            return None, df_metrics

    elif method == "ci-based":
        perc = np.arange(0, 1.1, 1 / n_bins)
        df_calib = pd.DataFrame()
        df_metrics = pd.DataFrame()

        for fold in df["fold"].unique():
            df_fold = df[df["fold"] == fold]
            y_target = df_fold["y"].values
            y_pred = df_fold["y_pred"].values
            y_var = df_fold["y_var"].values

            count_arr = np.vstack(
                [
                    np.abs(y_target - y_pred)
                    <= stats.norm.interval(
                        q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var)
                    )[1]
                    for q in perc
                ]
            )
            count = np.mean(count_arr, axis=1)
            ECE = np.mean(np.abs(count - perc))
            marginal_var = np.var(y_target - y_pred)
            dof = np.sum(np.cov(y_pred, y_target)) / marginal_var
            chi_2 = np.sum((y_target - y_pred) ** 2 / y_var) / (len(y_target) - 1 - dof)

            if df_calib.empty:
                df_calib = pd.DataFrame(
                    {
                        "percentile": perc,
                        "confidence": count,
                        "fold": fold,
                    }
                )
                df_metrics = pd.DataFrame(
                    {
                        "fold": fold,
                        "ECE": ECE,
                        "chi_2": chi_2,
                    },
                    index=[0],
                )
            else:
                df_ = pd.DataFrame(
                    {
                        "percentile": perc,
                        "confidence": count,
                        "fold": fold,
                    }
                )
                df_calib = pd.concat([df_calib, df_])
                df_ = pd.DataFrame(
                    {
                        "fold": fold,
                        "ECE": ECE,
                        "chi_2": chi_2,
                    },
                    index=[0],
                )
                df_metrics = pd.concat([df_metrics, df_])
    return df_calib.reset_index(drop=True), df_metrics.reset_index(drop=True)


def compute_all_calibration_metrics(
    model_name: str, split_method: str, limit_n: bool = False
):
    df_ref = pd.read_csv("data/DMS_substitutions.csv")
    prediction_dir = Path("results/predictions")
    df_metrics = pd.DataFrame(
        columns=["DMS_id", "fold_variable_name", "fold", "ECE", "chi_2", "ENCE", "CV"]
    )
    num_quantiles = 10
    n_bins = 5

    if limit_n:
        # Ablation.
        df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
    elif split_method in ["fold_rand_multiples", "domain"]:
        df_ref = df_ref[df_ref["includes_multiple_mutants"]]
        df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
        df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]

    for dataset in tqdm(df_ref["DMS_id"].unique()):
        prediction_path = prediction_dir / dataset / f"{model_name}_{split_method}.csv"
        df = pd.read_csv(prediction_path)

        if split_method == "domain":
            df["fold"] = 0

        _, df_eb_metrics = compute_calibration_metrics(
            df, method="error-based", n_bins=n_bins
        )
        _, df_ci_metrics = compute_calibration_metrics(
            df, method="ci-based", n_bins=num_quantiles
        )
        # Combine
        df_m = pd.merge(df_eb_metrics, df_ci_metrics, on="fold")

        df_m["DMS_id"] = dataset
        df_m["fold_variable_name"] = split_method

        if df_metrics.empty:
            df_metrics = df_m
        else:
            df_metrics = pd.concat([df_metrics, df_m])

    calibration_dir = Path("results/calibration_metrics")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    df_metrics = df_metrics.reset_index(drop=True)
    df_metrics.to_csv(
        calibration_dir / f"metrics_{model_name}_{split_method}.csv", index=False
    )


def compute_calibration_plot_values(
    model_name: str, split_method: str, calibration_method: str, limit_n: bool = False
):
    df_ref = pd.read_csv("data/DMS_substitutions.csv")
    prediction_dir = Path("results/ProteinGym/predictions")

    if calibration_method == "error-based":
        n_bins = 5
        df_metrics = pd.DataFrame(
            columns=["DMS_id", "fold_variable_name", "fold", "ENCE", "CV"]
        )
    elif calibration_method == "ci-based":
        n_bins = 10
        df_metrics = pd.DataFrame(
            columns=["DMS_id", "fold_variable_name", "fold", "ECE", "chi_2"]
        )

    if limit_n:
        df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
    elif split_method in ["multiples", "domain"]:
        df_ref = df_ref[df_ref["includes_multiple_mutants"]]
        df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
        df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]

    for dataset in tqdm(df_ref["DMS_id"].unique()):
        prediction_path = prediction_dir / dataset / f"{model_name}_{split_method}.csv"
        # if dataset == "BRCA2_HUMAN_Erwood_2022_HEK293T":
        # continue
        df = pd.read_csv(prediction_path)

        df_calib, _ = compute_calibration_metrics(
            df, method="error-based", n_bins=n_bins
        )
        if df_calib is None:
            continue

        df_calib["DMS_id"] = dataset
        df_calib["fold_variable_name"] = split_method
        if df_metrics.empty:
            df_metrics = df_calib
        else:
            df_metrics = pd.concat([df_metrics, df_calib])

    calibration_dir = Path("results/calibration_metrics")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    df_metrics = df_metrics.reset_index(drop=True)
    df_metrics.to_csv(
        calibration_dir / f"xy_{model_name}_{calibration_method}_{split_method}.csv",
        index=False,
    )
