"""Plots confidence interval-based and error-based calibration curves for four assays across splits"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import COLORS
from src.process_results.utils import compute_calibration_metrics

warnings.simplefilter(action="ignore", category=FutureWarning)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"


def reliability_diagram_custom(model_name, errorbars: bool = True):
    # Plots confidence interval-based calibration curve for model_name.
    prediction_dir = Path("results/predictions")
    fig_dir = Path("figures/calibration")
    fig_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Tsuboyama_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    number_quantiles = 10

    fig, ax = plt.subplots(4, 3, figsize=(6.5, 8.5), sharex="all", sharey="all")
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")

            # Compute calibration metrics
            df_ci, df_ci_metrics = compute_calibration_metrics(
                df, method="ci-based", n_bins=number_quantiles
            )

            if errorbars:
                mean_val = df_ci.groupby(["percentile"], as_index=False).agg(
                    {"confidence": "mean"}
                )["confidence"]
                std_val = df_ci.groupby(["percentile"], as_index=False).agg(
                    {"confidence": "std"}
                )["confidence"]
                perc = df_ci["percentile"].unique()
                ax[i, j].errorbar(
                    x=perc,
                    y=mean_val,
                    fmt="o",
                    yerr=std_val,
                    ecolor=COLORS[j],
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                    capsize=2.5,
                    alpha=0.75,
                )
                ax[i, j].plot(
                    perc,
                    mean_val,
                    color=COLORS[j],
                    alpha=0.4,
                )
                ax[i, j].errorbar(
                    x=perc,
                    y=mean_val,
                    fmt="o",
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                )
            else:
                # Average curve
                sns.scatterplot(
                    data=df_ci,
                    x="percentile",
                    y="confidence",
                    # color=COLORS[j],
                    ax=ax[i, j],
                    legend=False,
                    hue="fold",
                    palette=COLORS,
                )
                sns.lineplot(
                    data=df_ci,
                    x="percentile",
                    y="confidence",
                    # color=COLORS[j],
                    ax=ax[i, j],
                    legend=False,
                    hue="fold",
                    palette=COLORS,
                )

            # Average and std of ECE
            ECE_avg = df_ci_metrics["ECE"].mean()
            ECE_std = df_ci_metrics["ECE"].std()
            if model_name == "ProteinNPT":
                ax[i, j].text(
                    0.5,
                    0.95,
                    f"ECE: {ECE_avg:.2f} (±{2*ECE_std:.2f})",
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=ax[i, j].transAxes,
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.5,
                        pad=3,
                    ),
                )
            else:
                ax[i, j].text(
                    0.5,
                    0.05,
                    f"ECE: {ECE_avg:.2f} (±{2*ECE_std:.2f})",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    transform=ax[i, j].transAxes,
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.5,
                        pad=3,
                    ),
                )

            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_ylim(0, 1)
            ax[i, j].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
            ax[i, j].set_aspect("equal", "box")

            if i == 0:
                ax[i, j].set_yticks(np.arange(0, 1.1, 0.2))
                ax[i, j].set_xticks(np.arange(0, 1.1, 0.2))
                ax[i, j].set_title(f"{method[5:-2].capitalize()}")
            if j == 0:
                ax[i, j].set_ylabel("Confidence", fontsize=10)
            else:
                ax[i, j].set_ylabel("")

            if i == len(datasets) - 1:
                ax[i, j].set_xlabel("Percentile", fontsize=10)
            else:
                ax[i, j].set_xlabel("")

    # Reduce space between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.13, hspace=0.15)
    plt.savefig(fig_dir / f"rel_diag_{model_name}_summary.png", dpi=125)
    plt.savefig(fig_dir / f"rel_diag_{model_name}_summary.pdf", dpi=125)
    plt.close()


def error_based_calibration_custom(model_name, errorbars: bool = True):
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Tsuboyama_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]
    prediction_dir = Path("results/predictions")
    fig_dir = Path("figures/calibration")
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    n_bins = 5

    fig, ax = plt.subplots(4, 3, figsize=(6.5, 8.5))

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")

            df_err, df_err_metrics = compute_calibration_metrics(
                df, method="error-based", n_bins=n_bins
            )

            ENCE_avg = df_err_metrics["ENCE"].mean()
            ENCE_std = df_err_metrics["ENCE"].std()
            CV_avg = df_err_metrics["CV"].mean()
            CV_std = df_err_metrics["CV"].std()

            df_grouped = df_err.groupby(["bin"], as_index=False)
            RMSE_mean = df_grouped.mean(numeric_only=True)["RMSE"]
            RMSE_std = df_grouped.std(numeric_only=True)["RMSE"]
            RMV_mean = df_grouped.mean(numeric_only=True)["RMV"]
            RMV_std = df_grouped.std(numeric_only=True)["RMV"]

            if errorbars:
                ax[i, j].errorbar(
                    x=RMV_mean,
                    y=RMSE_mean,
                    fmt="o",
                    xerr=RMV_std,
                    yerr=RMSE_std,
                    ecolor=COLORS[j],
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                    capsize=2.5,
                    alpha=0.75,
                )
                ax[i, j].plot(
                    RMV_mean,
                    RMSE_mean,
                    color=COLORS[j],
                    alpha=0.4,
                )
                ax[i, j].errorbar(
                    x=RMV_mean,
                    y=RMSE_mean,
                    fmt="o",
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                )
            else:
                sns.scatterplot(
                    data=df_err,
                    x="RMV",
                    y="RMSE",
                    color=COLORS[j],
                    ax=ax[i, j],
                    legend=False,
                    hue="fold",
                    palette=COLORS,
                )
                sns.lineplot(
                    data=df_err,
                    x="RMV",
                    y="RMSE",
                    color=COLORS[j],
                    ax=ax[i, j],
                    legend=False,
                    hue="fold",
                    palette=COLORS,
                )

            ax[i, j].text(
                x=0.5,
                y=0.05,
                s=f"ENCE: {ENCE_avg:.2f} (±{2*ENCE_std:.2f})\n"
                + r"$c_v$"
                + f": {CV_avg:.2f} (±{2*CV_std:.2f})",
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=ax[i, j].transAxes,
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.7,
                    pad=3,
                ),
            )

            # Add dashed diagonal line
            y_min = RMSE_mean.min() - 1.5 * RMSE_std.max()
            y_max = RMSE_mean.max() + 1.5 * RMSE_std.max()
            x_min = RMV_mean.min() - 1.5 * RMV_std.max()
            x_max = RMV_mean.max() + 1.5 * RMV_std.max()

            ax[i, j].plot(
                [min(x_min, y_min), max(x_max, y_max)],
                [min(x_min, y_min), max(x_max, y_max)],
                "k--",
                linewidth=1,
                alpha=0.5,
            )

            # Set ax lims
            ax[i, j].set_xlim(x_min, x_max)
            ax[i, j].set_ylim(y_min - 0.5 * RMSE_std.max(), y_max)

            if i == 0:
                ax[i, j].set_title(f"{method[5:-2].capitalize()}")
            if j == 0:
                ax[i, j].set_ylabel("RMSE", fontsize=10)
            if i == len(datasets) - 1:
                ax[i, j].set_xlabel("RMV", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(fig_dir / f"error_based_{model_name}_summary.png", dpi=125)
    plt.savefig(fig_dir / f"error_based_{model_name}_summary.pdf")
    plt.close()


if __name__ == "__main__":
    for model_name in ["kermut", "ProteinNPT"]:
        reliability_diagram_custom(model_name, errorbars=True)
        error_based_calibration_custom(model_name, errorbars=True)
