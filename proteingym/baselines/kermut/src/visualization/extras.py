"""Extra plotting functionality, e.g., calibration curves for single datasets"""

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


def plot_y_vs_y_pred(
    dataset: list,
    methods: list,
    model_name: str,
):
    prediction_dir = Path("results/predictions")
    fig_dir = Path("figures", "calibration", "extras")
    folds = [0, 1, 2, 3, 4]

    if len(methods) > 1:
        fig, ax = plt.subplots(
            len(methods),
            5,
            figsize=(12, len(methods) * 2.5),
            sharex="all",
            sharey="all",
        )

        for i, method in enumerate(methods):
            prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
            df = pd.read_csv(prediction_path)
            folds = sorted(df["fold"].unique())
            for j, fold in enumerate(folds):
                df_fold = df.loc[df["fold"] == fold]
                y_err = 2 * np.sqrt(df_fold["y_var"].values)
                ax[i, j].errorbar(
                    y=df_fold["y_pred"],
                    x=df_fold["y"],
                    fmt="o",
                    yerr=y_err,
                    ecolor=COLORS[i],
                    markerfacecolor=COLORS[i],
                    markeredgecolor="white",
                    capsize=2.5,
                )
                # Add dotted grey diagonal line
                y_min = min(df["y"].min(), df["y_pred"].min())
                y_max = max(df["y"].max(), df["y_pred"].max())

                ax[i, j].plot(
                    [y_min, y_max],
                    [y_min, y_max],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2.5,
                )
                if i == 0:
                    ax[i, j].set_title(f"Test fold {fold}")
                # if i == (len(methods) - 1):
                if j == 0:
                    ax[i, j].set_ylabel(f"{method}")

        plt.suptitle(f"{dataset} - {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_pred_vs_true_{model_name}.png", dpi=125)
        plt.close()

    elif len(methods) == 1:
        fig, ax = plt.subplots(
            1,
            5,
            figsize=(12, 1.5 * 2.5),
            sharex="all",
            sharey="all",
        )
        method = methods[0]
        prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
        df = pd.read_csv(prediction_path)
        if "train" in df:
            df = df[df["train"] == False]
        folds = sorted(df["fold"].unique())
        for i, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]
            y_err = np.sqrt(df_fold["y_var"].values)
            # sns.scatterplot(
            # data=df_fold,
            # x="y_pred",
            # y="y",
            # color=COLORS[j],
            # ax=ax[i, j],
            # )
            ax[i].errorbar(
                y=df_fold["y_pred"],
                x=df_fold["y"],
                fmt="o",
                yerr=y_err,
                ecolor=COLORS[i],
                markerfacecolor=COLORS[i],
                markeredgecolor="white",
                capsize=2.5,
            )
            # Add dotted grey diagonal line
            y_min = min(df["y"].min(), df["y_pred"].min())
            y_max = max(df["y"].max(), df["y_pred"].max())

            ax[i].plot(
                [y_min, y_max],
                [y_min, y_max],
                "k--",
                linewidth=1,
                alpha=0.5,
                zorder=2.5,
            )
            ax[i].set_title(f"Test fold {fold}")
            if i == 0:
                ax[i].set_ylabel(f"{method}")

        plt.suptitle(f"{dataset} - {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_{method}_{model_name}.png", dpi=125)
        plt.close()


def plot_reliability_diagram(
    dataset: str,
    methods: list,
    model_name: str,
):
    """
    Plot reliability diagram for a given dataset and model
    """
    plt.rcParams["font.family"] = "serif"

    prediction_dir = Path("results/predictions", dataset)
    fig_dir = Path("figures", "calibration", "extras")
    fig_dir.mkdir(parents=True, exist_ok=True)

    number_quantiles = 10

    fig, ax = plt.subplots(
        2, len(methods), figsize=(len(methods) * 2, 4), sharex="all", sharey="all"
    )
    if ax.ndim == 1:
        ax = ax.reshape(1, -1)

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / f"{model_name}_{method}.csv")
        df_ci, df_ci_metrics = compute_calibration_metrics(
            df, method="ci-based", n_bins=number_quantiles
        )

        sns.scatterplot(
            data=df_ci,
            x="percentile",
            y="confidence",
            hue="fold",
            palette=COLORS,
            ax=ax[0, i],
            legend=False,
        )
        sns.lineplot(
            data=df_ci,
            x="percentile",
            y="confidence",
            hue="fold",
            palette=COLORS,
            ax=ax[0, i],
            legend=False,
        )

        sns.scatterplot(
            data=df_ci.groupby(["percentile"], as_index=False).agg(
                {"confidence": "mean"}
            ),
            x="percentile",
            y="confidence",
            color=COLORS[i],
            ax=ax[1, i],
            legend=False,
        )
        sns.lineplot(
            data=df_ci,
            x="percentile",
            y="confidence",
            color=COLORS[i],
            ax=ax[1, i],
            legend=False,
        )

        ECE_avg = df_ci_metrics["ECE"].mean()
        ECE_std = df_ci_metrics["ECE"].std()

        # Add box with EVE and sharpness value in top left corner of ax
        ax[1, i].text(
            0.95,
            0.05,
            f"ECE: {ECE_avg:.2f} (±{2*ECE_std:.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[1, i].transAxes,
            fontsize=7,
            bbox=dict(
                # boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, pad=2
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
                pad=2,
            ),
        )

        ax[0, i].set_ylabel("Confidence", fontsize=10)
        ax[0, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_yticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_title(f"{method}")

        ax[1, i].set_xlabel("Percentile", fontsize=10)
        ax[1, i].set_ylabel("Confidence", fontsize=10)
        ax[1, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[1, i].set_yticks(np.arange(0, 1.1, 0.2))

        # Add dotted black line from (0,0) to (1,1)
        ax[0, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[1, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[0, i].set_aspect("equal", "box")
        ax[1, i].set_aspect("equal", "box")

    plt.suptitle(f"{dataset} -- {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"ci_based_{dataset}_{model_name}.png", dpi=125)
    plt.close()


def plot_error_based_calibration_curve(
    dataset: str,
    model_name: str,
    methods=["fold_random_5", "fold_modulo_5", "fold_contiguous_5"],
    fix_axes: bool = True,
):
    """
    Plot error-based calibration curve for a given dataset and model
    """

    plt.rcParams["font.family"] = "serif"

    prediction_dir = Path("results/predictions", dataset)
    fig_dir = Path("figures", "calibration", "extras")
    fig_dir.mkdir(parents=True, exist_ok=True)

    number_quantiles = 5

    fig, ax = plt.subplots(2, 3, figsize=(6.5, 4))  # , sharex="col", sharey="col")
    if ax.ndim == 1:
        ax = ax.reshape(1, -1)

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / f"{model_name}_{method}.csv")
        df_err, df_err_metrics = compute_calibration_metrics(
            df, method="error-based", n_bins=number_quantiles
        )

        # For each fold
        for j, fold in enumerate(df_err["fold"].unique()):
            df_fold = df_err[df_err["fold"] == fold]
            sns.scatterplot(
                data=df_fold,
                x="RMV",
                y="RMSE",
                color=COLORS[j],
                legend=False,
                ax=ax[0, i],
            )
            sns.lineplot(
                data=df_fold,
                x="RMV",
                y="RMSE",
                color=COLORS[j],
                legend=False,
                ax=ax[0, i],
            )

        # Comined
        df_grouped = df_err.groupby(["bin"], as_index=False)
        RMSE_mean = df_grouped.mean(numeric_only=True)["RMSE"]
        RMSE_std = df_grouped.std(numeric_only=True)["RMSE"]
        RMV_mean = df_grouped.mean(numeric_only=True)["RMV"]
        RMV_std = df_grouped.std(numeric_only=True)["RMV"]

        ax[1, i].errorbar(
            x=RMV_mean,
            y=RMSE_mean,
            fmt="o",
            xerr=RMV_std,
            yerr=RMSE_std,
            ecolor=COLORS[i],
            markerfacecolor=COLORS[i],
            markeredgecolor="white",
            capsize=2.5,
        )

        ENCE_avg = df_err_metrics["ENCE"].mean()
        ENCE_std = df_err_metrics["ENCE"].std()
        CV_avg = df_err_metrics["CV"].mean()
        CV_std = df_err_metrics["CV"].std()

        # Add box with EVE and sharpness value in top left corner of ax
        ax[1, i].text(
            x=0.95,
            y=0.05,
            s=f"ENCE: {ENCE_avg:.2f} (±{2*ENCE_std:.2f})\n"
            + r"$c_v$"
            + f": {CV_avg:.2f} (±{2*CV_std:.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[1, i].transAxes,
            fontsize=9,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
                pad=3,
            ),
        )

        # Add dashed diagonal line
        y_min, y_max = min(df_err["RMV"].min(), df_err["RMSE"].min()), max(
            df_err["RMV"].max(), df_err["RMSE"].max()
        )

        if fix_axes:
            ax[0, i].plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1, alpha=0.5)
            ax[1, i].plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1, alpha=0.5)

        if i == 0:
            ax[0, i].set_ylabel("RMSE", fontsize=10)
            ax[1, i].set_ylabel("RMSE", fontsize=10)
        else:
            ax[0, i].set_ylabel("")
            ax[1, i].set_ylabel("")

        for j in range(3):
            ax[1, j].set_xlabel("RMV", fontsize=10)

    plt.suptitle(f"{dataset} -- {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"e_based_{dataset}_{model_name}.png", dpi=125)
    plt.close()
