"""Plots true values vs. mean predictions (with errorbars) for four assays across splits"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import COLORS

warnings.simplefilter(action="ignore", category=FutureWarning)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"


def y_true_vs_pred_custom(model_name):
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Tsuboyama_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]

    prediction_dir = Path("results/predictions")
    fig_dir = Path("figures/calibration/y_vs_y_pred")
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    folds = [0, 1, 2, 3, 4]
    for dataset in datasets:
        fig, ax = plt.subplots(
            5,
            3,
            figsize=(6, 8),
            sharex="all",
            sharey="all",
        )

        for j, method in enumerate(methods):
            prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
            df = pd.read_csv(prediction_path)
            for i, fold in enumerate(folds):
                df_fold = df.loc[df["fold"] == fold]
                y_err = 2 * np.sqrt(df_fold["y_var"].values)
                ax[i, j].errorbar(
                    y=df_fold["y_pred"],
                    x=df_fold["y"],
                    fmt="o",
                    yerr=y_err,
                    ecolor=COLORS[j],
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                    capsize=2.5,
                    markersize=3.5,
                    alpha=0.7,
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
                    ax[i, j].set_title(f"{method[5:-2].capitalize()}", fontsize=10)
                # if i == (len(methods) - 1):
                if j == 0:
                    ax[i, j].set_ylabel(f"Prediction (fold {fold+1})", fontsize=10)
                if i == (len(folds) - 1):
                    ax[i, j].set_xlabel("Target", fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(
            fig_dir / f"{dataset}_pred_vs_true_{model_name}.png",
            dpi=125,
            bbox_inches="tight",
        )
        plt.savefig(
            fig_dir / f"{dataset}_pred_vs_true_{model_name}.pdf", bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    for model_name in ["kermut", "ProteinNPT"]:
        y_true_vs_pred_custom(model_name)
