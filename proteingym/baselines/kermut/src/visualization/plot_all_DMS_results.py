"""Script to generate plots showing performance per assay."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import COLORS


def comparison_plot():
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    split_methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    model_names = ["ProteinNPT", "Kermut"]
    colors = [COLORS[0], COLORS[4]]

    # Two plots are generated for each split method, one for the upper half of the dataframe and one for the lower half.
    for split_method in split_methods:
        score_path = Path(
            "results/summary/Spearman",
            f"DMS_substitutions_Spearman_DMS_level_{split_method}.csv",
        )

        for part in ["upper", "lower"]:
            df = pd.read_csv(score_path)
            x_min = df[model_names].min().min()
            # For older ProteinGym versions only:
            df["DMS_id"] = df["DMS_id"].str.replace("Tsuboyama", "Tsuboyama")
            df = df.sort_values(by="Kermut", ascending=True)
            df = df.reset_index(drop=True)

            # Take half of dataframe
            if part == "lower":
                df = df.iloc[: len(df) // 2]
            else:
                df = df.iloc[len(df) // 2 :]

            df["rank"] = range(len(df))

            fig, ax = plt.subplots(figsize=(6, 16))
            max_vals = df[model_names].max(axis=1)
            ax.hlines(
                y=df["rank"],
                xmin=x_min - 0.05,
                xmax=max_vals,
                color="gray",
                linestyles="dashed",
                alpha=0.5,
                linewidth=0.5,
            )

            for i, model_name in enumerate(model_names):
                ax.scatter(
                    x=df[model_name],
                    y=df["rank"],
                    label=model_name,
                    color=colors[i],
                    alpha=1,
                    s=25,
                )
            ax.set_xlim(df[model_names].min().min() - 0.05, 1.0 + 0.05)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["DMS_id"])
            ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
            ax.yaxis.set_tick_params(labelsize=6)
            ax.set_ylabel("")
            ax.set_ylim(-1, len(df) + 1)
            ax.yaxis.set_tick_params(width=1, length=4)
            manual_x_ticks = [0, 0.25, 0.5, 0.75, 1.0]
            ax.set_xticks(manual_x_ticks)
            ax.yaxis.grid(False)

            # Add legend
            ax.legend(loc="upper right", fontsize=8)
            # Change legend to 3 cols and place above figure
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8
            )

            plt.tight_layout()
            plt.savefig(
                f"figures/full_DMS/Spearman_comparison_plot_{split_method}_{part}.pdf"
            )
            plt.savefig(
                f"figures/full_DMS/Spearman_comparison_plot_{split_method}_{part}.png",
                dpi=125,
            )

    # Repeat for the average
    score_path = Path(
        "results/ProteinGym/summary/Spearman",
        "DMS_substitutions_Spearman_DMS_level.csv",
    )

    for part in ["upper", "lower"]:
        df = pd.read_csv(score_path)
        x_min = df[model_names].min().min()
        df["DMS_id"] = df["DMS_id"].str.replace("Tsuboyama", "Tsuboyama")
        df = df.sort_values(by="Kermut", ascending=True)
        df = df.reset_index(drop=True)

        # Take top half of the dataframe
        if part == "lower":
            df = df.iloc[: len(df) // 2]
        else:
            df = df.iloc[len(df) // 2 :]

        df["rank"] = range(len(df))

        fig, ax = plt.subplots(figsize=(6, 16))
        max_vals = df[model_names].max(axis=1)
        ax.hlines(
            y=df["rank"],
            xmin=x_min - 0.05,
            xmax=max_vals,
            color="gray",
            linestyles="dashed",
            alpha=0.5,
            linewidth=0.5,
        )

        for i, model_name in enumerate(model_names):
            ax.scatter(
                x=df[model_name],
                y=df["rank"],
                label=model_name,
                color=colors[i],
                alpha=1,
                s=25,
            )
        ax.set_xlim(df[model_names].min().min() - 0.05, 1.0 + 0.05)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["DMS_id"])
        ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.set_ylabel("")
        ax.set_ylim(-1, len(df) + 1)
        ax.yaxis.set_tick_params(width=1, length=4)
        manual_x_ticks = [0, 0.25, 0.5, 0.75, 1.0]
        ax.set_xticks(manual_x_ticks)
        ax.yaxis.grid(False)

        # Add legend
        ax.legend(loc="upper right", fontsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)

        plt.tight_layout()
        plt.savefig(f"figures/full_DMS/Spearman_comparison_plot_average_{part}.pdf")
        plt.savefig(
            f"figures/full_DMS/Spearman_comparison_plot_average_{part}.png", dpi=125
        )
        plt.show()


if __name__ == "__main__":
    comparison_plot()
