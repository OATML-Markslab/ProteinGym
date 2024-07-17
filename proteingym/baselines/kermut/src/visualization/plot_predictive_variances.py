"""Creates violin plots showing mean preditive variances over domains"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import COLORS


def domain_comparison(model_names: List[str], orientation: str = "horizontal"):
    # Plot params
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    fig_dir = Path("figures", "calibration")

    # Filter data
    df_ref = pd.read_csv("data/DMS_substitutions.csv")
    df_ref = df_ref.loc[
        (
            (df_ref["includes_multiple_mutants"])
            & (df_ref["DMS_total_number_mutants"] < 7500)
        )
    ]
    df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]

    methods = [
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
        "fold_rand_multiples",
        "domain",
    ]

    domain_mapping = {
        "fold_random_5_1": r"1M$\rightarrow$1M" + " (rand.)",
        "fold_modulo_5_1": r"1M$\rightarrow$1M" + " (mod.)",
        "fold_contiguous_5_1": r"1M$\rightarrow$1M" + " (cont.)",
        "fold_rand_multiples_1": r"1M/2M$\rightarrow$1M" + " (rand.)",
        "fold_rand_multiples_2": r"1M/2M$\rightarrow$2M" + " (rand.)",
        # "domain_1": r"1M$\rightarrow$1M" + " (rand.)",
        "domain_2": r"1M$\rightarrow$2M" + " (extrap.)",
    }

    for i, model_name in enumerate(model_names):
        df_avg = pd.DataFrame()
        for method in methods:
            df = pd.DataFrame()
            for dataset in df_ref["DMS_id"].unique():
                df_ = pd.read_csv(
                    Path(
                        "results/ProteinGym/predictions",
                        dataset,
                        f"{model_name}_{method}.csv",
                    )
                )
                df_["dataset"] = dataset
                df = pd.concat([df, df_])

            df["domain"] = method
            df["model_name"] = model_name
            df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
            df = df.groupby(
                ["domain", "dataset", "model_name", "n_mutations"], as_index=False
            ).mean(numeric_only=True)
            if df_avg.empty:
                df_avg = df
            else:
                df_avg = pd.concat([df_avg, df])

        df_avg["n_mutations"] = df_avg["n_mutations"].astype(int)
        df_avg["domain"] = df_avg["domain"] + "_" + df_avg["n_mutations"].astype(str)

        # Plotting
        if orientation == "vertical":
            fig, ax = plt.subplots(
                **{"figsize": (2.5, 7.5), "sharex": "all", "sharey": "all"},
            )
            sns.violinplot(
                data=df_avg[df_avg["domain"] != "domain_1"],
                y="domain",
                x="y_var",
                hue="domain",
                palette=COLORS[: len(domain_mapping)],
                ax=ax,
                saturation=1,
                hue_order=domain_mapping.keys(),
                legend="full",
            )
            ax.set_ylabel("")
            ax.set_xlabel(r"Predictive variance $\hat{\sigma}^2$", fontsize=12)

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [domain_mapping[label] for label in labels]
            ax.legend(
                handles,
                new_labels,
                title="Domain",
                title_fontsize=12,
                fontsize=10,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.05),
            )
            ax.set_yticks([])
            for xtick in ax.get_xticklabels():
                xtick.set_fontsize(10)

            plt.tight_layout()
            fig_path = fig_dir / f"{model_name}_domain_variance_summary_vertical"
            plt.savefig(f"{fig_path}.png", dpi=125)
            plt.savefig(f"{fig_path}.pdf")
            plt.close()
        else:
            fig, ax = plt.subplots(
                **{"figsize": (7.5, 2.5), "sharex": "all", "sharey": "all"},
            )
            sns.violinplot(
                data=df_avg[df_avg["domain"] != "domain_1"],
                x="domain",
                y="y_var",
                hue="domain",
                palette=COLORS[: len(domain_mapping)],
                ax=ax,
                saturation=1,
                hue_order=domain_mapping.keys(),
                legend="full",
            )
            ax.set_xlabel("")
            ax.set_ylabel(r"Predictive variance $\hat{\sigma}^2$", fontsize=12)

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [domain_mapping[label] for label in labels]
            ax.legend(
                handles,
                new_labels,
                title="Domain",
                title_fontsize=12,
                fontsize=10,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
            )
            ax.set_xticks([])
            for ytick in ax.get_yticklabels():
                ytick.set_fontsize(10)

            plt.tight_layout()
            fig_path = fig_dir / f"{model_name}_domain_variance_summary_horizontal"
            plt.savefig(f"{fig_path}.png", dpi=125)
            plt.savefig(f"{fig_path}.pdf")
            plt.close()


if __name__ == "__main__":
    model_names = ["kermut", "kermut_no_m_constant_mean"]
    for model in model_names:
        domain_comparison(model_names, orientation="horizontal")
