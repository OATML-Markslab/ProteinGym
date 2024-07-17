"""Script to compute all calibration metrics and plot them across domains.
Assumes metrics have been computed via src/process_results/calibration_results.py."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import COLORS


def main():
    model_names = [
        "kermut",
        "kermut_no_m_constant_mean",
    ]
    split_methods = [
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
        "fold_rand_multiples",
        "domain",
    ]

    # Load results
    df = pd.DataFrame()
    for model_name in model_names:
        for split_method in split_methods:
            df_ = pd.read_csv(
                f"results/calibration_metrics/{model_name}_{split_method}.csv"
            )
            df_["model_name"] = model_name
            if df.empty:
                df = df_
            else:
                df = pd.concat([df, df_])

    name_map = {
        "kermut": "Kermut",
        "kermut_no_m_constant_mean": "Baseline GP",
    }
    fold_map = {
        "fold_random_5": "Random",
        "fold_modulo_5": "Modulo",
        "fold_contiguous_5": "Contiguous",
        "fold_rand_multiples": "Multiples",
        "domain": "Extrapolation",
    }
    # Rename CV column to r"$c_v$
    df = df.rename(columns={"CV": r"$c_v$"})

    df = df.drop(columns=["fold"])
    df["model_name"] = df["model_name"].map(name_map)
    df["fold_variable_name"] = df["fold_variable_name"].map(fold_map)

    # Aggregate
    df_agg = df.groupby(
        ["fold_variable_name", "model_name", "DMS_id"], as_index=False
    ).mean(numeric_only=True)
    fold_order = ["Random", "Modulo", "Contiguous", "Multiples", "Extrapolation"]
    df_agg["fold_variable_name"] = pd.Categorical(
        df_agg["fold_variable_name"], categories=fold_order, ordered=True
    )
    df_agg = df_agg.sort_values("fold_variable_name")

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"

    kwargs = {
        "x": "fold_variable_name",
        "hue": "model_name",
        "hue_order": ["Kermut", "Baseline GP"],
        "palette": COLORS,
        "saturation": 1,
        "errorbar": "se",
        "capsize": 0.25,
        "width": 0.8,
        "err_kws": {"linewidth": 1.5},
    }

    fig, ax = plt.subplots(
        1,
        3,
        **{"figsize": (7.5, 2.5), "sharex": "all"},
    )
    sns.barplot(data=df_agg, y="ECE", ax=ax[0], **kwargs, legend=False)
    sns.barplot(data=df_agg, y="ENCE", ax=ax[1], **kwargs)
    sns.barplot(data=df_agg, y=r"$c_v$", ax=ax[2], **kwargs, legend=False)

    # Collect legend and place under central subplot
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.075)
    )
    ax[1].get_legend().remove()
    for a in ax:
        a.set_xlabel("")
        labels = [item.get_text() for item in a.get_xticklabels()]
        a.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        a.set_title(a.get_ylabel())
        a.set_ylabel("")
        a.tick_params(axis="both", which="major", labelsize=8)

    plt.tight_layout()
    plt.savefig(
        "figures/calibration/calibration_metrics_barplot.png",
        dpi=125,
        bbox_inches="tight",
    )
    plt.savefig(
        "figures/calibration/calibration_metrics_barplot.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
