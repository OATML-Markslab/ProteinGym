import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import COLORS


def DMS_histogram_multiples():
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"

    df_ref = pd.read_csv("data/DMS_substitutions.csv")
    # Filter data for fair comparison
    df_ref = df_ref[df_ref["includes_multiple_mutants"]]
    df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
    df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]

    df = pd.DataFrame()
    for dataset in df_ref["DMS_id"].tolist():
        df_ = pd.read_csv(f"data/substitutions_multiples/{dataset}.csv")
        if df.empty:
            df = df_
        else:
            df = pd.concat([df, df_])

    df["Number of mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    sns.histplot(
        data=df,
        x="DMS_score",
        hue="Number of mutations",
        bins=50,
        kde=True,
        ax=ax,
        palette=COLORS,
    )
    ax.set_xlabel("DMS score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/ProteinGym/DMS_histogram_multiples.png", dpi=125)
    plt.savefig("figures/ProteinGym/DMS_histogram_multiples.pdf")


if __name__ == "__main__":
    DMS_histogram_multiples()
