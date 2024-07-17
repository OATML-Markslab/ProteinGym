"""Creates boxplots showing slope and intercept of error-based calibration curves"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src import COLORS


def calibration_boxplot():
    """
    Generates boxplots over intercept and slope for linear fits to error-based calibration curves.
    Assumes metric have been precomputed.
    """
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "serif"
    split_schemes = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    model_names = ["kermut", "kermut_no_m_constant_mean"]
    calibration_path = Path("results/calibration_metrics")

    for model_name in model_names:
        df = pd.DataFrame()
        for split_scheme in split_schemes:
            df_ = pd.read_csv(
                calibration_path / f"{model_name}_error-based_{split_scheme}.csv"
            )
            df_ = df_.groupby(["fold", "DMS_id"]).apply(
                lambda x: sm.OLS(x["RMSE"], sm.add_constant(x["RMV"])).fit().params
            )
            df_ = df_.reset_index(drop=True)
            df[f"{split_scheme}_intercept"] = df_["const"]
            df[f"{split_scheme}_slope"] = df_["RMV"]

        fig, ax = plt.subplots(2, 1, figsize=(4, 4))
        for j, coef in enumerate(["intercept", "slope"]):
            df_long = pd.melt(
                df,
                value_vars=[f"{split_scheme}_{coef}" for split_scheme in split_schemes],
                var_name="split_scheme",
                value_name=coef,
            )

            sns.boxplot(
                data=df_long,
                ax=ax[j],
                x="split_scheme",
                y=coef,
                hue="split_scheme",
                palette=COLORS,
                showfliers=False,
                saturation=1,
                width=0.5,
            )
            if coef == "slope":
                ax[j].axhline(1, color="black", linestyle="--", alpha=0.3)
            else:
                ax[j].axhline(0, color="black", linestyle="--", alpha=0.3)
            if j == 1:
                ax[j].set_xticklabels(
                    [
                        split_scheme.split("_")[1].capitalize()
                        for split_scheme in split_schemes
                    ]
                )
                ax[j].set_ylim(-3, 4.5)
            else:
                ax[j].set_xticklabels([])
                ax[j].set_ylim(-2.2, 2.5)
            ax[j].set_xlabel("")
            ax[j].set_ylabel(coef.capitalize())

        plt.tight_layout()

        plt.savefig(
            f"figures/calibration/boxplot_eb_calib_{model_name}.pdf",
            bbox_inches="tight",
        )
        plt.savefig(
            f"figures/calibration/boxplot_eb_calib_{model_name}.png",
            bbox_inches="tight",
            dpi=125,
        )


if __name__ == "__main__":
    calibration_boxplot()
