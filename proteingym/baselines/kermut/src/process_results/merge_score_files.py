"""This script merges the predictions across all models, assays, and splits together with reference results from ProteinGym"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


def main(ablation: bool = False):
    # Load reference file for assay details and ProteinGym results
    df_ref = pd.read_csv("data/DMS_substitutions.csv")
    df_baseline_results = pd.read_csv(
        "results/baselines/DMS_supervised_substitutions_scores.csv"
    )

    results_dir = Path("results/predictions")
    df_results = pd.DataFrame(
        columns=["fold_variable_name", "MSE", "Spearman", "assay_id", "model_name"]
    )

    # By default, use main split schemes
    methods = [
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
    ]

    if not ablation:
        # Models are renamed for better readability
        # The results should be available for all 217 assays and all 3 splits
        new_names = {
            "kermut": "Kermut",
        }
        out_path = Path("results", "merged_scores.csv")
        intermediary_path = Path("results", "kermut_scores.csv")

    else:
        # Models are renamed for better readability
        new_names = {
            "kermut": "Kermut",
            "kermut_constant_mean": "Kermut (const. mean)",
            "kermut_no_m": "Kermut (no m)",
            "kermut_no_m_constant_mean": "Kermut (no m, const. mean)",
            "kermut_no_d": "Kermut (no distance)",
            "kermut_no_p": "Kermut (no p)",
            "kermut_no_g": "Kermut (no global)",
            "kermut_no_h": "Kermut (no Hellinger)",
            "kermut_no_hp": "Kermut (no Hellinger/p)",
            "kermut_proteinmpnn": "Kermut (ProteinMPNN)",
            "kermut_trancepteve": "Kermut (TranceptEVE)",
            "kermut_esmif1": "Kermut (ESM-IF1)",
            "kermut_eve": "Kermut (EVE)",
            "kermut_gemme": "Kermut (GEMME)",
            "kermut_vespa": "Kermut (VESPA)",
        }
        out_path = Path("results", "merged_ablation_scores.csv")

        # The results should be available for all 174 assays and all 3 splits:
        df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
        df_ref = df_ref[df_ref["DMS_id"] != "BRCA2_HUMAN_Erwood_2022_HEK293T"]

    models = list(new_names.keys())
    dms_finished = []
    for dataset in tqdm(df_ref["DMS_id"].unique()):
        # Process each dataset separately. If results are missing at any point for a dataset, it is skipped.
        try:
            df_dms = pd.DataFrame()
            for model in models:
                for method in methods:
                    df = pd.read_csv(results_dir / dataset / f"{model}_{method}.csv")
                    # Spearman correlation and MSE across CV folds
                    corr, _ = spearmanr(df["y"].values, df["y_pred"].values)
                    mse = ((df["y"] - df["y_pred"]) ** 2).mean()
                    df = pd.DataFrame(
                        [
                            {
                                "assay_id": dataset,
                                "model_name": model,
                                "fold_variable_name": method,
                                "Spearman": corr,
                                "MSE": mse,
                            }
                        ]
                    )

                    if df_dms.empty:
                        df_dms = df
                    else:
                        df_dms = pd.concat([df_dms, df])

            dms_finished.append(dataset)
            if df_results.empty:
                df_results = df_dms
            else:
                df_results = pd.concat([df_results, df_dms])

        except FileNotFoundError:
            print(f"File not found: {dataset}/{model}_{method}.csv")
            continue

    # Extract UniProt ID from reference file
    df_avg = pd.merge(
        left=df_results,
        right=df_ref[["DMS_id", "UniProt_ID"]],
        how="left",
        left_on="assay_id",
        right_on="DMS_id",
    )

    # Rename for concatenation
    df_avg = df_avg.rename(
        columns={
            "UniProt_ID": "UniProt_id",
            "model_name": "model_name_raw",
            "Spearman": "Spearman_fitness",
            "MSE": "loss_fitness",
        }
    )
    df_avg["model_name"] = df_avg["model_name_raw"].map(new_names)
    df_avg = df_avg[
        [
            "model_name",
            "model_name_raw",
            "assay_id",
            "UniProt_id",
            "fold_variable_name",
            "loss_fitness",
            "Spearman_fitness",
        ]
    ]
    
    if not ablation:
        df_avg.to_csv(intermediary_path, index=False)

    # Concatenate with reference results
    df_cat = pd.concat([df_avg, df_baseline_results])

    # Inconsistent assay_id names
    assay_id_mapping = {
        "A0A140D2T1_ZIKV_Sourisseau_growth_2019": "A0A140D2T1_ZIKV_Sourisseau_2019",
        "A4D664_9INFA_Soh_CCL141_2019": "A4D664_9INFA_Soh_2019",
        "VKOR1_HUMAN_Chiasson_activity_2020": "VKOR1_HUMAN_Chiasson_2020_activity",
        "VKOR1_HUMAN_Chiasson_abundance_2020": "VKOR1_HUMAN_Chiasson_2020_abundance",
        "SPIKE_SARS2_Starr_bind_2020": "SPIKE_SARS2_Starr_2020_binding",
        "SPIKE_SARS2_Starr_expr_2020": "SPIKE_SARS2_Starr_2020_expression",
        "RL401_YEAST_Mavor_2016": "RL40A_YEAST_Mavor_2016",
        "RL401_YEAST_Roscoe_2013": "RL40A_YEAST_Roscoe_2013",
        "RL401_YEAST_Roscoe_2014": "RL40A_YEAST_Roscoe_2014",
        "P53_HUMAN_Giacomelli_WT_Nutlin_2018": "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
        "P53_HUMAN_Giacomelli_NULL_Nutlin_2018": "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
        "SRC_HUMAN_Ahler_CD_2019": "SRC_HUMAN_Ahler_2019",
        "CAPSD_AAV2S_Sinai_substitutions_2021": "CAPSD_AAV2S_Sinai_2021",
        "HXK4_HUMAN_Gersing_2022": "HXK4_HUMAN_Gersing_2022_activity",
        "CP2C9_HUMAN_Amorosi_activity_2021": "CP2C9_HUMAN_Amorosi_2021_activity",
        "CP2C9_HUMAN_Amorosi_abundance_2021": "CP2C9_HUMAN_Amorosi_2021_abundance",
        "DYR_ECOLI_Thompson_plusLon_2019": "DYR_ECOLI_Thompson_2019",
        "GCN4_YEAST_Staller_induction_2018": "GCN4_YEAST_Staller_2018",
        "TPOR_HUMAN_Bridgford_S505N_2020": "TPOR_HUMAN_Bridgford_2020",
        "R1AB_SARS2_Flynn_growth_2022": "R1AB_SARS2_Flynn_2022",
        "P53_HUMAN_Giacomelli_NULL_Etoposide_2018": "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
        "NRAM_I33A0_Jiang_standard_2016": "NRAM_I33A0_Jiang_2016",
        "MTH3_HAEAE_Rockah-Shmuel_2015": "MTH3_HAEAE_RockahShmuel_2015",
        "B3VI55_LIPST_Klesmith_2015": "LGK_LIPST_Klesmith_2015",
    }

    # Replace assay_id with mapping ignoring the ones not in the mapping
    df_cat["assay_id"] = df_cat["assay_id"].apply(lambda x: assay_id_mapping.get(x, x))

    # If using only a subset of splits
    if len(methods) != 3:
        df_cat = df_cat[df_cat["fold_variable_name"].isin(methods)]

    # Keep only finished_dms
    df_cat = df_cat[df_cat["assay_id"].isin(dms_finished)]

    # Check that all assays have the same number of models
    assert len(df_cat["assay_id"].value_counts().unique()) == 1
    assert len(df_cat["model_name"].value_counts().unique()) == 1

    df_cat.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Whether to merge the ablation results or the full results",
    )
    args = parser.parse_args()
    main(ablation=args.ablation)
