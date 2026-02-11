import os

import click
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


def output_score(assays_dir: str, outputs_dir: str, index_file: str) -> None:

    index_df = pd.read_csv(index_file)
    spearmans = {}

    for DMS_id in tqdm(index_df["DMS_id"].unique(), desc="Scoring", ncols=80):
        output_file = os.path.join(outputs_dir, f"{DMS_id}.tmp")
        if not os.path.exists(output_file):
            continue

        assay_file = os.path.join(assays_dir, f"{DMS_id}.csv")
        assay_df = pd.read_csv(assay_file)
        output_df = pd.read_csv(output_file)

        # Remove the DMS_id from the sequence id and keep only sequence index within the assay
        output_df["sequence_id"] = output_df["sequence_id"].apply(lambda x: x.split("+")[1]).astype(int)
        output_df = output_df.sort_values(by="sequence_id").set_index("sequence_id")

        assert len(assay_df) == len(output_df), "Assay and output files must have the same number of rows"

        # merge the pandas df vertically
        score_df = pd.merge(output_df, assay_df, left_index=True, right_index=True)[["mutant", "mutated_sequence", "log_likelihood"]]
        score_df.to_csv(os.path.join(outputs_dir, f"{DMS_id}.csv"), index=False)
        y_true = assay_df["DMS_score"].values
        y_pred = output_df["log_likelihood"].values

        spearman_corr, _ = spearmanr(y_true, y_pred)
        spearmans[DMS_id] = spearman_corr

    index_df["spearman_corr"] = index_df["DMS_id"].apply(lambda x: spearmans.get(x, None))
    print(index_df[index_df["spearman_corr"].notna()][["DMS_id", "spearman_corr"]])

    spearman_df = index_df[index_df["spearman_corr"].notna()]
    aggregated_spearman = (
        spearman_df.groupby(["coarse_selection_type", "UniProt_ID"])["spearman_corr"]
        .mean()
        .groupby("coarse_selection_type")
        .mean()
        .mean()
        .item()
    )
    print(f"Aggregated Spearman: {aggregated_spearman}")

