import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description='ProteinGym score merging')
    parser.add_argument('--DMS_assays_location', type=str, default='~/.cache/ProteinGym/ProteinGym/DMS_assays/substitutions',
                        help='Path to folder containing all model scores')
    parser.add_argument('--model_scores_location', type=str, default='~/.cache/ProteinGym/model_scores/supervised_substitutions',
                        help='Path to folder containing all model scores')
    parser.add_argument('--merged_scores_dir', type=str, default="~/.cache/ProteinGym/model_scores/supervised_substitutions/merged_scores",
                        help='Name of folder where all merged scores should be stored (in model_scores_location)')
    parser.add_argument('--mutation_type', default='substitutions',
                        type=str, help='Type of mutations (substitutions | indels)')
    parser.add_argument('--dataset', default='DMS', type=str,
                        help='Dataset to merge (DMS | clinical)')
    parser.add_argument('--DMS_reference_file', default='reference_files/DMS_substitutions.csv',
                        type=str, help='Path to reference file containing DMS assay information')
    parser.add_argument('--config_file', default='config.json',
                        type=str, help='Path to config file containing model information')
    args = parser.parse_args()
    # fmt: on

    # Determine paths
    DMS_assays_location = Path(args.DMS_assays_location)
    model_scores_location = Path(args.model_scores_location)
    merged_scores_dir = Path(args.merged_scores_dir)
    DMS_reference_file = Path(args.DMS_reference_file)
    config_file = Path(args.config_file)

    reference_file = pd.read_csv(DMS_reference_file)
    list_DMS = reference_file["DMS_id"]

    with open(config_file) as f:
        config = json.load(f)
    reference_field = f"model_list_supervised_{args.mutation_type}_{args.dataset}"

    if args.mutation_type == "indels":
        cv_schemes = ["fold_random_5"]
    else:
        cv_schemes = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

    list_models = config[reference_field].keys()
    for cv_scheme in cv_schemes:
        df_spearman = pd.DataFrame(columns=["DMS_id", *list_models])
        df_mse = pd.DataFrame(columns=["DMS_id", *list_models])
        df_spearman["DMS_id"] = reference_file["DMS_id"]
        df_mse["DMS_id"] = reference_file["DMS_id"]

        pbar = tqdm(enumerate(list_DMS), total=len(list_DMS))
        for DMS_index, DMS_id in pbar:
            pbar.set_description(f"Processing {cv_scheme} for DMS assay {DMS_id}")
            DMS_filename = reference_file["DMS_filename"][
                reference_file["DMS_id"] == DMS_id
            ].values[0]
            full_DMS_path = DMS_assays_location / DMS_filename
            if full_DMS_path.exists():
                DMS_file = pd.read_csv(full_DMS_path)
            else:
                print(f"Could not find DMS file {full_DMS_path}. Skipping.")
                continue
            if "mutated_sequence" not in DMS_file:
                DMS_file["mutated_sequence"] = DMS_file["mutant"]

            score_files = {}
            all_model_scores = DMS_file
            orig_DMS_length = len(all_model_scores)
            for model in list_models:
                mutant_merge_key = config[reference_field][model]["key"]
                input_score_name = config[reference_field][model]["input_score_name"]
                # Mutant merge key depends on the model for subs
                DMS_mutant_column = (
                    mutant_merge_key
                    if args.mutation_type == "substitutions"
                    else "mutated_sequence"
                )

                score_files[model] = pd.read_csv(
                    model_scores_location
                    / cv_scheme
                    / config[reference_field][model]["location"]
                    / f"{DMS_id}.csv"
                )
                score_files[model] = score_files[model].rename(
                    columns={"sequence": "mutated_sequence", input_score_name: model}
                )
                score_files[model] = score_files[model][[mutant_merge_key, model]]
                score_files[model].drop_duplicates(inplace=True)
                score_files[model] = (
                    score_files[model].groupby(mutant_merge_key).mean().reset_index()
                )
                # check that score_files[model][mutant_merge_key] and all_model_scores[DMS_mutant_column] are the same
                if (
                    set(score_files[model][mutant_merge_key])
                    & set(all_model_scores[DMS_mutant_column])
                    == set()
                ):
                    print(
                        f"Warning: No overlap on mutants for {DMS_id} with model {model}. Skipping"
                    )
                    continue
                elif set(score_files[model][mutant_merge_key]) < set(
                    all_model_scores[DMS_mutant_column]
                ):
                    # print difference between two key sets
                    print(
                        "WARNING: {model} and {DMS_id} do not have the same mutants. Skipping."
                    )
                    continue

                score_files[model] = score_files[model].rename(
                    columns={mutant_merge_key: DMS_mutant_column}
                )
                all_model_scores = pd.merge(
                    all_model_scores,
                    score_files[model],
                    on=DMS_mutant_column,
                    how="left",
                )

                if len(all_model_scores) != orig_DMS_length:
                    print(
                        f"WARNING: Merge on {model} for {DMS_id} changed length. mutant_merge_keys are likely different between them."
                    )
                    print(f"Length DMS: {orig_DMS_length}".format(orig_DMS_length))
                    print(f"Length {model}: {len(score_files[model])}")
                    print(
                        f"Length all_model_score unique keys: {len(all_model_scores[mutant_merge_key].unique())}"
                    )
                    print(
                        f"Length DMS unique keys: {len(score_files[model][mutant_merge_key].unique())}"
                    )
                    print(f"Length merged file: {format(len(all_model_scores))}")
                    continue
            num_mutants_expected = reference_file[reference_file["DMS_id"] == DMS_id][
                "DMS_number_single_mutants"
            ].values[0]
            if len(all_model_scores) != num_mutants_expected:
                print(
                    f"Warning: Insufficient mutants for {DMS_id}: {len(all_model_scores)}, expected {num_mutants_expected}. Original DMS file length: {orig_DMS_length}"
                )
            # Compute metrics
            for model in list_models:
                spearman = (
                    all_model_scores[[model, "DMS_score"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                )
                mse = (
                    (all_model_scores[model] - all_model_scores["DMS_score"]) ** 2
                ).mean()
                df_spearman.loc[DMS_index, model] = spearman
                df_mse.loc[DMS_index, model] = mse

        cv_scores_path = merged_scores_dir / "Spearman" / f"{cv_scheme}.csv"
        cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
        df_spearman.to_csv(cv_scores_path, index=False)

        cv_scores_path = merged_scores_dir / "MSE" / f"{cv_scheme}.csv"
        cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
        df_mse.to_csv(cv_scores_path, index=False)


if __name__ == "__main__":
    main()
