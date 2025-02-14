import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm



def main():
    # fmt: off
    parser = argparse.ArgumentParser(description='ProteinGym score merging')
    parser.add_argument('--DMS_assays_location', type=str, default='~/.cache/ProteinGym/ProteinGym/DMS_assays/substitutions', help='Path to folder containing all model scores')
    parser.add_argument('--model_scores_location', type=str, default='~/.cache/ProteinGym/model_scores/supervised_substitutions', help='Path to folder containing all model scores')
    parser.add_argument('--merged_scores_dir', type=str, default="~/.cache/ProteinGym/model_scores/supervised_substitutions/merged_scores", help='Name of folder where all merged scores should be stored (in model_scores_location)')
    parser.add_argument('--mutation_type', default='substitutions', type=str, help='Type of mutations (substitutions | indels)')
    parser.add_argument('--dataset', default='DMS', type=str, help='Dataset to merge (DMS | clinical)')
    parser.add_argument('--DMS_reference_file', default='reference_files/DMS_substitutions.csv', type=str, help='Path to reference file containing DMS assay information')
    parser.add_argument('--config_file', default='config.json', type=str, help='Path to config file containing model information')
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
    list_models = config[reference_field].keys()

    if args.mutation_type == "indels":
        cv_schemes = ["fold_random_5"]
    else:
        cv_schemes = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

    df_merged = pd.DataFrame(
        columns=["DMS_id", "model_name", "fold_variable_name", "Spearman", "MSE"]
    )

    for cv_scheme in cv_schemes:
        df_spearman = reference_file[["DMS_id"]].copy()
        df_mse = reference_file[["DMS_id"]].copy()
        df_spearman[list(list_models)] = None
        df_mse[list(list_models)] = None
        merged_scores_dir_cv = merged_scores_dir / cv_scheme
        merged_scores_dir_cv.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(enumerate(list_DMS), total=len(list_DMS))
        for DMS_index, DMS_id in pbar:
            pbar.set_description(f"Processing {cv_scheme} for DMS assay {DMS_id}")
            DMS_filename = reference_file["DMS_filename"][
                reference_file["DMS_id"] == DMS_id
            ].values[0]
            DMS_filename = reference_file.loc[
                reference_file["DMS_id"] == DMS_id, "DMS_filename"
            ].values[0]
            full_DMS_path = DMS_assays_location / DMS_filename
            if full_DMS_path.exists():
                DMS_file = pd.read_csv(full_DMS_path)
            else:
                print(f"Could not find DMS file {full_DMS_path}. Skipping.")
                continue

            if "mutated_sequence" not in DMS_file:
                DMS_file["mutated_sequence"] = DMS_file["mutant"]

            # all_model_scores = DMS_file[
            #     ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin", cv_scheme]
            # ]
            orig_DMS_length = len(DMS_file)
            for model in list_models:
                mutant_merge_key = config[reference_field][model]["key"]
                input_score_name = config[reference_field][model]["input_score_name"]
                label_name = config[reference_field][model]["label_name"]
                # Mutant merge key depends on the model for subs
                DMS_mutant_column = (
                    mutant_merge_key
                    if args.mutation_type == "substitutions"
                    else "mutated_sequence"
                )

                score_path = (
                    model_scores_location
                    / cv_scheme
                    / config[reference_field][model]["location"]
                    / f"{DMS_id}.csv"
                )
                if not score_path.exists():
                    print(f"Could not find score file {score_path}. Skipping.")
                    continue

                df_scores = pd.read_csv(score_path)
                df_scores = df_scores.rename(
                    columns={"sequence": "mutated_sequence", input_score_name: model}
                )
                df_scores = df_scores[[mutant_merge_key, model, label_name]]
                df_scores = (
                    df_scores.groupby(mutant_merge_key, as_index=False)
                    .mean()
                    .reset_index(drop=True)
                )
                # check that score_files[model][mutant_merge_key] and all_model_scores[DMS_mutant_column] are the same
                if (
                    set(df_scores[mutant_merge_key]) & set(DMS_file[DMS_mutant_column])
                    == set()
                ):
                    print(
                        f"Warning: No overlap on mutants for {DMS_id} with model {model}. Skipping"
                    )
                    continue
                elif set(df_scores[mutant_merge_key]) != set(
                    DMS_file[DMS_mutant_column]
                ):
                    print(
                        "WARNING: {model} and {DMS_id} do not have the same mutants for {cv_scheme} scheme. Skipping."
                    )
                    continue

                df_scores = df_scores.rename(
                    columns={mutant_merge_key: DMS_mutant_column}
                )
                # all_model_scores = pd.merge(
                #     all_model_scores,
                #     df_scores[[DMS_mutant_column, model]],
                #     on=DMS_mutant_column,
                #     how="left",
                # )

                if len(df_scores) != orig_DMS_length:
                    print(
                        f"WARNING: Merge on {model} for {DMS_id} changed length. mutant_merge_keys are likely different between them."
                    )
                    print(f"Length DMS: {orig_DMS_length}")
                    print(f"Length {model}: {len(df_scores)}")
                    continue
                num_mutants_expected = reference_file.loc[
                    reference_file["DMS_id"] == DMS_id, "DMS_number_single_mutants"
                ].values[0]
                if len(df_scores) != num_mutants_expected:
                    print(
                        f"Warning: Insufficient mutants for {DMS_id}: {len(DMS_file)}, expected {num_mutants_expected}. Original DMS file length: {orig_DMS_length}"
                    )

                # Compute metrics
                spearman = df_scores[label_name].corr(
                    df_scores[model], method="spearman"
                )
                mse = ((df_scores[label_name] - df_scores[model]) ** 2).mean()

                df_spearman.loc[df_spearman["DMS_id"] == DMS_id, model] = spearman
                df_mse.loc[df_mse["DMS_id"] == DMS_id, model] = mse
            # all_model_scores.to_csv(merged_scores_dir_cv / f"{DMS_id}.csv", index=False)

        df_spearman = df_spearman.melt(
            id_vars=["DMS_id"], var_name="model_name", value_name="Spearman"
        )
        df_mse = df_mse.melt(
            id_vars=["DMS_id"], var_name="model_name", value_name="MSE"
        )
        df = pd.merge(df_spearman, df_mse, on=["DMS_id", "model_name"])
        df["fold_variable_name"] = cv_scheme
        df_merged = pd.concat([df_merged, df])

    df_merged = df_merged.sort_values(
        by=["DMS_id", "model_name", "fold_variable_name"]
    ).reset_index(drop=True)
    cv_scores_path = merged_scores_dir / "merged_scores.csv"
    cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(cv_scores_path, index=False)


if __name__ == "__main__":
    main()
