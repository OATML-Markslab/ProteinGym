# fmt: off
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

"""
This is the script used to compute statistics for the supervised scoring models. 
It uses the score files output from the runs done for the ProteinNPT paper, and the 
code to run those supervised models is available in the ProteinNPT repo 
"""


def compute_bootstrap_standard_error_functional_categories(df, number_assay_reshuffle=10000, top_model="ProteinNPT"):
    """
    Computes the non-parametric bootstrap standard error for the mean estimate of a given performance metric (eg., Spearman, AUC) across DMS assays (ie., the sample standard deviation of the mean across bootstrap samples)
    """
    model_errors = {}
    for model_name, group in tqdm(df.groupby("model_name")):
        group_centered = group.subtract(df.loc[top_model],axis=0)
        mean_performance_across_samples = {}
        for category, group2 in group_centered.groupby("coarse_selection_type"):
            mean_performance_across_samples[category] = []
            for sample in range(number_assay_reshuffle):
                mean_performance_across_samples[category].append(group2.sample(frac=1.0, replace=True).mean(axis=0)) #Resample a dataset of the same size (with replacement) then take the sample mean
            mean_performance_across_samples[category]=pd.DataFrame(data=mean_performance_across_samples[category])
        categories = list(mean_performance_across_samples.keys())
        combined_averages = mean_performance_across_samples[categories[0]].copy()
        for category in categories[1:]:
            combined_averages += mean_performance_across_samples[category]
        combined_averages /= len(categories)
        model_errors[model_name] = combined_averages.std(ddof=1)
    return pd.DataFrame(model_errors).transpose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProteinGym supervised stats script')
    parser.add_argument('--input_scoring_file', type=str, help='Name of csv-file in long format containing all assay, model, split, and metric combinations.')
    parser.add_argument('--output_performance_file_folder', default='./outputs', type=str, help='Name of folder where to save performance analysis files')
    parser.add_argument('--DMS_reference_file_path', default="reference_files/DMS_substitutions.csv", type=str, help='Reference file with list of DMSs to consider')
    parser.add_argument('--top_model', type=str, default="ProteinNPT", help='Best performing model to compute standard errors relative to.')
    parser.add_argument('--number_assay_reshuffle', type=int, default=10000, help="Number of times to resample the data to compute bootstrap standard errors")
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    args = parser.parse_args()
    
    metrics = ["Spearman", "MSE"]
    score_column = {"Spearman": "Spearman", "MSE": "MSE"}
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/constants.json") as f:
        constants = json.load(f)
    if not os.path.exists(args.output_performance_file_folder):
        os.makedirs(args.output_performance_file_folder)

    ref_df = pd.read_csv(args.DMS_reference_file_path)
    ref_df["MSA_Neff_L_category"] = ref_df["MSA_Neff_L_category"].apply(lambda x: x[0].upper() + x[1:] if type(x) == str else x)
    score_df = pd.read_csv(args.input_scoring_file)
    score_df = score_df.merge(ref_df[["DMS_id","MSA_Neff_L_category","coarse_selection_type","taxon", "UniProt_ID"]],on="DMS_id",how="left")
    score_df = score_df[["model_name", "DMS_id", "UniProt_ID", "MSA_Neff_L_category", "coarse_selection_type", "taxon", "fold_variable_name", *score_column.values()]]
    if args.indel_mode:
        cv_schemes = ["fold_random_5"]
    else:
        cv_schemes = ["fold_random_5","fold_modulo_5","fold_contiguous_5"]
    for metric in metrics:
        if not os.path.exists(os.path.join(args.output_performance_file_folder,f"{metric}")):
            os.makedirs(os.path.join(args.output_performance_file_folder,f"{metric}"))
        output_folder = os.path.join(args.output_performance_file_folder,f"{metric}")
        all_DMS_perf = None 
        all_DMS_cv_schemes_perf = {cv_scheme:None for cv_scheme in cv_schemes}
        for DMS_id in tqdm(ref_df["DMS_id"].unique()):
            performance_all_DMS = {} 
            performance_all_DMS_cv_scheme = {cv_scheme:{} for cv_scheme in cv_schemes}
            score_subset = score_df[score_df['DMS_id']==DMS_id]
            models = score_subset["model_name"].unique()  
            for model in models:
                performance_all_DMS[model] = 0.0
            for cv_scheme in cv_schemes:
                cv_subset = score_subset[score_subset["fold_variable_name"]==cv_scheme]
                for model in models:
                    performance_all_DMS[model] += cv_subset[score_column[metric]][cv_subset["model_name"]==model].values[0]/len(cv_schemes)
                    performance_all_DMS_cv_scheme[cv_scheme][model] = cv_subset[score_column[metric]][cv_subset["model_name"]==model].values[0]
            performance_all_DMS = pd.DataFrame.from_dict(performance_all_DMS,orient="index").reset_index(names="model_names")
            performance_all_DMS.columns = ["model_names",DMS_id]
            performance_all_DMS_cv_scheme = {cv_scheme:pd.DataFrame.from_dict(performance_all_DMS_cv_scheme[cv_scheme],orient="index").reset_index(names="model_names") for cv_scheme in cv_schemes}
            for cv_scheme in cv_schemes:
                performance_all_DMS_cv_scheme[cv_scheme].columns = ["model_names",DMS_id]
            if all_DMS_perf is None:
                all_DMS_perf = performance_all_DMS
                all_DMS_cv_schemes_perf = {cv_scheme:performance_all_DMS_cv_scheme[cv_scheme] for cv_scheme in cv_schemes}
            else:
                all_DMS_perf = all_DMS_perf.merge(performance_all_DMS,on="model_names",how="inner")
                all_DMS_cv_schemes_perf = {cv_scheme:all_DMS_cv_schemes_perf[cv_scheme].merge(performance_all_DMS_cv_scheme[cv_scheme],on="model_names",how="inner") for cv_scheme in cv_schemes}
        all_DMS_perf = all_DMS_perf.set_index("model_names").transpose().reset_index(names="DMS_id")
        all_DMS_perf.columns = [constants["supervised_clean_names"][x] if x in constants["supervised_clean_names"] else x for x in all_DMS_perf.columns]
        if args.indel_mode:
            all_DMS_perf.round(3).to_csv(os.path.join(output_folder,f"DMS_indels_{metric}_DMS_level.csv"),index=False)
        else:
            all_DMS_perf.round(3).to_csv(os.path.join(output_folder,f"DMS_substitutions_{metric}_DMS_level.csv"),index=False)
        for cv_scheme in cv_schemes:
            all_DMS_cv_schemes_perf[cv_scheme] = all_DMS_cv_schemes_perf[cv_scheme].set_index("model_names").transpose().reset_index(names="DMS_id")
            all_DMS_cv_schemes_perf[cv_scheme].columns = [constants["supervised_clean_names"][x] if x in constants["supervised_clean_names"] else x for x in all_DMS_cv_schemes_perf[cv_scheme].columns]
            if args.indel_mode:
                all_DMS_cv_schemes_perf[cv_scheme].round(3).to_csv(os.path.join(output_folder,f"DMS_indels_{metric}_DMS_level_{cv_scheme}.csv"),index=False)
            else:
                all_DMS_cv_schemes_perf[cv_scheme].round(3).to_csv(os.path.join(output_folder,f"DMS_substitutions_{metric}_DMS_level_{cv_scheme}.csv"),index=False)

        def pivot_model_df(df, value_column, score_column):
            df = df[["model_name",value_column,score_column]]
            df = df.pivot(index="model_name",columns=value_column,values=score_column)
            return df

        # computing function groupings within CV schemes, then averaging them 
        all_summary_performance = None 
        for cv_scheme in cv_schemes:
            cv_subset = score_df[score_df["fold_variable_name"] == cv_scheme]
            if len(cv_subset) == 0:
                raise ValueError("No scores found for cross-validation scheme {}".format(cv_scheme))
            cv_uniprot_function = cv_subset.groupby(["model_name","UniProt_ID","coarse_selection_type"]).mean(numeric_only=True)
            bootstrap_standard_error = compute_bootstrap_standard_error_functional_categories(cv_uniprot_function,top_model=args.top_model,number_assay_reshuffle=args.number_assay_reshuffle)
            bootstrap_standard_error = bootstrap_standard_error[score_column[metric]].reset_index()
            bootstrap_standard_error.columns = ["model_name",f"Bootstrap_standard_error_{metric}"]
            cv_function_average = cv_uniprot_function.groupby(["model_name","coarse_selection_type"]).mean()
            cv_final_average = cv_function_average.groupby("model_name").mean()
            performance_by_MSA_depth = cv_subset.groupby(["model_name","UniProt_ID","MSA_Neff_L_category"]).mean(numeric_only=True).groupby(["model_name","MSA_Neff_L_category"]).mean(numeric_only=True)
            performance_by_taxon = cv_subset.groupby(["model_name","UniProt_ID","taxon"]).mean(numeric_only=True).groupby(["model_name","taxon"]).mean(numeric_only=True)
            performance_by_MSA_depth = pivot_model_df(performance_by_MSA_depth.reset_index(),"MSA_Neff_L_category",score_column[metric])
            performance_by_MSA_depth.columns = ['Low_MSA_depth','Medium_MSA_depth','High_MSA_depth']
            performance_by_taxon = pivot_model_df(performance_by_taxon.reset_index(),"taxon",score_column[metric])
            performance_by_taxon.columns = ['Taxa_Human','Taxa_Other_Eukaryote','Taxa_Prokaryote','Taxa_Virus']
            cv_function_average = pivot_model_df(cv_function_average.reset_index(),"coarse_selection_type",score_column[metric])
            cv_function_average.columns = ["Function_"+x for x in cv_function_average.columns]
            cv_final_average = cv_final_average.reset_index()[["model_name",score_column[metric]]].copy()
            cv_final_average.columns = ["model_name",f"Average_{metric}"]
            summary_performance = pd.merge(cv_final_average,performance_by_MSA_depth,on="model_name",how="inner")
            summary_performance = pd.merge(summary_performance,performance_by_taxon,on="model_name",how="inner")
            summary_performance = pd.merge(summary_performance,cv_function_average,on="model_name",how="inner")   
            summary_performance = pd.merge(summary_performance,bootstrap_standard_error,on="model_name",how="inner")
            if all_summary_performance is None:
                all_summary_performance = summary_performance.set_index("model_name")/len(cv_schemes)
                all_summary_performance[f"Average_{metric}_{cv_scheme}"] = all_summary_performance[f"Average_{metric}"]*len(cv_schemes)
            else:
                ignore_columns = [f"Average_{metric}_{cv_approach}" for cv_approach in cv_schemes]
                all_summary_performance[[column for column in all_summary_performance.columns if column not in ignore_columns]] += summary_performance.set_index("model_name")/len(cv_schemes)
                all_summary_performance[f"Average_{metric}_{cv_scheme}"] = summary_performance[f"Average_{metric}"].values
        all_summary_performance = all_summary_performance.reset_index(names="Model_name")
        if metric == "MSE":
            ascending = True
        else:
            ascending = False
        all_summary_performance.sort_values(by=f"Average_{metric}",ascending=ascending,inplace=True)
        all_summary_performance.index = range(1,len(all_summary_performance)+1)
        all_summary_performance.index.name = 'Model_rank'
        all_summary_performance = all_summary_performance.round(3)
        all_summary_performance["Model_name"] = all_summary_performance["Model_name"].apply(lambda x: constants["supervised_clean_names"][x] if x in constants["supervised_clean_names"] else x)
        all_summary_performance["References"] = all_summary_performance["Model_name"].apply(lambda x: constants["supervised_model_references"][x] if x in constants["supervised_model_references"] else "")
        all_summary_performance["Model details"] = all_summary_performance["Model_name"].apply(lambda x: constants["supervised_model_details"][x] if x in constants["supervised_model_details"] else "")
        all_summary_performance["Model type"] = all_summary_performance["Model_name"].apply(lambda x: constants["supervised_model_types"][x] if x in constants["supervised_model_types"] else "")
        if args.indel_mode:
            all_summary_performance["Function_Binding"] = "N/A"
            column_order = ["Model_name","Model type",f"Average_{metric}",f"Bootstrap_standard_error_{metric}",f"Average_{metric}_fold_random_5","Function_Activity","Function_Binding","Function_Expression","Function_OrganismalFitness","Function_Stability","Low_MSA_depth","Medium_MSA_depth","High_MSA_depth","Taxa_Human","Taxa_Other_Eukaryote","Taxa_Prokaryote","Taxa_Virus","References","Model details"]
        else:
            column_order = ["Model_name","Model type",f"Average_{metric}",f"Bootstrap_standard_error_{metric}",f"Average_{metric}_fold_random_5",f"Average_{metric}_fold_modulo_5",f"Average_{metric}_fold_contiguous_5","Function_Activity","Function_Binding","Function_Expression","Function_OrganismalFitness","Function_Stability","Low_MSA_depth","Medium_MSA_depth","High_MSA_depth","Taxa_Human","Taxa_Other_Eukaryote","Taxa_Prokaryote","Taxa_Virus","References","Model details"]
        all_summary_performance = all_summary_performance[column_order]
        if args.indel_mode:
            all_summary_performance.to_csv(os.path.join(output_folder,f"Summary_performance_DMS_indels_{metric}.csv"))
        else:
            all_summary_performance.to_csv(os.path.join(output_folder,f"Summary_performance_DMS_substitutions_{metric}.csv"))
