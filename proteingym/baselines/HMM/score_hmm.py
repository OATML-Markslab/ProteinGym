import argparse 
import pandas as pd 
import os 
import numpy as np 

"""
This script scores a folder of variants using the HMM model.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score a folder of variants using the HMM model')   
    #We may pass in all required information about the dataset via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--DMS_reference_file', type=str, help='Path to reference file with list of target sequences (and filepaths to their associated variants) to score')
    parser.add_argument('--DMS_index', type=int, help='Index of sequence and variants to score in reference file')
    parser.add_argument("--hmmer_path", type=str, help="Path to hmmer installation")
    parser.add_argument('--DMS_folder', type=str, help='Path to folder that contains the protein variants for each target sequence')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument("--intermediate_outputs_folder", type=str, default="./intermediate_outputs", help="Path to folder to write intermediate outputs to")
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')

    #Fields to be passed manually if reference file is not used
    parser.add_argument('--target_seq', default=None, type=str, help='Full wild type sequence that is mutated in the experiment')
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of experiment file')
    parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')
    parser.add_argument("--mutant_column",default="mutant",type=str)
    parser.add_argument("--mutated_sequence_column",default="mutated_sequence",type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_scores_folder):
        os.makedirs(args.output_scores_folder)
    if not os.path.exists(args.intermediate_outputs_folder):    
        os.makedirs(args.intermediate_outputs_folder)
    if not os.path.exists(args.intermediate_outputs_folder + os.sep + "hmm"):
        os.mkdir(args.intermediate_outputs_folder + os.sep + "hmm")
    if not os.path.exists(args.intermediate_outputs_folder + os.sep + "fa"):
        os.mkdir(args.intermediate_outputs_folder + os.sep + "fa")
    if not os.path.exists(args.intermediate_outputs_folder + os.sep + "a2m"):
        os.mkdir(args.intermediate_outputs_folder + os.sep + "a2m")
    if not os.path.exists(args.intermediate_outputs_folder + os.sep + "raw_scores"):
        os.mkdir(args.intermediate_outputs_folder + os.sep + "raw_scores")
    hmm_path = args.intermediate_outputs_folder + os.sep + "hmm"
    fa_path = args.intermediate_outputs_folder + os.sep + "fa"
    raw_score_path = args.intermediate_outputs_folder + os.sep + "raw_scores"
    
    if args.DMS_reference_file:
        DMS_mapfile = pd.read_csv(args.DMS_reference_file)
        list_target_sequences = DMS_mapfile["DMS_id"].tolist()
        DMS_id=list_target_sequences[args.DMS_index]
        print(f"Computing HMM scores for {DMS_id})")
        target_seq = DMS_mapfile["target_seq"][DMS_mapfile["DMS_id"]==DMS_id].values[0].upper()
        DMS_file_name = DMS_mapfile["DMS_filename"][DMS_mapfile["DMS_id"]==DMS_id].values[0]
        MSA_data_file = DMS_mapfile["MSA_filename"].tolist()[args.DMS_index]
        MSA_data_file = args.MSA_folder + os.sep + MSA_data_file if type(MSA_data_file) != float else None
        MSA_start = DMS_mapfile["MSA_start"].tolist()[args.DMS_index]
        MSA_start = int(MSA_start) - 1 if not np.isnan(MSA_start) else None # MSA_start typically based on 1-indexing 
        MSA_end = DMS_mapfile["MSA_end"].tolist()[args.DMS_index]
        MSA_end = int(MSA_end) if not np.isnan(MSA_end) else None
    else:
        target_seqs=args.target_seq
        DMS_file_names=args.DMS_file_name
        DMS_id = DMS_file_names.split(".")[0]
        MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
        MSA_start = args.MSA_start - 1 # MSA_start based on 1-indexing
        MSA_end = args.MSA_end
    # mutated_sequence_column = "mutant"
    mutated_sequence_column = "mutated_sequence"
    # checking all a3m files in alignment and calling reformat if an a2m or hmm file are not already generated 
    print("Checking for alignment files and reformatting a3m files if needed")
    if MSA_data_file is None:
        print(f"Alignment required for HMM scoring")
        exit()
    basename = os.path.splitext(MSA_data_file)[0].split(os.sep)[-1]
    # Building HMMs from a2ms if HMMs do not already exist
    print("Building HMMs if they don't already exist")
    if not os.path.exists(hmm_path + os.sep + basename + ".hmm"):
        os.system(f"{args.hmmer_path}" + os.sep + "bin" + os.sep + 'hmmbuild --amino ' + f"{hmm_path + os.sep + basename + '.hmm'} {MSA_data_file}")

    print("Writing out variants to fasta file for hmm scoring")
    if not os.path.exists(args.DMS_folder + os.sep + DMS_file_name):
        print(f"Warning: {DMS_file_name} not found in {args.DMS_folder}. Skipping.")
    variant_df = pd.read_csv(args.DMS_folder + os.sep + DMS_file_name)
    with open(fa_path + os.sep + basename + ".fa", "w+") as f:
        f.write(f">WT\n{target_seq}\n")
        for j, row in variant_df.iterrows():
            f.write(f">Mutant {j}\n{row[mutated_sequence_column]}\n")

    print("Scoring variants with HMM forward-backward algorithm") 
    hmm_file = hmm_path + os.sep + basename + ".hmm"
    fa_file = fa_path + os.sep + basename + ".fa"
    output_file = raw_score_path + os.sep + basename + ".csv"
    os.system(f"{args.hmmer_path}" + os.sep + "src" + os.sep + f"generic_fwdback_example {hmm_file} {fa_file} > {output_file}")

    # Postprocessing scores to have label columns, wt ratio scores 
    print("Postprocessing scores to match output format")
    if not os.path.exists(os.path.join(args.DMS_folder,DMS_file_name)):
        print(f"Warning: {DMS_file_name} not found in {args.DMS_folder}. Skipping.") 
    df = pd.read_csv(os.path.join(raw_score_path, basename + ".csv"))
    # removing extra whitespace from seq_name column
    df["seq_name"] = df["seq_name"].apply(lambda x: x.replace(" ",""))
    wt_logprob = df[df["seq_name"] == "WT"]["logprob"].values[0]
    df["wt_ratio"] = df["logprob"].astype(float) - float(wt_logprob)
    # Occasionaly there are characters in sequences outside the alphabet of the hmm model, result in logprobs of -inf for both sequences and nan model scores. 
    # We set those NaNs equal to 0 here, assuming no difference between the mutants as it's outside the hmm alphabet. 
    df["wt_ratio"] = df["wt_ratio"].fillna(0)
    # Dropping WT score here to match other output formats 
    df = df[df["seq_name"] != "WT"]
    df = df.rename(columns={"seq":"mutant"})
    variant_df = pd.read_csv(os.path.join(args.DMS_folder, DMS_file_name))
    df = df.merge(variant_df[["mutant","DMS_score"]], on="mutant", how="left")
    df.to_csv(os.path.join(args.output_scores_folder,DMS_id + ".csv"), index=False)