import pandas as pd 
import os 
import argparse
import subprocess 
import json 
import numpy as np 
import tqdm 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Tranception scoring')

    parser.add_argument("--cache_location", type=str, help="Location of T5 weight cache", default=None)    
    parser.add_argument("--skip_VESPA_computation",type=str,help="Skip running VESPA if it's been prerun and we just want to compute fitness.")
    parser.add_argument("--wt_fasta_file",type=str,default="./WT_sequences.fasta",help="Location of fasta file containing all wild type sequences to compute VESPA scores for.")
    parser.add_argument("--vespa_tmp_dir",type=str,default="./vespa_tmp",help="Location of temporary files for VESPA")
    parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument("--MSA_data_folder", type=str, help="Folder where MSAs are stored", default=None)
    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
    parser.add_argument("--DMS_index_range_start",type=int, help="Start of range of DMS assays", default=None)
    parser.add_argument("--DMS_index_range_end",type=int, help="Index of DMS assay in reference file", default=None)
    
    args = parser.parse_args()
    if not os.path.exists(args.vespa_tmp_dir):
        os.makedirs(args.vespa_tmp_dir)
    
    if args.DMS_reference_file_path is not None:
        if args.DMS_index_range_start is not None and args.DMS_index_range_end is not None:
            DMS_reference_file = pd.read_csv(args.DMS_reference_file_path)
            DMS_filenames = DMS_reference_file['DMS_filename'][args.DMS_index_range_start:args.DMS_index_range_end+1].tolist()
            MSA_filenames = DMS_reference_file['MSA_filename'][args.DMS_index_range_start:args.DMS_index_range_end+1].unique().tolist()
            target_seqs = DMS_reference_file['target_seq'][args.DMS_index_range_start:args.DMS_index_range_end+1].tolist()
        else:
            DMS_reference_file = pd.read_csv(args.DMS_reference_file_path)
            DMS_filenames = [DMS_reference_file['DMS_filename'][args.DMS_index]]
            MSA_filenames = [DMS_reference_file['MSA_filename'][args.DMS_index]]
            target_seqs = [DMS_reference_file['target_seq'][args.DMS_index]]
    else:
        DMS_filenames = [args.DMS_file_name]
        MSA_filenames = [args.MSA_file_name]
        target_seqs = [args.target_seq]
    if not args.skip_VESPA_computation:
        # Build fasta file of wild types sequences based on first sequence of each alignment in MSA_filenames
        all_WT_sequences_fasta = args.vespa_tmp_dir + os.sep + args.wt_fasta_file
        index_map = {"indices":[],"MSA_filename":[]}
        for i,filename in enumerate(MSA_filenames):
            index_map["indices"].append(i)
            index_map["MSA_filename"].append(filename)
            f = os.path.join(args.MSA_data_folder, filename)
            target_seq=""
            with open(f, 'r') as msa_data:
                for i, line in enumerate(msa_data):
                    line = line.rstrip()
                    if line.startswith(">") and i==0:
                        with open(all_WT_sequences_fasta,'a+') as seq_wt_file:
                            seq_wt_file.write(line+"\n")
                    elif line.startswith(">"):
                        break
                    else:
                        with open(all_WT_sequences_fasta,'a+') as seq_wt_file:
                            seq_wt_file.write(line+"\n")
        subprocess.run(["vespa",args.wt_fasta_file,"--prott5_weights_cache", args.cache_location, "--vespa"], cwd=args.vespa_tmp_dir, check=True)
    else:
        index_map = {"indices":[],"MSA_filename":[]}
        for i,filename in enumerate(MSA_filenames):
            index_map["indices"].append(i)
            index_map["MSA_filename"].append(filename)
        print(f"Skipping VESPA computation, scoring DMS assays {DMS_filenames} with precomputed VESPA scores. Note that this assumes that the WT sequence file contains the wild type sequences in the same order as the DMS reference file.")
    # Score each DMS assay
    assert os.path.exists(f"{args.vespa_tmp_dir}/vespa_run_directory/output/map.json")
    map_dict = json.load(open(f"{args.vespa_tmp_dir}/vespa_run_directory/output/map.json","r"))
    index_df = pd.DataFrame(index_map)
    index_df["VESPA_scoring_file_name"] = index_df["indices"].apply(lambda x: map_dict[str(x)])
    index_df = index_df.rename(columns={"indices":"VESPA_scoring_file_index"})
    DMS_reference_file_subset = DMS_reference_file.iloc[args.DMS_index_range_start:args.DMS_index_range_end+1]
    DMS_reference_file_subset = DMS_reference_file_subset.merge(index_df, how="left",on="MSA_filename")
    list_DMS = DMS_reference_file_subset["DMS_id"]
    VESPA_scores_folder = f"{args.vespa_tmp_dir}/vespa_run_directory/output"
    for DMS_id in list_DMS:
        print(DMS_id)
        DMS_filename = DMS_reference_file_subset["DMS_filename"][DMS_reference_file_subset["DMS_id"]==DMS_id].values[0]
        VESPA_scores_filename = int(DMS_reference_file_subset["VESPA_scoring_file_index"][DMS_reference_file_subset["DMS_id"]==DMS_id].values[0])
        print(VESPA_scores_filename)
        VESPA_scores_filename = str(VESPA_scores_filename) + '.csv'
        DMS_file = pd.read_csv(args.DMS_data_folder+os.sep+DMS_filename) #mutant,mutated_sequence,DMS_score,DMS_score_bin
        VESPA_scores = pd.read_csv(VESPA_scores_folder+os.sep+VESPA_scores_filename, sep=";")
        MSA_start = DMS_reference_file_subset["MSA_start"][DMS_reference_file_subset["DMS_id"]==DMS_id].values[0]
        VESPA_scores['mutant'] = VESPA_scores['Mutant'].apply(lambda x: x[0] + str(int(int(x[1:-1])+MSA_start)) + x[-1])
        VESPA_scores['VESPA'] = np.log(1 - VESPA_scores['VESPA']) #log proba of being functional (raw score is proba that mutation has an "effect")
        VESPA_scores['VESPAl'] = np.log(1 - VESPA_scores['VESPAl'])
        mapping_mutant_VESPA={}
        mapping_mutant_VESPAl={}
        for mutant in tqdm.tqdm(DMS_file['mutant']):
            VESPA_score_singles_sum = 0
            VESPAl_score_singles_sum = 0
            #Proba of multiple mutants to be benign is the product that each mutant is benign
            num_synonomous = 0
            for single in mutant.split(":"):
                # skipping synomymous mutations
                if single[0] == single[-1]:
                    num_synonomous += 1
                    continue
                VESPA_score_singles_sum += VESPA_scores['VESPA'][VESPA_scores['mutant']==single].values[0]
                VESPAl_score_singles_sum += VESPA_scores['VESPAl'][VESPA_scores['mutant']==single].values[0]
            assert VESPA_score_singles_sum!=0 or num_synonomous == len(mutant.split(":")), "Missing VESPA scores"
            mapping_mutant_VESPA[mutant] = VESPA_score_singles_sum
            mapping_mutant_VESPAl[mutant] = VESPAl_score_singles_sum
        mapping_mutant_VESPA = pd.DataFrame.from_dict(mapping_mutant_VESPA, orient='index').reset_index()
        mapping_mutant_VESPA.columns = ['mutant','VESPA']
        mapping_mutant_VESPAl = pd.DataFrame.from_dict(mapping_mutant_VESPAl, orient='index').reset_index()
        mapping_mutant_VESPAl.columns = ['mutant','VESPAl']
        final_scores_VESPA = pd.merge(DMS_file,mapping_mutant_VESPA, how='left',on='mutant')
        final_scores_VESPA = pd.merge(final_scores_VESPA,mapping_mutant_VESPAl, how='left',on='mutant')
        print(final_scores_VESPA)