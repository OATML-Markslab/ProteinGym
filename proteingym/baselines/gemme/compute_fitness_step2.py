import os
import pandas as pd
import argparse
import subprocess
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scoring_utils import set_mutant_offset, undo_mutant_offset

"""
Run GEMME on selected DMS and saves fitness scores
Note that GEMME has JET2 as a dependency (and psiblast if used for sequence search)
and requires installations of java, python2, and R
GEMME also assumes that the first sequence of the alignment is the query sequence 
"""
if __name__ == "__main__":
    """
    parse the GEMME output
    """
    parser = argparse.ArgumentParser(description='GEMME scoring part 2')
    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')

    parser.add_argument("--temp_folder", default="./gemme_tmp", type=str, help="Path to temporary folder to store intermediate files")
    
    args = parser.parse_args()

    if args.DMS_reference_file_path:
        mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[args.DMS_index]
        print("Compiling scores for DMS: "+str(DMS_id))
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        MSA_start = mapping_protein_seq_DMS["MSA_start"][args.DMS_index]
    else:
        DMS_file_name=args.DMS_file_name
        DMS_id = DMS_file_name.split(".")[0]
        MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
        MSA_start = args.MSA_start

    # get mutant files in right format for GEMME 
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name)
    condensed_DMS_id = DMS_id.replace("_","").replace(".","")
    full_temp_folder = args.temp_folder + os.sep + condensed_DMS_id
    
    # find file with suffix _evolCombi.txt
    for file in os.listdir(full_temp_folder):
        if file.endswith("_evolCombi.txt"):
            evol_combi_file = file
            break
    else:
        raise ValueError("GEMME output file not found")
    score_df = pd.read_csv(full_temp_folder + os.sep + evol_combi_file, sep=" ")
    score_df = score_df.reset_index().rename(columns={"index":"mutant","x":"GEMME_score"})
    if not type(score_df["mutant"][0]) == str:
        print("Weird R dataframe conversion error inside GEMME, remapping mutants to dataframe")
        score_df["mutant"] = DMS_data["mutant"]
    else:
        score_df["mutant"] = score_df["mutant"].apply(undo_mutant_offset, MSA_start=MSA_start)
        score_df["mutant"] = score_df["mutant"].apply(lambda x: x.replace(",",":"))
    DMS_data_merged = pd.merge(DMS_data, score_df, on="mutant", how="left")
    DMS_data_merged.to_csv(args.output_scores_folder + os.sep + DMS_id + ".csv", index=False)
    os.system("rm -rf {}".format(full_temp_folder))
