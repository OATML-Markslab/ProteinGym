import pandas as pd 
import os 
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
    Script to generate the bash script to score substitutions with GEMME
    """
    parser = argparse.ArgumentParser(description='GEMME scoring part 1')
    #We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
    #Fields to be passed manually if reference file is not used
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
    parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')
    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')

    # GEMME parameters     
    parser.add_argument("--temp_folder", default="./gemme_tmp", type=str, help="Path to temporary folder to store intermediate files")
    parser.add_argument("--GEMME_path", default="/n/groups/marks/software/GEMME/GEMME", type=str, help="Path to GEMME installation")
    parser.add_argument("--JET_path", default="/n/groups/marks/software/JET2/JET2", type=str, help="Path to JET2 installation")
    parser.add_argument("--nseqs", type=int, default=20000)

    # output bash script to run GEMME
    parser.add_argument("--output_script", default=None, type=str, help="Output bash script to run GEMME")
    args = parser.parse_args()

    if not os.path.isdir(args.temp_folder):
        os.mkdir(args.temp_folder)

    if args.DMS_reference_file_path:
        mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[args.DMS_index]
        print("Generate GEMME run script for DMS: "+str(DMS_id))
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        MSA_data_file = args.MSA_folder + os.sep + mapping_protein_seq_DMS["MSA_filename"][args.DMS_index] if args.MSA_folder is not None else None
        MSA_start = mapping_protein_seq_DMS["MSA_start"][args.DMS_index]
        MSA_end = mapping_protein_seq_DMS["MSA_end"][args.DMS_index]
    else:
        target_seq=args.target_seq
        DMS_file_name=args.DMS_file_name
        DMS_id = DMS_file_name.split(".")[0]
        MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
        MSA_start = args.MSA_start
        MSA_end = args.MSA_end
    # This is necessary because GEMME splits filenames and things on periods and underscores and dashes and things, so we need to remove those to avoid 
    # any filepath/name issues 
    condensed_DMS_id = DMS_id.replace("_","").replace(".","")
    if not os.path.isdir(args.temp_folder + os.sep + condensed_DMS_id):
        os.mkdir(args.temp_folder + os.sep + condensed_DMS_id)
    full_temp_folder = args.temp_folder + os.sep + condensed_DMS_id
    # get mutant files in right format for GEMME 
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name)
    # mutant_temp_file = DMS_id.replace(".","-").replace("_","@") + "_mutants.txt"
    mutant_temp_file = condensed_DMS_id + "_mutants.txt"
    if "mutant" not in DMS_data.columns:
        raise ValueError("DMS data file must contain a column named 'mutant' with the mutant sequences to score")
    env = os.environ.copy()
    env["GEMME_PATH"] = args.GEMME_path
    env["JET_PATH"] = args.JET_path 
    with open(full_temp_folder + os.sep + mutant_temp_file, "w+") as mutant_file:
        for mutant in DMS_data["mutant"]:
            offset_mutant = set_mutant_offset(mutant, MSA_start)
            mutant_csv = offset_mutant.replace(":", ",")
            mutant_file.write(mutant_csv + "\n")

    # Converting alignment to uppercase if necessary, only uppercases lines that are not headers
    MSA_upper_file = condensed_DMS_id + "_MSA_upper.txt"
    if MSA_data_file is not None:
        with open(MSA_data_file, "r") as f:
            lines = f.readlines()
        with open(full_temp_folder + os.sep + MSA_upper_file, "w") as f:
            for i,line in enumerate(lines):
                if line[0] == ">":
                    # if i == 0:
                        # f.write(">" + condensed_DMS_id + "\n")
                    # else:
                    if not "/" in line:
                        newline = line.split("\t")[0].replace("_","").replace(".","").rstrip() + "/" + str(MSA_start) + "-" + str(MSA_end) + "\n"
                        f.write(newline)
                    else:
                        f.write(line.replace("_","").replace(".",""))
                else:
                    f.write(line.upper())
    else:
        raise ValueError("MSA data file must be provided to run GEMME")

    # output bash script to run GEMME in a container 
    command = "cd {}\npython2.7 {}/gemme.py {} -r input -f {} -m {} -N {}\n".format(full_temp_folder, args.GEMME_path, MSA_upper_file, MSA_upper_file, mutant_temp_file, args.nseqs)
    # print(command)
    with open(args.output_script, 'w') as f_out:
        f_out.write(command)
    
    # proc_obj = subprocess.run(command, shell=True, env=env, cwd=full_temp_folder, check=True)
