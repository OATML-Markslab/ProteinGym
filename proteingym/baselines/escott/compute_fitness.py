"""
Run ESCOTT on selected DMS index in ProteinGym and save fitness scores.

Note that ESCOTT has the following dependencies that have to be installed:
    - JET2 (muscle, java, and optionally psiblast, clustalw)
    - R
    - DSSP
Due to the difficulty in ESCOTT installation, we recommend using the Docker image
provided by the authors. The current version can be dowloaded with:
docker pull tekpinar/prescott-docker:v1.6.0.
This script assumes that the Docker image is running and accessible, and
folders containing inputs and output files are mounted to the Docker container.
This is done automatically when using the scoring_ESCOTT_substitutions.sh script in
scripts/scoring_DMS_zero_shot folder.
ESCOTT documentation with installation instructions and examples can be found at:
http://gitlab.lcqb.upmc.fr/tekpinar/PRESCOTT

For the correct ESCOTT execution, the first sequence in MSA file has to be the
query sequence, and MSA and PDB file must span the exact same region of the protein.
In this script, default input files are manipulated to match these requirements.
"""

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pdb_utils


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
aa2idx = {aa: i for i, aa in enumerate(AA_VOCAB)}


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--DMS_reference_file_path",
        default=None,
        type=str,
        help="Path to reference file with list of DMS to score",
    )
    parser.add_argument(
        "--DMS_index", default=0, type=int, help="Index of DMS assay in reference file"
    )
    parser.add_argument(
        "--DMS_data_folder", type=str, help="Path to folder that contains all DMS assay datasets"
    )
    parser.add_argument(
        "--output_scores_folder",
        default="./",
        type=str,
        help="Name of folder to write model scores to",
    )
    parser.add_argument("--MSA_folder", default=".", type=str, help="Path to MSA file")
    parser.add_argument("--structure_data_folder", default=".", type=str, help="Path to PDB file")
    parser.add_argument(
        "--temp_folder",
        default="./escott_tmp",
        type=str,
        help="Path to temporary folder to store intermediate files",
    )
    parser.add_argument("--nseqs", type=int, default=40000)
    args = parser.parse_args()
    return args


def parse_alignment(ali_file):
    """
    Parse alignment file and return dictionary of sequences.
    Headers are modified to avoid errors in ESCOTT.
    """
    with open(ali_file, "r") as f:
        lines = f.readlines()
    seqs = {}
    for line in lines:
        if line[0] == ">":
            seq_id = line[1:].strip().replace("_", "").replace(".", "")
            seqs[seq_id] = ""
        else:
            seqs[seq_id] += line.strip().upper().replace(".", "-")
    return seqs


def extract_scores(predictions, mutants, offset):
    """Extract scores for a given assay from the full mutational landscape."""
    scores = []
    for mut in mutants:
        score = 0
        for m in mut.split(":"):  # multiple mutations case
            pos, mut_aa = int(m[1:-1]) - offset, m[-1]
            score += predictions[pos, aa2idx[mut_aa]]
        scores.append(score)
    return scores


def main(args):

    if not os.path.isdir(args.temp_folder):
        os.mkdir(args.temp_folder)

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    print("Computing scores for DMS: " + str(DMS_id))
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][
        mapping_protein_seq_DMS["DMS_id"] == DMS_id
    ].values[0]
    MSA_data_file = os.path.join(
        args.MSA_folder, mapping_protein_seq_DMS["MSA_filename"][args.DMS_index]
    )
    MSA_start = mapping_protein_seq_DMS["MSA_start"][args.DMS_index]
    MSA_end = mapping_protein_seq_DMS["MSA_end"][args.DMS_index]

    DMS_data = pd.read_csv(os.path.join(args.DMS_data_folder, DMS_file_name))
    if "mutant" in DMS_data.columns:
        mutants = DMS_data["mutant"].tolist()
    else:
        raise ValueError(
            "DMS data file must contain a column named 'mutant' with the mutant sequences to score"
        )

    # Create subfolder for single assays in temporary folder, removing some annoying characters
    # Use absolute path to avoid issues with subprocess call
    assay_folder = os.path.abspath(
        os.path.join(args.temp_folder, DMS_id.replace("_", "").replace(".", ""))
    )
    if not os.path.isdir(assay_folder):
        os.mkdir(assay_folder)

    # since ESCOTT also uses pdb files, and they have to match MSA files, we iterate over
    # pdb chunks for a single assay, cutting the MSA file accordingly
    # Only after processing each chunk we extract scores from the output file for
    # the corresponding mutants
    pdb_filenames = (
        mapping_protein_seq_DMS["pdb_file"][mapping_protein_seq_DMS["DMS_id"] == DMS_id]
        .values[0]
        .split("|")
    )  # if sequence is large (eg., BRCA2_HUMAN) the structure is split in several chunks
    pdb_ranges = (
        mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"] == DMS_id]
        .values[0]
        .split("|")
    )
    all_scores = []

    try:
        for pdb_index, pdb_filename in enumerate(pdb_filenames):
            pdb_file = os.path.join(args.structure_data_folder, pdb_filename)
            pdb_start, pdb_end = [int(x) for x in pdb_ranges[pdb_index].split("-")]

            # check if pdb range is contained in MSA range, otherwise manipulate pdb file
            if not (pdb_start >= MSA_start and pdb_end <= MSA_end):
                print("PDB range not contained in MSA range, PDB file manipulation needed")
                pdb_start_range = max(pdb_start, MSA_start) - pdb_start + 1  # 1-based index
                pdb_end_range = min(pdb_end, MSA_end) - pdb_start + 2
                residue_numbers = list(range(pdb_start_range, pdb_end_range))
                new_pdb_file = os.path.join(assay_folder, "cut_" + pdb_filename)
                pdb_utils.filter_residues(pdb_file, new_pdb_file, residue_numbers)

                pdb_file = new_pdb_file
                pdb_start = max(pdb_start, MSA_start)
                pdb_end = min(pdb_end, MSA_end)

            # Both pdb index and msa index in reference file are 1-based
            starting_msa_idx = pdb_start - MSA_start
            endings_msa_idx = pdb_end - MSA_start + 1

            # set offset for scores extraction at the end
            if pdb_index == 0:
                offset = pdb_start

            # Create temporary MSA file for this pdb chunk, with correct format for ESCOTT
            MSA_tmp_file = os.path.join(assay_folder, "MSA_" + str(pdb_index) + ".fasta")
            if MSA_data_file is not None:
                sequence_dict = parse_alignment(MSA_data_file)
                with open(MSA_tmp_file, "w") as f:
                    for id, seq in sequence_dict.items():
                        f.write(">" + id + "\n")
                        f.write(seq[starting_msa_idx:endings_msa_idx] + "\n")
            else:
                raise ValueError("MSA data file must be provided to run ESCOTT")

            # run ESCOTT on this pdb chunk
            command = f"escott {MSA_tmp_file} -p {pdb_file} -N {args.nseqs}"
            output = subprocess.run(command, shell=True, cwd=assay_folder, capture_output=True)
            if output.returncode != 0:
                raise ValueError(f"Error occurred while running ESCOTT: {output.stderr.decode()}")

            # parse output files and find the file with suffix _evolCombi.txt
            for file in os.listdir(assay_folder):
                if file.endswith("_evolCombi.txt"):
                    evol_combi_file = file
                    break
            else:
                raise ValueError("ESCOTT output file not found, an error occurred")

            scores = pd.read_csv(
                os.path.join(assay_folder, evol_combi_file), sep="\s+", index_col=0
            ).values.transpose()
            all_scores.append(scores)

        # concatenate scores from all pdb chunks
        all_scores = np.concatenate(all_scores, axis=0)

        # extract scores for selected mutants
        mutant_scores = extract_scores(all_scores, mutants, offset)
        DMS_data["ESCOTT_score"] = mutant_scores
        DMS_data.to_csv(os.path.join(args.output_scores_folder, f"{DMS_id}.csv"), index=False)
        print(f"Scores for DMS {DMS_id} computed successfully")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        os.system(f"rm -rf {assay_folder}")


if __name__ == "__main__":
    args = get_args()
    main(args)
