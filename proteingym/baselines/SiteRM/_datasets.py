"""
This module creates the training MSAs for CherryML based on the raw data.

This involves carefully extracting the region of interest from the MSA,
and formatting the MSAs in the format expected by CherryML.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import random

from cherryml import io as cherryml_io
from cherryml import caching as cherryml_caching


GAP_CHARACTER = "-"


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_transitions_dir"],
)
def create_transitions_from_pairs_and_time(
    families: List[str],
    transition_pairs_dir: str,
    t: float,
    output_transitions_dir: Optional[str] = None,
    _version: str = "2024_05_01_v1",
):
    for family in families:
        transitions = []
        lines = open(
            os.path.join(
                transition_pairs_dir,
                family + ".txt"
            ), "r"
        )
        for line in lines:
            x, y = line.split(" ")
            y = y.replace("\n", "")  # Remove endline character
            # print(f"len(x) = {len(x)}")
            # print(f"len(y) = {len(y)}")
            assert(len(x) == len(y))
            transitions.append((x, y, t))
        cherryml_io.write_transitions(
            transitions,
            os.path.join(
                output_transitions_dir,
                family + ".txt"
            )
        )
        cherryml_caching.secure_parallel_output(
            output_transitions_dir,
            family
        )


def translate_msa(
    a2m_path: str,
    subsample_n_sequences: int,
    random_seed: int = 42,
) -> Tuple[Dict[str, str], int, int]:
    """
    Created the training MSA from the a2m ProteinGym file.

    It also tries to find the MSA start and end indices.
    If not found, returns -1 for both.
    """
    # Need to manually read the MSA
    protein_names_and_sequences = []  # Type: List[Tuple[str, str]]
    lengths = []
    start_idx_0_based = None
    end_idx_0_based = None
    seen_protein_names = set()
    start_idx_0_based, end_idx_0_based = -1, -1
    with open(a2m_path, "r") as a2m_file:
        curr_protein = ""
        protein_name = None
        for line_idx, line in enumerate(a2m_file.read().split("\n")):
            if line_idx == 0:
                # This is the reference protein -- we need to extract the indices.
                try:
                    start_idx_0_based, end_idx_0_based = line.split('/')[-1].split('-')
                    start_idx_0_based = int(start_idx_0_based) - 1
                    end_idx_0_based = int(end_idx_0_based) - 1
                    assert(start_idx_0_based >= 0)
                    assert(end_idx_0_based >= 0)
                except:
                    print(f"WARNING: MSA file {a2m_path} does not specify the start and end indices.")
            if line.startswith(">"):
                # Push the previous protein, which we have already built
                if curr_protein != "":
                    assert protein_name is not None
                    protein_name = f"protein-{len(lengths)}"  # Override because their MSAs have duplicate sequence names!!!
                    protein_names_and_sequences.append(
                        (
                            protein_name,
                            curr_protein
                        )
                    )
                    if protein_name in seen_protein_names:
                        raise ValueError(
                            f"Issue parsing protein names in file {a2m_path} . "
                            f"Found duplicate name: {protein_name} . Might be "
                            f"because I am parsing the MSA file incorrectly."
                        )
                    seen_protein_names.add(protein_name)
                    lengths.append(len(curr_protein))
                protein_name = line.split("\t")[0][1:]
                curr_protein = ""
            else:
                curr_protein += line.upper().replace(".", "-")
    assert(len(set(lengths)) == 1)

    # Remove duplicates from protein_names_and_sequences
    def make_unique(
        protein_names_and_sequences: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        protein_names_and_sequences_unique = []
        seen_proteins = set()
        for protein_name, protein_sequence in protein_names_and_sequences:
            if protein_sequence not in seen_proteins:
                protein_names_and_sequences_unique.append(
                    (
                        protein_name,
                        protein_sequence
                    )
                )
                seen_proteins.add(protein_sequence)
        return protein_names_and_sequences_unique
    # print(f"Size of MSA before removing duplicates: {len(protein_names_and_sequences)}")
    protein_names_and_sequences = make_unique(protein_names_and_sequences)
    # print(f"Size of MSA after removing duplicates: {len(protein_names_and_sequences)}")
    # Check uniqueness
    assert(
        len(
            set(
                [x[1] for x in protein_names_and_sequences]
            )
        ) == len(protein_names_and_sequences)
    )

    # Create a random number generator object
    rng = random.Random(random_seed)
    sampled_indices = [0] + rng.sample(
        range(1, len(protein_names_and_sequences), 1),
        min(
            subsample_n_sequences,
            len(protein_names_and_sequences)
        ) - 1
    )
    assert(len(set(sampled_indices)) == min(subsample_n_sequences, len(protein_names_and_sequences)))
    sampled_protein_names_and_sequences = [
        protein_names_and_sequences[i]
        for i in sampled_indices
    ]
    assert(len(set(sampled_protein_names_and_sequences)) == min(subsample_n_sequences, len(protein_names_and_sequences)))
    subsampled_msa = dict(sampled_protein_names_and_sequences)
    # print(f"subsampled_msa = {subsampled_msa}")
    # print(f"subsampled_msa length = {len(subsampled_msa)}")
    return subsampled_msa, start_idx_0_based, end_idx_0_based


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_msa_dir",
        "output_aligned_region_0_based_dir",
    ],
)
def DMS_substitutions_dataset_create_training_msas(
    DMS_reference_file_path: str,
    DMS_data_folder: str,
    DMS_MSA_data_folder: str,
    families: List[str],  # With the list of DMS_id
    subsample_n_sequences: int = 1000,
    output_msa_dir: Optional[str] = None,
    output_aligned_region_0_based_dir: Optional[str] = None,
    _version: str = "2024_05_05__10_29_am",
):
    """
    Create training MSAs for DMS substitutions dataset.
    """
    # We need to convert the training alignment into standard format, etc.
    for family_idx, family in enumerate(families):
        DMS_id = family
        mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        
        DMS_data_path = DMS_data_folder + os.sep + DMS_file_name
        DMS_data = pd.read_csv(DMS_data_path, low_memory=False)

        DMS_MSA_file_name = mapping_protein_seq_DMS["MSA_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        DMS_MSA_file_path = DMS_MSA_data_folder + os.sep + DMS_MSA_file_name
        print(f"Creating standardized data for: {DMS_id} ({family_idx + 1} / {len(families)}). DMS file path:\n{DMS_MSA_file_path}")
        if not os.path.exists(DMS_MSA_file_path):
            raise ValueError(
                f"DMS_MSA_file_path {DMS_MSA_file_path} does not exist!"
            )

        start_idx_0_based, end_idx_0_based = (
            int(mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]["MSA_start"]) - 1,
            int(mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]["MSA_end"]) - 1
        )
        print(f"start_idx_0_based = {start_idx_0_based}")
        print(f"end_idx_0_based = {end_idx_0_based}")

        subsetted_msa, start_idx_0_based_msa, end_idx_0_based_msa = translate_msa(
            a2m_path=DMS_MSA_file_path,
            subsample_n_sequences=subsample_n_sequences,
        )
        # print(f"start_idx_0_based_msa = {start_idx_0_based_msa}")
        # print(f"end_idx_0_based_msa = {end_idx_0_based_msa}")
        # assert(False)
        if start_idx_0_based_msa != -1 and end_idx_0_based_msa != -1:
            if start_idx_0_based != start_idx_0_based_msa or end_idx_0_based != end_idx_0_based_msa:
                raise ValueError(
                    f"start_idx_0_based = {start_idx_0_based}\n"
                    f"start_idx_0_based_msa = {start_idx_0_based_msa}\n"
                    f"end_idx_0_based = {end_idx_0_based}\n"
                    f"end_idx_0_based_msa = {end_idx_0_based_msa}\n"
                )

        cherryml_io.write_msa(
            subsetted_msa,
            os.path.join(
                output_msa_dir,
                family + ".txt",
            ),
        )
        cherryml_caching.secure_parallel_output(output_msa_dir, family)

        with open(
            os.path.join(
                output_aligned_region_0_based_dir,
                family + ".txt",
            ),
            "w"
        ) as f:
            f.write(f"{start_idx_0_based} {end_idx_0_based}")
        cherryml_caching.secure_parallel_output(output_aligned_region_0_based_dir, family)


def get_dms_substitutions_families(
    DMS_reference_file_path: str,
) -> List[str]:
    mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
    res = sorted(list(set(mapping_protein_seq_DMS["DMS_id"])))
    if len(res) not in [217, 2525]:
        raise ValueError(
            f"Expected 217 DMS assays or 2525 clinical assays. "
            f"Found: {len(res)} instead. Assays: {res}."
        )
    return res


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_transition_pairs_dir"],
)
def create_test_transition_pairs(
    families: List[str],
    DMS_reference_file_path: str,
    DMS_data_folder: str,
    output_transition_pairs_dir: Optional[str] = None,
    _version: str = "2024_04_30_v1",
):
    for family in families:
        DMS_id = family
        mapping_protein_seq_DMS = pd.read_csv(DMS_reference_file_path)
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        
        DMS_data_path = DMS_data_folder + os.sep + DMS_file_name
        DMS_data = pd.read_csv(DMS_data_path, low_memory=False)
        ys = np.array(DMS_data['mutated_sequence'])
        xs = []
        mutated_xs = []
        mutated_ys = []
        for protein, mutantions_concatenated in zip(DMS_data['mutated_sequence'], DMS_data['mutant']):
            mutated_x = ""
            mutated_y = ""
            source_protein_chars = [aa for aa in protein]

            mutations = mutantions_concatenated.split(':')
            for mutation in mutations:
                original_aa, zero_based_idx, new_aa = mutation[0], int(mutation[1:-1]) - 1, mutation[-1]
                mutated_x += original_aa
                mutated_y += new_aa
                if protein[zero_based_idx] != new_aa:
                    raise ValueError(f"Protein {protein} should be mutant, i.e. {original_aa} mutating at zero-based position {zero_based_idx} to {new_aa}, but there is not {new_aa} at index {zero_based_idx}.")
                source_protein_chars[zero_based_idx] = original_aa

            xs.append(''.join(source_protein_chars))
            assert(len(mutated_x) == len(mutated_y))
            assert(len(mutated_x) > 0)
            assert(len(mutated_x) == 1 + mutantions_concatenated.count(':'))
            mutated_xs.append(mutated_x)
            mutated_ys.append(mutated_y)
        # Confirm that all the xs are the same (it's a DMS after all)
        if len(set(xs)) != 1:
            raise ValueError(
                f"There should be only one source protein in a DMS assay, "
                f"but found more than one! This assumption is used to speed "
                f"up likelihood computations under the LG model."
            )
        pairs = list(zip(xs, ys))
        pairs_str = "\n".join(f"{x} {y}" for (x, y) in pairs)
        with open(
            os.path.join(
                output_transition_pairs_dir,
                family + ".txt"
            ),
            "w"
        ) as test_transition_pairs_file:
            test_transition_pairs_file.write(pairs_str)
        cherryml_caching.secure_parallel_output(output_transition_pairs_dir, family)


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_transition_pairs_dir"]
)
def substring_the_aligned_region(
    families: List[str],
    transition_pairs_dir: str,
    aligned_region_0_based_dir: str,
    output_transition_pairs_dir: Optional[str] = None,
    _version: str = "2024_04_30_v1",
):
    for family in families:
        transition_pairs_strs = open(
            os.path.join(
                transition_pairs_dir,
                family + ".txt"
            )
        ).read().split('\n')
        aligned_region_0_based_str = open(
            os.path.join(
                aligned_region_0_based_dir,
                family + ".txt"
            )
        ).read()
        start_idx_0_based = int(aligned_region_0_based_str.split(" ")[0])
        end_idx_0_based = int(aligned_region_0_based_str.split(" ")[1])
        transition_pairs_aligned_region_strs = []
        for line in transition_pairs_strs:
            x, y = line.split(" ")
            assert(len(x) == len(y))
            x_aligned_region = x[start_idx_0_based:(end_idx_0_based + 1)]
            y_aligned_region = y[start_idx_0_based:(end_idx_0_based + 1)]
            transition_pairs_aligned_region_strs.append(f"{x_aligned_region} {y_aligned_region}")
        transition_pairs_aligned_region_str = "\n".join(transition_pairs_aligned_region_strs)
        open(
            os.path.join(
                output_transition_pairs_dir,
                family + ".txt"
            ), "w"
        ).write(transition_pairs_aligned_region_str)
        cherryml_caching.secure_parallel_output(output_transition_pairs_dir, family)


def DMS_substitutions_dataset(
    DMS_reference_file_path: str,
    DMS_data_folder: str,
    DMS_MSA_data_folder: str,
    subsample_n_sequences: int = 100,
    max_families: int = None,  # For testing
):
    """
    Create training MSAs for DMS substitutions benchmark.
    """

    families = get_dms_substitutions_families(
        DMS_reference_file_path=DMS_reference_file_path,
    )
    if max_families is not None:
        families = families[:max_families]

    msa_dirs = DMS_substitutions_dataset_create_training_msas(
        DMS_reference_file_path=DMS_reference_file_path,
        DMS_data_folder=DMS_data_folder,
        DMS_MSA_data_folder=DMS_MSA_data_folder,
        families=families,
        subsample_n_sequences=subsample_n_sequences,
    )
    train_msa_dir = msa_dirs["output_msa_dir"]
    aligned_region_0_based_dir = msa_dirs["output_aligned_region_0_based_dir"]
    # print(f"train_msa_dir = {train_msa_dir}")
    # print(f"aligned_region_0_based_dir = {aligned_region_0_based_dir}")

    test_transition_pairs_dir = create_test_transition_pairs(
        families=families,
        DMS_reference_file_path=DMS_reference_file_path,
        DMS_data_folder=DMS_data_folder,
    )["output_transition_pairs_dir"]
    # print(f"test_transition_pairs_dir = {test_transition_pairs_dir}")

    test_transition_pairs_aligned_region_dir = substring_the_aligned_region(
        families=families,
        transition_pairs_dir=test_transition_pairs_dir,
        aligned_region_0_based_dir=aligned_region_0_based_dir,
    )["output_transition_pairs_dir"]
    # print(f"test_transition_pairs_aligned_region_dir = {test_transition_pairs_aligned_region_dir}")

    res_dict = {
        "families": families,
        "train_msa_dir": train_msa_dir,
        "train_aligned_region_0_based_dir": aligned_region_0_based_dir,
        "test_transition_pairs_dir": test_transition_pairs_dir,
        "test_transition_pairs_aligned_region_dir": test_transition_pairs_aligned_region_dir,
    }
    return res_dict


def clinical_substitutions_dataset(
    clinical_reference_file_path: str,
    clinical_data_folder: str,
    clinical_MSA_data_folder: str,
    subsample_n_sequences: int = 100,
    max_families: int = None,  # For testing
):
    return DMS_substitutions_dataset(
        DMS_reference_file_path=clinical_reference_file_path,
        DMS_data_folder=clinical_data_folder,
        DMS_MSA_data_folder=clinical_MSA_data_folder,
        subsample_n_sequences=subsample_n_sequences,
        max_families=max_families,
    )
    

if __name__ == '__main__':
    # This takes around 5 minutes to run. It will generate all the training MSAs for CherryML.
    cherryml_caching.set_cache_dir("_cache_cherryml")
    cherryml_caching.set_log_level(9)
    cherryml_caching.set_read_only(False)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "../../../input_data/ProteinGym")
    git_dir = os.path.join(dir_path, "../../..")
    DMS_reference_file_path = os.path.abspath(f"{git_dir}/reference_files/DMS_substitutions.csv")
    DMS_data_folder = os.path.abspath(f"{data_dir}/DMS_ProteinGym_substitutions/DMS_ProteinGym_substitutions")
    DMS_MSA_data_folder = os.path.abspath(f"{data_dir}/DMS_msa_files/DMS_msa_files")
    clinical_reference_file_path = os.path.abspath(f"{git_dir}/reference_files/clinical_substitutions.csv")
    clinical_data_folder = os.path.abspath(f"{data_dir}/clinical_ProteinGym_substitutions")
    clinical_MSA_data_folder = os.path.abspath(f"{data_dir}/clinical_msa_files/subs")

    ##### DMS substitutions
    res_dict = DMS_substitutions_dataset(
        DMS_reference_file_path=DMS_reference_file_path,
        DMS_data_folder=DMS_data_folder,
        DMS_MSA_data_folder=DMS_MSA_data_folder,
        subsample_n_sequences=100000000,
        max_families=217,
    )
    ##### clinical substitutions
    res_dict = clinical_substitutions_dataset(
        clinical_reference_file_path=clinical_reference_file_path,
        clinical_data_folder=clinical_data_folder,
        clinical_MSA_data_folder=clinical_MSA_data_folder,
        subsample_n_sequences=100000000,
        max_families=2525,
    )

    print("\n".join(f"{key}: {value}" for (key, value) in res_dict.items() if key != "families"))
