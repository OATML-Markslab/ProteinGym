
"""
Run RSALOR mutational landscape predictions on DMS datasets.
"""

# Imports ----------------------------------------------------------------------
import os
import argparse
try:
    from rsalor import MSA
    from rsalor.structure import Structure
    from rsalor.utils import CSV
except:
    raise ImportError("Import of pip package 'rsalor' failed. Please install the package with 'pip install rsalor'.")


# Main -------------------------------------------------------------------------
def main():

    # Constants
    MUTATION_PROPERTY = "mutant"
    PREDICTION_PROPERTY = "RSALOR"
    REMOVE_PROPERTIES_IN_OUTPUT = ["DMS_score", "DMS_score_bin"]
    SEP = ","
    CHAIN = "A"
    MAX_CPUS = 16
    
    # Set number of used CPUs
    n_cpu_total = os.cpu_count()
    n_cpu_used = max(1, min(n_cpu_total // 2, MAX_CPUS))

    # Parse arguments
    parser = argparse.ArgumentParser(description='RSALOR arguments')
    parser.add_argument("--DMS_reference_file_path", type=str, help="Datasets reference '.csv' file.")
    parser.add_argument("--DMS_data_folder", type=str, help="Initial DMS datasets folder")
    parser.add_argument("--MSA_folder", type=str, help="MSA ('.a2m' or '.fasta') folder")
    parser.add_argument("--DMS_structure_folder", type=str, help="PDB ('.pdb') folder")
    parser.add_argument("--output_scores_folder", type=str, help="Output folder")
    args = parser.parse_args()

    # Read reference file
    dataset_reference_path = args.DMS_reference_file_path
    dataset_reference = CSV(sep=SEP, name="DMS reference").read(dataset_reference_path)
    print(f"\nEvaluate RSALOR on {len(dataset_reference)} DMS datasets")
    print(f" * dataset_reference_path: '{dataset_reference_path}'")
    dataset_reference.show()

    # Init output folder if required
    if not os.path.isdir(args.output_scores_folder):
        print(f"\nCreate new output directory '{args.output_scores_folder}'.")
        os.mkdir(args.output_scores_folder)

    # Loop on DMS datasets
    print(f"\nRun RSALOR on datasets from '{args.DMS_data_folder}' ...")
    for i, dataset_entry in enumerate(dataset_reference):

        # Init metadata
        dms_name = dataset_entry["DMS_id"]
        msa_name = dataset_entry["MSA_filename"]
        pdb_name = dataset_entry["pdb_file"]
        resid_shift = int(dataset_entry["MSA_start"]) - 1
        print(f"\n * Run RSALOR {i+1} / {len(dataset_reference)}: '{dms_name}'")

        # Set paths
        dataset_input_path = os.path.join(args.DMS_data_folder, f"{dms_name}.csv")
        dataset_output_path = os.path.join(args.output_scores_folder, f"{dms_name}.csv")
        msa_path = os.path.join(args.MSA_folder, f"{msa_name}")
        pdb_path = os.path.join(args.DMS_structure_folder, f"{pdb_name}")

        # Run RSALOR
        if "|" not in pdb_name: # normal case
            msa = MSA(
                msa_path, pdb_path, CHAIN,
                num_threads=n_cpu_used,
                verbose=False, disable_warnings=True,
            )
        else: # one single case when the PDB structure is splitted in 3 files ...
            # Oh God why you put 3 different '.pdb' files for one protein instead of merging them in one file :( ... 
            # (I am not supposed to do that with my code)

            # Set all PDB paths
            pdb_paths_list = [os.path.join(args.DMS_structure_folder, f"{pdb_name_i}") for pdb_name_i in pdb_name.split("|")]

            # Init MSA object with first structure
            msa = MSA(
                msa_path, pdb_paths_list[0], CHAIN,
                num_threads=n_cpu_used,
                verbose=False, disable_warnings=True,
            )

            # Manually merge all structures to one single structure
            structures = [Structure(pdb_path_i, CHAIN) for pdb_path_i in pdb_paths_list]
            main_structure = structures[0]
            for i in range(1, len(structures)):
                for residue in structures[i].residues:
                    main_structure.residues.append(residue)
            for resid, res in enumerate(main_structure.residues):
                res.position = str(resid+1) # re-index all residues
            main_structure.chain_residues = [res for res in main_structure.residues if res.chain == main_structure.chain]
            main_structure.residues_map = {res.resid: res for res in main_structure.residues}
            main_structure.sequence.sequence = "".join(res.amino_acid.one for res in main_structure.chain_residues)

            # Manually inject merged structure in the MSA object
            msa.structure = main_structure
            msa._align_structure_to_sequence()
            msa._init_weights()
            msa._init_counts()

        # Output RSALOR scores
        rsalor_single_scores = msa.get_scores()

        # Format scores (and shift mutation residue id reference because of MSA files shifts)
        rsalor_single_scores_map = {
            s["mutation_fasta"][0] + str(int(s["mutation_fasta"][1:-1])+resid_shift) + s["mutation_fasta"][-1]: s
            for s in rsalor_single_scores
        }

        # Read dataset
        dataset = CSV(sep=SEP, name=dms_name).read(dataset_input_path)
        for property_to_remove in REMOVE_PROPERTIES_IN_OUTPUT:
            dataset.remove_col(property_to_remove)

        # Assign predicted values
        dataset.add_empty_col(PREDICTION_PROPERTY, allow_replacement=True)
        for mutation_entry in dataset:
            mutations_arr = mutation_entry[MUTATION_PROPERTY].split(":")
            mutation_entry[PREDICTION_PROPERTY] = sum([rsalor_single_scores_map[mut]["RSA*LOR"] for mut in mutations_arr])

        # Save output
        print(f"   - save output to '{dataset_output_path}'")
        dataset.show()
        dataset.write(dataset_output_path)

    # Log DONE
    print("\nDONE.")


# Execution --------------------------------------------------------------------
if __name__ == "__main__":
    main()
