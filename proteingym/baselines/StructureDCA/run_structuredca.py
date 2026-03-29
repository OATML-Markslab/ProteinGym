
"""
Run StructureDCA mutational landscape predictions on DMS datasets.
"""

# Imports ----------------------------------------------------------------------
import os
import argparse
import time
try:
    from structuredca import StructureDCA
    from structuredca.structuredca import CSV
except:
    raise ImportError("Import of pip package 'structuredca' failed. Please install the package with 'pip install structuredca'.")


# Main -------------------------------------------------------------------------
def main():

    # Constants
    t1_total = time.time()
    MUTATION_PROPERTY = "mutant"
    PREDICTION_PROPERTY = "StructureDCA"
    PREDICTION_WITH_RSA_PROPERTY = "StructureDCA[RSA]"
    REMOVE_PROPERTIES_IN_OUTPUT = ["DMS_score", "DMS_score_bin"]
    SEP = ","
    CHAIN = "A"
    MAX_CPUS = 16

    # Set number of used CPUs
    n_cpu_total = os.cpu_count()
    if n_cpu_total is None:
        n_cpu_total = 1
    n_cpu_used = max(1, min(n_cpu_total // 2, MAX_CPUS))
    print("\n\nStructureDCA Runs Settings -------------------------------------------------")
    print(f" * Used CPUs: {n_cpu_used}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='StructureDCA arguments')
    parser.add_argument("--reference_file_path", type=str, help="Datasets reference '.csv' file.")
    parser.add_argument("--data_folder", type=str, help="Initial DMS datasets folder")
    parser.add_argument("--MSA_folder", type=str, help="MSA ('.a2m' or '.fasta') folder")
    parser.add_argument("--structure_folder", type=str, help="PDB ('.pdb') folder")
    parser.add_argument("--output_scores_folder", type=str, help="Output folder")
    args = parser.parse_args()
    print(f" * reference_file_path: '{args.reference_file_path}'")
    print(f" * data_folder: '{args.data_folder}'")
    print(f" * MSA_folder: '{args.MSA_folder}'")
    print(f" * structure_folder: '{args.structure_folder}'")
    print(f" * output_scores_folder: '{args.output_scores_folder}'")
    
    # Read reference file
    dataset_reference_path = args.reference_file_path
    dataset_reference = CSV.read(dataset_reference_path, sep=SEP, name="DMS reference")
    print(f"\nEvaluate StructureDCA on {len(dataset_reference)} DMS datasets")
    print(f" * dataset_reference_path: '{dataset_reference_path}'")
    dataset_reference.show()

    # Init output folder if required
    if not os.path.isdir(args.output_scores_folder):
        print(f"\nCreate new output directory '{args.output_scores_folder}'.")
        os.mkdir(args.output_scores_folder)

    # Loop on DMS datasets
    print(f"\nRun StructureDCA on datasets from '{args.data_folder}' ...")
    for i, dataset_entry in enumerate(dataset_reference):

        # Init metadata
        dms_name = dataset_entry["DMS_id"]
        msa_name = dataset_entry["MSA_filename"]
        pdb_name = dataset_entry["pdb_file"]
        resid_shift = int(dataset_entry["MSA_start"]) - 1
        print(f"\n * Run StructureDCA {i+1} / {len(dataset_reference)}: '{dms_name}'")

        # Set paths
        dataset_input_path = os.path.join(args.data_folder, f"{dms_name}.csv")
        dataset_output_path = os.path.join(args.output_scores_folder, f"{dms_name}.csv")
        if os.path.exists(dataset_output_path):
            print(f"Already computed scores for {dms_name}")
            continue
        msa_path = os.path.join(args.MSA_folder, f"{msa_name}")
        pdb_path = os.path.join(args.structure_folder, f"{pdb_name}")

        # Run StructureDCA
        t1 = time.time()
        sdca = StructureDCA(
            msa_path, pdb_path, CHAIN, # input data
            use_contacts_plddt_filter=True, # when working with AlphaFold 3D structures that may contain low pLDDT regions
            num_threads=n_cpu_used, # set number of used threads
            verbose=False, disable_warnings=True, # disable all logs
        )

        # Read dataset
        dataset = CSV.read(dataset_input_path, sep=SEP, name=dms_name)
        for property_to_remove in REMOVE_PROPERTIES_IN_OUTPUT:
            dataset.remove_col(property_to_remove)

        # Assign predicted values
        dataset.add_empty_col(PREDICTION_PROPERTY, allow_replacement=True)
        dataset.add_empty_col(PREDICTION_WITH_RSA_PROPERTY, allow_replacement=True)
        for mutation_entry in dataset:
            mutations_fasta = mutation_entry[MUTATION_PROPERTY]
            # map mutation as referenced in the fasta file to its msa coordinates
            # (sometimes the MSA range is smaller than the full fasta file)
            mutations_msa = ":".join([
                sin_mut[0] + str(int(sin_mut[1:-1])-resid_shift) + sin_mut[-1]
                for sin_mut in mutations_fasta.split(":")
            ])
            # Evalut dE of the mutation according to StructureDCA
            mutation_entry[PREDICTION_PROPERTY] = float(sdca.eval_mutation(mutations_msa, reweight_by_rsa=False))
            # Evalut dE of the mutation according to StructureDCA[RSA]
            mutation_entry[PREDICTION_WITH_RSA_PROPERTY] = float(sdca.eval_mutation(mutations_msa, reweight_by_rsa=True))
        t2 = time.time()
        print(f"   - done in {t2-t1:.1f} sec.")

        # Save output
        print(f"   - save output to '{dataset_output_path}'")
        dataset.show()
        dataset.write(dataset_output_path)

    # Log DONE
    t2_total = time.time()
    print(f"\nDONE. Total time: {t2_total - t1_total:.1f} sec.")


# Execution --------------------------------------------------------------------
if __name__ == "__main__":
    main()
