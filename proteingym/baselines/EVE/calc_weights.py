# Basically train_VAE.py but just calculating the weights
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import data_utils


def create_argparser():
    parser = argparse.ArgumentParser(description='VAE')

    # If we don't have a mapping file, just use a single MSA path
    parser.add_argument("--MSA_filepath", type=str, help="Full path to MSA")

    # If we have a mapping file with one MSA path per line
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored', required=True)
    parser.add_argument('--DMS_reference_file_path', type=str, help='List of proteins and corresponding MSA file name', required=True)
    parser.add_argument('--DMS_index', type=int, help='Row index of protein in input mapping file', required=True)
    parser.add_argument('--MSA_weights_location', type=str,
                        help='Location where weights for each sequence in the MSA will be stored', required=True)
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument("--num_cpus", type=int, help="Number of CPUs to use", default=1)
    parser.add_argument("--skip_existing", help="Will quit gracefully if weights file already exists", action="store_true", default=False)
    parser.add_argument("--overwrite", help="Will overwrite existing weights file", action="store_true", default=False)
    parser.add_argument("--calc_method", choices=["evcouplings", "eve", "both", "identity"], help="Method to use for calculating weights. Note: Both produce the same results as we modified the evcouplings numba code to mirror the eve calculation", default="evcouplings")
    parser.add_argument("--threshold_focus_cols_frac_gaps", type=float,
                        help="Maximum fraction of gaps allowed in focus columns - see data_utils.MSA_processing")
    return parser


def main(args):
    print("Arguments:", args)

    weights_file = None

    if args.MSA_filepath is not None:
        assert os.path.isfile(args.MSA_filepath), f"MSA filepath {args.MSA_filepath} doesn't exist"
        msa_location = args.MSA_filepath
    else:
        # Use mapping file
        assert os.path.isfile(args.DMS_reference_file_path), f"DMS reference file {args.DMS_reference_file_path} doesn't seem to exist"
        mapping_file = pd.read_csv(args.DMS_reference_file_path)
        protein_name = mapping_file['MSA_filename'][args.DMS_index].split(".a2m")[0]        
        msa_location = args.MSA_data_folder + os.sep + mapping_file['MSA_filename'][args.DMS_index]
        print("Protein name: " + str(protein_name))
        # If weights_file is in the df_mapping, use that instead
        if "weight_file_name" in mapping_file.columns:
            weights_file = args.MSA_weights_location + os.sep + mapping_file["weight_file_name"][args.DMS_index]
            print("Using weights filename from mapping file:", weights_file)

    print("MSA file: " + str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
        print(f"Using custom theta value {theta} instead of loading from mapping file.")
    else:
        try:
            theta = float(mapping_file['MSA_theta'][args.DMS_index])
        except KeyError as e:
            # Overriding previous errors is bad, but we're being nice to the user
            raise KeyError("Couldn't load theta from mapping file. "
                           "NOT using default value of theta=0.2; please specify theta manually. Specific line:",
                           mapping_file[args.DMS_index],
                           "Previous error:", e)
        assert not np.isnan(theta), "Theta is NaN, please provide a custom theta value"

    print("Theta MSA re-weighting: " + str(theta))

    # Using data_kwargs so that if options aren't set, they'll be set to default values
    data_kwargs = {}
    if args.threshold_focus_cols_frac_gaps is not None:
        print("Using custom threshold_focus_cols_frac_gaps: ", args.threshold_focus_cols_frac_gaps)
        data_kwargs['threshold_focus_cols_frac_gaps'] = args.threshold_focus_cols_frac_gaps

    if not os.path.isdir(args.MSA_weights_location):
        os.makedirs(args.MSA_weights_location, exist_ok=True)
        raise NotADirectoryError(f"{args.MSA_weights_location} is not a directory."
                                 f"Could create it automatically, but at the moment raising an error.")
    else:
        print(f"MSA weights directory: {args.MSA_weights_location}")

    if weights_file is None:
        print("Weights filename not found - writing to new file")
        weights_file = args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'

    print(f"Writing to {weights_file}")
    # First check that the weights file doesn't exist
    if os.path.isfile(weights_file) and not args.overwrite:
        if args.skip_existing:
            print("Weights file already exists, skipping, since --skip_existing was specified")
            exit(0)
        else:
            raise FileExistsError(f"File {weights_file} already exists. "
                                  f"Please delete it if you want to re-calculate it. "
                                  f"If you want to skip existing files, use --skip_existing.")

    # The msa_data processing has a side effect of saving a weights file
    _ = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=weights_file,
        num_cpus=args.num_cpus,
        weights_calc_method=args.calc_method,
        overwrite_weights=args.overwrite,
        debug_only_weights=True,
        **data_kwargs,
    )


if __name__ == '__main__':
    start = time.perf_counter()
    parser = create_argparser()
    args = parser.parse_args()
    main(args)
    end = time.perf_counter()
    print(f"calc_weights.py took {end-start:.2f} seconds in total.")
