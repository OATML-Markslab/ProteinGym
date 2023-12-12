import datetime
import os,sys
import json
import argparse
from resource import getrusage, RUSAGE_SELF

import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--DMS_reference_file_path', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--random_seeds',type=int,nargs="+", help='Random seed for VAE model initialization')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--DMS_data_folder', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_scores_folder', type=str, help='Output location of computed evol indices')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument("--skip_existing", action="store_true", help="Skip scoring if output file already exists")
    parser.add_argument("--aggregation_method", choices=["full", "batch", "online"], default="full", help="Method to aggregate evol indices")
    parser.add_argument("--threshold_focus_cols_frac_gaps", type=float,
                        help="Maximum fraction of gaps allowed in focus columns - see data_utils.MSA_processing")
    args = parser.parse_args()

    print("Arguments:", args)

    assert os.path.isfile(args.DMS_reference_file_path), 'MSA list file does not exist: {}'.format(args.DMS_reference_file_path)
    mapping_file = pd.read_csv(args.DMS_reference_file_path)
    DMS_id = mapping_file['DMS_id'][args.protein_index]
    protein_name = mapping_file['MSA_filename'][args.protein_index].split(".a2m")[0]
    DMS_filename = mapping_file['DMS_filename'][args.protein_index]
    mutant = mapping_file['DMS_filename'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['MSA_filename'][args.protein_index]
    DMS_mutant_column = "mutant"
    if not DMS_filename.startswith(DMS_id):
        print(f"Warning: DMS id does not match DMS filename: {DMS_id} vs {DMS_filename}. Continuing for now.")

    # Check filepaths are valid
    evol_indices_output_filename = os.path.join(args.output_scores_folder, f'{DMS_id}.csv')


    if os.path.isfile(evol_indices_output_filename):
        print("Output file already exists: " + str(evol_indices_output_filename))

        if args.skip_existing:
            print("Skipping scoring since args.skip_existing is True")
            sys.exit(0)
        else:
            print("Overwriting existing file: " + str(evol_indices_output_filename))
            print("To skip scoring for existing files, use --skip_existing")
    # Check if surrounding directory exists
    else:
        print("Output file: " + str(evol_indices_output_filename))
        assert os.path.isdir(os.path.dirname(evol_indices_output_filename)), \
            'Output directory does not exist: {}. Please create directory before running script.\nOutput filename given: {}.\nDebugging curdir={}'\
            .format(os.path.dirname(evol_indices_output_filename), evol_indices_output_filename, os.getcwd())

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['MSA_theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    # Using data_kwargs so that if options aren't set, they'll be set to default values
    data_kwargs = {}
    if args.threshold_focus_cols_frac_gaps is not None:
        print("Using custom threshold_focus_cols_frac_gaps: ", args.threshold_focus_cols_frac_gaps)
        data_kwargs['threshold_focus_cols_frac_gaps'] = args.threshold_focus_cols_frac_gaps

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=False,  # Don't need weights for evol indices
            **data_kwargs,
    )


    args.mutations_location = args.DMS_data_folder + os.sep + DMS_filename
    for seed in args.random_seeds:
        model_name = protein_name + f"_seed_{seed}"
        print("Model name: "+str(model_name))

        model_params = json.load(open(args.model_parameters_location))

        model = VAE_model.VAE_model(
                        model_name=model_name,
                        data=data,
                        encoder_parameters=model_params["encoder_parameters"],
                        decoder_parameters=model_params["decoder_parameters"],
                        random_seed=42
        )
        model = model.to(model.device)
        checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name
        assert os.path.isfile(checkpoint_name), 'Checkpoint file does not exist: {}'.format(checkpoint_name)

        try:
            checkpoint = torch.load(checkpoint_name, map_location=model.device)  # Added map_location so that this works with CPU too
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
        except Exception as e:
            print("Unable to load VAE model checkpoint {}".format(checkpoint_name))
            raise e

        list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
            msa_data=data,
            list_mutations_location=args.mutations_location,
            mutant_column=DMS_mutant_column,
            num_samples=args.num_samples_compute_evol_indices,
            batch_size=args.batch_size,
            aggregation_method=args.aggregation_method
        )

        df = {}
        df['mutant'] = list_valid_mutations
        df[f'evol_indices_seed_{seed}'] = evol_indices
        df = pd.DataFrame(df)

        if os.path.exists(evol_indices_output_filename) and seed != args.random_seeds[0]:
            prev_df = pd.read_csv(evol_indices_output_filename)
            prev_len = len(prev_df)
            df = pd.merge(prev_df, df, on='mutant', how='inner')
            # checking that the mutants match after the first seed (first seed will overwrite original score file)
            assert len(df) == len(prev_df), "Length of merged dataframe doesn't match previous length, mutants must not match across seeds"
            df.to_csv(evol_indices_output_filename, index=False)
        else:
            df.to_csv(evol_indices_output_filename, index=False)
