"""
Usage (call this from the associated bash script):
$ bash scoring_SiteRM_substitutions.sh
"""
import multiprocessing
import os
import argparse
import tqdm
import time
import logging

import numpy as np
import pandas as pd
import random

import tempfile
from cherryml import io as cherryml_io
from cherryml import utils as cherryml_utils
from cherryml.phylogeny_estimation import fast_tree
from ._datasets import DMS_substitutions_dataset, create_transitions_from_pairs_and_time, clinical_substitutions_dataset, GAP_CHARACTER
from cherryml import learn_site_specific_rate_matrices
from cherryml.caching import secure_parallel_output
from cherryml.utils import get_process_args
from cherryml import markov_chain

import sys
# # Get the directory of the current script
# current_script_path = os.path.realpath(__file__)
# # Calculate the path to the directory containing the 'caching' package
# package_directory = os.path.join(current_script_path, '../../..')
# # Normalize the path to resolve any '..'
# normalized_path = os.path.normpath(package_directory)
# # Append this path to sys.path if not already included
# if normalized_path not in sys.path:
#     sys.path.append(normalized_path)

from cherryml import caching as cherryml_caching

from typing import Dict, List, Optional, Tuple

def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


ALPHABET = cherryml_utils.get_amino_acids() + [GAP_CHARACTER]


def _transition_ll(
    x: str,
    y: str,
    matrix_exponentials: List[pd.DataFrame],
    matrix_exponential_indices: List[int],
    alphabet_to_int: Dict[str, int],
) -> float:
    assert(len(x) == len(y))
    assert(len(x) == len(matrix_exponential_indices))
    res = 0.0
    for i, idx in enumerate(matrix_exponential_indices):
        res += np.log(
            matrix_exponentials[idx][
                alphabet_to_int[x[i]],
                alphabet_to_int[y[i]]
            ]
        )
    return res


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_likelihood_dir"],
    exclude_args=["num_processes"],
)
def evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site_custom(
    transitions_dir: str,
    families: List[str],
    model_dir: str,
    alphabet: List[str],
    condition_on_non_gap: bool = False,
    num_processes: int = 1,
    output_likelihood_dir: Optional[str] = None,
    _version: str = "2024_05_02_v1",
):
    if condition_on_non_gap:
        raise NotImplementedError
    logger = logging.getLogger(__name__)
    logger.info(f"Going to evaluate SiteRM model on {len(families)} families using {num_processes} cores.")

    alphabet_to_int = {
        alphabet[i]: i for i in range(len(alphabet))
    }
    for f_idx, family in enumerate(families):
        print(f"Processing family: {family} ({f_idx + 1} / {len(families)})")
        transitions = cherryml_io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        assert(
            len(
                set(
                    (x, t) for (x, y, t) in transitions
                )
            ) == 1
        )
        t = transitions[0][2]

        site_specific_rate_matrices = np.stack(
                cherryml_io.read_pickle(
                os.path.join(model_dir, family + ".txt")
            ),
            axis=0,
        )
        num_sites = site_specific_rate_matrices.shape[0]
        matrix_exponentials = [
            matrix_exponential_reversible(
                rate_matrix=site_specific_rate_matrices[site_id, :, :],
                exponents=[t],  # Since we are using the same time for all positions.
            )[0, :, :]
            for site_id in range(num_sites)
        ]

        lls = []
        time_log_prob_x_x_t = 0
        time_log_probs_xi_xi_t = 0
        time_log_probs_yi_xi_t = 0

        # First, we compute the universal offset log P(x | x, t)
        # print(f"Going to compute log_prob_x_x_t ...")
        st = time.time()
        log_prob_x_x_t = _transition_ll(
            x=transitions[0][0],
            y=transitions[0][0],  # Here is where we feed in y=x
            matrix_exponentials=matrix_exponentials,
            matrix_exponential_indices=list(range(len(transitions[0][0]))),
            alphabet_to_int=alphabet_to_int,
        )
        time_log_prob_x_x_t += time.time() - st
        # print(f"log_prob_x_x_t = {log_prob_x_x_t}")

        for (x, y, t) in transitions:
            assert(len(x) == len(y))
            assert(len(x) == len(matrix_exponentials))
            mutated_indices = [
                i for i in range(len(x))
                if x[i] != y[i]
            ]
            mutated_x = "".join(
                [x[i] for i in mutated_indices]
            )
            mutated_y = "".join(
                [y[i] for i in mutated_indices]
            )

            # print(f"x = {x}")
            # print(f"y = {y}")
            # print(f"mutated_indices = {mutated_indices}")

            # We use the following identity to speed things up:
            # log P(y | x, t) = log P(x | x, t) + \sum_{i: x_i != y_i} log P(y_i | x_i, t)/P(x_i | x_i, t)
            # Now we compute the 'lost' likelihood for each test datapoint
            # print(f"Going to compute log_probs_xi_xi_t ...")
            st = time.time()
            log_probs_xi_xi_t = _transition_ll(
                x=mutated_x,
                y=mutated_x,  # Here is where we feed in y=x
                matrix_exponentials=matrix_exponentials,
                matrix_exponential_indices=mutated_indices,
                alphabet_to_int=alphabet_to_int,
            )
            time_log_probs_xi_xi_t += time.time() - st
            # Finally, we compute the 'won' likelihoods
            # print(f"Going to compute log_probs_yi_xi_t ...")
            st = time.time()
            log_probs_yi_xi_t = _transition_ll(
                x=mutated_x,
                y=mutated_y,
                matrix_exponentials=matrix_exponentials,
                matrix_exponential_indices=mutated_indices,
                alphabet_to_int=alphabet_to_int,
            )
            time_log_probs_yi_xi_t += time.time() - st
            # print(f"Done!")
            # Now we just use the identity to get the true log transition probabilities
            log_probs_y_x_t = log_prob_x_x_t + log_probs_yi_xi_t - log_probs_xi_xi_t
            lls.append(
                log_probs_y_x_t
            )
            # print(f"log_probs_y_x_t = {log_probs_y_x_t}")
        print(f"time_log_prob_x_x_t = {time_log_prob_x_x_t}")
        print(f"time_log_probs_xi_xi_t = {time_log_probs_xi_xi_t}")
        print(f"time_log_probs_yi_xi_t = {time_log_probs_yi_xi_t}")
        cherryml_io.write_transitions_log_likelihood(
            lls,
            os.path.join(
                output_likelihood_dir, family + ".txt"
            )
        )
        cherryml_caching.secure_parallel_output(
            output_dir=output_likelihood_dir, parallel_arg=family
        )


def _map_func(args):
    assert(len(args) == 14)
    families = args[0]
    tree = args[1]
    msa_dir = args[2]
    alphabet = args[3]
    regularization_rate_matrix = args[4]
    regularization_strength = args[5]
    device = args[6]
    num_rate_categories = args[7]
    alphabet_for_site_rate_estimation = args[8]
    rate_matrix_for_site_rate_estimation = args[9]
    num_epochs = args[10]
    quantization_grid_num_steps = args[11]
    use_vectorized_implementation = args[12]
    output_dir = args[13]
    for family_id, family in enumerate(families):
        msa_path = os.path.join(msa_dir, family + ".txt")
        msa = cherryml_io.read_msa(msa_path)
        print(f"Processing family {family_id + 1} / {len(families)} ({family}). Family len: {len(list(msa.values())[0])}, family size: {len(msa)}.")
        st = time.time()
        res_dict = learn_site_specific_rate_matrices(
            tree=tree,
            msa=msa,
            alphabet=alphabet,
            regularization_rate_matrix=regularization_rate_matrix,
            regularization_strength=regularization_strength,
            device=device,
            num_rate_categories=num_rate_categories,
            alphabet_for_site_rate_estimation=alphabet_for_site_rate_estimation,
            rate_matrix_for_site_rate_estimation=rate_matrix_for_site_rate_estimation,
            num_epochs=num_epochs,
            quantization_grid_num_steps=quantization_grid_num_steps,
            use_vectorized_implementation=use_vectorized_implementation,
        )
        learnt_rate_matrices = res_dict["learnt_rate_matrices"]
        learnt_site_rates = res_dict["learnt_site_rates"]
        learnt_tree = res_dict["learnt_tree"]
        sum_of_times = 0.0
        profiling_str_breakdown = ""
        for k, v in res_dict.items():
            if k.startswith("time_"):
                print(k, v)
                sum_of_times += float(v)
                profiling_str_breakdown += f"{k}: {v}\n"
        profiling_str = f"Family: {family}\nSum of times: {sum_of_times}\nTotal time: {time.time() - st}\nFamily len: {len(list(msa.values())[0])}\nFamily size: {len(msa)}\nTime breakdown:\n{profiling_str_breakdown}"
        print(profiling_str)
        cherryml_io.write_pickle(learnt_rate_matrices, os.path.join(output_dir, family + ".txt"))
        cherryml_io.write_site_rates(learnt_site_rates, os.path.join(output_dir, family + ".site_rates"))
        cherryml_io.write_tree(learnt_tree, os.path.join(output_dir, family + ".tree"))
        cherryml_io.write_str(profiling_str, os.path.join(output_dir, family + ".profiling"))
        secure_parallel_output(output_dir, family)


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes", "device"],
    output_dirs=["output_dir"],
)
def train_site_specific_rate_matrix_model__cached(
    msa_dir: str,
    families: List[str],
    regularization_rate_matrix_path: str,
    regularization_strength: float,
    device: str,
    num_rate_categories: int,
    rate_matrix_for_site_rate_estimation_path: str,
    num_processes: int,
    num_epochs: int = 100,
    quantization_grid_num_steps: int = 64,
    output_dir: Optional[str] = None,
):
    """
    Wrapper for CherryML's `learn_site_specific_rate_matrices`.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Going to train SiteRM model on {len(families)} families using {num_processes} cores.")

    map_args = [
        [
            get_process_args(process_rank, num_processes, families),  # families
            None,  # tree
            msa_dir,  # msa_dir
            ALPHABET[:],  # alphabet
            cherryml_io.read_rate_matrix(regularization_rate_matrix_path),  # regularization_rate_matrix
            regularization_strength,  # regularization_strength
            device,  # device
            num_rate_categories,  # num_rate_categories
            cherryml_utils.get_amino_acids(),  # alphabet_for_site_rate_estimation
            cherryml_io.read_rate_matrix(rate_matrix_for_site_rate_estimation_path),  # rate_matrix_for_site_rate_estimation
            num_epochs,  # num_epochs.
            quantization_grid_num_steps,  # quantization_grid_num_steps.
            False,  # use_vectorized_implementation
            output_dir,  # output_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func, map_args),
                total=len(map_args),
            )
        )
    logger.info("Done learning SiteRM model for all MSAs!")


def matrix_exponential_reversible(
    rate_matrix: np.array,
    exponents: List[float],
) -> np.array:
    """
    Compute matrix exponential (batched).

    Args:
        rate_matrix: Rate matrix for which to compute the matrix exponential
        exponents: List of exponents.
    Returns:
        3D tensor where res[:, i, i] contains exp(rate_matrix * exponents[i])
    """
    return markov_chain.matrix_exponential_reversible(
        exponents=exponents,
        fact=markov_chain.FactorizedReversibleModel(rate_matrix),
        device="cpu",
    )


def _condition_on_non_gap(conditional_probability_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: Assumes that the gap state is the last one in the alphabet!
    """
    if conditional_probability_matrix.columns[-1] != "-":
        raise ValueError(
            "It is assumed that the gap state is the last one! "
            "Last state was instead: "
            f"{conditional_probability_matrix.columns[-1]}"
        )

    data = conditional_probability_matrix.values.copy()
    row_sums = np.sum(data[:, :-1], axis=1, keepdims=True)
    data[:, :-1] /= row_sums
    data[:, -1] = 1.0

    res = pd.DataFrame(
        data,
        index=conditional_probability_matrix.index,
        columns=conditional_probability_matrix.columns
    )
    return res


def evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
    transitions: List[Tuple[str, str, float]],
    site_specific_rate_matrices: np.array,
    alphabet: List[str],
    condition_on_non_gap: bool = False,
) -> List[List[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the
    site-specific rate matrix model.

    It is assumed that the site_specific_rate_matrices are reversible.

    The log-likelihood under the site-specific rate matrix model is given by:
    P(y[i] | x[i], t) = log( exp(rate_matrix[i] * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        site_specific_rate_matrices: The site-specific rate matrices of the model;
            3D array indexed by [site_id, state_1, state_2]
        alphabet: Alphabet (of states).
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
    Returns:
        lls: The log-likelihood of each transition.
    """
    assert(len(transitions[0][0]) == site_specific_rate_matrices.shape[0])
    num_sites = len(transitions[0][0])
    matrix_exponentials = [
        matrix_exponential_reversible(
            rate_matrix=site_specific_rate_matrices[site_id, :, :],
            exponents=[t for (x, y, t) in transitions],
        )
        for site_id in range(num_sites)
    ]  # Indexed by [site_id][transition_id, :, :]
    res = []
    for i, (x, y, t) in enumerate(transitions):
        if len(x) != len(y):
            raise ValueError(
                f"Transition has two sequences of different lengths: {x}, {y}."
            )
        mexp_dfs = [
            pd.DataFrame(
                matrix_exponentials[site_id][i, :, :],
                index=alphabet,
                columns=alphabet,
            )
            for site_id in range(num_sites)
        ]  # Indexed by [site_id][:, :]
        if condition_on_non_gap:
            mexp_dfs = [
                _condition_on_non_gap(
                    mexp_dfs[site_id]
                )
                for site_id in range(num_sites)
            ]  # Indexed by [site_id][:, :]
        assert len(x) == len(y)
        assert len(x) == site_specific_rate_matrices.shape[0]
        lls = [
            np.log(
                mexp_dfs[site_id].at[x[site_id], y[site_id]]
            )
            for site_id in range(len(x))
        ]
        res.append(lls)
    return res


def _evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached__map_func(
    args: List,
):
    """
    Auxiliary version of
    "evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached"
    used for multiprocessing.
    """
    assert len(args) == 7
    transitions_dir = args[0]
    families = args[1]
    model_dir = args[2]
    output_transitions_log_likelihood_dir = args[3]
    output_transitions_log_likelihood_per_site_dir = args[4]
    condition_on_non_gap = args[5]
    alphabet = args[6]
    for family in families:
        transitions = cherryml_io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        site_specific_rate_matrices = cherryml_io.read_pickle(os.path.join(model_dir, f"{family}.txt"))

        ##### Now do the per-site LLs
        st = time.time()
        transitions_log_likelihood_per_site = (
            evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site(
                transitions=transitions,
                site_specific_rate_matrices=site_specific_rate_matrices,
                alphabet=alphabet,
                condition_on_non_gap=condition_on_non_gap,
            )
        )
        cherryml_io.write_transitions_log_likelihood_per_site(
            transitions_log_likelihood_per_site=transitions_log_likelihood_per_site,
            transitions_log_likelihood_per_site_path=os.path.join(
                output_transitions_log_likelihood_per_site_dir, family + ".txt"
            ),
        )
        secure_parallel_output(
            output_dir=output_transitions_log_likelihood_per_site_dir,
            parallel_arg=family,
        )
        ##### Now do total LLs
        transitions_log_likelihood = [
            sum(x) for x in transitions_log_likelihood_per_site
        ]
        cherryml_io.write_transitions_log_likelihood(
            transitions_log_likelihood=transitions_log_likelihood,
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )

        profiling_str = f"Total time: {time.time() - st}\n"
        for profiling_path in [  # (We write it in both places)
            os.path.join(
                output_transitions_log_likelihood_per_site_dir,
                family + ".profiling"
            ),
            os.path.join(
                output_transitions_log_likelihood_dir,
                family + ".profiling"
            )
        ]:
            with open(profiling_path, "w") as f:
                f.write(profiling_str)


@cherryml_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args=["num_processes"],
    exclude_args_if_default=["condition_on_non_gap"],
    write_extra_log_files=True,
)
def evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    families: List[str],
    model_dir: str,
    alphabet: List[str],
    condition_on_non_gap: bool = False,
    num_processes: int = 1,
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2024_04_26_v1",
) -> None:
    """
    Compute transitions log-likelihood under the site-specific rate matrix model.

    Rate matrices must be stored in {model_dir}/{family}.txt

    Args:
        transitions_dir: The directory with the transitions for which to
            compute the log-likelihood. The transitions for family 'family'
            should be in the file '{family}.txt'
        families: List of families for which to compute the log-likelihood.
        model_dir: The directory containing the rate matrices in the files
            '{family}.txt'
        alphabet: The alphabet (e.g. amino acids, or amino acids + gap)
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
        num_processes: How many processes to use to paralellize the likelihood
            evaluation. The parallelization is family-based.
        output_transitions_log_likelihood_dir: Where the log-likelihoods will
            get written. The log-likelihoods for family 'family' will be in
            the file '{family}.txt', with one line per transition.
        output_transitions_log_likelihood_per_site_dir: Where the per-site
            log-likelihoods will get written. The log-likelihoods for family
            'family' will be in the file '{family}.txt', with one line per
            transition.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Going to evaluate site-specific rate matrix model on "
        f"{len(families)} families using {num_processes} processes"
    )

    map_args = [
        [
            transitions_dir,
            get_process_args(process_rank, num_processes, families),
            model_dir,
            output_transitions_log_likelihood_dir,
            output_transitions_log_likelihood_per_site_dir,
            condition_on_non_gap,
            alphabet,
        ]
        for process_rank in range(num_processes)
    ]

    map_func = _evaluate_site_specific_rate_matrix_model_transitions_log_likelihood__cached__map_func
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(
                        map_func,
                        map_args,
                    ),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(
                    map_func,
                    map_args,
                ),
                total=len(map_args),
            )
        )


def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with SiteRM model.
    """
    parser = argparse.ArgumentParser(description='SiteRM model scoring')

    parser.add_argument('--subsample_n_sequences', default=100000000, type=int, help='How many sequences to subsample from each MSA.')

    # Method hyperparameters
    parser.add_argument('--tree_estimator_name', default="FastCherries", type=str, help='Tree estimator name.')
    parser.add_argument('--tree_estimation_rate_matrix_path', default="proteingym/baselines/SiteRM/lg.txt", type=str, help='Tree estimator rate matrix path.')
    parser.add_argument('--regularization_rate_matrix_path', default="proteingym/baselines/SiteRM/lg_with_gaps.txt" ,type=str, help='Rate matrix used as a prior to generate pseudocounts.')
    parser.add_argument('--regularization_strength', default=0.5, type=float, help='Between 0 (no regularization) and 1 (fully regularized)')
    parser.add_argument('--num_rate_categories', default=20, type=int, help='Number of rate categories. Used only for generating pseudocounts.')
    parser.add_argument('--evolutionary_time', default=1.0, type=float, help='Evolutionary time used to compute fitness scores. Basically, the evolutionary radius around the source protein to consider.')

    parser.add_argument('--DMS_reference_file_path', type=str, help='Path of DMS reference')
    parser.add_argument('--DMS_data_folder', type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_MSA_data_folder', type=str, help='Path of DMS folder (containing .a2m files)')
    parser.add_argument('--clinical_reference_file_path', type=str, help='Path of clinical reference')
    parser.add_argument('--clinical_data_folder', type=str, help='Path of clinical folder')
    parser.add_argument('--clinical_MSA_data_folder', type=str, help='Path of clinical folder (containing .a2m files)')
    parser.add_argument('--DMS_indices', type=int, default=list(range(217)), nargs='+', help='Indices of DMS folders')
    parser.add_argument('--output_scores_folder', default=None, type=str, help='Name of folder to write model scores to')
    parser.add_argument('--num_processes', default=1, type=int, help='Number of processes used to parallelize.')

    parser.add_argument('--device', type=str, help='Whether to use "cpu" or "cuda" to train the SiteRM model.')

    parser.add_argument('--clinical', action="store_true", help='Whether to use clinical instead')
    args = parser.parse_args()

    if args.clinical:
        # Just replace the DMS args since those get pulled down in the code (we are just "wrapping" the DMS code)
        args.DMS_reference_file_path = args.clinical_reference_file_path
        args.DMS_data_folder = args.clinical_data_folder
        args.DMS_MSA_data_folder = args.clinical_MSA_data_folder

    args.DMS_reference_file_path = os.path.abspath(args.DMS_reference_file_path)
    args.DMS_data_folder = os.path.abspath(args.DMS_data_folder)
    args.DMS_MSA_data_folder = os.path.abspath(args.DMS_MSA_data_folder)

    if args.clinical:
        dataset_dict = clinical_substitutions_dataset(
            subsample_n_sequences=args.subsample_n_sequences,
            clinical_reference_file_path=args.DMS_reference_file_path,
            clinical_data_folder=args.DMS_data_folder,
            clinical_MSA_data_folder=args.DMS_MSA_data_folder,
            max_families=len(args.DMS_indices),
        )
    else:
        dataset_dict = DMS_substitutions_dataset(
            subsample_n_sequences=args.subsample_n_sequences,
            DMS_reference_file_path=args.DMS_reference_file_path,
            DMS_data_folder=args.DMS_data_folder,
            DMS_MSA_data_folder=args.DMS_MSA_data_folder,
            max_families=len(args.DMS_indices),
        )

    # Need to set the families to only the desired subset
    families = [
        dataset_dict["families"][i]
        for i in args.DMS_indices
    ]
    dataset_dict["families"] = families

    model_dir = train_site_specific_rate_matrix_model__cached(
        msa_dir=dataset_dict["train_msa_dir"],
        families=dataset_dict["families"],
        regularization_rate_matrix_path=args.regularization_rate_matrix_path,
        regularization_strength=args.regularization_strength,
        device=args.device,
        num_rate_categories=args.num_rate_categories,
        rate_matrix_for_site_rate_estimation_path=args.tree_estimation_rate_matrix_path,
        num_processes=args.num_processes,
        num_epochs=100,
        quantization_grid_num_steps=64,
    )["output_dir"]

    # Report the slowest family
    families_and_total_times_and_profiling_strs = []
    families_and_fastcherries_times_and_profiling_strs = []
    for family in families:
        profiling_path = os.path.join(model_dir, family + ".profiling")
        assert(os.path.exists(profiling_path))
        with open(profiling_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Total time:"):
                    families_and_total_times_and_profiling_strs.append(
                        (float(line.split(":")[-1]), "".join(lines))
                    )
                if line.startswith("time_estimate_tree"):
                    families_and_fastcherries_times_and_profiling_strs.append(
                        (float(line.split(":")[-1]), "".join(lines))
                    )
    families_and_total_times_and_profiling_strs = sorted(families_and_total_times_and_profiling_strs, key=lambda x: x[0])
    print(f"Slowest family in total runtime:\n{families_and_total_times_and_profiling_strs[-1][1]}")
    families_and_fastcherries_times_and_profiling_strs = sorted(families_and_fastcherries_times_and_profiling_strs, key=lambda x: x[0])
    print(f"Slowest family for FastCherries:\n{families_and_fastcherries_times_and_profiling_strs[-1][1]}")


    # Now we compute the model scores. This will require creating a transitions
    # dataset with the times added, to be able to use the evaluation API...
    transitions_dir = create_transitions_from_pairs_and_time(
        families=families,
        transition_pairs_dir=dataset_dict["test_transition_pairs_aligned_region_dir"],
        t=args.evolutionary_time,
    )["output_transitions_dir"]
    print(f"transitions_dir = {transitions_dir}")

    ##### LL computation using custom computation (for computational speedup)
    lls_dir = evaluate_site_specific_rate_matrix_model_transitions_log_likelihood_per_site_custom(
        transitions_dir=transitions_dir,
        families=families,
        model_dir=model_dir,
        alphabet=ALPHABET[:],
        condition_on_non_gap=False,
        num_processes=args.num_processes,
    )["output_likelihood_dir"]
    print(f"lls_dir = {lls_dir}")

    if not os.path.exists(args.output_scores_folder):
        os.makedirs(args.output_scores_folder)

    for DMS_index in args.DMS_indices:
        # Now all the old boilerplate...
        mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[DMS_index]
        print(f"Computing scores for: {DMS_id} ({DMS_index + 1} / {len(args.DMS_indices)}) with SiteRM-{args.regularization_strength} model")
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        
        DMS_data_path = args.DMS_data_folder + os.sep + DMS_file_name
        DMS_data = pd.read_csv(DMS_data_path, low_memory=False)

        DMS_MSA_file_name = mapping_protein_seq_DMS["MSA_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        DMS_MSA_file_path = args.DMS_MSA_data_folder + os.sep + DMS_MSA_file_name
        if not os.path.exists(DMS_MSA_file_path):
            raise ValueError(
                f"DMS_MSA_file_path {DMS_MSA_file_path} does not exist!"
            )

        family = DMS_id
        lls = cherryml_io.read_transitions_log_likelihood(
            os.path.join(
                lls_dir,
                family + ".txt"
            )
        )

        model_scores = lls

        DMS_data['SiteRM_score'] = model_scores
        scoring_filename = args.output_scores_folder+os.sep+DMS_id+'.csv'
        if args.clinical:
            DMS_data[['mutant','SiteRM_score','DMS_bin_score']].to_csv(scoring_filename, index=False)
        else:
            DMS_data[['mutant','SiteRM_score','DMS_score']].to_csv(scoring_filename, index=False)


if __name__ == '__main__':
    cherryml_caching.set_cache_dir("_cache_cherryml")
    cherryml_caching.set_log_level(9)
    cherryml_caching.set_read_only(False)
    main()
