import multiprocessing
import time
from collections import defaultdict

import numba
from numba import prange
from numba_progress import ProgressBar

import numpy as np
from tqdm import tqdm


def compute_weight_eve(seq, list_seq, theta):
    # seq shape: (L * alphabet_size,)
    number_non_empty_positions = np.sum(seq)  # = np.dot(seq,seq), assuming it is a flattened one-hot matrix
    if number_non_empty_positions > 0:
        # Dot product of one-hot vectors x and y = (x == y).sum()
        matches = np.dot(list_seq, seq)
        denom = matches / number_non_empty_positions  # number_non_empty_positions = np.dot(seq,seq)
        denom = np.sum(denom > 1 - theta)  # Lood: Keeping >, and changing EVCouplings code to >
        return 1 / denom
    else:
        return 0.0  # return 0 weight if sequence is fully empty


def _compute_weight_global(i):
    seq = list_seq_global[i]
    # seq shape: (L * alphabet_size,)
    number_non_empty_positions = np.sum(seq)  # = np.dot(seq,seq), assuming it is a flattened one-hot matrix
    if number_non_empty_positions > 0:
        matches = np.dot(list_seq_global, seq)
        denom = matches / number_non_empty_positions  # number_non_empty_positions = np.dot(seq,seq)
        denom = np.sum(denom > 1 - theta_global)  # Lood: Keeping >, and changing EVCouplings code to >
        return 1 / denom
    else:
        return 0.0  # return 0 weight if sequence is fully empty


def _init_worker_calc_eve(list_seq, theta):
    # Initialize the worker process
    # Note: Using global is not ideal, but not sure how else
    # It should be safe since processes have private global variables
    global list_seq_global
    global theta_global
    list_seq_global = list_seq
    theta_global = theta


def compute_sequence_weights(list_seq, theta, num_cpus=1):
    _N, _seq_len, _alphabet_size = list_seq.shape  # = len(self.seq_name_to_sequence.keys()), len(self.focus_cols), len(self.alphabet)
    list_seq = list_seq.reshape((_N, _seq_len * _alphabet_size))
    print(f"Using {num_cpus} cpus for EVE weights computation")

    if num_cpus > 1:
        # Compute weights in parallel
        with multiprocessing.Pool(processes=num_cpus, initializer=_init_worker_calc_eve, initargs=(list_seq, theta)) as pool:
            # func = functools.partial(compute_weight, list_seq=list_seq, theta=theta)
            chunksize = max(min(8, int(_N / num_cpus / 4)), 1)
            print("chunksize: " + str(chunksize))
            # imap: Lazy version of map
            # Parallel progress bars are complicated, so just used a single one
            weights_map = tqdm(pool.imap(_compute_weight_global, range(_N), chunksize=chunksize),
                               total=_N, desc="Computing weights parallel EVE")
            weights = np.array(list(weights_map))
    else:
        weights_map = map(lambda seq: compute_weight_eve(seq, list_seq=list_seq, theta=theta), list_seq)
        weights = np.array(list(tqdm(weights_map, total=_N, desc="Computing weights serial EVE")))

    return weights


def is_empty_sequence_matrix(matrix, empty_value):
    assert len(matrix.shape) == 2, f"Matrix must be 2D; shape={matrix.shape}"
    assert isinstance(empty_value, (int, float)), f"empty_value must be a number; type={type(empty_value)}"
    # Check for each sequence if all positions are equal to empty_value
    empty_idx = np.all((matrix == empty_value), axis=1)
    return empty_idx


# See calc_num_cluster_members_nogaps
@numba.jit(nopython=True)  # , fastmath=True, parallel=True)
def calc_num_clusters_i(matrix, identity_threshold, invalid_value, i: int, L_non_gaps: float):
    N, L = matrix.shape
    L_non_gaps = 1.0 * L_non_gaps  # Show numba it's a float

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_clusters_i = 1  # Self
    # compare all pairs of sequences
    for j in range(N):
        if i == j:
            continue
        pair_matches = 0
        for k in range(L):
            # Edit(Lood): Don't count gaps as matches
            if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                pair_matches += 1
        # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
        # Note: Changed >= to > to match EVE / DeepSequence code
        if pair_matches / L_non_gaps > identity_threshold:
            num_clusters_i += 1

    return num_clusters_i


# Below are util functions copied from EVCouplings: https://github.com/debbiemarkslab/EVcouplings
# This code looks slow but it's because it's written as a numba kernel
# Fastmath should be safe here, as we can assume that there are no NaNs in the input etc.
@numba.jit(nopython=True)  # , fastmath=True, parallel=True
def calc_num_cluster_members_nogaps(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/align/alignment.py#L1172
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_matches = 0
            for k in range(L):
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors[i] += 1
            if pair_matches / L_non_gaps[j] > identity_threshold:
                num_neighbors[j] += 1

    return num_neighbors


def calc_weights_evcouplings(matrix_mapped, identity_threshold, empty_value, num_cpus=1):
    """
        From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
        Calculate weights for sequences in alignment by
        clustering all sequences with sequence identity
        greater or equal to the given threshold.
        Parameters
        ----------
        identity_threshold : float
            Sequence identity threshold
        """
    empty_idx = is_empty_sequence_matrix(matrix_mapped,
                                         empty_value=empty_value)  # e.g. sequences with just gaps or lowercase, no valid AAs
    N = matrix_mapped.shape[0]

    # Original EVCouplings code structure, plus gap handling
    if num_cpus != 1:
        # Numba native parallel:
        # print("Calculating weights using Numba parallel (experimental) since num_cpus > 1. "
        #       "If you want to disable multiprocessing set num_cpus=1.")
        # print("Default number of threads for Numba:", numba.config.NUMBA_NUM_THREADS)
        # # num_cpus > numba.config.NUMBA_NUM_THREADS will give an error.
        # # But we'll leave it so that the user has to be explicit.
        # numba.set_num_threads(num_cpus)
        # print("Set number of threads to:", numba.get_num_threads())
        # num_cluster_members = calc_num_cluster_members_nogaps_parallel(matrix_mapped[~empty_idx], identity_threshold,
        #                                                                invalid_value=empty_value)
        
        
        update_frequency=1000
        with ProgressBar(total=N, update_interval=30, miniters=update_frequency) as progress:  # can also use tqdm mininterval, maxinterval etc
            num_cluster_members = calc_num_cluster_members_nogaps_parallel_print(matrix_mapped[~empty_idx], identity_threshold,
                                                                       invalid_value=empty_value, progress_proxy=progress, update_frequency=update_frequency)
    #     print("Num CPUs for EVCouplings code:", num_cpus)
    #     print(
    #         f"Calculating weights using Numba JIT and multiprocessing (experimental) since num_cpus ({num_cpus}) > 1. "
    #         "If you want to disable multiprocessing set num_cpus=1.")
    #     with multiprocessing.Pool(processes=num_cpus, initializer=_init_worker_ev,
    #                               initargs=(matrix_mapped[~empty_idx], empty_value, identity_threshold)) as pool:
    #         # Simply: Chunksize is between 1 and 64, preferably N / num_cpus / 16,
    #         # so every CPU gets a 16th of their expected total every time they ask for more work.
    #         #  Too small values: Too much overhead sending simple indexes to workers, and them sending back results.
    #         #  Too large: May wait a while for the last worker's task to finish.
    #         chunksize = max(1, min(64, int(N / num_cpus / 16)))
    #         print("chunksize: " + str(chunksize))

    #         # imap: Lazy version of map
    #         # Parallel progress bars are complicated and pollute logs
    #         cluster_map = tqdm(pool.imap(_worker_func, range(N), chunksize=chunksize), total=N, mininterval=1)
    #         num_cluster_members = np.array(list(cluster_map))
    else:
        num_cluster_members = calc_num_cluster_members_nogaps(matrix_mapped[~empty_idx], identity_threshold,
                                                              invalid_value=empty_value)

    # Empty sequences: weight 0
    weights = np.zeros((N))
    weights[~empty_idx] = 1.0 / num_cluster_members
    return weights


# Multiprocessing with numba jit:
def _init_worker_ev(matrix, empty_value, identity_threshold):
    global matrix_mapped_global
    matrix_mapped_global = matrix
    L = matrix.shape[1]
    global empty_value_global
    empty_value_global = empty_value
    global identity_threshold_global
    identity_threshold_global = identity_threshold
    global L_i_global
    L_i_global = L - np.sum(matrix == empty_value, axis=1)
    print("Initialising worker")
    global global_func_num_clusters_i
    global_func_num_clusters_i = _global_calc_cluster_factory()
    try:
        start = time.perf_counter()
        _ = global_func_num_clusters_i(0)  # Timeout, and numba verbosity?
        end = time.perf_counter()
        print(f"Initialising worker took: {end - start:.2f}")
    except Exception as e:
        print("Worker initialisation failed:", e)
        raise e
    print("Function compiled")


def _worker_func(i):
    return global_func_num_clusters_i(i)


def _global_calc_cluster_factory():
    # @numba.jit(nopython=True)
    def func(i):
        return calc_num_clusters_i(matrix_mapped_global, identity_threshold_global, empty_value_global, i,
                                   L_non_gaps=L_i_global[i])
    return func


# Copied from EVCouplings
def map_from_alphabet(alphabet, default):
    """
    Creates a mapping dictionary from a given alphabet.
    Parameters
    ----------
    alphabet : str
        Alphabet for remapping. Elements will
        be remapped according to alphabet starting
        from 0
    default : Elements in matrix that are not
        contained in alphabet will be treated as
        this character
    Raises
    ------
    ValueError
        For invalid default character
    """
    map_ = {
        c: i for i, c in enumerate(alphabet)
    }

    try:
        default = map_[default]
    except KeyError:
        raise ValueError(
            "Default {} is not in alphabet {}".format(default, alphabet)
        )

    return defaultdict(lambda: default, map_)


# Copied from EVCouplings
def map_matrix(matrix, map_):
    """
    Map elements in a numpy array using alphabet
    Parameters
    ----------
    matrix : np.array
        Matrix that should be remapped
    map_ : defaultdict
        Map that will be applied to matrix elements
    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)

# The main function

@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
        Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1  # num_neighbors_i = 0  # TODO why did I make this 0 again? Probably because I thought I'd have to count i == j
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                if matrix[i, k] == matrix[j, k] and matrix[
                    i, k] != invalid_value:  # Edit(Lood): Don't count gaps as matches
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors

@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel_print(matrix, identity_threshold, invalid_value, progress_proxy=None, update_frequency=1000):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
        Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1  # num_neighbors_i = 0  # TODO why did I make this 0 again? Probably because I thought I'd have to count i == j
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                if matrix[i, k] == matrix[j, k] and matrix[
                    i, k] != invalid_value:  # Edit(Lood): Don't count gaps as matches
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i
        if progress_proxy is not None and i % update_frequency == 0:
            progress_proxy.update(update_frequency)

    return num_neighbors

########################
# Failed JIT parallel ideas:
# N, L = matrix_mapped[~empty_idx].shape
# L_i = L - np.sum(matrix_mapped[~empty_idx] == empty_value, axis=1)

# Idea 1: Calculate the full pair matrix (i,j) = number of matches between sequence i and sequence j
# neighbour_matrix = calc_num_pairs(matrix_mapped[~empty_idx], identity_threshold, invalid_value=empty_value)
# num_cluster_members = (pairs_matrix / L_i[:, None] >= identity_threshold).sum(axis=1)

# Idea 2: Calculate the pair matrix but applying thresholding inside the loop
# num_cluster_members = neighbour_matrix.sum(axis=1)
# num_cluster_members = calc_num_cluster_members_nogaps_all_vs_all(
#     matrix_mapped[~empty_idx], identity_threshold, invalid_value=empty_value  # matrix_mapped[~empty_idx]
# )

# Idea 3)

# Inside calling function calc_weights_evcouplings_parallel
# if num_cpus > 1:
# Compute weights in parallel
# print("Num CPUs for EVCouplings code:", num_cpus)
# with multiprocessing.Pool(processes=num_cpus, initializer=init_worker_ev, initargs=(matrix_mapped[~empty_idx], empty_value, identity_threshold)) as pool:
#     # func = functools.partial(compute_weight, list_seq=list_seq, theta=theta)
#     chunksize = max(min(32, int(N / num_cpus / 4)), 1)
#     print("chunksize: " + str(chunksize))
#     # imap: Lazy version of map
#     # Parallel progress bars are complicated
#     cluster_map = tqdm(pool.imap(_worker_func, range(N), chunksize=chunksize), total=N)

# @numba.jit(nopython=True, fastmath=True, parallel=True)
# def calc_num_pairs(matrix, identity_threshold, invalid_value):
#     """
#     From EVCouplings: https://github.com/debbiemarkslab/EVcouplings
#     Calculate number of sequences in alignment
#     within given identity_threshold of each other
#     Parameters
#     ----------
#     matrix : np.array
#         N x L matrix containing N sequences of length L.
#         Matrix must be mapped to range(0, num_symbols) using
#         map_matrix function
#     identity_threshold : float
#         Sequences with at least this pairwise identity will be
#         grouped in the same cluster.
#     Returns
#     -------
#     np.array
#         Vector of length N containing number of cluster
#         members for each sequence (inverse of sequence
#         weight)
#     """
#     N, L = matrix.shape
#     # L = 1.0 * L  # need to tell numba that L is a float
#
#     # Empty sequences are filtered out before this function and are ignored
#     # minimal cluster size is 1 (self)
#     # L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
#     neighbour_matrix = np.eye(N)  # dtype=np.bool
#     # Crucial: We assume none of the sequences are empty
#     # Construct a loop that counts a neighbour if the pairwise identity is above the threshold
#     pairs_j = np.zeros(N, dtype=np.int32)
#     for i in range(N):
#         # Calculate the non-gapped length of sequence i
#         # L_i = np.sum(matrix[i] != invalid_value)  # Can either use L_i or L_j to calculate the neighbor matrix, the output will simply be transposed
#         pairs_j[:] = 0
#         for j in range(N):
#             num_pairs = 0
#             for k in range(L):
#                 if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
#                     num_pairs += 1
#             pairs_j[j] = num_pairs  # Could also just add this as an array at the end of j loop
#         neighbour_matrix[i] = pairs_j  # Could also calc identity threshold here
#
#     return neighbour_matrix

# @numba.jit(nopython=True, parallel=True)
# def num_cluster_members_from_pair(matrix, identity_threshold, invalid_value, L_i):
#     N = matrix.shape[0]
#     pairs_matrix = np.zeros((N))
#     for i in prange(N):
#         pairs_matrix[i] = calc_num_clusters_i(matrix, identity_threshold=identity_threshold,
#                                               invalid_value=invalid_value, i=i,
#                                               L_non_gaps=L_i[i])
#     return pairs_matrix

# Slower than numpy.prange
# @numba.jit(nopython=True, parallel=True)
# def func_all_i(matrix, identity_threshold, invalid_value, L_i):
#     N = matrix.shape[0]
#     num_clusters = np.zeros(N)
#     for i in prange(N):
#         num_clusters[i] = calc_num_clusters_i(matrix, identity_threshold=identity_threshold,
#                                               invalid_value=invalid_value, i=i, L_non_gaps=L_i[i])
#     return num_clusters