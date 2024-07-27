# This class is copied from the EVE codebase https://github.com/OATML-Markslab/EVE, removing the progress bar (since we're calculating weights on the fly so don't need one)
import multiprocessing
import time
from collections import defaultdict

import numba
from numba import prange
# from numba_progress import ProgressBar

import numpy as np
from tqdm import tqdm

def calc_weights_fast(matrix_mapped, identity_threshold, empty_value, num_cpus=1):
    """
        Modified from EVCouplings: https://github.com/debbiemarkslab/EVcouplings
        
        Note: Numba by default uses `multiprocessing.cpu_count()` threads. 
        On a cluster where a process might only have access to a subset of CPUs, this may be less than the number of CPUs available.
        The caller should ideally use len(os.sched_getaffinity(0)) to get the number of CPUs available to the process.
        
        Calculate weights for sequences in alignment by
        clustering all sequences with sequence identity
        greater or equal to the given threshold.
        Parameters
        ----------
        identity_threshold : float
            Sequence identity threshold
        """
    empty_idx = is_empty_sequence_matrix(matrix_mapped, empty_value=empty_value)  # e.g. sequences with just gaps or lowercase, no valid AAs
    N = matrix_mapped.shape[0]

    # Original EVCouplings code structure, plus gap handling
    if num_cpus != 1:
        # print("Calculating weights using Numba parallel (experimental) since num_cpus > 1. If you want to disable multiprocessing set num_cpus=1.")
        # print("Default number of threads for Numba:", numba.config.NUMBA_NUM_THREADS)
        
        # num_cpus > numba.config.NUMBA_NUM_THREADS will give an error.
        # But we'll leave it so that the user has to be explicit.
        numba.set_num_threads(num_cpus)
        print("Set number of threads to:", numba.get_num_threads())  # Sometimes Numba uses all the CPUs anyway
        
        num_cluster_members = calc_num_cluster_members_nogaps_parallel(matrix_mapped[~empty_idx], identity_threshold,
                                                                       invalid_value=empty_value)
        
    else:
        # Use the serial version
        num_cluster_members = calc_num_cluster_members_nogaps(matrix_mapped[~empty_idx], identity_threshold,
                                                              invalid_value=empty_value)

    # Empty sequences: weight 0
    weights = np.zeros((N))
    weights[~empty_idx] = 1.0 / num_cluster_members
    return weights

# Below are util functions copied from EVCouplings
def is_empty_sequence_matrix(matrix, empty_value):
    assert len(matrix.shape) == 2, f"Matrix must be 2D; shape={matrix.shape}"
    assert isinstance(empty_value, (int, float)), f"empty_value must be a number; type={type(empty_value)}"
    # Check for each sequence if all positions are equal to empty_value
    empty_idx = np.all((matrix == empty_value), axis=1)
    return empty_idx


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


# Fastmath should be safe here, as we can assume that there are no NaNs in the input etc.
@numba.jit(nopython=True, fastmath=True)  #parallel=True
def calc_num_cluster_members_nogaps(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/align/alignment.py#L1172.
    Modified to use non-gapped length and not counting gaps as sequence similarity matches.
    
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


@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """
    Parallel implementation of calc_num_cluster_members_nogaps above.
    
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
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
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
    Modified calc_num_cluster_members_nogaps_parallel to add tqdm progress bar - useful for multi-hour weights calc.
    
    progress_proxy : numba_progress.ProgressBar
        A handle on the progress bar to update
    update_frequency : int
        Similar to miniters in tqdm, how many iterations between updating the progress bar (which then will only print every `update_interval` seconds)
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
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i
        if progress_proxy is not None and i % update_frequency == 0:
            progress_proxy.update(update_frequency)

    return num_neighbors
