import hashlib
import math
import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numba as nb
import numpy as np
import torch
from pydantic import BaseModel, Field

from poet.utils import hash_of_string_list


def compute_hamming_csim_np(
    seqs: np.ndarray,
    ungapped_msa: np.ndarray,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> np.ndarray:
    """
    This function has an awkward spec. The point was to test
    compute_homology_weights_np and demonstrate its flaws wrt handling gap tokens.

    Compute the hamming similiarity between sequences in seqs and ungapped_msa, among
    non-gap tokens.

    Assumes

    - seqs[gap tokens in seqs] == gap_token
    - ungapped_msa[gap tokens in ungapped_msa] == gap_token_mask

    Example
    ---
    1. hamming_sim(
        ABC,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 2

    2. hamming_sim(
        A-C,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 1

    3. hamming_sim(
        AB-,  # a sequence in seqs
        ABA,  # a sequence in ungapped_msa
    ) = 2

    4. hamming_sim(
        AB-,  # a sequence in seqs
        AB-,  # a sequence in ungapped_msa
    ) = 2  # not 3 b/c of the matching gaps

    5. hamming_sim(
        ABC,  # a sequence in seqs
        AB-,  # a sequence in ungapped_msa
    ) = 2
    """
    return (seqs[:, np.newaxis] == ungapped_msa).sum(axis=2, dtype=np.uint16)


def compute_hamming_csim_torch(
    seqs: torch.Tensor,
    ungapped_msa: torch.Tensor,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> torch.Tensor:
    return (seqs.unsqueeze(1) == ungapped_msa).sum(dim=2)


@nb.njit(locals={"sim": nb.uint16})
def hamming_sim(x, y, N):
    # Compute the Hamming sim between two sequences x and y
    sim = 0
    for i in range(N):
        if x[i] == y[i]:
            sim += 1
    return sim


@nb.njit(parallel=True)
def compute_hamming_csim_nb(
    seqs: np.ndarray,
    ungapped_msa: np.ndarray,
    gap_token: int = 20,
    gap_token_mask: int = 255,
) -> np.ndarray:
    """See compute_hamming_csim_np"""
    N1, M = seqs.shape
    N2, _ = ungapped_msa.shape
    sims = np.zeros((N1, N2), dtype=np.uint16)  # Initialize an array to store sims

    # Compute sims between all pairs of sequences
    for i in nb.prange(N1):
        for j in range(N2):
            sims[i, j] = hamming_sim(seqs[i], ungapped_msa[j], M)

    return sims


def _compute_homology_weights(
    ungapped_msa: np.ndarray,
    gap_token: int,
    gap_token_mask: int,
    theta: float,
    hamming_csim_func: Callable,
    max_memory: int = 20,
    can_use_torch: bool = True,
) -> np.ndarray:
    use_torch = can_use_torch and torch.cuda.is_available()
    if use_torch:
        hamming_csim_func = compute_hamming_csim_torch
    batch_size = math.floor(
        2
        * 1024
        * 1024
        * 1024
        / (ungapped_msa.shape[0] * ungapped_msa.shape[1])
        * max_memory
        / 40
    )

    batch_size = 1 if batch_size == 0 else batch_size

    neighbors = []
    if not use_torch:
        masked_ungapped_msa = ungapped_msa.copy()
    else:
        ungapped_msa = torch.from_numpy(ungapped_msa).byte().cuda()
        masked_ungapped_msa = ungapped_msa.clone()
    masked_ungapped_msa[masked_ungapped_msa == gap_token] = gap_token_mask
    for b_start in range(0, len(ungapped_msa), batch_size):
        b_end = b_start + batch_size
        seqs = ungapped_msa[b_start:b_end]

        sim = hamming_csim_func(
            seqs=seqs,
            ungapped_msa=masked_ungapped_msa,
            gap_token=gap_token,
            gap_token_mask=gap_token_mask,
        )
        if not use_torch:
            sim = sim / (seqs != gap_token).sum(axis=1, keepdims=True)
            d = 1 - sim
            assert ((d >= 0) & (d <= 1)).all()
            this_neighbors = (d <= theta).sum(axis=1)
        else:
            sim = sim / (seqs != gap_token).sum(dim=1, keepdim=True)
            d = 1 - sim
            assert ((d >= 0) & (d <= 1)).all()
            this_neighbors = (d <= theta).sum(dim=1).cpu()
        neighbors.append(this_neighbors)
    return np.concatenate(neighbors)


def compute_homology_weights(
    ungapped_msa: np.ndarray,
    theta: float = 0.2,
    gap_token: int = 20,
    gap_token_mask: int = 255,
    hamming_csim_func: Callable = compute_hamming_csim_nb,
    result_cache_dir: Optional[Path] = None,
    can_use_torch: bool = True,
) -> tuple[int, np.ndarray]:
    """
    Calculate the effective number of sequences and sampling probability for the NEIGHBORS and NEIGHBORS_NO_LIMIT sampling methods using numpy.

    Parameters:

        ungapped_msa (np.ndarray): The MSA (from .fa).
        theta (float, optional): A parameter used to determine the similarity between sequences. Default is 0.2.
        gap_token (int, optional): The token representing gaps in the (Uniprot21 encoded) MSA. Default is 20.
        gap_token_mask (int): token for masking gaps. should be a token not representing any other value.

    Returns:

        tuple[int, np.ndarray]: A tuple containing the effective number of sequences and the sampling probability for each sequence in the MSA.
    """
    assert gap_token >= 0
    if result_cache_dir is not None:
        result_cache_dir = result_cache_dir / "compute_homology_weights"
        result_cache_dir.mkdir(exist_ok=True, parents=True)
        result_hash = hashlib.sha1(ungapped_msa.view(np.uint8)).hexdigest()
        additional_hash_components = []
        additional_hash_components.append(f"{theta=}")
        additional_hash_components.append(f"{gap_token=}")
        additional_hash_components.append(f"{gap_token_mask=}")
        result_hash = hash_of_string_list(additional_hash_components + [result_hash])
        result_filepath = (result_cache_dir / result_hash).with_suffix(".pkl")
        if result_filepath.is_file():
            return pickle.load(open(result_filepath, "rb"))

    neighbors = _compute_homology_weights(
        ungapped_msa=ungapped_msa,
        gap_token=gap_token,
        gap_token_mask=gap_token_mask,
        theta=theta,
        hamming_csim_func=hamming_csim_func,
        can_use_torch=can_use_torch,
    )
    n_eff = np.sum(1 / neighbors)

    p = 1 / neighbors
    p /= np.sum(p)

    if result_cache_dir is not None:
        pickle.dump((n_eff, p), open(result_filepath, "wb"))
    return n_eff, p


class TopSampler(BaseModel):
    sampler_type: Literal["top"] = "top"

    def get_weights(
        self, msa: np.ndarray, gap_token: int, result_cache_dir: Optional[Path] = None
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        return None, None

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        assert weights is None
        return np.arange(len(msa))


class RandomSampler(BaseModel):
    sampler_type: Literal["random"] = "random"

    def get_weights(
        self, msa: np.ndarray, gap_token: int, result_cache_dir: Optional[Path] = None
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        return None, None

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        assert weights is None
        rng = np.random.default_rng(seed) if seed is not None else np.random
        return rng.permutation(len(msa))


class NeighborsSampler(BaseModel):
    sampler_type: Literal["neighbors"] = "neighbors"
    theta: float = 0.2
    can_use_torch: bool = True

    def get_weights(
        self, msa: np.ndarray, gap_token: int, result_cache_dir: Optional[Path] = None
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        assert msa.dtype == np.uint8
        return compute_homology_weights(
            ungapped_msa=msa,
            theta=self.theta,
            gap_token=gap_token,
            gap_token_mask=255,
            result_cache_dir=result_cache_dir,
            can_use_torch=self.can_use_torch,
        )

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        assert weights is not None
        if len(msa) == 0:
            return np.array([], dtype=int)
        size = len(msa)
        rng = np.random.default_rng(seed) if seed is not None else np.random
        return rng.choice(len(msa), replace=False, size=size, p=weights / weights.sum())


class MSASampler(BaseModel):
    # TODO: refactor msa sampling code...
    method: Union[TopSampler, RandomSampler, NeighborsSampler] = Field(
        ..., discriminator="sampler_type"
    )
    force_include_first: bool = False
    max_similarity: float = 1.0
    max_dissimilarity: float = 1.0

    def _get_sim_filtered_idxs(self, msa: np.ndarray) -> np.ndarray:
        nonnormalized_sim = (msa == msa[[0]]).sum(axis=1)
        normfactor = msa.shape[1]
        norm_sim = nonnormalized_sim / normfactor

        assert (norm_sim.min() >= 0) and (norm_sim.max() <= 1)
        dsim = 1 - norm_sim

        max_sim_filter = norm_sim <= self.max_similarity
        max_dissim_filter = dsim <= self.max_dissimilarity
        return np.where(max_sim_filter & max_dissim_filter)[0]

    def get_sample_idxs(
        self,
        msa: np.ndarray,
        gap_token: int,
        seed: Optional[int] = None,
        result_cache_dir: Optional[Path] = None,
    ) -> np.ndarray:
        _, weights = self.method.get_weights(
            msa=msa, gap_token=gap_token, result_cache_dir=result_cache_dir
        )

        original_msa_sample_idxs = np.arange(len(msa))
        sample_idxs = self._get_sim_filtered_idxs(msa)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]
        msa = msa[sample_idxs]
        weights = weights[sample_idxs]

        sample_idxs = self.method.get_sample_idxs(msa=msa, weights=weights, seed=seed)
        original_msa_sample_idxs = original_msa_sample_idxs[sample_idxs]
        del msa, weights

        if self.force_include_first:
            original_msa_sample_idxs = np.concatenate(
                [[0], original_msa_sample_idxs[original_msa_sample_idxs != 0]]
            )
        return original_msa_sample_idxs
