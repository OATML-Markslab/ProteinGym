"""Utility functions for data processing and modelling"""

import time
from pathlib import Path
from typing import Sequence

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from kermut import AA_TO_IDX, ALPHABET

PROTEINGYM_DIR = Path(__file__).resolve().parents[5]
KERMUT_DIR = Path(__file__).resolve().parents[2]


def load_zero_shot(dataset: str, zero_shot_method: str) -> pd.DataFrame:
    zero_shot_dir = KERMUT_DIR / "data/zero_shot_fitness_predictions" / zero_shot_method
    zero_shot_col = zero_shot_name_to_col(zero_shot_method)

    if zero_shot_method == "TranceptEVE":
        zero_shot_dir = zero_shot_dir / "TranceptEVE_L"
    if zero_shot_method == "ESM2":
        zero_shot_dir = zero_shot_dir / "650M"

    df_zero = pd.read_csv(zero_shot_dir / f"{dataset}.csv")
    # Average duplicates
    df_zero = df_zero[["mutant", zero_shot_col]].groupby("mutant").mean().reset_index()
    return df_zero


def zero_shot_name_to_col(key) -> str:
    return {
        "ProteinMPNN": "pmpnn_ll",
        "ESM_IF1": "esmif1_ll",
        "EVE": "evol_indices_ensemble",
        "TranceptEVE": "avg_score",
        "GEMME": "GEMME_score",
        "VESPA": "VESPA",
        "ESM2": "esm2_t33_650M_UR50D",
        "MSA_Transformer": "esm_msa1b_t12_100M_UR50S_ensemble",
    }[key]


def load_embeddings(
    dataset: str,
    df: pd.DataFrame,
    multiples: bool = False,
    embedding_type: str = "ESM2",
) -> torch.Tensor:
    if multiples:
        emb_path = (
            Path(f"data/embeddings/substitutions_multiples/{embedding_type}")
            / f"{dataset}.h5"
        )
    else:
        emb_path = (
            Path(f"data/embeddings/substitutions_singles/{embedding_type}")
            / f"{dataset}.h5"
        )
    emb_path = KERMUT_DIR / emb_path
    # Check if file exists
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}.")

    # Occasional issues with reading the file due to concurrent access
    tries = 0
    while tries < 10:
        try:
            with h5py.File(emb_path, "r", locking=True) as h5f:
                embeddings = torch.tensor(h5f["embeddings"][:]).float()
                mutants = [x.decode("utf-8") for x in h5f["mutants"][:]]
            break
        except OSError:
            tries += 1
            time.sleep(10)
            pass

    # If not already mean-pooled
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=1)

    # Keep entries that are in the dataset
    keep = [x in df["mutant"].tolist() for x in mutants]
    embeddings = embeddings[keep]
    mutants = np.array(mutants)[keep]
    # Ensures matching indices
    idx = [df["mutant"].tolist().index(x) for x in mutants]
    embeddings = embeddings[idx]
    return embeddings


def prepare_gp_kwargs(DMS_id: str, wt_sequence: str, cfg: DictConfig):
    # Prepare arguments for gp/kernel
    gp_kwargs = {"use_zero_shot": cfg.gp.use_zero_shot}
    if cfg.gp.use_mutation_kernel:
        tokenizer = hydra.utils.instantiate(cfg.gp.mutation_kernel.tokenizer)
        wt_sequence = tokenizer(wt_sequence).squeeze()
        conditional_probs = np.load(
            KERMUT_DIR / f"data/conditional_probs/ProteinMPNN/{DMS_id}.npy"
        )

        gp_kwargs["wt_sequence"] = wt_sequence
        gp_kwargs["conditional_probs"] = torch.tensor(conditional_probs).float()
        gp_kwargs["km_cfg"] = cfg.gp.mutation_kernel
        gp_kwargs["use_global_kernel"] = cfg.gp.use_global_kernel

        if cfg.gp.mutation_kernel.use_distances:
            coords = np.load(KERMUT_DIR / f"data/structures/coords/{DMS_id}.npy")
            gp_kwargs["coords"] = torch.tensor(coords).float()
    else:
        tokenizer = None
    return gp_kwargs, tokenizer


def load_proteingym_dataset(dataset: str, multiples: bool = False) -> pd.DataFrame:
    if multiples:
        base_path = PROTEINGYM_DIR / "data/substitutions_multiples"
    else:
        base_path = PROTEINGYM_DIR / "data/substitutions_singles"
    df = pd.read_csv(base_path / f"{dataset}.csv")
    return df.reset_index(drop=True)


def hellinger_distance(p: torch.tensor, q: torch.tensor) -> torch.Tensor:
    """Compute Hellinger distance between input distributions:

    HD(p, q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))

    Args:
        x1 (torch.Tensor): Shape (n, 20)
        x2 (torch.Tensor): Shape (n, 20)

    Returns:
        torch.Tensor: Shape (n, n)
    """
    batch_size = p.shape[0]
    # Compute only the lower triangular elements if p == q
    if torch.allclose(p, q):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=-1)
        hellinger_tril = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[tril_i]) - torch.sqrt(q[tril_j])) ** 2, dim=1)
        )
        hellinger_matrix = torch.zeros((batch_size, batch_size))
        hellinger_matrix[tril_i, tril_j] = hellinger_tril
        hellinger_matrix[tril_j, tril_i] = hellinger_tril
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        hellinger = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[mesh_i]) - torch.sqrt(q[mesh_j])) ** 2, dim=1)
        )
        hellinger_matrix = hellinger.reshape(batch_size, batch_size)
    return hellinger_matrix.float()


class Tokenizer:
    """Tokenizer for amino acid sequences. Converts sequences to one-hot encoded tensors."""

    def __init__(self, flatten: bool = True):
        super().__init__()
        # Uses the standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY
        self.alphabet = list(ALPHABET)
        self.flatten = flatten
        self._aa_to_tok = AA_TO_IDX
        self._tok_to_aa = {v: k for k, v in self._aa_to_tok.items()}

    def encode(self, batch: Sequence[str]) -> torch.LongTensor:
        batch_size = len(batch)
        seq_len = len(batch[0])
        toks = torch.zeros((batch_size, seq_len, 20))
        for i, seq in enumerate(batch):
            for j, aa in enumerate(seq):
                toks[i, j, self._aa_to_tok[aa]] = 1

        if self.flatten:
            # Check if batch is str
            if isinstance(batch, str):
                return toks.squeeze().flatten().long()
            else:
                return toks.reshape(batch_size, seq_len * 20).long()
        else:
            return toks.squeeze().long()

    def __call__(self, batch: Sequence[str]):
        return self.encode(batch)
