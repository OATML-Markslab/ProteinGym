"""Utility functions for data processing and modelling"""

import time
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src import AA_TO_IDX, ALPHABET


def load_zero_shot(dataset: str, zero_shot_method: str) -> pd.DataFrame:
    zero_shot_dir = Path("data/zero_shot_fitness_predictions") / zero_shot_method
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
        if cfg.gp.mutation_kernel.conditional_probs_method == "ProteinMPNN":
            conditional_probs = np.load(
                Path(f"data/conditional_probs/ProteinMPNN/{DMS_id}.npy")
            )
        else:
            raise NotImplementedError

        gp_kwargs["wt_sequence"] = wt_sequence
        gp_kwargs["conditional_probs"] = torch.tensor(conditional_probs).float()
        gp_kwargs["km_cfg"] = cfg.gp.mutation_kernel
        gp_kwargs["use_global_kernel"] = cfg.gp.use_global_kernel

        if cfg.gp.mutation_kernel.use_distances:
            coords = np.load(f"data/structures/coords/{DMS_id}.npy")
            gp_kwargs["coords"] = torch.tensor(coords).float()
    else:
        tokenizer = None
    return gp_kwargs, tokenizer


def load_proteingym_dataset(dataset: str, multiples: bool = False) -> pd.DataFrame:
    if multiples:
        base_path = Path("data/substitutions_multiples")
    else:
        base_path = Path("data/substitutions_singles")
    df = pd.read_csv(base_path / f"{dataset}.csv")

    df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
    return df.reset_index(drop=True)


def prepare_datasets(
    cfg: DictConfig, use_multiples: bool = False
) -> Tuple[List[str], List[str]]:
    # Datasets require >48GB VRAM
    large_datasets = [
        "POLG_CXB3N_Mattenberger_2021",
        "POLG_DEN26_Suphatrakul_2023",
    ]

    df_ref = pd.read_csv(Path("data", "DMS_substitutions.csv"))
    # Filter reference file according to experiment setup
    if cfg.dataset == "benchmark":
        if use_multiples:
            df_ref = df_ref[df_ref["includes_multiple_mutants"]]
            df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
            # Remove GCN4_YEAST_Staller_2018 due to very high mutation count
            df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
        else:
            # Ignore large datasets
            df_ref = df_ref[~df_ref["DMS_id"].isin(large_datasets)]
    elif cfg.dataset == "ablation":
        df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
    elif cfg.dataset == "large":
        df_ref = df_ref[df_ref["DMS_id"].isin(large_datasets)]
    else:
        # Single dataset
        df_ref = df_ref[df_ref["DMS_id"] == cfg.dataset]

    df_ref = df_ref.sort_values(by="DMS_id")
    datasets = df_ref["DMS_id"].tolist()
    sequences = df_ref["target_seq"].tolist()

    # Determine which datasets to process
    model_name = cfg.custom_name if "custom_name" in cfg else cfg.gp.name
    split_method = cfg.split_method
    overwrite = cfg.overwrite
    if overwrite:
        output_dataset = datasets
        output_sequences = sequences
    else:
        # If not overwrite, run only on missing datasets
        output_dataset = []
        output_sequences = []
        for dataset, seq in zip(datasets, sequences):
            out_path = (
                Path("results/predictions")
                / dataset
                / f"{model_name}_{split_method}.csv"
            )
            if not out_path.exists():
                # Dataset only processed if method does not require inter-residue distances
                if dataset == "BRCA2_HUMAN_Erwood_2022_HEK293T":
                    if cfg.gp.use_mutation_kernel:
                        if not cfg.gp.mutation_kernel.use_distances:
                            output_dataset.append(dataset)
                            output_sequences.append(seq)

                        else:
                            print(f"Skipping {dataset} (use_distances=True)")
                    else:
                        output_dataset.append(dataset)
                        output_sequences.append(seq)
                else:
                    output_dataset.append(dataset)
                    output_sequences.append(seq)

    return output_dataset, output_sequences


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
