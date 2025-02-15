from pathlib import Path

import h5py
import numpy as np
import polars as pl
import rich.progress as progress
import torch
from jaxtyping import Float

from vespag.utils.type_hinting import PrecisionType


class PerResidueDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        embedding_file: Path,
        annotation_file: Path,
        cluster_file: Path,
        precision: PrecisionType,
        device: torch.device,
        max_len: int,
        limit_cache: bool = False,
    ):
        self.precision = precision
        self.device = device
        self.dtype = torch.float if precision == "float" else torch.half
        self.limit_cache = limit_cache

        self.cluster_df = pl.read_csv(cluster_file)
        self.protein_embeddings = {
            key: torch.tensor(np.array(data[()]), device=self.device, dtype=self.dtype)
            for key, data in progress.track(
                h5py.File(embedding_file, "r").items(),
                description=f"Loading embeddings from {embedding_file}",
                transient=True,
            )
            if key in self.cluster_df["protein_id"]
        }
        self.protein_annotations = {
            key: torch.tensor(
                np.array(data[()][:max_len]), device=self.device, dtype=self.dtype
            )
            for key, data in progress.track(
                h5py.File(annotation_file, "r").items(),
                description=f"Loading annotations from {annotation_file}",
                transient=True,
            )
            if key in self.cluster_df["protein_id"]
        }

        self.residue_embeddings = torch.cat(
            [
                self.protein_embeddings[protein_id]
                for protein_id in progress.track(
                    self.cluster_df["protein_id"],
                    description="Pre-loading embeddings",
                    transient=True,
                )
            ]
        )
        self.residue_annotations = torch.cat(
            [
                self.protein_annotations[protein_id]
                for protein_id in progress.track(
                    self.cluster_df["protein_id"],
                    description="Pre-loading annotations",
                    transient=True,
                )
            ]
        )

    def __getitem__(
        self, idx
    ) -> tuple[
        Float[torch.Tensor, "length embedding_dim"], Float[torch.Tensor, "length 20"]
    ]:
        embedding = self.residue_embeddings[idx]
        annotation = self.residue_annotations[idx]
        if self.precision == "half":
            embedding = embedding.half()
            annotation = annotation.half()
        else:
            embedding = embedding.float()
            annotation = annotation.float()

        embedding = embedding.clone()
        annotation = annotation.clone()
        return embedding, annotation

    def __len__(self):
        return self.residue_embeddings.shape[0]
