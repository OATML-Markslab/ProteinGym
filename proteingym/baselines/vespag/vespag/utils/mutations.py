from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import polars as pl
import rich
import torch
from jaxtyping import Float

from .utils import AMINO_ACIDS


@dataclass
class SAV:
    position: int
    from_aa: str
    to_aa: str
    one_indexed: bool = False

    @classmethod
    def from_sav_string(
        cls, sav_string: str, one_indexed: bool = False, offset: int = 0
    ) -> SAV:
        from_aa, to_aa = sav_string[0], sav_string[-1]
        position = int(sav_string[1:-1]) - offset
        if one_indexed:
            position -= 1
        return SAV(position, from_aa, to_aa, one_indexed=one_indexed)

    def __str__(self) -> str:
        pos = self.position
        if self.one_indexed:
            pos += 1
        return f"{self.from_aa}{pos}{self.to_aa}"

    def __hash__(self):
        return hash(str(self))


@dataclass
class Mutation:
    savs: list[SAV]

    @classmethod
    def from_mutation_string(
        cls, mutation_string: str, one_indexed: bool = False, offset: int = 0
    ) -> Mutation:
        return Mutation(
            [
                SAV.from_sav_string(sav_string, one_indexed=one_indexed, offset=offset)
                for sav_string in mutation_string.split(":")
            ]
        )

    def __str__(self) -> str:
        return ":".join([str(sav) for sav in self.savs])

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        yield from self.savs


def mask_non_mutations(
    gemme_prediction: Float[torch.Tensor, "length 20"], wildtype_sequence
) -> Float[torch.Tensor, "length 20"]:
    """
    Simply set the predicted effect of the wildtype amino acid at each position (i.e. all non-mutations) to 0
    """
    gemme_prediction[
        torch.arange(len(wildtype_sequence)),
        torch.tensor([AMINO_ACIDS.index(aa) for aa in wildtype_sequence]),
    ] = 0.0

    return gemme_prediction


def read_mutation_file(
    mutation_file: Path, one_indexed: bool = False
) -> dict[str, list[SAV]]:
    mutations_per_protein = defaultdict(list)
    for row in pl.read_csv(mutation_file).iter_rows():
        mutations_per_protein[row[0]].append(
            Mutation.from_mutation_string(row[1], one_indexed)
        )

    return mutations_per_protein


def compute_mutation_score(
    y: Float[torch.Tensor, "length 20"],
    mutation: Union[Mutation, SAV],
    alphabet: str = AMINO_ACIDS,
    normalize: bool = False,
    pbar: rich.progress.Progress = None,
    progress_id: int = None,
) -> float:
    if pbar:
        pbar.advance(progress_id)

    if isinstance(mutation, Mutation):
        score = sum(
            [y[sav.position][alphabet.index(sav.to_aa)].item() for sav in mutation]
        )
    else:
        score = y[mutation.position][alphabet.index(mutation.to_aa)].item()

    if normalize:
        score = 1 / (1 + math.exp(-score))
    return score
