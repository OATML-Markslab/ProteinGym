"""AF2 structure parsing with length-independent residue geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


@dataclass(frozen=True)
class StructureContext:
    """Residue-indexed structure quantities for one selected AF2 chain."""

    residue_keys: tuple[tuple[str, int, str], ...]
    aa: tuple[str, ...]
    xyz: np.ndarray
    burial: np.ndarray
    plddt: np.ndarray
    contact_count: np.ndarray

    def index_for_alignment_position(self, position: int, pdb_start: int) -> int | None:
        """Map a DMS/MSA one-based position to sequential PDB residue index."""

        index = position - pdb_start
        if index < 0 or index >= len(self.aa):
            return None
        return index

    def pair_distance(self, left_index: int, right_index: int) -> float:
        return float(np.linalg.norm(self.xyz[left_index] - self.xyz[right_index]))


def _select_chain(model, chain_id: str | None):
    chains = list(model.get_chains())
    if chain_id is not None:
        for chain in chains:
            if chain.id == chain_id:
                return chain
        raise ValueError(f"requested PDB chain is missing: {chain_id}")
    for chain in chains:
        if chain.id == "A":
            return chain
    if not chains:
        raise ValueError("AF2 structure contains no chains")
    return chains[0]


def parse_af2_structure(
    path: str | Path,
    *,
    chain_id: str | None = None,
    contact_radius: float = 8.0,
) -> StructureContext:
    """Parse standard residues and compute relative SASA, contacts, pLDDT.

    Burial is the within-protein percentile rank of negative residue SASA.
    This avoids the invalid contact-count / protein-length normalization used
    by legacy PureGraph sensors. C-alpha is used for Gly and C-beta otherwise,
    so Gly residues are covered rather than silently dropped.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("puregraph", str(path))
    model = next(structure.get_models())
    chain = _select_chain(model, chain_id)
    residues = [
        residue for residue in chain.get_residues()
        if residue.resname in THREE_TO_ONE and "CA" in residue
    ]
    if len(residues) < 2:
        raise ValueError(f"AF2 chain has fewer than two standard residues: {path}")

    # Biopython's Shrake-Rupley computes residue SASA from the complete atom set.
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(model, level="R")
    sasa = np.asarray([float(getattr(residue, "sasa", np.nan)) for residue in residues])
    if not np.isfinite(sasa).all():
        raise ValueError(f"AF2 residue SASA contains non-finite values: {path}")

    coords = []
    plddt = []
    for residue in residues:
        atom = residue["CB"] if "CB" in residue else residue["CA"]
        coords.append(np.asarray(atom.coord, dtype=np.float64))
        plddt.append(float(residue["CA"].bfactor))
    xyz = np.asarray(coords)
    distances = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
    contact_count = ((distances < contact_radius) & (distances > 0)).sum(axis=1).astype(np.float64)
    # Percentile rank is deterministic and invariant to protein length.
    burial = pd.Series(-sasa).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    confidence = np.clip(np.asarray(plddt, dtype=np.float64) / 100.0, 0.0, 1.0)
    keys = tuple((chain.id, int(residue.id[1]), residue.resname) for residue in residues)
    return StructureContext(
        residue_keys=keys,
        aa=tuple(THREE_TO_ONE[residue.resname] for residue in residues),
        xyz=xyz,
        burial=burial,
        plddt=confidence,
        contact_count=contact_count,
    )
