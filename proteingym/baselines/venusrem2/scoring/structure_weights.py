import os
from typing import List, Optional

import numpy as np


def _torch():
    import torch

    return torch


MAX_ASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0, "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def resolve_pdb_file(
    protein_name: Optional[str], pdb_file: Optional[str] = None, pdb_dir: Optional[str] = None
) -> Optional[str]:
    if pdb_file and os.path.exists(pdb_file):
        return pdb_file
    if protein_name and pdb_dir:
        candidate = os.path.join(pdb_dir, f"{protein_name}.pdb")
        if os.path.exists(candidate):
            return candidate
    return None


def _rsa_tensor_from_values(seq_len: int, rsa_vals: List[float]):
    torch = _torch()
    weights = torch.zeros(seq_len, 1, dtype=torch.float32)
    n = min(seq_len, len(rsa_vals))
    if n > 0:
        weights[:n, 0] = torch.tensor(rsa_vals[:n], dtype=torch.float32)
    return weights


def _load_rsa_with_mdtraj(resolved: str, seq_len: int):
    import mdtraj as md

    traj = md.load(resolved)
    sasa = md.shrake_rupley(traj, mode="residue")[0]
    residues = list(traj.topology.residues)
    rsa_vals: List[float] = []
    for res, sasa_val in zip(residues, sasa):
        aa = THREE_TO_ONE.get(res.name.upper(), None)
        if aa is None or aa not in MAX_ASA:
            rsa_vals.append(0.0)
            continue
        asa = float(sasa_val) * 100.0
        rsa = asa / MAX_ASA[aa]
        rsa_vals.append(float(np.clip(rsa, 0.0, 1.5)))
    return _rsa_tensor_from_values(seq_len, rsa_vals)


def _load_rsa_with_biopython(resolved: str, seq_len: int):
    from Bio.PDB import PDBParser, ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", resolved)
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    sr.compute(structure, level="R")

    rsa_vals: List[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                aa = THREE_TO_ONE.get(residue.get_resname().upper(), None)
                if aa is None or aa not in MAX_ASA:
                    rsa_vals.append(0.0)
                    continue
                asa = float(getattr(residue, "sasa", 0.0))
                rsa = asa / MAX_ASA[aa]
                rsa_vals.append(float(np.clip(rsa, 0.0, 1.5)))
        break
    return _rsa_tensor_from_values(seq_len, rsa_vals)


_EXPERIMENTAL_EXPDTA = (
    "X-RAY",
    "NEUTRON",
    "ELECTRON CRYSTALLOGRAPHY",
    "ELECTRON MICROSCOPY",
    "SOLUTION NMR",
    "SOLID-STATE NMR",
    "FIBER DIFFRACTION",
    "POWDER DIFFRACTION",
    "NMR",
)
_PREDICTED_HINTS = (
    "ALPHAFOLD",
    "COLABFOLD",
    "ESMFOLD",
    "OMEGAFOLD",
    "ROSETTAFOLD",
    "OPENFOLD",
    "THEORETICAL MODEL",
    "AF2",
    "AF3",
    "PLDDT",
)
_REFINEMENT_HINTS = ("PHENIX", "REFMAC", "BUSTER", "SHELX", "REFINE")


def _header_lines(path: str, limit: int = 400) -> list[str]:
    lines: list[str] = []
    try:
        with open(path, "r", errors="replace") as handle:
            for i, raw in enumerate(handle):
                if i >= limit:
                    break
                rec = raw[:6].strip().upper()
                if rec in {"ATOM", "HETATM"}:
                    break
                lines.append(raw.rstrip("\n"))
    except OSError:
        return []
    return lines


def _sample_ca_bfactors(path: str, limit: int = 256) -> list[float]:
    values: list[float] = []
    try:
        with open(path, "r", errors="replace") as handle:
            for raw in handle:
                if not raw.startswith("ATOM"):
                    continue
                if len(raw) < 66:
                    continue
                if raw[12:16].strip() != "CA":
                    continue
                try:
                    values.append(float(raw[60:66]))
                except ValueError:
                    continue
                if len(values) >= limit:
                    break
    except OSError:
        return []
    return values


def classify_pdb_origin(path: str) -> tuple[str, str]:
    """Return ``(kind, detail)``: predicted / experimental / unknown.

    Experimental structures (crystal, NMR, cryo-EM) store temperature factors in
    the B-factor column, not pLDDT.
    """
    header = "\n".join(_header_lines(path)).upper()
    expdta = ""
    for line in header.splitlines():
        if line.startswith("EXPDTA"):
            expdta = line[6:].strip()
            break

    if expdta:
        if any(key in expdta for key in _PREDICTED_HINTS):
            return "predicted", expdta
        if any(key in expdta for key in _EXPERIMENTAL_EXPDTA):
            return "experimental", expdta or "experimental"

    if any(key in header for key in _PREDICTED_HINTS):
        return "predicted", "predicted-model remark"
    if any(key in header for key in _REFINEMENT_HINTS) and any(
        key in header for key in ("X-RAY", "CRYSTAL", "DIFFRACTION", "NMR", "MICROSCOPY")
    ):
        return "experimental", "refinement of an experimental structure"

    bfactors = _sample_ca_bfactors(path)
    if bfactors and max(bfactors) > 100.5:
        return "experimental", "B-factors > 100 (temperature factors, not pLDDT)"
    return "unknown", ""


def plddt_skip_reason(path: Optional[str]) -> Optional[str]:
    """Why pLDDT should not be read from this PDB, or None if it is allowed."""
    if not path:
        return None
    kind, detail = classify_pdb_origin(path)
    if kind != "experimental":
        return None
    label = detail or "experimental"
    return (
        f"PDB is a crystal/experimental structure ({label}); "
        "no pLDDT (B-factor is a temperature factor). Skipping pLDDT decay."
    )


def load_residue_rsa_weights_from_pdb(
    seq_len: int,
    protein_name: Optional[str],
    pdb_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
):
    resolved = resolve_pdb_file(protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir)
    if resolved is None:
        return None

    for loader in (_load_rsa_with_mdtraj, _load_rsa_with_biopython):
        try:
            return loader(resolved, seq_len)
        except Exception:
            continue
    return None


def load_residue_plddt_from_pdb(
    seq_len: int,
    protein_name: Optional[str],
    pdb_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
):
    resolved = resolve_pdb_file(protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir)
    if resolved is None:
        return None
    if plddt_skip_reason(resolved):
        return None
    try:
        from Bio.PDB import PDBParser
        torch = _torch()
        parser_pdb = PDBParser(QUIET=True)
        structure = parser_pdb.get_structure("prot", resolved)
        model = list(structure.get_models())[0]
        plddt_vals: List[float] = []
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                ca = residue["CA"] if "CA" in residue else list(residue.get_atoms())[0]
                plddt_vals.append(float(ca.get_bfactor()))
        weights = torch.zeros(seq_len, 1, dtype=torch.float32)
        n = min(seq_len, len(plddt_vals))
        if n > 0:
            weights[:n, 0] = torch.tensor(plddt_vals[:n], dtype=torch.float32) / 100.0
        return weights.clamp(0.0, 1.0)
    except Exception:
        return None
