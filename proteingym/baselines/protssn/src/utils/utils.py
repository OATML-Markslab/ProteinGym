# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import biotite.structure
import numpy as np
import pandas as pd
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import *
from Bio import SeqIO

def param_num(model):
    total = sum([param.numel() for param in model.parameters() if param.requires_grad])
    num_M = total/1e6
    if num_M >= 1000:
        return "Number of parameter: %.2fB" % (num_M/1e3)
    else:
        return "Number of parameter: %.2fM" % (num_M)

def create_mutant_file(root):
    root = "data/evaluation/ProteinGym_substitutions/"
    proteins = os.listdir(root)
    for protein in proteins:
        files = os.listdir(os.path.join(root, protein))
        files_ = sorted([file for file in files if file.split(".")[1].isdigit()])
        df = pd.read_table(os.path.join(root,protein, files_[0]))
        if len(files_) > 1:
            for i in range(1, len(files_)):
                file_ = pd.read_table(os.path.join(root,protein, files_[i]))
                df = pd.concat([df, file_])
            df.to_csv(os.path.join(root, protein, f"{protein}.tsv"), sep="\t", index=False)
        else:
            df.to_csv(os.path.join(root, protein, f"{protein}.tsv"), sep="\t", index=False)


def read_fasta(file_path, key):
    return str(getattr(SeqIO.read(file_path, 'fasta'), key))

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    # elif fpath.endswith('pdb'):
    else:
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """

    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)

def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.

    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
        torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )
