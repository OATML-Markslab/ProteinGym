# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
import io, tempfile, os, sys
from typing import Any, Mapping, Optional
from utils import residue_constants
from Bio.PDB import PDBParser
import numpy as np
import sys

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&`'*()_+-,.;\/{}|:<>?[]€ƒ…†‡‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ¡¢£¤¥¦§¨©ª«¬�­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

def extend_PDB_Chains(extend_to=500):
    """
    Extend the PDB chains range

    Parameter
    ---------------
    extend_to: int, Extend the chain name range to given range

    Return
    ---------------
    PDB_CHAIN_IDS: New PDB Chain ID string
    """
    global PDB_CHAIN_IDS
    global PDB_MAX_CHAINS

    used_chains = set([ ord(char) for char in PDB_CHAIN_IDS ])
    asc = 33
    while len(PDB_CHAIN_IDS) < extend_to:
        if asc not in used_chains and chr(asc).isprintable():
            PDB_CHAIN_IDS += chr(asc)
        asc += 1
    PDB_MAX_CHAINS = extend_to
    return PDB_CHAIN_IDS

# @dataclasses.dataclass(frozen=False)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray = np.array([])  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray = np.array([])  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray = np.array([])  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray = np.array([])  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray = np.array([])  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray = None  # [num_res, num_atom_type]

  # Secondary structure. 1: Loop, 2: Helix, 3: Sheet
  occupancies: np.ndarray = None  # [num_res, num_atom_type]

  # Molecular type: protein, RNA, DNA
  molecular_type: str = 'protein'

  def __init__(self,  atom_positions = None,   aatype = None, 
                      atom_mask = None,        residue_index = None, 
                      chain_index = None,      b_factors = None, 
                      occupancies = None,      molecular_type=None):
    
    if atom_positions is not None:
      self.atom_positions = atom_positions
    if aatype is not None:
      self.aatype = aatype
    if atom_mask is not None:
      self.atom_mask = atom_mask
    if residue_index is not None:
      self.residue_index = residue_index
    if chain_index is not None:
      self.chain_index = chain_index
    if b_factors is not None:
      self.b_factors = b_factors
    if occupancies is not None:
      self.occupancies = occupancies
    if molecular_type is not None:
      self.molecular_type = molecular_type

  def __post_init__(self):
    if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
      raise ValueError(
          f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains '
          'because these cannot be written to PDB format.')
    
    assert self.molecular_type in ('protein', 'RNA', 'DNA')
    atom_type_num = residue_constants.atom_type_num if self.molecular_type == 'protein' else residue_constants.na_atom_type_num

    if len(self.atom_positions) == 0:
      self.atom_positions   = np.zeros(shape=[0,atom_type_num,3])
    if len(self.aatype) == 0:
      self.aatype           = np.zeros(shape=[0])
    if len(self.chain_index) == 0:
      self.chain_index      = np.zeros(shape=[0])
    if len(self.atom_mask) == 0:
      self.atom_mask        = np.zeros(shape=[0,atom_type_num])
    
    if self.b_factors is None or len(self.b_factors) == 0:
      self.b_factors        = np.zeros(shape=[self.aatype.shape[0], atom_type_num])
    if self.occupancies is None or len(self.occupancies) == 0:
      self.occupancies      = np.ones(shape=[self.aatype.shape[0],  atom_type_num])
    
    if self.atom_positions.ndim != 3:
      raise ValueError(f'expect atom_positions dimension of 3, but got {self.atom_positions.shape}')
    if self.aatype.ndim != 1:
      raise ValueError(f'expect aatype dimension of 1, but got {self.aatype.shape}')
    if self.residue_index.ndim != 1:
      raise ValueError(f'expect residue_index dimension of 1, but got {self.residue_index.shape}')
    if self.chain_index is not None and self.chain_index.ndim != 1:
      raise ValueError(f'expect chain_index dimension of 1, but got {self.chain_index.shape}')
    if self.b_factors is not None and self.b_factors.ndim != 2:
      raise ValueError(f'expect b_factors dimension of 2, but got {self.b_factors.shape}')
    if self.occupancies is not None and self.occupancies.ndim != 2:
        raise ValueError(f'expect occupancies dimension of 2, but got {self.occupancies.shape}')

    num_res = self.atom_positions.shape[0]
    assert num_res == self.aatype.shape[0]
    assert num_res == self.chain_index.shape[0]
    assert num_res == self.atom_mask.shape[0]
    assert num_res == self.b_factors.shape[0]
    assert num_res == self.occupancies.shape[0]

    self.aatype        = self.aatype.astype(np.int64)
    self.chain_index   = self.chain_index.astype(np.int64)
    self.atom_mask     = self.atom_mask.astype(np.int64)
    self.residue_index = self.residue_index.astype(np.int64)

  def slice(self, i: int, j: int, ch_idx: Optional[int] = None, by: str = 'residue_index'):
    """
    Slice protein object with residue_index or index

    Parameters
    ---------------
    i: int, start index (close)
    j: int, end start (open)
    ch_idx: int, chain index
    by: str, 'residue_index' or 'index'
    """
    assert by in ('residue_index', 'index')

    if by == 'residue_index':
      mask = (i <= self.residue_index) & (self.residue_index < j)
    else:
      mask = np.zeros(self.aatype.shape[0]).astype(np.bool_)
      mask[i:j] = True

    if ch_idx is not None:
      mask &= (self.chain_index == ch_idx)

    return self.filter_by_mask(mask)

  def __add__(self, prot2):
    """
    Combine two proteins
    """
    assert self.molecular_type == prot2.molecular_type
    ret_b_factors = (self.b_factors is not None) and (prot2.b_factors is not None)
    ret_occupancies = (self.occupancies is not None) and (prot2.occupancies is not None)

    return Protein(
      np.concatenate([self.atom_positions, prot2.atom_positions]),
      np.concatenate([self.aatype, prot2.aatype]),
      np.concatenate([self.atom_mask, prot2.atom_mask]),
      np.concatenate([self.residue_index, prot2.residue_index]),
      np.concatenate([self.chain_index, prot2.chain_index]),
      np.concatenate([self.b_factors, prot2.b_factors]) if ret_b_factors else None,
      np.concatenate([self.occupancies, prot2.occupancies]) if ret_occupancies else None,
      self.molecular_type
    )

  def filter_by_mask(self, mask: np.ndarray):
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
    return Protein(
      self.atom_positions[mask],
      self.aatype[mask],
      self.atom_mask[mask],
      self.residue_index[mask],
      self.chain_index[mask],
      self.b_factors[mask] if self.b_factors is not None else None,
      self.occupancies[mask] if self.occupancies is not None else None,
      self.molecular_type
    )

  def filter_by_chain(self, chain_id_or_name):
    assert isinstance(chain_id_or_name, (int, str))
    if isinstance(chain_id_or_name, str):
      chain_id = PDB_CHAIN_IDS.index(chain_id_or_name)
    else:
      chain_id = chain_id_or_name
    mask = (self.chain_index == chain_id)
    return self.filter_by_mask(mask)
  
  def weight(self):
    """
    Return the molecular weight (KDa)
    """

    atom_types = residue_constants.atom_types if self.molecular_type == 'protein' else residue_constants.na_atom_types

    atom_masses = {"C": 12.01, "N": 14.01, "O": 16.00, "S": 32.06, "P": 30.97}
    weight = 0
    for residue in self.atom_mask:
        for atom_idx in range(len(residue)):
            atom_mask = residue[atom_idx]
            if atom_mask == 1:
                weight += atom_masses[atom_types[atom_idx][0]]
    return weight / 1_000

  def __repr__(self):

    # restypes_with_x = residue_constants.restypes_with_x if self.molecular_type == 'protein' else residue_constants.nuctypes_with_x

    text = f"Molecular weight: {self.weight():.3f}KDa\n"
    chain2seq = self.seq(as_seq=False, with_gap=True)
    for ch_name, ch_seq in chain2seq.items():
      text += f"{ch_name} ({len(ch_seq)}): {ch_seq}\n"
    return text

  # def __str__(self):
  #   return to_pdb(self)
  
  def __getitem__(self, val):
    """
    Acceptable values:
    str: chain name
    int: residue index
    slice: residue range
    """
    num_chain = len(np.unique(self.chain_index))
    if isinstance(val, str):
      return self.filter_by_chain(val)
    elif isinstance(val, int):
      assert num_chain == 1, f"Expect one chain in protein, but got {num_chain} chains"
      if val < 0:
        val = self.residue_index[-1] + 1 + val
      return self.slice(val, val+1)
    elif isinstance(val, slice):
      assert num_chain == 1, f"Expect one chain in protein, but got {num_chain} chains"
      start, stop, step = val.start, val.stop, val.step
      if start is None:
        start = 0
      elif start < 0:
        start = self.residue_index[-1] + 1 + start
        assert start >= 0, f"Expect start >= - ({self.residue_index[-1]} + 1)"
      if stop is None:
        stop = self.residue_index[-1] + 1
      elif stop < 0:
        stop = self.residue_index[-1] + 1 + stop
        assert stop > 0, f"Expect stop > - ({self.residue_index[-1]} + 1)"
      if step is None:
        step = 1

      assert start < stop, f"Expect start < stop, but got start={start}, stop={stop}"

      acceptable_residue_index = np.array([ int(d) for d in range(start, stop, step) ])
      mask = np.isin(self.residue_index, acceptable_residue_index)
      return self.filter_by_mask(mask)
    else:
      raise ValueError(f"Expect val be one of str,int,slice, but got {type(val)}")
  
  def re_index(self):
    for ch_idx in np.unique(self.chain_index):
        ch_idx = int(ch_idx)
        chain_mask = (self.chain_index == ch_idx)
        self.residue_index[chain_mask] = np.arange(1, chain_mask.sum()+1)
  
  def seq(self, as_seq=False, with_gap=False):
    """
    Return the sequence of current protein

    Parameters
    ------------
    as_seq: (True) return the concatenated sequence. (False) return dict of sequence
    with_gap: return the sequence with gap
    
    Return
    ------------
    as_seq=True, return str
    as_seq=False, return dict
    """

    restypes_with_x    = residue_constants.restypes_with_x if self.molecular_type == 'protein' else residue_constants.nuctypes_with_x
    return_seq_dict = {}
    return_seq = ""

    uniq_chs, ch_order = np.unique(self.chain_index, return_index=True)
    uniq_chs = uniq_chs[np.argsort(ch_order)]
    for ch_idx in uniq_chs:
      ch_idx  = int(ch_idx)
      ch_prot = self.filter_by_chain(ch_idx)
      ch_name = PDB_CHAIN_IDS[ch_idx]
      if ch_prot.residue_index.min() == 0:
          ch_prot.residue_index = ch_prot.residue_index + 1
      ch_res  = ['-'] * ch_prot.residue_index.max()
      for res_idx, res_aatype in zip(ch_prot.residue_index, ch_prot.aatype):
          ch_res[res_idx-1] = restypes_with_x[res_aatype]
      ch_seq  = "".join(ch_res)
      return_seq_dict[ch_name] = ch_seq if with_gap else ch_seq.replace('-', '')
      return_seq += ch_seq if with_gap else ch_seq.replace('-', '')
    
    if as_seq:
      return return_seq
    else:
      return return_seq_dict

  def save(self, file):
    from af2_features import save_prot_as_pdb, save_prot_as_cif
    if file.endswith('.cif'):
      save_prot_as_cif(self, file)
    else:
      save_prot_as_pdb(self, file)

def from_pdb_string(pdb_str: str, 
                    chain_id: Optional[str] = None, 
                    molecular_type: Optional[str] = None, 
                    # read_res_with_insertion_code:bool = False,
                    insertion_code_process: str = 'error',
                    stderr=sys.stderr) -> Protein:
  """Takes a PDB string and constructs a Protein object.

  WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

  Args:
    pdb_str: The contents of the pdb file
    chain_id: If chain_id is specified (e.g. A), then only that chain
      is parsed. Otherwise all chains are parsed.
    molecular_type: None or protein, DNA or RNA
    insertion_code_process: error, insert or ignore
    stderr: print error information

  Returns:
    A new `Protein` parsed from the pdb contents.
  """

  if molecular_type is not None:
    assert molecular_type in ('protein', 'DNA', 'RNA')
  assert insertion_code_process in ('error', 'insert', 'ignore')

  pdb_fh = io.StringIO(pdb_str)
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('none', pdb_fh)
  models = list(structure.get_models())
  if len(models) != 1:
    #raise ValueError(f'Only single model PDBs are supported. Found {len(models)} models.')
    print(f"Only single model PDBs are supported. Found {len(models)} models. Use the 1st model.", file=stderr)
  model = models[0]

  atom_positions = []
  aatype = []
  atom_mask = []
  residue_index = []
  chain_ids = []
  b_factors = []
  occupancies = []

  ## Read residues with insertion code
  cur_chain = None
  residue_increase_on_chain = 0

  for chain in model:
    if chain_id is not None and chain.id != chain_id:
      continue
    if cur_chain is None or cur_chain != chain.id:
      cur_chain = chain.id
      residue_increase_on_chain = 0
    for res in chain:
      if res.id[0] != ' ':
        print(f"Warning: Unexpected residue -- {res.id}, ignored", file=stderr)
        continue
      if res.id[2] != ' ':
        if insertion_code_process == 'insert':
          print(f"Warning: PDB contains an insertion code at chain {chain.id} and residue index {res.id[1]} -- {res.id}. Insert into chain", file=stderr)
          residue_increase_on_chain += 1
        elif insertion_code_process == 'ignore':
          print(f"Warning: PDB contains an insertion code at chain {chain.id} and residue index {res.id[1]} -- {res.id}. Ingored", file=stderr)
          continue
        else:
          raise ValueError(
              f'PDB contains an insertion code at chain {chain.id} and residue '
              f'index {res.id[1]}. These are not supported.')
      resname = res.resname.strip()
      if len(resname) == 3:
        ## Protein
        restype_3to1 = residue_constants.restype_3to1
        restype_order = residue_constants.restype_order
        atom_type_num = residue_constants.atom_type_num
        atom_order = residue_constants.atom_order
        atom_types = residue_constants.atom_types
        restype_num = residue_constants.restype_num
        if molecular_type is not None:
          if molecular_type != 'protein':
            continue
        molecular_type = 'protein'
      elif len(resname) == 2:
        ## DNA
        restype_3to1 = { 'DA': 'A', 'DT': 'T', 'DC': 'C', 'DG': 'G' }
        restype_order = residue_constants.nuctype_order
        atom_type_num = residue_constants.na_atom_type_num
        atom_order = residue_constants.na_atom_order
        atom_types = residue_constants.na_atom_types
        restype_num = residue_constants.nuctype_num
        if molecular_type is not None:
          if molecular_type != 'DNA':
            continue
        molecular_type = 'DNA'
      elif len(resname) == 1:
        ## RNA
        restype_3to1 = { 'A': 'A', 'U': 'U', 'C': 'C', 'G': 'G' }
        restype_order = residue_constants.nuctype_order
        atom_type_num = residue_constants.na_atom_type_num
        atom_order = residue_constants.na_atom_order
        atom_types = residue_constants.na_atom_types
        restype_num = residue_constants.nuctype_num
        if molecular_type is not None:
          if molecular_type != 'RNA':
            continue
        molecular_type = 'RNA'
      else:
        print(f"Unexpected resname: {resname}, ignored", file=stderr)
        # len(resname) == 4
        #ATOM     58  CA AARG A  20     131.717  89.202 105.130  0.50 78.68           C
        #ATOM     59  CB AARG A  20     131.422  88.374 106.378  0.50 78.68           C
        continue

      res_shortname = restype_3to1.get(resname, 'X')
      restype_idx = restype_order.get(res_shortname, restype_num)
      pos = np.zeros((atom_type_num, 3))
      mask = np.zeros((atom_type_num,))
      res_b_factors = np.zeros((atom_type_num,))
      res_occupancy = np.zeros((atom_type_num,))
      for atom in res:
        if atom.name not in atom_types:
          continue
        pos[atom_order[atom.name]] = atom.coord
        mask[atom_order[atom.name]] = 1.
        res_b_factors[atom_order[atom.name]] = 0 if atom.bfactor is None else atom.bfactor
        res_occupancy[atom_order[atom.name]] = 0 if atom.occupancy is None else atom.occupancy
      if np.sum(mask) < 0.5:
        # If no known atom positions are reported for the residue then skip it.
        continue
      aatype.append(restype_idx)
      atom_positions.append(pos)
      atom_mask.append(mask)
      residue_index.append(res.id[1]+residue_increase_on_chain)
      chain_ids.append(chain.id)
      b_factors.append(res_b_factors)
      occupancies.append(res_occupancy)

  # Chain IDs are usually characters so map these to ints.
  unique_chain_ids = np.unique(chain_ids)
  chain_id_mapping = {cid: PDB_CHAIN_IDS.index(cid) for n, cid in enumerate(unique_chain_ids)}
  chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

  return Protein(
      atom_positions=np.array(atom_positions),
      atom_mask=np.array(atom_mask),
      aatype=np.array(aatype),
      residue_index=np.array(residue_index),
      chain_index=chain_index,
      b_factors=np.array(b_factors),
      occupancies=np.array(occupancies),
      molecular_type=molecular_type)


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein, gap_ter_threshold=600.0) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  if prot.molecular_type == 'protein':
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types
  elif prot.molecular_type == 'RNA':
    nuctypes = residue_constants.nuctypes + ['X']
    res_1to3 = lambda r: nuctypes[r]
    atom_types = residue_constants.na_atom_types
  else:
    nuctypes = residue_constants.nuctypes + ['X']
    res_1to3 = lambda r: 'D' + nuctypes[r]
    atom_types = residue_constants.na_atom_types

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors
  occupancies = prot.occupancies
  if occupancies is None:
    occupancies = np.ones_like(b_factors)

  if np.any(aatype > residue_constants.restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  last_residue_index = residue_index[0]
  # last_res_ca_xyz = atom_positions[0, 1] if atom_mask[0, 1] == 1.0 else None
  last_resolved_ca_xyz = atom_positions[0, 1] if atom_mask[0, 1] == 1.0 else None # Last resolved Ca XYZ
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i] or (last_chain_index == chain_index[i] and last_residue_index > residue_index[i]):
      pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.
      atom_index = min(atom_index, 9999)
    #elif last_res_ca_xyz is not None and atom_mask[i, 1] == 1.0 and \
    #    np.sqrt(np.sum((last_res_ca_xyz - atom_positions[i, 1])**2)) >= gap_ter_threshold:
    elif last_resolved_ca_xyz is not None and atom_mask[i, 1] == 1.0 and \
          np.sqrt(np.sum((last_resolved_ca_xyz - atom_positions[i, 1])**2)) >= gap_ter_threshold:
      pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.
      atom_index = min(atom_index, 9999)
    # last_res_ca_xyz = atom_positions[i, 1] if atom_mask[i, 1] == 1.0 else None
    if atom_mask[i, 1] == 1:
      last_resolved_ca_xyz = atom_positions[i, 1]

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor, occupancy in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i], occupancies[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      # occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1
      atom_index = min(atom_index, 9999)
    
    last_residue_index = residue_index[i]

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]), chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.


def ideal_atom_mask(prot: Protein) -> np.ndarray:
  """Computes an ideal atom mask.

  `Protein.atom_mask` typically is defined according to the atoms that are
  reported in the PDB. This function computes a mask according to heavy atoms
  that should be present in the given sequence of amino acids.

  Args:
    prot: `Protein` whose fields are `numpy.ndarray` objects.

  Returns:
    An ideal atom mask.
  """
  return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True) -> Protein:
  """Assembles a protein from a prediction.

  Args:
    features: Dictionary holding model inputs.
    result: Dictionary holding model outputs.
    b_factors: (Optional) B-factors to use for the protein.
    remove_leading_feature_dimension: Whether to remove the leading dimension
      of the `features` values.

  Returns:
    A protein instance.
  """
  fold_output = result['structure_module']

  def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
    return arr[0] if remove_leading_feature_dimension else arr

  if 'asym_id' in features:
    chain_index = _maybe_remove_leading_dim(features['asym_id'])
  else:
    chain_index = np.zeros_like(_maybe_remove_leading_dim(features['aatype']))

  if b_factors is None:
    b_factors = np.zeros_like(fold_output['final_atom_mask'])

  return Protein(
      aatype=_maybe_remove_leading_dim(features['aatype']),
      atom_positions=fold_output['final_atom_positions'],
      atom_mask=fold_output['final_atom_mask'],
      residue_index=_maybe_remove_leading_dim(features['residue_index']) + 1,
      chain_index=chain_index,
      b_factors=b_factors)

def from_protein(prot: Protein, 
              atom_positions=None,
              aatype=None,
              atom_mask=None,
              residue_index=None, 
              chain_index=None,
              b_factors=None,
              occupancies=None) -> Protein:
    if isinstance(chain_index, int) or isinstance(chain_index, float):
        chain_index = np.ones_like(prot.chain_index) * chain_index
    return Protein(
        prot.atom_positions if atom_positions is None else atom_positions,
        prot.aatype         if aatype         is None else aatype,
        prot.atom_mask      if atom_mask      is None else atom_mask,
        prot.residue_index  if residue_index  is None else residue_index,
        prot.chain_index    if chain_index    is None else chain_index,
        prot.b_factors      if b_factors      is None else b_factors,
        prot.occupancies    if occupancies    is None else occupancies,
        prot.molecular_type
    )


