#!/usr/bin/env python

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.internal_coords import *
from Bio.PDB.ic_rebuild import structure_rebuild_test
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBIO import PDBIO

import numpy as np
import pandas as pd
import os


cif_parser = MMCIFParser(QUIET=True)
pdb_parser = PDBParser(QUIET=True)
io=PDBIO()
angles_names = ['phi', "psi", "omg", "tau", "chi1", "chi2", "chi3", "chi4", "chi5"]
# default 1.4. 1.8 is less than a single amino acid chain break and accounts for a freak bond 
# (pdb outliers and disordered AF regions)
IC_Chain.MaxPeptideBond = 1.8 
        
        
def getStructureObject(pdb_file, chain='A'): # if you supply the structure from processPdb, the chain is always A
    '''
    Creates structure object specifically for angles extraction.
    Бывает, что atom_to_internal_coordinates что-то не нравится (resultDict['pass'] == False), но я с этим не разбиралась ещё.
    '''
    # parse pdb file as structure object
    structure = pdb_parser.get_structure('', pdb_file)
    structure = structure[0][chain] 
    
    # format structure for angles retrieval
    structure.atom_to_internal_coordinates(verbose=True)
    resultDict = structure_rebuild_test(structure)
    if resultDict['pass'] == False:
        print('atom_to_internal_coordinates went wrong :(')
    return structure # returns structure anyway


def AnglesFromStructure(structure):
    '''
    Computes angles from structure object and returns a dataframe with amino acid, residue number and all angles
    '''
    aa_lst, resnum_lst, angles_lst, bfactors_lst, coords_lst = [], [], [], [], []
    residues = [i for i in structure.get_residues()]

    # compute angles
    for residue in residues:
        R = ''.join([f'{residue.get_full_id()[3][i]}' for i in range(1,3) if residue.get_full_id()[3][i] != ' ']) # in case residue number contains letters in residue.get_full_id()[3][2]
        resnum_lst.append(R)
        aa_lst.append(residue.get_resname())
        # https://biopython.org/docs/dev/api/Bio.PDB.internal_coords.html
        if residue.internal_coord != None:
            angles_tuple = tuple([residue.internal_coord.get_angle(angle) for angle in angles_names])
            angles_lst.append(angles_tuple)
        else:
            angles_lst.append(tuple(len(angles_names)*[None])) 
            
        bfactors_lst.append(residue['CA'].bfactor)
        coords_lst.append(residue['CA'].coord)
            
    # make a dataframe
    angles_df = pd.DataFrame(np.around(np.array(angles_lst, dtype=np.float64),2))
    angles_df.columns = angles_names
    df = pd.DataFrame({'aa':aa_lst, 'residue_number':resnum_lst})
    df = pd.concat([df, angles_df], axis=1)
    df['bfactor'] = bfactors_lst
    coords = np.array(coords_lst)
    df[['x', 'y', 'z']] = coords
    return df
