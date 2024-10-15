import os

import numpy as np
from deli import load_json, save_json, save, load
from tqdm.auto import tqdm
import torch

from Bio.PDB.Polypeptide import protein_letters_3to1

from mulan.pdb_utils import AnglesFromStructure, getStructureObject
from mulan.foldseek_utils import get_struc_seq


def preproc_df(data, rot_len, all_amino_acids):
    # drop unknown amino acids
    data = data[data.aa.isin(all_amino_acids + ['MSE'])]
    data['aa'] = [protein_letters_3to1[cur_aa] if cur_aa in all_amino_acids 
                  else 'X' for cur_aa in data['aa']]

    data['max_len'] = data.aa.map(rot_len)
    data = data[~np.isnan(data.max_len)]
    data['max_len'] = data['max_len'].astype(int)

    return data


class Tokenizer():
    def __init__(self, use_foldseek_sequences):
        self.all_amino_acids = ['ASN', 'TRP', 'GLY', 'ILE', 'ASP', 'VAL', 'LYS', 'PHE',
                                'ARG', 'HIS', 'MET', 'THR', 'SER', 'GLN', 'LEU', 'ALA', 'CYS',
                                'PRO', 'GLU', 'TYR']

        self.rot_len_full = {"N": 2, "W": 2, "G": 0, "I": 2, "D": 2, "V": 1, "K": 4, "F": 2, 
                             "R": 5, "H": 2, "M": 3, "T": 1, "S": 1, "Q": 3, "L": 2, "A": 0, 
                             "C": 1, "P": 2, "E": 3, "Y": 2, "X": 5}
        self.rot_len_full['X'] = 5
        self.one_letter_aas = list(self.rot_len_full.keys())
        self.nan_fill_value = 182
        self.pad_value = 4.
        self.use_foldseek_sequences = use_foldseek_sequences

    def tokenize_protein(self, data):

        splitted_protein = []
        data = preproc_df(data, self.rot_len_full, all_amino_acids=self.all_amino_acids)
        if len(data) == 0:
            return splitted_protein

        angle_cols = ['phi', 'psi'] + [f'chi{i}' for i in range(1, 6)]
        data.fillna(self.nan_fill_value, inplace=True)
        data[angle_cols] = data[angle_cols].apply(np.deg2rad)

        for _, row in data.iterrows():
            cols_to_save = ['aa', 'bfactor', 'x', 'y', 'z'] + \
                           angle_cols[:2 + row['max_len']]
                           
            splitted_protein.append(row[cols_to_save].values.tolist())
        
        return splitted_protein
    
    def get_angles_from_pdb(self, pdb_path):
        angles_df = AnglesFromStructure(
            getStructureObject(pdb_path, chain='A'))
        return angles_df
    
    def get_foldseek_seq(self, pdb_path, plddt_mask, foldseek_path):
        # Extract the "A" chain from the pdb file and encode it into a struc_seq
        # pLDDT is used to mask low-confidence regions if "plddt_mask" is True
        parsed_seqs = get_struc_seq(foldseek_path, pdb_path, ["A"], plddt_mask=plddt_mask)["A"]
        seq, foldseek_seq, combined_seq = parsed_seqs
        return combined_seq
        
    def preproc_structural_data(self, proteins, protein_names):
        result_protein_angles = []
        result_sequences = []
        result_protein_names = []
        for protein, name in tqdm(zip(proteins, protein_names), desc='Preproc structural data'):
            protein_seq = ''.join([protein_aa[0] for protein_aa in protein])
            if len(protein_seq) < 1:
                continue
            padded_protein = [protein_aa[1:5 + 2 + len(protein_aa) - 5] + 
                            [self.pad_value] * (12 - len(protein_aa))
                            for protein_aa in protein]

            padded_protein = np.array(padded_protein)
            result_protein_angles.append(padded_protein)
            result_sequences.append(protein_seq)
            result_protein_names.append(name)

        return result_sequences, result_protein_angles, result_protein_names
    
    def save_dataset(self, data_path, dataset_path, split, is_experimental_structure, 
                     extract_foldseek, protein_ids=None, foldseek_path='bin/foldseek'):
        if protein_ids is None:
            protein_ids = [fname for fname in os.listdir(data_path) 
                           if fname.endswith('.pdb') or fname.endswith('.cif')]

        os.makedirs(dataset_path, exist_ok=True)
        all_proteins = []
        protein_names = []

        combined_foldseek_sequences = []
        for fname in tqdm(protein_ids, desc='Tokenizing data'):
            preprocessed_pdb_path = os.path.join(data_path, fname)
            data = self.get_angles_from_pdb(preprocessed_pdb_path)

            # foldseek extraction
            if extract_foldseek:
                combined_foldseek_seq = self.get_foldseek_seq(preprocessed_pdb_path, 
                                                              plddt_mask=not is_experimental_structure,
                                                              foldseek_path=foldseek_path)
                combined_foldseek_sequences.append(combined_foldseek_seq)
                assert int(len(combined_foldseek_seq) / 2) == len(data)

            # set all plddts to 100 in case of the experimental structure
            if is_experimental_structure:
                data['bfactor'] = 100.
                
            # tokenize protein
            splitted_protein = self.tokenize_protein(data)
            all_proteins.append(splitted_protein)
            protein_names.append(fname[:-4])

        res_sequences, res_angles, res_names = self.preproc_structural_data(all_proteins, protein_names)

        seq_lengths = [len(seq) for seq in res_sequences]
        res_angles = np.concatenate(res_angles)
        assert sum(seq_lengths) == res_angles.shape[0]

        save_json(res_names, os.path.join(dataset_path, f'{split}_names.json'))
        save_json(res_sequences, os.path.join(dataset_path, f'{split}_sequences.json'))
        save(res_angles, os.path.join(dataset_path, f'{split}_angles.npy.gz'))
        if extract_foldseek:
            save(combined_foldseek_sequences, 
                 os.path.join(dataset_path, f'{split}_foldseek_masked_sequences.json'))

    def load_dataset(self, dataset_path, split):
        if not os.path.exists(os.path.join(dataset_path, f'{split}_names.json')):
            return None, None, None
        
        res_names = load_json(os.path.join(dataset_path, f'{split}_names.json'))
        if self.use_foldseek_sequences:
            print('LOAD FOLDSEEK in tokenizer')
            res_sequences = load_json(os.path.join(dataset_path, f'{split}_foldseek_masked_sequences.json'))
        else:
            res_sequences = load_json(os.path.join(dataset_path, f'{split}_sequences.json'))
        res_angles = torch.from_numpy(load(os.path.join(dataset_path, f'{split}_angles.npy.gz')))

        return res_sequences, res_angles, res_names
