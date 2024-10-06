from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

PROTEINGYM_DIR = Path(__file__).resolve().parents[5]
KERMUT_DIR = Path(__file__).resolve().parents[2]


def main():
    """Extracts the coordinates of the alpha carbons for all proteins in the ProteinGym benchmark"""
    df_ref = pd.read_csv(PROTEINGYM_DIR / "reference_files/DMS_substitutions.csv")
    pdb_dir = PROTEINGYM_DIR / "structure_files/DMS_subs"
    distance_dir = KERMUT_DIR / "data/structures/coords"
    distance_dir.mkdir(exist_ok=True, parents=True)

    for row in tqdm(df_ref.itertuples()):
        uniprot_id = row.UniProt_ID
        dms_id = row.DMS_id

        out_path = distance_dir / f"{dms_id}.npy"
        pdb_path = pdb_dir / f"{uniprot_id}.pdb"

        if out_path.exists():
            print(f"Skipping {dms_id} as it already exists.")
            continue

        try:
            # Fails for BRCA2_HUMAN
            structure = PDBParser().get_structure(uniprot_id, pdb_path)

        except FileNotFoundError:
            print(f"Could not find PDB file for {uniprot_id}")
            continue

        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            coords.append(atom.get_coord())

        coords = np.array(coords)

        # Special cases
        if dms_id == "A0A140D2T1_ZIKV_Sourisseau_2019":
            min_pos = 291 - 1
            max_pos = min_pos + len(coords)
            seq_len = len(row.target_seq)
            full_coords = np.full((seq_len, 3), np.nan)
            full_coords[min_pos:max_pos] = coords
            coords = full_coords
        elif dms_id == "POLG_HCVJF_Qi_2014":
            min_pos = 1982 - 1
            max_pos = min_pos + len(coords)
            seq_len = len(row.target_seq)
            full_coords = np.full((seq_len, 3), np.nan)
            full_coords[min_pos:max_pos] = coords
            coords = full_coords

        np.save(out_path, coords)


if __name__ == "__main__":
    main()
