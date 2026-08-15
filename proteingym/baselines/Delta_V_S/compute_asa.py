#!/usr/bin/env python3
"""
Compute per-residue solvent accessibility from AlphaFold structures
and store in a SQLite database for agent queries.

Uses Biopython's Shrake-Rupley algorithm for ASA computation.
Normalizes to RSA using Sander & Rost maximum accessibility values.

Usage: python3 compute_asa.py
"""

import csv, json, os, sys, sqlite3
import numpy as np

import argparse
_p = argparse.ArgumentParser(description="Compute per-residue ASA/RSA from AlphaFold structures")
_p.add_argument("--DMS_reference_file_path", required=True, help="DMS_substitutions.csv")
_p.add_argument("--pdb_cache", default="pdb_cache", help="Folder used by download_structures.py")
_p.add_argument("--output_db", default="protein_structures.db", help="Output SQLite DB path")
_a = _p.parse_args()

DATA_DIR = _a.pdb_cache
REF_CSV = _a.DMS_reference_file_path
PDB_CACHE = DATA_DIR
MAPPING_FILE = os.path.join(PDB_CACHE, "uniprot_mapping.json")
DB_PATH = _a.output_db

# Sander & Rost (1994) maximum ASA values for each amino acid
# Used to normalize absolute ASA to relative ASA (0-1)
MAX_ASA = {
    'ALA': 106.0, 'ARG': 248.0, 'ASN': 157.0, 'ASP': 163.0, 'CYS': 135.0,
    'GLN': 198.0, 'GLU': 194.0, 'GLY':  84.0, 'HIS': 184.0, 'ILE': 169.0,
    'LEU': 164.0, 'LYS': 205.0, 'MET': 188.0, 'PHE': 197.0, 'PRO': 136.0,
    'SER': 130.0, 'THR': 142.0, 'TRP': 227.0, 'TYR': 222.0, 'VAL': 142.0,
}

# Three-letter to one-letter mapping
AA3_TO_AA1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# RSA thresholds for burial classification
# Based on Sander & Rost conventions:
# RSA < 0.09 = buried (core)
# RSA 0.09-0.20 = intermediate (partly buried)
# RSA 0.20-0.40 = accessible
# RSA > 0.40 = exposed (surface)
def classify_rsa(rsa):
    if rsa < 0.09:
        return "core"
    elif rsa < 0.20:
        return "buried"
    elif rsa < 0.40:
        return "intermediate"
    else:
        return "surface"


def compute_asa_for_pdb(pdb_path):
    """Compute per-residue ASA using Biopython's Shrake-Rupley."""
    from Bio.PDB import PDBParser, ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Compute ASA
    sr = ShrakeRupley()
    sr.compute(structure, level="R")  # residue level

    results = []
    model = structure[0]  # first model only

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":  # skip hetero atoms
                continue
            resname = residue.get_resname()
            if resname not in AA3_TO_AA1:
                continue

            aa1 = AA3_TO_AA1[resname]
            resnum = residue.id[1]  # PDB residue number
            asa = residue.sasa

            # Normalize to RSA
            max_asa = MAX_ASA.get(resname, 150.0)
            rsa = asa / max_asa if max_asa > 0 else 0.0
            rsa = min(rsa, 1.0)  # cap at 1.0

            burial = classify_rsa(rsa)

            results.append({
                "resnum": resnum,
                "aa": aa1,
                "asa": round(asa, 2),
                "rsa": round(rsa, 4),
                "burial": burial,
            })

    return results


def main():
    # Load mapping
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    # Load protein metadata
    with open(REF_CSV) as f:
        proteins = list(csv.DictReader(f))

    # Remove old DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create tables
    c.execute("""
        CREATE TABLE asa (
            protein_id TEXT,
            uniprot_id TEXT,
            position INTEGER,
            wt_aa TEXT,
            asa REAL,
            rsa REAL,
            burial_class TEXT,
            PRIMARY KEY (protein_id, position)
        )
    """)

    c.execute("""
        CREATE TABLE protein_info (
            protein_id TEXT PRIMARY KEY,
            uniprot_id TEXT,
            uniprot_acc TEXT,
            pdb_file TEXT,
            pdb_range TEXT,
            has_structure INTEGER,
            seq_len INTEGER,
            source_organism TEXT,
            selection_type TEXT,
            coarse_selection_type TEXT
        )
    """)

    c.execute("CREATE INDEX idx_asa_protein ON asa(protein_id)")
    c.execute("CREATE INDEX idx_asa_burial ON asa(protein_id, burial_class)")

    computed = 0
    no_structure = 0

    for i, p in enumerate(proteins):
        dms_id = p["DMS_id"]
        uid = p["UniProt_ID"]
        acc = mapping.get(uid, "")
        pdb_name = p["pdb_file"]
        pdb_range = p["pdb_range"]
        seq_len = int(p["seq_len"])

        # Insert protein info
        c.execute("""INSERT OR REPLACE INTO protein_info VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            dms_id, uid, acc, pdb_name, pdb_range,
            1 if os.path.exists(os.path.join(PDB_CACHE, f"{acc}.pdb")) else 0,
            seq_len,
            p["source_organism"],
            p["selection_type"],
            p["coarse_selection_type"],
        ))

        pdb_path = os.path.join(PDB_CACHE, f"{acc}.pdb")
        if not os.path.exists(pdb_path):
            no_structure += 1
            continue

        try:
            residues = compute_asa_for_pdb(pdb_path)

            # Map PDB residue numbers to mutation positions
            # ProteinGym mutations use positions in the target_seq coordinate system
            # PDB range tells us the mapping: pdb_range "291-794" means PDB res 291 = pos 291 in target_seq
            # For most proteins, PDB residue numbers match UniProt/target_seq positions directly

            for res in residues:
                c.execute("""INSERT OR REPLACE INTO asa VALUES (?,?,?,?,?,?,?)""", (
                    dms_id,
                    uid,
                    res["resnum"],  # PDB residue number = target_seq position
                    res["aa"],
                    res["asa"],
                    res["rsa"],
                    res["burial"],
                ))

            computed += 1

        except Exception as e:
            print(f"  ERROR computing ASA for {dms_id}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(proteins)}] {computed} computed, {no_structure} no structure")
            conn.commit()

    conn.commit()

    # Summary
    c.execute("SELECT COUNT(*) FROM asa")
    total_residues = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT protein_id) FROM asa")
    proteins_with_asa = c.fetchone()[0]
    c.execute("SELECT burial_class, COUNT(*) FROM asa GROUP BY burial_class")
    burial_dist = c.fetchall()

    print(f"\nDone!")
    print(f"  Proteins with ASA data: {proteins_with_asa}")
    print(f"  Total residues: {total_residues}")
    print(f"  No structure: {no_structure}")
    print(f"  Burial distribution:")
    for cls, count in burial_dist:
        print(f"    {cls}: {count} ({100*count/total_residues:.1f}%)")

    conn.close()
    print(f"\nDatabase: {DB_PATH}")


if __name__ == "__main__":
    main()
