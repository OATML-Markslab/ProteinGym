"""Build ProSST structure-token FASTA files from PDB structures.

Same quantizer VenusREM uses (`PdbQuantizer`). VenusREM2 scoring calls this
when `struc_seq/{K}/{assay}.fasta` is missing and a PDB is available.
"""

import os
import sys
from argparse import ArgumentParser

PROSST_ENSEMBLE_SIZES = (20, 128, 512, 1024, 2048, 4096)
_VENUSREM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "venusrem", "venusrem")
)


def _quantizer(vocab_size):
    import biotite.structure as structure

    # biotite>=1.2 renamed filter_backbone; VenusREM's quantizer still imports the old name.
    if not hasattr(structure, "filter_backbone") and hasattr(structure, "filter_peptide_backbone"):
        structure.filter_backbone = structure.filter_peptide_backbone
    if _VENUSREM_ROOT not in sys.path:
        sys.path.insert(0, _VENUSREM_ROOT)
    from structure.quantizer import PdbQuantizer

    return PdbQuantizer(structure_vocab_size=vocab_size, therads=1)


def token_fasta_path(out_dir, vocab_size, protein_name):
    return os.path.join(out_dir, str(vocab_size), f"{protein_name}.fasta")


def pdb_path_for(pdb_dir, protein_name):
    for suffix in (".pdb", ".PDB"):
        path = os.path.join(pdb_dir, f"{protein_name}{suffix}")
        if os.path.exists(path):
            return path
    return None


def write_token_fasta(path, protein_name, tokens):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f">{protein_name}\n")
        handle.write(",".join(str(int(token)) for token in tokens))
        handle.write("\n")


def ensure_structure_tokens(pdb_dir, out_dir, protein_names, vocab_sizes):
    """Write any missing `struc_seq/{K}/{name}.fasta`. Returns the number written."""
    pending = []
    for vocab_size in vocab_sizes:
        for protein_name in protein_names:
            fasta_path = token_fasta_path(out_dir, vocab_size, protein_name)
            if os.path.exists(fasta_path):
                continue
            pdb_file = pdb_path_for(pdb_dir, protein_name) if pdb_dir else None
            if pdb_file is None:
                raise FileNotFoundError(
                    f"Missing structure tokens {fasta_path} and no PDB for {protein_name} in {pdb_dir}"
                )
            pending.append((vocab_size, protein_name, pdb_file, fasta_path))
    if not pending:
        return 0

    written = 0
    for vocab_size in vocab_sizes:
        batch = [item for item in pending if item[0] == vocab_size]
        if not batch:
            continue
        print(f">>> Building ProSST structure tokens from PDB (K={vocab_size}, n={len(batch)})")
        processor = _quantizer(vocab_size)
        for _, protein_name, pdb_file, fasta_path in batch:
            tokens = processor(pdb_file, return_residue_seq=False)
            write_token_fasta(fasta_path, protein_name, tokens)
            written += 1
            print(f">>> wrote {fasta_path}")
    return written


def main():
    parser = ArgumentParser(description="PDB to ProSST structure-token FASTA for VenusREM2")
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vocab_size", nargs="+", type=int, default=list(PROSST_ENSEMBLE_SIZES))
    parser.add_argument("--protein_name", nargs="*", default=None, help="Assay stems; default: every PDB in --pdb_dir")
    args = parser.parse_args()
    if args.protein_name:
        names = list(args.protein_name)
    else:
        names = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(args.pdb_dir)
            if name.lower().endswith(".pdb")
        )
    if not names:
        raise SystemExit(f"No PDB files in {args.pdb_dir}")
    written = ensure_structure_tokens(args.pdb_dir, args.out_dir, names, args.vocab_size)
    print(f">>> structure tokens ready under {args.out_dir} ({written} new files)")


if __name__ == "__main__":
    main()
