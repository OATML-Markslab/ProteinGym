import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from tqdm import tqdm
from structure.quantizer import PdbQuantizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing PDB files",)
    parser.add_argument("--pdb_file", type=str, default=None, help="PDB file",)
    parser.add_argument("--vocab_size", type=int, default=[2048], nargs='+', help="Vocabulary size",)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files",)
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory",)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.vocab_size:
        os.makedirs(os.path.join(args.out_dir, str(v)), exist_ok=True)
    
    if args.pdb_dir is not None:
        pdb_files = os.listdir(args.pdb_dir)
        for pdb_file in tqdm(pdb_files):
            pdb_name = pdb_file[:-4]
            for v in args.vocab_size:
                out_file = os.path.join(args.out_dir, str(v), pdb_name+'.fasta')
                if os.path.exists(out_file) and not args.overwrite:
                    continue
                processor = PdbQuantizer(structure_vocab_size=v)
                result = processor(os.path.join(args.pdb_dir, pdb_file), return_residue_seq=False)
                result = [str(i) for i in result]
                with open(out_file, "w") as f:
                    f.write(f'>{pdb_name}\n')
                    f.write(','.join(result))
                    
    elif args.pdb_file is not None:
        pdb_name = args.pdb_file.split(".")[0]
        for v in args.vocab_size:
            processor = PdbQuantizer(structure_vocab_size=v)
            result = processor(args.pdb_file, return_residue_seq=False)
            result = [str(i) for i in result]
            with open(os.path.join(args.out_dir, str(v), pdb_name+'.fasta'), "w") as f:
                f.write(f'>{pdb_name}\n')
                f.write(','.join(result))