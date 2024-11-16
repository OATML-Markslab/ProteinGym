import os
import argparse
import pandas as pd
import biotite.structure.io as bsio
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--pdb_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="plddt.csv")
    parser.add_argument("--type", type=str, choices=["residue", "protein"], default="protein")
    args = parser.parse_args()
    
    if args.pdb_dir is not None:
        if args.type == "protein":
            out_info = {"pdb": [], "plddt": []}
        elif args.type == "residue":
            out_info = {"pdb": []}
        pdbs = sorted(os.listdir(args.pdb_dir))
        for pdb in tqdm(pdbs):
            pdb_file = os.path.join(args.pdb_dir, pdb)
            struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
            if args.type == "protein":
                plddt = struct.b_factor.mean()
                out_info["pdb"].append(pdb)
                out_info["plddt"].append(plddt)
            elif args.type == "residue":
                out_info["pdb"].append(pdb)
                res_list = []
                for res, plddt in zip(struct.res_id, struct.b_factor):
                    if res not in res_list:
                        if not out_info.get(res):
                            out_info[res] = []
                        res_list.append(res)
                        out_info[res].append(plddt)
        pd.DataFrame(out_info).to_csv(args.out_file, index=False)
    else:
        struct = bsio.load_structure(args.pdb_file, extra_fields=["b_factor"])
        if args.type == "protein":
            plddt = struct.b_factor.mean()
        elif args.type == "residue":
            res_list = []
            plddt = []
            for res, b_factor in zip(struct.res_id, struct.b_factor):
                if res not in res_list:
                    res_list.append(res)
                    plddt.append(b_factor)
        print(plddt)