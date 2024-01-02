import os, sys
# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
import argparse
from src.dataset.cath_dataset import CathDataset
from src.dataset.mutant_dataset import MutantDataset
from src.utils.dataset_utils import NormalizeProtein


def build_cath_dataset(args, split):
    dataset = CathDataset(
        root=args.cath_dataset,
        split=split,
        divide_num=1,
        divide_idx=0,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        set_length=None,
        p=args.noise_ratio,
        normalize_file=f'norm/cath_k{args.c_alpha_max_neighbors}_mean_attr.pt',
    )
    return dataset


def build_mutant_dataset(args):
    mm_dataset = MutantDataset(
        root=args.mutant_dataset_dir,
        name=args.mutant_name,
        raw_dir=args.mutant_dataset_dir+"/DATASET",
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        pre_transform=NormalizeProtein(
            filename=f'norm/cath_k{args.c_alpha_max_neighbors}_mean_attr.pt'
        ),
    )
    return mm_dataset

def prepare_train_val_dataset(args):
    # load protein dataset like CATHs40
    train_dataset = build_cath_dataset(args, "train")
    val_dataset = build_cath_dataset(args, "val")
    
    return train_dataset, val_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=30, help="number of nearest nodes to build graph")
    
    parser.add_argument("--data_type", type=str, choices=["cath", "mutant"], default="cath", help="type of dataset")
    # build cath train dataset
    parser.add_argument("--cath_dataset", type=str, default="data/cath40_k30", help="name of cath dataset")
    parser.add_argument("--noise_ratio", type=float, default=0.05)
    
    # build zero-shot mutant prediction dataset
    parser.add_argument("--mutant_dataset_dir",type=str,default="data/proteingym-benchmark",help="dir of mutation dataset")
    parser.add_argument("--mutant_name",type=str,default="proteingym_k30",help="name of mutation dataset")

    args = parser.parse_args()
    
    if args.data_type == "cath":
        cath_dataset = build_cath_dataset(args, "train")
    elif args.data_type == "mutant":
        mutant_dataset = build_mutant_dataset(args)