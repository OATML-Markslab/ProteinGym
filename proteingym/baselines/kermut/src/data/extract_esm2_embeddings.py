"""Script to extract ESM-2 embeddings for ProteinGym DMS assays.
Adapted from https://github.com/facebookresearch/esm/blob/main/scripts/extract.py"""

import argparse
from pathlib import Path

import h5py
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, pretrained
from tqdm import tqdm


def extract_single_embeddings(
    model: torch.nn.Module,
    alphabet: Alphabet,
    dataset: str,
    overwrite: bool = False,
    toks_per_batch: int = 8192,
    nogpu: bool = False,
) -> None:
    output_path = Path("data/embeddings/substitutions_singles/ESM2", f"{dataset}.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if overwrite:
            print(f"Overwriting existing embeddings for {dataset}.")
        else:
            print(f"Embeddings for {dataset} already exists. Skipping.")
            return

    print(f"--- Extracting embeddings for {dataset} ---")
    # Load dataset sequences
    df = pd.read_csv(
        Path(
            "data/substitutions_singles",
            f"{dataset}.csv",
        )
    )
    mutants = df["mutant"].tolist()
    sequences = df["mutated_sequence"].tolist()
    batched_dataset = FastaBatchedDataset(
        sequence_strs=sequences, sequence_labels=mutants
    )

    batches = batched_dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        batched_dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length=1022),
        batch_sampler=batches,
    )

    repr_layers = [33]

    all_labels = []
    all_representations = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(1022, len(strs[i]))
                all_labels.append(label)
                all_representations.append(
                    representations[33][i, 1 : truncate_len + 1]
                    .mean(axis=0)
                    .clone()
                    .numpy()
                )

    assert mutants == all_labels
    embeddings_dict = {
        "embeddings": all_representations,
        "mutants": mutants,
    }

    # Store data as HDF5
    with h5py.File(output_path, "w") as h5f:
        for key, value in embeddings_dict.items():
            h5f.create_dataset(key, data=value)


def extract_multiple_embeddings(
    model: torch.nn.Module,
    alphabet: Alphabet,
    dataset: str,
    overwrite: bool = False,
    toks_per_batch: int = 8192,
    nogpu: bool = False,
) -> None:
    output_path = Path(
        "data", "embeddings", "substitutions_multiples", "ESM2", f"{dataset}.h5"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if overwrite:
            print(f"Overwriting existing embeddings for {dataset}.")
        else:
            print(f"Embeddings for {dataset} already exists. Skipping.")
            return

    print(f"--- Extracting embeddings for {dataset} ---")
    # Load dataset sequences
    df = pd.read_csv(
        Path(
            "data/substitutions_multiples",
            f"{dataset}.csv",
        )
    )
    mutants = df["mutant"].tolist()
    sequences = df["mutated_sequence"].tolist()
    batched_dataset = FastaBatchedDataset(
        sequence_strs=sequences, sequence_labels=mutants
    )

    batches = batched_dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        batched_dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length=1022),
        batch_sampler=batches,
    )

    repr_layers = [33]

    all_labels = []
    all_representations = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(1022, len(strs[i]))
                all_labels.append(label)
                all_representations.append(
                    representations[33][i, 1 : truncate_len + 1]
                    .mean(axis=0)
                    .clone()
                    .numpy()
                )

    assert mutants == all_labels
    embeddings_dict = {
        "embeddings": all_representations,
        "mutants": mutants,
    }

    # Store data as HDF5
    with h5py.File(output_path, "w") as h5f:
        for key, value in embeddings_dict.items():
            h5f.create_dataset(key, data=value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--toks_per_batch", type=int, default=16384, help="maximum batch size"
    )
    parser.add_argument("--which", type=str, default="singles")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    args = parser.parse_args()

    if args.dataset == "all":
        df_ref = pd.read_csv("data/DMS_substitutions.csv")
        if args.which == "multiples":
            df_ref = df_ref[df_ref["includes_multiple_mutants"]]
            df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 10000]
        datasets = df_ref["DMS_id"].tolist()
    else:
        datasets = [args.dataset]

    # Load model
    model_path = Path("models", "esm2_t33_650M_UR50D.pt")
    model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU.")

    for dataset in tqdm(datasets):
        if args.which == "singles":
            extract_single_embeddings(
                model,
                alphabet,
                dataset,
                args.overwrite,
                args.toks_per_batch,
                args.nogpu,
            )
        elif args.which == "multiples":
            extract_multiple_embeddings(
                model,
                alphabet,
                dataset,
                args.overwrite,
                args.toks_per_batch,
                args.nogpu,
            )
