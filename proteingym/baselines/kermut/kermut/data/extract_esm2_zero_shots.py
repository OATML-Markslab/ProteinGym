"""Adapted from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm

PROTEINGYM_DIR = Path(__file__).resolve().parents[5]
KERMUT_DIR = Path(__file__).resolve().parents[2]


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    mutations = row.split(":")
    score = 0
    for mutation in mutations:
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert (
            sequence[idx] == wt
        ), "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (
            token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        ).item()

    return score


def compute_zero_shot(DMS_id: str, model, alphabet, nogpu: bool, overwrite: bool):
    file_out = Path(
        "data", "zero_shot_fitness_predictions", "ESM2/650M", f"{DMS_id}.csv"
    )
    file_out = KERMUT_DIR / file_out
    if file_out.exists() and not overwrite:
        print(f"Predictions for {DMS_id} already exist. Skipping.")
        return
    else:
        print(f"--- {DMS_id} ---")

    # Load data
    df_ref = pd.read_csv(PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv")
    df_wt = df_ref.loc[df_ref["DMS_id"] == DMS_id]
    reference_seq = df_wt["target_seq"].iloc[0]

    file_in = Path("data", "substitutions_singles", f"{DMS_id}.csv")
    file_in = PROTEINGYM_DIR / file_in
    df = pd.read_csv(file_in)

    score_key = "esm2_t33_650M_UR50D"
    batch_converter = alphabet.get_batch_converter()
    data = [
        ("protein1", reference_seq),
    ]
    _, _, batch_tokens = batch_converter(data)

    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            if torch.cuda.is_available() and not nogpu:
                batch_tokens_masked = batch_tokens_masked.cuda()
            token_probs = torch.log_softmax(
                model(batch_tokens_masked)["logits"], dim=-1
            )
        all_token_probs.append(token_probs[:, i])  # vocab size
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    df[score_key] = df.apply(
        lambda row: label_row(
            row["mutations"],
            reference_seq,
            token_probs,
            alphabet,
            1,
        ),
        axis=1,
    )

    df.to_csv(file_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    parser.add_argument("--DMS_idx", type=str, required=True)
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    model_path = Path("models") / "esm2_t33_650M_UR50D.pt"
    model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU.")

    df_ref = pd.read_csv(PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv")
    DMS_idx = args.DMS_idx
    DMS_id = df_ref.loc[DMS_idx, "DMS_id"]

    compute_zero_shot(
        DMS_id=DMS_id,
        model=model,
        alphabet=alphabet,
        nogpu=args.nogpu,
        overwrite=args.overwrite,
    )
