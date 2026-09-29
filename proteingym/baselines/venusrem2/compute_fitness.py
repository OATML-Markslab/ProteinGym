"""Score ProteinGym substitution assays with VenusREM2.

VenusREM2 is the vrh calibration head on the six official ProSST checkpoints
(K = 20, 128, 512, 1024, 2048, 4096). Each checkpoint is scored with
entropy-weighted MSA fusion, gated CCD on the raw logits, and RSA / pLDDT
above-mean decay. The reported column is the per-protein z-mean of those six
members.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from get_struc_seq import ensure_structure_tokens
from scoring import (
    apply_alignment_prior,
    build_calibration_terms,
    load_alignment_count_matrix,
    load_residue_plddt_from_pdb,
    load_residue_rsa_weights_from_pdb,
    parse_alpha_arg,
    parse_background_weight_arg,
    plddt_skip_reason,
    resolve_mix_weights,
    resolve_pdb_file,
    score_mutations_batch,
)

PROSST_ENSEMBLE_SIZES = (20, 128, 512, 1024, 2048, 4096)
PROSST_ENSEMBLE_IDS = tuple(f"AI4Protein/ProSST-{k}" for k in PROSST_ENSEMBLE_SIZES)
ENSEMBLE_COLUMN = "VenusREM2"

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        seq = str(record.seq).strip()
        if seq:
            return seq
    raise ValueError(f"No sequence found in {fasta}")


def read_names(fasta_dir):
    return sorted(path.stem for path in Path(fasta_dir).glob("*.fasta"))


def tokenize_structure_sequence(structure_sequence):
    shifted = [i + 3 for i in structure_sequence]
    shifted = [1, *shifted, 2]
    return torch.tensor([shifted], dtype=torch.long)


def infer_model_max_residue_len(model, tokenizer):
    config = getattr(model, "config", None)
    pos_embed_type = getattr(config, "position_embedding_type", None) if config else None
    candidates = []
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value > 2:
            overhead = 2
            if pos_embed_type == "absolute":
                pos_embed = getattr(getattr(model, "esm", model), "embeddings", None)
                pos_embed = getattr(pos_embed, "position_embeddings", None)
                pad_idx = getattr(pos_embed, "padding_idx", None)
                if isinstance(pad_idx, int) and pad_idx > 0:
                    overhead = pad_idx + 3
            candidates.append(value - overhead)
    if candidates:
        return min(candidates)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 2 < tok_max < 1_000_000:
        return tok_max - 2
    return None


def resolve_structure_fasta(struc_seq_dir, protein_name, model_name):
    vocab = model_name.split("-")[-1]
    nested = os.path.join(struc_seq_dir, vocab, f"{protein_name}.fasta")
    if os.path.exists(nested):
        return nested
    flat = os.path.join(struc_seq_dir, f"{protein_name}.fasta")
    if os.path.exists(flat):
        return flat
    raise FileNotFoundError(
        f"Missing ProSST structure tokens for {protein_name} (K={vocab}): {nested}"
    )


def resolve_alignment(aln_dir, protein_name):
    if not aln_dir:
        return None
    for suffix in (".a2m", ".a3m", ".fasta"):
        path = os.path.join(aln_dir, f"{protein_name}{suffix}")
        if os.path.exists(path):
            return path
    return None


def member_column(model_name):
    return f"{ENSEMBLE_COLUMN}__{model_name.split('/')[-1]}"


def zmean_columns(frame, columns):
    mats = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scale = float(np.nanstd(values))
        if scale > 0:
            mats.append((values - np.nanmean(values)) / scale)
        else:
            mats.append(np.zeros_like(values, dtype=float))
    return np.nanmean(np.vstack(mats), axis=0)


def _log(protein_name, message):
    print(f"[{protein_name}] {message}")


@torch.no_grad()
def forward_prosst_logits(
    model,
    tokenizer,
    sequence,
    structure_sequence,
    max_residue_len=None,
    long_seq_overlap=256,
):
    seq_len = len(sequence)
    if len(structure_sequence) != seq_len:
        raise ValueError(
            f"Structure token length {len(structure_sequence)} != sequence length {seq_len}"
        )

    def _window(window_seq, window_structure):
        tokenized = tokenizer([window_seq], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=tokenize_structure_sequence(window_structure).to(device),
            labels=input_ids,
        )
        return torch.log_softmax(outputs.logits[0, 1:-1, :], dim=-1)

    if max_residue_len is None or seq_len <= max_residue_len:
        return _window(sequence, structure_sequence)

    overlap = max(0, int(long_seq_overlap))
    if overlap >= max_residue_len:
        overlap = max(0, max_residue_len // 2)
    step = max(1, max_residue_len - overlap)
    print(
        f">>> Long sequence L={seq_len} > {max_residue_len}; "
        f"sliding window={max_residue_len}, overlap={overlap}"
    )
    aggregated = None
    counts = None
    start = 0
    while start < seq_len:
        end = min(seq_len, start + max_residue_len)
        window_logits = _window(sequence[start:end], structure_sequence[start:end])
        window_len = min(window_logits.size(0), end - start)
        window_logits = window_logits[:window_len]
        if aggregated is None:
            aggregated = torch.zeros(
                seq_len, window_logits.size(-1), device=window_logits.device, dtype=window_logits.dtype
            )
            counts = torch.zeros(seq_len, 1, device=window_logits.device, dtype=window_logits.dtype)
        aggregated[start : start + window_len] += window_logits
        counts[start : start + window_len] += 1.0
        if end >= seq_len:
            break
        start += step
    return aggregated / counts.clamp_min(1.0)


def score_from_logits(
    logits,
    tokenizer,
    sequence,
    mutant_df,
    protein_name,
    aa_seq_aln_file=None,
    pdb_file=None,
    pdb_dir=None,
    alpha="entropy",
    background_weight="one_minus_alpha",
    scoring_mode="calibrated_margin",
    calibrate_on_raw=True,
    use_rsa_decay=True,
    rsa_decay_mode="above_mean",
    use_plddt_decay=True,
    plddt_decay_mode="above_mean",
    task_type="default",
    disable_adaptive_ccd=False,
    aln_count_cache_dir=None,
):
    alpha_spec = parse_alpha_arg(alpha)
    background_spec = parse_background_weight_arg(background_weight)
    backbone_logits = logits
    vocab = tokenizer.get_vocab()
    count_matrix = None
    aln_start = aln_end = 0
    if aa_seq_aln_file and os.path.exists(aa_seq_aln_file):
        try:
            count_matrix, aln_start, aln_end = load_alignment_count_matrix(
                aa_seq_aln_file, tokenizer, cache_dir=aln_count_cache_dir, protein_name=protein_name
            )
            count_matrix = count_matrix.to(logits.device)
        except Exception as exc:
            _log(protein_name, f"MSA count matrix failed ({exc}); mix will skip entropy-α")
            count_matrix = None
    if alpha_spec == "entropy" and count_matrix is None:
        _log(
            protein_name,
            "No usable MSA; entropy-α cannot run, using α=0 (CCD / RSA / pLDDT still apply)",
        )
    try:
        mix_alpha, mix_bg, mix_features = resolve_mix_weights(
            raw_logits=backbone_logits,
            count_matrix=count_matrix,
            aln_start=aln_start,
            aln_end=aln_end,
            sequence=sequence,
            vocab=vocab,
            alpha_spec=alpha_spec if count_matrix is not None else (
                0.0 if alpha_spec == "entropy" else alpha_spec
            ),
            background_spec=background_spec,
        )
        if mix_features.get("n"):
            _log(
                protein_name,
                "vrh mix α={:.4f} β={:.4f} ρ={:.4f} H̄={:.4f} ρπ={:.4f}".format(
                    mix_alpha,
                    mix_bg,
                    float(mix_features.get("rho", float("nan"))),
                    float(mix_features.get("hbar", float("nan"))),
                    float(mix_features.get("rho_pi", float("nan"))),
                ),
            )
    except Exception as exc:
        if alpha_spec == "entropy":
            _log(protein_name, f"entropy-α failed ({exc}); falling back to α=0.8, β=0.2")
            mix_alpha, mix_bg = 0.8, 0.2
        else:
            mix_alpha = float(alpha_spec)
            mix_bg = 1.0 - mix_alpha if background_spec == "one_minus_alpha" else float(background_spec)

    if mix_alpha != 0:
        logits = apply_alignment_prior(
            logits=logits,
            tokenizer=tokenizer,
            alpha=mix_alpha,
            aa_seq_aln_file=aa_seq_aln_file,
            count_matrix_cache_dir=aln_count_cache_dir,
            protein_name=protein_name,
            show_progress=False,
        )
    else:
        _log(protein_name, "No alignment matrix used")

    raw_for_calib = backbone_logits if calibrate_on_raw else None
    rsa_weights = None
    if use_rsa_decay:
        rsa_weights = load_residue_rsa_weights_from_pdb(
            seq_len=len(sequence), protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir
        )
        if rsa_weights is None:
            _log(protein_name, "RSA requested but computation failed; ignoring RSA")
    plddt_weights = None
    if use_plddt_decay:
        resolved_pdb = resolve_pdb_file(protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir)
        skip = plddt_skip_reason(resolved_pdb)
        if skip:
            _log(protein_name, skip)
        else:
            plddt_weights = load_residue_plddt_from_pdb(
                seq_len=len(sequence), protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir
            )
            if plddt_weights is None:
                _log(protein_name, "pLDDT requested but loading failed; ignoring pLDDT")

    calibration_terms = build_calibration_terms(
        logits=logits,
        sequence=sequence,
        vocab=vocab,
        scoring_mode=scoring_mode,
        raw_logits=raw_for_calib,
        rsa_weights=rsa_weights,
        plddt_weights=plddt_weights,
        disable_adaptive_ccd=disable_adaptive_ccd,
    )
    if scoring_mode == "calibrated_margin":
        _log(
            protein_name,
            "CCD adaptive={} bg_scale={:.4f} bg_consistency={:.4f}".format(
                not disable_adaptive_ccd,
                float(calibration_terms.get("bg_scale", 1.0)),
                float(calibration_terms.get("bg_consistency", 0.0)),
            ),
        )
    return score_mutations_batch(
        mutants=mutant_df["mutant"].tolist(),
        logits=logits,
        sequence=sequence,
        vocab=vocab,
        scoring_mode=scoring_mode,
        background_weight=float(mix_bg),
        calibration_terms=calibration_terms,
        use_rsa_decay=use_rsa_decay,
        rsa_decay_mode=rsa_decay_mode,
        use_plddt_decay=use_plddt_decay,
        plddt_decay_mode=plddt_decay_mode,
        task_type=task_type,
    )


def _first_existing(base_dir, names):
    for name in names:
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            return path
    return os.path.join(base_dir, names[0])


def _spearman(frame, column):
    if "DMS_score" not in frame.columns or column not in frame.columns:
        return float("nan")
    corr = spearmanr(frame["DMS_score"], frame[column]).correlation
    return float(corr) if corr == corr else float("nan")


def main():
    parser = ArgumentParser(description="VenusREM2 zero-shot scoring on ProteinGym substitutions")
    parser.add_argument("--model_name", nargs="+", default=list(PROSST_ENSEMBLE_IDS))
    parser.add_argument("--base_dir", default=None, help="Folder with sequences, structure tokens, MSA, mutants, pdbs")
    parser.add_argument("--aa_seq_dir", default=None)
    parser.add_argument("--struc_seq_dir", default=None)
    parser.add_argument("--aa_seq_aln_dir", default=None)
    parser.add_argument("--mutant_dir", default=None)
    parser.add_argument("--pdb_dir", default=None)
    parser.add_argument("--output_scores_folder", required=True)
    parser.add_argument("--alpha", default="entropy")
    parser.add_argument("--background_weight", default="one_minus_alpha")
    parser.add_argument("--scoring_mode", default="calibrated_margin", choices=["calibrated_margin", "log_odds"])
    parser.add_argument("--no_calibrate_on_raw", action="store_true")
    parser.add_argument("--no_rsa_decay", action="store_true")
    parser.add_argument("--no_plddt_decay", action="store_true")
    parser.add_argument("--rsa_decay_mode", default="above_mean")
    parser.add_argument("--plddt_decay_mode", default="above_mean")
    parser.add_argument("--task_type", default="default", choices=["default", "surface"])
    parser.add_argument("--disable_adaptive_ccd", action="store_true")
    parser.add_argument("--long_seq_overlap", type=int, default=256)
    parser.add_argument("--max_residue_len", type=int, default=None)
    parser.add_argument("--aln_count_cache_dir", default=None)
    args = parser.parse_args()

    if args.base_dir:
        args.aa_seq_dir = args.aa_seq_dir or _first_existing(args.base_dir, ["aa_seq", "residue_sequence"])
        if args.struc_seq_dir is None:
            for name in ("struc_seq", "structure_sequence"):
                candidate = os.path.join(args.base_dir, name)
                if os.path.isdir(candidate):
                    args.struc_seq_dir = candidate
                    break
            if args.struc_seq_dir is None:
                args.struc_seq_dir = os.path.join(args.base_dir, "struc_seq")
        args.mutant_dir = args.mutant_dir or _first_existing(args.base_dir, ["substitutions"])
        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = _first_existing(args.base_dir, ["aa_seq_aln_a2m", "msa"])
        if args.pdb_dir is None:
            for name in ("pdb", "pdbs"):
                candidate = os.path.join(args.base_dir, name)
                if os.path.isdir(candidate):
                    args.pdb_dir = candidate
                    break
    for label, path in (
        ("aa_seq_dir", args.aa_seq_dir),
        ("mutant_dir", args.mutant_dir),
    ):
        if not path or not os.path.isdir(path):
            raise SystemExit(f"Missing {label}: {path}")
    if not args.struc_seq_dir:
        raise SystemExit("Missing struc_seq_dir")
    os.makedirs(args.struc_seq_dir, exist_ok=True)

    os.makedirs(args.output_scores_folder, exist_ok=True)
    protein_names = read_names(args.aa_seq_dir)
    if not protein_names:
        raise SystemExit(f"No FASTA files in {args.aa_seq_dir}")
    print(f">>> VenusREM2 on {len(protein_names)} proteins, {len(args.model_name)} ProSST checkpoints")
    print(f">>> device={device} alpha={args.alpha} mode={args.scoring_mode}")
    vocab_sizes = []
    for model_name in args.model_name:
        vocab = model_name.split("-")[-1]
        if vocab.isdigit():
            vocab_sizes.append(int(vocab))
    if args.pdb_dir:
        ensure_structure_tokens(args.pdb_dir, args.struc_seq_dir, protein_names, vocab_sizes)

    summary_rows = {name: {} for name in protein_names}
    for model_name in args.model_name:
        column = member_column(model_name)
        print(f">>> Loading {model_name}")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        max_len = args.max_residue_len or infer_model_max_residue_len(model, tokenizer)
        print(f">>> max residue length: {max_len}")

        for protein_name in tqdm(protein_names, desc=column):
            out_path = os.path.join(args.output_scores_folder, f"{protein_name}.csv")
            mutant_file = out_path if os.path.exists(out_path) else os.path.join(
                args.mutant_dir, f"{protein_name}.csv"
            )
            mutant_df = pd.read_csv(mutant_file)
            if column not in mutant_df.columns:
                sequence = read_seq(os.path.join(args.aa_seq_dir, f"{protein_name}.fasta"))
                structure_sequence = [
                    int(token) for token in read_seq(
                        resolve_structure_fasta(args.struc_seq_dir, protein_name, model_name)
                    ).split(",")
                ]
                pdb_file = None
                if args.pdb_dir:
                    candidate = os.path.join(args.pdb_dir, f"{protein_name}.pdb")
                    if os.path.exists(candidate):
                        pdb_file = candidate
                logits = forward_prosst_logits(
                    model,
                    tokenizer,
                    sequence,
                    structure_sequence,
                    max_residue_len=max_len,
                    long_seq_overlap=args.long_seq_overlap,
                )
                mutant_df[column] = score_from_logits(
                    logits=logits,
                    tokenizer=tokenizer,
                    sequence=sequence,
                    mutant_df=mutant_df,
                    protein_name=protein_name,
                    aa_seq_aln_file=resolve_alignment(args.aa_seq_aln_dir, protein_name),
                    pdb_file=pdb_file,
                    pdb_dir=args.pdb_dir,
                    alpha=args.alpha,
                    background_weight=args.background_weight,
                    scoring_mode=args.scoring_mode,
                    calibrate_on_raw=not args.no_calibrate_on_raw,
                    use_rsa_decay=not args.no_rsa_decay,
                    rsa_decay_mode=args.rsa_decay_mode,
                    use_plddt_decay=not args.no_plddt_decay,
                    plddt_decay_mode=args.plddt_decay_mode,
                    task_type=args.task_type,
                    disable_adaptive_ccd=args.disable_adaptive_ccd,
                    aln_count_cache_dir=args.aln_count_cache_dir,
                )
                mutant_df.to_csv(out_path, index=False)
            summary_rows[protein_name][column] = _spearman(mutant_df, column)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    member_cols = [member_column(name) for name in args.model_name]
    if len(member_cols) >= 2:
        for protein_name in protein_names:
            out_path = os.path.join(args.output_scores_folder, f"{protein_name}.csv")
            frame = pd.read_csv(out_path)
            present = [col for col in member_cols if col in frame.columns]
            if len(present) >= 2:
                frame[ENSEMBLE_COLUMN] = zmean_columns(frame, present)
                frame.to_csv(out_path, index=False)
            summary_rows[protein_name][ENSEMBLE_COLUMN] = _spearman(frame, ENSEMBLE_COLUMN)

    summary = pd.DataFrame(
        [{"protein": name, **scores} for name, scores in summary_rows.items()]
    )
    summary_path = os.path.join(args.output_scores_folder, "summary_spearman.csv")
    summary.to_csv(summary_path, index=False)
    if ENSEMBLE_COLUMN in summary.columns and summary[ENSEMBLE_COLUMN].notna().any():
        print(f">>> {ENSEMBLE_COLUMN} mean Spearman: {summary[ENSEMBLE_COLUMN].mean():.4f}")
    print(f">>> wrote {args.output_scores_folder}")


if __name__ == "__main__":
    main()
