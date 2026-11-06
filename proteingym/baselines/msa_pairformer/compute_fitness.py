"""ProteinGym scoring adapter for the MSA Pairformer paper results.

Adapted from Figure5B_ProteinGym_VEP_ProteinGym_MSAs.py. The masked-marginal
calculation, sampling with replacement, and sequence windows follow that script.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dms_mapping", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dms_index", type=int, help="Zero-based assay index")
    selection.add_argument("--dms-id", help="ProteinGym DMS_id")
    parser.add_argument("--dms-input", type=Path, required=True, help="Official assay directory")
    parser.add_argument("--dms-output", type=Path, required=True, help="Prediction directory")
    parser.add_argument("--msa-path", type=Path, required=True, help="Combined Logan/ProteinGym MSAs")
    parser.add_argument("--msa-weights-folder", type=Path, required=True)
    parser.add_argument("--msa-mapping", type=Path,
                        help="Optional CSV: DMS_id, MSA_filename, weight_file_name")
    parser.add_argument("--proteingym-repo", type=Path, default=Path(__file__).resolve().parents[3])
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--model-dir", type=Path, help="Local Hugging Face snapshot directory")
    checkpoint.add_argument("--model-revision", help="Pinned Hugging Face revision; not 'main'")
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--msa-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0, help="Python MSA sampling seed")
    parser.add_argument("--numpy-seed", type=int, default=42)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-variants", type=int,
                        help="Score a subset for reproduction checks; never use for benchmark submission")
    parser.add_argument("--validate-inputs", action="store_true", help="Check inputs without loading a model")
    args = parser.parse_args()
    if args.msa_samples < 1 or args.num_cpus < 1:
        parser.error("--msa-samples and --num-cpus must be positive")
    if args.max_variants is not None and args.max_variants < 1:
        parser.error("--max-variants must be positive")
    if args.model_revision in {"main", "master"}:
        parser.error("Use an immutable checkpoint revision")
    return args


def read_query(path):
    with path.open() as handle:
        if not handle.readline().startswith(">"):
            raise ValueError(f"Missing FASTA query header: {path}")
        parts = []
        for line in handle:
            if line.startswith(">"):
                break
            parts.append(line.strip())
    query = "".join(parts)
    if not query or re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", query):
        raise ValueError(f"Expected an ungapped uppercase query: {path}")
    return query


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd

    reference = pd.read_csv(args.dms_mapping)
    if args.dms_index is not None:
        if not 0 <= args.dms_index < len(reference):
            raise ValueError("--dms_index is outside the reference table")
        row = reference.iloc[args.dms_index]
    else:
        matches = reference[reference.DMS_id == args.dms_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one reference row for {args.dms_id}")
        row = matches.iloc[0]
    assay_id = row.DMS_id
    msa_filename = f"{assay_id}.a3m"
    weight_filename = row.weight_file_name
    if args.msa_mapping is not None:
        mapping = pd.read_csv(args.msa_mapping)
        matches = mapping[mapping.DMS_id == assay_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one MSA mapping row for {assay_id}")
        msa_filename = matches.iloc[0].MSA_filename
        weight_filename = matches.iloc[0].weight_file_name
    msa_path = args.msa_path / msa_filename
    weight_path = args.msa_weights_folder / weight_filename
    for path in [msa_path, weight_path, args.dms_input / row.DMS_filename]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.model_dir is not None:
        for name in ["model_cuex.bin", "confind_contact.bin", "contact.bin"]:
            if not (args.model_dir / name).is_file():
                raise FileNotFoundError(args.model_dir / name)
    # Never silently recompute weights: their order must match the original MSA.
    weights = np.load(weight_path, allow_pickle=False)
    if weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"Invalid sequence weights: {weight_path}")
    query = read_query(msa_path)
    target = row.target_seq.upper()
    offset = target.find(query)
    if offset < 0 or target.find(query, offset + 1) >= 0:
        raise ValueError(f"MSA query must have one exact placement in target_seq: {assay_id}")
    variants = pd.read_csv(args.dms_input / row.DMS_filename)
    if variants.mutant.isna().any() or variants.mutant.duplicated().any():
        raise ValueError("Mutant keys must be present and unique")
    if args.max_variants is not None:
        variants = variants.head(args.max_variants).copy()
    elif len(variants) != int(row.DMS_total_number_mutants):
        raise ValueError("Assay does not contain the full expected variant set")
    mutations = []
    for mutant in variants.mutant:
        terms = mutant.split(":")
        if not all(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]", t) for t in terms):
            raise ValueError(f"Unsupported mutant: {mutant}")
        parsed = [(int(t[1:-1]), t[0], t[-1]) for t in terms]
        for position, wt, _ in parsed:
            if not offset < position <= offset + len(query) or target[position - 1] != wt:
                raise ValueError(f"Mutation does not match MSA query: {mutant}")
        mutations.append(parsed)
    inputs = {"DMS_id": assay_id, "variants": len(variants), "query_length": len(query),
              "target_length": len(target), "query_offset": offset, "weights": len(weights),
              "msa_path": str(msa_path), "weight_path": str(weight_path)}
    print(json.dumps(inputs, indent=2), flush=True)
    if args.validate_inputs:
        return
    suffix = ".subset.csv" if args.max_variants is not None else ".csv"
    output = args.dms_output / f"{assay_id}{suffix}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    import torch
    from torch.nn.functional import one_hot
    from MSA_Pairformer.model import MSAPairformer
    from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks
    sys.path.insert(0, str(args.proteingym_repo.resolve() / "proteingym"))
    from utils.msa_utils import MSA_processing

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; request a GPU or explicitly use --device cpu")
    np.random.seed(args.numpy_seed)
    # ProteinGym's class/function bodies are identical to the paper's helpers.
    # Cached weights avoid the unused dense one-hot representation of the full MSA.
    processed = MSA_processing(MSA_location=str(msa_path), weights_location=str(weight_path),
                               use_weights=True, num_cpus=args.num_cpus,
                               skip_one_hot_encodings=True)
    random.seed(args.seed)
    sampled = [(processed.focus_seq_name, processed.raw_seq_name_to_sequence[processed.focus_seq_name])]
    non_query_weights = np.array([w for key, w in processed.seq_name_to_weight.items()
                                  if key != processed.focus_seq_name])
    non_query_sequences = [(key, seq) for key, seq in processed.seq_name_to_sequence.items()
                          if key != processed.focus_seq_name]
    if non_query_sequences:
        if non_query_weights.sum() <= 0:
            raise ValueError("Non-query sampling weights sum to zero")
        non_query_weights = non_query_weights / non_query_weights.sum()
        sampled.extend(random.choices(non_query_sequences, weights=non_query_weights, k=args.msa_samples - 1))
    sequences = [(name, ("".join(seq) if isinstance(seq, list) else seq).upper()) for name, seq in sampled]
    array = np.array([list(seq) for _, seq in sequences])
    tokens = torch.from_numpy(np.vectorize(aa2tok_d.get)(array))
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokens.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = [t.to(device) for t in (mask, msa_mask, full_mask, pairwise_mask)]
    onehot = one_hot(tokens, num_classes=len(set(aa2tok_d.values()))).float().unsqueeze(0).to(device)

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer", revision=args.model_revision,
                         cache_dir=str(args.model_cache) if args.model_cache else None,
                         allow_patterns=["model_cuex.bin", "confind_contact.bin", "contact.bin"]))
    checkpoint_files = [model_dir / name for name in ["model_cuex.bin", "confind_contact.bin", "contact.bin"]]
    state = {}
    for path in checkpoint_files:
        state.update(torch.load(path, weights_only=True, map_location="cpu"))
    model = MSAPairformer()
    model.load_state_dict(state)
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    del model.contact_head
    del model.confind_contact_head
    if not args.no_compile:
        model = torch.compile(model, dynamic=True)
    positions = list(dict.fromkeys(p for terms in mutations for p, _, _ in terms))
    log_probs = torch.zeros(len(target), len(aa2tok_d) - 2)
    for number, position in enumerate(positions, 1):
        msa_position = position - offset - 1
        wt_index = onehot[0, 0, msa_position].argmax()
        onehot[0, 0, msa_position] = 0
        onehot[0, 0, msa_position, aa2tok_d["<mask>"]] = 1
        lower = max(0, msa_position - 512) if onehot.shape[2] > 1024 and onehot.shape[1] > 2048 else 0
        upper = min(onehot.shape[2], lower + 1024) if onehot.shape[2] > 1024 and onehot.shape[1] > 2048 else onehot.shape[2]
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            result = model(msa=onehot[:, :, lower:upper].to(torch.bfloat16),
                           mask=mask[:, lower:upper], msa_mask=msa_mask,
                           full_mask=full_mask[:, :, lower:upper], pairwise_mask=pairwise_mask[:, lower:upper, lower:upper],
                           complex_chain_break_indices=None, return_seq_weights=False,
                           return_pairwise_repr_layer_idx=None, return_msa_repr_layer_idx=None,
                           return_cb_contacts=False, return_confind_contacts=False, query_only=True,
                           store_pairwise_repr_cpu=False, store_msa_repr_cpu=False)
        log_probs[position - 1] = result["logits"][0, 0, msa_position - lower].log_softmax(dim=-1).float().cpu()
        onehot[0, 0, msa_position] = 0
        onehot[0, 0, msa_position, wt_index] = 1
        print(f"{assay_id}: masked site {number}/{len(positions)}", flush=True)
    scores = []
    for terms in mutations:
        pos = [p - 1 for p, _, _ in terms]
        alt = [aa2tok_d[aa] for _, _, aa in terms]
        wt = [aa2tok_d[aa] for _, aa, _ in terms]
        scores.append(torch.sum(log_probs[pos, alt] - log_probs[pos, wt]).item())
    if not np.isfinite(scores).all():
        raise ValueError("Model produced nonfinite scores")
    args.dms_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mutant": variants.mutant.to_numpy(), "mutation_score": scores}).to_csv(output, index=False)
    provenance = {**inputs, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "checkpoint_sha256": {p.name: sha256(p) for p in checkpoint_files},
                  "msa_sha256": sha256(msa_path), "weights_sha256": sha256(weight_path)}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
