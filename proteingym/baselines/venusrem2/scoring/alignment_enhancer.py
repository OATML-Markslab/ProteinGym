import hashlib
import os
import random

import numpy as np
import torch
from tqdm import tqdm


def read_multi_fasta(file_path):
    sequences = {}
    current_sequence = ""
    header = None
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_sequence and header is not None:
                    sequences[header] = (
                        "".join(c for c in current_sequence if not c.islower())
                        .upper().replace("-", "<pad>").replace(".", "<pad>")
                    )
                    current_sequence = ""
                header = line
            else:
                current_sequence += line
        if current_sequence and header is not None:
            sequences[header] = (
                "".join(c for c in current_sequence if not c.islower())
                .upper().replace("-", "<pad>").replace(".", "<pad>")
            )
    return sequences


# ---------------------------------------------------------------------------
# Fast count-matrix builder (bypasses HuggingFace tokenizer)
# ---------------------------------------------------------------------------

def _vocab_hash(tokenizer):
    """8-char hash identifying a tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    key = str(sorted(vocab.items()))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _build_lut(tokenizer):
    """Build an ASCII ord → token-id lookup table from *tokenizer*'s vocab."""
    vocab = tokenizer.get_vocab()
    pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    lut = np.full(128, pad_idx, dtype=np.int32)
    for token, idx in vocab.items():
        if len(token) == 1 and ord(token) < 128:
            lut[ord(token)] = idx
    lut[ord('-')] = pad_idx
    lut[ord('.')] = pad_idx
    return lut, pad_idx


def _fast_count_matrix(file_path, tokenizer, logger=None, protein_name=None):
    """Build a normalised log-softmax count matrix directly from an a2m/a3m file.

    Streams through the file in batches to keep memory bounded, and uses a
    simple ASCII lookup table instead of the HuggingFace tokenizer.

    Returns
    -------
    count_matrix : Tensor [L, vocab_size]  (log-softmax normalised)
    aln_start    : int   (0-based inclusive)
    aln_end      : int   (0-based exclusive)
    """
    vocab_size = tokenizer.vocab_size
    lut, pad_idx = _build_lut(tokenizer)
    BATCH = 50_000

    # -- first pass: dimensions + alignment span --
    aln_start, aln_end = 0, 0
    max_len = 0
    n_seqs = 0
    first_header = None

    with open(file_path, "r") as f:
        cur_len = 0
        seen_header = False
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if seen_header:
                    max_len = max(max_len, cur_len)
                    n_seqs += 1
                cur_len = 0
                seen_header = True
                if first_header is None:
                    first_header = line
            elif seen_header:
                cur_len += sum(1 for char in line if not char.islower())
        if seen_header:
            max_len = max(max_len, cur_len)
            n_seqs += 1

    if first_header:
        try:
            span = first_header.split("/")[-1].split("-")
            aln_start, aln_end = int(span[0]) - 1, int(span[1])
        except Exception:
            aln_start, aln_end = 0, max_len
    else:
        aln_start, aln_end = 0, max_len

    if logger:
        logger.debug(
            f"Fast alignment: {n_seqs} seqs × {max_len} pos, span=[{aln_start+1},{aln_end}]",
            protein=protein_name,
        )

    # -- second pass: stream sequences in batches, accumulate counts --
    counts = np.zeros((max_len, vocab_size), dtype=np.float64)
    batch_buf = np.full((BATCH, max_len), pad_idx, dtype=np.int32)
    bi = 0

    def _flush(buf, n):
        flat_pos = np.arange(max_len, dtype=np.int64)[np.newaxis, :].repeat(n, axis=0)
        flat_idx = flat_pos.ravel() * vocab_size + buf[:n].ravel().astype(np.int64)
        c = np.bincount(flat_idx, minlength=max_len * vocab_size)
        counts[:] += c.reshape(max_len, vocab_size)

    with open(file_path, "r") as f:
        cur_chars = bytearray()
        seen_header = False
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if seen_header:
                    arr = np.frombuffer(bytes(cur_chars), dtype=np.uint8).copy()
                    ids = lut[np.clip(arr, 0, 127)]
                    slen = len(ids)
                    batch_buf[bi, :slen] = ids
                    if slen < max_len:
                        batch_buf[bi, slen:] = pad_idx
                    bi += 1
                    if bi == BATCH:
                        _flush(batch_buf, bi)
                        batch_buf[:] = pad_idx
                        bi = 0
                    cur_chars = bytearray()
                seen_header = True
            elif seen_header:
                aligned = "".join(char for char in line if not char.islower())
                cur_chars.extend(aligned.encode("ascii", errors="replace"))
        if seen_header:
            arr = np.frombuffer(bytes(cur_chars), dtype=np.uint8).copy()
            ids = lut[np.clip(arr, 0, 127)]
            slen = len(ids)
            batch_buf[bi, :slen] = ids
            if slen < max_len:
                batch_buf[bi, slen:] = pad_idx
            bi += 1
        if bi > 0:
            _flush(batch_buf, bi)

    cm = torch.from_numpy(counts).float()
    row_sums = cm.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cm = cm / row_sums
    cm = torch.log_softmax(cm, dim=-1)
    return cm, aln_start, aln_end


# ---------------------------------------------------------------------------
# Count-matrix cache
# ---------------------------------------------------------------------------

def _cm_cache_path(cache_dir, file_path, tokenizer, suffix="aa"):
    """Deterministic cache path for a (alignment file, tokenizer) pair.

    Includes the parent directory name (e.g. msa_af2_a2m vs msa_evc_a2m_clean)
    so that different MSA sources for the same protein get distinct cache files.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parent = os.path.basename(os.path.dirname(file_path)) or "_"
    vh = _vocab_hash(tokenizer)
    return os.path.join(cache_dir, f"{parent}__{stem}__{vh}__{suffix}.pt")


def _get_or_build_count_matrix(
    file_path, tokenizer, cache_dir=None,
    is_structure=False, logger=None, protein_name=None,
):
    """Return (count_matrix, aln_start, aln_end), using cache when available."""
    suffix = "struc" if is_structure else "aa"

    if cache_dir is not None:
        cp = _cm_cache_path(cache_dir, file_path, tokenizer, suffix)
        if os.path.exists(cp):
            cached = torch.load(cp, map_location="cpu", weights_only=True)
            if logger:
                logger.debug(f"Count-matrix cache hit: {os.path.basename(cp)}", protein=protein_name)
            return cached["count_matrix"], cached["aln_start"], cached["aln_end"]

    cm, aln_start, aln_end = _fast_count_matrix(
        file_path, tokenizer, logger=logger, protein_name=protein_name,
    )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{cp}.tmp.{os.getpid()}"
        torch.save({"count_matrix": cm, "aln_start": aln_start, "aln_end": aln_end}, tmp_path)
        os.replace(tmp_path, cp)
        if logger:
            logger.debug(f"Count-matrix cache saved: {os.path.basename(cp)}", protein=protein_name)

    return cm, aln_start, aln_end


# ---------------------------------------------------------------------------
# Legacy tokenizer-based builders (kept for reference / fallback)
# ---------------------------------------------------------------------------

def count_matrix_from_residue_alignment(
    tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None
):
    alignment_seqs = list(alignment_dict.values())
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split("/")[-1].split("-")
    except Exception:
        aln_start, aln_end = 1, len(alignment_seqs[0])
    if verbose:
        msg1 = f"Alignment span: start={aln_start}, end={aln_end}"
        msg2 = f"Tokenizing residue alignments: n={len(alignment_seqs)}"
        if logger is not None:
            logger.debug(msg1, protein=protein_name)
            logger.debug(msg2, protein=protein_name)
        else:
            print(f">>> {msg1}")
            print(f">>> {msg2}")
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:, 1:-1]
    return alignment_ids, int(aln_start) - 1, int(aln_end)


def count_matrix_from_structure_alignment(
    tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None
):
    alignment_seqs = list(alignment_dict.values())
    if verbose:
        msg = f"Tokenizing structure alignments: n={len(alignment_seqs)}"
        if logger is not None:
            logger.debug(msg, protein=protein_name)
        else:
            print(f">>> {msg}")
    if len(alignment_seqs) == 0:
        return None
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:, 1:-1]
    return alignment_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_alignment_count_matrix(
    file_path,
    tokenizer,
    cache_dir=None,
    is_structure=False,
    logger=None,
    protein_name=None,
):
    """Public wrapper: (count_matrix, aln_start, aln_end)."""
    return _get_or_build_count_matrix(
        file_path,
        tokenizer,
        cache_dir=cache_dir,
        is_structure=is_structure,
        logger=logger,
        protein_name=protein_name,
    )


def apply_alignment_prior(
    logits,
    tokenizer,
    alpha,
    aa_seq_aln_file=None,
    struc_seq_aln_file=None,
    sample_ratio=1.0,
    sample_times=1,
    show_progress=True,
    quiet=False,
    logger=None,
    protein_name=None,
    count_matrix_cache_dir=None,
):
    if alpha == 0:
        return logits

    device = logits.device
    use_fast = (sample_ratio >= 1.0 and sample_times <= 1)

    if aa_seq_aln_file is not None and struc_seq_aln_file is None:
        if logger is not None:
            logger.info("Using residue sequence alignment matrix", protein=protein_name)

        if use_fast:
            count_matrix, aln_start, aln_end = _get_or_build_count_matrix(
                aa_seq_aln_file, tokenizer,
                cache_dir=count_matrix_cache_dir,
                is_structure=False, logger=logger, protein_name=protein_name,
            )
            count_matrix = count_matrix.to(device)
            aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * count_matrix
            logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)
        else:
            # Legacy path with sampling support
            alignment_dict = read_multi_fasta(aa_seq_aln_file)
            alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            for _ in range(sample_times):
                if sample_ratio < 1.0:
                    curr_size = int(len(alignment_matrix) * sample_ratio)
                    sample_indices = random.sample(range(len(alignment_matrix)), curr_size)
                    alignment_matrix_sample = alignment_matrix[sample_indices]
                else:
                    alignment_matrix_sample = alignment_matrix

                cm = torch.zeros(alignment_matrix_sample.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(alignment_matrix_sample.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    cm[i] = torch.bincount(
                        alignment_matrix_sample[:, i], minlength=tokenizer.vocab_size
                    )

                cm = (cm / cm.sum(dim=1, keepdim=True)).to(device)
                cm = torch.log_softmax(cm, dim=-1)
                aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * cm
                logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)

    if struc_seq_aln_file is not None and aa_seq_aln_file is None:
        if logger is not None:
            logger.info("Using structure sequence alignment matrix", protein=protein_name)

        if use_fast:
            count_matrix, _, _ = _get_or_build_count_matrix(
                struc_seq_aln_file, tokenizer,
                cache_dir=count_matrix_cache_dir,
                is_structure=True, logger=logger, protein_name=protein_name,
            )
            count_matrix = count_matrix.to(device)
            logits = (1 - alpha) * logits + alpha * count_matrix
        else:
            alignment_dict = read_multi_fasta(struc_seq_aln_file)
            alignment_matrix = count_matrix_from_structure_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            if alignment_matrix is not None:
                cm = torch.zeros(alignment_matrix.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(alignment_matrix.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    cm[i] = torch.bincount(
                        alignment_matrix[:, i], minlength=tokenizer.vocab_size
                    )
                cm = (cm / cm.sum(dim=1, keepdim=True)).to(device)
                cm = torch.log_softmax(cm, dim=-1)
                logits = (1 - alpha) * logits + alpha * cm

    if aa_seq_aln_file is not None and struc_seq_aln_file is not None:
        if logger is not None:
            logger.info("Using both residue and structure sequence alignment matrix", protein=protein_name)
        plm_logits = logits.clone()

        if use_fast:
            struc_cm, _, _ = _get_or_build_count_matrix(
                struc_seq_aln_file, tokenizer,
                cache_dir=count_matrix_cache_dir,
                is_structure=True, logger=logger, protein_name=protein_name,
            )
            struc_cm = struc_cm.to(device)
            logits = (1 - alpha) * plm_logits + alpha * struc_cm

            aa_cm, aln_start, aln_end = _get_or_build_count_matrix(
                aa_seq_aln_file, tokenizer,
                cache_dir=count_matrix_cache_dir,
                is_structure=False, logger=logger, protein_name=protein_name,
            )
            aa_cm = aa_cm.to(device)
            aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * aa_cm
            logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)
        else:
            alignment_dict = read_multi_fasta(struc_seq_aln_file)
            structure_alignment_matrix = count_matrix_from_structure_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            if structure_alignment_matrix is not None:
                cm = torch.zeros(structure_alignment_matrix.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(structure_alignment_matrix.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    cm[i] = torch.bincount(
                        structure_alignment_matrix[:, i], minlength=tokenizer.vocab_size
                    )
                cm = (cm / cm.sum(dim=1, keepdim=True)).to(device)
                cm = torch.log_softmax(cm, dim=-1)
                logits = (1 - alpha) * plm_logits + alpha * cm

            alignment_dict = read_multi_fasta(aa_seq_aln_file)
            residue_alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            cm = torch.zeros(residue_alignment_matrix.size(1), tokenizer.vocab_size)
            for i in tqdm(
                range(residue_alignment_matrix.size(1)),
                disable=(not show_progress) or quiet,
                leave=False,
            ):
                cm[i] = torch.bincount(
                    residue_alignment_matrix[:, i], minlength=tokenizer.vocab_size
                )
            cm = (cm / cm.sum(dim=1, keepdim=True)).to(device)
            cm = torch.log_softmax(cm, dim=-1)
            aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * cm
            logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)

    return logits
