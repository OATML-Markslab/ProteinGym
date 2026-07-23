"""ProteinGym-compatible A2M and MSA preprocessing utilities."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ORDER)
GAP_BYTE = ord("-")
LOWER_TRANSLATION = np.arange(256, dtype=np.uint8)
for _aa in AA_ORDER:
    LOWER_TRANSLATION[ord(_aa)] = ord(_aa.lower())
VALID_PREPROCESSED_BYTES = np.zeros(256, dtype=bool)
for _aa in AA_ORDER:
    VALID_PREPROCESSED_BYTES[ord(_aa)] = True
    VALID_PREPROCESSED_BYTES[ord(_aa.lower())] = True
VALID_PREPROCESSED_BYTES[GAP_BYTE] = True


def normalize_match_states(sequence: str) -> str:
    """Remove lowercase insertion states while preserving match-state gaps."""

    return "".join(char for char in sequence.strip() if not char.islower() and char != ".")


def parse_a2m_alignment(path: str | Path) -> OrderedDict[str, str]:
    """Read a raw A2M file and preserve record order and casing."""

    records: OrderedDict[str, str] = OrderedDict()
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if name is not None:
                    records[name] = "".join(chunks)
                name = line[1:].strip()
                chunks = []
            elif line.strip():
                chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks)
    if not records:
        raise ValueError(f"A2M file contains no sequences: {path}")
    return records


def parse_a2m_records(path: str | Path) -> list[str]:
    """Read an A2M file and return generic normalized match-state sequences."""

    records = [normalize_match_states(sequence) for sequence in parse_a2m_alignment(path).values()]
    lengths = {len(record) for record in records}
    if len(lengths) != 1:
        raise ValueError(f"A2M match-state lengths disagree: {path}: {sorted(lengths)[:8]}")
    return records


def preprocess_records_like_proteingym(
    records: OrderedDict[str, str],
    threshold_sequence_frac_gaps: float = 0.5,
    threshold_focus_cols_frac_gaps: float = 1.0,
    remove_indeterminate: bool = True,
) -> OrderedDict[str, str]:
    """Mirror ProteinGym/EVE preprocessing for the official MSA weights.

    This follows the ProteinGym baseline logic:
    1. Replace '.' with '-'.
    2. Uppercase all residues.
    3. Remove columns that are gaps in the focus sequence.
    4. Remove fragment-like sequences with too many gaps.
    5. Mark non-focus columns as lowercase instead of dropping them.
    6. Optionally drop sequences with indeterminate symbols.
    """

    if not records:
        raise ValueError("A2M preprocessing requires at least one record")
    if not 0.0 <= threshold_sequence_frac_gaps <= 1.0:
        raise ValueError(f"invalid fragment threshold: {threshold_sequence_frac_gaps}")
    if not 0.0 <= threshold_focus_cols_frac_gaps <= 1.0:
        raise ValueError(f"invalid focus threshold: {threshold_focus_cols_frac_gaps}")

    names = list(records)
    cleaned = [sequence.replace(".", "-").upper() for sequence in records.values()]
    focus_len = len(cleaned[0])
    matrix = np.full((len(cleaned), focus_len), GAP_BYTE, dtype=np.uint8)
    for index, sequence in enumerate(cleaned):
        encoded = np.frombuffer(sequence.encode("ascii", errors="ignore"), dtype=np.uint8)[:focus_len]
        matrix[index, : len(encoded)] = encoded

    matrix = matrix[:, matrix[0] != GAP_BYTE]
    if matrix.shape[1] == 0:
        raise ValueError("focus-sequence gap filtering removed every column")

    gaps = matrix == GAP_BYTE
    seq_keep = gaps.mean(axis=1) <= threshold_sequence_frac_gaps
    if not np.any(seq_keep):
        raise ValueError("fragment filtering removed every sequence")
    focus_keep = gaps[seq_keep].mean(axis=0) <= threshold_focus_cols_frac_gaps

    output: OrderedDict[str, str] = OrderedDict()
    kept_names = [name for name, keep in zip(names, seq_keep, strict=True) if keep]
    kept_matrix = matrix[seq_keep]
    if np.all(focus_keep):
        valid_rows = (
            VALID_PREPROCESSED_BYTES[kept_matrix].all(axis=1)
            if remove_indeterminate
            else np.ones(len(kept_names), dtype=bool)
        )
        for name, row, valid in zip(kept_names, kept_matrix, valid_rows, strict=True):
            if valid:
                output[name] = row.tobytes().decode("ascii")
    else:
        for name, row in zip(kept_names, kept_matrix, strict=True):
            current = row.copy()
            current[~focus_keep] = LOWER_TRANSLATION[current[~focus_keep]]
            if remove_indeterminate and not VALID_PREPROCESSED_BYTES[current].all():
                continue
            output[name] = current.tobytes().decode("ascii")
    if not output:
        raise ValueError("indeterminate filtering removed every sequence")
    return output


def extract_focus_sequences(records: OrderedDict[str, str]) -> list[str]:
    """Extract the uppercase focus columns exactly as ProteinGym baselines do."""

    if not records:
        raise ValueError("focus extraction requires at least one record")
    focus_sequence = next(iter(records.values()))
    if focus_sequence and focus_sequence == focus_sequence.upper() and "-" not in focus_sequence:
        output = list(records.values())
        lengths = {len(sequence) for sequence in output}
        if len(lengths) != 1:
            raise ValueError(f"ProteinGym focus-sequence lengths disagree: {sorted(lengths)[:8]}")
        return output
    focus_columns = [index for index, aa in enumerate(focus_sequence) if aa != "-" and aa == aa.upper()]
    if not focus_columns:
        raise ValueError("no focus columns remain after preprocessing")
    output = ["".join(sequence[index].upper() for index in focus_columns) for sequence in records.values()]
    lengths = {len(sequence) for sequence in output}
    if len(lengths) != 1:
        raise ValueError(f"ProteinGym focus-sequence lengths disagree: {sorted(lengths)[:8]}")
    return output


def load_proteingym_focus_sequences(
    path: str | Path,
    *,
    threshold_sequence_frac_gaps: float = 0.5,
    threshold_focus_cols_frac_gaps: float = 1.0,
    remove_indeterminate: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """Return ProteinGym-compatible focus sequences plus prediction-side counts."""

    raw_records = parse_a2m_alignment(path)
    processed = preprocess_records_like_proteingym(
        raw_records,
        threshold_sequence_frac_gaps=threshold_sequence_frac_gaps,
        threshold_focus_cols_frac_gaps=threshold_focus_cols_frac_gaps,
        remove_indeterminate=remove_indeterminate,
    )
    sequences = extract_focus_sequences(processed)
    return sequences, {
        "msa_raw_sequences": len(raw_records),
        "msa_processed_sequences": len(processed),
        "msa_focus_length": len(sequences[0]),
    }


def load_weights(path: str | Path, n_sequences: int) -> np.ndarray:
    """Load sequence weights and reject silent length mismatches."""

    weights = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if len(weights) != n_sequences:
        raise ValueError(
            f"MSA weight length mismatch: {path}: weights={len(weights)}, sequences={n_sequences}"
        )
    if not np.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError(f"MSA weights are invalid: {path}")
    return weights


def weighted_conservation(
    sequences: Iterable[str],
    weights: np.ndarray,
    start: int,
    end: int,
) -> dict[int, float]:
    """Compute weighted entropy conservation on normalized match columns.

    Positions are one-based in the reference sequence. Gaps and unknown
    states are excluded from each column denominator. The implementation is
    deterministic and contains no DMS-dependent operation.
    """

    records = list(sequences)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(records) != len(weights):
        raise ValueError(f"sequence/weight mismatch: {len(records)} != {len(weights)}")
    if start < 1 or end < start:
        raise ValueError(f"invalid MSA range: {start}-{end}")
    if end - start + 1 > len(records[0]):
        raise ValueError(f"MSA range exceeds aligned length: {start}-{end} > {len(records[0])}")

    index = {aa: i for i, aa in enumerate(AA_ORDER)}
    max_entropy = float(np.log(len(AA_ORDER)))
    output: dict[int, float] = {}
    for offset, position in enumerate(range(start, end + 1)):
        counts = np.zeros(len(AA_ORDER), dtype=np.float64)
        observed = 0.0
        for sequence, weight in zip(records, weights):
            aa = sequence[offset]
            if aa in index:
                counts[index[aa]] += weight
                observed += weight
        if observed <= 0:
            output[position] = 0.0
            continue
        probabilities = counts[counts > 0] / observed
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        output[position] = 1.0 - min(entropy / max_entropy, 1.0)
    return output
