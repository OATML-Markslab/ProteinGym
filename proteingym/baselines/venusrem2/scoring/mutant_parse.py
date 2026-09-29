"""Parse substitution strings used in mutant CSVs."""

from __future__ import annotations

import re
from typing import Mapping

_SUB = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")


class MutantParseError(ValueError):
    """A mutant string is malformed or does not match the wild-type sequence."""


def parse_substitution(sub: str, sequence: str, vocab: Mapping[str, int]) -> tuple[str, int, str]:
    """Return ``(wt, 0-based index, mt)`` for ``A42G`` / ``A42G:L10M`` pieces."""
    text = str(sub).strip()
    match = _SUB.fullmatch(text)
    if match is None:
        raise MutantParseError(
            f"Bad mutant {text!r}; expected WtPosMt like A42G (colon-separated for multi-site)"
        )
    wt, pos, mt = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
    idx = pos - 1
    if idx < 0 or idx >= len(sequence):
        raise MutantParseError(
            f"Position {pos} in {text!r} is outside sequence length {len(sequence)}"
        )
    seq_wt = sequence[idx].upper()
    if seq_wt != wt:
        raise MutantParseError(
            f"Wild-type mismatch at {pos}: sequence has {seq_wt}, mutant says {wt}"
        )
    if wt not in vocab:
        raise MutantParseError(f"Wild-type {wt} in {text!r} is not in the model vocab")
    if mt not in vocab:
        raise MutantParseError(f"Mutant {mt} in {text!r} is not in the model vocab")
    return wt, idx, mt
