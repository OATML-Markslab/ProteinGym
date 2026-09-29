"""VenusREM2 scoring head.

Copied from the vrh recipe in VenusREM-Harness and kept on the same formulas:
entropy-weighted MSA fusion, gated CCD, RSA / pLDDT above-mean decay.
"""

from .alignment_enhancer import apply_alignment_prior, load_alignment_count_matrix
from .entropy_alpha import parse_alpha_arg, parse_background_weight_arg, resolve_mix_weights
from .mutant_parse import MutantParseError, parse_substitution
from .scoring_heads import build_calibration_terms, score_mutations_batch
from .structure_weights import (
    load_residue_plddt_from_pdb,
    load_residue_rsa_weights_from_pdb,
    plddt_skip_reason,
    resolve_pdb_file,
)

__all__ = [
    "MutantParseError",
    "apply_alignment_prior",
    "build_calibration_terms",
    "load_alignment_count_matrix",
    "load_residue_plddt_from_pdb",
    "load_residue_rsa_weights_from_pdb",
    "parse_alpha_arg",
    "parse_background_weight_arg",
    "parse_substitution",
    "plddt_skip_reason",
    "resolve_mix_weights",
    "resolve_pdb_file",
    "score_mutations_batch",
]
