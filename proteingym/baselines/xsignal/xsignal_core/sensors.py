"""Legacy-compatible corrected PureGraph GE sensors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .a2m import AA_ORDER, load_proteingym_focus_sequences, load_weights, weighted_conservation
from .structure import StructureContext

AA_VOLUMES = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}
AA_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
AA_STABILITY = {
    "A": 0.0, "C": 0.3, "D": -0.9, "E": -0.8, "F": 0.5,
    "G": -1.2, "H": -0.3, "I": 0.7, "K": -1.1, "L": 0.5,
    "M": 0.4, "N": -0.7, "P": -0.3, "Q": -0.6, "R": -0.8,
    "S": -0.5, "T": -0.2, "V": 0.4, "W": 1.0, "Y": 0.2,
}
AA_CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1,
    "I": 0, "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0,
    "R": 1, "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}
GAP_STATE = len(AA_ORDER)


def parse_single_mutation(value: str) -> tuple[str, int, str] | None:
    """Parse only a single substitution; leave multi-mutants for a later track."""

    match = MUTATION_RE.fullmatch(str(value).strip())
    if not match:
        return None
    wt, position, mutant = match.groups()
    return wt, int(position), mutant


def parse_mutation_tokens(value: str) -> list[tuple[str, int, str]]:
    """Parse all substitution tokens separated by ':'."""

    parsed: list[tuple[str, int, str]] = []
    for token in str(value).strip().split(":"):
        match = MUTATION_RE.fullmatch(token.strip())
        if not match:
            continue
        wt, position, mutant = match.groups()
        parsed.append((wt, int(position), mutant))
    return parsed


def required_position_pairs(values: list[str] | tuple[str, ...] | np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for value in values:
        positions = sorted({position for _wt, position, _mutant in parse_mutation_tokens(str(value))})
        for left_index, left in enumerate(positions):
            for right in positions[left_index + 1 :]:
                pairs.add((left, right))
    return pairs


def _mutation_delta_vector(parsed: tuple[str, int, str]) -> dict[str, float]:
    wt, _position, mutant = parsed
    return {
        "hydro": (AA_HYDRO.get(mutant, 0.0) - AA_HYDRO.get(wt, 0.0)) / 9.0,
        "volume": (AA_VOLUMES.get(mutant, 130.0) - AA_VOLUMES.get(wt, 130.0)) / max(AA_VOLUMES.values()),
        "charge": float(AA_CHARGE.get(mutant, 0) - AA_CHARGE.get(wt, 0)),
        "to_pro": float(mutant == "P") - float(wt == "P"),
        "to_gly": float(mutant == "G") - float(wt == "G"),
        "to_cys": float(mutant == "C") - float(wt == "C"),
    }


def _structure_context_single(parsed: tuple[str, int, str], structure: StructureContext, pdb_start: int) -> float:
    wt, position, mutant = parsed
    index = structure.index_for_alignment_position(position, pdb_start)
    if index is None:
        return 0.0
    burial = float(structure.burial[index])
    confidence = float(structure.plddt[index])
    if burial < 0.15:
        return 0.0
    importance = burial * (0.5 + 0.5 * confidence)
    delta_volume = abs(AA_VOLUMES.get(mutant, 130.0) - AA_VOLUMES.get(wt, 130.0)) / max(AA_VOLUMES.values())
    wt_hydro = AA_HYDRO.get(wt, 0.0)
    mutant_hydro = AA_HYDRO.get(mutant, 0.0)
    hydro_penalty = 0.0
    if wt_hydro > 1.0 and mutant_hydro < -1.0:
        hydro_penalty = 1.0
    elif wt_hydro < -1.0 and mutant_hydro > 1.0:
        hydro_penalty = 0.8
    elif abs(wt_hydro - mutant_hydro) > 3.0:
        hydro_penalty = 0.5
    aa_penalty = 0.0
    if mutant == "P" and wt != "P":
        aa_penalty = 0.5
    if wt == "C" and mutant != "C":
        aa_penalty = 1.0
    if mutant == "G" and wt != "G":
        aa_penalty = 0.3
    disruption = importance * max(delta_volume * 0.7 + hydro_penalty * 0.3, aa_penalty * 0.15)
    return float(-disruption)


def structure_context_score(value: str, structure: StructureContext, pdb_start: int) -> float:
    parsed = parse_single_mutation(value)
    if parsed is None:
        return 0.0
    return _structure_context_single(parsed, structure, pdb_start)


def _tm_stability_single(parsed: tuple[str, int, str], structure: StructureContext, pdb_start: int) -> float:
    wt, position, mutant = parsed
    index = structure.index_for_alignment_position(position, pdb_start)
    if index is None:
        return 0.0
    burial = float(structure.burial[index])
    confidence = float(structure.plddt[index])
    delta_tm = AA_STABILITY.get(mutant, 0.0) - AA_STABILITY.get(wt, 0.0)
    delta_charge = AA_CHARGE.get(mutant, 0) - AA_CHARGE.get(wt, 0)
    return float(burial * delta_tm * confidence + (1.0 - burial) * abs(delta_charge) * 0.15 * confidence)


def tm_stability_score(value: str, structure: StructureContext, pdb_start: int) -> float:
    parsed = parse_single_mutation(value)
    if parsed is None:
        return 0.0
    return _tm_stability_single(parsed, structure, pdb_start)


def _msa_structure_joint_single(
    parsed: tuple[str, int, str],
    conservation: dict[int, float],
    structure: StructureContext,
    pdb_start: int,
) -> float:
    wt, position, mutant = parsed
    index = structure.index_for_alignment_position(position, pdb_start)
    if index is None:
        return 0.0
    cons = float(conservation.get(position, 0.0))
    burial = float(structure.burial[index])
    confidence = float(structure.plddt[index])
    delta_volume = abs(AA_VOLUMES.get(mutant, 130.0) - AA_VOLUMES.get(wt, 130.0)) / max(AA_VOLUMES.values())
    wt_hydro = AA_HYDRO.get(wt, 0.0)
    mutant_hydro = AA_HYDRO.get(mutant, 0.0)
    bio_penalty = 0.0
    if wt_hydro > 1.0 and mutant_hydro < -1.0:
        bio_penalty = 1.0
    elif wt_hydro < -1.0 and mutant_hydro > 1.0:
        bio_penalty = 0.8
    elif abs(wt_hydro - mutant_hydro) > 3.0:
        bio_penalty = 0.5
    if mutant == "P" and wt != "P":
        bio_penalty = max(bio_penalty, 0.5)
    if wt == "C" and mutant != "C":
        bio_penalty = max(bio_penalty, 1.0)
    core_score = burial * delta_volume * (0.5 + 0.5 * cons)
    functional_score = (1.0 - burial) * cons * bio_penalty
    return float(-(core_score * 0.65 + functional_score * 0.35) * confidence)


def msa_structure_joint_score(
    value: str,
    conservation: dict[int, float],
    structure: StructureContext,
    pdb_start: int,
) -> float:
    parsed = parse_single_mutation(value)
    if parsed is None:
        return 0.0
    return _msa_structure_joint_single(parsed, conservation, structure, pdb_start)


def _mean_over_valid(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def structure_context_composable_score(value: str, structure: StructureContext, pdb_start: int) -> float:
    return _mean_over_valid(
        [
            _structure_context_single(parsed, structure, pdb_start)
            for parsed in parse_mutation_tokens(value)
        ]
    )


def tm_stability_composable_score(value: str, structure: StructureContext, pdb_start: int) -> float:
    return _mean_over_valid(
        [
            _tm_stability_single(parsed, structure, pdb_start)
            for parsed in parse_mutation_tokens(value)
        ]
    )


def msa_structure_joint_composable_score(
    value: str,
    conservation: dict[int, float],
    structure: StructureContext,
    pdb_start: int,
) -> float:
    return _mean_over_valid(
        [
            _msa_structure_joint_single(parsed, conservation, structure, pdb_start)
            for parsed in parse_mutation_tokens(value)
        ]
    )


def contact_pair_ge_score(
    value: str,
    structure: StructureContext,
    pdb_start: int,
    *,
    contact_radius: float = 8.0,
) -> float:
    parsed_tokens = parse_mutation_tokens(value)
    if len(parsed_tokens) < 2:
        return 0.0

    indexed: list[tuple[int, dict[str, float]]] = []
    for parsed in parsed_tokens:
        index = structure.index_for_alignment_position(parsed[1], pdb_start)
        if index is None:
            continue
        indexed.append((index, _mutation_delta_vector(parsed)))
    if len(indexed) < 2:
        return 0.0

    pair_scores: list[float] = []
    for left_offset, (left_index, left_vec) in enumerate(indexed):
        for right_index, right_vec in indexed[left_offset + 1 :]:
            distance = structure.pair_distance(left_index, right_index)
            if not np.isfinite(distance) or distance <= 0.0 or distance > contact_radius:
                continue
            local_burial = 0.5 * (float(structure.burial[left_index]) + float(structure.burial[right_index]))
            confidence = np.sqrt(float(structure.plddt[left_index]) * float(structure.plddt[right_index]))
            contact_weight = np.exp(-((distance / 6.5) ** 2)) * confidence * (0.35 + 0.65 * local_burial)

            same_charge = max(0.0, left_vec["charge"] * right_vec["charge"])
            opposite_charge = max(0.0, -left_vec["charge"] * right_vec["charge"])
            dual_expand = max(0.0, left_vec["volume"]) * max(0.0, right_vec["volume"])
            dual_shrink = max(0.0, -left_vec["volume"]) * max(0.0, -right_vec["volume"])
            volume_comp = max(0.0, -left_vec["volume"] * right_vec["volume"])
            hydro_mismatch = abs(left_vec["hydro"] - right_vec["hydro"])
            special = (
                abs(left_vec["to_pro"]) + abs(right_vec["to_pro"])
                + 0.6 * (abs(left_vec["to_gly"]) + abs(right_vec["to_gly"]))
                + 0.4 * (abs(left_vec["to_cys"]) + abs(right_vec["to_cys"]))
            )

            penalty = (
                0.55 * same_charge
                + 0.35 * dual_expand
                + 0.22 * dual_shrink
                + 0.18 * hydro_mismatch
                + 0.14 * special
            )
            compensation = 0.28 * opposite_charge + 0.14 * volume_comp
            pair_scores.append(float(contact_weight * (compensation - penalty)))
    return _mean_over_valid(pair_scores)


def build_msa_pair_log_tables_with_metadata(
    msa_path: str | Path,
    weight_path: str | Path,
    start: int,
    end: int,
    mutant_values: list[str] | tuple[str, ...] | np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, int]]:
    sequences, metadata = load_proteingym_focus_sequences(
        msa_path,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_indeterminate=True,
    )
    expected_length = end - start + 1
    if metadata["msa_focus_length"] != expected_length:
        raise ValueError(
            f"MSA focus length mismatch: {msa_path}: focus={metadata['msa_focus_length']}, expected={expected_length}"
        )
    weights = load_weights(weight_path, len(sequences))
    pairs = required_position_pairs(mutant_values)
    if not pairs:
        meta = dict(metadata)
        meta["msa_weight_sequences"] = int(len(weights))
        meta["msa_weight_sum_floor"] = int(np.floor(weights.sum()))
        meta["msa_pair_requested"] = 0
        meta["msa_pair_scored"] = 0
        return {}, meta

    selected_positions = sorted({position for pair in pairs for position in pair if start <= position <= end})
    position_to_col = {position: index for index, position in enumerate(selected_positions)}
    state_matrix = np.full((len(sequences), len(selected_positions)), GAP_STATE, dtype=np.int16)
    for row_index, sequence in enumerate(sequences):
        for col_index, position in enumerate(selected_positions):
            aa = sequence[position - start]
            state_matrix[row_index, col_index] = AA_TO_INDEX.get(aa, GAP_STATE)

    n_states = len(AA_ORDER) + 1
    total = float(weights.sum()) + pseudocount * (n_states * n_states)
    tables: dict[tuple[int, int], np.ndarray] = {}
    for left, right in sorted(pairs):
        left_col = position_to_col.get(left)
        right_col = position_to_col.get(right)
        if left_col is None or right_col is None:
            continue
        keys = state_matrix[:, left_col].astype(np.int32) * n_states + state_matrix[:, right_col].astype(np.int32)
        counts = np.bincount(keys, weights=weights, minlength=n_states * n_states).astype(np.float64)
        probs = (counts.reshape(n_states, n_states) + pseudocount) / total
        tables[(left, right)] = np.log(np.clip(probs[: len(AA_ORDER), : len(AA_ORDER)], 1e-12, 1.0))

    meta = dict(metadata)
    meta["msa_weight_sequences"] = int(len(weights))
    meta["msa_weight_sum_floor"] = int(np.floor(weights.sum()))
    meta["msa_pair_requested"] = int(len(pairs))
    meta["msa_pair_scored"] = int(len(tables))
    return tables, meta


def contact_evo_pair_ge_score(
    value: str,
    pair_log_tables: dict[tuple[int, int], np.ndarray],
    structure: StructureContext,
    pdb_start: int,
    *,
    contact_radius: float = 8.0,
) -> float:
    parsed_tokens = parse_mutation_tokens(value)
    if len(parsed_tokens) < 2:
        return 0.0

    indexed: list[tuple[int, int, tuple[str, int, str]]] = []
    for parsed in parsed_tokens:
        index = structure.index_for_alignment_position(parsed[1], pdb_start)
        if index is None:
            continue
        indexed.append((parsed[1], index, parsed))
    if len(indexed) < 2:
        return 0.0

    pair_scores: list[float] = []
    for left_offset, (left_pos, left_index, left_parsed) in enumerate(indexed):
        for right_pos, right_index, right_parsed in indexed[left_offset + 1 :]:
            distance = structure.pair_distance(left_index, right_index)
            if not np.isfinite(distance) or distance <= 0.0 or distance > contact_radius:
                continue

            swapped = False
            pair = (left_pos, right_pos)
            if left_pos > right_pos:
                pair = (right_pos, left_pos)
                swapped = True
            table = pair_log_tables.get(pair)
            if table is None:
                continue
            parsed_left = right_parsed if swapped else left_parsed
            parsed_right = left_parsed if swapped else right_parsed
            wt_left, _pos_left, mt_left = parsed_left
            wt_right, _pos_right, mt_right = parsed_right
            wt_left_index = AA_TO_INDEX.get(wt_left)
            wt_right_index = AA_TO_INDEX.get(wt_right)
            mt_left_index = AA_TO_INDEX.get(mt_left)
            mt_right_index = AA_TO_INDEX.get(mt_right)
            if None in {wt_left_index, wt_right_index, mt_left_index, mt_right_index}:
                continue

            log_wt_wt = float(table[wt_left_index, wt_right_index])
            log_mt_mt = float(table[mt_left_index, mt_right_index])
            log_mt_wt = float(table[mt_left_index, wt_right_index])
            log_wt_mt = float(table[wt_left_index, mt_right_index])
            epistasis = log_mt_mt + log_wt_wt - log_mt_wt - log_wt_mt

            local_burial = 0.5 * (float(structure.burial[left_index]) + float(structure.burial[right_index]))
            confidence = np.sqrt(float(structure.plddt[left_index]) * float(structure.plddt[right_index]))
            contact_weight = np.exp(-((distance / 6.5) ** 2)) * confidence * (0.35 + 0.65 * local_burial)
            pair_scores.append(float(contact_weight * epistasis))
    return _mean_over_valid(pair_scores)


def corrected_msa_conservation_with_metadata(
    msa_path: str | Path,
    weight_path: str | Path,
    start: int,
    end: int,
) -> tuple[dict[int, float], dict[str, int]]:
    sequences, metadata = load_proteingym_focus_sequences(
        msa_path,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_indeterminate=True,
    )
    expected_length = end - start + 1
    if metadata["msa_focus_length"] != expected_length:
        raise ValueError(
            f"MSA focus length mismatch: {msa_path}: focus={metadata['msa_focus_length']}, expected={expected_length}"
        )
    weights = load_weights(weight_path, len(sequences))
    metadata = dict(metadata)
    metadata["msa_weight_sequences"] = int(len(weights))
    metadata["msa_weight_sum_floor"] = int(np.floor(weights.sum()))
    return weighted_conservation(sequences, weights, start, end), metadata


def corrected_msa_conservation(msa_path: str | Path, weight_path: str | Path, start: int, end: int) -> dict[int, float]:
    conservation, _ = corrected_msa_conservation_with_metadata(msa_path, weight_path, start, end)
    return conservation
