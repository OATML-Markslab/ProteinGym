from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_AND_TOKENIZER_BY_DEVICE_NAME: dict[str, tuple[Any, Any]] = {}
VOCABULARY_BY_TOKENIZER_IDENTIFIER: dict[int, dict[str, int]] = {}
SCORE_CACHE_BY_DEVICE_AND_SEQUENCE: dict[
    str,
    OrderedDict[tuple[str, tuple[int, ...]], torch.Tensor],
] = {}
SCORE_CACHE_CAPACITY = 32
MUTATION_TOKEN_PATTERN = re.compile(r"([A-Z\*])(\d+)([A-Z\*])")

AAS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AAS)}
EPS = torch.finfo(torch.float32).tiny
DEFAULT_SCORE_COLUMN_NAME = "ProSST_2048_PIT_tail_rank_score"


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_name", required=True, type=str)
    argument_parser.add_argument("--base_dir", type=str)
    argument_parser.add_argument("--residue_dir", type=str)
    argument_parser.add_argument("--structure_dir", type=str)
    argument_parser.add_argument("--mutant_dir", type=str)
    argument_parser.add_argument("--reference_file_path", required=True, type=str)
    argument_parser.add_argument("--output_scores_folder", required=True, type=str)
    argument_parser.add_argument("--score_column_name", default=DEFAULT_SCORE_COLUMN_NAME, type=str)
    argument_parser.add_argument("--device", default="", type=str)
    return argument_parser.parse_args()


def resolve_device_name(explicit_device_name: str) -> str:
    if explicit_device_name:
        return explicit_device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_benchmark_root_directory(base_directory_path: Path) -> Path:
    nested_benchmark_directory_path = base_directory_path / "proteingym_benchmark"
    if nested_benchmark_directory_path.exists():
        return nested_benchmark_directory_path
    return base_directory_path


def resolve_model_quantization_directory_name(model_name: str) -> str:
    model_name_suffix = model_name.split("/")[-1]
    if "-" not in model_name_suffix:
        return model_name_suffix
    return model_name_suffix.split("-")[-1]


def read_single_fasta_sequence(file_path: Path) -> str:
    sequence_part_list: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith(">"):
            continue
        sequence_part_list.append(stripped_line)
    if not sequence_part_list:
        raise ValueError(f"Missing sequence content in FASTA file: {file_path}")
    return "".join(sequence_part_list)


def read_structure_token_list(file_path: Path) -> list[int]:
    structure_text = read_single_fasta_sequence(file_path)
    return [int(token_text) for token_text in structure_text.split(",") if token_text]


def load_model_and_tokenizer(model_name: str, device_name: str) -> tuple[Any, Any]:
    cache_key = f"{model_name}::{device_name}"
    if cache_key not in MODEL_AND_TOKENIZER_BY_DEVICE_NAME:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        model.cls.predictions.decoder.weight = model.prosst.embeddings.word_embeddings.weight
        model = model.to(device_name).eval()
        MODEL_AND_TOKENIZER_BY_DEVICE_NAME[cache_key] = (model, tokenizer)
    return MODEL_AND_TOKENIZER_BY_DEVICE_NAME[cache_key]


def tokenize_structure_token_list(structure_token_list: list[int]) -> torch.Tensor:
    return torch.tensor(
        [[1] + [structure_token + 3 for structure_token in structure_token_list] + [2]],
        dtype=torch.long,
    )


def get_vocabulary_by_token(tokenizer: Any) -> dict[str, int]:
    tokenizer_identifier = id(tokenizer)
    if tokenizer_identifier not in VOCABULARY_BY_TOKENIZER_IDENTIFIER:
        VOCABULARY_BY_TOKENIZER_IDENTIFIER[tokenizer_identifier] = tokenizer.get_vocab()
    return VOCABULARY_BY_TOKENIZER_IDENTIFIER[tokenizer_identifier]


def get_canonical_probability_tensor(
    model: Any,
    tokenizer: Any,
    target_sequence: str,
    structure_token_list: list[int],
    device_name: str,
) -> torch.Tensor:
    tokenized_sequence = tokenizer([target_sequence], return_tensors="pt")
    input_identifier_tensor = tokenized_sequence["input_ids"].to(device_name)
    attention_mask_tensor = tokenized_sequence["attention_mask"].to(device_name)
    structure_identifier_tensor = tokenize_structure_token_list(structure_token_list).to(device_name)

    with torch.inference_mode():
        model_output = model(
            input_ids=input_identifier_tensor,
            attention_mask=attention_mask_tensor,
            ss_input_ids=structure_identifier_tensor,
            return_dict=True,
        )
        vocabulary_by_token = get_vocabulary_by_token(tokenizer)
        canonical_identifier_tensor = torch.tensor(
            [vocabulary_by_token[amino_acid] for amino_acid in AAS],
            dtype=torch.long,
            device=device_name,
        )
        canonical_logit_tensor = model_output.logits[:, 1:-1, :].float().index_select(
            dim=-1,
            index=canonical_identifier_tensor,
        )
        return torch.softmax(canonical_logit_tensor, dim=-1)[0]


def center(values: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return values - (probs * values).sum(dim=-1, keepdim=True)


def logit(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(EPS).clamp_max(1 - EPS)
    return values.log() - (-values).log1p()


def groups(sorted_values: torch.Tensor) -> torch.Tensor:
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    return starts.cumsum(dim=0) - 1


def protein_rank_residual(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1)
    order = torch.argsort(flat)
    group_ids = groups(flat[order])
    counts = torch.bincount(group_ids, minlength=int(group_ids[-1]) + 1)
    lower = counts.cumsum(0)
    upper = counts.flip(0).cumsum(0).flip(0)
    ties = (counts[group_ids].to(values.dtype) - 1) / 2
    residual = (lower[group_ids].to(values.dtype) - ties).log() - (
        upper[group_ids].to(values.dtype) - ties
    ).log()
    ranked = torch.empty_like(flat)
    ranked[order] = residual
    return ranked.reshape_as(values)


def protein_pit_residual(
    values: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:
    flat = values.reshape(-1)
    order = torch.argsort(flat)
    group_ids = groups(flat[order])
    size = int(group_ids[-1]) + 1
    weights = (probs / probs.shape[0]).reshape(-1)[order]
    masses = torch.zeros(size, dtype=values.dtype, device=values.device)
    masses.scatter_add_(0, group_ids, weights)
    counts = torch.bincount(group_ids, minlength=size)
    mass_cdf = masses.cumsum(0) - masses / 2
    rank_cdf = (counts.cumsum(0).to(values.dtype) - counts.to(values.dtype) / 2) / flat.numel()
    residual = logit(mass_cdf[group_ids]) - logit(rank_cdf[group_ids])
    ranked = torch.empty_like(flat)
    ranked[order] = residual
    return ranked.reshape_as(values)


def compute_cached_score_matrix(
    model: Any,
    tokenizer: Any,
    target_sequence: str,
    structure_token_list: list[int],
    device_name: str,
) -> torch.Tensor:
    cache_key = (target_sequence, tuple(structure_token_list))
    if device_name not in SCORE_CACHE_BY_DEVICE_AND_SEQUENCE:
        SCORE_CACHE_BY_DEVICE_AND_SEQUENCE[device_name] = OrderedDict()
    cache_by_sequence = SCORE_CACHE_BY_DEVICE_AND_SEQUENCE[device_name]
    if cache_key in cache_by_sequence:
        cache_by_sequence.move_to_end(cache_key)
        return cache_by_sequence[cache_key]

    probs = get_canonical_probability_tensor(
        model=model,
        tokenizer=tokenizer,
        target_sequence=target_sequence,
        structure_token_list=structure_token_list,
        device_name=device_name,
    )

    lower = (probs.unsqueeze(-2) <= probs.unsqueeze(-1)).to(probs.dtype)
    tail_mass = (lower * probs.unsqueeze(-2)).sum(dim=-1).clamp_min(EPS)
    tail_rank = lower.sum(dim=-1).to(probs.dtype) / probs.shape[-1]
    local = center(tail_mass.log() - tail_rank.log(), probs)
    global_pit = center(protein_pit_residual(local, probs), probs)
    global_rank = center(protein_rank_residual(local), probs)
    score_matrix = (local + global_pit + global_rank).cpu()

    cache_by_sequence[cache_key] = score_matrix
    if len(cache_by_sequence) > SCORE_CACHE_CAPACITY:
        cache_by_sequence.popitem(last=False)
    return score_matrix


@lru_cache(maxsize=262144)
def parse_mutation_string(mutation_string: str) -> tuple[tuple[str, str, int], ...]:
    normalized_mutation_string = mutation_string.strip().upper()
    if not normalized_mutation_string or normalized_mutation_string in {
        "WT",
        "W",
        "WILDTYPE",
        "WILD-TYPE",
        "NONE",
    }:
        return ()

    for token in [
        "MUT:",
        "MUT=",
        "VARIANT:",
        "VARIANT=",
        "SUBSTITUTIONS:",
        "SUBSTITUTION:",
        "MUTATION:",
        "MUTATIONS:",
        "MISSENSE:",
        "AA:",
        "P.",
        "C.",
        "G.",
        "R.",
        "HGVS:",
    ]:
        normalized_mutation_string = normalized_mutation_string.replace(token, "")
    for character in "()[]{}\"'":
        normalized_mutation_string = normalized_mutation_string.replace(character, "")
    normalized_mutation_string = normalized_mutation_string.replace("->", "").replace(">", "").replace("=", "")
    normalized_mutation_string = normalized_mutation_string.replace("TER", "*").replace("STOP", "*")
    for separator in [",", ";", "/", "|", "_", " ", "+", "."]:
        normalized_mutation_string = normalized_mutation_string.replace(separator, ":")

    mutation_part_list = [
        mutation_part.strip()
        for mutation_part in normalized_mutation_string.split(":")
        if mutation_part.strip()
    ]
    if not mutation_part_list:
        return ()

    parsed_mutation_list: list[tuple[str, str, int]] = []
    for mutation_part in mutation_part_list:
        mutation_part = "".join(mutation_part.split()).strip(":.")
        if len(mutation_part) >= 2 and mutation_part[1] == "." and mutation_part[0] in {
            "P",
            "C",
            "G",
            "M",
            "R",
        }:
            mutation_part = mutation_part[2:]
        if mutation_part in {
            "WT",
            "W",
            "WILDTYPE",
            "WILD-TYPE",
            "NONE",
            "P",
            "C",
            "G",
            "M",
            "R",
            "SYNONYMOUS",
        }:
            continue

        direct_match = MUTATION_TOKEN_PATTERN.fullmatch(mutation_part)
        if direct_match is not None:
            wild_type_amino_acid, position_text, mutant_amino_acid = direct_match.groups()
            position_value = int(position_text)
            if position_value <= 0:
                raise ValueError(f"Malformed mutation string: {mutation_string}")
            parsed_mutation_list.append(
                (wild_type_amino_acid, mutant_amino_acid, position_value - 1)
            )
            continue

        cleaned_mutation_part = re.sub(r"[^A-Z0-9\*]", "", mutation_part)
        embedded_match_list = list(MUTATION_TOKEN_PATTERN.finditer(cleaned_mutation_part))
        if embedded_match_list:
            leftover_text = MUTATION_TOKEN_PATTERN.sub("", cleaned_mutation_part)
            if leftover_text:
                raise ValueError(f"Malformed mutation string: {mutation_string}")
            matched_text = "".join(match.group(0) for match in embedded_match_list)
            if matched_text != cleaned_mutation_part:
                raise ValueError(f"Malformed mutation string: {mutation_string}")
            for embedded_match in embedded_match_list:
                wild_type_amino_acid, position_text, mutant_amino_acid = embedded_match.groups()
                position_value = int(position_text)
                if position_value <= 0:
                    raise ValueError(f"Malformed mutation string: {mutation_string}")
                parsed_mutation_list.append(
                    (wild_type_amino_acid, mutant_amino_acid, position_value - 1)
                )
            continue

        raise ValueError(f"Malformed mutation string: {mutation_string}")
    return tuple(parsed_mutation_list)


def predict_fitness(
    model_name: str,
    target_sequence: str,
    mutation_string_list: list[str],
    structure_token_list: list[int],
    device_name: str,
) -> list[float]:
    if not mutation_string_list:
        return []

    model, tokenizer = load_model_and_tokenizer(model_name=model_name, device_name=device_name)
    score_matrix = compute_cached_score_matrix(
        model=model,
        tokenizer=tokenizer,
        target_sequence=target_sequence,
        structure_token_list=structure_token_list,
        device_name=device_name,
    )
    target_sequence_length = len(target_sequence)

    fitness_score_list: list[float] = []
    for mutation_string in mutation_string_list:
        total_variant_score = 0.0
        mutation_count = 0
        is_variant_valid = True

        try:
            parsed_mutation_list = parse_mutation_string(mutation_string)
        except ValueError:
            fitness_score_list.append(0.0)
            continue

        for wild_type_amino_acid, mutant_amino_acid, position_index in parsed_mutation_list:
            if position_index < 0 or position_index >= target_sequence_length:
                is_variant_valid = False
                break
            if target_sequence[position_index] != wild_type_amino_acid:
                is_variant_valid = False
                break

            mutant_index = AA_TO_INDEX.get(mutant_amino_acid)
            if mutant_index is None:
                is_variant_valid = False
                break

            total_variant_score += float(score_matrix[position_index, mutant_index])
            mutation_count += 1

        if not is_variant_valid:
            fitness_score_list.append(0.0)
            continue
        fitness_score_list.append(total_variant_score if mutation_count > 0 else 0.0)

    return fitness_score_list


def build_assay_identifier_suffix(assay_identifier: str) -> str:
    assay_identifier_part_list = assay_identifier.split("_")
    if len(assay_identifier_part_list) <= 2:
        return assay_identifier
    return "_".join(assay_identifier_part_list[2:])


def build_assay_identifier_prefix(assay_identifier: str) -> str:
    assay_identifier_part_list = assay_identifier.split("_")
    if len(assay_identifier_part_list) <= 2:
        return assay_identifier
    return "_".join(assay_identifier_part_list[:2])


def build_sequence_to_file_stem_list_by_sequence(
    residue_directory_path: Path,
) -> dict[str, list[str]]:
    file_stem_list_by_sequence: dict[str, list[str]] = {}
    for residue_fasta_path in residue_directory_path.glob("*.fasta"):
        candidate_sequence = read_single_fasta_sequence(residue_fasta_path)
        file_stem_list_by_sequence.setdefault(candidate_sequence, []).append(residue_fasta_path.stem)
    return file_stem_list_by_sequence


def resolve_benchmark_file_stem(
    residue_directory_path: Path,
    assay_identifier: str,
    assay_file_name: str,
    target_sequence: str,
    file_stem_list_by_sequence: dict[str, list[str]],
) -> str:
    assay_file_stem = Path(assay_file_name).stem
    for candidate_file_stem in [assay_identifier, assay_file_stem]:
        candidate_residue_fasta_path = residue_directory_path / f"{candidate_file_stem}.fasta"
        if candidate_residue_fasta_path.exists():
            candidate_sequence = read_single_fasta_sequence(candidate_residue_fasta_path)
            if candidate_sequence == target_sequence:
                return candidate_file_stem

    matching_file_stem_list = list(file_stem_list_by_sequence.get(target_sequence, []))
    if len(matching_file_stem_list) == 1:
        return matching_file_stem_list[0]

    assay_identifier_suffix = build_assay_identifier_suffix(assay_identifier)
    suffix_matched_file_stem_list = [
        file_stem
        for file_stem in matching_file_stem_list
        if build_assay_identifier_suffix(file_stem) == assay_identifier_suffix
    ]
    if len(suffix_matched_file_stem_list) == 1:
        return suffix_matched_file_stem_list[0]

    assay_identifier_prefix = build_assay_identifier_prefix(assay_identifier)
    prefix_matched_file_stem_list = [
        file_stem
        for file_stem in matching_file_stem_list
        if build_assay_identifier_prefix(file_stem) == assay_identifier_prefix
    ]
    if len(prefix_matched_file_stem_list) == 1:
        return prefix_matched_file_stem_list[0]

    raise FileNotFoundError(
        "Could not resolve benchmark file stem for assay "
        f"{assay_identifier}; sequence-matched candidates={matching_file_stem_list}"
    )


def load_reference_row_list(reference_file_path: Path) -> list[dict[str, str]]:
    with reference_file_path.open(newline="", encoding="utf-8") as reference_file:
        return list(csv.DictReader(reference_file))


def resolve_directory_path_list(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    if arguments.base_dir:
        benchmark_root_directory_path = resolve_benchmark_root_directory(Path(arguments.base_dir))
        residue_directory_path = Path(arguments.residue_dir) if arguments.residue_dir else (
            benchmark_root_directory_path / "residue_sequence"
        )
        if arguments.structure_dir:
            structure_directory_path = Path(arguments.structure_dir)
        else:
            structure_directory_name = resolve_model_quantization_directory_name(arguments.model_name)
            structure_directory_path = (
                benchmark_root_directory_path / "structure_sequence" / structure_directory_name
            )
            if not structure_directory_path.exists():
                structure_directory_path = benchmark_root_directory_path / "structure_sequence"
        mutant_directory_path = Path(arguments.mutant_dir) if arguments.mutant_dir else (
            benchmark_root_directory_path / "substitutions"
        )
        return residue_directory_path, structure_directory_path, mutant_directory_path

    if not arguments.residue_dir or not arguments.structure_dir or not arguments.mutant_dir:
        raise ValueError(
            "Either --base_dir or all of --residue_dir, --structure_dir, and --mutant_dir must be provided."
        )
    return Path(arguments.residue_dir), Path(arguments.structure_dir), Path(arguments.mutant_dir)


def main() -> None:
    arguments = parse_arguments()
    device_name = resolve_device_name(arguments.device)
    reference_file_path = Path(arguments.reference_file_path)
    output_scores_directory_path = Path(arguments.output_scores_folder)
    output_scores_directory_path.mkdir(parents=True, exist_ok=True)

    residue_directory_path, structure_directory_path, mutant_directory_path = (
        resolve_directory_path_list(arguments)
    )
    file_stem_list_by_sequence = build_sequence_to_file_stem_list_by_sequence(
        residue_directory_path=residue_directory_path
    )
    reference_row_list = load_reference_row_list(reference_file_path=reference_file_path)

    print(
        f"Scoring {len(reference_row_list)} ProteinGym substitution assays "
        f"with {arguments.model_name} on {device_name}"
    )
    for assay_index, reference_row in enumerate(reference_row_list, start=1):
        assay_identifier = reference_row["DMS_id"]
        assay_file_name = reference_row["DMS_filename"]
        target_sequence = reference_row["target_seq"]
        benchmark_file_stem = resolve_benchmark_file_stem(
            residue_directory_path=residue_directory_path,
            assay_identifier=assay_identifier,
            assay_file_name=assay_file_name,
            target_sequence=target_sequence,
            file_stem_list_by_sequence=file_stem_list_by_sequence,
        )
        structure_token_list = read_structure_token_list(
            structure_directory_path / f"{benchmark_file_stem}.fasta"
        )

        mutant_file_path = mutant_directory_path / assay_file_name
        if not mutant_file_path.exists():
            mutant_file_path = mutant_directory_path / f"{benchmark_file_stem}.csv"
        mutant_data_frame = pd.read_csv(mutant_file_path)
        mutation_string_list = mutant_data_frame["mutant"].astype(str).tolist()

        fitness_score_list = predict_fitness(
            model_name=arguments.model_name,
            target_sequence=target_sequence,
            mutation_string_list=mutation_string_list,
            structure_token_list=structure_token_list,
            device_name=device_name,
        )

        output_data_frame = mutant_data_frame[["mutant"]].copy()
        output_data_frame[arguments.score_column_name] = fitness_score_list
        if "DMS_score" in mutant_data_frame.columns:
            output_data_frame["DMS_score"] = mutant_data_frame["DMS_score"]
        output_data_frame.to_csv(
            output_scores_directory_path / f"{assay_identifier}.csv",
            index=False,
        )
        print(f"[{assay_index}/{len(reference_row_list)}] wrote {assay_identifier}.csv")


if __name__ == "__main__":
    main()
