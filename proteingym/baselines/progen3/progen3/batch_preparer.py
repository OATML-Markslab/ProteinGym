import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from progen3.common import dist

from .tokenizer import END_OF_SPAN_TOKEN, get_tokenizer

CLM_PATTERN = re.compile(r"^[A-Z]+$")
GLM_PATTERN = re.compile(r"^[A-Z]+\[GLM\](?:\d+\-\d+\-\d+;)*\d+\-\d+\-\d+;?$")


@dataclass
class DataPrepConfig:
    fuzzy_span_len_factor: float = 0.2
    max_glm_spans: int = 50


class ProGen3BatchPreparer:
    """
    Takes a batch of sequences and prepares them to be fed into the model's forward pass.
    """

    def __init__(
        self,
        data_prep_config: Optional[DataPrepConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        tokenizer = get_tokenizer()
        super().__init__()
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.rng = rng or np.random.default_rng(0)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.token_to_id("<pad>")

    def get_batch_kwargs(  # type: ignore[override]
        self,
        sequences: list[str],
        device: torch.device = torch.device("cpu"),
        reverse: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        NOTE: This function assumes all sequences are in 1->2 direction only.
        Passing reverse sequences will result in incorrect encoding.
        """
        sequence_encodings = [self.prepare_singleseq(sequence, reverse) for sequence in sequences]
        padded_encodings = self.pad_encodings(sequence_encodings)
        padded_encodings = {k: v.to(device=device, non_blocking=True) for k, v in padded_encodings.items()}

        return padded_encodings

    def pad_encodings(self, sequence_encodings: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        padding_value = {
            "input_ids": self.pad_token_id,
            "labels": self.pad_token_id,
            "position_ids": 0,
            "sequence_ids": 0,
        }
        padded_batch = {}
        for key, padding_value in padding_value.items():
            padded_batch[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings],
                batch_first=True,
                padding_value=padding_value,
            ).to(dtype=torch.long)

        return padded_batch

    def get_generation_kwargs(self, sequence: str, reverse_sequences: bool) -> dict[str, torch.Tensor]:
        """
        NOTE: This function assumes both sequence and context are in 1->2 direction only.
        Passing reverse sequences will result in incorrect encoding.
        """
        single_seq_encoding = self.prepare_singleseq(sequence, reverse_sequences)
        prefix_length = single_seq_encoding["metadata"]["prefix_length"]

        input_ids = single_seq_encoding["input_ids"][:prefix_length]
        sequence_ids = single_seq_encoding["sequence_ids"][:prefix_length]
        position_ids = single_seq_encoding["position_ids"][:prefix_length]

        return {
            "input_ids": input_ids.unsqueeze(0).to(dist.get_device()),
            "sequence_ids": sequence_ids.unsqueeze(0).to(dist.get_device()),
            "position_ids": position_ids.unsqueeze(0).to(dist.get_device()),
        }

    def prepare_singleseq(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        """
        NOTE: This function assumes sequence is in 1->2 direction only.
        Passing reverse sequence as first argument will result in incorrect encoding.
        """
        example = (self.prepare_clm if not is_glm_instance(sequence) else self.prepare_glm)(sequence, reverse_sequence)
        return example

    def prepare_clm(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        sequence = "1" + sequence + "2"
        if reverse_sequence:
            sequence = sequence[::-1]

        tokens = self.tokenizer.encode(f"<bos>{sequence}<eos>").ids
        return {
            "input_ids": torch.tensor(tokens),
            "labels": torch.tensor(tokens),
            "position_ids": torch.arange(len(tokens)),
            "sequence_ids": torch.zeros(len(tokens)),
            # Metadata for generation
            # remove <1/2><eos> from the end for generation
            "metadata": {"prefix_length": len(tokens) - 2},
        }

    def prepare_glm(self, sequence: str, reverse_sequence: bool) -> dict[str, Any]:
        sequence, masking_info = get_spans_to_mask(sequence)
        spans_to_mask = sorted(masking_info.keys())
        remaining_spans = get_remaining_spans_from_infill_spans(spans_to_mask, len(sequence))
        tokens = list(sequence)
        if reverse_sequence:
            tokens = tokens[::-1]
            spans_to_mask = [(len(tokens) - e, len(tokens) - s) for s, e in spans_to_mask]
            remaining_spans = [(len(tokens) - e, len(tokens) - s) for s, e in remaining_spans]
            masking_info = {(len(tokens) - e, len(tokens) - s): L for (s, e), L in masking_info.items()}

        infill_span_ids = self.rng.choice(self.data_prep_config.max_glm_spans, len(spans_to_mask))
        infill_span_ids = [f"<span_{i}>" for i in infill_span_ids]  # type: ignore

        all_spans = sorted(
            [(*x, True, infill_span_ids[i]) for i, x in enumerate(spans_to_mask)]
            + [(*x, False, "") for x in remaining_spans]
        )

        prefix_tokens, suffix_tokens = [], []
        prefix_pos_ids, suffix_pos_ids = [], []

        pos_id_start = 0 + 2  # 0 for <bos>, 1 for <term_1>
        for s, e, is_infill_span, span_id in all_spans:
            if is_infill_span:
                # If span is infill span, add it to suffix and replace it with span_id in prefix
                span_suffix_tokens = [span_id] + tokens[s:e] + [END_OF_SPAN_TOKEN]
                suffix_tokens.extend(span_suffix_tokens)
                suffix_pos_ids.extend(list(range(pos_id_start, pos_id_start + len(span_suffix_tokens))))

                prefix_tokens.append(span_id)
                prefix_pos_ids.append(pos_id_start)
                pos_id_start += 1

                # The infill length L may be different from the span length (for example for miniaturization)
                L = masking_info[(s, e)]
                fuzzy_diff = np.floor(L * self.data_prep_config.fuzzy_span_len_factor)
                pos_id_start += L + int(fuzzy_diff)
            else:
                # If span is remaining span, add it to prefix
                prefix = tokens[s:e]
                prefix_tokens.extend(deepcopy(prefix))
                prefix_pos_ids.extend(list(range(pos_id_start, pos_id_start + len(prefix))))
                pos_id_start += len(prefix)

        term_1, term_2 = ("1", "2") if not reverse_sequence else ("2", "1")

        prefix_tokens = ["<bos_glm>", term_1] + prefix_tokens + [term_2, "<eos>"]
        prefix_labels = ["<pad>"] * len(prefix_tokens)
        prefix_pos_ids = [0, 1] + prefix_pos_ids + [pos_id_start, pos_id_start + 1]

        input_tokens = prefix_tokens + suffix_tokens
        labels = prefix_labels + [x if not x.startswith("<span_") else "<pad>" for x in suffix_tokens]
        pos_ids = prefix_pos_ids + suffix_pos_ids

        input_tokens_ids = self.tokenizer.encode("".join(input_tokens)).ids
        labels_ids = self.tokenizer.encode("".join(labels)).ids

        return {
            "input_ids": torch.tensor(input_tokens_ids),
            "labels": torch.tensor(labels_ids),
            "position_ids": torch.tensor(pos_ids),
            "sequence_ids": torch.zeros(len(input_tokens_ids)),
            # Metadata for generation
            # +1 for first span_id in suffix which we want to keep
            "metadata": {"prefix_length": len(prefix_pos_ids) + 1},
        }


def assert_valid_instance(sequence: str) -> None:
    if len(sequence) == 0:
        return

    if not CLM_PATTERN.match(sequence) and not GLM_PATTERN.match(sequence):
        raise ValueError(f"Sequence is not a valid CLM or GLM instance: {sequence}")


def is_glm_instance(sequence: str) -> bool:
    return GLM_PATTERN.match(sequence) is not None


def get_spans_to_mask(sequence: str) -> tuple[str, dict[tuple[int, int], int]]:
    spans = {}
    sequence, spans_str = sequence.split("[GLM]")
    spans_str = spans_str.strip(";")
    for span in spans_str.split(";"):
        s, e, length = span.split("-")
        spans[(int(s), int(e))] = int(length)
    return sequence, spans


def prepare_glm_string_from_spans(spans: dict[tuple[int, int], int]) -> str:
    return "[GLM]" + ";".join(f"{s}-{e}-{v}" for (s, e), v in spans.items())


def get_remaining_spans_from_infill_spans(
    infill_spans: list[tuple[int, int]], num_tokens: int
) -> list[tuple[int, int]]:
    remaining_spans = []
    start = 0
    for s, e in infill_spans:
        assert s >= 0 and e <= num_tokens, f"Span {s}-{e} is invalid for sequence of length {num_tokens}."
        if start < s:
            remaining_spans.append((start, s))
        start = e
    if start < num_tokens:
        remaining_spans.append((start, num_tokens))

    return remaining_spans
