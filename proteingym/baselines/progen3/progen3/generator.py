import logging
import math
import re
from copy import deepcopy
from typing import Iterator, NamedTuple, Optional

import pandas as pd
import torch
import torch.distributed
from tqdm import tqdm
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerateDecoderOnlyOutput, GenerationConfig

from progen3.batch_preparer import (
    ProGen3BatchPreparer,
    assert_valid_instance,
    get_spans_to_mask,
    is_glm_instance,
    prepare_glm_string_from_spans,
)
from progen3.common import dist
from progen3.common.dist import generate
from progen3.modeling import ProGen3ForCausalLM
from progen3.tools.utils import batched, write_fasta_sequences

logger = logging.getLogger(__name__)


class GenerationResult(NamedTuple):
    generation: str  # aka raw generation from the model
    sequence: Optional[str]  # cleaned, validated and compiled sequence in the forward direction


class ProGen3Generator:
    OUTPUT_GENERATIONS_PATTERN = "{output_dir}/{prompt_id}.gen.fasta"
    OUTPUT_SEQUENCES_PATTERN = "{output_dir}/{prompt_id}.seq.fasta"

    def __init__(
        self,
        model: ProGen3ForCausalLM,
        max_batch_tokens: int = 65536,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        self.model = model
        self.model.eval()
        self.batch_preparer = ProGen3BatchPreparer()

        # eos_token_id is set in run_generation()
        self.default_gen_config = GenerationConfig(
            do_sample=True,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.batch_preparer.tokenizer.padding["pad_id"],
        )

        self.max_batch_tokens = max_batch_tokens

    def generate(
        self,
        prompt: str,
        num_sequences: int,
        min_new_tokens: int,
        max_new_tokens: int,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Iterator[GenerationResult]:
        prompt, direction = parse_directed_prompt(prompt)
        # After above operation, prompt is in 1->2 format only (including GLM spans)
        assert_valid_instance(prompt)

        # Prepare input
        reverse_sequence = direction == "rev"
        input_encoding = self.batch_preparer.get_generation_kwargs(prompt, reverse_sequence)
        num_input_tokens = len(input_encoding["input_ids"][0])
        logger.info(f"Generating for {prompt} ({direction}) with {num_input_tokens} input tokens")

        # Fill out generation config
        end_token = "<eos>" if not is_glm_instance(prompt) else "<eos_span>"
        end_token_id = self.batch_preparer.tokenizer.token_to_id(end_token)
        assert end_token_id is not None, f"End token {end_token} not found in tokenizer"

        gen_config = deepcopy(gen_config or self.default_gen_config)
        gen_config.eos_token_id = end_token_id
        gen_config.max_new_tokens = max_new_tokens + 2
        gen_config.min_new_tokens = min_new_tokens + 2

        batch_size = max(1, math.floor(self.max_batch_tokens / (num_input_tokens + gen_config.max_new_tokens)))

        # Prepare key-value cache with the prompt
        cached_length = num_input_tokens - 1
        key_value_cache: DynamicCache | None = None
        if cached_length > 0:
            with torch.no_grad():
                cached_encoding = {k: v[:, :cached_length] for k, v in input_encoding.items()}
                key_value_cache = self.model(**cached_encoding, use_cache=True, return_dict=True).past_key_values
            assert key_value_cache is not None, "Key-value cache must be non-None"
            assert key_value_cache.get_seq_length(0) == cached_length, f"Cache must have {cached_length} tokens."
            torch.cuda.empty_cache()

        # Generate sequences
        pbar = tqdm(
            total=num_sequences,
            ncols=80,
            disable=dist.get_rank() != 0,
            desc=f"Sequences generated (Rank {dist.get_rank()}): ",
        )
        for i in range(0, num_sequences, batch_size):
            num_gen = min(num_sequences - i, batch_size)

            if key_value_cache:
                key_value_cache.batch_repeat_interleave(num_gen)
            gen_config.num_return_sequences = num_gen
            outputs: GenerateDecoderOnlyOutput = generate(
                self.model, **input_encoding, generation_config=gen_config, past_key_values=key_value_cache
            )
            if key_value_cache:
                key_value_cache.crop(cached_length)
                key_value_cache.batch_select_indices([0])
                assert key_value_cache.get_seq_length(0) == cached_length, f"Cache must have {cached_length} tokens."

            # Get generated sequences
            input_token_ids = input_encoding["input_ids"][0].cpu().numpy().tolist()
            input_token_ids_in_completions = outputs.sequences[:, :num_input_tokens].tolist()
            assert all(
                input_token_ids == x for x in input_token_ids_in_completions
            ), "Input tokens must be the same in prompt and completion."
            completions = outputs.sequences[:, num_input_tokens:].tolist()
            decoded_completions = self.batch_preparer.tokenizer.decode_batch(completions, skip_special_tokens=False)
            pbar.update(num_gen)
            for decoded_completion in decoded_completions:
                compiled_completion = compile_generation(prompt, decoded_completion, direction)
                yield GenerationResult(sequence=compiled_completion, generation=decoded_completion)

    def run(self, prompt_file: str, output_dir: str, n_per_prompt: int) -> None:
        prompt_df = pd.read_csv(prompt_file)
        for _, row in prompt_df.iterrows():
            prompt_id, prompt = row["id"], row["sequence"]
            min_new_tokens = row["min_new_tokens"]
            max_new_tokens = row["max_new_tokens"]
            generations_per_rank = math.ceil(n_per_prompt / dist.get_world_size())

            results = []
            generation_iterator = self.generate(prompt, generations_per_rank, min_new_tokens, max_new_tokens)

            for batch_generations in batched(generation_iterator, 500):
                if torch.distributed.is_initialized():
                    all_batch_generations: list = [None for _ in range(dist.get_world_size())]
                    torch.distributed.all_gather_object(all_batch_generations, batch_generations)
                    batch_generations = [result for rank_results in all_batch_generations for result in rank_results]
                results.extend(batch_generations)

            results = results[:n_per_prompt]
            if dist.get_rank() == 0:
                self._save_generations(prompt_id, results, output_dir)

    def _save_generations(self, prompt_id: str, results: list[GenerationResult], output_dir: str) -> None:
        generation_seqs = {}
        sequence_seqs = {}
        for i, result in enumerate(results):
            new_seq_id = str(i)
            generation_seqs[new_seq_id] = result.generation
            if result.sequence:
                sequence_seqs[new_seq_id] = result.sequence

        generations_file = self.OUTPUT_GENERATIONS_PATTERN.format(output_dir=output_dir, prompt_id=prompt_id)
        sequences_file = self.OUTPUT_SEQUENCES_PATTERN.format(output_dir=output_dir, prompt_id=prompt_id)
        write_fasta_sequences(generations_file, generation_seqs)
        write_fasta_sequences(sequences_file, sequence_seqs)


def completion_validity_pattern(mode: str, direction: str) -> re.Pattern:
    pattern = r"[ACDEFGHIKLMNPQRSTVWY]+"  # all non-X standard amino acids
    if mode == "CLM":
        terminal = "1" if direction == "rev" else "2"
        pattern += "(" + re.escape(terminal + "<eos>") + ")"
    elif mode == "GLM":
        pattern += "(" + re.escape("<eos_span>") + ")"

    return re.compile(pattern)


def compile_generation(prompt: str, completion: str, direction: str) -> str | None:
    mode = "GLM" if is_glm_instance(prompt) else "CLM"
    pattern = completion_validity_pattern(mode, direction)
    match = pattern.match(completion)
    if match is None:
        return None

    # Remove special tokens from the completion
    stripped_completion = re.sub(r"[^A-Z]", "", completion)

    # original prompt is always 1->2, whereas completion is either:
    # direction fwd: 1->2
    # direction rev: 2->1
    match (mode, direction):
        case ("CLM", "fwd"):
            compiled_completion = prompt + stripped_completion
        case ("CLM", "rev"):
            compiled_completion = stripped_completion[::-1] + prompt
        case ("GLM", "fwd"):
            prompt, spans = get_spans_to_mask(prompt)
            assert len(spans) == 1, f"GLM can only have one span, but got {spans} spans."
            s, e = next(iter(spans.keys()))
            compiled_completion = prompt[:s] + stripped_completion + prompt[e:]
        case ("GLM", "rev"):
            prompt, spans = get_spans_to_mask(prompt)
            assert len(spans) == 1, f"GLM can only have one span, but got {spans} spans."
            s, e = next(iter(spans.keys()))
            compiled_completion = prompt[:s] + stripped_completion[::-1] + prompt[e:]
        case _:
            raise ValueError(f"Invalid mode or direction: {mode} {direction}")

    return compiled_completion


def parse_directed_prompt(prompt: str) -> tuple[str, str]:
    assert prompt.startswith("1") or prompt.startswith("2"), "Prompt must start with 1 or 2"
    is_fwd = prompt.startswith("1")
    prompt = prompt[1:]

    is_glm = is_glm_instance(prompt)

    match (is_glm, is_fwd):
        case (False, True):
            return prompt, "fwd"
        case (False, False):
            return prompt[::-1], "rev"
        case (True, True):
            return prompt, "fwd"
        case (True, False):
            sequence, spans = get_spans_to_mask(prompt)
            sequence = sequence[::-1]
            spans = {(len(sequence) - e, len(sequence) - s): v for (s, e), v in spans.items()}
            post_string = prepare_glm_string_from_spans(spans)
            return sequence + post_string, "rev"
        case _:
            raise ValueError(f"Invalid mode or direction: {is_glm} {is_fwd}")
