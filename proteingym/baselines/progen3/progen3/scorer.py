import itertools
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
import torch.distributed
import torch.nn as nn
from Bio import SeqIO
from tqdm import tqdm

from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.common import dist
from progen3.modeling import MoeCausalOutputWithPast, ProGen3ForCausalLM

IndexedSequence = tuple[int, str]


class ProGen3Scorer:
    def __init__(
        self,
        model: ProGen3ForCausalLM,
        max_batch_tokens: int = 65536,
        reduction: str = "mean",
    ):
        super().__init__()
        self.batch_preparer = ProGen3BatchPreparer()
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Reduction must be one of {['mean', 'sum']}")
        self.reduction = reduction
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.model.eval()

    def group_by_length(self, indexed_sequences: list[IndexedSequence]) -> list[list[IndexedSequence]]:
        batches: list[list[IndexedSequence]] = [[]]
        for idx, seq in sorted(indexed_sequences, key=lambda idx_seq: (len(idx_seq[1]), idx_seq[0])):
            if len(batches[-1]) > 0 and len(seq) * (len(batches[-1]) + 1) > self.max_batch_tokens:
                batches.append([])
            batches[-1].append((idx, seq))

        return batches

    def batch_sequences(self, sequences: list[str]) -> list[list[int]]:  # type: ignore[override]
        """
        Batches the sequences and returns indices for the current rank
        We want to keep sequences of similar length together.
        Ensures that no batch exceeds max_batch_tokens
        """
        indexed_sequences: list[IndexedSequence] = list(enumerate(sequences))
        indexed_batches = self.group_by_length(indexed_sequences)
        batches = [[item[0] for item in batch] for batch in indexed_batches]  # type: ignore[no-redef,misc]

        assert sorted(sum(batches, [])) == list(
            range(len(sequences))
        ), "Batches must contain all indices with no repetition"

        world_size = dist.get_world_size()
        extra_batches_needed = (world_size - len(batches)) % world_size
        # create extra batches to make the total number of batches divisible by the world size
        batches += [batches[0][:1]] * extra_batches_needed
        assert len(batches) % world_size == 0

        return batches[dist.get_rank() :: world_size]  # type: ignore[return-value]

    @torch.no_grad
    def score_batch(self, sequences: list[str]) -> dict[str, list[float]]:
        kwargs_n_to_c = self.batch_preparer.get_batch_kwargs(sequences, device=dist.get_device(), reverse=False)
        output_batch = self._log_likelihoods(kwargs_n_to_c)

        kwargs_c_to_n = self.batch_preparer.get_batch_kwargs(sequences, device=dist.get_device(), reverse=True)
        output_rev_batch = self._log_likelihoods(kwargs_c_to_n)
        scores: dict[str, list[float]] = {"log_likelihood": [], "perplexity": []}
        for i in range(len(sequences)):
            ll_batch, ll_rev_batch = output_batch[i], output_rev_batch[i]
            ll = (ll_batch + ll_rev_batch) / 2
            scores["log_likelihood"].append(ll.item())
            scores["perplexity"].append(torch.exp(-ll).item())
        return scores

    def _log_likelihoods(self, model_forward_kwargs: dict[str, Any]) -> torch.Tensor:
        output: MoeCausalOutputWithPast = self.model(
            input_ids=model_forward_kwargs["input_ids"],
            labels=model_forward_kwargs["labels"],
            sequence_ids=model_forward_kwargs["sequence_ids"],
            position_ids=model_forward_kwargs["position_ids"],
            return_dict=True,
        )
        labels = model_forward_kwargs["labels"]
        target_mask = labels != self.model.config.pad_token_id

        targets = labels[..., 1:].contiguous()
        target_mask = target_mask[..., 1:].contiguous()
        logits = output.logits[..., :-1, :].contiguous().to(torch.float32)
        flat_logits = logits.view(-1, logits.shape[-1])
        nll = nn.functional.cross_entropy(flat_logits, targets.view(-1), reduction="none").view(targets.shape)
        nll = (nll * target_mask.to(nll)).sum(dim=1)
        if self.reduction == "mean":
            nll = nll / target_mask.sum(dim=1)
        return -nll.detach()

    def evaluate(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary mapping a scoring metric to a tensor of scores.

        In a distributed setting, each rank will compute scores for its own sequences,
        and then we will all_gather the scores from each rank to get the full tensor of scores.
        Each rank is responsible for its own indices in the full tensor.
        """

        sequence_batch_indices: list[list[int]] = self.batch_sequences(sequences)
        scores: dict[str, list[float]] = defaultdict(list)
        pbar = tqdm(desc="Scored sequences: ", ncols=80, disable=dist.get_rank() != 0)
        for indices in sequence_batch_indices:
            sequence_batch = [sequences[i] for i in indices]
            batch_scores: dict[str, list[float]] = self.score_batch(sequence_batch)
            for metric, float_list in batch_scores.items():
                scores[metric] += float_list

            batch_size = torch.tensor(len(sequence_batch), device=dist.get_device())
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(batch_size, op=torch.distributed.ReduceOp.SUM)
            pbar.update(batch_size.item())

        sequence_indices = list(itertools.chain.from_iterable(sequence_batch_indices))

        if torch.distributed.is_initialized():
            # gather scores and sequence batch indices from all ranks
            scores, sequence_indices = self._dist_get_rank_scores_and_indices(scores, sequence_indices)

        ordered_scores = self._order_scores_by_indices(scores, sequence_indices)
        return ordered_scores

    def _dist_get_rank_scores_and_indices(
        self,
        scores: dict[str, list[float]],
        sequence_indices: list[int],
    ) -> tuple[dict[str, list[float]], list[int]]:
        """
        Concatenates scores and sequence batch indices from all ranks
        Puts scores and indices in the same form as the local scores and indices
        """
        all_rank: list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(all_rank, (scores, sequence_indices))

        all_scores: dict[str, list[float]] = defaultdict(list)
        all_indices: list[int] = []
        for rank_scores, rank_indices in all_rank:  # type: ignore
            for metric, float_list in rank_scores.items():  # type: ignore
                all_scores[metric] += float_list
            all_indices.extend(rank_indices)

        return all_scores, all_indices

    def _order_scores_by_indices(
        self, scores: dict[str, list[float]], sequence_indices: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Scores are not ordered. We need to order them based on the sequence_indices.
        """
        output: dict[str, torch.Tensor] = {}
        for metric, float_list in scores.items():
            if metric not in output:
                output[metric] = torch.zeros(max(sequence_indices) + 1)
            for i, idx in enumerate(sequence_indices):
                output[metric][idx] = float_list[i]
        return output

    def run(self, fasta_path: str, output_path: str) -> None:
        sequences = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)

        seq_ids = sorted(list(sequences.keys()))
        seqs = [sequences[seq_id] for seq_id in seq_ids]

        scores = self.evaluate(seqs)

        if dist.get_rank() == 0:
            df = pd.DataFrame(scores, index=seq_ids)
            df.index.name = "sequence_id"
            df.to_csv(output_path)
