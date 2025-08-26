from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypedDict
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet, S3DiAlphabet
from poet_2.alphabets import Uniprot21, append_startstop
from poet_2.models.modules.packed_sequence import PackedTensorSequences
from poet_2.models.poet_2 import N_ATOMB, PoET2


@contextmanager
def default_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def load_model(
    path: Path | str, device: torch.device, dtype: torch.dtype | None = None
) -> PoET2:
    if dtype is None:
        model_dtype = torch.float16
        if device.type == "cpu":
            model_dtype = torch.float32  # half precision not supported on cpu
    else:
        model_dtype = dtype
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "init_args" in ckpt["hyper_parameters"].keys():
        hparams = ckpt["hyper_parameters"]["init_args"]["model_spec"]["init_args"]
    else:
        hparams = ckpt["hyper_parameters"]["model_spec"]["init_args"]
    with default_dtype(model_dtype), torch.device(device):
        model = PoET2(**hparams)
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    )
    model.mask_token_s3di = S3DiAlphabet().mask_token
    model = model.to(model_dtype)
    jit_warmup(model=model)
    return model


# SEQUENCE ENCODING STUFF


@dataclass
class NamedInput:
    sequence: bytes | str
    plddt: npt.NDArray[np.float32] = field(init=False)  # Shape: (L,)
    atomx: npt.NDArray[np.float32] = field(
        init=False
    )  # Shape: (L, 3, 3) Atom Order: N-Ca-C

    def __init__(
        self,
        sequence: bytes | str,
        plddt: npt.NDArray[np.float32] | None = None,
        atomx: npt.NDArray[np.float32] | None = None,
    ) -> None:
        if plddt is not None:
            assert len(plddt) == len(sequence)
        if atomx is not None:
            assert len(atomx) == len(sequence)
        self.length = len(sequence)
        self.sequence = sequence
        if plddt is not None:
            self.plddt = plddt
        else:
            self.plddt = np.full((self.length,), np.nan, dtype=np.float32)
        if atomx is not None:
            self.atomx = atomx
        else:
            self.atomx = np.full((self.length, 3, 3), np.nan, dtype=np.float32)

    def __len__(self) -> int:
        return self.length

    def enforce(self) -> Self:
        # assuming the input is either a pdb or afdb structure
        # in which case, can enforce based on atomx, and then based on plddt
        # i.e. don't need to consider enforcing both at the same time
        # enforce constraints for pdb structures
        if self.atomx is not None and self.plddt is not None:
            is_nan_any = np.isnan(self.atomx).reshape(len(self.atomx), -1).any(-1)
            self.plddt[is_nan_any] = np.nan
            self.atomx[is_nan_any] = np.nan
        # enforce constraints for afdb structures
        self.mask_by_plddt()
        return self

    def mask_by_plddt(self, plddt_threshold: float = 70) -> Self:
        if self.plddt is None:
            return self
        if self.atomx is not None:
            self.atomx[self.plddt < plddt_threshold] = np.nan
        return self


Input = bytes | str | NamedInput


class TokenizedData(TypedDict):
    seqs: torch.Tensor
    segment_sizes: torch.Tensor
    plddts: torch.Tensor | None
    s3dis: torch.Tensor | None
    atomxs: torch.Tensor | None
    atombs: torch.Tensor | None


def _tokenize_seqs(
    inputs: Sequence[Input],
    alphabet: Uniprot21 = Alphabet(),
    alphabet_s3di: Uniprot21 = S3DiAlphabet(),
    allow_return_none: bool = True,
    auto_append_startstop: bool = True,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    assert len(inputs) > 0
    has_structure = False
    named_inputs: Sequence[NamedInput] = []
    for input_ in inputs:
        if not isinstance(input_, NamedInput):
            input_ = NamedInput(input_)
        if isinstance(input_.sequence, str):
            input_ = replace(input_, sequence=input_.sequence.encode())
        has_structure = input_.plddt is not None or input_.atomx is not None
        named_inputs.append(input_)

    seqs, plddts, s3dis, atomxs, atombs = [], [], [], [], []
    for input_ in named_inputs:
        seq = alphabet.encode(input_.sequence)
        if auto_append_startstop:
            seq = append_startstop(seq, alphabet=alphabet)
        seqs.append(torch.from_numpy(seq))
        if allow_return_none and not has_structure:
            continue
        plddt = np.full_like(seqs[-1], np.nan, dtype=np.float32)
        s3di = np.full_like(seqs[-1], alphabet_s3di.mask_token, dtype=np.uint8)
        atomx = np.full((len(seq), 3, 3), np.nan, dtype=np.float32)
        if input_.plddt is not None:
            if auto_append_startstop:
                plddt[1:-1] = input_.plddt
            else:
                plddt[:] = input_.plddt
        if input_.atomx is not None:
            if auto_append_startstop:
                atomx[1:-1] = input_.atomx
            else:
                atomx[:] = input_.atomx
        assert len(plddt) == len(atomx) == len(s3di) == len(seq)
        plddts.append(torch.from_numpy(plddt))
        s3dis.append(torch.from_numpy(s3di))
        atomxs.append(torch.from_numpy(atomx))
        atombs.append(atomb_from_atomx(atomx))
    if allow_return_none and not has_structure:
        return seqs, None, None, None, None
    else:
        return seqs, plddts, s3dis, atomxs, atombs


def tokenize_seqs(
    inputs: Sequence[Input],
    device: torch.device = torch.device("cpu"),
    alphabet: Uniprot21 = Alphabet(),
    alphabet_s3di: Uniprot21 = S3DiAlphabet(),
    auto_append_startstop: bool = True,
) -> TokenizedData:
    seqs, plddts, s3dis, atomxs, atombs = _tokenize_seqs(
        inputs=inputs,
        alphabet=alphabet,
        alphabet_s3di=alphabet_s3di,
        auto_append_startstop=auto_append_startstop,
    )

    segment_sizes = torch.tensor([len(s) for s in seqs], device=device).unsqueeze(1)
    seqs = (
        pad_sequence(seqs, batch_first=True, padding_value=alphabet.mask_token)
        .to(device)
        .long()
    )
    if plddts is None:
        assert s3dis is None and atomxs is None and atombs is None
        return {
            "seqs": seqs,
            "segment_sizes": segment_sizes,
            "plddts": None,
            "s3dis": None,
            "atomxs": None,
            "atombs": None,
        }
    assert s3dis is not None and atomxs is not None and atombs is not None
    assert len(seqs) == len(plddts) == len(s3dis) == len(atomxs) == len(atombs)
    plddts = pad_sequence(plddts, batch_first=True, padding_value=torch.nan).to(device)
    s3dis = (
        pad_sequence(s3dis, batch_first=True, padding_value=alphabet_s3di.mask_token)
        .to(device)
        .long()
    )
    atomxs = pad_sequence(atomxs, batch_first=True, padding_value=torch.nan).to(device)
    atombs = pad_sequence(atombs, batch_first=True, padding_value=torch.nan).to(device)
    return {
        "seqs": seqs,
        "segment_sizes": segment_sizes,
        "plddts": plddts,
        "s3dis": s3dis,
        "atomxs": atomxs,
        "atombs": atombs,
    }


def tokenize_seq_of_seqs(
    inputs: Sequence[Sequence[Input]],
    device: torch.device = torch.device("cpu"),
    alphabet: Uniprot21 = Alphabet(),
    alphabet_s3di: Uniprot21 = S3DiAlphabet(),
    auto_append_startstop: bool = True,
) -> TokenizedData:
    """
    Tokenizes a batch of sequences of sequences.

    A sequence of sequences with no sequences will be tokenized as the empty prompt.
    The empty prompt consists of a single token, the masked token.
    """
    seqs, segment_sizes, plddts, s3dis, atomxs, atombs = [], [], [], [], [], []
    for input_ in inputs:
        if len(input_) > 0:
            seqs_, plddts_, s3dis_, atomxs_, atombs_ = _tokenize_seqs(
                inputs=input_,
                alphabet=alphabet,
                alphabet_s3di=alphabet_s3di,
                allow_return_none=False,
                auto_append_startstop=auto_append_startstop,
            )
        else:
            seqs_ = [torch.full((1,), alphabet.mask_token, dtype=torch.uint8)]
            plddts_ = [torch.full((1,), torch.nan, dtype=torch.float32)]
            s3dis_ = [torch.full((1,), alphabet_s3di.mask_token, dtype=torch.uint8)]
            atomxs_ = [torch.full((1, 3, 3), torch.nan, dtype=torch.float32)]
            atombs_ = [torch.full((1, N_ATOMB), torch.nan, dtype=torch.float32)]
        assert plddts_ is not None and s3dis_ is not None
        assert atomxs_ is not None and atombs_ is not None
        seqs.append(torch.cat(seqs_))
        segment_sizes.append(torch.tensor([len(s) for s in seqs_]))
        plddts.append(torch.cat(plddts_))
        s3dis.append(torch.cat(s3dis_))
        atomxs.append(torch.cat(atomxs_))
        atombs.append(torch.cat(atombs_))
    seqs = (
        pad_sequence(seqs, batch_first=True, padding_value=alphabet.mask_token)
        .to(device)
        .long()
    )
    segment_sizes = pad_sequence(segment_sizes, batch_first=True).to(device)
    plddts = pad_sequence(plddts, batch_first=True, padding_value=torch.nan).to(device)
    s3dis = (
        pad_sequence(s3dis, batch_first=True, padding_value=alphabet_s3di.mask_token)
        .to(device)
        .long()
    )
    atomxs = pad_sequence(atomxs, batch_first=True, padding_value=torch.nan).to(device)
    atombs = pad_sequence(atombs, batch_first=True, padding_value=torch.nan).to(device)
    return {
        "seqs": seqs,
        "segment_sizes": segment_sizes,
        "plddts": plddts,
        "s3dis": s3dis,
        "atomxs": atomxs,
        "atombs": atombs,
    }


class QueryData(TypedDict):
    ref_idxs: torch.Tensor
    known: torch.Tensor


def encode_query_data(
    encoder_batch: TokenizedData,
    decoder_batch: TokenizedData,
    query_idxs_batch: Sequence[Sequence[int | None]],
    alphabet: Uniprot21 = Alphabet(),
) -> QueryData:
    ref_idxs = torch.full_like(decoder_batch["seqs"], -100, dtype=torch.long)
    known = torch.full_like(decoder_batch["seqs"], False, dtype=torch.bool)
    for batch_idx, (x_seqs, x_seqlens, y_seqlens, query_idxs) in enumerate(
        zip(
            encoder_batch["seqs"],
            encoder_batch["segment_sizes"],
            decoder_batch["segment_sizes"],
            query_idxs_batch,
            strict=True,
        )
    ):
        x_cu_seqlens = F.pad(x_seqlens[x_seqlens > 0].cumsum(dim=0), (1, 0)).tolist()
        y_cu_seqlens = F.pad(y_seqlens[y_seqlens > 0].cumsum(dim=0), (1, 0)).tolist()
        for idx, query_idx in enumerate(query_idxs):
            if query_idx is None:
                continue
            ref_idxs[batch_idx, y_cu_seqlens[idx] : y_cu_seqlens[idx + 1]] = (
                torch.arange(
                    x_cu_seqlens[query_idx],
                    x_cu_seqlens[query_idx + 1],
                    dtype=torch.long,
                    device=ref_idxs.device,
                )
            )
            known[batch_idx, y_cu_seqlens[idx] : y_cu_seqlens[idx + 1]] = (
                x_seqs[x_cu_seqlens[query_idx] : x_cu_seqlens[query_idx + 1]]
                != alphabet.mask_token
            )
    return {"ref_idxs": ref_idxs, "known": known}


# STRUCTURE ENCODING STUFF


_ATOMB_TRIU = torch.triu_indices(9, 9, offset=1)


def atomb_from_atomx(atomx: npt.NDArray[np.float32]) -> torch.Tensor:
    if atomx.shape[0] == 1:
        return torch.full((1, N_ATOMB), torch.nan, dtype=torch.half)
    assert atomx.shape[0] >= 3
    atomx = atomx[:, :3, :]  # use backbone N-Ca-C atoms only, (L, 3, 3)
    left, center, right = atomx[:-2], atomx[1:-1], atomx[2:]
    atoms = np.concatenate((left, center, right), axis=1)  # (L, 9, 3)
    atoms = torch.from_numpy(atoms)
    distances = torch.cdist(atoms, atoms)
    distances = distances[:, _ATOMB_TRIU[0], _ATOMB_TRIU[1]].half()
    return F.pad(distances, (0, 0, 1, 1), value=torch.nan)


def atomb_from_atomx_torch(atomx: torch.Tensor) -> torch.Tensor:
    assert atomx.shape[0] >= 3
    atomx = atomx[:, :3, :]  # use backbone N-Ca-C atoms only, (L, 3, 3)
    left, center, right = atomx[:-2], atomx[1:-1], atomx[2:]
    atoms = torch.concatenate((left, center, right), dim=1)  # (L, 9, 3)
    distances = torch.cdist(atoms, atoms)
    distances = distances[:, _ATOMB_TRIU[0], _ATOMB_TRIU[1]].half()
    return F.pad(distances, (0, 0, 1, 1), value=torch.nan)


# HIGH LEVEL SCORING FUNCTIONS


def compute_memory(
    model: PoET2, prompts: Sequence[Sequence[Input]]
) -> tuple[list[PackedTensorSequences], torch.Tensor]:
    device = next(model.parameters()).device
    tokenized = tokenize_seq_of_seqs(prompts, device=device)
    return model.outputs_with_ref_values(
        xs=tokenized["seqs"],
        segment_sizes=tokenized["segment_sizes"],
        xs_plddts=tokenized["plddts"],
        xs_s3dis=tokenized["s3dis"],
        xs_atomxs=tokenized["atomxs"],
        xs_atombs=tokenized["atombs"],
    )


def score_sequences_given_memory(
    model: PoET2,
    memory: list[PackedTensorSequences],
    sequences: Sequence[Input],
    seqid: torch.Tensor | None = None,
    ys_ref_values: torch.Tensor | None = None,
    self_prompt: torch.Tensor | None = None,
    alphabet: Uniprot21 = Alphabet(),
) -> torch.Tensor:
    device = next(iter(model.parameters())).device
    tokenized = tokenize_seqs(sequences, device=device, alphabet=alphabet)
    # compute the sequence logps given memory
    if seqid is not None:
        assert ys_ref_values is not None
        seqid = seqid.expand(*tokenized["seqs"].size(), 2)
    if ys_ref_values is not None:
        if memory[0].cu_seqlens.numel() - 1 > 1:
            raise NotImplementedError("batch memory with ys_ref_values not supported")
        if seqid is None:
            ys_refs = torch.arange(0, tokenized["seqs"].size(1), device=device)
            ys_ref_values = ys_ref_values[ys_refs.tile(tokenized["seqs"].size(0))]
        else:
            ys_ref_values = ys_ref_values[[0]]
    logits = model.outputs_from_memory(
        decoder=model.clm_decoder,
        memory=memory,
        ys=tokenized["seqs"],
        ys_segment_sizes=tokenized["segment_sizes"],
        ys_plddts=tokenized["plddts"],
        ys_s3dis=tokenized["s3dis"],
        ys_atomxs=tokenized["atomxs"],
        ys_atombs=tokenized["atombs"],
        ys_seqids=seqid,
        ys_ref_values=ys_ref_values,
    ).logits
    assert logits is not None
    logits = logits.chunk(2, dim=2)[0]
    target = tokenized["seqs"]

    logits = logits[:, :-1]
    target = target[:, 1:]
    # fmt: off
    if self_prompt is not None:
        self_prompt = self_prompt[1:]

        idx = alphabet.gap_token
        query_one_hot = F.one_hot(self_prompt, logits.size(-1)).to(logits.device)
        logits = F.log_softmax(logits, dim=-1)

        # this does not enforce consistency
        log_p_match = logits[..., idx].unsqueeze(-1) # the log probability of emitting '-'
        log_p_match_token = torch.log(query_one_hot) + log_p_match # emitting '-' means the token in the mask

        logits = torch.logaddexp(logits, log_p_match_token)
        logits[..., idx] = -np.inf # we've converted outputting '-' to the token, so this should be -inf

        # this version enforces consitency
        # is_match = (self_prompt != alphabet.mask_token).float().unsqueeze(-1)
        # logits[..., idx] = -np.inf
        # logits = torch.log_softmax(logits, dim=-1)
        # logits = torch.logaddexp(torch.log(1-is_match) + logits, torch.log(is_match) + torch.log(query_one_hot))
    # fmt: on
    return -(
        F.cross_entropy(
            logits.transpose(1, 2),
            target,
            ignore_index=alphabet.mask_token,
            reduction="none",
        )
        .float()
        .sum(dim=1)
    )


def score_sequences(
    model: PoET2,
    memory: list[PackedTensorSequences],
    sequences: Sequence[Input],
    seqid: torch.Tensor | None = None,
    ys_ref_values: torch.Tensor | None = None,
    self_prompt: Input | None = None,
    alphabet: Uniprot21 = Alphabet(),
    batch_size: int = 256,
    verbose: bool = False,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if self_prompt is not None:
        assert seqid is None and ys_ref_values is not None
        self_prompt_ = tokenize_seqs([self_prompt])["seqs"].squeeze(0)
        assert (self_prompt_ != alphabet.gap_token).all()
        self_prompt_ = self_prompt_.to(device)
    else:
        self_prompt_ = None
    logps = torch.empty(len(sequences), dtype=torch.float32, device=device)
    if verbose:
        iterator = trange(0, len(sequences), batch_size)
    else:
        iterator = range(0, len(sequences), batch_size)
    for i in iterator:
        logps[i : i + batch_size] = score_sequences_given_memory(
            model=model,
            memory=memory,
            sequences=sequences[i : i + batch_size],
            seqid=seqid,
            ys_ref_values=ys_ref_values,
            self_prompt=self_prompt_,
            alphabet=alphabet,
        )
    return logps


# JIT CONFIG


def jit_warmup(
    model: PoET2,
    alphabet: Alphabet = Alphabet(),
):
    device = next(iter(model.parameters())).device
    with torch.inference_mode():
        x = b"$WAAAGH*$WAAGW*"
        segment_sizes = [8, 7]
        x = alphabet.encode(x)
        x = torch.from_numpy(x).long().to(device)
        segment_sizes = torch.tensor(segment_sizes).long().to(device)
        _ = model.forward(
            xs=x.unsqueeze(0).expand(4, -1),
            xs_plddts=None,
            xs_s3dis=(
                torch.full_like(x.unsqueeze(0).expand(4, -1), alphabet.mask_token)
                if model.s3di_embed is not None
                else None
            ),
            xs_atomxs=None,
            xs_atombs=None,
            xs_segment_sizes=segment_sizes.unsqueeze(0).expand(4, -1),
            mlm_ys=x.unsqueeze(0).expand(2, -1),
            mlm_ys_seqids=None,
            mlm_ys_plddts=None,
            mlm_ys_s3dis=(
                torch.full_like(x.unsqueeze(0).expand(2, -1), alphabet.mask_token)
                if model.s3di_embed is not None
                else None
            ),
            mlm_ys_atomxs=None,
            mlm_ys_atombs=None,
            mlm_ys_refs=torch.full_like(x.unsqueeze(0).expand(2, -1), -100),
            mlm_ys_segment_sizes=segment_sizes.unsqueeze(0).expand(2, -1),
            clm_ys=x.unsqueeze(0).expand(2, -1),
            clm_ys_seqids=None,
            clm_ys_plddts=None,
            clm_ys_s3dis=(
                torch.full_like(x.unsqueeze(0).expand(2, -1), alphabet.mask_token)
                if model.s3di_embed is not None
                else None
            ),
            clm_ys_atomxs=None,
            clm_ys_atombs=None,
            clm_ys_refs=torch.full_like(x.unsqueeze(0).expand(2, -1), -100),
            clm_ys_segment_sizes=segment_sizes.unsqueeze(0).expand(2, -1),
        )
