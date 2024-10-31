import re
from pathlib import Path
from typing import Union

import h5py
import rich.progress as progress
import torch
import typer
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from typing_extensions import Annotated

from vespag.utils import get_device
from vespag.utils.type_hinting import EmbeddingType

model_names = {
    "esm2": "facebook/esm2_t36_3B_UR50D",
    "prott5": "Rostlab/prot_t5_xl_uniref50",
}


class Embedder:
    def __init__(
        self, pretrained_path: Union[Path, str], cache_dir: Path = None
    ) -> None:
        device = get_device()
        self.device = device

        if "t5" in pretrained_path:
            tokenizer_class = T5Tokenizer
            encoder_class = T5EncoderModel
        else:
            tokenizer_class = AutoTokenizer
            encoder_class = AutoModel

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained_path, **kwargs, do_lower_case=False
        )
        self.encoder = encoder_class.from_pretrained(pretrained_path, **kwargs).to(
            device
        )
        self.encoder = (
            self.encoder.half()
            if device == torch.device("cuda:0")
            else self.encoder.float()
        )

    @staticmethod
    def batch(sequences: dict[str, str], max_batch_length: int) -> list[dict[str, str]]:
        batches = []
        current_batch = {}
        for n, (id, sequence) in enumerate(sequences.items()):
            if (
                sum(map(len, current_batch.values()))
                + min(len(sequence), max_batch_length)
                > max_batch_length
            ):
                batches.append(current_batch)
                current_batch = {id: sequence}
            else:
                current_batch[id] = sequence
        batches.append(current_batch)

        return batches

    def embed(
        self, sequences: dict[str, str], max_batch_length: int = 4096
    ) -> dict[str, torch.tensor]:
        batches = self.batch(sequences, max_batch_length)

        with progress.Progress(
            *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
        ) as pbar, torch.no_grad():
            embedding_progress = pbar.add_task(
                "Computing embeddings", total=sum(map(len, sequences.values()))
            )
            embeddings = {}
            for batch in batches:
                input_sequences = [
                    " ".join(list(re.sub(r"[UZOB]", "X", seq)))
                    for seq in batch.values()
                ]
                input_tokens = self.tokenizer.batch_encode_plus(
                    input_sequences,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                    max_length=max_batch_length,
                ).to(self.device)
                raw_embeddings = self.encoder(**input_tokens)
                embeddings.update(
                    {
                        id: raw_embeddings.last_hidden_state[i, 1 : len(seq) + 1]
                        .detach()
                        .float()
                        .cpu()
                        for i, (id, seq) in enumerate(batch.items())
                    }
                )
                pbar.advance(embedding_progress, sum(map(len, batch.values())))
            return embeddings

    @staticmethod
    def save_embeddings(embeddings: dict[str, torch.tensor], h5_path: Path) -> None:
        h5_path.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(h5_path, "w") as f:
            for id, emb in embeddings.items():
                f.create_dataset(id, data=emb.numpy())


def generate_embeddings(
    input_fasta_file: Annotated[Path, typer.Argument(help="Path of input FASTA file")],
    output_h5_file: Annotated[
        Path, typer.Argument(help="Path for saving HDF5 file with computed embeddings")
    ],
    cache_dir: Annotated[
        Path,
        typer.Option(
            "-c", "--cache-dir", help="Custom path to download model checkpoints to"
        ),
    ],
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "-e",
            "--embedding-type",
            case_sensitive=False,
            help="Type of embeddings to generate",
        ),
    ] = EmbeddingType.esm2,
    pretrained_path: Annotated[
        str,
        typer.Option("--pretrained-path", help="Path or URL of pretrained transformer"),
    ] = None,
):
    if embedding_type and not pretrained_path:
        pretrained_path = model_names[embedding_type]

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta_file, "fasta")}
    embedder = Embedder(pretrained_path, cache_dir)
    embeddings = embedder.embed(sequences)
    Embedder.save_embeddings(embeddings, output_h5_file)


if __name__ == "__main__":
    typer.run(generate_embeddings)
