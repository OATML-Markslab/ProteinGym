import logging
import random
from itertools import islice
from pathlib import Path
from typing import Iterator, List, TypeVar

import numpy as np
import torch

from progen3.common.model_loading import get_model
from progen3.modeling import ProGen3ForCausalLM

AVAILABLE_MODELS = [
    "Profluent-Bio/progen3-3b",
    "Profluent-Bio/progen3-1b",
    "Profluent-Bio/progen3-762m",
    "Profluent-Bio/progen3-339m",
    "Profluent-Bio/progen3-219m",
    "Profluent-Bio/progen3-112m",
]
FILE_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)


def get_progen3_model(model_name: str, use_fsdp: bool) -> ProGen3ForCausalLM:
    """
    Initialize model only on rank 0 in cpu. Rest are initialized with empty weights.
    Returns a FSDP wrapped model.
    """
    if model_name not in AVAILABLE_MODELS:
        logger.warning(f"Model {model_name} not in AVAILABLE_MODELS; assuming its a local path.")

    model = get_model(
        model_name_or_path=model_name, model_class=ProGen3ForCausalLM, fsdp=use_fsdp, dtype=torch.bfloat16
    )
    return model


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_fasta_sequences(file_path: str, sequences: dict[str, str]) -> None:
    with open(file_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")


T = TypeVar("T")


def batched(iterator: Iterator[T], n: int) -> Iterator[List[T]]:
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            return
        yield batch
