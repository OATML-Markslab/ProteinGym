from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import torch
import torch.multiprocessing as mp
from rich.logging import RichHandler

from vespag.models import FNN, MinimalCNN

from .type_hinting import Architecture, EmbeddingType

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_MODEL_PARAMETERS = {
    "architecture": "fnn",
    "model_parameters": {"hidden_dims": [256], "dropout_rate": 0.2},
    "embedding_type": "esm2",
}


def save_async(obj, pool: mp.Pool, path: Path, mkdir: bool = True):
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    pool.apply_async(torch.save, (obj, path))


def load_model_from_config(
    architecture: str, model_parameters: dict, embedding_type: str
):
    if architecture == "fnn":
        model = FNN(
            hidden_layer_sizes=model_parameters["hidden_dims"],
            input_dim=get_embedding_dim(embedding_type),
            dropout_rate=model_parameters["dropout_rate"],
        )
    elif architecture == "cnn":
        model = MinimalCNN(
            input_dim=get_embedding_dim(embedding_type),
            n_channels=model_parameters["n_channels"],
            kernel_size=model_parameters["kernel_size"],
            padding=model_parameters["padding"],
            fnn_hidden_layers=model_parameters["fully_connected_layers"],
            cnn_dropout_rate=model_parameters["dropout"]["cnn"],
            fnn_dropout_rate=model_parameters["dropout"]["fnn"],
        )
    return model


def load_model(
    architecture: Architecture,
    model_parameters: dict,
    embedding_type: EmbeddingType,
    checkpoint_file: Path = None,
) -> torch.nn.Module:
    checkpoint_file = checkpoint_file or Path.cwd() / "model_weights/state_dict_v2.pt"
    model = load_model_from_config(architecture, model_parameters, embedding_type)
    model.load_state_dict(torch.load(checkpoint_file))
    return model


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger("rich")
    logger.setLevel(logging.INFO)
    return logger


def get_embedding_dim(embedding_type: EmbeddingType) -> int:
    if embedding_type == "prott5":
        return 1024
    elif embedding_type == "esm2":
        return 2560


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_precision() -> Literal["half", "float"]:
    if "cuda" in str(get_device()):
        return "half"
    else:
        return "float"
