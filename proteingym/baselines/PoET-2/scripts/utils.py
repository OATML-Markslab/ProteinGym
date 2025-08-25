import hashlib
import io
import string
from pathlib import Path

import numpy as np
import numpy.typing as npt

import pyzstd

from poet_2.alphabets import Uniprot21
from poet_2.fasta import parse_stream


ASCII_LOWERCASE_BYTES = string.ascii_lowercase.encode()


def read_maybe_zstd_compressed_file(path: Path) -> bytes:
    data = open(path, "rb").read()
    if path.suffix == ".zst":
        data = pyzstd.decompress(data)
    return data


def get_names_and_seqs_from_fastalike(
    filepath: Path,
) -> tuple[list[bytes], list[bytes]]:
    names, sequences = [], []
    for name, sequence in parse_stream(
        io.BytesIO(read_maybe_zstd_compressed_file(filepath)), upper=False
    ):
        names.append(name)
        sequences.append(sequence)
    return names, sequences


def get_encoded_msa_from_a3m_seqs(
    msa_sequences: list[bytes], alphabet: Uniprot21
) -> npt.NDArray[np.uint8]:
    return np.vstack(
        [
            alphabet.encode(s.translate(None, delete=ASCII_LOWERCASE_BYTES))
            for s in msa_sequences
        ]
    )


def hash_of_list(lst: list[bytes | str]) -> str:
    m = hashlib.sha1()
    for elt in lst:
        if isinstance(elt, str):
            elt = elt.encode("utf-8")
        m.update(elt)
    return m.hexdigest()


def get_numpy_seed(string: str) -> int:
    # Generate a hash of the string using hashlib
    hash_object = hashlib.sha256(string.encode())
    hash_hex = hash_object.hexdigest()

    # Convert the hash to an integer and use it as the seed for numpy
    seed = int(hash_hex, 16) % (2**32 - 1)
    return seed
