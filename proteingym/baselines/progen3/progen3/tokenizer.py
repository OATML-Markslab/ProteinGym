import os

from tokenizers import Tokenizer

END_OF_SPAN_TOKEN = "<eos_span>"  # nosec
PAD_TOKEN_ID = 0


def get_tokenizer() -> Tokenizer:
    fname = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(fname)
    assert (
        tokenizer.padding["pad_id"] == PAD_TOKEN_ID
    ), f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"

    return tokenizer
