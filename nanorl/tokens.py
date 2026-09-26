"""Turning text into numbers and back, without downloading a tokenizer.

nano-family models speak bytes: every character of any text becomes one of its
UTF-8 byte values (0-255), plus three special numbers — one for padding, one
that says "the answer starts here", one that says "it ends here". That is 259
numbers in total, so *any* text on Earth round-trips through the model with
nothing to install and nothing to train first.

A trained BPE tokenizer (as in nanobrain) compresses better; byte-level is the
honest floor that never fails on a character nobody predicted.
"""

from __future__ import annotations

import torch

PAD = 256  # filler for short examples; the loss never looks at it
BOS = 257  # begin-of-sequence
EOS = 258  # end-of-sequence
VOCAB_SIZE = 259  # 256 byte values + the three specials

_SPECIALS = {PAD, BOS, EOS}


def encode(text: str) -> list[int]:
    """Text -> [BOS, byte, byte, ..., EOS]. Never fails on any character."""
    return [BOS] + list(text.encode("utf-8")) + [EOS]


def decode(ids) -> str:
    """Token ids -> text, dropping the specials; bad bytes become a marker."""
    data = bytes(i for i in ids if i not in _SPECIALS)
    return data.decode("utf-8", errors="replace")


def encode_to_seq_len(text: str, seq_len: int) -> tuple[list[int], list[int]]:
    """One example, cut or padded to ``seq_len``, with its mask.

    Returns (token ids, mask) where the mask is 1 on real tokens and 0 on
    padding. An empty text still gets BOS and EOS so no example is empty.
    """
    seq = encode(text)
    real = len(seq)
    if real > seq_len:
        seq = seq[:seq_len]
        seq[-1] = EOS  # never leave an answer without its ending
        real = seq_len
    return seq + [PAD] * (seq_len - real), [1] * real + [0] * (seq_len - real)


def pad_batch(texts, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A list of texts -> (token ids [batch, seq_len], mask [batch, seq_len])."""
    rows, masks = zip(*(encode_to_seq_len(t, seq_len) for t in texts))
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )
