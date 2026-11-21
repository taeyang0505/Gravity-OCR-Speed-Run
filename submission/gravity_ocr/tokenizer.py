"""Byte-level BPE vocabulary: encoding, raw byte pieces and decoding.

Encoding (text → ids) is delegated to the ``tokenizers`` library reading the
model's own ``tokenizer.json``.  Everything the runtime does with *output*
tokens works on raw bytes instead: a Korean syllable is three UTF-8 bytes and
the vocabulary is free to split it between tokens, so a piece is only text once
its neighbours are known.  :meth:`Vocab.piece` returns those bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

# ids of the control tokens the runtime needs by name
SPECIAL = {
    "eos": "<|endoftext|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "image_start": "<|begin_of_image|>",
    "image_end": "<|end_of_image|>",
    "image": "<|image|>",
    "mask": "<|mask|>",
}


def _byte_decoder() -> dict[str, int]:
    """The printable stand-in each byte gets in a byte-level BPE vocabulary.

    Bytes that are printable in Latin-1 stand for themselves; the remaining 68
    (controls, space, DEL, 0x80–0xA0, soft hyphen) are numbered from U+0100 in
    byte order.
    """
    keep = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table = {chr(b): b for b in keep}
    extra = 0
    for b in range(256):
        if b not in table.values():
            table[chr(256 + extra)] = b
            extra += 1
    return table


class Vocab:
    def __init__(self, tokenizer_json: str):
        self._tok = Tokenizer.from_str(tokenizer_json)
        spec = json.loads(tokenizer_json)
        vocab: dict[str, int] = spec["model"]["vocab"]
        added = {a["id"]: a["content"] for a in spec.get("added_tokens", [])}
        self.size = max(max(vocab.values()), max(added, default=0)) + 1
        self.base_size = len(vocab)
        dec = _byte_decoder()
        pieces: list[bytes] = [b""] * self.size
        for text, idx in vocab.items():
            pieces[idx] = bytes(dec[ch] for ch in text)
        self.special_ids = set(added)
        self.special_text = added
        for idx, text in added.items():
            pieces[idx] = text.encode("utf-8")
        self._pieces = pieces
        self._ids = {v: k for k, v in added.items()}
        for name, text in SPECIAL.items():
            setattr(self, f"{name}_id", self._ids[text])

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        return cls(Path(path).read_text(encoding="utf-8"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def piece(self, token: int) -> bytes:
        """Raw bytes of one token (control tokens give their literal name)."""
        return self._pieces[token]

    def is_special(self, token: int) -> bool:
        return token in self.special_ids

    def decode_bytes(self, ids, skip_special: bool = True) -> bytes:
        return b"".join(self._pieces[i] for i in ids if not (skip_special and i in self.special_ids))

    def decode(self, ids, skip_special: bool = True) -> str:
        return self.decode_bytes(ids, skip_special).decode("utf-8", errors="replace")
