"""UnifiedNgram (plan Part 5): token trigrams baked from the model's own layer 0, and the
deterministic draft look-ahead.

**Table.**  Domain seed trigrams ``(t₋₂, t₋₁, t₀)`` – equipment codes of both schemes,
units and numbers, the table markup the model writes, the Korean field labels of the
forms – are run through the input embedding and layer 0; the 1536-wide responses at the
third position are stacked, their top 128 right singular vectors form ``synth_proj``
``[128, 1536]`` and every trigram keeps its 128 coefficients in INT8 (one scale per
row).  Rows live in 65,536 buckets; a bucket collision keeps the row with the larger
energy ``Σ z²``.  Every row carries a 64-bit ``key_tag`` of the trigram computed with a
hash independent of the bucket index, and a lookup returns a residual only when the tag
matches exactly – an unrelated trigram that lands in the same bucket gets nothing.  The
residual ``gate · z · (synth_proj + learned_residual_proj)`` is added to the hidden
state after layer 0; ``learned_residual_proj`` starts at zero (a slot for a later
fine-tune with the model frozen).

**Look-ahead.**  A table of ``context → continuation`` built from the same seeds feeds
the speculative decoder (``decode.Lookahead``); the verification pass keeps a proposed
token only when the model would have produced it, so it can only change the number of
passes, never the text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import decode, language
from .drawing import codes

BUCKETS = 1 << 16
RANK = 128
GATE = 0.08
_M64 = (1 << 64) - 1


def _mix(x: int) -> int:
    """splitmix64 finaliser."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def pack(a: int, b: int, c: int) -> int:
    return (a << 34) | (b << 17) | c  # token ids are < 2**17


def bucket_of(a: int, b: int, c: int) -> int:
    return _mix(pack(a, b, c)) & (BUCKETS - 1)


def tag_of(a: int, b: int, c: int) -> int:
    """Independent 64-bit check value of the trigram (never 0: 0 marks an empty row)."""
    return _mix(pack(a, b, c) ^ 0xD6E8FEB86659FD93) or 1


def seed_texts(ledger_codes=(), extra=()) -> list[str]:
    """Strings whose trigrams are baked: codes, units, numbers, table markup, form labels."""
    from .surgery import domain_corpus

    labels = ["설비코드", "센서태그", "항목", "정상치", "경보한계", "트립한계", "점검자", "사번", "센서 태그", "현장 실측치",
              "단위", "점검 일시", "위치: ", "도면번호: ", "기술책임자: ", "검토일 ", "KKS: ", "자산코드: ", "(EL.+", "(EL.-"]  # fmt: skip
    rows = ["<table><tr><td>설비코드</td><td>센서태그</td><td>항목</td><td>정상치</td><td>경보한계</td><td>트립한계</td></tr>",
            "</td></tr><tr><td>", "</td><td>", "</td></tr></table>"]  # fmt: skip
    return list(extra) + labels + rows + list(ledger_codes) + domain_corpus()


@dataclass
class NgramTable:
    proj: np.ndarray  # [RANK, 1536] float32 (synth_proj)
    learned: np.ndarray  # [RANK, 1536] float32, zero (learned_residual_proj)
    coef: np.ndarray  # [BUCKETS, RANK] int8
    scale: np.ndarray  # [BUCKETS] float32
    tags: np.ndarray  # [BUCKETS] uint64, 0 = empty
    gate: float = GATE

    @property
    def nbytes(self) -> int:
        return int(self.coef.nbytes + self.scale.nbytes + self.tags.nbytes + self.proj.nbytes + self.learned.nbytes)

    @property
    def rows(self) -> int:
        return int((self.tags != 0).sum())

    def lookup(self, a: int, b: int, c: int) -> np.ndarray | None:
        i = bucket_of(a, b, c)
        if int(self.tags[i]) != tag_of(a, b, c):
            return None
        z = self.coef[i].astype(np.float32) * self.scale[i]
        return self.gate * (z @ (self.proj + self.learned))

    def residual(self, history: list[int], tokens: list[int]) -> np.ndarray | None:
        """Rows to add after layer 0 for ``tokens`` following ``history`` (``None``: nothing matched)."""
        seq = list(history[-2:]) + list(tokens)
        out = None
        for j in range(len(tokens)):
            k = j + len(seq) - len(tokens)
            if k < 2:
                continue
            r = self.lookup(seq[k - 2], seq[k - 1], seq[k])
            if r is not None:
                if out is None:
                    out = np.zeros((len(tokens), self.proj.shape[1]), np.float32)
                out[j] = r
        return out

    def save(self, writer, prefix: str = "ngram.") -> None:
        writer.add_array(prefix + "synth_proj", self.proj)
        writer.add_array(prefix + "learned_residual_proj", self.learned)
        writer.add_array(prefix + "coef", self.coef)
        writer.add_array(prefix + "scale", self.scale)
        writer.add_array(prefix + "tags", self.tags)

    @classmethod
    def load(cls, src, prefix: str = "ngram.", gate: float = GATE) -> "NgramTable":
        g = lambda n: np.asarray(src.get(prefix + n))  # noqa: E731 - memory-mapped views
        return cls(g("synth_proj"), g("learned_residual_proj"), g("coef"), g("scale"), g("tags"), gate)


def trigrams(vocab, texts) -> list[tuple[int, int, int]]:
    seen = {}
    for t in texts:
        ids = vocab.encode(t)
        for i in range(2, len(ids)):
            seen.setdefault((ids[i - 2], ids[i - 1], ids[i]), None)
    return list(seen)


def layer0_responses(model, grams: list[tuple[int, int, int]], batch: int = 256) -> np.ndarray:
    """Hidden state after layer 0 at the third position of each trigram (CPU path)."""
    lm = model.lm
    out = np.empty((len(grams), language.WIDTH), np.float32)
    pos = np.repeat(np.arange(3), 3).reshape(3, 3)
    for i, g in enumerate(grams):
        x = model.embed(list(g))
        hidden: list = []
        cache = lm.new_cache(8)
        # one layer is enough: stop after collecting the input of layer 1
        hook_state = {"x": None}

        def hook(layer, h, _s=hook_state):
            if layer == 1:
                _s["x"] = np.array(h[-1], np.float32)
                raise _Stop
            return h

        lm.layer_hook = hook
        try:
            lm.forward(x, pos, cache, hidden=hidden)
        except _Stop:
            pass
        finally:
            lm.layer_hook = None
        out[i] = hook_state["x"]
    return out


class _Stop(Exception):
    pass


def build(model, texts: list[str], rank: int = RANK) -> NgramTable:
    """Bake the table from the trigrams of ``texts`` (model on a CPU device)."""
    grams = trigrams(model.vocab, texts)
    resp = layer0_responses(model, grams)
    _, _, vt = np.linalg.svd(resp.astype(np.float64), full_matrices=False)
    proj = vt[:rank].astype(np.float32)
    z = resp @ proj.T
    energy = (z.astype(np.float64) ** 2).sum(axis=1)
    coef = np.zeros((BUCKETS, rank), np.int8)
    scale = np.zeros(BUCKETS, np.float32)
    tags = np.zeros(BUCKETS, np.uint64)
    best = np.full(BUCKETS, -1.0)
    for (a, b, c), zi, e in zip(grams, z, energy):
        i = bucket_of(a, b, c)
        if e <= best[i]:
            continue  # the bucket keeps the row with more energy
        s = float(np.abs(zi).max()) / 127 or 1.0
        coef[i] = np.clip(np.rint(zi / s), -127, 127).astype(np.int8)
        scale[i] = s
        tags[i] = tag_of(a, b, c)
        best[i] = e
    return NgramTable(proj, np.zeros_like(proj), coef, scale, tags)


def lookahead(vocab, texts: list[str], order: int = 2, max_len: int = 6) -> decode.Lookahead:
    """Deterministic continuations (a context seen with two different continuations keeps their common prefix)."""
    la = decode.Lookahead(order, max_len)
    for t in texts:
        la.add(vocab.encode(t))
    return la


def code_texts(ledger_codes) -> list[str]:
    """How codes appear in outputs: alone, after their label and inside table cells."""
    out = []
    for c in ledger_codes:
        sc = codes.scheme_of(c)
        out += [c, f"{(sc or codes.KKS).label}: {c}", f"<td>{c}</td>"]
    return out
