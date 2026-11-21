"""VQ codebook bypass (plan Part 1.2): simple merge groups skip the vision tower.

A merge group (28×28 px, one image token) is *simple* when it is blank or its ink is
only straight line pixels (pipes, rules, frames: long horizontal / vertical runs) –
no glyph stroke.  The codebook is built on the calibration crops:

* key   – the group's four ``patch_embed`` outputs (4 × 1024), the tower's first layer,
          so a lookup costs one small matrix product and no attention;
* value – the image token the full tower gave the group (24 blocks + merger), averaged
          over the groups of the word.

K-means puts the keys into up to 4,096 words.  A word is *enabled* only when its
groups agreed on their output (relative spread of the values below ``MAX_SPREAD``)
and it has at least ``MIN_MEMBERS`` groups: what a line looks like after the tower
depends on its neighbours, and a word whose groups disagree is never used.  At run
time a simple group whose key lies within its word's radius takes the word's value
instead of going through the tower (``prune.encode(..., level="vq")``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import FACTOR, MERGE

WORDS = 4096
MAX_SPREAD = 0.15
MIN_MEMBERS = 3
LINE_UNITS = 2.2


@dataclass
class Codebook:
    keys: np.ndarray  # [k, 4096] float32
    values: np.ndarray  # [k, 1536] float32
    radius: np.ndarray  # [k] max member distance (keys)
    enabled: np.ndarray  # [k] bool

    def save(self, writer, prefix: str = "vq.") -> None:
        writer.add_array(prefix + "keys", self.keys.astype(np.float16))
        writer.add_array(prefix + "values", self.values.astype(np.float16))
        writer.add_array(prefix + "radius", self.radius.astype(np.float32))
        writer.add_array(prefix + "enabled", self.enabled.astype(np.uint8))

    @classmethod
    def load(cls, src, prefix: str = "vq.") -> "Codebook":
        g = lambda n: np.asarray(src.get(prefix + n))  # noqa: E731
        return cls(g("keys").astype(np.float32), g("values").astype(np.float32), g("radius"), g("enabled").astype(bool))

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Word of every key and whether it is a usable hit."""
        if not len(keys):
            return np.zeros(0, np.int64), np.zeros(0, bool)
        d2 = (keys * keys).sum(1)[:, None] - 2 * keys @ self.keys.T + (self.keys * self.keys).sum(1)[None, :]
        word = d2.argmin(axis=1)
        dist = np.sqrt(np.maximum(d2[np.arange(len(keys)), word], 0))
        return word, self.enabled[word] & (dist <= self.radius[word])


def simple_groups(rgb: np.ndarray) -> np.ndarray:
    """Blank or line-only merge groups (merge-group order) of an on-grid image."""
    from .drawing.ink import Ink, long_runs

    ink = Ink.of_crop(rgb)
    n = max(int(round(LINE_UNITS * ink.unit)), 12)
    line = long_runs(ink.mask, n, 1) | long_runs(ink.mask, n, 0)
    other = ink.mask & ~line
    h, w = ink.mask.shape
    g = other.reshape(h // FACTOR, FACTOR, w // FACTOR, FACTOR).any(axis=(1, 3))
    return ~g.reshape(-1)


def group_keys(model, patches: np.ndarray) -> np.ndarray:
    """``patch_embed`` outputs of every merge group, concatenated (``[groups, 4 · 1024]``)."""
    w = _patch_embed(model)
    x = patches @ w[0].T + w[1]
    per = MERGE * MERGE
    return np.ascontiguousarray(x.reshape(-1, per * x.shape[1]), dtype=np.float32)


def _patch_embed(model):
    cached = getattr(model, "_patch_embed_f32", None)
    if cached is None:
        from .vision import PREFIX, reshape_weight

        wt = reshape_weight("patch_embed.proj.weight", np.asarray(model.src.get(PREFIX + "patch_embed.proj.weight"), np.float32))
        cached = (np.ascontiguousarray(wt), np.asarray(model.src.get(PREFIX + "patch_embed.proj.bias"), np.float32))
        model._patch_embed_f32 = cached
    return cached


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    c = x[rng.choice(len(x), k, replace=False)].copy()
    xx = (x * x).sum(1)[:, None]
    for _ in range(iters):
        d2 = xx - 2 * x @ c.T + (c * c).sum(1)[None, :]
        a = d2.argmin(axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(axis=0)
    d2 = xx - 2 * x @ c.T + (c * c).sum(1)[None, :]
    return c, d2.argmin(axis=1)


def build(keys: np.ndarray, values: np.ndarray, words: int = WORDS) -> Codebook:
    c, a = kmeans(keys, words)
    k = c.shape[0]
    vals = np.zeros((k, values.shape[1]), np.float32)
    radius = np.zeros(k, np.float32)
    enabled = np.zeros(k, bool)
    for j in range(k):
        m = a == j
        if not m.any():
            continue
        v = values[m]
        mean = v.mean(axis=0)
        vals[j] = mean
        radius[j] = float(np.sqrt(((keys[m] - c[j]) ** 2).sum(1)).max())
        spread = float(np.mean(np.linalg.norm(v - mean, axis=1)) / max(np.linalg.norm(mean), 1e-6))
        enabled[j] = m.sum() >= MIN_MEMBERS and spread <= MAX_SPREAD
    return Codebook(c.astype(np.float32), vals, radius, enabled)
