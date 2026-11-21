"""Small numeric helpers shared by the vision tower and the language model."""

from __future__ import annotations

import numpy as np
from scipy.special import erf, expit

F32 = np.float32


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F32) + F32(eps))
    return x * scale * weight


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = np.mean(x, axis=-1, keepdims=True, dtype=F32)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True, dtype=F32)
    return xc / np.sqrt(var + F32(eps)) * weight + bias


def silu(x: np.ndarray) -> np.ndarray:
    return x * expit(x)


def gelu(x: np.ndarray) -> np.ndarray:
    return (x * F32(0.5) * (F32(1.0) + erf(x * F32(0.7071067811865476)))).astype(F32, copy=False)


def softmax_inplace(x: np.ndarray) -> np.ndarray:
    x -= np.max(x, axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= np.sum(x, axis=-1, keepdims=True)
    return x


def top_k(logits: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Indices and softmax probabilities of the ``k`` largest logits (ties → lowest index)."""
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.lexsort((idx, -logits[idx]))]
    top = logits[idx[0]]
    z = np.sum(np.exp((logits - top).astype(np.float64)))
    return idx, (np.exp((logits[idx] - top).astype(np.float64)) / z).astype(F32)
