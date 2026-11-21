"""Block-wise weight quantisation codecs (numpy only).

Every scheme works on a flat float32 vector that is cut into fixed blocks; the
row structure of the matrix is ignored (rows of every tensor in this model are
multiples of 256 wide, so blocks never straddle a row in a harmful way).

Schemes
-------
``f32``   raw float32                                        32      bpw
``f16``   raw float16                                        16      bpw
``q8``    symmetric  8-bit, block 32, f16 scale               8.5    bpw
``q6``    symmetric  6-bit, block 16, 8-bit scale / sb f16    6.5625 bpw
``q5``    asymmetric 5-bit, block 32, 6-bit scale+min / sb    5.5    bpw
``q4``    asymmetric 4-bit, block 32, 6-bit scale+min / sb    4.5    bpw
``q3``    symmetric  3-bit, block 16, 6-bit scale / sb f16    3.4375 bpw
``nf4``   normal-float 4-bit codebook, block 64, f16 absmax   4.25   bpw

"sb" is a super-block of 256 weights holding one f16 scale (and one f16 min
scale for the asymmetric schemes); the per-block scales are stored relative to
it.  The layouts follow the well known "k-quant" idea, but the code here is an
independent numpy implementation with its own container.

The scale search is a weighted least-squares grid search: for a handful of
candidate step sizes the weights are rounded, the optimal affine map for that
rounding is solved in closed form, and the candidate with the smallest weighted
error wins.  Importance weights are ``rms(block) + |x|`` so large weights (which
dominate the matmul output) are reproduced more faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPER = 256

# Quantiles of N(0,1) normalised to [-1, 1] as defined in the QLoRA paper
# (Dettmers et al., 2023, "NormalFloat").
NF4_LEVELS = np.array(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Scheme:
    name: str
    bits: int
    block: int
    symmetric: bool
    scale_bits: int  # bits of the per-block scale (0 → f16 scale stored directly)

    @property
    def bpw(self) -> float:
        if self.name == "f32":
            return 32.0
        if self.name == "f16":
            return 16.0
        if self.name == "q8":
            return 8 + 16 / 32
        if self.name == "nf4":
            return 4 + 16 / 64
        per_block = self.scale_bits * (1 if self.symmetric else 2)
        per_super = 16 * (1 if self.symmetric else 2)
        return self.bits + per_block / self.block + per_super / SUPER


SCHEMES: dict[str, Scheme] = {
    "f32": Scheme("f32", 32, 1, True, 0),
    "f16": Scheme("f16", 16, 1, True, 0),
    "q8": Scheme("q8", 8, 32, True, 0),
    "q6": Scheme("q6", 6, 16, True, 8),
    "q5": Scheme("q5", 5, 32, False, 6),
    "q4": Scheme("q4", 4, 32, False, 6),
    "q3": Scheme("q3", 3, 16, True, 6),
    "nf4": Scheme("nf4", 4, 64, True, 0),
}

# From most to least precise; used by the allocation search.
LADDER = ["f16", "q8", "q6", "q5", "q4", "q3"]


# --------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------
def pack_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    """Pack unsigned integers ``< 2**nbits`` into a little-endian bit stream."""
    v = np.ascontiguousarray(vals, dtype=np.uint8).reshape(-1)
    if nbits == 8:
        return v
    if nbits == 4:
        if v.size % 2:
            v = np.concatenate([v, np.zeros(1, np.uint8)])
        return (v[0::2] | (v[1::2] << 4)).astype(np.uint8)
    bits = np.unpackbits(v[:, None], axis=1, bitorder="little")[:, :nbits]
    return np.packbits(bits.reshape(-1), bitorder="little")


def unpack_bits(buf: np.ndarray, nbits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`; returns ``count`` uint8 values."""
    b = np.asarray(buf, dtype=np.uint8).reshape(-1)
    if nbits == 8:
        return b[:count]
    if nbits == 4:
        out = np.empty(b.size * 2, np.uint8)
        out[0::2] = b & 15
        out[1::2] = b >> 4
        return out[:count]
    if nbits == 6 and count % 4 == 0:
        t = b[: count // 4 * 3].reshape(-1, 3).astype(np.uint32)
        w = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16)
        out = np.empty((w.size, 4), np.uint8)
        for i in range(4):
            out[:, i] = (w >> (6 * i)) & 63
        return out.reshape(-1)
    bits = np.unpackbits(b, bitorder="little")[: count * nbits].reshape(count, nbits)
    out = np.zeros(count, np.uint8)
    for i in range(nbits):
        out |= bits[:, i] << i
    return out


# --------------------------------------------------------------------------
# scale search
# --------------------------------------------------------------------------
def _blocks(x: np.ndarray, block: int) -> tuple[np.ndarray, int]:
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    n = flat.size
    pad = (-n) % SUPER
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    return flat.reshape(-1, block), n


def _importance(xb: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(xb * xb, axis=1, keepdims=True))
    return rms + np.abs(xb)


def _fit_symmetric(xb: np.ndarray, w: np.ndarray, nmax: int, steps=range(-9, 10)):
    """Per block: scale ``d`` minimising Σ w (d·q − x)², q ∈ [−nmax−1, nmax]."""
    amax = np.max(np.abs(xb), axis=1, keepdims=True)
    safe = np.where(amax > 0, amax, 1.0)
    best_err = np.full((xb.shape[0], 1), np.inf, np.float32)
    best_d = np.zeros((xb.shape[0], 1), np.float32)
    for s in steps:
        inv = (nmax + 0.1 * s) / safe
        q = np.clip(np.rint(xb * inv), -nmax - 1, nmax)
        num = np.sum(w * xb * q, axis=1, keepdims=True)
        den = np.sum(w * q * q, axis=1, keepdims=True)
        d = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).astype(np.float32)
        err = np.sum(w * (d * q - xb) ** 2, axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    return best_d


def _fit_asymmetric(xb: np.ndarray, w: np.ndarray, nmax: int, steps: int = 20):
    """Per block: ``d``, ``m ≥ 0`` minimising Σ w (d·q − m − x)², q ∈ [0, nmax]."""
    lo = np.minimum(np.min(xb, axis=1, keepdims=True), 0.0)
    hi = np.max(xb, axis=1, keepdims=True)
    rng = hi - lo
    safe = np.where(rng > 0, rng, 1.0)
    sum_w = np.sum(w, axis=1, keepdims=True)
    sum_x = np.sum(w * xb, axis=1, keepdims=True)
    best_err = np.full((xb.shape[0], 1), np.inf, np.float32)
    best_d = np.zeros((xb.shape[0], 1), np.float32)
    best_m = np.zeros((xb.shape[0], 1), np.float32)
    for s in range(steps + 1):
        inv = (nmax - 1.0 + 0.1 * s) / safe
        q = np.clip(np.rint((xb - lo) * inv), 0, nmax)
        sum_l = np.sum(w * q, axis=1, keepdims=True)
        sum_l2 = np.sum(w * q * q, axis=1, keepdims=True)
        sum_xl = np.sum(w * xb * q, axis=1, keepdims=True)
        det = sum_w * sum_l2 - sum_l * sum_l
        ok = det > 0
        dsafe = np.where(ok, det, 1.0)
        d = np.where(ok, (sum_w * sum_xl - sum_x * sum_l) / dsafe, 0.0)
        off = np.where(ok, (sum_l2 * sum_x - sum_l * sum_xl) / dsafe, 0.0)
        # the offset must stay ≤ 0 so that it is stored as a magnitude
        pos = off > 0
        d = np.where(pos, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), d)
        off = np.where(pos, 0.0, off)
        err = np.sum(w * (d * q + off - xb) ** 2, axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d).astype(np.float32)
        best_m = np.where(better, -off, best_m).astype(np.float32)
    return best_d, best_m


def _to_super(v: np.ndarray, per_super: int, levels: int):
    """Quantise per-block positive scalars relative to the super-block maximum."""
    vs = v.reshape(-1, per_super)
    top = np.max(vs, axis=1, keepdims=True).astype(np.float16).astype(np.float32)
    # f16 rounding may fall below the true maximum; nudge up so ratios stay ≤ 1
    low = top < np.max(vs, axis=1, keepdims=True)
    top = np.where(low, np.nextafter(top.astype(np.float16), np.float16(np.inf)).astype(np.float32), top)
    safe = np.where(top > 0, top, 1.0)
    idx = np.clip(np.rint(vs / safe * levels), 0, levels).astype(np.uint8)
    return top.astype(np.float16).reshape(-1), idx.reshape(-1)


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------
CHUNK = 1 << 20  # values encoded at a time (a multiple of SUPER)


def quantize(x: np.ndarray, scheme: str, chunk: int = CHUNK) -> dict[str, np.ndarray]:
    """Encode ``x``; returns named uint8/float16 parts for the container.

    Large tensors are encoded ``chunk`` values at a time: every block and every
    super-block is fitted on its own and every packed stream is byte aligned at
    a super-block, so the pieces concatenate to exactly the whole-tensor result
    while the scratch memory stays bounded.
    """
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if scheme in ("f32", "f16") or flat.size <= chunk:
        return _quantize(flat if scheme not in ("f32", "f16") else x, scheme)
    chunk -= chunk % SUPER
    pieces = [_quantize(flat[i : i + chunk], scheme) for i in range(0, flat.size, chunk)]
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def _quantize(x: np.ndarray, scheme: str) -> dict[str, np.ndarray]:
    sc = SCHEMES[scheme]
    if scheme == "f32":
        return {"w": np.ascontiguousarray(x, dtype=np.float32)}
    if scheme == "f16":
        return {"w": np.ascontiguousarray(x, dtype=np.float32).astype(np.float16)}

    xb, _ = _blocks(x, sc.block)
    if scheme == "nf4":
        amax = np.max(np.abs(xb), axis=1, keepdims=True).astype(np.float16)
        safe = np.where(amax > 0, amax, 1.0).astype(np.float32)
        norm = xb / safe
        mid = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2
        idx = np.searchsorted(mid, norm.reshape(-1)).astype(np.uint8)
        return {"q": pack_bits(idx, 4), "d": amax.reshape(-1)}

    w = _importance(xb)
    if scheme == "q8":
        d = _fit_symmetric(xb, w, 127).astype(np.float16)
        df = d.astype(np.float32)
        q = np.clip(np.rint(xb / np.where(df != 0, df, 1.0)), -128, 127).astype(np.int8)
        return {"q": q.reshape(-1).view(np.uint8), "d": d.reshape(-1)}

    nmax = (1 << (sc.bits - 1)) - 1 if sc.symmetric else (1 << sc.bits) - 1
    levels = (1 << sc.scale_bits) - 1
    per_super = SUPER // sc.block
    if sc.symmetric:
        # the rounding keeps the sign of x, so the least-squares scale is ≥ 0
        d = _fit_symmetric(xb, w, nmax)
        top, idx = _to_super(d, per_super, levels)
        dq = np.repeat(top.astype(np.float32), per_super)[:, None] * (idx[:, None] / levels)
        q = np.clip(np.rint(xb / np.where(dq != 0, dq, 1.0)), -nmax - 1, nmax)
        q = np.where(dq != 0, q, 0)
        return {
            "q": pack_bits((q + nmax + 1).astype(np.uint8), sc.bits),
            "s": pack_bits(idx, sc.scale_bits),
            "d": top,
        }
    d, m = _fit_asymmetric(xb, w, nmax)
    dtop, didx = _to_super(d, per_super, levels)
    mtop, midx = _to_super(m, per_super, levels)
    dq = np.repeat(dtop.astype(np.float32), per_super)[:, None] * (didx[:, None] / levels)
    mq = np.repeat(mtop.astype(np.float32), per_super)[:, None] * (midx[:, None] / levels)
    q = np.clip(np.rint((xb + mq) / np.where(dq != 0, dq, 1.0)), 0, nmax)
    q = np.where(dq != 0, q, 0)
    return {
        "q": pack_bits(q.astype(np.uint8), sc.bits),
        "s": pack_bits(didx, sc.scale_bits),
        "m": pack_bits(midx, sc.scale_bits),
        "d": dtop,
        "dm": mtop,
    }


def dequantize(parts: dict[str, np.ndarray], scheme: str, shape) -> np.ndarray:
    """Decode parts written by :func:`quantize` back to float32 of ``shape``.

    Runs in the native decoder when it is loaded (checked bit for bit against
    :func:`dequantize_np` at load time).
    """
    from . import native

    k = native.load()
    if k is not None and k.has("codec") and scheme in native.SCHEME_IDS and len(shape) >= 1:
        n = int(np.prod(shape))
        rows = int(shape[0]) if len(shape) >= 2 else 1
        lin = native.Linear(parts, scheme, (rows, n // max(rows, 1)))
        return k.dequant(lin).reshape(shape)
    return dequantize_np(parts, scheme, shape)


def dequantize_np(parts: dict[str, np.ndarray], scheme: str, shape) -> np.ndarray:
    sc = SCHEMES[scheme]
    n = int(np.prod(shape))
    if scheme in ("f32", "f16"):
        return np.asarray(parts["w"]).astype(np.float32).reshape(shape)
    padded = n + ((-n) % SUPER)
    nblk = padded // sc.block
    if scheme == "nf4":
        idx = unpack_bits(parts["q"], 4, padded)
        out = NF4_LEVELS[idx].reshape(nblk, sc.block)
        out = out * np.asarray(parts["d"]).astype(np.float32)[:, None]
        return out.reshape(-1)[:n].reshape(shape)
    if scheme == "q8":
        q = np.asarray(parts["q"]).view(np.int8).astype(np.float32).reshape(nblk, sc.block)
        out = q * np.asarray(parts["d"]).astype(np.float32)[:, None]
        return out.reshape(-1)[:n].reshape(shape)

    levels = (1 << sc.scale_bits) - 1
    per_super = SUPER // sc.block
    q = unpack_bits(parts["q"], sc.bits, padded).astype(np.float32).reshape(nblk, sc.block)
    dtop = np.repeat(np.asarray(parts["d"]).astype(np.float32), per_super)
    dq = dtop * (unpack_bits(parts["s"], sc.scale_bits, nblk).astype(np.float32) / levels)
    if sc.symmetric:
        nmax = (1 << (sc.bits - 1)) - 1
        out = (q - (nmax + 1)) * dq[:, None]
    else:
        mtop = np.repeat(np.asarray(parts["dm"]).astype(np.float32), per_super)
        mq = mtop * (unpack_bits(parts["m"], sc.scale_bits, nblk).astype(np.float32) / levels)
        out = q * dq[:, None] - mq[:, None]
    return out.reshape(-1)[:n].reshape(shape)


def roundtrip(x: np.ndarray, scheme: str) -> np.ndarray:
    """Quantise then dequantise (what the runtime will actually compute with)."""
    if scheme == "f32":
        return np.asarray(x, dtype=np.float32)
    return dequantize(quantize(x, scheme), scheme, x.shape)


def packed_bytes(parts: dict[str, np.ndarray]) -> int:
    return int(sum(np.asarray(p).nbytes for p in parts.values()))
