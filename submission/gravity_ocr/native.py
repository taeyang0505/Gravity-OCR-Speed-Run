"""Rust CPU kernels (``rust_kernels/``), called through ``ctypes`` without copies.

Every array goes to the library as the address of its numpy buffer; packed
weights go as addresses into the memory-mapped package, so a quantised tensor
is multiplied where it lies and is never expanded to float32.

The library is optional.  :func:`load` finds it, checks every kernel once
against the numpy code it replaces (the decoders bit for bit, everything else
within ``TOL``) and only then hands it out; a kernel that disagrees is left out
and its caller keeps the numpy path.  ``GRAVITY_OCR_NATIVE=0`` turns the
library off, ``GRAVITY_OCR_KERNELS=<path>`` points to a build elsewhere.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import quant

F32 = np.float32
TOL = 1e-3
ABI = 2  # rust_kernels/src/lib.rs VERSION: the exported signatures this module binds

SCHEME_IDS = {"f32": 0, "f16": 1, "bf16": 2, "q8": 3, "q6": 4, "q5": 5, "q4": 6, "q3": 7, "nf4": 8}
ERRORS = {-1: "unknown scheme", -2: "missing part", -3: "part too short", -4: "shape not supported", -5: "cache too small"}

_c = ctypes
_P = _c.c_void_p
_Z = _c.c_size_t


class QTensor(_c.Structure):
    _fields_ = [
        ("scheme", _c.c_uint32), ("_pad", _c.c_uint32), ("rows", _c.c_uint64), ("cols", _c.c_uint64),
        ("q", _P), ("q_len", _c.c_uint64), ("d", _P), ("d_len", _c.c_uint64),
        ("s", _P), ("s_len", _c.c_uint64), ("m", _P), ("m_len", _c.c_uint64),
        ("dm", _P), ("dm_len", _c.c_uint64),
    ]  # fmt: skip


class NativeError(RuntimeError):
    pass


def _addr(a) -> int | None:
    return None if a is None else a.ctypes.data


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=F32)


def _inplace(a, dtype=F32) -> np.ndarray:
    """An array the kernel writes into: it must already be contiguous of ``dtype``."""
    if not (isinstance(a, np.ndarray) and a.dtype == dtype and a.flags.c_contiguous and a.flags.writeable):
        raise ValueError("in-place argument must be a writeable C-contiguous array of the right dtype")
    return a


class Linear:
    """A weight matrix as packed parts plus the descriptor the library reads."""

    __slots__ = ("parts", "scheme", "rows", "cols", "struct", "addr")

    def __init__(self, parts: dict, scheme: str, shape):
        rows, cols = int(shape[0]), int(np.prod(shape[1:]))
        self.parts = {k: np.ascontiguousarray(v) for k, v in parts.items()}  # keeps the buffers alive
        self.scheme, self.rows, self.cols = scheme, rows, cols
        st = QTensor(scheme=SCHEME_IDS[scheme], rows=rows, cols=cols)
        main = self.parts.get("q", self.parts.get("w"))
        st.q, st.q_len = _addr(main), main.nbytes
        for key in ("d", "s", "m", "dm"):
            if key in self.parts:
                setattr(st, key, _addr(self.parts[key]))
                setattr(st, key + "_len", self.parts[key].nbytes)
        self.struct = st
        self.addr = _c.addressof(st)

    @classmethod
    def dense(cls, w: np.ndarray) -> "Linear":
        w = _f32(w)
        return cls({"w": w}, "f32", w.shape)

    @property
    def nbytes(self) -> int:
        return int(sum(p.nbytes for p in self.parts.values()))


def parts_of(src, name: str) -> tuple[dict, str, tuple]:
    """Packed parts of a tensor in a package, or the raw BF16/F16/F32 of a safetensors file."""
    scheme = src.scheme(name)
    if hasattr(src, "parts"):
        return src.parts(name), scheme, src.shape(name)
    raise TypeError(f"{type(src).__name__} cannot hand out packed parts")


class Kernels:
    """The loaded library; ``ok`` holds the kernels that passed the self-check."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lib = lib = _c.CDLL(str(self.path))
        lib.gk_version.restype, lib.gk_version.argtypes = _c.c_uint32, []
        if lib.gk_version() != ABI:
            raise OSError(f"library ABI {lib.gk_version()}, this runtime needs {ABI} (rebuild rust_kernels)")
        sig = {
            "gk_features": (_c.c_uint32, []),
            "gk_set_threads": (_c.c_uint32, [_c.c_uint32]),
            "gk_threads": (_c.c_uint32, []),
            "gk_set_scalar": (None, [_c.c_int32]),
            "gk_check": (_c.c_int32, [_P]),
            "gk_dequant": (_c.c_int32, [_P, _P]),
            "gk_qmatmul": (_c.c_int32, [_P, _P, _Z, _P, _P, _c.c_uint32]),
            "gk_rms_norm": (None, [_P, _Z, _Z, _P, _c.c_float, _P]),
            "gk_rms_norm_add": (None, [_P, _P, _Z, _Z, _P, _c.c_float]),
            "gk_layer_norm": (None, [_P, _Z, _Z, _P, _P, _c.c_float, _P]),
            "gk_swiglu_split": (None, [_P, _Z, _Z, _P]),
            "gk_silu_mul": (None, [_P, _P, _Z, _P]),
            "gk_lm_attention": (_c.c_int32, [_P, _P, _P, _P, _P, _P, _P, _Z, _Z, _Z, _c.c_int32, _P]),
            "gk_vis_attention": (_c.c_int32, [_P, _P, _P, _c.c_float, _P, _P, _Z, _P]),
            "gk_gray_chroma": (None, [_P, _Z, _P, _P, _P]),
            "gk_threshold": (None, [_P, _Z, _c.c_uint8, _c.c_int32, _P]),
            "gk_dilate": (None, [_P, _Z, _Z, _Z, _P]),
            "gk_long_runs": (None, [_P, _Z, _Z, _Z, _c.c_uint32, _P]),
            "gk_bridges": (None, [_P, _P, _Z, _Z, _c.c_uint32, _Z, _P]),
            "gk_damp": (None, [_P, _P, _Z, _c.c_uint8]),
            "gk_label": (_c.c_int64, [_P, _Z, _Z, _P, _P, _Z, _P]),
            "gk_select": (None, [_P, _Z, _P, _Z, _P]),
            "gk_patchify": (None, [_P, _Z, _Z, _c.c_float, _P, _P, _P]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        self.version = lib.gk_version()
        self.features = lib.gk_features()
        self.ok: set[str] = set()
        self.report: dict[str, str] = {}

    def has(self, *names: str) -> bool:
        return all(n in self.ok for n in names)

    # -- threads -----------------------------------------------------------
    def set_threads(self, n: int = 0) -> int:
        return int(self.lib.gk_set_threads(n))

    @property
    def threads(self) -> int:
        return int(self.lib.gk_threads())

    # -- weights -----------------------------------------------------------
    def dequant(self, lin: Linear) -> np.ndarray:
        out = np.empty((lin.rows, lin.cols), F32)
        rc = self.lib.gk_dequant(lin.addr, out.ctypes.data)
        if rc:
            raise NativeError(ERRORS.get(rc, rc))
        return out

    def linear(self, lin: Linear, x: np.ndarray, bias=None, out=None, accumulate: bool = False) -> np.ndarray:
        """``x · Wᵀ + bias``; with ``accumulate`` the product is added to ``out``."""
        x = _f32(x)
        n = x.shape[0]
        if out is None:
            if accumulate:
                raise ValueError("accumulate needs an output")
            out = np.empty((n, lin.rows), F32)
        _inplace(out)
        bias = None if bias is None else _f32(bias)
        rc = self.lib.gk_qmatmul(lin.addr, x.ctypes.data, n, out.ctypes.data, _addr(bias), 1 if accumulate else 0)
        if rc:
            raise NativeError(f"qmatmul: {ERRORS.get(rc, rc)}")
        return out

    # -- row operations ----------------------------------------------------
    def rms_norm(self, x, weight, eps=1e-5, out=None):
        x = _f32(x)
        d = x.shape[-1]
        out = np.empty_like(x) if out is None else _inplace(out)
        self.lib.gk_rms_norm(x.ctypes.data, x.size // d, d, _f32(weight).ctypes.data, eps, out.ctypes.data)
        return out

    def rms_norm_add(self, x, y, weight, eps=1e-5):
        """``x += rms_norm(y, weight)`` in place."""
        d = _inplace(x).shape[-1]
        self.lib.gk_rms_norm_add(x.ctypes.data, _f32(y).ctypes.data, x.size // d, d, _f32(weight).ctypes.data, eps)
        return x

    def layer_norm(self, x, weight, bias, eps=1e-5, out=None):
        x = _f32(x)
        d = x.shape[-1]
        out = np.empty_like(x) if out is None else _inplace(out)
        w, b = _f32(weight), _f32(bias)
        self.lib.gk_layer_norm(x.ctypes.data, x.size // d, d, w.ctypes.data, b.ctypes.data, eps, out.ctypes.data)
        return out

    def swiglu_split(self, u, inter, out=None):
        u = _f32(u)
        n = u.shape[0]
        out = np.empty((n, inter), F32) if out is None else _inplace(out)
        self.lib.gk_swiglu_split(u.ctypes.data, n, inter, out.ctypes.data)
        return out

    def silu_mul(self, gate, up, out=None):
        gate, up = _f32(gate), _f32(up)
        out = np.empty_like(gate) if out is None else _inplace(out)
        self.lib.gk_silu_mul(gate.ctypes.data, up.ctypes.data, gate.size, out.ctypes.data)
        return out

    # -- attention ---------------------------------------------------------
    def lm_attention(self, q, k, v, cos, sin, kt, vc, past, causal):
        """q ``[t, 2048]``, k / v ``[t, 1024]`` (q, k rotated in place); caches of one layer."""
        t = _inplace(q).shape[0]
        _inplace(k), _inplace(kt), _inplace(vc)
        v, cos, sin = _f32(v), _f32(cos), _f32(sin)
        out = np.empty((t, q.shape[1]), F32)
        rc = self.lib.gk_lm_attention(
            q.ctypes.data, k.ctypes.data, v.ctypes.data, cos.ctypes.data, sin.ctypes.data,
            kt.ctypes.data, vc.ctypes.data, vc.shape[1], past, t, 1 if causal else 0, out.ctypes.data,
        )  # fmt: skip
        if rc:
            raise NativeError(f"lm_attention: {ERRORS.get(rc, rc)}")
        return out

    def vis_attention(self, qkv, q_norm, k_norm, cos, sin, eps=1e-5):
        """``qkv`` ``[n, 3072]`` (q, k normalised and rotated in place) → ``[n, 1024]``."""
        n = _inplace(qkv).shape[0]
        q_norm, k_norm, cos, sin = _f32(q_norm), _f32(k_norm), _f32(cos), _f32(sin)
        out = np.empty((n, qkv.shape[1] // 3), F32)
        rc = self.lib.gk_vis_attention(
            qkv.ctypes.data, q_norm.ctypes.data, k_norm.ctypes.data, eps, cos.ctypes.data, sin.ctypes.data, n, out.ctypes.data
        )
        if rc:
            raise NativeError(f"vis_attention: {ERRORS.get(rc, rc)}")
        return out

    # -- pixels ------------------------------------------------------------
    def gray_chroma(self, rgb, chroma: bool = True):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        gray = np.empty((h, w), np.uint8)
        ch = np.empty((h, w), np.uint8) if chroma else None
        hist = np.empty(256, np.uint64)
        self.lib.gk_gray_chroma(rgb.ctypes.data, h * w, gray.ctypes.data, _addr(ch), hist.ctypes.data)
        return gray, ch, hist

    def below(self, src, thr: int) -> np.ndarray:
        src = np.ascontiguousarray(src, dtype=np.uint8)
        out = np.empty(src.shape, bool)
        self.lib.gk_threshold(src.ctypes.data, src.size, int(thr), 0, out.ctypes.data)
        return out

    def at_least(self, src, thr: int) -> np.ndarray:
        src = np.ascontiguousarray(src, dtype=np.uint8)
        out = np.empty(src.shape, bool)
        self.lib.gk_threshold(src.ctypes.data, src.size, int(thr), 1, out.ctypes.data)
        return out

    def dilate(self, mask, radius: int) -> np.ndarray:
        m = np.ascontiguousarray(mask, dtype=bool)
        out = np.empty_like(m)
        self.lib.gk_dilate(m.ctypes.data, m.shape[0], m.shape[1], int(radius), out.ctypes.data)
        return out

    def long_runs(self, mask, min_len: int, axis: int = 1) -> np.ndarray:
        m = np.ascontiguousarray(mask, dtype=bool)
        out = np.empty_like(m)
        self.lib.gk_long_runs(m.ctypes.data, m.shape[0], m.shape[1], int(min_len), int(axis), out.ctypes.data)
        return out

    def bridges(self, line, text, axis: int, reach: int) -> np.ndarray:
        line = np.ascontiguousarray(line, dtype=bool)
        text = np.ascontiguousarray(text, dtype=bool)
        out = np.empty_like(line)
        self.lib.gk_bridges(line.ctypes.data, text.ctypes.data, line.shape[0], line.shape[1], int(axis), int(reach), out.ctypes.data)
        return out

    def damp(self, rgb, mask, luma: int) -> None:
        """``rgb[mask] = max(rgb[mask], luma)`` in place (``rgb`` must be contiguous uint8)."""
        m = np.ascontiguousarray(mask, dtype=bool)
        _inplace(rgb, np.uint8)
        self.lib.gk_damp(rgb.ctypes.data, m.ctypes.data, m.size, int(luma))

    def label(self, mask, boxes: bool = True, counts: bool = False):
        """``ndimage.label`` with 8-connectivity, plus ``find_objects`` boxes ``[k, (y0, x0, y1, x1)]``
        (and, with ``counts``, the pixels of every component)."""
        m = np.ascontiguousarray(mask, dtype=bool)
        h, w = m.shape
        labels = np.empty((h, w), np.int32)
        cap = h * w // 2 + 1 if (boxes or counts) else 0
        box = np.empty((cap, 4), np.int32) if boxes else None
        cnt = np.empty(cap, np.int64) if counts else None
        n = int(self.lib.gk_label(m.ctypes.data, h, w, labels.ctypes.data, _addr(box), cap, _addr(cnt)))
        out = (labels, n, (box[:n] if boxes else None))
        return out + (cnt[:n],) if counts else out

    def select(self, labels: np.ndarray, flags: np.ndarray) -> np.ndarray:
        """Mask of the pixels whose label is flagged (``flags[0]`` is the background)."""
        lab = np.ascontiguousarray(labels, dtype=np.int32)
        f = np.ascontiguousarray(flags, dtype=bool)
        out = np.empty(lab.shape, bool)
        self.lib.gk_select(lab.ctypes.data, lab.size, f.ctypes.data, f.size, out.ctypes.data)
        return out

    def patchify(self, rgb, grid, mean, std, inv255) -> np.ndarray:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        gh, gw = grid
        out = np.empty((gh * gw, 3 * 14 * 14), F32)
        self.lib.gk_patchify(rgb.ctypes.data, gh, gw, F32(inv255), _f32(mean).ctypes.data, _f32(std).ctypes.data, out.ctypes.data)
        return out


# ------------------------------------------------------------------------
# finding and checking the library
# ------------------------------------------------------------------------
def _lib_names() -> list[str]:
    if sys.platform == "win32":
        return ["gravity_kernels.dll"]
    if sys.platform == "darwin":
        return ["libgravity_kernels.dylib"]
    return ["libgravity_kernels.so"]


def candidates() -> list[Path]:
    out = []
    env = os.environ.get("GRAVITY_OCR_KERNELS")
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    for name in _lib_names():
        out.append(here.parent / "rust_kernels" / "target" / "release" / name)  # a fresh build wins
        out.append(here / "_native" / name)
    return out


def _close(a, b, tol=TOL) -> bool:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)):
        return False
    return float(np.max(np.abs(a - b), initial=0.0)) <= tol * max(1.0, float(np.max(np.abs(b), initial=0.0)))


def self_check(k: Kernels) -> None:
    """Run every kernel once on small inputs against the numpy code; fills ``k.ok``."""
    from scipy import ndimage

    from . import ops
    from .drawing import crops, ink
    from .preprocess import MEAN, STD, to_patches_np

    rng = np.random.default_rng(20261113)
    ok, report = k.ok, k.report

    def check(name, fn):
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001 - any failure disables the kernel
            res = f"error {type(e).__name__}: {e}"
        if res is True:
            ok.add(name)
            report[name] = "ok"
        else:
            report[name] = "failed" if res is False else str(res)

    # decoders: bit for bit, vector and scalar code, a ragged tail included
    def codecs():
        for scheme in quant.SCHEMES:
            for shape, scale in (((6, 512), 0.1), ((3, 100), 0.03), ((4, 256), 2e-6)):  # 2e-6: subnormal f16 scales
                x = (rng.standard_normal(shape) * scale).astype(F32)
                parts = quant.quantize(x, scheme)
                want = quant.dequantize_np(parts, scheme, shape)
                lin = Linear(parts, scheme, shape)
                for scalar in (0, 1):
                    k.lib.gk_set_scalar(scalar)
                    got = k.dequant(lin)
                    if not np.array_equal(got.view(np.uint32), want.reshape(lin.rows, lin.cols).view(np.uint32)):
                        k.lib.gk_set_scalar(0)
                        return f"{scheme} {shape} scalar={scalar} differs"
        k.lib.gk_set_scalar(0)
        bf = (rng.standard_normal((4, 300)).astype(F32).view(np.uint32) >> 16).astype(np.uint16)
        want = (bf.astype(np.uint32) << 16).view(F32)
        return np.array_equal(k.dequant(Linear({"w": bf}, "bf16", bf.shape)), want)

    def matmul():
        for scheme, shape, n in (("q4", (40, 768), 7), ("q6", (24, 512), 1), ("nf4", (16, 256), 50), ("f32", (33, 588), 5), ("f16", (9, 260), 4)):
            x = rng.standard_normal((n, shape[1])).astype(F32)
            w = (rng.standard_normal(shape) * 0.05).astype(F32)
            parts = quant.quantize(w, scheme)
            wq = quant.dequantize_np(parts, scheme, shape)
            bias = rng.standard_normal(shape[0]).astype(F32)
            lin = Linear(parts, scheme, shape)
            want = x @ wq.T + bias
            if not _close(k.linear(lin, x, bias), want):
                return f"{scheme} {shape} n={n}"
            acc = rng.standard_normal((n, shape[0])).astype(F32)
            if not _close(k.linear(lin, x, None, out=acc.copy(), accumulate=True), acc + x @ wq.T):
                return f"{scheme} accumulate"
            # one row alone gives the same bits as inside a batch
            one = k.linear(lin, x[n // 2 : n // 2 + 1], bias)
            if not np.array_equal(one[0], k.linear(lin, x, bias)[n // 2]):
                return f"{scheme} batch invariance"
        return True

    def norms():
        x = rng.standard_normal((5, 1536)).astype(F32) * 3
        y = rng.standard_normal((5, 1536)).astype(F32)
        w = rng.standard_normal(1536).astype(F32)
        b = rng.standard_normal(1536).astype(F32)
        if not _close(k.rms_norm(x, w), ops.rms_norm(x, w)):
            return "rms_norm"
        if not _close(k.rms_norm_add(x.copy(), y, w), x + ops.rms_norm(y, w)):
            return "rms_norm_add"
        return _close(k.layer_norm(x, w, b), ops.layer_norm(x, w, b))

    def activations():
        u = (rng.standard_normal((3, 2 * 200)) * 4).astype(F32)
        if not _close(k.swiglu_split(u, 200), u[:, 200:] * ops.silu(u[:, :200])):
            return "swiglu"
        g = (rng.standard_normal((4, 37)) * 4).astype(F32)
        v = rng.standard_normal((4, 37)).astype(F32)
        return _close(k.silu_mul(g, v), ops.silu(g) * v)

    def lm_attention():
        from . import language as L

        for past, t, causal in ((5, 1, True), (3, 4, True), (2, 6, False)):
            cap = 64
            pos = np.stack([np.arange(past, past + t)] * 3, axis=1)
            cos, sin = L.rope_tables(pos)
            q = rng.standard_normal((t, 2048)).astype(F32)
            kk = rng.standard_normal((t, 1024)).astype(F32)
            v = rng.standard_normal((t, 1024)).astype(F32)
            k_old = rng.standard_normal((8, past, 128)).astype(F32)
            v_old = rng.standard_normal((8, past, 128)).astype(F32)
            kt = np.zeros((8, 128, cap), F32)
            vc = np.zeros((8, cap, 128), F32)
            kt[:, :, :past] = k_old.transpose(0, 2, 1)
            vc[:, :past] = v_old
            got = k.lm_attention(q.copy(), kk.copy(), v, cos, sin, kt, vc, past, causal)
            # numpy reference: the code of language.LanguageModel.forward
            qr = L._rotate(q.reshape(t, 16, 128), cos, sin)
            kr = L._rotate(kk.reshape(t, 8, 128), cos, sin)
            keys = np.concatenate([k_old, kr.transpose(1, 0, 2)], axis=1)
            vals = np.concatenate([v_old, v.reshape(t, 8, 128).transpose(1, 0, 2)], axis=1)
            qg = qr.reshape(t, 8, 2, 128).transpose(1, 2, 0, 3).reshape(8, 2 * t, 128)
            att = (qg @ keys.transpose(0, 2, 1)) * F32(128**-0.5)
            if causal and t > 1:
                att[:, :, past:] += np.tile(np.triu(np.full((t, t), -np.inf, F32), k=1), (2, 1))
            ops.softmax_inplace(att)
            want = (att @ vals).reshape(8, 2, t, 128).transpose(2, 0, 1, 3).reshape(t, 2048)
            if not _close(got, want):
                return f"output past={past} t={t} causal={causal}"
            if not np.array_equal(kt[:, :, past : past + t], kr.transpose(1, 2, 0)) or not np.array_equal(
                vc[:, past : past + t], v.reshape(t, 8, 128).transpose(1, 0, 2)
            ):
                return "cache rows"
        return True

    def vis_attention():
        from . import vision as V
        from .preprocess import patch_positions

        grid = (4, 6)
        n = grid[0] * grid[1]
        cos, sin = V._rope_tables(patch_positions(*grid))
        qkv = rng.standard_normal((n, 3 * 1024)).astype(F32)
        qn = rng.standard_normal(64).astype(F32)
        kn = rng.standard_normal(64).astype(F32)
        got = k.vis_attention(qkv.copy(), qn, kn, cos, sin)
        x = qkv.reshape(n, 3, 16, 64)
        q = V._rotate(ops.rms_norm(x[:, 0], qn), cos, sin).transpose(1, 0, 2)
        kk = V._rotate(ops.rms_norm(x[:, 1], kn), cos, sin).transpose(1, 2, 0)
        v = x[:, 2].transpose(1, 0, 2)
        att = ops.softmax_inplace((q @ kk) * F32(64**-0.5))
        want = (att @ v).transpose(1, 0, 2).reshape(n, 1024)
        return _close(got, want)

    def pixels():
        rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
        gray, ch, hist = k.gray_chroma(rgb)
        if not (np.array_equal(gray, ink.gray_of_np(rgb)) and np.array_equal(ch, ink.chroma_of_np(rgb))):
            return "gray/chroma"
        if not np.array_equal(hist, np.bincount(gray.reshape(-1), minlength=256)):
            return "histogram"
        if not np.array_equal(k.below(gray, 100), gray < 100):
            return "threshold"
        mask = rng.random((41, 67)) < 0.08
        for r in (1, 2, 3):
            want = ndimage.maximum_filter(mask, size=2 * r + 1, mode="constant", cval=0)
            if not np.array_equal(k.dilate(mask, r), want):
                return f"dilate r={r}"
        runs = rng.random((41, 67)) < 0.7
        for axis in (0, 1):
            if not np.array_equal(k.long_runs(runs, 5, axis), ink.long_runs_np(runs, 5, axis)):
                return f"long_runs axis={axis}"
        line = rng.random((30, 40)) < 0.6
        text = (rng.random((30, 40)) < 0.3) & ~line
        for axis in (0, 1):
            if not np.array_equal(k.bridges(line, text, axis, crops.BRIDGE_REACH), crops.bridges_np(line, text, axis)):
                return f"bridges axis={axis}"
        blobs = rng.random((60, 80)) < 0.45
        lab, n, boxes, cnt = k.label(blobs, counts=True)
        want, wn = ndimage.label(blobs, structure=ink.EIGHT)
        if n != wn or not np.array_equal(lab, want):
            return "label"
        if not np.array_equal(cnt, np.bincount(want.reshape(-1), minlength=n + 1)[1:]):
            return "label counts"
        flags = rng.random(n + 1) < 0.5
        flags[0] = False
        if not np.array_equal(k.select(lab, flags), flags[want]):
            return "select"
        objs = ndimage.find_objects(want)
        for i, sl in enumerate(objs):
            if tuple(boxes[i]) != (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop):
                return "label boxes"
        img = rgb.copy()
        m = rng.random((37, 53)) < 0.3
        k.damp(img, m, 215)
        want = rgb.copy()
        want[m] = np.maximum(want[m], 215)
        if not np.array_equal(img, want):
            return "damp"
        grid = (4, 6)
        pic = rng.integers(0, 256, (grid[0] * 14, grid[1] * 14, 3), dtype=np.uint8)
        want_p, _ = to_patches_np(pic)
        got_p = k.patchify(pic, grid, MEAN, STD, np.float32(1 / 255))
        return np.array_equal(got_p.view(np.uint32), want_p.view(np.uint32))

    t0 = time.perf_counter()
    check("codec", codecs)
    check("matmul", matmul)
    check("norm", norms)
    check("act", activations)
    check("lm_attention", lm_attention)
    check("vis_attention", vis_attention)
    check("pixels", pixels)
    report["seconds"] = f"{time.perf_counter() - t0:.3f}"


_LOADED: dict = {}


def load(verbose: bool = False) -> Kernels | None:
    """The checked library, or ``None`` (not built, switched off, or failed)."""
    if "k" in _LOADED:
        return _LOADED["k"]
    k = None
    if os.environ.get("GRAVITY_OCR_NATIVE", "1") not in ("0", "off", "false", "no"):
        for path in candidates():
            if path.is_file():
                try:
                    k = Kernels(path)
                except (OSError, AttributeError) as e:  # AttributeError: a library older than this runtime
                    warnings.warn(f"gravity_ocr: cannot load {path}: {e}", stacklevel=2)
                    continue
                self_check(k)
                failed = {n: r for n, r in k.report.items() if n != "seconds" and r != "ok"}
                if failed:
                    warnings.warn(f"gravity_ocr: native kernels disabled where the check failed: {failed}", stacklevel=2)
                if verbose:
                    print(f"native kernels {path} v{k.version} features={k.features:#x} threads={k.threads} {k.report}")
                break
    _LOADED["k"] = k
    return k


def set_threads(n: int = 0) -> int:
    """Configure the Rust Rayon thread pool size (``1`` disables internal parallelism for lock-free worker pools)."""
    k = load()
    return k.set_threads(n) if k is not None else 0


def reset() -> None:
    _LOADED.clear()
