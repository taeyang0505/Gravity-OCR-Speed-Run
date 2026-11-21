"""GPU engine (``rust_gpu/``, built on candle-rocm-vulkan): ROCm, Vulkan or CUDA.

The engine holds the weights on the device – packed as stored on ROCm (the HIP
kernels decode them in registers), decoded once to float32 on the other
backends – and runs whole vision and language passes; Python keeps the
orchestration (prompt, embedding rows from the memory-mapped package,
decoding loop) and only moves inputs and results.

:func:`load` opens the best device that is there (``"auto"``: CUDA, then
ROCm, then Vulkan), checks the engine once against the numpy code (decoders
bit for bit on the ROCm kernels, everything else within ``TOL``) and returns
``None`` when nothing usable is found, so the caller falls back to the CPU.
``GRAVITY_OCR_GPU=0`` switches the GPU off, ``GRAVITY_OCR_GPU_LIB=<path>``
points to a build elsewhere.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import sys
import time
import warnings
import weakref
from pathlib import Path

import numpy as np

from . import quant
from .native import QTensor, Linear, parts_of

F32 = np.float32
TOL = 1e-3
ABI = 2  # rust_gpu/src/lib.rs VERSION: the exported signatures this module binds
REST = "lm_head.rest.weight"  # the output rows a sliced head cut (rust_gpu engine.rs REST)
KINDS = {"rocm": 1, "vulkan": 2, "cuda": 3}
AUTO_ORDER = ("cuda", "rocm", "vulkan")

_c = ctypes
_P = _c.c_void_p
_Z = _c.c_size_t
_I = _c.c_int64


class GpuError(RuntimeError):
    pass


def _lib_names() -> list[str]:
    if sys.platform == "win32":
        return ["gravity_gpu.dll"]
    if sys.platform == "darwin":
        return ["libgravity_gpu.dylib"]
    return ["libgravity_gpu.so"]


def candidates() -> list[Path]:
    out = []
    env = os.environ.get("GRAVITY_OCR_GPU_LIB")
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    for name in _lib_names():
        out.append(here.parent / "rust_gpu" / "target" / "release" / name)  # a fresh build wins
        out.append(here / "_native" / name)
    return out


def _dll_dirs() -> None:
    """Windows resolves the HIP / CUDA runtime DLLs only from directories it is told about."""
    if sys.platform != "win32":
        return
    seen = set()
    for var in ("HIP_PATH", "ROCM_PATH", "CUDA_PATH"):
        root = os.environ.get(var)
        if not root:
            continue
        # CUDA 13 keeps its runtime DLLs (cublas, curand, nvrtc) in bin\x64
        for sub in ("bin", "bin/x64"):
            if (Path(root) / sub).is_dir():
                d = str((Path(root) / sub).resolve())
                if d not in seen:
                    seen.add(d)
                    os.add_dll_directory(d)


class Library:
    def __init__(self, path: Path):
        _dll_dirs()
        # CUDA: load the kernel modules (cuBLAS, candle's) when first launched, not all at once –
        # measured on Windows: 1.46 GB committed host memory for the context vs 2.89 GB eager
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        self.path = Path(path)
        self.lib = lib = _c.CDLL(str(self.path))
        lib.gg_version.restype, lib.gg_version.argtypes = _c.c_uint32, []
        if lib.gg_version() != ABI:
            raise OSError(f"library ABI {lib.gg_version()}, this runtime needs {ABI} (rebuild rust_gpu)")
        sig = {
            "gg_backends": (_c.c_uint32, []),
            "gg_device_count": (_I, [_c.c_uint32]),
            "gg_last_error": (_Z, [_c.c_char_p, _Z]),
            "gg_open": (_P, [_c.c_uint32, _c.c_uint32]),
            "gg_close": (None, [_P]),
            "gg_describe": (_Z, [_P, _c.c_char_p, _Z]),
            "gg_set_option": (_I, [_P, _c.c_uint32, _I]),
            "gg_release": (_I, [_P]),
            "gg_trim": (_I, [_P]),
            "gg_mem": (_I, [_P, _P, _P]),
            "gg_weight_bytes": (_c.c_uint64, [_P]),
            "gg_synchronize": (_I, [_P]),
            "gg_set_tensor": (_I, [_P, _c.c_char_p, _P, _c.c_int32]),
            "gg_head_rows": (_I, [_P]),
            "gg_vision": (_I, [_P, _P, _Z, _P, _P, _P, _Z, _P]),
            "gg_cache_new": (_P, [_P, _Z]),
            "gg_cache_grow": (_I, [_P, _P, _Z, _Z]),
            "gg_cache_free": (None, [_P]),
            "gg_lm": (_I, [_P, _P, _P, _Z, _P, _P, _Z, _c.c_int32, _Z, _P, _P]),
            "gg_test_dequant": (_I, [_P, _P, _P]),
            "gg_test_linear": (_I, [_P, _P, _P, _Z, _P, _P]),
            "gg_bench_linear": (_c.c_double, [_P, _P, _Z, _c.c_uint32, _Z]),
            "gg_test_lm_attention": (_I, [_P, _P, _P, _P, _P, _P, _P, _P, _Z, _Z, _c.c_int32, _P, _P]),
            "gg_test_vis_attention": (_I, [_P, _P, _P, _P, _P, _P, _Z, _P]),
            "gg_test_rows": (_I, [_P, _c.c_uint32, _P, _Z, _Z, _P, _P, _P]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        # optional exports (newer libraries): the language pass that also checks the cut output rows
        self.has_rest = hasattr(lib, "gg_lm_rest")
        if self.has_rest:
            lib.gg_lm_rest.restype = _I
            lib.gg_lm_rest.argtypes = [_P, _P, _P, _Z, _P, _P, _Z, _c.c_int32, _Z, _P, _P, _P]
        self.has_rest_max = hasattr(lib, "gg_rest_max")
        if self.has_rest_max:
            lib.gg_rest_max.restype = _I
            lib.gg_rest_max.argtypes = [_P, _P, _Z, _Z, _P, _P]
        # optional exports: dropping a group of weights, half / FP8 weights, compressed KV caches
        mem_sig = {
            "gg_drop": (_I, [_P, _c.c_char_p]),
            "gg_cache_bytes": (_c.c_uint64, [_P]),
            "gg_kv_row_bytes": (_I, [_P, _c.c_uint32]),
            "gg_test_kv_attention": (_I, [_P, _c.c_uint32, _P, _P, _P, _P, _P, _P, _P, _Z, _Z, _c.c_int32, _P]),
        }
        self.has_memory = all(hasattr(lib, n) for n in mem_sig)
        if self.has_memory:
            for name, (res, args) in mem_sig.items():
                fn = getattr(lib, name)
                fn.restype, fn.argtypes = res, args
        # optional exports: GGML block formats (the GGUF pyramid model, gravity_ocr/gguf.py)
        ggml_sig = {
            "gg_ggml_row_bytes": (_I, [_c.c_uint32, _Z]),
            "gg_ggml_quantize": (_I, [_P, _c.c_uint32, _P, _P, _Z]),
            "gg_ggml_dequant": (_I, [_c.c_uint32, _Z, _Z, _P, _Z, _P]),
            "gg_set_ggml": (_I, [_P, _c.c_char_p, _c.c_uint32, _Z, _Z, _P, _Z]),
            "gg_test_ggml_linear": (_I, [_P, _c.c_uint32, _Z, _Z, _P, _Z, _P, _Z, _P]),
            "gg_gguf_info": (_I, [_c.c_char_p, _P, _P]),
        }
        self.has_ggml = all(hasattr(lib, n) for n in ggml_sig)
        if self.has_ggml:
            for name, (res, args) in ggml_sig.items():
                fn = getattr(lib, name)
                fn.restype, fn.argtypes = res, args
        self.version = lib.gg_version()
        self.backends = [k for k, bit in (("rocm", 1), ("vulkan", 2), ("cuda", 4)) if lib.gg_backends() & bit]

    def error(self) -> str:
        buf = _c.create_string_buffer(4096)
        self.lib.gg_last_error(buf, len(buf))
        return buf.value.decode("utf-8", "replace")

    def check(self, rc: int, what: str) -> None:
        if rc < 0:
            raise GpuError(f"{what}: {self.error()}")

    def devices(self, kind: str) -> int:
        n = self.lib.gg_device_count(KINDS[kind])
        return max(int(n), 0)


def _addr(a) -> int | None:
    return None if a is None else a.ctypes.data


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=F32)


class Cache:
    """Key/value cache of the language model on the device."""

    def __init__(self, engine: "Engine", capacity: int, row_bytes: int | None = None, guard=None):
        self.engine = engine
        self.capacity = int(capacity)
        self.row_bytes = row_bytes or 2 * 16 * 8 * 128 * 4  # bytes per position (format of the cache)
        self.guard = guard  # called with the bytes a growth allocates before it does (residency)
        self.ptr = engine.lib.lib.gg_cache_new(engine.ptr, self.capacity)
        if not self.ptr:
            raise GpuError(f"cache: {engine.lib.error()}")
        self.len = 0
        self.transposed = False

    def free(self) -> None:
        if self.ptr:
            self.engine.lib.lib.gg_cache_free(self.ptr)
            self.ptr = None

    def reserve(self, extra: int) -> None:
        need = self.len + extra
        if need <= self.capacity:
            return
        cap = max(need, self.capacity * 2)
        if self.guard is not None:
            # the old buffers live until the copy is done: the new ones must fit next to them;
            # when doubling does not fit, grow to exactly what is needed
            cap = self.guard(self.row_bytes * cap, self.row_bytes * need) // self.row_bytes
        self.engine.lib.check(self.engine.lib.lib.gg_cache_grow(self.engine.ptr, self.ptr, cap, self.len), "cache grow")
        self.capacity = cap

    @property
    def nbytes(self) -> int:
        if self.ptr and self.engine.lib.has_memory:
            return int(self.engine.lib.lib.gg_cache_bytes(self.ptr))
        return self.row_bytes * self.capacity

    def __del__(self):
        try:
            self.free()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


_ENGINES: "weakref.WeakSet[Engine]" = weakref.WeakSet()


@atexit.register
def _shutdown() -> None:
    """Give every device allocation back before the HIP runtime is torn down (it can hang otherwise)."""
    for e in list(_ENGINES):
        try:
            e.close()
        except Exception:  # noqa: BLE001
            pass


class Engine:
    def __init__(self, lib: Library, kind: str, ordinal: int = 0):
        self.lib = lib
        self.kind = kind
        self.ptr = lib.lib.gg_open(KINDS[kind], ordinal)
        if not self.ptr:
            raise GpuError(f"open {kind}: {lib.error()}")
        self.info = self.describe()
        self.report: dict[str, str] = {}
        self.ok = False
        self._caches: "weakref.WeakSet[Cache]" = weakref.WeakSet()
        _ENGINES.add(self)

    def describe(self) -> dict:
        buf = _c.create_string_buffer(4096)
        self.lib.lib.gg_describe(self.ptr, buf, len(buf))
        return json.loads(buf.value.decode("utf-8", "replace"))

    OPTIONS = {"gemv_max": 1, "custom": 2, "flash": 3, "qgemm_max": 4, "vis_attn": 5, "chunk_bytes": 6, "lm_attn": 7,
               "weight_store": 8, "half_compute": 9, "decode_bytes": 10, "kv_format": 11}  # fmt: skip

    def set_option(self, key: str, value: int) -> None:
        k = self.OPTIONS[key]
        self.lib.check(self.lib.lib.gg_set_option(self.ptr, k, int(value)), f"option {key}")
        self.info = self.describe()

    def drop(self, prefix: str) -> int:
        """Drop the weights named ``prefix…`` and settle the device memory; returns their bytes."""
        if not self.lib.has_memory:
            raise GpuError("this GPU library cannot drop weights (rebuild rust_gpu)")
        rc = self.lib.lib.gg_drop(self.ptr, prefix.encode("utf-8"))
        self.lib.check(rc, f"drop {prefix}")
        return int(rc)

    def kv_row_bytes(self, fmt: int) -> int:
        """Bytes of one cached position in KV format ``fmt`` (all layers, keys and values)."""
        if not self.lib.has_memory:
            return 2 * 16 * 8 * 128 * 4
        rc = self.lib.lib.gg_kv_row_bytes(self.ptr, int(fmt))
        self.lib.check(rc, "kv row bytes")
        return int(rc)

    def mem(self) -> tuple[int, int]:
        free, total = _c.c_uint64(), _c.c_uint64()
        self.lib.check(self.lib.lib.gg_mem(self.ptr, _c.addressof(free), _c.addressof(total)), "mem")
        return int(free.value), int(total.value)

    @property
    def weight_bytes(self) -> int:
        return int(self.lib.lib.gg_weight_bytes(self.ptr))

    def synchronize(self) -> None:
        self.lib.check(self.lib.lib.gg_synchronize(self.ptr), "synchronize")

    def trim(self) -> None:
        """Hand the allocator's cached blocks back to the driver (keeps VRAM for other processes)."""
        self.lib.check(self.lib.lib.gg_trim(self.ptr), "trim")

    def close(self) -> None:
        """Free the caches and weights, return the memory to the driver, drop the engine."""
        if self.ptr:
            for c in list(self._caches):
                c.free()
            self.lib.lib.gg_release(self.ptr)
            self.lib.lib.gg_close(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- weights ---------------------------------------------------------
    def set_linear(self, name: str, lin: Linear, vector: bool = False) -> None:
        self.lib.check(self.lib.lib.gg_set_tensor(self.ptr, name.encode("utf-8"), lin.addr, 1 if vector else 0), f"upload {name}")

    def set_tensor(self, name: str, parts: dict, scheme: str, shape, vector: bool = False) -> None:
        if vector:
            shape = (1, int(np.prod(shape)))
        self.set_linear(name, Linear(parts, scheme, shape), vector)

    def set_ggml(self, name: str, kind: str, shape, data: np.ndarray) -> None:
        """Matrix ``name`` as GGML blocks (e.g. a view into a memory-mapped GGUF file), kept and
        multiplied in that format on the device."""
        from .gguf import GGML

        if not self.lib.has_ggml:
            raise GpuError("this GPU library has no GGML support (rebuild rust_gpu)")
        rows, cols = int(shape[0]), int(np.prod(shape[1:]))
        data = np.ascontiguousarray(data, dtype=np.uint8)
        rc = self.lib.lib.gg_set_ggml(self.ptr, name.encode("utf-8"), GGML[kind], rows, cols, data.ctypes.data, data.nbytes)
        self.lib.check(rc, f"upload {name} ({kind})")

    def set_array(self, name: str, value: np.ndarray, vector: bool = False) -> None:
        value = _f32(value)
        shape = (1, value.size) if vector or value.ndim == 1 else value.shape
        self.set_linear(name, Linear({"w": value.reshape(shape)}, "f32", shape), vector)

    @property
    def head_rows(self) -> int:
        return int(self.lib.lib.gg_head_rows(self.ptr))

    # -- passes ----------------------------------------------------------
    def vision(self, patches: np.ndarray, cos: np.ndarray, sin: np.ndarray, segs=None) -> np.ndarray:
        """Image tokens of one image, or of a batch whose patch counts are ``segs``."""
        patches, cos, sin = _f32(patches), _f32(cos), _f32(sin)
        n = patches.shape[0]
        out = np.empty((n // 4, 1536), F32)
        seg = None if not segs else np.asarray(segs, np.uint64)
        rc = self.lib.lib.gg_vision(self.ptr, patches.ctypes.data, n, cos.ctypes.data, sin.ctypes.data, _addr(seg), 0 if seg is None else seg.size, out.ctypes.data)
        self.lib.check(rc, "vision")
        return out

    def new_cache(self, capacity: int, row_bytes: int | None = None, guard=None) -> Cache:
        c = Cache(self, capacity, row_bytes, guard)
        self._caches.add(c)
        return c

    def lm(self, cache: Cache, x, cos, sin, past: int, causal: bool, start: int, inject=None, rest: bool = False):
        """Logits of rows ``start..``; with ``rest`` also ``(best logit, its row)`` of the cut output rows."""
        x, cos, sin = _f32(x), _f32(cos), _f32(sin)
        t = x.shape[0]
        inject = None if inject is None else _f32(inject)
        out = np.empty((t - start, self.head_rows), F32)
        if rest:
            best = np.empty((t - start, 2), F32)
            rc = self.lib.lib.gg_lm_rest(self.ptr, cache.ptr, x.ctypes.data, t, cos.ctypes.data, sin.ctypes.data, past, 1 if causal else 0,
                                         start, _addr(inject), out.ctypes.data, best.ctypes.data)  # fmt: skip
            self.lib.check(rc, "lm")
            return out, (best[:, 0].copy(), best[:, 1].astype(np.int64))
        rc = self.lib.lib.gg_lm(self.ptr, cache.ptr, x.ctypes.data, t, cos.ctypes.data, sin.ctypes.data, past, 1 if causal else 0, start,
                                _addr(inject), out.ctypes.data)  # fmt: skip
        self.lib.check(rc, "lm")
        return out

    def rest_max(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the cut output rows (``lm_head.rest.weight``) on uncertain hidden states ``x [n, k]``."""
        if not self.lib.has_rest_max:
            raise GpuError("this GPU library has no gg_rest_max export (rebuild rust_gpu)")
        x = _f32(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, k = int(x.shape[0]), int(x.shape[1])
        out_max = np.empty(n, F32)
        out_id = np.empty(n, np.uint32)
        rc = self.lib.lib.gg_rest_max(self.ptr, x.ctypes.data, n, k, out_max.ctypes.data, out_id.ctypes.data)
        self.lib.check(rc, "rest_max")
        return out_max, out_id.astype(np.int64)


# ------------------------------------------------------------------------
# self-check
# ------------------------------------------------------------------------
def _close(a, b, tol=TOL) -> bool:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)):
        return False
    return float(np.max(np.abs(a - b), initial=0.0)) <= tol * max(1.0, float(np.max(np.abs(b), initial=0.0)))


def self_check(e: Engine) -> bool:
    """The engine's kernels against the numpy code on small inputs; fills ``e.report``."""
    from . import ops
    from . import language as L
    from . import vision as V
    from .preprocess import patch_positions

    rng = np.random.default_rng(20261113)
    lib = e.lib.lib
    report = e.report

    def check(name, fn):
        try:
            res = fn()
        except Exception as x:  # noqa: BLE001
            res = f"error {type(x).__name__}: {x}"
        report[name] = "ok" if res is True else ("failed" if res is False else str(res))

    def dequant():
        if not e.info.get("custom"):
            return True  # decoded on the host by the checked CPU codec
        for scheme in quant.SCHEMES:
            for scale in (0.05, 2e-6):  # the second gives subnormal f16 scales
                shape = (8, 512)
                x = (rng.standard_normal(shape) * scale).astype(F32)
                parts = quant.quantize(x, scheme)
                want = quant.dequantize_np(parts, scheme, shape)
                lin = Linear(parts, scheme, shape)
                out = np.empty(shape, F32)
                e.lib.check(lib.gg_test_dequant(e.ptr, lin.addr, out.ctypes.data), "dequant")
                if not np.array_equal(out.view(np.uint32), want.view(np.uint32)):
                    return f"{scheme} scale={scale} differs"
        return True

    def linear():
        for scheme, shape, n in (("q4", (40, 768), 1), ("q6", (24, 512), 5), ("q8", (33, 256), 32), ("nf4", (16, 512), 40),
                                 ("q3", (8, 512), 3), ("q5", (16, 256), 64), ("bf16", (12, 512), 2), ("f32", (9, 588), 50)):  # fmt: skip
            x = rng.standard_normal((n, shape[1])).astype(F32)
            w = (rng.standard_normal(shape) * 0.05).astype(F32)
            if scheme == "bf16":
                raw = (w.view(np.uint32) >> 16).astype(np.uint16)
                parts, wq = {"w": raw}, (raw.astype(np.uint32) << 16).view(F32)
            else:
                parts = quant.quantize(w, scheme)
                wq = quant.dequantize_np(parts, scheme, shape)
            bias = rng.standard_normal(shape[0]).astype(F32)
            lin = Linear(parts, scheme, shape)
            out = np.empty((n, shape[0]), F32)
            e.lib.check(lib.gg_test_linear(e.ptr, lin.addr, x.ctypes.data, n, bias.ctypes.data, out.ctypes.data), "linear")
            if not _close(out, x @ wq.T + bias):
                return f"{scheme} {shape} n={n}"
        return True

    def rows():
        x = (rng.standard_normal((5, 1536)) * 3).astype(F32)
        w = rng.standard_normal(1536).astype(F32)
        b = rng.standard_normal(1536).astype(F32)
        out = np.empty_like(x)
        e.lib.check(lib.gg_test_rows(e.ptr, 0, x.ctypes.data, 5, 1536, w.ctypes.data, None, out.ctypes.data), "rms")
        if not _close(out, ops.rms_norm(x, w)):
            return "rms_norm"
        e.lib.check(lib.gg_test_rows(e.ptr, 1, x.ctypes.data, 5, 1536, w.ctypes.data, b.ctypes.data, out.ctypes.data), "ln")
        if not _close(out, ops.layer_norm(x, w, b)):
            return "layer_norm"
        u = (rng.standard_normal((3, 400)) * 4).astype(F32)
        o2 = np.empty((3, 200), F32)
        e.lib.check(lib.gg_test_rows(e.ptr, 2, u.ctypes.data, 3, 200, None, None, o2.ctypes.data), "swiglu")
        if not _close(o2, u[:, 200:] * ops.silu(u[:, :200])):
            return "swiglu"
        g = (rng.standard_normal((4, 64)) * 4).astype(F32)
        v = rng.standard_normal((4, 64)).astype(F32)
        o3 = np.empty_like(g)
        e.lib.check(lib.gg_test_rows(e.ptr, 3, g.ctypes.data, 4, 64, v.ctypes.data, None, o3.ctypes.data), "silu_mul")
        return _close(o3, ops.silu(g) * v)

    def lm_attention():
        # one row, a few rows, tiles of 8 rows with a partial last one, more than one chunk of keys
        for past, t, causal in ((5, 1, True), (3, 4, True), (2, 6, False), (0, 3, True), (40, 11, True), (37, 17, False), (1100, 2, True)):
            pos = np.stack([np.arange(past, past + t)] * 3, axis=1)
            cos, sin = L.rope_tables(pos)
            q = rng.standard_normal((t, 2048)).astype(F32)
            k = rng.standard_normal((t, 1024)).astype(F32)
            v = rng.standard_normal((t, 1024)).astype(F32)
            k_old = rng.standard_normal((past, 8, 128)).astype(F32)
            v_old = rng.standard_normal((past, 8, 128)).astype(F32)
            out = np.empty((t, 16, 128), F32)
            kc = np.empty((past + t, 8, 128), F32)
            rc = lib.gg_test_lm_attention(e.ptr, q.ctypes.data, k.ctypes.data, v.ctypes.data, cos.ctypes.data, sin.ctypes.data,
                                          k_old.ctypes.data if past else None, v_old.ctypes.data if past else None,
                                          past, t, 1 if causal else 0, out.ctypes.data, kc.ctypes.data)  # fmt: skip
            e.lib.check(rc, "lm attention")
            qr = L._rotate(q.reshape(t, 16, 128), cos, sin)
            kr = L._rotate(k.reshape(t, 8, 128), cos, sin)
            keys = np.concatenate([k_old.transpose(1, 0, 2), kr.transpose(1, 0, 2)], axis=1)
            vals = np.concatenate([v_old.transpose(1, 0, 2), v.reshape(t, 8, 128).transpose(1, 0, 2)], axis=1)
            qg = qr.reshape(t, 8, 2, 128).transpose(1, 2, 0, 3).reshape(8, 2 * t, 128)
            att = (qg @ keys.transpose(0, 2, 1)) * F32(128**-0.5)
            if causal and t > 1:
                att[:, :, past:] += np.tile(np.triu(np.full((t, t), -np.inf, F32), k=1), (2, 1))
            ops.softmax_inplace(att)
            want = (att @ vals).reshape(8, 2, t, 128).transpose(2, 0, 1, 3).reshape(t, 16, 128)
            if not _close(out, want):
                return f"output past={past} t={t} causal={causal}"
            if not _close(kc[past:], kr, 1e-6):
                return "cache rows"
        return True

    def vis_attention():
        grid = (4, 6)
        n = grid[0] * grid[1]
        cos, sin = V._rope_tables(patch_positions(*grid))
        qkv = rng.standard_normal((n, 3 * 1024)).astype(F32)
        qn = rng.standard_normal(64).astype(F32)
        kn = rng.standard_normal(64).astype(F32)
        out = np.empty((n, 16, 64), F32)
        e.lib.check(lib.gg_test_vis_attention(e.ptr, qkv.ctypes.data, qn.ctypes.data, kn.ctypes.data, cos.ctypes.data, sin.ctypes.data, n, out.ctypes.data), "vis attention")
        x = qkv.reshape(n, 3, 16, 64)
        q = V._rotate(ops.rms_norm(x[:, 0], qn), cos, sin).transpose(1, 0, 2)
        k = V._rotate(ops.rms_norm(x[:, 1], kn), cos, sin).transpose(1, 2, 0)
        v = x[:, 2].transpose(1, 0, 2)
        att = ops.softmax_inplace((q @ k) * F32(64**-0.5))
        want = (att @ v).transpose(1, 0, 2)
        return _close(out, want)

    t0 = time.perf_counter()
    check("dequant", dequant)
    check("linear", linear)
    check("rows", rows)
    check("lm_attention", lm_attention)
    check("vis_attention", vis_attention)
    report["seconds"] = f"{time.perf_counter() - t0:.3f}"
    e.ok = all(v == "ok" for k, v in report.items() if k != "seconds")
    return e.ok


_LIB: dict = {}


def library() -> Library | None:
    if "lib" in _LIB:
        return _LIB["lib"]
    lib = None
    if os.environ.get("GRAVITY_OCR_GPU", "1") not in ("0", "off", "false", "no"):
        for path in candidates():
            if path.is_file():
                try:
                    lib = Library(path)
                    break
                except OSError as x:
                    warnings.warn(f"gravity_ocr: cannot load {path}: {x}", stacklevel=2)
    _LIB["lib"] = lib
    return lib


def load(kind: str = "auto", verbose: bool = False) -> Engine | None:
    """A checked engine on ``kind`` (``auto``/``gpu``: the first that works), or ``None``."""
    lib = library()
    if lib is None:
        return None
    order = AUTO_ORDER if kind in ("auto", "gpu") else (kind,)
    for k in order:
        if k not in lib.backends or lib.devices(k) <= 0:
            continue
        try:
            e = Engine(lib, k)
        except GpuError as x:
            warnings.warn(f"gravity_ocr: {x}", stacklevel=2)
            continue
        self_check(e)
        if verbose:
            print(f"gpu engine {k}: {e.info} check={e.report}")
        if e.ok:
            return e
        warnings.warn(f"gravity_ocr: {k} engine failed its self-check: {e.report}", stacklevel=2)
        e.close()
    return None


# ------------------------------------------------------------------------
# model parts on the engine
# ------------------------------------------------------------------------
def _parts_bytes(parts: dict) -> int:
    return int(sum(np.asarray(v).nbytes for v in parts.values()))


class _Part:
    """A group of weights on the engine.  With a :class:`~gravity_ocr.residency.Residency` the
    group is uploaded when a pass needs it and may be dropped in between (``swap``); without
    one it is uploaded now and stays.  Weights replaced after construction (``set`` /
    ``set_parts``: a sliced output matrix, surgery, quantisation sweeps) are kept on the host
    and uploaded again with the group."""

    GROUP = ""
    PREFIXES: tuple = ()

    def __init__(self, engine: Engine, src, residency=None):
        self.e = engine
        self.src = src
        self.res = residency
        self.overrides: dict[str, tuple] = {}  # name → (parts, scheme, shape, vector)

    def _start(self) -> None:
        if self.res is None:
            self._load()
        else:
            self.res.register(self.GROUP, self.PREFIXES, self._load, self._specs)

    def _ready(self) -> None:
        if self.res is not None:
            self.res.enter(self.GROUP)

    @property
    def resident(self) -> bool:
        return self.res is None or self.res.is_resident(self.GROUP)

    def _put_override(self, name: str, parts: dict, scheme: str, shape, vector: bool) -> None:
        self.overrides[name] = (parts, scheme, tuple(shape), vector)
        if self.resident:

            def up() -> None:
                self.e.set_tensor(name, parts, scheme, shape, vector=vector)

            if self.res is None:
                up()
            else:
                self.res.apply(self.GROUP, up)

    def _prepare(self, name: str) -> None:
        """The residency's store of tensor ``name`` (the pyramid store picks it per tensor)."""
        if self.res is not None:
            self.res.prepare(name)

    def _upload_override(self, name: str) -> bool:
        if name not in self.overrides:
            return False
        self._prepare(name)
        parts, scheme, shape, vector = self.overrides[name]
        self.e.set_tensor(name, parts, scheme, shape, vector=vector)
        return True

    def _ggml(self, name: str) -> bool:
        """Is ``name`` GGML blocks in the source (a GGUF file)?"""
        is_ggml = getattr(self.src, "is_ggml", None)
        return bool(is_ggml and is_ggml(name))

    def _upload_src(self, name: str, vector: bool = False) -> None:
        """Tensor ``name`` from the source: GGML blocks as they are, anything else as codec parts."""
        self._prepare(name)
        if self._ggml(name):
            self.e.set_ggml(name, self.src.kind(name), self.src.shape(name), self.src.raw(name))
        else:
            self.e.set_tensor(name, *parts_of(self.src, name), vector=vector)

    def _spec_of(self, name: str, matrix: bool):
        from .residency import Spec

        if self._ggml(name):
            return Spec(tuple(self.src.shape(name)), self.src.scheme(name), True, self.src.stored_bytes(name), name=name)
        return Spec(tuple(self.src.shape(name)), self.src.scheme(name), matrix, name=name)

    def _done(self, name: str) -> None:
        """Tensor ``name`` is on the device: its pages of the memory-mapped source leave this
        process's working set right away, so the peak during an upload is about one tensor, not
        a whole group (measured over 20 swapped crops: peak working set 1.93 → 0.99 GB,
        docs/REPORT.md §18)."""
        rel = getattr(self.src, "release_pages", None)
        if rel is not None and name in self.src:
            rel(name)

    def release(self) -> None:
        """Drop this group from the device now (it comes back with the next pass)."""
        if self.res is not None:
            self.res.release(self.GROUP)


class GpuVision(_Part):
    GROUP = "vision"
    PREFIXES = ("model.visual.",)

    def __init__(self, engine: Engine, src, residency=None):
        super().__init__(engine, src, residency)
        self._start()

    def _load(self) -> None:
        from . import vision as V

        for name in V.tensor_names():
            if not self._upload_override(name):
                self.upload(name, self.src)
                self._done(name)

    def _specs(self) -> list:
        from . import vision as V
        from .residency import Spec

        out = []
        for name in V.tensor_names():
            short = name[len(V.PREFIX) :]
            if name in self.overrides:
                parts, scheme, shape, vector = self.overrides[name]
                out.append(Spec(tuple(shape), scheme, not vector, _parts_bytes(parts), name=name))
            elif V.is_matrix(short) and short in V.RESHAPED and not getattr(self.src, "reshaped", False):
                # uploaded as float32 after the reshape (the two frames of the patch embedding summed)
                n = int(np.prod(self.src.shape(name)))
                rows = self.src.shape(name)[0]
                cols = n // rows // (2 if short == "patch_embed.proj.weight" else 1)
                out.append(Spec((rows, cols), "f32", True, name=name))
            else:
                out.append(self._spec_of(name, V.is_matrix(short)))
        return out

    def upload(self, name: str, src) -> None:
        from . import vision as V

        short = name[len(V.PREFIX) :]
        self._prepare(name)
        if src is self.src and self._ggml(name):
            self._upload_src(name)
        elif V.is_matrix(short) and (short not in V.RESHAPED or getattr(src, "reshaped", False)):
            self.e.set_tensor(name, *parts_of(src, name))
        elif V.is_matrix(short):
            self.e.set_array(name, V.reshape_weight(short, np.asarray(src.get(name), F32)))
        else:
            self.e.set_tensor(name, *parts_of(src, name), vector=True)

    def set(self, name: str, value: np.ndarray) -> None:
        from . import vision as V

        short = name[len(V.PREFIX) :]
        w = _f32(V.reshape_weight(short, np.asarray(value, F32)))
        vector = not V.is_matrix(short)
        shape = (1, w.size) if vector or w.ndim == 1 else w.shape
        self._put_override(name, {"w": w.reshape(shape)}, "f32", shape, vector)

    def set_parts(self, name: str, parts: dict, scheme: str, shape) -> None:
        from . import vision as V

        short = name[len(V.PREFIX) :]
        if V.is_matrix(short) and short not in V.RESHAPED:
            self._put_override(name, parts, scheme, shape, False)
        else:
            self.set(name, quant.dequantize(parts, scheme, shape))

    def forward(self, patches: np.ndarray, grid: tuple[int, int], taps: dict | None = None, rows=None) -> np.ndarray:
        from .preprocess import patch_positions
        from .vision import _rope_tables

        if taps is not None:
            raise NotImplementedError("taps need a CPU device")
        self._ready()
        if self.res is not None:
            self.res.vision_batches([patches.shape[0]])
        pos = patch_positions(*grid)
        cos, sin = _rope_tables(pos if rows is None else pos[rows])
        return self.e.vision(patches, cos, sin)

    def forward_batch(self, items: list) -> list[np.ndarray]:
        """``[(patches, grid, rows or None), ...]`` in one pass (or in as many as the device memory
        takes); attention stays inside each image."""
        self._ready()
        if self.res is None:
            return self._batch(items)
        out = []
        for b in self.res.vision_batches([p.shape[0] for p, _, _ in items]):
            if len(b) > 1:
                out += self._batch([items[i] for i in b])
            else:
                patches, grid, rows = items[b[0]]
                out.append(self.forward(patches, grid, rows=rows))
        return out

    def _batch(self, items: list) -> list[np.ndarray]:
        from .preprocess import patch_positions
        from .vision import _rope_tables

        tabs = []
        for patches, grid, rows in items:
            pos = patch_positions(*grid)
            tabs.append(_rope_tables(pos if rows is None else pos[rows]))
        segs = [p.shape[0] for p, _, _ in items]
        out = self.e.vision(np.concatenate([p for p, _, _ in items]), np.concatenate([c for c, _ in tabs]),
                            np.concatenate([s for _, s in tabs]), segs)  # fmt: skip
        return np.split(out, np.cumsum([s // 4 for s in segs])[:-1])


class GpuLanguage(_Part):
    GROUP = "language"
    PREFIXES = ("model.language_model.", "lm_head.")

    def __init__(self, engine: Engine, src, head_parts: tuple | None, residency=None):
        super().__init__(engine, src, residency)
        # (parts, scheme, shape) of the output matrix in the source; None: taken from the source
        # by name (GGML blocks of a GGUF file)
        self.head_parts = head_parts
        self.layer_hook = None
        self.rest = False
        self.last_rest = None  # (best logit, its row in the cut rows) per output row of the last pass
        self._start()

    def _load(self) -> None:
        from . import language as L

        for name in L.tensor_names():
            if not self._upload_override(name):
                self._upload_src(name, vector=not L.is_matrix(name[len(L.PREFIX) :]))
                self._done(name)
        if not self._upload_override(L.HEAD):
            if self.head_parts is None:
                self._upload_src(L.HEAD)
            else:
                self._prepare(L.HEAD)
                self.e.set_tensor(L.HEAD, *self.head_parts)
            self._done(L.HEAD)
        self._upload_override(REST)

    def _specs(self) -> list:
        from . import language as L
        from .residency import Spec

        out = []
        for name in L.tensor_names():
            if name in self.overrides:
                parts, scheme, shape, vector = self.overrides[name]
                out.append(Spec(tuple(shape), scheme, not vector, _parts_bytes(parts), name=name))
            else:
                out.append(self._spec_of(name, L.is_matrix(name[len(L.PREFIX) :])))
        for name in (L.HEAD, REST):
            if name in self.overrides:
                parts, scheme, shape, _ = self.overrides[name]
                out.append(Spec(tuple(shape), scheme, True, _parts_bytes(parts), name=name))
            elif name == L.HEAD and self.head_parts is None:
                out.append(self._spec_of(L.HEAD, True))
            elif name == L.HEAD:
                parts, scheme, shape = self.head_parts
                out.append(Spec(tuple(shape), scheme, True, _parts_bytes(parts), name=L.HEAD))
        return out

    @property
    def head_rows(self) -> int:
        from . import language as L

        if L.HEAD in self.overrides:
            return int(self.overrides[L.HEAD][2][0])
        return int((self.head_parts[2] if self.head_parts is not None else self.src.shape(L.HEAD))[0])

    def set_rest(self, parts: dict, scheme: str, shape) -> bool:
        """The output rows the slice cut; ``False`` when the library cannot check them."""
        if not self.e.lib.has_rest:
            return False
        self._put_override(REST, parts, scheme, shape, False)
        self.rest = True
        return True

    def set(self, name: str, value: np.ndarray) -> None:
        from . import language as L

        value = _f32(value)
        vector = not L.is_matrix(name[len(L.PREFIX) :]) and name != L.HEAD
        shape = (1, value.size) if vector or value.ndim == 1 else value.shape
        self._put_override(name, {"w": value.reshape(shape)}, "f32", shape, vector)

    def set_parts(self, name: str, parts: dict, scheme: str, shape) -> None:
        from . import language as L

        if name == L.HEAD or L.is_matrix(name[len(L.PREFIX) :]):
            self._put_override(name, parts, scheme, shape, False)
        else:
            self.set(name, quant.dequantize(parts, scheme, shape))

    def new_cache(self, capacity: int) -> Cache:
        self._ready()
        if self.res is not None:
            return self.res.cache(capacity)
        return self.e.new_cache(capacity)

    def forward_logits(self, x, pos, cache: Cache, mode="causal", commit=True, last_only=False, start: int = 0, inject=None,
                       rest: bool = True) -> np.ndarray:  # fmt: skip
        from .language import rope_tables

        self._ready()
        if self.layer_hook is not None:
            x = self.layer_hook(0, x)
        t = x.shape[0]
        past = cache.len
        cache.reserve(t)
        cos, sin = rope_tables(pos)
        check = self.rest and rest
        logits = self.e.lm(cache, x, cos, sin, past, mode == "causal", t - 1 if last_only else start, inject, rest=check)
        self.last_rest = None
        if check:
            logits, self.last_rest = logits
        if commit:
            cache.len = past + t
        return logits[0] if last_only else logits
