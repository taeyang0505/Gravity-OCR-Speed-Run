"""GGUF files: the "pyramid" model – every matrix in the GGML block format its measured
sensitivity allows (Q8_0 / Q6_K / Q5_K / Q4_K), quantised once from the original BF16.

A GPU engine multiplies the blocks as they are (``rust_gpu/src/ggml.rs``: candle ``QMatMul``,
CUDA MMVQ / MMQ kernels), so the device holds ≈ 0.7 GB instead of 1.9 GB of bf16 weights and no
matrix is ever expanded.  Written by ``tools/build_gguf.py``; opened by
:meth:`gravity_ocr.GravityOCR.from_gguf`.

Format: standard GGUF v3 (``ggml/docs/gguf.md``; candle's reader opens it, see
``gguf_info``).  Tensors keep the checkpoint names; the two vision weights the runtime keeps in
another layout (``vision.RESHAPED``) are stored in the runtime layout (``gravity.reshaped``).
Metadata ``gravity.config`` / ``gravity.block_diffusion`` / ``gravity.tokenizer`` carry the
json files of the checkpoint, ``gravity.plan`` the format of every matrix.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32

# ggml.h type ids and (values per block, bytes per block)
GGML = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8, "q3_k": 11, "q4_k": 12, "q5_k": 13, "q6_k": 14, "bf16": 30}
GGML_NAMES = {v: k for k, v in GGML.items()}
BLOCK = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q4_0": (32, 18), "q8_0": (32, 34), "q3_k": (256, 110),
         "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210)}  # fmt: skip
DENSE = ("f32", "f16", "bf16")

# gguf metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i", _F32: "<f", _BOOL: "<?", _U64: "<Q",
           _I64: "<q", _F64: "<d"}  # fmt: skip


def nbytes(kind: str, shape) -> int:
    """Bytes of a tensor of ``shape`` in GGML type ``kind`` (rows must hold whole blocks)."""
    bs, ts = BLOCK[kind]
    n = int(np.prod(shape))
    if shape[-1] % bs:
        raise ValueError(f"{kind}: rows of {shape[-1]} values are not whole {bs}-value blocks")
    return n // bs * ts


def fits(kind: str, shape) -> bool:
    return shape[-1] % BLOCK[kind][0] == 0


# ------------------------------------------------------------------------------------- writing
def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", _BOOL) + struct.pack("<?", v)
    if isinstance(v, int):
        if 0 <= v < 1 << 32:
            return struct.pack("<I", _U32) + struct.pack("<I", v)
        return struct.pack("<I", _I64) + struct.pack("<q", v)
    if isinstance(v, float):
        return struct.pack("<I", _F32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", _STR) + _str(v)
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return struct.pack("<IIQ", _ARR, _STR, len(v)) + b"".join(_str(x) for x in v)
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return struct.pack("<IIQ", _ARR, _I64, len(v)) + b"".join(struct.pack("<q", x) for x in v)
    raise TypeError(f"no GGUF value type for {type(v).__name__}")


@dataclass
class Entry:
    """One tensor to write: ``produce()`` returns exactly ``nbytes(kind, shape)`` bytes."""

    name: str
    shape: tuple
    kind: str
    produce: Callable[[], object]

    @property
    def nbytes(self) -> int:
        return nbytes(self.kind, self.shape)


def write(path: str | Path, meta: dict, entries: list[Entry], log=None) -> int:
    """Write a GGUF v3 file; tensors are produced one at a time (never all in memory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"general.alignment": ALIGNMENT, **meta}
    head = bytearray(MAGIC + struct.pack("<IQQ", VERSION, len(entries), len(meta)))
    for k, v in meta.items():
        head += _str(k) + _value(v)
    offsets, off = [], 0
    for e in entries:
        off = (off + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        offsets.append(off)
        dims = [int(d) for d in reversed(e.shape)]  # ggml order: the fastest dimension first
        head += _str(e.name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<IQ", GGML[e.kind], off)
        off += e.nbytes
    head += b"\0" * (-len(head) % ALIGNMENT)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(head)
        base = len(head)
        for e, o in zip(entries, offsets):
            f.write(b"\0" * (base + o - f.tell()))
            data = e.produce()
            buf = memoryview(np.ascontiguousarray(data)).cast("B") if isinstance(data, np.ndarray) else memoryview(data)
            if buf.nbytes != e.nbytes:
                raise ValueError(f"{e.name}: {buf.nbytes} bytes produced, {e.nbytes} expected for {e.kind} {e.shape}")
            f.write(buf)
            if log:
                log(e)
        total = f.tell()
    os.replace(tmp, path)
    return total


# ------------------------------------------------------------------------------------- reading
class GgufFile:
    """Read-only view of a GGUF file (memory-mapped), with the interface of the other weight
    sources: ``shape`` / ``scheme`` / ``parts`` / ``get`` / ``rows`` / ``release_pages``."""

    reshaped = True  # vision.RESHAPED weights are stored in the runtime layout

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        mm = self._mm
        pos = 0

        def take(fmt: str):
            nonlocal pos
            size = struct.calcsize(fmt)
            out = struct.unpack(fmt, bytes(mm[pos : pos + size]))
            pos += size
            return out if len(out) > 1 else out[0]

        def string() -> str:
            nonlocal pos
            n = take("<Q")
            s = bytes(mm[pos : pos + n]).decode("utf-8")
            pos += n
            return s

        def value(t: int):
            if t == _STR:
                return string()
            if t == _ARR:
                et, n = take("<I"), take("<Q")
                return [value(et) for _ in range(n)]
            return take(_SCALAR[t])

        if bytes(mm[:4]) != MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        pos = 4
        version, n_tensors, n_kv = take("<IQQ")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        self.meta: dict = {}
        for _ in range(n_kv):
            k = string()
            self.meta[k] = value(take("<I"))
        infos = []
        for _ in range(n_tensors):
            name = string()
            nd = take("<I")
            dims = [take("<Q") for _ in range(nd)]
            kind, off = take("<IQ")
            infos.append((name, tuple(reversed(dims)), GGML_NAMES.get(kind, f"type{kind}"), off))
        align = int(self.meta.get("general.alignment", ALIGNMENT))
        base = (pos + align - 1) // align * align
        self.tensors: dict[str, dict] = {}
        for name, shape, kind, off in infos:
            n = nbytes(kind, shape) if kind in BLOCK else 0
            self.tensors[name] = {"shape": shape, "kind": kind, "offset": base + off, "nbytes": n}

    # -- metadata ---------------------------------------------------------------
    def meta_json(self, key: str, default=None):
        v = self.meta.get(key)
        return default if v is None else json.loads(v)

    # -- the weight-source interface ------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def names(self) -> list[str]:
        return list(self.tensors)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.tensors[name]["shape"])

    def kind(self, name: str) -> str:
        """The GGML type of a tensor (``q4_k``, ``bf16`` …)."""
        return self.tensors[name]["kind"]

    def is_ggml(self, name: str) -> bool:
        """A block-quantised tensor (uploaded with ``Engine.set_ggml``, not as codec parts)."""
        return name in self.tensors and self.kind(name) not in DENSE

    def scheme(self, name: str) -> str:
        k = self.kind(name)
        return k if k in DENSE else "ggml:" + k

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return self._mm[t["offset"] : t["offset"] + t["nbytes"]]

    def stored_bytes(self, name: str) -> int:
        return int(self.tensors[name]["nbytes"])

    def parts(self, name: str) -> dict[str, np.ndarray]:
        k = self.kind(name)
        if k not in DENSE:
            raise TypeError(f"{name} is {k}: GGML blocks go to the GPU engine as they are (Engine.set_ggml)")
        buf = self.raw(name)
        return {"w": buf.view("<f4") if k == "f32" else buf.view("<u2")}

    def get(self, name: str) -> np.ndarray:
        """Float32 values (GGML blocks decoded by the GPU library on the CPU)."""
        k, shape = self.kind(name), self.shape(name)
        buf = self.raw(name)
        if k == "f32":
            return np.array(buf.view("<f4")).reshape(shape)
        if k == "bf16":
            return (buf.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if k == "f16":
            return buf.view("<f2").astype(np.float32).reshape(shape)
        return dequant(k, shape, buf)

    def rows(self, name: str, ids) -> np.ndarray:
        """Rows ``ids`` of a matrix as float32, read only where needed (the embedding table)."""
        k, (nrows, ncols) = self.kind(name), self.shape(name)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= nrows):
            raise IndexError(f"row out of range for {name}")
        if k in DENSE:
            width = 4 if k == "f32" else 2
            table = self.raw(name).reshape(nrows, ncols * width)
            sel = np.ascontiguousarray(table[ids]).reshape(-1)
            if k == "f32":
                return sel.view("<f4").reshape(ids.size, ncols).copy()
            if k == "bf16":
                return (sel.view("<u2").astype(np.uint32) << 16).view(np.float32).reshape(ids.size, ncols)
            return sel.view("<f2").astype(np.float32).reshape(ids.size, ncols)
        rb = nbytes(k, (1, ncols))
        sel = np.ascontiguousarray(self.raw(name).reshape(nrows, rb)[ids]).reshape(-1)
        return dequant(k, (ids.size, ncols), sel)

    def release_pages(self, name: str | None = None) -> bool:
        """Hand the pages read so far – of tensor ``name``, or of the whole file – back to the
        system's file cache (``memory.release_mapped``)."""
        from .memory import release_mapped

        return release_mapped(self._mm if name is None else self.raw(name), self._mm)


def dequant(kind: str, shape, data: np.ndarray) -> np.ndarray:
    """GGML blocks as float32 (through the GPU library's CPU code; no device needed)."""
    from .gpu import library

    lib = library()
    if lib is None or not lib.has_ggml:
        raise RuntimeError("decoding GGML blocks needs the gravity_gpu library (python tools/build_kernels.py)")
    rows, cols = int(shape[0]), int(np.prod(shape[1:]))
    data = np.ascontiguousarray(data, dtype=np.uint8)
    out = np.empty(rows * cols, np.float32)
    lib.check(lib.lib.gg_ggml_dequant(GGML[kind], rows, cols, data.ctypes.data, data.nbytes, out.ctypes.data), f"dequant {kind}")
    return out.reshape(shape)


def quantize(kind: str, parts: dict, scheme: str, shape, imatrix: np.ndarray | None = None) -> np.ndarray:
    """A tensor (codec ``parts`` in ``scheme``) as GGML ``kind`` blocks (CPU, rows in parallel)."""
    from .gpu import library
    from .native import Linear

    lib = library()
    if lib is None or not lib.has_ggml:
        raise RuntimeError("GGML quantisation needs the gravity_gpu library (python tools/build_kernels.py)")
    lin = Linear(parts, scheme, shape)
    out = np.empty(nbytes(kind, (lin.rows, lin.cols)), np.uint8)
    w = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
    rc = lib.lib.gg_ggml_quantize(lin.addr, GGML[kind], None if w is None else w.ctypes.data, out.ctypes.data, out.nbytes)
    lib.check(rc, f"quantize {kind}")
    return out


def check_with_candle(path: str | Path) -> tuple[int, int]:
    """``(tensors, metadata entries)`` as candle's GGUF reader sees the file."""
    import ctypes

    from .gpu import library

    lib = library()
    if lib is None or not lib.has_ggml:
        raise RuntimeError("needs the gravity_gpu library")
    t, k = ctypes.c_uint64(), ctypes.c_uint64()
    lib.check(lib.lib.gg_gguf_info(str(path).encode("utf-8"), ctypes.byref(t), ctypes.byref(k)), "gguf info")
    return int(t.value), int(k.value)
