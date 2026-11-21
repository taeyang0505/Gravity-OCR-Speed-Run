"""Single-file model package (``.gocr``) and the original-weights reader.

Layout of a package::

    magic   8 bytes  b"GOCRPK01"
    data    tensor parts, each aligned to 64 bytes
    index   UTF-8 JSON  {"meta": {...}, "tensors": {name: {...}}}
    tail    u64 index offset, u64 index length, b"GOCRPK01"

The index sits at the end so the writer is a single pass.  The reader maps the
file once (``np.memmap``); a tensor is only touched when it is asked for, and
:meth:`Package.rows` gathers individual rows of a matrix straight from the map,
which is how the full 59,283-row input embedding stays on disk.

Besides tensors the package carries opaque blobs (``tokenizer.json``, tables)
under the scheme ``"blob"``.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import quant

MAGIC = b"GOCRPK01"
ALIGN = 64


class PackageWriter:
    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(MAGIC)
        self._pos = len(MAGIC)
        self.meta: dict = dict(meta or {})
        self.tensors: dict[str, dict] = {}

    def _write(self, arr: np.ndarray) -> int:
        pad = (-self._pos) % ALIGN
        if pad:
            self._f.write(b"\0" * pad)
            self._pos += pad
        off = self._pos
        raw = np.ascontiguousarray(arr)
        self._f.write(raw.tobytes())
        self._pos += raw.nbytes
        return off

    def add_parts(self, name: str, parts: dict[str, np.ndarray], scheme: str, shape) -> int:
        if name in self.tensors:
            raise KeyError(f"duplicate tensor {name}")
        entry = {"scheme": scheme, "shape": [int(s) for s in shape], "parts": {}}
        total = 0
        for key, arr in parts.items():
            arr = np.asarray(arr)
            entry["parts"][key] = {
                "dtype": arr.dtype.str,
                "count": int(arr.size),
                "offset": self._write(arr),
            }
            total += arr.nbytes
        entry["bytes"] = total
        self.tensors[name] = entry
        return total

    def add(self, name: str, array: np.ndarray, scheme: str) -> int:
        """Quantise ``array`` with ``scheme`` and store it; returns stored bytes."""
        array = np.asarray(array)
        return self.add_parts(name, quant.quantize(array, scheme), scheme, array.shape)

    def add_array(self, name: str, array: np.ndarray) -> int:
        """Store an array verbatim in its own dtype (integer tables, f32 bases)."""
        array = np.asarray(array)
        return self.add_parts(name, {"w": array}, "raw", array.shape)

    def add_blob(self, name: str, data: bytes) -> int:
        buf = np.frombuffer(data, dtype=np.uint8)
        return self.add_parts(name, {"w": buf}, "blob", [buf.size])

    def close(self) -> None:
        index = json.dumps({"meta": self.meta, "tensors": self.tensors}, ensure_ascii=False).encode("utf-8")
        pad = (-self._pos) % ALIGN
        self._f.write(b"\0" * pad)
        off = self._pos + pad
        self._f.write(index)
        self._f.write(struct.pack("<QQ", off, len(index)))
        self._f.write(MAGIC)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._f.closed:
            self.close()


class Package:
    """Read-only view of a ``.gocr`` file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        if bytes(self._mm[:8]) != MAGIC or bytes(self._mm[-8:]) != MAGIC:
            raise ValueError(f"{path} is not a GOCR package")
        off, length = struct.unpack("<QQ", bytes(self._mm[-24:-8]))
        index = json.loads(bytes(self._mm[off : off + length]).decode("utf-8"))
        self.meta: dict = index["meta"]
        self.tensors: dict[str, dict] = index["tensors"]

    def release_pages(self, name: str | None = None) -> bool:
        """Hand the pages read so far – of tensor ``name``, or of the whole file – back to the
        system's file cache (``memory.release_mapped``): once the weights are on a GPU this
        process need not keep them resident."""
        from .memory import release_mapped

        if name is None:
            return release_mapped(self._mm)
        return all([release_mapped(self._part(name, k), self._mm) for k in self.tensors[name]["parts"]])

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def names(self) -> list[str]:
        return list(self.tensors)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.tensors[name]["shape"])

    def scheme(self, name: str) -> str:
        return self.tensors[name]["scheme"]

    def _part(self, name: str, key: str) -> np.ndarray:
        p = self.tensors[name]["parts"][key]
        dt = np.dtype(p["dtype"])
        off = p["offset"]
        return self._mm[off : off + p["count"] * dt.itemsize].view(dt)

    def parts(self, name: str) -> dict[str, np.ndarray]:
        return {k: self._part(name, k) for k in self.tensors[name]["parts"]}

    def get(self, name: str) -> np.ndarray:
        """Float32 tensor (dequantised) or the stored array for ``raw`` entries."""
        entry = self.tensors[name]
        scheme = entry["scheme"]
        if scheme == "blob":
            raise TypeError(f"{name} is a blob; use blob()")
        if scheme == "raw":
            return np.asarray(self._part(name, "w")).reshape(entry["shape"])
        return quant.dequantize(self.parts(name), scheme, entry["shape"])

    def rows(self, name: str, ids) -> np.ndarray:
        """Rows ``ids`` of a matrix as float32, read from disk only where needed."""
        entry = self.tensors[name]
        scheme = entry["scheme"]
        nrows, ncols = entry["shape"]
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= nrows):
            raise IndexError(f"row out of range for {name}")
        if scheme in ("f32", "f16", "raw"):
            w = self._part(name, "w").reshape(nrows, ncols)
            return np.asarray(w[ids], dtype=np.float32)
        if scheme == "q8":
            if ncols % 32:
                raise ValueError("q8 row gather needs a width that is a multiple of 32")
            q = self._part(name, "q").view(np.int8).reshape(nrows, ncols)
            d = self._part(name, "d").reshape(nrows, ncols // 32)
            out = np.asarray(q[ids], dtype=np.float32).reshape(ids.size, ncols // 32, 32)
            out *= np.asarray(d[ids], dtype=np.float32)[:, :, None]
            return out.reshape(ids.size, ncols)
        if scheme in quant.SCHEMES and ncols % quant.SUPER == 0:
            # every part holds the same number of entries per 256-value superblock: gather the
            # superblocks of the wanted rows and decode only those (bit-identical to the whole decode)
            per_row = ncols // quant.SUPER
            n_super = nrows * per_row
            blocks = (ids[:, None] * per_row + np.arange(per_row)[None, :]).reshape(-1)
            sub = {}
            for key in entry["parts"]:
                part = self._part(name, key)
                sub[key] = np.ascontiguousarray(part.reshape(n_super, part.size // n_super)[blocks].reshape(-1))
            return quant.dequantize(sub, scheme, (ids.size, ncols))
        raise ValueError(f"row gather is not supported for scheme {scheme}")

    def blob(self, name: str) -> bytes:
        if self.tensors[name]["scheme"] != "blob":
            raise TypeError(f"{name} is not a blob")
        return bytes(self._part(name, "w"))

    def stored_bytes(self, name: str) -> int:
        return int(self.tensors[name]["bytes"])


class SafetensorsFile:
    """Minimal reader for the original ``model.safetensors`` (BF16/F16/F32).

    The format is an 8-byte little-endian header length, a JSON header and raw
    little-endian data; no third-party code is needed.  numpy has no bfloat16,
    so BF16 is widened by placing the 16 bits in the upper half of a float32.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        (hlen,) = struct.unpack("<Q", bytes(self._mm[:8]))
        header = json.loads(bytes(self._mm[8 : 8 + hlen]).decode("utf-8"))
        self.metadata = header.pop("__metadata__", {})
        self._base = 8 + hlen
        self.tensors: dict[str, dict] = header

    def release_pages(self, name: str | None = None) -> bool:
        """Hand the pages read so far – of tensor ``name``, or of the whole file – back to the
        system's file cache (``memory.release_mapped``): once the weights are on a GPU this
        process need not keep them resident."""
        from .memory import release_mapped

        if name is None:
            return release_mapped(self._mm)
        return release_mapped(self._raw(name)[0], self._mm)

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def names(self) -> list[str]:
        return list(self.tensors)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.tensors[name]["shape"])

    def scheme(self, name: str) -> str:
        return {"BF16": "bf16", "F16": "f16", "F32": "f32"}[self.tensors[name]["dtype"]]

    def _raw(self, name: str) -> tuple[np.ndarray, str, tuple[int, ...]]:
        t = self.tensors[name]
        a, b = t["data_offsets"]
        return self._mm[self._base + a : self._base + b], t["dtype"], tuple(t["shape"])

    @staticmethod
    def _widen(buf: np.ndarray, dtype: str) -> np.ndarray:
        if dtype == "BF16":
            return (buf.view("<u2").astype(np.uint32) << 16).view(np.float32)
        if dtype == "F16":
            return buf.view("<f2").astype(np.float32)
        if dtype == "F32":
            return np.array(buf.view("<f4"))
        raise TypeError(f"unsupported dtype {dtype}")

    def get(self, name: str) -> np.ndarray:
        buf, dtype, shape = self._raw(name)
        return self._widen(buf, dtype).reshape(shape)

    def parts(self, name: str) -> dict[str, np.ndarray]:
        """The stored values as they lie in the file (BF16/F16 as uint16), for the native kernels."""
        buf, dtype, _ = self._raw(name)
        return {"w": buf.view("<u2") if dtype in ("BF16", "F16") else buf.view("<f4")}

    def rows(self, name: str, ids) -> np.ndarray:
        buf, dtype, shape = self._raw(name)
        width = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        table = buf.reshape(shape[0], shape[1] * width)
        return self._widen(np.ascontiguousarray(table[ids]).reshape(-1), dtype).reshape(ids.size, shape[1])
