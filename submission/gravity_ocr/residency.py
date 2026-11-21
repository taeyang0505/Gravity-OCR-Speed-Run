"""Device memory of the GPU engine: which weight groups are on the device, in which precision,
and the KV caches – chosen from the memory the machine has *now*, never past it.

The structure follows blueprint-sandbox ``core/src/ai/residency.rs`` (no code taken from it):

* the unit is a **group** of weights (``vision``: the CogViT tower, ``language``: the GLM layers
  and the output matrix); a **phase** needs one group – reading an image the vision tower,
  writing the text the language model;
* with the ``swap`` policy (default) entering a phase drops every other group **before** the
  new one is uploaded, and the device memory pool is settled (synchronize → workspaces → pool
  trim) so the driver really has the memory back.  The vision tower is therefore off the
  device while the text is decoded, and the language model while an image is encoded;
  ``auto`` keeps the other group as long as everything fits;
* every upload, KV cache and cache growth is checked against the free device memory minus a
  headroom for activations.  A group that does not fit in its preferred precision is stored in
  a smaller one (float32 → the bf16 / f16 of the checkpoint → FP8 E4M3 with a scale per
  segment of at most 1024 values) and a cache in a smaller format (float32 → FP8 → TurboQuant
  4 / 3 bit); when nothing fits a :class:`GpuMemoryError` is raised *before* the allocation – on
  Windows the driver would otherwise spill silently into system memory, or fail mid-pass;
* the precisions and KV formats are switched on only after they passed a check against numpy
  on this device (:func:`probe`);
* after an upload, the pages of the memory-mapped checkpoint that it read leave the process's
  working set (:func:`gravity_ocr.memory.release_mapped`: they stay in the system file cache for
  the next swap); measured on Windows, each swap otherwise left 0.8 – 1.1 GB of checkpoint pages
  counted as memory in use.

Environment (all optional):

``GRAVITY_OCR_RESIDENCY``        ``swap`` (default) | ``auto``
``GRAVITY_OCR_WEIGHTS``          ``auto`` (default) | ``f32`` | ``half`` | ``fp8``: preferred storage
``GRAVITY_OCR_HALF_COMPUTE``     ``native`` (default: bf16 / f16 matmul, f32 accumulation) |
                                 ``f32`` (widen to f32 first: bit-exact, slower)
``GRAVITY_OCR_KV``               ``auto`` (default: float32 while the cache takes at most a quarter of
                                 the free device memory, else FP8, then TurboQuant 4 / 3 bit) |
                                 ``f32`` | ``fp8`` | ``turbo4`` | ``turbo3`` (the first one tried)
``GRAVITY_OCR_VRAM_HEADROOM_MB`` device memory kept free for activations (default 512)
``GRAVITY_OCR_VRAM_BUDGET_MB``   upper bound of what the weights and caches may take
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import memory
from .ops import F32

MB = 1 << 20
STORES = {"f32": 0, "half": 1, "fp8": 2, "nvfp4": 3, "mxfp8": 4, "mxfp4": 5}
KV_FORMATS = {"f32": 0, "fp8": 1, "turbo4": 2, "turbo3": 3, "nvfp4": 4, "mxfp8": 5, "mxfp4": 6}
# bytes a position (all layers, K and V): 128 KB, 35 / 34 KB, 20 / 19 / 18 KB, 14 KB
KV_ORDER = ("f32", "mxfp8", "fp8", "nvfp4", "mxfp4", "turbo4", "turbo3")  # largest first
POLICIES = ("swap", "auto")
KV_SHARE = 0.25  # KV "auto": a float32 cache may take this share of the free device memory
DEFAULT_HEADROOM_MB = 512
# vision activations per patch row (float32: residual, qkv, q/k/v and their head-major copies,
# gate/up/silu of the MLP), measured peak ≈ 100 KB; a margin on top
VIS_ROW_BYTES = 128 << 10
VIS_CHUNK = 256 * MB  # rust_gpu Engine::chunk_bytes default: scores of one vision attention chunk
VIS_MIN_CHUNK = 16 * MB
CODEC_MAX_ROW = 1024
HALF_SCHEMES = ("bf16", "f16")

ENV_POLICY = "GRAVITY_OCR_RESIDENCY"
ENV_WEIGHTS = "GRAVITY_OCR_WEIGHTS"
ENV_COMPUTE = "GRAVITY_OCR_HALF_COMPUTE"
ENV_KV = "GRAVITY_OCR_KV"
ENV_HEADROOM = "GRAVITY_OCR_VRAM_HEADROOM_MB"
ENV_BUDGET = "GRAVITY_OCR_VRAM_BUDGET_MB"


class GpuMemoryError(MemoryError):
    """Not enough device memory for a group, a cache or a pass – raised before allocating."""


def _env(name: str, default: str, allowed) -> str:
    v = os.environ.get(name, "").strip().lower() or default
    if v not in allowed:
        warnings.warn(f"gravity_ocr: {name}={v!r} is not one of {sorted(allowed)}; using {default!r}", stacklevel=3)
        return default
    return v


def _env_mb(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) * MB if v else None
    except ValueError:
        warnings.warn(f"gravity_ocr: {name}={v!r} is not a number of MB", stacklevel=3)
        return None


# weight / KV codecs (candle accel::formats): (bits per value, values per scale block)
CODECS = {"fp8": (8, 0), "mxfp8": (8, 32), "nvfp4": (4, 16), "mxfp4": (4, 32)}


def codec_segment(cols: int, fmt: str = "fp8") -> int | None:
    """Segment length of a coded weight row (rust_gpu ``codec_segment``): the largest divisor of
    ``cols`` up to 1024 holding whole scale blocks, at least 64 unless the row is shorter."""
    block = CODECS[fmt][1] or 1
    for s in range(min(cols, CODEC_MAX_ROW), 0, -1):
        if cols % s == 0 and s % block == 0 and (s >= 64 or s == cols):
            return s
    return None


def fp8_segment(cols: int) -> int | None:
    return codec_segment(cols, "fp8")


def codec_row_words(fmt: str, dim: int) -> int:
    """u32 words of one codec row of ``dim`` values (``accel::formats::row_layout``)."""
    bits, block = CODECS[fmt]
    aux = -(-(dim // block) // 4) if block else 0
    return -(-dim * bits // 32) + aux + 2


@dataclass
class Spec:
    """One tensor of a group as it will be uploaded."""

    shape: tuple
    scheme: str
    matrix: bool
    packed: int = 0  # bytes of the packed parts (custom ROCm path)
    name: str = ""

    def bytes(self, store: str, custom: bool) -> int:
        n = int(np.prod(self.shape))
        if not self.matrix:
            return n * 4
        if self.scheme.startswith("ggml:"):
            return self.packed + 512  # GGML blocks stay as they are (+ candle's row padding)
        if custom and self.packed and self.scheme not in ("f32", *HALF_SCHEMES):
            return self.packed + 16 * 5  # stays packed on the device
        half = self.scheme in HALF_SCHEMES
        if store in CODECS:
            rows, cols = self.shape[0], int(np.prod(self.shape[1:]))
            seg = codec_segment(cols, store)
            if seg is not None:
                return rows * (cols // seg) * codec_row_words(store, seg) * 4
            store = "half"
        if store == "half" and half:
            return n * 2
        return n * 4


@dataclass
class Group:
    name: str
    prefixes: list
    load: object  # () -> None: uploads every tensor (the engine's weight_store is set)
    specs: object  # () -> list[Spec]


@dataclass
class Transition:
    action: str  # load | drop
    group: str
    store: str
    bytes: int
    free_before: int
    free_after: int
    seconds: float


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)):
        return float("inf")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# relative error a precision / KV format may have on the probes to be switched on
# (GGML: against its own decoded blocks, so only the Q8_1 rounding of the activations counts)
# (weights on random matrices: FP8 2.1 %, MXFP8 2.6 %, NVFP4 8.8 %, MXFP4 9.4 %; KV attention: FP8 3.6 %, MXFP8
# 4.2 %, NVFP4 13.5 %, MXFP4 16.3 %, TurboQuant 4 bit 13.8 %, 3 bit 25.6 % - RTX 3050)
TOL = {"half": 1e-5, "half_native": 2e-2, "fp8": 6e-2, "mxfp8": 6e-2, "nvfp4": 0.12, "mxfp4": 0.13,
       "kv_fp8": 5e-2, "kv_mxfp8": 6e-2, "kv_nvfp4": 0.16, "kv_mxfp4": 0.18, "kv_turbo4": 0.15, "kv_turbo3": 0.25,
       "ggml": 2e-2}  # fmt: skip


WEIGHT_PROBES = {"half": (1, 0), "half_native": (1, 1), "fp8": (2, 0), "nvfp4": (3, 0), "mxfp8": (4, 0), "mxfp4": (5, 0)}  # (weight_store, half_compute)


def probe_weights(e, key: str) -> float:
    """Relative error of a bf16 matrix product through weight path ``key`` (``half``: stored
    bf16, widened to f32; ``half_native``: bf16 matmul; ``fp8``) against numpy."""
    from .gpu import Linear

    store, compute = WEIGHT_PROBES[key]
    rng = np.random.default_rng(20261118)
    shape, n = (40, 1536), 5
    w = (rng.standard_normal(shape) * 0.05).astype(F32)
    raw = (w.view(np.uint32) >> 16).astype(np.uint16)
    wq = (raw.astype(np.uint32) << 16).view(F32)
    x = rng.standard_normal((n, shape[1])).astype(F32)
    bias = rng.standard_normal(shape[0]).astype(F32)
    want = x.astype(np.float64) @ wq.T.astype(np.float64) + bias
    lin = Linear({"w": raw}, "bf16", shape)
    keep = (e.info.get("weight_store"), e.info.get("half_compute"))
    try:
        e.set_option("weight_store", store)
        e.set_option("half_compute", compute)
        out = np.empty((n, shape[0]), F32)
        e.lib.check(e.lib.lib.gg_test_linear(e.ptr, lin.addr, x.ctypes.data, n, bias.ctypes.data, out.ctypes.data), key)
        return _rel(out, want)
    finally:
        e.set_option("weight_store", {"f32": 0, "half": 1, "fp8": 2}.get(keep[0], 0))
        e.set_option("half_compute", int(keep[1] or 0))


def probe_kv(e, name: str) -> float:
    """Largest relative error of the attention through a ``name`` KV cache against the float32
    cache (single rows, a few rows, a draft block without mask, a cache without old rows)."""
    from . import language as L

    lib = e.lib.lib

    def out_of(fmt: int, past: int, t: int, causal: bool, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        pos = np.stack([np.arange(past, past + t)] * 3, axis=1)
        cos, sin = L.rope_tables(pos)
        q = r.standard_normal((t, 2048)).astype(F32)
        k = r.standard_normal((t, 1024)).astype(F32)
        v = r.standard_normal((t, 1024)).astype(F32)
        ko = r.standard_normal((max(past, 1), 1024)).astype(F32)
        vo = r.standard_normal((max(past, 1), 1024)).astype(F32)
        out = np.empty((t, 2048), F32)
        rc = lib.gg_test_kv_attention(e.ptr, fmt, q.ctypes.data, k.ctypes.data, v.ctypes.data, cos.ctypes.data, sin.ctypes.data,
                                      ko.ctypes.data, vo.ctypes.data, past, t, 1 if causal else 0, out.ctypes.data)  # fmt: skip
        e.lib.check(rc, f"kv attention format {fmt}")
        return out

    cases = ((40, 1, True), (37, 5, True), (100, 32, False), (0, 6, True))
    return max(_rel(out_of(KV_FORMATS[name], p, t, c, i), out_of(0, p, t, c, i)) for i, (p, t, c) in enumerate(cases))


def probe_ggml(e, kind: str) -> float:
    """Largest relative error of a GGML ``kind`` matrix multiplied on the device (as blocks:
    MMVQ for a few rows, MMQ for many, activations as Q8_1) against the decoded blocks in numpy."""
    from . import gguf

    rng = np.random.default_rng(20261119)
    shape = (48, 1536)
    w = (rng.standard_normal(shape) * 0.05).astype(F32)
    blocks = gguf.quantize(kind, {"w": w}, "f32", shape)
    wq = gguf.dequant(kind, shape, blocks).astype(np.float64)
    worst = 0.0
    for n in (1, 5, 40):
        x = rng.standard_normal((n, shape[1])).astype(F32)
        out = np.empty((n, shape[0]), F32)
        rc = e.lib.lib.gg_test_ggml_linear(e.ptr, gguf.GGML[kind], shape[0], shape[1], blocks.ctypes.data, blocks.nbytes,
                                           x.ctypes.data, n, out.ctypes.data)  # fmt: skip
        e.lib.check(rc, f"ggml linear {kind}")
        worst = max(worst, _rel(out, x.astype(np.float64) @ wq.T))
    return worst


class Caps:
    """Weight paths and KV formats that passed their probe on an engine, probed on first use
    (the TurboQuant / FP8 kernels are compiled at their first launch: probing only what is
    used keeps the model load short)."""

    def __init__(self, engine):
        self.e = engine
        self.errors: dict[str, float | str] = {}
        self.ok: dict[str, bool] = {"kv_f32": True}

    def __call__(self, key: str) -> bool:
        """``half`` / ``half_native`` / ``fp8`` (weights) or ``kv_<format>``."""
        if key not in self.ok:
            if not self.e.lib.has_memory:
                self.ok[key] = False
                return False
            try:
                if key.startswith("ggml_"):
                    if not self.e.lib.has_ggml:
                        raise RuntimeError("library without GGML support")
                    err = probe_ggml(self.e, key[5:])
                    tol = TOL["ggml"]
                elif key.startswith("kv_"):
                    err = probe_kv(self.e, key[3:])
                    tol = TOL[key]
                else:
                    err = probe_weights(self.e, key)
                    tol = TOL[key]
                self.errors[key] = err
                self.ok[key] = err <= tol
            except Exception as x:  # noqa: BLE001
                self.errors[key] = f"{type(x).__name__}: {x}"
                self.ok[key] = False
        return self.ok[key]

    def report(self) -> dict:
        return {k: (round(self.errors[k], 5) if isinstance(self.errors.get(k), float) else self.errors.get(k, 0.0)) if v else "off"
                for k, v in self.ok.items()}  # fmt: skip


def probe(e) -> dict:
    """Every weight path and KV format at once: ``{key: relative error or "off"}``."""
    caps = Caps(e)
    for key in (*WEIGHT_PROBES, *("kv_" + k for k in KV_ORDER[1:])):
        caps(key)
    return caps.report()


class Residency:
    """Which groups are on the engine, in which precision; caches and passes sized to fit."""

    def __init__(self, engine, policy: str | None = None, weights: str | None = None, half_compute: str | None = None,
                 kv: str | None = None, headroom: int | None = None, budget: int | None = None, caps: Caps | None = None,
                 release_pages=None):  # fmt: skip
        self.e = engine
        # called after every upload: the mapped checkpoint pages the upload read leave this
        # process's working set (they stay in the system file cache for the next swap)
        self.release_pages = release_pages
        self.policy = policy or _env(ENV_POLICY, "swap", POLICIES)
        self.weights = weights or _env(ENV_WEIGHTS, "auto", ("auto", "native", "pyramid", *STORES))
        compute = half_compute or _env(ENV_COMPUTE, "native", ("native", "f32"))
        self.kv = kv or _env(ENV_KV, "auto", ("auto", "native", *KV_FORMATS))
        # what the GPU computes natively (candle accel profile): FP8 from sm_89, FP4 / MX from
        # sm_100; its 8-bit (FP8, MXFP8 on Blackwell) and 4-bit (NVFP4) float formats
        prof = engine.info.get("profile") or {}
        self.native_fp8 = bool(prof.get("native_fp8"))
        self.native_fp4 = bool(prof.get("native_fp4"))
        self.fp8c = prof.get("fp8_format") if prof.get("fp8_format") in CODECS else "fp8"
        self.fp4c = prof.get("fp4_format") if prof.get("fp4_format") in CODECS else "nvfp4"
        self.arch = prof.get("arch", "")
        self.loading: str | None = None  # the store of the group being uploaded
        self.headroom = headroom if headroom is not None else (_env_mb(ENV_HEADROOM) or DEFAULT_HEADROOM_MB * MB)
        self.budget = budget if budget is not None else _env_mb(ENV_BUDGET)
        self.groups: dict[str, Group] = {}
        self.resident: dict[str, str] = {}  # group → store
        self.history: list[Transition] = []
        self.custom = bool(engine.info.get("custom"))
        self.caps = caps or Caps(engine)
        self.half_native = compute == "native" and self.caps("half_native")
        engine.set_option("half_compute", 1 if self.half_native else 0)
        self.vis_chunk = VIS_CHUNK

    # ------------------------------------------------------------------ bookkeeping
    def register(self, name: str, prefixes: list, load, specs) -> None:
        self.groups[name] = Group(name, list(prefixes), load, specs)

    def is_resident(self, name: str) -> bool:
        return name in self.resident

    def used(self) -> int:
        """Bytes of weights and live caches on the engine."""
        caches = 0
        for c in list(self.e._caches):
            if c.ptr:
                caches += c.nbytes
        return self.e.weight_bytes + caches

    def free(self) -> int:
        """Device memory this process may still take: the driver's free memory (plus the pool's
        idle bytes), bounded by the budget.  A device without a memory query counts as roomy."""
        free, total = self.e.mem()
        if total == 0:
            free = 1 << 50
        if self.budget is not None:
            free = min(free, max(self.budget - self.used(), 0))
        return int(free)

    def need(self, name: str, store: str) -> int:
        return sum(s.bytes(self.tensor_store(store, s.name), self.custom) for s in self.groups[name].specs())

    def tensor_store(self, store: str, name: str) -> str:
        """The store of one tensor: ``pyramid`` gives the measured 8-bit class (q8 / q6 / q5) the
        GPU's 8-bit float format and the 4-bit class (q4 / nf4 / q3) its 4-bit one (NVFP4)."""
        if store != "pyramid":
            return store
        from .pyramid import width_class

        return self.fp8c if width_class(name) == 8 else self.fp4c

    def store_order(self) -> list[str]:
        """Stores to try for a group, preferred first, smaller ones after (never a larger one):
        f32 -> half -> 8-bit float -> pyramid (8 / 4-bit by the measured allocation) -> 4-bit float.

        ``auto`` starts at half (bf16 matmul) or f32; ``native`` at what the GPU computes
        natively - the pyramid on FP4 GPUs (sm_100+), the 8-bit format on FP8 ones (sm_89+),
        like ``auto`` elsewhere.  The candle fork has no FP8 / FP4 tensor-core GEMM for weights:
        coded rows are decoded by chunks for the matmul, so these stores save memory, not time."""
        w = self.weights
        if w == "native":
            w = "pyramid" if self.native_fp4 else (self.fp8c if self.native_fp8 else "auto")
        if w == "auto":
            w = "half" if self.half_native else "f32"
        eight = w if w in ("fp8", "mxfp8") else self.fp8c
        four = w if w in ("nvfp4", "mxfp4") else self.fp4c
        ladder = ["f32", "half", eight, "pyramid", four]
        start = {"fp8": 2, "mxfp8": 2, "nvfp4": 4, "mxfp4": 4}.get(w, ladder.index(w) if w in ladder else 0)
        return ladder[start:]

    def store_ok(self, store: str) -> bool:
        if store == "f32":
            return True
        if store == "half":
            return self.half_native or self.caps("half")
        if store == "pyramid":
            return self.caps(self.fp8c) and self.caps(self.fp4c)
        return self.caps(store)

    def prepare(self, name: str) -> None:
        """Before tensor ``name`` of the group being loaded is uploaded: its store (pyramid)."""
        if self.loading == "pyramid":
            self.e.set_option("weight_store", STORES[self.tensor_store("pyramid", name)])

    def stores(self) -> list[str]:
        """The stores of ``store_order`` that work here (probing the ones not tried yet)."""
        return [s for s in self.store_order() if self.store_ok(s)] or ["f32"]

    def kv_order(self) -> list[str]:
        """KV formats to try, largest first.  ``auto``: float32 (while the cache stays under
        ``KV_SHARE`` of the free memory) on GPUs without native FP8, then the GPU's 8-bit float
        format, NVFP4 where FP4 is native, TurboQuant 4 / 3 bit; on FP8 GPUs (sm_89+) the 8-bit
        format first.  ``native``: the compressed formats only."""
        if self.kv in ("auto", "native"):
            tail = [self.fp8c] + (["nvfp4"] if self.native_fp4 else []) + ["turbo4", "turbo3"]
            return tail if self.kv == "native" or self.native_fp8 else ["f32", *tail]
        return list(KV_ORDER[KV_ORDER.index(self.kv) :])

    def kv_formats(self) -> list[str]:
        """The KV formats of ``kv_order`` that work here (probing all of them)."""
        return [f for f in self.kv_order() if self.caps("kv_" + f)] or ["f32"]

    # ------------------------------------------------------------------ groups
    def _drop(self, name: str) -> None:
        g = self.groups[name]
        free0 = self.free()
        t0 = time.perf_counter()
        freed = sum(self.e.drop(p) for p in g.prefixes)
        store = self.resident.pop(name)
        self.history.append(Transition("drop", name, store, freed, free0, self.free(), time.perf_counter() - t0))

    def _drop_others(self, keep: str | None) -> bool:
        others = [g for g in self.resident if g != keep]
        for g in others:
            self._drop(g)
        return bool(others)

    def enter(self, name: str) -> None:
        """Make group ``name`` resident (dropping the others first under ``swap``)."""
        if name in self.resident and (self.policy != "swap" or len(self.resident) == 1):
            return
        if self.policy == "swap":
            self._drop_others(name)
        if name in self.resident:
            return
        self._load(name)

    def _load(self, name: str) -> None:
        g = self.groups[name]
        # precision is lost last: the preferred store next to the other groups, then the other
        # groups dropped, then smaller stores (probed only when they are needed)
        dropped = False
        for store in self.store_order():
            need = self.need(name, store)
            if need + self.headroom > self.free() and not dropped:
                dropped = True
                self._drop_others(name)
            free = self.free()
            if need + self.headroom <= free and self.store_ok(store):
                if store != self.store_order()[0]:
                    warnings.warn(f"gravity_ocr: {name} weights stored as {store} ({need / MB:.0f} MB): "
                                  f"{free / MB:.0f} MB free on the device", stacklevel=4)  # fmt: skip
                self._upload(g, store, need, free)
                return
        smallest = min(self.need(name, s) for s in self.stores())
        raise GpuMemoryError(f"{name} weights need at least {smallest / MB:.0f} MB + {self.headroom / MB:.0f} MB headroom, "
                             f"{self.free() / MB:.0f} MB free on the device (close other programs using the GPU, "
                             f"or lower {ENV_HEADROOM})")  # fmt: skip

    def _upload(self, g: Group, store: str, need: int, free0: int) -> None:
        avail, _ = memory.system()
        # conversions run by row chunks of this many float32 bytes in host and device memory
        chunk = 64 * MB if avail > memory.ram_reserve() + 512 * MB else 16 * MB
        self.e.set_option("decode_bytes", chunk)
        self.e.set_option("weight_store", STORES[self.tensor_store(store, "")])
        t0 = time.perf_counter()
        before = self.e.weight_bytes
        self.loading = store
        try:
            g.load()
        except Exception:
            for p in g.prefixes:  # leave no half-loaded group behind
                self.e.drop(p)
            raise
        finally:
            self.loading = None
        # GGML blocks (a GGUF file) keep their own formats: the store only applies to the rest
        if any(s.scheme.startswith("ggml:") for s in g.specs()):
            store = "ggml"
        self.resident[g.name] = store
        self.e.trim()
        self.history.append(Transition("load", g.name, store, self.e.weight_bytes - before, free0, self.free(), time.perf_counter() - t0))
        if self.release_pages is not None:
            self.release_pages()
        memory.relieve()

    def apply(self, name: str, fn) -> None:
        """Run ``fn()`` (an upload of one tensor) with the store of resident group ``name``."""
        store = self.tensor_store(self.resident[name], "")  # pyramid: the 8-bit class
        self.e.set_option("weight_store", STORES.get(store, STORES["f32"]))  # "ggml": f32 for the rest
        fn()

    def release(self, name: str) -> None:
        if name in self.resident:
            self._drop(name)

    def release_all(self) -> None:
        for name in list(self.resident):
            self._drop(name)

    def feasible(self) -> bool:
        """Can every group fit on its own in its smallest store (nothing else resident)?"""
        free = self.free() + self.used()
        return all(any(self.need(n, s) + self.headroom <= free and self.store_ok(s) for s in self.store_order()) for n in self.groups)

    # ------------------------------------------------------------------ caches and passes
    def room(self, nbytes: int, keep: str) -> bool:
        """``nbytes`` + headroom free on the device, dropping the groups other than ``keep``
        when needed (whatever the policy: no allocation may run the device out of memory)."""
        if nbytes + self.headroom <= self.free():
            return True
        if self._drop_others(keep) and nbytes + self.headroom <= self.free():
            return True
        self.e.trim()
        return nbytes + self.headroom <= self.free()

    def cache(self, capacity: int):
        """A KV cache of ``capacity`` positions in the first format that fits.  ``auto`` takes
        float32 (the fastest: −13 % decode speed for FP8, held-out set) only while the cache
        stays under ``KV_SHARE`` of the free device memory, then FP8, then TurboQuant."""
        for fmt in self.kv_order():
            if not self.caps("kv_" + fmt):
                continue
            row = self.e.kv_row_bytes(KV_FORMATS[fmt])
            if self.kv == "auto" and fmt == "f32" and row * capacity > KV_SHARE * max(self.free() - self.headroom, 0):
                continue
            if self.room(row * capacity, "language"):
                self.e.set_option("kv_format", KV_FORMATS[fmt])
                return self.e.new_cache(capacity, row, guard=self._grow)
        raise GpuMemoryError(f"no room for a KV cache of {capacity} positions ({self.free() / MB:.0f} MB free on the device)")

    def _grow(self, want: int, need: int) -> int:
        """Bytes a cache may grow to: ``want`` (doubling) when it fits, else ``need``."""
        if self.room(want, "language"):
            return want
        if self.room(need, "language"):
            return need
        raise GpuMemoryError(f"no room to grow a KV cache to {need / MB:.0f} MB ({self.free() / MB:.0f} MB free on the device)")

    def vision_batches(self, sizes: list[int]) -> list[list[int]]:
        """Indices of the images of a vision batch cut into passes whose activations fit; the
        score chunk of the attention is shrunk when a single image needs it."""
        chunk = VIS_CHUNK
        room = self.free() - self.headroom // 2
        out, cur, rows = [], [], 0
        for i, n in enumerate(sizes):
            if cur and (rows + n) * VIS_ROW_BYTES + chunk > room:
                out.append(cur)
                cur, rows = [], 0
            cur.append(i)
            rows += n
        if cur:
            out.append(cur)
        worst = max(sum(sizes[i] for i in b) for b in out)
        if worst * VIS_ROW_BYTES + chunk > room:
            chunk = room - worst * VIS_ROW_BYTES
            if chunk < VIS_MIN_CHUNK:
                raise GpuMemoryError(f"an image of {worst} patches needs ≈{(worst * VIS_ROW_BYTES + VIS_MIN_CHUNK) / MB:.0f} MB "
                                     f"for the vision pass, {room / MB:.0f} MB free on the device")  # fmt: skip
        if chunk != self.vis_chunk:  # shrunk for this pass, or back to the default
            self.e.set_option("chunk_bytes", chunk)
            self.vis_chunk = chunk
        return out

    # ------------------------------------------------------------------ report
    def snapshot(self) -> dict:
        free, total = self.e.mem()
        avail, ram = memory.system()
        return {
            "policy": self.policy,
            "resident": dict(self.resident),
            "weight_mb": round(self.e.weight_bytes / MB, 1),
            "free_mb": round(free / MB, 1),
            "total_mb": round(total / MB, 1),
            "headroom_mb": round(self.headroom / MB, 1),
            "half_compute": "native" if self.half_native else "f32",
            "weights": self.weights,
            "gpu": {"arch": self.arch, "native_fp8": self.native_fp8, "native_fp4": self.native_fp4,
                    "fp8_format": self.fp8c, "fp4_format": self.fp4c},
            "kv": next((f for f in self.kv_order() if self.caps("kv_" + f)), "f32"),
            "ram_available_mb": round(avail / MB, 1),
            "ram_total_mb": round(ram / MB, 1),
            "caps": self.caps.report(),
            "transitions": len(self.history),
        }
