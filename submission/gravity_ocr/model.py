"""The assembled model: image → image tokens → prefill → next-token logits.

``device`` picks where it runs:

``"auto"``    the best that is there and passes its self-check: a GPU (CUDA,
              ROCm, Vulkan), else the native CPU kernels, else numpy
``"gpu"``     any GPU; ``"cuda"`` / ``"rocm"`` / ``"vulkan"`` one backend
``"cpu"``     the native CPU kernels on the packed weights (numpy if missing)
``"numpy"``   the float32 reference

A device that is asked for but not usable falls back down the same chain
with a warning; ``model.device`` says what was taken.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gpu, language, native, preprocess, vision
from .container import Package, SafetensorsFile
from .language import KVCache, LanguageModel
from .ops import F32
from .residency import GpuMemoryError, Residency
from .tokenizer import Vocab
from .vision import VisionTower

TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
PROMPT_HEAD = "[gMASK]<sop><|user|>\n<|begin_of_image|>"
PROMPT_TAIL = "<|end_of_image|>{prompt}<|assistant|>\n"
HEAD_IDS = "lm_head.ids"
REST = gpu.REST  # the rows the slice cut, kept to check that none of them wins
REST_IDS = "lm_head.rest_ids"
DEVICES = ("auto", "gpu", "cuda", "rocm", "vulkan", "cpu", "numpy")
GPU_DEVICES = ("auto", "gpu", "cuda", "rocm", "vulkan")


def cpu_kernels(device: str):
    """The native kernels for ``device`` when every model kernel passed its check."""
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}")
    if device == "numpy":
        return None
    k = native.load()
    if k is None or not k.has("codec", "matmul", "norm", "act", "lm_attention", "vis_attention"):
        return None
    return k


@dataclass
class Session:
    """Decoding state after a prefill: the cache and where the next token goes."""

    cache: KVCache
    next_pos: int
    logits: np.ndarray  # logits of the last prefill row (compact vocabulary)
    prompt_tokens: int
    image_tokens: int
    timing: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # committed output tokens
    rest: tuple | None = None  # (logit, token) of the best row cut from the head, for ``logits`` (see ``GravityOCR.rest_at``)

    def accept(self, tokens) -> None:
        """Commit ``tokens`` whose cache rows were written by an uncommitted pass."""
        n = len(tokens)
        self.cache.len += n
        self.next_pos += n
        self.history += list(tokens)


class GravityOCR:
    def __init__(self, src, vocab: Vocab, config: dict, block_diffusion: dict | None = None, device: str = "auto",
                 rescue: bool = True):  # fmt: skip
        self.src = src
        self.vocab = vocab
        self.config = config
        self.eos = set(config["text_config"]["eos_token_id"])
        bd = block_diffusion or {}
        self.bd_size = int(bd.get("bd_size", 32))
        self.mask_id = int(bd.get("mask_id", vocab.mask_id))
        self.ngram = None  # UnifiedNgram table (ngram.NgramTable); residuals after layer 0 when set
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        ids = np.asarray(src.get(HEAD_IDS)) if HEAD_IDS in src else None
        self.rest_lin, self._rest, self.rest_ids = None, None, None
        self.last_rest = None  # (logits, tokens) of the best cut row of every row of the last ``run``
        self.gpu = gpu.load(device) if device in GPU_DEVICES else None
        if device in GPU_DEVICES[1:] and self.gpu is None:
            warnings.warn(f"gravity_ocr: no usable {device} device; running on the CPU", stacklevel=2)
        self.kernels = cpu_kernels("cpu" if device != "numpy" else "numpy")
        # which weight groups are on the device, in which precision, and the KV caches
        # (gravity_ocr.residency): the vision tower is dropped while the text is decoded
        self.residency = None
        ggml = getattr(src, "is_ggml", None)
        self.ggml = bool(ggml and any(ggml(n) for n in src.names()))  # a GGUF pyramid model: GPU only
        if self.ggml and (self.gpu is None or not self.gpu.lib.has_ggml):
            raise RuntimeError("a GGUF model (GGML blocks) runs on a GPU engine with GGML support; "
                               "for the CPU use GravityOCR.from_pretrained / from_package")  # fmt: skip
        if self.gpu is not None and self.gpu.lib.has_memory:
            self.residency = Residency(self.gpu, release_pages=getattr(src, "release_pages", None))
        if self.ggml:
            # every GGML format of the file must pass its check on this device first
            from .residency import Caps

            caps = self.residency.caps if self.residency is not None else Caps(self.gpu)
            bad = sorted(k for k in {src.kind(n) for n in src.names() if ggml(n)} if not caps("ggml_" + k))
            if bad:
                raise RuntimeError(f"the {self.gpu.kind} engine failed its check of GGML {bad}: {caps.errors}")
        if self.gpu is not None:
            head = None if self.ggml and ggml(language.HEAD) else native.parts_of(src, language.HEAD)
            self.vision = gpu.GpuVision(self.gpu, src, self.residency)
            self.lm = gpu.GpuLanguage(self.gpu, src, head, self.residency)
            if self.residency is not None and not self.residency.feasible():
                if self.ggml:
                    raise GpuMemoryError(f"the {self.gpu.kind} device has too little free memory for the model "
                                         f"({self.residency.snapshot()['free_mb']} MB)")  # fmt: skip
                warnings.warn(f"gravity_ocr: the {self.gpu.kind} device has too little free memory for the model "
                              f"({self.residency.snapshot()['free_mb']} MB); running on the CPU", stacklevel=2)  # fmt: skip
                self.gpu.close()
                self.gpu, self.residency = None, None
        if self.gpu is not None:
            self.device = f"gpu-{self.gpu.kind}"
            self.head_lin, self._head = None, None
            self._set_head_ids(ids)
        else:
            self.device = "cpu-native" if self.kernels is not None else "numpy"
            self.vision = VisionTower(src, self.kernels)
            self.lm = LanguageModel(src, self.kernels)
            if self.kernels is not None:
                self.set_head_parts(*native.parts_of(src, language.HEAD), ids)
            else:
                self.set_head(src.get(language.HEAD), ids)
        if rescue and REST in src and REST_IDS in src:
            self.set_rest_parts(*native.parts_of(src, REST), np.asarray(src.get(REST_IDS)))

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, folder: str | Path, device: str = "auto") -> "GravityOCR":
        """Original weights (``model.safetensors`` + json files), no surgery."""
        folder = Path(folder)
        src = SafetensorsFile(folder / "model.safetensors")
        vocab = Vocab.from_file(folder / "tokenizer.json")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        bd_path = folder / "block_diffusion.json"
        bd = json.loads(bd_path.read_text(encoding="utf-8")) if bd_path.exists() else None
        return cls(src, vocab, config, bd, device)

    @classmethod
    def from_gguf(cls, path: str | Path, device: str = "auto") -> "GravityOCR":
        """A GGUF pyramid model written by ``tools/build_gguf.py``: every matrix in the GGML
        format its measured sensitivity allows, multiplied as it is on the GPU (no CPU path)."""
        from .gguf import GgufFile

        src = GgufFile(path)
        vocab = Vocab(src.meta["gravity.tokenizer"])
        return cls(src, vocab, src.meta_json("gravity.config"), src.meta_json("gravity.block_diffusion"), device)

    @classmethod
    def from_package(cls, path: str | Path, device: str = "auto", ngram_gate: float = 0.0, rescue: bool = True) -> "GravityOCR":
        """A ``.gocr`` package written by ``tools/build_model.py``.

        ``ngram_gate > 0`` switches the UnifiedNgram residual on (measured: no gain at
        the plan's 0.08, a loss at 0.16 – off by default; ``tools/ngram_eval.py``).
        ``rescue=False`` leaves the check of the rows the output slice cut out (a GPU pass
        of 16 rows saves ~1.5 ms; a token outside the slice is then replaced by the best
        kept one, e.g. ``PRESS`` → ``RESS``).
        """
        src = Package(path)
        vocab = Vocab(src.blob("tokenizer.json").decode("utf-8"))
        model = cls(src, vocab, src.meta["config"], src.meta.get("block_diffusion"), device, rescue=rescue)
        if ngram_gate > 0 and "ngram.tags" in src:
            from .ngram import NgramTable

            model.ngram = NgramTable.load(src, gate=ngram_gate)
        return model

    def set_head(self, weight: np.ndarray, ids: np.ndarray | None) -> None:
        """Output matrix and, when it is sliced, the full token id of each row."""
        w = np.ascontiguousarray(weight, dtype=F32)
        if self.kernels is not None:
            self.head_lin, self._head = native.Linear.dense(w), None
        else:
            self.head_lin, self._head = None, w
        self._set_head_ids(ids)

    def set_head_parts(self, parts: dict, scheme: str, shape, ids: np.ndarray | None) -> None:
        """Packed output matrix (native path; the numpy path decodes it)."""
        if self.kernels is None:
            from .quant import dequantize

            self.set_head(dequantize(parts, scheme, shape), ids)
            return
        self.head_lin, self._head = native.Linear(parts, scheme, shape), None
        self._set_head_ids(ids)

    def set_rest_parts(self, parts: dict, scheme: str, shape, ids: np.ndarray) -> None:
        """The output rows the slice cut (full ids ``ids``).  Every pass then also finds the best
        of them; where it beats every kept row, that token is taken – what the whole head would
        have picked.  A slice can cut a token the model wants (a word outside the domain corpus,
        e.g. ``PRESS``) and the best kept row is then a wrong piece (``RESS``)."""
        ids = np.asarray(ids, dtype=np.int64)
        if self.gpu is not None:
            if not self.lm.set_rest(parts, scheme, shape):
                warnings.warn("gravity_ocr: this GPU library cannot check the cut output rows (rebuild rust_gpu)", stacklevel=2)
                return
        elif self.kernels is not None:
            self.rest_lin = native.Linear(parts, scheme, shape)
        else:
            from .quant import dequantize

            self._rest = np.ascontiguousarray(dequantize(parts, scheme, shape), dtype=F32)
        self.rest_ids = ids

    def rest_at(self, i: int) -> tuple | None:
        """``(logit, token)`` of the best cut row for row ``i`` of the last ``run`` (``None`` without them)."""
        if self.last_rest is None:
            return None
        vals, toks = self.last_rest
        return float(vals[i]), int(toks[i])

    def _rest_of(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(h, F32).reshape(-1, h.shape[-1])
        r = self.kernels.linear(self.rest_lin, h) if self.rest_lin is not None else h @ self._rest.T
        arg = r.argmax(axis=1)
        return r[np.arange(r.shape[0]), arg], self.rest_ids[arg]

    @property
    def head(self) -> np.ndarray:
        """The output matrix as float32 (decoded on first use on the native path)."""
        if self._head is None:
            if self.head_lin is None:
                raise NotImplementedError("the output matrix lives on the GPU")
            self._head = self.kernels.dequant(self.head_lin)
        return self._head

    @property
    def memory(self) -> dict | None:
        """Device memory and what is on it (``None`` off the GPU)."""
        return None if self.residency is None else self.residency.snapshot()

    @property
    def head_rows(self) -> int:
        if self.gpu is not None:
            return self.lm.head_rows
        return self.head_lin.rows if self.head_lin is not None else self._head.shape[0]

    def slice_head(self, ids, rescue: bool = False) -> None:
        """Keep only the output rows of the full-vocabulary ``ids`` (ascending), on any device.

        ``rescue`` keeps the other rows too, to check every pass that none of them wins
        (``set_rest_parts``)."""
        ids = np.asarray(ids, dtype=np.int64)
        if self.head_ids is not None:
            raise ValueError("the output matrix is already sliced")
        if self.ggml:
            raise NotImplementedError("slicing the output matrix of a GGUF model: slice before tools/build_gguf.py")
        parts, scheme, shape = native.parts_of(self.src, language.HEAD)

        def rows(which: np.ndarray) -> tuple[dict, str, tuple]:
            if scheme in ("bf16", "f16", "f32", "raw"):
                return {"w": np.ascontiguousarray(np.asarray(parts["w"]).reshape(shape[0], -1)[which])}, scheme, (which.size, shape[1])
            from .quant import dequantize

            return {"w": np.ascontiguousarray(dequantize(parts, scheme, shape)[which])}, "f32", (which.size, shape[1])

        new, sch, shp = rows(ids)
        if self.gpu is not None:
            self.lm.set_parts(language.HEAD, new, sch, shp)
            self._set_head_ids(ids)
        else:
            self.set_head_parts(new, sch, shp, ids)
        if rescue:
            rest = np.setdiff1d(np.arange(shape[0], dtype=np.int64), ids)
            self.set_rest_parts(*rows(rest), rest)

    def _set_head_ids(self, ids: np.ndarray | None) -> None:
        self.head_ids = None if ids is None else np.asarray(ids, dtype=np.int64)
        if self.head_ids is None:
            self.eos_rows = np.array(sorted(self.eos), dtype=np.int64)
        else:
            self.eos_rows = np.flatnonzero(np.isin(self.head_ids, list(self.eos)))

    # ------------------------------------------------------------------ pieces
    def embed(self, ids) -> np.ndarray:
        return self.src.rows(language.EMBED, ids)

    def token_of(self, row: int) -> int:
        """Full-vocabulary id of an output row."""
        return int(row) if self.head_ids is None else int(self.head_ids[row])

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        if self.gpu is not None:
            raise NotImplementedError("on the GPU the logits come out of the language pass")
        if self.head_lin is not None:
            h = np.asarray(hidden, F32)
            out = self.kernels.linear(self.head_lin, h.reshape(-1, h.shape[-1]))
            return out.reshape(*h.shape[:-1], out.shape[-1])
        return hidden @ self._head.T

    def encode_image(self, image, fit: str = "pad", taps: dict | None = None) -> tuple[np.ndarray, tuple[int, int]]:
        """Image → image tokens ``[rows·cols, 1536]`` and the merged grid ``(rows, cols)``."""
        rgb = preprocess.fit_image(image, fit)
        patches, (gh, gw) = preprocess.to_patches(rgb)
        tokens = self.vision.forward(patches, (gh, gw), taps)
        return tokens, (gh // preprocess.MERGE, gw // preprocess.MERGE)

    def prompt_ids(self, prompt: str, image_tokens: int) -> list[int]:
        head = self.vocab.encode(PROMPT_HEAD)
        tail = self.vocab.encode(PROMPT_TAIL.format(prompt=prompt))
        return head + [self.vocab.image_id] * image_tokens + tail

    @staticmethod
    def prompt_positions(head: int, grid: tuple[int, int], tail: int) -> tuple[np.ndarray, int]:
        """3-axis positions of the prompt rows and the position after them.

        Text advances all three axes together; an image token at merged cell
        ``(r, c)`` sits at ``(base, base + r, base + c)`` and the image as a whole
        advances the text position by ``max(rows, cols)``.
        """
        rows, cols = grid
        pos = np.empty((head + rows * cols + tail, 3), np.int64)
        pos[:head] = np.arange(head)[:, None]
        rr, cc = np.divmod(np.arange(rows * cols), cols)
        pos[head : head + rows * cols] = np.stack([np.zeros_like(rr), rr, cc], axis=-1) + head
        after = head + max(rows, cols)
        pos[head + rows * cols :] = (after + np.arange(tail))[:, None]
        return pos, after + tail

    # ------------------------------------------------------------------ prefill
    def prefill(
        self,
        image_tokens: np.ndarray,
        grid: tuple[int, int],
        prompt: str = TEXT_PROMPT,
        hidden: list | None = None,
        capacity: int = 0,
        keep: np.ndarray | None = None,
    ) -> Session:
        """Run the prompt.  ``keep`` (boolean per image token) leaves the others out of the
        prefill; the kept tokens keep their grid positions."""
        t0 = time.perf_counter()
        if keep is not None:
            image_tokens = image_tokens[keep]
        ids = self.prompt_ids(prompt, image_tokens.shape[0])
        head = ids.index(self.vocab.image_id)
        tail = len(ids) - head - image_tokens.shape[0]
        x = self.embed(ids)
        x[head : head + image_tokens.shape[0]] = image_tokens
        full = grid[0] * grid[1]
        pos, next_pos = self.prompt_positions(head, grid, tail)  # positions of the whole grid …
        if keep is not None:  # … of which the kept image rows keep theirs
            pos = np.concatenate([pos[:head], pos[head : head + full][keep], pos[head + full :]])
        cache = self.lm.new_cache(max(capacity, len(ids) + 128))
        logits = self.run(x, pos, cache, hidden=hidden, last_only=True)
        return Session(cache, next_pos, logits, len(ids), image_tokens.shape[0], {"prefill": time.perf_counter() - t0}, rest=self.rest_at(0))

    def step(self, session: Session, tokens, mode: str = "causal", commit: bool = True) -> np.ndarray:
        """Feed ``tokens`` at the next positions; returns logits ``[len(tokens), V]``."""
        n = len(tokens)
        pos = np.repeat(session.next_pos + np.arange(n), 3).reshape(n, 3)
        inject = self.ngram.residual(session.history, list(tokens)) if self.ngram is not None else None
        # a block draft is verified by the next causal pass, so it needs no check of the cut rows
        out = self.run(self.embed(tokens), pos, session.cache, mode=mode, commit=commit, inject=inject, rest=mode != "block")
        if commit:
            session.next_pos += n
            session.history += list(tokens)
        return out

    def run(self, x, pos, cache, mode="causal", commit=True, hidden=None, last_only=False, start: int = 0, inject=None,
            rest: bool = True) -> np.ndarray:  # fmt: skip
        """Language pass over rows ``x`` → logits of rows ``start..`` (or of the last row only).

        ``inject`` (``[rows, 1536]`` or ``None``) is added to the hidden state after layer 0.
        ``rest``: also find the best cut output row of every row (``rest_at``), when the head has them.
        """
        if self.gpu is not None:
            if hidden is not None:
                raise NotImplementedError("hidden states need a CPU device")
            logits = self.lm.forward_logits(x, pos, cache, mode, commit, last_only, start, inject, rest=rest)
            lr = self.lm.last_rest  # (best logit, row of the cut rows) per row
            self.last_rest = None if lr is None or self.rest_ids is None else (lr[0], self.rest_ids[lr[1]])
            return logits
        if inject is not None:
            outer = self.lm.layer_hook
            self.lm.layer_hook = lambda i, h: (h + inject if i == 1 else h) if outer is None else outer(i, h + inject if i == 1 else h)
        try:
            out = self.lm.forward(x, pos, cache, mode=mode, commit=commit, hidden=hidden)
        finally:
            if inject is not None:
                self.lm.layer_hook = outer
        rows = out[-1] if last_only else out[start:]
        self.last_rest = self._rest_of(rows) if rest and self.rest_ids is not None else None
        return self.logits(rows)
