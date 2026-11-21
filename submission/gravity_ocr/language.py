"""GLM decoder (16 layers, grouped-query attention, 3-axis RoPE).

One forward handles any number of new rows on top of a key/value cache.  Two
attention patterns are offered:

``causal``  row *i* sees the cache and new rows ``0..i`` – prefill, plain
            autoregressive steps and the verification pass.
``block``   every new row sees the cache and *all* new rows – the
            block-diffusion draft, where a block of mask tokens is filled in
            one pass.

The numpy path is the reference.  With ``kernels`` (``gravity_ocr.native``)
the matrices stay packed and every layer runs as: fused RMSNorm → three packed
projections → fused RoPE + cache write + grouped-query attention → packed
output projection with the post-norm residual fused → fused RMSNorm → packed
gate/up → fused SwiGLU → packed down projection with the residual fused.
"""

from __future__ import annotations

import numpy as np

from .ops import F32, rms_norm, silu, softmax_inplace

PREFIX = "model.language_model."
DEPTH = 16
WIDTH = 1536
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
GROUP = HEADS // KV_HEADS
INTER = 4608
EPS = 1e-5
ROPE_SECTIONS = (16, 24, 24)  # frequencies driven by the t, row and column position
CACHE_STEP = 64  # the native cache grows in whole key panels

LAYER_TENSORS = (
    "input_layernorm.weight", "post_self_attn_layernorm.weight",
    "post_attention_layernorm.weight", "post_mlp_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
)  # fmt: skip
MATRICES = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
}  # fmt: skip
EMBED = PREFIX + "embed_tokens.weight"
FINAL_NORM = PREFIX + "norm.weight"
HEAD = "lm_head.weight"


def tensor_names() -> list[str]:
    return [f"{PREFIX}layers.{i}.{n}" for i in range(DEPTH) for n in LAYER_TENSORS] + [FINAL_NORM]


def is_matrix(short: str) -> bool:
    return short.split(".", 2)[-1] in MATRICES


_INV_FREQ = 1.0 / (10000.0 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
_AXIS = np.repeat(np.arange(3), ROPE_SECTIONS)


def rope_tables(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``pos`` ``[T, 3]`` → cos / sin ``[T, 64]`` for adjacent-pair rotation."""
    ang = pos[:, _AXIS].astype(np.float64) * _INV_FREQ
    return np.ascontiguousarray(np.cos(ang), F32), np.ascontiguousarray(np.sin(ang), F32)


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate pairs ``(x[2i], x[2i+1])``; ``x`` is ``[T, heads, 128]``."""
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos[:, None, :] - b * sin[:, None, :]
    out[..., 1::2] = b * cos[:, None, :] + a * sin[:, None, :]
    return out


class KVCache:
    """Keys and values of every layer.

    ``transposed`` (the native layout) keeps the keys as ``[layer, kv head,
    dim, key]`` so that the attention kernel reads eight keys per vector, and
    sizes the cache in whole panels of 64 keys.
    """

    def __init__(self, capacity: int = 512, transposed: bool = False):
        self.transposed = transposed
        cap = self._round(capacity)
        self.k = self._empty_k(cap)
        self.v = np.empty((DEPTH, KV_HEADS, cap, HEAD_DIM), F32)
        self.len = 0

    def _round(self, cap: int) -> int:
        return -(-cap // CACHE_STEP) * CACHE_STEP if self.transposed else cap

    def _empty_k(self, cap: int) -> np.ndarray:
        if self.transposed:
            return np.empty((DEPTH, KV_HEADS, HEAD_DIM, cap), F32)
        return np.empty((DEPTH, KV_HEADS, cap, HEAD_DIM), F32)

    @property
    def capacity(self) -> int:
        return self.v.shape[2]

    def reserve(self, extra: int) -> None:
        need = self.len + extra
        cap = self.capacity
        if need <= cap:
            return
        cap = self._round(max(need, cap * 2))
        k = self._empty_k(cap)
        if self.transposed:
            k[..., : self.len] = self.k[..., : self.len]
        else:
            k[:, :, : self.len] = self.k[:, :, : self.len]
        v = np.empty((DEPTH, KV_HEADS, cap, HEAD_DIM), F32)
        v[:, :, : self.len] = self.v[:, :, : self.len]
        self.k, self.v = k, v

    def keys(self, layer: int, n: int) -> np.ndarray:
        """``[kv head, n, dim]`` of one layer, whatever the layout."""
        k = self.k[layer]
        return k[:, :, :n].transpose(0, 2, 1) if self.transposed else k[:, :n]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


class LanguageModel:
    """Weights by their short names; ``kernels`` switches to the packed native path."""

    def __init__(self, src, kernels=None):
        self.k = kernels
        self.w: dict[str, np.ndarray] = {}
        self.lin: dict = {}
        for name in tensor_names():
            short = name[len(PREFIX) :]
            if self.k is not None and is_matrix(short):
                from .native import Linear, parts_of

                self.lin[short] = Linear(*parts_of(src, name))
            else:
                self.set(name, src.get(name))
        # hook: called with (layer index, hidden rows) before a layer runs
        self.layer_hook = None

    @property
    def native(self) -> bool:
        return self.k is not None

    def set(self, name: str, value: np.ndarray) -> None:
        short = name[len(PREFIX) :]
        value = np.ascontiguousarray(value, dtype=F32)
        if self.k is not None and is_matrix(short):
            from .native import Linear

            self.lin[short] = Linear.dense(value)
            self.w.pop(short, None)
        else:
            self.w[short] = value

    def set_parts(self, name: str, parts: dict, scheme: str, shape) -> None:
        short = name[len(PREFIX) :]
        if self.k is not None and is_matrix(short):
            from .native import Linear

            self.lin[short] = Linear(parts, scheme, shape)
        else:
            from .quant import dequantize

            self.set(name, dequantize(parts, scheme, shape))

    def new_cache(self, capacity: int) -> KVCache:
        return KVCache(capacity, transposed=self.k is not None)

    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.w.values()) + sum(v.nbytes for v in self.lin.values()))

    def forward(
        self,
        x: np.ndarray,
        pos: np.ndarray,
        cache: KVCache,
        mode: str = "causal",
        commit: bool = True,
        hidden: list | None = None,
    ) -> np.ndarray:
        """Run ``x`` ``[T, 1536]`` at positions ``pos`` ``[T, 3]``; returns the final-norm rows.

        With ``commit=False`` the cache length is left unchanged, so the rows
        written by this call are overwritten by the next one (draft passes).
        """
        if self.k is not None:
            return self._forward_native(x, pos, cache, mode, commit, hidden)
        w = self.w
        t = x.shape[0]
        past = cache.len
        cache.reserve(t)
        cos, sin = rope_tables(pos)
        mask = None
        if mode == "causal" and t > 1:
            mask = np.triu(np.full((t, t), -np.inf, F32), k=1)
            mask = np.tile(mask, (GROUP, 1))  # query rows are laid out [group member, token]
        scale = F32(HEAD_DIM**-0.5)
        for i in range(DEPTH):
            p = f"layers.{i}."
            if self.layer_hook is not None:
                x = self.layer_hook(i, x)
            if hidden is not None:
                hidden.append(x)
            h = rms_norm(x, w[p + "input_layernorm.weight"])
            q = _rotate((h @ w[p + "self_attn.q_proj.weight"].T).reshape(t, HEADS, HEAD_DIM), cos, sin)
            k = _rotate((h @ w[p + "self_attn.k_proj.weight"].T).reshape(t, KV_HEADS, HEAD_DIM), cos, sin)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(t, KV_HEADS, HEAD_DIM)
            if cache.transposed:
                cache.k[i, :, :, past : past + t] = k.transpose(1, 2, 0)
            else:
                cache.k[i, :, past : past + t] = k.transpose(1, 0, 2)
            cache.v[i, :, past : past + t] = v.transpose(1, 0, 2)
            keys = cache.keys(i, past + t)
            vals = cache.v[i, :, : past + t]
            # query heads 2g and 2g+1 share key/value head g
            qg = q.reshape(t, KV_HEADS, GROUP, HEAD_DIM).transpose(1, 2, 0, 3).reshape(KV_HEADS, GROUP * t, HEAD_DIM)
            att = (qg @ keys.transpose(0, 2, 1)) * scale
            if mask is not None:
                att[:, :, past:] += mask
            softmax_inplace(att)
            o = (att @ vals).reshape(KV_HEADS, GROUP, t, HEAD_DIM).transpose(2, 0, 1, 3).reshape(t, HEADS * HEAD_DIM)
            x = x + rms_norm(o @ w[p + "self_attn.o_proj.weight"].T, w[p + "post_self_attn_layernorm.weight"])
            h = rms_norm(x, w[p + "post_attention_layernorm.weight"])
            u = h @ w[p + "mlp.gate_up_proj.weight"].T
            m = (u[:, INTER:] * silu(u[:, :INTER])) @ w[p + "mlp.down_proj.weight"].T
            x = x + rms_norm(m, w[p + "post_mlp_layernorm.weight"])
        if hidden is not None:
            hidden.append(x)
        if commit:
            cache.len = past + t
        return rms_norm(x, w["norm.weight"])

    def _forward_native(self, x, pos, cache: KVCache, mode: str, commit: bool, hidden: list | None) -> np.ndarray:
        k, w, lin = self.k, self.w, self.lin
        if not cache.transposed:
            raise ValueError("the native path needs a cache from new_cache()")
        x = np.array(x, dtype=F32, order="C", copy=True)  # updated in place below
        t = x.shape[0]
        past = cache.len
        cache.reserve(t)
        cos, sin = rope_tables(pos)
        causal = mode == "causal"
        for i in range(DEPTH):
            p = f"layers.{i}."
            if self.layer_hook is not None:
                x = np.array(self.layer_hook(i, x), dtype=F32, order="C", copy=True)
            if hidden is not None:
                hidden.append(x.copy())
            h = k.rms_norm(x, w[p + "input_layernorm.weight"], EPS)
            q = k.linear(lin[p + "self_attn.q_proj.weight"], h)
            kk = k.linear(lin[p + "self_attn.k_proj.weight"], h)
            v = k.linear(lin[p + "self_attn.v_proj.weight"], h)
            o = k.lm_attention(q, kk, v, cos, sin, cache.k[i], cache.v[i], past, causal)
            a = k.linear(lin[p + "self_attn.o_proj.weight"], o)
            k.rms_norm_add(x, a, w[p + "post_self_attn_layernorm.weight"], EPS)
            h = k.rms_norm(x, w[p + "post_attention_layernorm.weight"], EPS)
            u = k.linear(lin[p + "mlp.gate_up_proj.weight"], h)
            m = k.swiglu_split(u, INTER)
            d = k.linear(lin[p + "mlp.down_proj.weight"], m)
            k.rms_norm_add(x, d, w[p + "post_mlp_layernorm.weight"], EPS)
        if hidden is not None:
            hidden.append(x.copy())
        if commit:
            cache.len = past + t
        return k.rms_norm(x, w["norm.weight"], EPS)
