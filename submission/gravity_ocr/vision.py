"""CogViT vision tower (24 blocks, 2-D RoPE) and the 2×2 merger.

Two ways to run it, same arithmetic:

* numpy (the reference): every weight is held as float32;
* native kernels (``gravity_ocr.native``): matrices stay packed as they are
  stored (memory-mapped package parts or raw BF16) and are multiplied by the
  fused decode-and-multiply kernel; norm, rotary embedding and attention run
  fused.  Only the patch embedding (its two time frames are added) and the
  downsample convolution (re-ordered) are re-shaped into float32 at load.
"""

from __future__ import annotations

import numpy as np

from .ops import F32, gelu, layer_norm, rms_norm, silu, softmax_inplace
from .preprocess import patch_positions

PREFIX = "model.visual."
DEPTH = 24
WIDTH = 1024
HEADS = 16
HEAD_DIM = WIDTH // HEADS
OUT = 1536
EPS = 1e-5

BLOCK_TENSORS = (
    "norm1.weight", "norm2.weight",
    "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
    "attn.q_norm.weight", "attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.gate_proj.bias", "mlp.up_proj.weight", "mlp.up_proj.bias",
    "mlp.down_proj.weight", "mlp.down_proj.bias",
)  # fmt: skip
HEAD_TENSORS = ("patch_embed.proj.weight", "patch_embed.proj.bias")
TAIL_TENSORS = (
    "post_layernorm.weight", "downsample.weight", "downsample.bias",
    "merger.proj.weight", "merger.post_projection_norm.weight", "merger.post_projection_norm.bias",
    "merger.gate_proj.weight", "merger.up_proj.weight", "merger.down_proj.weight",
)  # fmt: skip
MATRICES = {
    "attn.qkv.weight", "attn.proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "patch_embed.proj.weight", "downsample.weight",
    "merger.proj.weight", "merger.gate_proj.weight", "merger.up_proj.weight", "merger.down_proj.weight",
}  # fmt: skip
RESHAPED = {"patch_embed.proj.weight", "downsample.weight"}


def tensor_names() -> list[str]:
    names = [PREFIX + n for n in HEAD_TENSORS]
    names += [f"{PREFIX}blocks.{i}.{n}" for i in range(DEPTH) for n in BLOCK_TENSORS]
    names += [PREFIX + n for n in TAIL_TENSORS]
    return names


def is_matrix(short: str) -> bool:
    return short.split("blocks.", 1)[-1].split(".", 1)[-1] in MATRICES or short in MATRICES


def _rope_tables(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos / sin ``[N, 64]``: first 16 frequencies follow the row, next 16 the column."""
    half = HEAD_DIM // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, half, 2, dtype=np.float64) / half))
    ang = (pos[:, :, None].astype(np.float64) * inv).reshape(pos.shape[0], -1)
    ang = np.concatenate([ang, ang], axis=-1)
    return np.ascontiguousarray(np.cos(ang), F32), np.ascontiguousarray(np.sin(ang), F32)


def _rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def reshape_weight(short: str, value: np.ndarray) -> np.ndarray:
    """The two weights the runtime stores in another layout than the checkpoint."""
    if short == "patch_embed.proj.weight":
        # [1024, 3, 2, 14, 14]: a still image is fed as two identical frames,
        # so the two temporal slices can be added once and for all.
        return value.reshape(WIDTH, 3, 2, -1).sum(axis=2).reshape(WIDTH, -1)
    if short == "downsample.weight":
        # conv 2×2 stride 2 over a merge group stored as (i, j, channel)
        return value.transpose(0, 2, 3, 1).reshape(OUT, 4 * WIDTH)
    return value


class VisionTower:
    """Weights by their short names; ``kernels`` switches to the packed native path."""

    def __init__(self, src, kernels=None):
        self.k = kernels
        self.w: dict[str, np.ndarray] = {}  # float32 tensors
        self.lin: dict = {}  # packed matrices (native path)
        for name in tensor_names():
            short = name[len(PREFIX) :]
            if self.k is not None and is_matrix(short) and short not in RESHAPED:
                from .native import Linear, parts_of

                self.lin[short] = Linear(*parts_of(src, name))
            else:
                self.set(name, src.get(name))

    @property
    def native(self) -> bool:
        return self.k is not None

    def set(self, name: str, value: np.ndarray) -> None:
        """Swap in a float32 tensor (original name)."""
        short = name[len(PREFIX) :]
        value = np.ascontiguousarray(reshape_weight(short, np.asarray(value, dtype=F32)))
        if self.k is not None and is_matrix(short):
            from .native import Linear

            self.lin[short] = Linear.dense(value)
            self.w.pop(short, None)
        else:
            self.w[short] = value

    def set_parts(self, name: str, parts: dict, scheme: str, shape) -> None:
        """Swap in a packed tensor (native path; the numpy path decodes it)."""
        short = name[len(PREFIX) :]
        if self.k is not None and is_matrix(short) and short not in RESHAPED:
            from .native import Linear

            self.lin[short] = Linear(parts, scheme, shape)
        else:
            from .quant import dequantize

            self.set(name, dequantize(parts, scheme, shape))

    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.w.values()) + sum(v.nbytes for v in self.lin.values()))

    # ------------------------------------------------------------------ numpy
    def patch_embed(self, patches: np.ndarray) -> np.ndarray:
        if self.k is not None:
            return self.k.linear(self.lin["patch_embed.proj.weight"], patches, self.w["patch_embed.proj.bias"])
        return patches @ self.w["patch_embed.proj.weight"].T + self.w["patch_embed.proj.bias"]

    def block(self, i: int, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        if self.k is not None:
            return self._block_native(i, x, cos, sin)
        w = self.w
        p = f"blocks.{i}."
        n = x.shape[0]
        h = rms_norm(x, w[p + "norm1.weight"])
        qkv = (h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]).reshape(n, 3, HEADS, HEAD_DIM)
        q = _rotate(rms_norm(qkv[:, 0], w[p + "attn.q_norm.weight"]), cos, sin).transpose(1, 0, 2)
        k = _rotate(rms_norm(qkv[:, 1], w[p + "attn.k_norm.weight"]), cos, sin).transpose(1, 2, 0)
        v = qkv[:, 2].transpose(1, 0, 2)
        out = np.empty((HEADS, n, HEAD_DIM), F32)
        scale = F32(HEAD_DIM**-0.5)
        for hd in range(HEADS):  # one head at a time bounds the N×N scratch memory
            att = softmax_inplace((q[hd] @ k[hd]) * scale)
            out[hd] = att @ v[hd]
        attn = out.transpose(1, 0, 2).reshape(n, WIDTH)
        x = x + attn @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"]
        h = rms_norm(x, w[p + "norm2.weight"])
        gate = silu(h @ w[p + "mlp.gate_proj.weight"].T + w[p + "mlp.gate_proj.bias"])
        up = h @ w[p + "mlp.up_proj.weight"].T + w[p + "mlp.up_proj.bias"]
        return x + (gate * up) @ w[p + "mlp.down_proj.weight"].T + w[p + "mlp.down_proj.bias"]

    def merge(self, x: np.ndarray) -> np.ndarray:
        """Post norm, 2×2 downsample and the merger MLP → ``[N/4, 1536]``."""
        w = self.w
        if self.k is not None:
            k, lin = self.k, self.lin
            x = k.rms_norm(x, w["post_layernorm.weight"], EPS)
            x = k.linear(lin["downsample.weight"], x.reshape(-1, 4 * WIDTH), w["downsample.bias"])
            h = k.linear(lin["merger.proj.weight"], x)
            h = gelu(k.layer_norm(h, w["merger.post_projection_norm.weight"], w["merger.post_projection_norm.bias"], EPS))
            g = k.linear(lin["merger.gate_proj.weight"], h)
            k.silu_mul(g, k.linear(lin["merger.up_proj.weight"], h), out=g)
            return k.linear(lin["merger.down_proj.weight"], g)
        x = rms_norm(x, w["post_layernorm.weight"])
        x = x.reshape(-1, 4 * WIDTH) @ w["downsample.weight"].T + w["downsample.bias"]
        h = x @ w["merger.proj.weight"].T
        h = gelu(layer_norm(h, w["merger.post_projection_norm.weight"], w["merger.post_projection_norm.bias"]))
        return (silu(h @ w["merger.gate_proj.weight"].T) * (h @ w["merger.up_proj.weight"].T)) @ w[
            "merger.down_proj.weight"
        ].T

    # ------------------------------------------------------------------ native
    def _block_native(self, i: int, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        """One block on the packed weights; ``x`` is updated in place and returned."""
        k, w, lin = self.k, self.w, self.lin
        p = f"blocks.{i}."
        h = k.rms_norm(x, w[p + "norm1.weight"], EPS)
        qkv = k.linear(lin[p + "attn.qkv.weight"], h, w[p + "attn.qkv.bias"])
        attn = k.vis_attention(qkv, w[p + "attn.q_norm.weight"], w[p + "attn.k_norm.weight"], cos, sin, EPS)
        k.linear(lin[p + "attn.proj.weight"], attn, w[p + "attn.proj.bias"], out=x, accumulate=True)
        h = k.rms_norm(x, w[p + "norm2.weight"], EPS)
        gate = k.linear(lin[p + "mlp.gate_proj.weight"], h, w[p + "mlp.gate_proj.bias"])
        up = k.linear(lin[p + "mlp.up_proj.weight"], h, w[p + "mlp.up_proj.bias"])
        k.silu_mul(gate, up, out=gate)
        k.linear(lin[p + "mlp.down_proj.weight"], gate, w[p + "mlp.down_proj.bias"], out=x, accumulate=True)
        return x

    def forward(self, patches: np.ndarray, grid: tuple[int, int], taps: dict | None = None, rows=None) -> np.ndarray:
        """``patches`` ``[N, 588]`` in merge-group order → image tokens ``[N/4, 1536]``.

        ``rows`` (whole merge groups) says which patch rows of the grid ``patches``
        are, when only some are passed (blank pruning); they keep their positions.
        """
        pos = patch_positions(*grid)
        cos, sin = _rope_tables(pos if rows is None else pos[rows])
        x = self.patch_embed(patches)
        if taps is not None:
            taps["patch_embed"] = x.copy()
        for i in range(DEPTH):
            x = self.block(i, x, cos, sin)
        if taps is not None:
            taps["last_hidden"] = x.copy()
        return self.merge(x)
