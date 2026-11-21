"""The measured "pyramid" allocation: which precision each group of matrices takes.

53 groups (vision bands × qkv / proj / mlp, downsample, merger, language bands × q / k / v / o /
gate_up / down, output matrix) were screened at q8 … q3 by teacher-forced ΔNLL on the GPU and
allocated under a ΔNLL budget of 0.002 (``docs/QUANT.md``, ``tools/quant_sweep.py`` /
``quant_alloc.py``).  The same allocation drives every compressed form of the model:

* the ``.gocr`` package (its own q3 … q8 / nf4 codecs, ``tools/build_model.py``);
* the GGUF file (GGML k-quants: ``GGML`` map, ``tools/build_gguf.py``);
* the "pyramid" weight store of a GPU engine (``gravity_ocr.residency``): the 8-bit class
  (q8 / q6 / q5) in the GPU's 8-bit float format (FP8 E4M3, MXFP8 on Blackwell), the 4-bit
  class (q4 / nf4 / q3) in its 4-bit format (NVFP4) – one quantisation from the BF16 checkpoint.
"""

from __future__ import annotations

from functools import lru_cache

from . import language, vision

VIS_BANDS = [(0, 0), (1, 7), (8, 15), (16, 22), (23, 23)]
VIS_KINDS = {"qkv": ["attn.qkv.weight"], "proj": ["attn.proj.weight"], "mlp_in": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
             "mlp_out": ["mlp.down_proj.weight"]}  # fmt: skip
LM_BANDS = [(0, 1), (2, 5), (6, 9), (10, 13), (14, 15)]
LM_KINDS = {"q": ["self_attn.q_proj.weight"], "k": ["self_attn.k_proj.weight"], "v": ["self_attn.v_proj.weight"],
            "o": ["self_attn.o_proj.weight"], "gate_up": ["mlp.gate_up_proj.weight"], "down": ["mlp.down_proj.weight"]}  # fmt: skip

# docs/QUANT.md "chosen" column (budget 0.002)
MEASURED = {
    "vis0-0.qkv": "q3", "vis0-0.proj": "q8", "vis0-0.mlp_in": "q3", "vis0-0.mlp_out": "q4",
    "vis1-7.qkv": "q4", "vis1-7.proj": "q4", "vis1-7.mlp_in": "q3", "vis1-7.mlp_out": "q4",
    "vis8-15.qkv": "q5", "vis8-15.proj": "q5", "vis8-15.mlp_in": "q5", "vis8-15.mlp_out": "q6",
    "vis16-22.qkv": "q8", "vis16-22.proj": "q8", "vis16-22.mlp_in": "q6", "vis16-22.mlp_out": "q6",
    "vis23-23.qkv": "q5", "vis23-23.proj": "q8", "vis23-23.mlp_in": "q4", "vis23-23.mlp_out": "q6",
    "vis.downsample": "q8", "vis.merger": "q5",
    "lm0-1.q": "q3", "lm0-1.k": "q4", "lm0-1.v": "q4", "lm0-1.o": "q8", "lm0-1.gate_up": "q4", "lm0-1.down": "q4",
    "lm2-5.q": "nf4", "lm2-5.k": "nf4", "lm2-5.v": "q4", "lm2-5.o": "q3", "lm2-5.gate_up": "q4", "lm2-5.down": "q4",
    "lm6-9.q": "nf4", "lm6-9.k": "nf4", "lm6-9.v": "q3", "lm6-9.o": "q5", "lm6-9.gate_up": "q4", "lm6-9.down": "q4",
    "lm10-13.q": "nf4", "lm10-13.k": "nf4", "lm10-13.v": "q5", "lm10-13.o": "q6", "lm10-13.gate_up": "q5", "lm10-13.down": "q4",
    "lm14-15.q": "q5", "lm14-15.k": "q3", "lm14-15.v": "q5", "lm14-15.o": "q5", "lm14-15.gate_up": "q4", "lm14-15.down": "q4",
    "lm.head": "q5",
}  # fmt: skip

# gravity scheme → GGML block format (blueprint-sandbox ai::pyramid: no Q3_K)
GGML = {"q8": "q8_0", "q6": "q6_k", "q5": "q5_k", "q4": "q4_k", "nf4": "q4_k", "q3": "q4_k"}
# gravity scheme → width class of the GPU float formats
CLASS = {"q8": 8, "q6": 8, "q5": 8, "q4": 4, "nf4": 4, "q3": 4}


def groups() -> dict[str, list[str]]:
    """Group name → the tensors it holds (the groups of ``tools/quant_sweep.py``)."""
    V, L = vision.PREFIX, language.PREFIX
    out: dict[str, list[str]] = {}
    for a, b in VIS_BANDS:
        for kind, names in VIS_KINDS.items():
            out[f"vis{a}-{b}.{kind}"] = [f"{V}blocks.{i}.{n}" for i in range(a, b + 1) for n in names]
    out["vis.downsample"] = [f"{V}downsample.weight"]
    out["vis.merger"] = [f"{V}merger.{n}.weight" for n in ("proj", "gate_proj", "up_proj", "down_proj")]
    for a, b in LM_BANDS:
        for kind, names in LM_KINDS.items():
            out[f"lm{a}-{b}.{kind}"] = [f"{L}layers.{i}.{n}" for i in range(a, b + 1) for n in names]
    out["lm.head"] = [language.HEAD]
    return out


def expand(assign: dict[str, str]) -> dict[str, str]:
    """``{group or tensor: scheme}`` → ``{tensor: scheme}``."""
    gs = groups()
    out = {}
    for key, scheme in assign.items():
        for t in gs.get(key, [key]):
            out[t] = scheme
    return out


@lru_cache(maxsize=1)
def measured() -> dict[str, str]:
    """``{tensor: scheme}`` of the measured allocation."""
    return expand(MEASURED)


def width_class(name: str) -> int:
    """8 or 4: the float-format width the measured allocation gives tensor ``name`` (8 for a
    matrix it does not name, e.g. the patch embedding)."""
    return CLASS.get(measured().get(name, "q8"), 8)
