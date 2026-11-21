"""Zero-copy ctypes wrapper for Track 2 native BF16 CUDA (sm_89) kernels.

Loads `gravity_ocr/_native/libtrack2_cuda_sm89.so` and dispatches directly on
`torch.cuda.current_stream().cuda_stream` so every call is 100% CUDA Graph capture safe.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

_LIB: Optional[ctypes.CDLL] = None
_TRIED = False

# Precomputed normalization constants matching Glm46VImageProcessorFast / gravity_ocr.preprocess:
#   out = (u8 / 255.0 - mean) / std = u8 * (1 / (255 * std)) + (-mean / std)
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_SCALE = tuple(1.0 / (255.0 * s) for s in _STD)
_BIAS = tuple(-m / s for m, s in zip(_MEAN, _STD))


def _find_lib() -> Optional[Path]:
    env_p = os.environ.get("TRACK2_CUDA_SO")
    if env_p and os.path.isfile(env_p):
        return Path(env_p)
    here = Path(__file__).resolve().parent / "_native" / "libtrack2_cuda_sm89.so"
    if here.is_file():
        return here
    return None


def load() -> Optional[ctypes.CDLL]:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = _find_lib()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.track2_u8_norm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.track2_u8_norm_bf16.restype = ctypes.c_int

        lib.track2_raw_rgb_patchify_norm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.track2_raw_rgb_patchify_norm_bf16.restype = ctypes.c_int

        lib.track2_chw_u8_patchify_norm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        lib.track2_chw_u8_patchify_norm_bf16.restype = ctypes.c_int

        lib.track2_vis_qkv_sdpa_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.track2_vis_qkv_sdpa_bf16.restype = ctypes.c_int

        lib.track2_bf16_argmax_cs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.track2_bf16_argmax_cs.restype = ctypes.c_int

        _LIB = lib
    except Exception:
        _LIB = None
    if _LIB is not None:
        try:
            _register_fused(_LIB)
        except Exception:  # load() must never raise: callers gate the whole shm / GPU image path on it
            pass
    return _LIB


def available() -> bool:
    return load() is not None


# ---------------------------------------------------------------------------------------------- v6 fused kernels
# Registered separately: a .so built from an older .cu (no fused symbols) still serves the kernels above.
_FUSED = False
_NORM = False
FUSED_ABI = 1
# arithmetic variants of the fused SiLU / FP8-scale math (picked by tools/test_fused_kernels.py: the pair whose outputs
# are bitwise equal to sgl_kernel.silu_and_mul + sgl_per_token_quant_fp8 on this build)
ACT_MODE = int(os.environ.get("TRACK2_FUSED_ACT_MODE", "0"))
QNT_MODE = int(os.environ.get("TRACK2_FUSED_QNT_MODE", "0"))


def _register_fused(lib: ctypes.CDLL) -> bool:
    """Optional kernels; never raises (a library built from an older .cu keeps serving the base kernels above)."""
    global _FUSED, _NORM
    vp, i32, i64, f32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
    try:
        lib.track2_fused_abi.argtypes = []
        lib.track2_fused_abi.restype = ctypes.c_int
        if int(lib.track2_fused_abi()) != FUSED_ABI:
            return False
        lib.track2_vis_qk_norm_rope_inplace.argtypes = [vp, vp, vp, vp, vp, i32, f32, i32, i32, vp]
        lib.track2_vis_qk_norm_rope_inplace.restype = i32
        lib.track2_silu_mul_quant_fp8.argtypes = [vp, i64, vp, vp, i32, i32, i32, i32, vp]
        lib.track2_silu_mul_quant_fp8.restype = i32
        lib.track2_quant_fp8_rows.argtypes = [vp, i64, vp, vp, i32, i32, i32, vp]
        lib.track2_quant_fp8_rows.restype = i32
        _FUSED = True
    except Exception:
        _FUSED = False
    try:  # added after ABI 1 builds shipped: optional, the kernels above keep working without it
        lib.track2_norm_quant_fp8.argtypes = [vp, i64, vp, i64, vp, f32, vp, f32, vp, i64, vp, vp, i32, i32, i32, i32, i32,
                                              vp]
        lib.track2_norm_quant_fp8.restype = i32
        _NORM = _FUSED
    except Exception:
        _NORM = False
    return _FUSED


def fused_available() -> bool:
    return load() is not None and _FUSED


def norm_available() -> bool:
    return load() is not None and _NORM


def vis_qk_norm_rope_inplace(
    qkv: torch.Tensor,
    qn_weight: torch.Tensor,
    kn_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K2 in place: Q / K head RMSNorm + NeoX 2D RoPE applied inside `qkv` ([..., 3 * H * 64], contiguous).

    Returns strided [S, H, 64] views (q, k, v) of `qkv` (last dim contiguous, row stride 3 * H * 64): FA2-varlen and
    SDPA take them as they are. Values are bitwise equal to vis_qkv_sdpa_bf16(..., out_shd=True)."""
    lib = load()
    assert lib is not None and _FUSED
    assert qkv.dtype == torch.bfloat16 and qkv.is_contiguous() and qkv.data_ptr() % 16 == 0
    s = int(qkv.numel() // (3 * num_heads * 64))
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    cos_is_bf16 = 1 if cos_c.dtype == torch.bfloat16 else 0
    if not cos_is_bf16 and cos_c.dtype != torch.float32:
        cos_c = cos_c.float()
        sin_c = sin_c.float()
    stream = torch.cuda.current_stream(qkv.device).cuda_stream
    err = lib.track2_vis_qk_norm_rope_inplace(
        qkv.data_ptr(), qn_weight.data_ptr(), kn_weight.data_ptr(), cos_c.data_ptr(), sin_c.data_ptr(),
        cos_is_bf16, float(eps), s, num_heads, stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_vis_qk_norm_rope_inplace CUDA error {err}")
    v3 = qkv.view(s, 3, num_heads, 64)
    return v3[:, 0], v3[:, 1], v3[:, 2]


def silu_mul_quant_fp8(gate_up: torch.Tensor, act_mode: Optional[int] = None,
                       qnt_mode: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16(silu(gate) * up) -> per-token FP8 (q [M, I] float8_e4m3fn, scale [M, 1] float32) in one kernel.

    Same outputs as sglang_per_token_quant_fp8(silu_and_mul(gate_up)) (bitwise for the build's ACT/QNT modes)."""
    lib = load()
    assert lib is not None and _FUSED
    two_i = int(gate_up.shape[-1])
    inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    if x2.stride(-1) != 1 or x2.data_ptr() % 16 or (x2.stride(0) * 2) % 16:
        x2 = x2.contiguous()
    m = int(x2.shape[0])
    q = torch.empty((m, inter), dtype=torch.float8_e4m3fn, device=gate_up.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=gate_up.device)
    if m == 0:
        return q, s
    stream = torch.cuda.current_stream(gate_up.device).cuda_stream
    err = lib.track2_silu_mul_quant_fp8(x2.data_ptr(), int(x2.stride(0)), q.data_ptr(), s.data_ptr(), m, inter,
                                        ACT_MODE if act_mode is None else int(act_mode),
                                        QNT_MODE if qnt_mode is None else int(qnt_mode), stream)
    if err != 0:
        raise RuntimeError(f"track2_silu_mul_quant_fp8 CUDA error {err}")
    return q, s


NORM_RED = int(os.environ.get("TRACK2_NORM_RED", "1"))  # RED=1 DIV=0: bitwise == flashinfer 0.6.7 norms
NORM_DIV = int(os.environ.get("TRACK2_NORM_DIV", "0"))
NORM_MODE_PLAIN, NORM_MODE_ADD, NORM_MODE_PRE_ADD, NORM_MODE_ROUND_ADD = 0, 1, 2, 3


def norm_quant_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    mode: int = NORM_MODE_PLAIN,
    weight0: Optional[torch.Tensor] = None,
    eps0: float = 0.0,
    want_y: bool = False,
    red: Optional[int] = None,
    div: Optional[int] = None,
):
    """(residual add +) RMSNorm (flashinfer arithmetic) -> per-token FP8 (sgl-kernel arithmetic), one kernel.

    mode 0: y = RMSNorm(x)                     mode 1: residual += x (in place), y = RMSNorm(residual) [fused_add_rmsnorm]
    mode 2: z = RMSNorm(x; weight0), residual += z, y = RMSNorm(residual)  (two norms + add)
    mode 3: residual = bf16(x + residual), y = RMSNorm(residual)            (torch add, then RMSNorm)
    Returns (q [M, H] float8_e4m3fn, scale [M, 1] float32) or, with want_y, (q, scale, y bf16)."""
    lib = load()
    assert lib is not None and _NORM
    hdim = int(x.shape[-1])
    x2 = x.reshape(-1, hdim)
    m = int(x2.shape[0])
    dev = x.device
    q = torch.empty((m, hdim), dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty((m, 1), dtype=torch.float32, device=dev)
    y = torch.empty((m, hdim), dtype=torch.bfloat16, device=dev) if want_y else None
    if m == 0:
        return (q, s, y) if want_y else (q, s)
    if mode != NORM_MODE_PLAIN:
        assert residual is not None
        r2 = residual.reshape(-1, hdim)
        assert r2.stride(-1) == 1 and r2.data_ptr() % 16 == 0 and (r2.stride(0) * 2) % 16 == 0
        r_ptr, r_stride = r2.data_ptr(), int(r2.stride(0))
    else:
        r_ptr, r_stride = None, 0
    assert x2.stride(-1) == 1 and x2.data_ptr() % 16 == 0 and (x2.stride(0) * 2) % 16 == 0
    stream = torch.cuda.current_stream(dev).cuda_stream
    err = lib.track2_norm_quant_fp8(
        x2.data_ptr(), int(x2.stride(0)), r_ptr, r_stride,
        weight0.data_ptr() if weight0 is not None else None, float(eps0),
        weight.data_ptr(), float(eps),
        y.data_ptr() if y is not None else None, hdim,
        q.data_ptr(), s.data_ptr(), m, hdim, int(mode),
        NORM_RED if red is None else int(red), NORM_DIV if div is None else int(div), stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_norm_quant_fp8 CUDA error {err}")
    return (q, s, y) if want_y else (q, s)


def quant_fp8_rows(x: torch.Tensor, qnt_mode: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quant of bf16 rows (the arithmetic of silu_mul_quant_fp8's second half)."""
    lib = load()
    assert lib is not None and _FUSED
    k = int(x.shape[-1])
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1 or x2.data_ptr() % 16 or (x2.stride(0) * 2) % 16:
        x2 = x2.contiguous()
    m = int(x2.shape[0])
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m == 0:
        return q, s
    stream = torch.cuda.current_stream(x.device).cuda_stream
    err = lib.track2_quant_fp8_rows(x2.data_ptr(), int(x2.stride(0)), q.data_ptr(), s.data_ptr(), m, k,
                                    QNT_MODE if qnt_mode is None else int(qnt_mode), stream)
    if err != 0:
        raise RuntimeError(f"track2_quant_fp8_rows CUDA error {err}")
    return q, s


def u8_norm_bf16(x_u8: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normalize `[N, 588]` uint8 single-frame patches directly into `[N, 588]` BF16 on GPU."""
    lib = load()
    assert lib is not None
    x_u8 = x_u8.contiguous()
    n = int(x_u8.shape[0])
    if out is None:
        out = torch.empty((n, 588), dtype=torch.bfloat16, device=x_u8.device)
    stream = torch.cuda.current_stream(x_u8.device).cuda_stream
    err = lib.track2_u8_norm_bf16(
        x_u8.data_ptr(),
        out.data_ptr(),
        n,
        _SCALE[0], _BIAS[0],
        _SCALE[1], _BIAS[1],
        _SCALE[2], _BIAS[2],
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_u8_norm_bf16 CUDA error {err}")
    return out


def raw_rgb_patchify_norm_bf16(
    rgb_u8: torch.Tensor,
    gh: int,
    gw: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused GPU 2x2 merge-group patchify + normalize from raw HWC uint8 RGB into `[gh*gw, 588]` BF16."""
    lib = load()
    assert lib is not None
    rgb_c = rgb_u8.contiguous()
    n = int(gh * gw)
    if out is None:
        out = torch.empty((n, 588), dtype=torch.bfloat16, device=rgb_c.device)
    stream = torch.cuda.current_stream(rgb_c.device).cuda_stream
    err = lib.track2_raw_rgb_patchify_norm_bf16(
        rgb_c.data_ptr(),
        out.data_ptr(),
        int(gh),
        int(gw),
        _SCALE[0], _BIAS[0],
        _SCALE[1], _BIAS[1],
        _SCALE[2], _BIAS[2],
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_raw_rgb_patchify_norm_bf16 CUDA error {err}")
    return out


def chw_u8_patchify_norm_bf16(
    rgb_chw_u8: torch.Tensor,
    gh: int,
    gw: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused GPU 2x2 merge-group patchify + normalize from CHW uint8 `[3, gh*14, gw*14]` -> `[gh*gw, 588]` BF16."""
    lib = load()
    assert lib is not None
    rgb_c = rgb_chw_u8.contiguous()
    n = int(gh * gw)
    if out is None:
        out = torch.empty((n, 588), dtype=torch.bfloat16, device=rgb_c.device)
    stream = torch.cuda.current_stream(rgb_c.device).cuda_stream
    err = lib.track2_chw_u8_patchify_norm_bf16(
        rgb_c.data_ptr(),
        out.data_ptr(),
        int(gh),
        int(gw),
        _SCALE[0], _BIAS[0],
        _SCALE[1], _BIAS[1],
        _SCALE[2], _BIAS[2],
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_chw_u8_patchify_norm_bf16 CUDA error {err}")
    return out


def vis_qkv_sdpa_bf16(
    qkv: torch.Tensor,
    qn_weight: torch.Tensor,
    kn_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int = 16,
    out_shd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV split + Q/K head RMSNorm + NeoX 2D RoPE -> `[1, H, S, 64]` (SDPA) or `[S, H, 64]` (Triton)."""
    lib = load()
    assert lib is not None
    qkv_c = qkv.contiguous()
    s = int(qkv_c.numel() // (3 * num_heads * 64))
    dev = qkv_c.device
    shape = (s, num_heads, 64) if out_shd else (1, num_heads, s, 64)
    q_out = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    k_out = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    v_out = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    cos_is_bf16 = 1 if cos_c.dtype == torch.bfloat16 else 0
    if not cos_is_bf16 and cos_c.dtype != torch.float32:
        cos_c = cos_c.float()
        sin_c = sin_c.float()
    stream = torch.cuda.current_stream(dev).cuda_stream
    err = lib.track2_vis_qkv_sdpa_bf16(
        qkv_c.data_ptr(),
        qn_weight.data_ptr(),
        kn_weight.data_ptr(),
        cos_c.data_ptr(),
        sin_c.data_ptr(),
        cos_is_bf16,
        1 if out_shd else 0,
        float(eps),
        q_out.data_ptr(),
        k_out.data_ptr(),
        v_out.data_ptr(),
        s,
        num_heads,
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_vis_qkv_sdpa_bf16 CUDA error {err}")
    return q_out, k_out, v_out


def bf16_argmax_cs(
    sub_logits: torch.Tensor,
    core_ids: Optional[torch.Tensor] = None,
    hidden: Optional[torch.Tensor] = None,
    mu_rest: Optional[torch.Tensor] = None,
    w_rest_bound: float = 0.0,
    check_cs: bool = False,
    v_valid: Optional[int] = None,
    out_id: Optional[torch.Tensor] = None,
    out_val: Optional[torch.Tensor] = None,
    out_rescue: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused BF16 vectorized argmax + core_ids remap + optional Cauchy-Schwarz bound check."""
    lib = load()
    assert lib is not None
    sub_c = sub_logits.contiguous()
    m, v_stride = int(sub_c.shape[0]), int(sub_c.shape[1])
    vv = int(v_valid) if v_valid is not None else v_stride
    dev = sub_c.device
    if out_id is None:
        out_id = torch.empty((m,), dtype=torch.int32, device=dev)
    if check_cs:
        if out_val is None:
            out_val = torch.empty((m,), dtype=torch.float32, device=dev)
        if out_rescue is None:
            out_rescue = torch.empty((m,), dtype=torch.uint8, device=dev)
    h_dim = int(hidden.shape[1]) if hidden is not None else 0
    stream = torch.cuda.current_stream(dev).cuda_stream
    err = lib.track2_bf16_argmax_cs(
        sub_c.data_ptr(),
        core_ids.data_ptr() if core_ids is not None else 0,
        hidden.contiguous().data_ptr() if (check_cs and hidden is not None) else 0,
        mu_rest.data_ptr() if (check_cs and mu_rest is not None) else 0,
        float(w_rest_bound),
        out_id.data_ptr(),
        out_val.data_ptr() if out_val is not None else 0,
        out_rescue.data_ptr() if (check_cs and out_rescue is not None) else 0,
        m,
        v_stride,
        vv,
        h_dim,
        1 if check_cs else 0,
        stream,
    )
    if err != 0:
        raise RuntimeError(f"track2_bf16_argmax_cs CUDA error {err}")
    return out_id, out_val, out_rescue
