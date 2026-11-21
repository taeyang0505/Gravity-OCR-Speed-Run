"""Zero-copy ctypes wrapper for Track 2 native CUTLASS Sm89 FP8xFP8->FP16-accum EVT GEMM.

Loads `gravity_ocr/_native/libtrack2_cutlass_f16.so` (compiled inside track2-base:v1 against CUTLASS 4.4.2)
and dispatches directly on `torch.cuda.current_stream().cuda_stream` with a pre-allocated static workspace
(100% CUDA Graph capture safe).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import torch

_LIB: Optional[ctypes.CDLL] = None
_TRIED = False
_WS: Optional[torch.Tensor] = None
_WS_BYTES = 16 * 1024 * 1024  # 16 MB static workspace (enough for Stream-K on any L40S shape)
_ENABLED = os.environ.get("TRACK2_CUTLASS_F16", "1") == "1"


def _find_lib() -> Optional[Path]:
    env_p = os.environ.get("TRACK2_CUTLASS_F16_SO")
    if env_p and os.path.isfile(env_p):
        return Path(env_p)
    here = Path(__file__).resolve().parent / "_native" / "libtrack2_cutlass_f16.so"
    if here.is_file():
        return here
    return None


def load() -> Optional[ctypes.CDLL]:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if not _ENABLED:
        return None
    path = _find_lib()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
        lib.track2_cutlass_f16_abi.argtypes = []
        lib.track2_cutlass_f16_abi.restype = ctypes.c_int
        if int(lib.track2_cutlass_f16_abi()) != 1:
            return None
        vp = ctypes.c_void_p
        i32 = ctypes.c_int
        lib.track2_cutlass_f16_mm.argtypes = [
            i32, vp, vp, vp, vp, vp, vp, i32, i32, i32, vp, ctypes.c_size_t, vp,
        ]
        lib.track2_cutlass_f16_mm.restype = i32
        _LIB = lib
    except Exception:
        _LIB = None
    return _LIB


def available() -> bool:
    return load() is not None


def _get_ws(device) -> torch.Tensor:
    global _WS
    if _WS is None or _WS.device != device:
        _WS = torch.empty(_WS_BYTES, dtype=torch.uint8, device=device)
    return _WS


def select_cfg(M: int, N: int, K: int) -> int:
    """L40S sm_89 sweep-verified CUTLASS FP8->FP16 config id (0..8), or -1 to defer to Triton/sgl_kernel.

    Config map (rust_gpu/kernels/track2_cutlass_f16.cu):
      0: 128x256x64_s3     1: 128x128x64_s4     2: 256x128x64_s3     3: 64x128x64_s5
      4: 64x256x64_s4      5: 64x64x64_s5       6: 64x128x128_s3     7: 64x128x64_s5_SK (Stream-K)
      8: 128x128x128_s3
    """
    if M < 128 or (N & 15) or (K & 15):
        return -1
    # ViT (K=1024 or N=1024)
    if K == 1024:
        if N == 3072:                       # ViT_qkv (1.09x-1.16x across all M)
            if M <= 3200:
                return 1
            if M <= 6400:
                return 0
            if M <= 10240:
                return 1
            return 0
        if N == 1024:                       # ViT_proj (1.04x-1.21x)
            if M <= 6400:
                return 1
            return 2
        if N == 8192:                       # ViT_gate_up (CUTLASS 1.09-1.12x <= 12K; Triton 1.36x > 12K)
            if M <= 12000:
                return 0
            return -1                       # defer to Triton 256x128x128 (690us vs 735us)
        return -1
    if K == 4096 and N == 1024:             # ViT_down (1.04x-1.16x for M >= 4096)
        if M < 4096:
            return -1
        if M <= 6400:
            return 8
        if M <= 10240:
            return 2
        return 0
    # LM (K=1536, 2048, 4608)
    if K == 1536:
        if N == 4096:                       # LM_qkv (1.04x-1.35x across all M)
            if M <= 256:
                return 3
            if M <= 512:
                return 1
            if M <= 1024:
                return 6
            if M <= 2560:
                return 1
            if M <= 3600:
                return 2
            return 0
        if N == 9216:                       # LM_gate_up (1.02x-1.78x)
            if M <= 256:
                return 3
            if M <= 512:
                return 4
            if M <= 1200:
                return 0
            if M <= 1900:
                return -1                   # sgl_kernel 114.9us vs 117.3us at 1600
            if M <= 2700:
                return 0
            if M <= 3600:
                return 1
            return 2
        return -1
    if N == 1536:
        if K == 2048:                       # LM_o_proj (1.07x-2.30x)
            if M <= 256:
                return 5
            if M <= 512:
                return 3
            if M <= 1024:
                return 1
            if M <= 1900:
                return 2
            if M <= 2700:
                return 0
            if M <= 3600:
                return 7                    # Stream-K
            return -1                       # sgl_kernel 65.5us vs 68.0us at 4200
        if K == 4608:                       # LM_down (1.04x-2.35x across ALL M!)
            if M <= 256:
                return 5
            if M <= 512:
                return 3
            if M <= 1024:
                return 4
            if M <= 1900:
                return 7                    # Stream-K (1.59x!)
            if M <= 2700:
                return 0
            if M <= 3600:
                return 7                    # Stream-K (1.17x!)
            return 1
    return -1


def mm(mat_a: torch.Tensor, mat_b: torch.Tensor, scales_a: torch.Tensor, scales_b: torch.Tensor,
       bias: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Run CUTLASS Sm89 FP8->FP16 EVT GEMM if a winning config matches (M, N, K), else return None.

    mat_a: [M, K] row-major float8_e4m3fn
    mat_b: [K, N] col-major view (stride(0) == 1, stride(1) == K) of [N, K] row-major float8_e4m3fn weight
    """
    lib = load()
    if lib is None:
        return None
    M, K = mat_a.shape
    N = mat_b.shape[1]
    if mat_a.stride(0) != K or mat_b.stride(1) != K:
        return None
    cfg_id = select_cfg(M, N, K)
    if cfg_id < 0:
        return None
    if cfg_id == 7 and torch.cuda.is_current_stream_capturing() and os.environ.get("TRACK2_CUTLASS_SK_GRAPH", "1") != "1":
        # Safe fallback to non-SK tile inside CUDA Graph if requested
        cfg_id = 2 if M <= 2000 else 3
    ws = _get_ws(mat_a.device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=mat_a.device)
    bias_ptr = bias.data_ptr() if bias is not None else 0
    stream = torch.cuda.current_stream(mat_a.device).cuda_stream
    rc = lib.track2_cutlass_f16_mm(
        cfg_id,
        mat_a.data_ptr(),
        mat_b.data_ptr(),
        scales_a.data_ptr(),
        scales_b.data_ptr(),
        bias_ptr,
        out.data_ptr(),
        M,
        N,
        K,
        ws.data_ptr(),
        _WS_BYTES,
        stream,
    )
    if rc != 0:
        return None
    return out
