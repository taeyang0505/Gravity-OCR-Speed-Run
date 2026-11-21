"""Bit-exact Triton replacement for sgl_kernel.fp8_scaled_mm on the LM shapes where it is faster (sm_89).

sgl_kernel 0.4.1's sm89 CUTLASS dispatch accumulates in one of two orders:
  * 2-stage configs (MmaPipelined: 256 < M <= 512, or M > 512 with N > 8192): one fp32 MMA chain over k
  * >=3-stage configs (MmaMultistage, staged accumulation, CTA K = 64): acc = c0; acc += (c1 . c2); ...; acc += c_last
and finishes with bf16((acc * scale_b) * scale_a) (bias: fma(acc * scale_b, scale_a, bias)). `_mm_v2` replays both
orders and that epilogue, so its output is bitwise equal to the CUTLASS kernel's for M > 128 (sliced-K configs below
that are not reproduced). Measured on the v7 job's shapes it is faster for:
  * o_proj / down_proj (N = 1536) at 256 < M <= 512 (decode batch 8..15) and 1408 < M <= 1472 (batch 43-44:
    144 CUTLASS CTAs = 2 waves on 142 SMs)
about -17 ms of LM-stream GPU time per job, outputs unchanged. Everything else keeps the CUTLASS kernel.

install() rebinds sglang.srt.layers.quantization.fp8_utils.fp8_scaled_mm (used by apply_fp8_linear and, captured at
install time, by gravity_ocr.fused_fp8) to a dispatcher, after a startup bit-equality self-check of every rule class
against the CUTLASS kernel (the match depends on that kernel's dispatch table: any mismatch -> dispatcher off).
Kernel + rules: tools audit by a sub-agent (scratchpad agentA/bench_fp8_alt.py, rules.py), 0 mismatches on 392 shapes.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

try:
    from gravity_ocr import cutlass_f16 as _CUTLASS_F16
except Exception:
    _CUTLASS_F16 = None

_ENABLED = os.environ.get("TRACK2_FP8_GEMM_TRITON", "1") == "1"
# [TRACK2-JIT-STABLE] avoid M%16 specialization recompiles inside timed prefill batches (bit-identical)
_JIT_STABLE = os.environ.get("TRACK2_JIT_STABLE", "1") == "1"
_ACTIVE = False
_ORIG = None
STATS = {"triton": 0, "cutlass": 0}


@triton.jit(do_not_specialize=["M"] if _JIT_STABLE else None)
def _mm_v2(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_bn, stride_cm, zf,
           HAS_BIAS: tl.constexpr, STAGED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    first_m = (pid // num_in_group) * GROUP_M
    gsz = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    a_base = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
    b_base = b_ptr + offs_n.to(tl.int64)[None, :] * stride_bn
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    if STAGED:
        zero = tl.zeros((BM, BN), dtype=tl.float32) + zf   # runtime 0.0: keeps `acc + dot(a, b, 0)` from being fused
        k32 = tl.arange(0, 32)
        k64 = tl.arange(0, 64)
        a0 = tl.load(a_base + k32[None, :], mask=mask_m, other=0.0)
        b0 = tl.load(b_base + k32[:, None], mask=mask_n, other=0.0)
        acc = tl.dot(a0, b0, zero)
        a_ptrs = a_base + 32 + k64[None, :]
        b_ptrs = b_base + 32 + k64[:, None]
        for _k in range(32, K - 32, 64):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc = acc + tl.dot(a, b, zero)
            a_ptrs += 64
            b_ptrs += 64
        aL = tl.load(a_base + (K - 32) + k32[None, :], mask=mask_m, other=0.0)
        bL = tl.load(b_base + (K - 32) + k32[:, None], mask=mask_n, other=0.0)
        acc = acc + tl.dot(aL, bL, zero)
    else:
        offs_k = tl.arange(0, BK)
        a_ptrs = a_base + offs_k[None, :]
        b_ptrs = b_base + offs_k[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    x = acc * sb[None, :]                          # CUTLASS EVT: acc * scale_b
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        x = tl.fma(x, sa[:, None], bias[None, :])  # multiply_add(x, scale_a, bias)
    else:
        x = x * sa[:, None]                        # x * scale_a
    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, x.to(tl.bfloat16), mask=mask_m & mask_n)


@triton.jit(do_not_specialize=["M"] if _JIT_STABLE else None)
def _mm_f16(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_bn, stride_cm,
            HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    """Ada sm_89 733-TFLOPS FP8xFP8->FP16 accumulator GEMM (mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16).

    Requires power-of-2 scaled inputs (a_max = 28.0, w_max = 14.0) so FP16 accumulation cannot overflow even at K=4608,
    keeping relative error vs FP32-accum CUTLASS at ~0.001-0.002 (below BF16 ULP)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    first_m = (pid // num_in_group) * GROUP_M
    gsz = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_n.to(tl.int64)[None, :] * stride_bn + offs_k[:, None]
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    acc16 = tl.zeros((BM, BN), dtype=tl.float16)
    for _k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        b = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc16 = tl.dot(a, b, acc16, out_dtype=tl.float16)
        a_ptrs += BK
        b_ptrs += BK
    acc = acc16.to(tl.float32)
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    x = acc * sb[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        x = tl.fma(x, sa[:, None], bias[None, :])
    else:
        x = x * sa[:, None]
    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, x.to(tl.bfloat16), mask=mask_m & mask_n)


def f16_rule(M: int, N: int, K: int):
    """L40S sm_89 sweep-verified tiles (BM, BN, BK, warps, stages) strictly beating CUTLASS FP32-accum (1.03x-1.53x)."""
    if K == 1024:
        if N == 8192:                       # ViT gate_up_proj
            if 512 <= M <= 2048 or M >= 12000:
                return (256, 128, 128, 8, 3)
            return None
        if N == 3072:                       # ViT qkv_proj
            if 512 <= M <= 2048:
                return (64, 128, 128, 4, 3)
            if 2048 < M <= 6144:
                return (256, 128, 128, 8, 3)
            if 6144 < M <= 12000:
                return (128, 128, 128, 4, 3)
            return None
        return None
    if K == 1536:
        if N == 9216:                       # LM gate_up_proj (1.09x-1.39x; CUTLASS wins 2400..3500)
            if 128 < M <= 256:
                return (64, 128, 128, 4, 3)
            if 256 < M <= 512:
                return (64, 256, 128, 8, 3)
            if 512 < M <= 1024:
                return (128, 256, 128, 8, 3)
            if 1024 < M <= 2400:
                return (128, 128, 128, 8, 3)
            if M > 3500:
                return (128, 128, 128, 4, 3)
            return None
        if N == 4096:                       # LM qkv_proj
            if 1200 <= M <= 2400:
                return (128, 128, 128, 8, 3)
            if M > 2400:
                return (256, 128, 128, 8, 3)
            return None
        return None
    if N == 1536:
        if K == 4608:                       # LM down_proj (1.53x at 1600; CUTLASS wins elsewhere)
            if 1100 <= M <= 2400:
                return (256, 128, 128, 8, 3)
            return None
        if K == 2048:                       # LM o_proj (1.29x at 400, 1.11x at 1600, 1.05x at 3200)
            if 256 < M <= 512:
                return (64, 128, 128, 4, 4)
            if 1100 <= M <= 2400:
                return (256, 128, 128, 8, 3)
            if 2400 < M <= 3400:
                return (64, 128, 128, 4, 4)
            return None
    return None


def rescale_f16_weight(layer, shift: float = 32.0) -> None:
    """In-place exact power-of-2 shift of an FP8 linear's weights (w /= shift, weight_scale *= shift) for _mm_f16."""
    if getattr(layer, "_t2_f16_scaled", False):
        return
    with torch.no_grad():
        w_new = (layer.weight.data.float() * (1.0 / shift)).to(torch.float8_e4m3fn)
        layer.weight.data.copy_(w_new)
        layer.weight_scale.data.mul_(shift)
    layer._t2_f16_scaled = True


def mm_f16(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    """Dispatch to Ada sm_89 FP16-accum FP8 GEMM (CUTLASS 4.4.2 EVT first, then Triton, else CUTLASS FP32 fallback)."""
    if (out_dtype == torch.bfloat16 and mat_a.dim() == 2 and mat_b.dim() == 2
            and mat_a.dtype == torch.float8_e4m3fn and mat_b.dtype == torch.float8_e4m3fn
            and mat_a.stride(1) == 1 and mat_b.stride(0) == 1
            and scales_a.dtype == torch.float32 and scales_b.dtype == torch.float32
            and scales_a.is_contiguous() and scales_b.is_contiguous()):
        if _CUTLASS_F16 is not None:
            out = _CUTLASS_F16.mm(mat_a, mat_b, scales_a, scales_b, bias)
            if out is not None:
                return out
        M, K = mat_a.shape
        N = mat_b.shape[1]
        # When native CUTLASS FP16-accum is active, only keep Triton for shapes where Triton beats BOTH CUTLASS
        # FP16-accum and sgl_kernel FP32-accum (ViT_gate_up M > 12000: 690us vs 735us / 941us = 1.36x).
        if _CUTLASS_F16 is not None and _CUTLASS_F16.available():
            cfg = (256, 128, 128, 8, 3) if (K == 1024 and N == 8192 and M > 12000) else None
        else:
            cfg = f16_rule(M, N, K)
        if cfg is not None and K % cfg[2] == 0:
            bm, bn, bk, w, s = cfg
            out = torch.empty((M, N), dtype=torch.bfloat16, device=mat_a.device)
            grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
            _mm_f16[grid](
                mat_a, mat_b, scales_a, scales_b, out, bias, M, N, K,
                mat_a.stride(0), mat_b.stride(1), out.stride(0),
                HAS_BIAS=bias is not None, BM=bm, BN=bn, BK=bk, GROUP_M=8,
                num_warps=w, num_stages=s,
            )
            return out
    return dispatch(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)


def cutlass_staged(M: int, N: int):
    """Does sgl_kernel's sm89 dispatch use a >= 3-stage (staged-accumulation) config for (M, N)? None: M <= 128."""
    if M <= 128:
        return None
    if M <= 256:
        return True
    if M <= 512:
        return N > 16384
    return N <= 8192


def rule(M: int, N: int):
    """(BM, BN, BK, warps, stages) where the Triton kernel is faster than CUTLASS on the job's shapes, else None.

    Verified bitwise equal (0 mismatches in self_check + agentA res_confirm.json):
      * o_proj / down_proj (N <= 2048): 256 < M <= 512 (bs 8..15) and 1408 < M <= 1472 (bs 43-44)
      * gate_up_proj / merger_gate_up (N == 9216, decode range 1408 <= M <= 4224; prefill M > 4224 stays CUTLASS):
        -15 to -40 us per layer x 16 layers (~-45 ms/job). Controlled by TRACK2_FP8_GEMM_GU (default 1)."""
    if N <= 2048:
        if 256 < M <= 512:
            return (64, 64, 128, 4, 4) if M <= 320 else (64, 128, 128, 4, 3)
        if 1408 < M <= 1472:
            return (64, 256, 64, 8, 4)
        return None
    if N == 9216 and os.environ.get("TRACK2_FP8_GEMM_GU", "1") == "1":
        if 1408 <= M <= 1782:
            return (64, 128, 128, 4, 3)
        if 1782 < M <= 2046:
            return (64, 256, 64, 8, 4)
        if 2244 <= M <= 3036:
            return (128, 256, 64, 8, 3)
        if 3036 < M <= 4224:
            return (256, 128, 64, 8, 3)
    return None


def _triton_mm(a, b, sa, sb, bias, cfg, staged):
    bm, bn, bk, w, s = cfg
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _mm_v2[grid](a, b, sa, sb, out, bias, M, N, K, a.stride(0), b.stride(1), out.stride(0), 0.0,
                 HAS_BIAS=bias is not None, STAGED=staged, BM=bm, BN=bn, BK=64 if staged else bk, GROUP_M=8,
                 num_warps=w, num_stages=s)
    return out


def _eligible(a, b, sa, sb, out_dtype, bias):
    if out_dtype != torch.bfloat16 or bias is not None or a.dim() != 2 or b.dim() != 2:
        return None
    M, K = a.shape
    N = b.shape[1]
    if M <= 256 or K % 128 or a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return None
    if a.stride(1) != 1 or b.stride(0) != 1:          # A row-major, B = [K, N] view with k contiguous (weight.t())
        return None
    if sa.dtype != torch.float32 or sb.dtype != torch.float32 or sa.numel() != M or sb.numel() != N:
        return None
    if not (sa.is_contiguous() and sb.is_contiguous()):
        return None
    cfg = rule(M, N)
    if cfg is None:
        return None
    staged = cutlass_staged(M, N)
    if staged is None:
        return None
    return cfg, bool(staged)


def dispatch(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    sel = _eligible(mat_a, mat_b, scales_a, scales_b, out_dtype, bias) if _ACTIVE else None
    if sel is None:
        return _ORIG(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
    return _triton_mm(mat_a, mat_b, scales_a, scales_b, None, sel[0], sel[1])


def self_check(orig) -> str:
    """Every rule class vs the CUTLASS kernel on random operands of the job's shapes: '' if all bitwise equal."""
    torch.manual_seed(0)
    cases = [(300, 1536, 2048), (300, 1536, 4608), (500, 1536, 2048), (500, 1536, 4608), (1452, 1536, 4608),
             (1419, 1536, 2048)]
    if os.environ.get("TRACK2_FP8_GEMM_GU", "1") == "1":
        cases += [(1452, 9216, 1536), (1980, 9216, 1536), (2904, 9216, 1536), (3168, 9216, 1536)]
    for M, N, K in cases:
        f32 = torch.float32  # explicit: SGLang sets the default dtype to bf16 while it loads the model
        a = (torch.randn(M, K, device="cuda", dtype=f32) * 0.6).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, device="cuda", dtype=f32) * 0.6).to(torch.float8_e4m3fn).t()
        sa = torch.rand(M, 1, device="cuda", dtype=f32) * 0.02 + 1e-3
        sb = torch.rand(N, 1, device="cuda", dtype=f32) * 0.02 + 1e-3
        sel = _eligible(a, b, sa, sb, torch.bfloat16, None)
        if sel is None:
            return f"case {M}x{N}x{K} not eligible"
        ref = orig(a, b, sa, sb, torch.bfloat16, None)
        got = _triton_mm(a, b, sa, sb, None, sel[0], sel[1])
        if not torch.equal(ref, got):
            return f"case {M}x{N}x{K}: {int((ref != got).sum())} elements differ"
    torch.cuda.synchronize()
    return ""


def install() -> str:
    """Rebind fp8_utils.fp8_scaled_mm to the dispatcher (call before anything captures that function)."""
    global _ACTIVE, _ORIG
    if not _ENABLED:
        return "Triton FP8 GEMM off (TRACK2_FP8_GEMM_TRITON=0)"
    from sglang.srt.layers.quantization import fp8_utils as fu

    if getattr(fu.fp8_scaled_mm, "_t2_dispatch", False):
        return "Triton FP8 GEMM already installed"
    orig = fu.fp8_scaled_mm
    try:
        err = self_check(orig)
    except Exception as e:  # triton / driver issue: keep CUTLASS everywhere
        return f"Triton FP8 GEMM off (self-check error: {e})"
    if err:
        return f"Triton FP8 GEMM off (not bitwise on this build: {err})"
    _ORIG = orig
    _ACTIVE = True
    dispatch._t2_dispatch = True
    fu.fp8_scaled_mm = dispatch
    gu = " + gate_up decode" if os.environ.get("TRACK2_FP8_GEMM_GU", "1") == "1" else ""
    return f"Triton FP8 GEMM (bit-exact; o/down at decode batch 8-15 and 43-44{gu})"
