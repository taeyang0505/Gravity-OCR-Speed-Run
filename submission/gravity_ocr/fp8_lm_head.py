"""Fused FP8 W8A8 / BF16 lm_head GEMM + Block-Argmax for Track 2 (L40S sm_89).

Standard `F.linear(hidden, W_pad)` + `gk_bf16_argmax_cs(logits)` writes `[M, 59288]` BF16 (~344 MB at bs=88, K=33)
to HBM and reads all 344 MB back just to find 1 int32 argmax per row (~3.05 ms/call, ~221 ms/job on v11).

This module fuses the GEMM with an in-register tile argmax:
  Stage 1: each CTA computes a `[BM, BN]` tile of `hidden @ W.T` using FP8 W8A8 Tensor Cores (FP32 accum or Ada
           sm_89 733-TFLOPS FP16 accum via `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` with power-of-2
           scaled operands a_max=28, w_max=28 so zero mantissa loss and zero FP16 overflow), multiplies by per-row
           `scale_b` (positive per-token `scale_a` cancels out of argmax), masks `n >= V_valid` to `-inf`, reduces
           `(max_val, argmax)` across the `BN` columns IN REGISTERS, and writes only `[M, N_tiles]` (~1-2 MB).
  Stage 2: reduces across the `N_tiles` partial maxes per row into `[M]` int32 (~8 us).

Modes (`TRACK2_FP8_LM_HEAD`):
  "off"   : stock K3 BF16 F.linear + bf16_argmax_cs
  "draft" : fused FP8 lm_head+argmax for draft rows 1..K-1 (100% bit-exact output: row 0 a0 and verify stay BF16)
  "both"  : fused FP8 lm_head+argmax for both draft and verify (row 0 a0 optionally BF16 via TRACK2_FP8_A0)
  "bf16"  : fused BF16 lm_head+argmax for both draft and verify (no FP8 quantization, skips 344 MB logits round-trip)

Switch: TRACK2_FP8_LM_HEAD_F16=1 enables the Ada sm_89 FP16-accum path (1.24x faster than FP32-accum FP8 lm_head).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from sgl_kernel import sgl_per_token_quant_fp8 as _sgl_quant_fp8
except Exception:
    _sgl_quant_fp8 = None

try:
    from gravity_ocr import cuda_sm89 as _cuda_sm89
except Exception:
    _cuda_sm89 = None


@triton.jit(do_not_specialize=["M"])
def _fp8_lm_head_argmax_stage1(
    a_ptr, b_ptr, sb_ptr, out_val_ptr, out_idx_ptr,
    M, N, K, V_valid,
    stride_am, stride_bn, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    F16_ACC: tl.constexpr,
):
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

    if F16_ACC:
        acc16 = tl.zeros((BM, BN), dtype=tl.float16)
        for _k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc16 = tl.dot(a, b, acc16, out_dtype=tl.float16)
            a_ptrs += BK
            b_ptrs += BK
        acc = acc16.to(tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=mask_m, other=0.0)
            b = tl.load(b_ptrs, mask=mask_n, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK
            b_ptrs += BK

    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
    # scale_a > 0 is constant per row m, so argmax_n(acc[m,n] * sb[n] * sa[m]) == argmax_n(acc[m,n] * sb[n])
    val = acc * sb[None, :]
    val = tl.where(offs_n[None, :] < V_valid, val, float("-inf"))
    best_v, best_local = tl.max(val, axis=1, return_indices=True)
    best_idx = (pid_n * BN + best_local).to(tl.int32)

    out_off = offs_m.to(tl.int64) * stride_om + pid_n
    tl.store(out_val_ptr + out_off, best_v, mask=offs_m < M)
    tl.store(out_idx_ptr + out_off, best_idx, mask=offs_m < M)


@triton.jit(do_not_specialize=["M"])
def _bf16_lm_head_argmax_stage1(
    a_ptr, b_ptr, out_val_ptr, out_idx_ptr,
    M, N, K, V_valid,
    stride_am, stride_bn, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
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

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        b = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK

    val = tl.where(offs_n[None, :] < V_valid, acc, float("-inf"))
    best_v, best_local = tl.max(val, axis=1, return_indices=True)
    best_idx = (pid_n * BN + best_local).to(tl.int32)

    out_off = offs_m.to(tl.int64) * stride_om + pid_n
    tl.store(out_val_ptr + out_off, best_v, mask=offs_m < M)
    tl.store(out_idx_ptr + out_off, best_idx, mask=offs_m < M)


@triton.jit(do_not_specialize=["M"])
def _argmax_stage2(
    val_ptr, idx_ptr, out_ptr,
    M, N_TILES, stride_m,
    NT_POW2: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= M:
        return
    offs = tl.arange(0, NT_POW2)
    mask = offs < N_TILES
    base = m.to(tl.int64) * stride_m
    v = tl.load(val_ptr + base + offs, mask=mask, other=float("-inf"))
    _, best_tile = tl.max(v, axis=0, return_indices=True)
    best_id = tl.load(idx_ptr + base + best_tile)
    tl.store(out_ptr + m, best_id)


def quantize_weight_fp8(W_bf16: torch.Tensor, f16_acc: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row (vocab channel) FP8 e4m3fn quantization of lm_head weight [V, K].

    Returns (W_fp8_t [K, V] column-major with K contiguous, scale_b [V] fp32).
    When f16_acc=True, shifts weights by exact power-of-2 1/32 (w_max = 14.0) so FP16 accum cannot overflow.
    """
    V, K = W_bf16.shape
    w_f32 = W_bf16.float()
    amax = w_f32.abs().amax(dim=1).clamp(min=1e-12)
    fp8_max = 14.0 if f16_acc else 448.0
    scale_b = (amax / fp8_max).contiguous()  # [V] fp32
    w_q = (w_f32 / scale_b[:, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()  # [V, K]
    return w_q.t(), scale_b  # w_q.t() has shape [K, V], stride (1, K)


def quantize_act_fp8(x_bf16: torch.Tensor, f16_acc: bool = False) -> torch.Tensor:
    """Fast per-token FP8 e4m3fn quantization of hidden states [M, K] (scale discarded: monotonic in argmax)."""
    M, K = x_bf16.shape
    if f16_acc:
        if _cuda_sm89 is not None and _cuda_sm89.fused_available():
            q, _ = _cuda_sm89.quant_fp8_rows(x_bf16, qnt_mode=3)
            return q
        amax = x_bf16.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
        return (x_bf16.float() * (28.0 / amax)).clamp(-28.0, 28.0).to(torch.float8_e4m3fn).contiguous()
    if _sgl_quant_fp8 is not None and x_bf16.is_contiguous():
        q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x_bf16.device)
        s = torch.empty((M, 1), dtype=torch.float32, device=x_bf16.device)
        _sgl_quant_fp8(x_bf16, q, s)
        return q
    amax = x_bf16.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
    return (x_bf16.float() * (448.0 / amax)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()


def _pick_fp8_cfg(M: int, f16_acc: bool = False):
    if f16_acc:
        # Swept on L40S sm_89 across M=25..4000: match BM to next_pow2(M) for small M so W_fp8 (91MB) is streamed ONCE!
        if M <= 32:
            return (32, 128, 128, 4, 3)
        if M <= 64:
            return (64, 256, 64, 8, 3)
        if M <= 128:
            return (128, 256, 128, 8, 3)
        if M <= 256:
            return (256, 128, 128, 8, 3)
        if M <= 768:
            return (64, 256, 128, 8, 3)
        return (128, 256, 128, 8, 3)
    # FP32-accum configs swept on L40S sm_89 across M=25..4000
    if M <= 32:
        return (32, 256, 128, 8, 3)
    if M <= 64:
        return (64, 256, 64, 8, 3)
    if M <= 128:
        return (128, 256, 64, 8, 3)
    if M <= 256:
        return (256, 128, 128, 8, 3)
    if M <= 768:
        return (64, 128, 128, 4, 3)
    return (128, 256, 64, 8, 3)


def _pick_bf16_cfg(M: int):
    if M <= 64:
        return (16, 128, 64, 4, 3)
    if M <= 256:
        return (32, 128, 64, 4, 3)
    return (64, 128, 64, 4, 3)


def fused_fp8_argmax(
    x_bf16: torch.Tensor,
    W_fp8_t: torch.Tensor,
    scale_b: torch.Tensor,
    v_valid: int,
    cfg: Optional[Tuple[int, int, int, int, int]] = None,
    f16_acc: bool = False,
) -> torch.Tensor:
    """[M, K] bf16 -> [M] int32 argmax via fused FP8 GEMM + in-register block argmax."""
    if not x_bf16.is_contiguous():
        x_bf16 = x_bf16.contiguous()
    M, K = x_bf16.shape
    N = W_fp8_t.shape[1]
    q_a = quantize_act_fp8(x_bf16, f16_acc=f16_acc)
    bm, bn, bk, warps, stages = cfg or _pick_fp8_cfg(M, f16_acc=f16_acc)
    n_tiles = triton.cdiv(N, bn)
    part_v = torch.empty((M, n_tiles), dtype=torch.float32, device=x_bf16.device)
    part_i = torch.empty((M, n_tiles), dtype=torch.int32, device=x_bf16.device)
    grid1 = (triton.cdiv(M, bm) * n_tiles,)
    _fp8_lm_head_argmax_stage1[grid1](
        q_a, W_fp8_t, scale_b, part_v, part_i,
        M, N, K, v_valid,
        q_a.stride(0), W_fp8_t.stride(1), part_v.stride(0),
        BM=bm, BN=bn, BK=bk, GROUP_M=8, F16_ACC=f16_acc,
        num_warps=warps, num_stages=stages,
    )
    out = torch.empty((M,), dtype=torch.int32, device=x_bf16.device)
    nt_pow2 = triton.next_power_of_2(max(2, n_tiles))
    _argmax_stage2[(M,)](
        part_v, part_i, out,
        M, n_tiles, part_v.stride(0),
        NT_POW2=nt_pow2,
        num_warps=4 if nt_pow2 <= 512 else 8,
    )
    return out


def fused_bf16_argmax(
    x_bf16: torch.Tensor,
    W_bf16_t: torch.Tensor,
    v_valid: int,
    cfg: Optional[Tuple[int, int, int, int, int]] = None,
) -> torch.Tensor:
    """[M, K] bf16 -> [M] int32 argmax via fused BF16 GEMM + in-register block argmax (zero 344 MB logits HBM)."""
    if not x_bf16.is_contiguous():
        x_bf16 = x_bf16.contiguous()
    M, K = x_bf16.shape
    N = W_bf16_t.shape[1]
    bm, bn, bk, warps, stages = cfg or _pick_bf16_cfg(M)
    n_tiles = triton.cdiv(N, bn)
    part_v = torch.empty((M, n_tiles), dtype=torch.float32, device=x_bf16.device)
    part_i = torch.empty((M, n_tiles), dtype=torch.int32, device=x_bf16.device)
    grid1 = (triton.cdiv(M, bm) * n_tiles,)
    _bf16_lm_head_argmax_stage1[grid1](
        x_bf16, W_bf16_t, part_v, part_i,
        M, N, K, v_valid,
        x_bf16.stride(0), W_bf16_t.stride(1), part_v.stride(0),
        BM=bm, BN=bn, BK=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    out = torch.empty((M,), dtype=torch.int32, device=x_bf16.device)
    nt_pow2 = triton.next_power_of_2(max(2, n_tiles))
    _argmax_stage2[(M,)](
        part_v, part_i, out,
        M, n_tiles, part_v.stride(0),
        NT_POW2=nt_pow2,
        num_warps=4 if nt_pow2 <= 512 else 8,
    )
    return out
