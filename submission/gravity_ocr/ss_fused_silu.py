"""Fused gate_up GEMM + SiLU*up for the Track 2 winning team's FP16-accumulate FP8 MLPs (team taktak, 2026-11-21).

The winner's MLP (gravity_ocr/fused_fp8.py) runs
    gate_up = mm_f16(q8, W_gate_up)            # [M, 2I] bf16 written to memory (CUTLASS FP16-acc EVT GEMM)
    q8, s   = silu_mul_quant_fp8(gate_up)       # reads [M, 2I] bf16, writes [M, I] fp8 + row scales
On the ViT (I = 4096, 189k patches x 24 blocks per job) and the LM (I = 4608) the [M, 2I] bf16 round trip is the largest
memory stream of the MLP. Here one Triton kernel computes both halves of an output tile (gate and up columns j, I + j)
with FP16 accumulation, applies the scales / bias, rounds g and u to bf16 exactly like the reference GEMM output, and
writes h = bf16(silu(g) * u) [M, I]; quant_fp8_rows(h) (the second half of silu_mul_quant_fp8, same arithmetic) then
gives the FP8 input of down_proj. Traffic per row: 2I*2 (write) + 2I*2 (read) + I -> I*2 + I*2 + I.

Same inputs and numerics regime as the winner's _mm_f16 (power-of-2 pre-scaled FP8: a_max 28, w_max 14, so the FP16
accumulator cannot overflow); the only arithmetic difference from the reference is the order of the scale multiply.
"""
import torch
import triton
import triton.language as tl

WARM = False   # set after the first launch (compiled): later launches may sit inside a CUDA-graph capture


@triton.jit(do_not_specialize=["M"])
def _gate_up_silu_f16(a_ptr, b_ptr, sa_ptr, sb_ptr, h_ptr, bias_ptr, M, I, K, stride_am, stride_bn, stride_hm,
                      HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(I, BN)
    num_in_group = GROUP_M * num_pid_n
    first_m = (pid // num_in_group) * GROUP_M
    gsz = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am + offs_k[None, :]
    bg_ptrs = b_ptr + offs_n.to(tl.int64)[None, :] * stride_bn + offs_k[:, None]
    bu_ptrs = bg_ptrs + I.to(tl.int64) * stride_bn
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < I
    acc_g = tl.zeros((BM, BN), dtype=tl.float16)
    acc_u = tl.zeros((BM, BN), dtype=tl.float16)
    for _k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        bg = tl.load(bg_ptrs, mask=mask_n, other=0.0)
        bu = tl.load(bu_ptrs, mask=mask_n, other=0.0)
        acc_g = tl.dot(a, bg, acc_g, out_dtype=tl.float16)
        acc_u = tl.dot(a, bu, acc_u, out_dtype=tl.float16)
        a_ptrs += BK
        bg_ptrs += BK
        bu_ptrs += BK
    nmask = offs_n < I
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    sbg = tl.load(sb_ptr + offs_n, mask=nmask, other=0.0)
    sbu = tl.load(sb_ptr + I + offs_n, mask=nmask, other=0.0)
    g = acc_g.to(tl.float32) * sbg[None, :]
    u = acc_u.to(tl.float32) * sbu[None, :]
    if HAS_BIAS:
        bg_ = tl.load(bias_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)
        bu_ = tl.load(bias_ptr + I + offs_n, mask=nmask, other=0.0).to(tl.float32)
        g = tl.fma(g, sa[:, None], bg_[None, :])
        u = tl.fma(u, sa[:, None], bu_[None, :])
    else:
        g = g * sa[:, None]
        u = u * sa[:, None]
    # the reference GEMM writes bf16 gate / up: round them the same way before the activation
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g))) * u
    h_ptrs = h_ptr + offs_m.to(tl.int64)[:, None] * stride_hm + offs_n[None, :]
    tl.store(h_ptrs, h.to(tl.bfloat16), mask=mask_m & mask_n)


def gate_up_silu(a, b_kn, sa, sb, bias=None, cfg=(256, 64, 128, 8, 3)):
    """a [M, K] fp8 row-major; b_kn [K, 2I] fp8 column-major view of the [2I, K] weight (stride(0) == 1, as SGLang passes
    it to fp8_scaled_mm); sa [M] / [M, 1] fp32; sb [2I] / [2I, 1] fp32; bias [2I] or None -> h [M, I] bf16"""
    M, K = a.shape
    I = b_kn.shape[1] // 2
    BM, BN, BK, warps, stages = cfg
    h = torch.empty((M, I), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(I, BN),)
    _gate_up_silu_f16[grid](a, b_kn, sa, sb, h, bias if bias is not None else h, M, I, K,
                            a.stride(0), b_kn.stride(1), h.stride(0),
                            HAS_BIAS=bias is not None, BM=BM, BN=BN, BK=BK, GROUP_M=8,
                            num_warps=warps, num_stages=stages)
    return h
