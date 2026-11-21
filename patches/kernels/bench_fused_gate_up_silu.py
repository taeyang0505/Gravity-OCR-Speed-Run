"""Benchmark: winner MLP front half (CUTLASS FP16-acc gate_up GEMM + silu_mul_quant_fp8) vs fused_gate_up_silu +
quant_fp8_rows (team taktak, 2026-11-21). Run on an otherwise idle L40S with the SGLang venv python:

  python bench_fused_gate_up_silu.py <team dir with gravity_ocr/>

Prints, per (shape, M): reference ms, fused ms for a few tile configs, and the error of the dequantized FP8 output
against the reference (relative L2) plus the fraction of FP8 codes that differ.
"""
import os
import sys

import torch
import triton

team = sys.argv[1]
sys.path.insert(0, team)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_ocr import cuda_sm89 as C  # noqa: E402
from gravity_ocr import cutlass_f16 as CF  # noqa: E402
from gravity_ocr import fp8_gemm as G  # noqa: E402
from fused_gate_up_silu import gate_up_silu  # noqa: E402

assert C.available() and CF.available(), "winner kernels not loadable"
torch.manual_seed(0)
dev = "cuda"
SHAPES = {"vit": (1024, 4096, [1024, 2048, 4096, 8192, 16384]),     # K, I, M list
          "lm": (1536, 4608, [256, 512, 1024, 2048, 4096])}
CFGS = [(256, 64, 128, 8, 3), (128, 64, 128, 4, 3), (128, 128, 128, 8, 3), (256, 128, 64, 8, 3), (64, 64, 128, 4, 4)]

for name, (K, I, Ms) in SHAPES.items():
    w = torch.randn(2 * I, K, device=dev) * 0.02
    sb = (w.abs().amax(1) / 14.0).contiguous()                       # w_max 14 regime of the winner's f16 path
    w8 = (w / sb[:, None]).to(torch.float8_e4m3fn)
    b_kn = w8.t()                                                    # [K, 2I] column-major view, stride(0) == 1
    bias = (torch.randn(2 * I, device=dev) * 0.02).to(torch.bfloat16) if name == "vit" else None
    for M in Ms:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        a8, sa = C.quant_fp8_rows(x, qnt_mode=3)                      # a_max 28 regime
        sa1 = sa.view(-1).contiguous()

        def ref():
            gu = G.mm_f16(a8, b_kn, sa, sb, torch.bfloat16, bias)    # the winner's dispatch: CUTLASS f16, Triton, fallback
            return C.silu_mul_quant_fp8(gu, qnt_mode=3)

        q_ref, s_ref = ref()
        deq_ref = q_ref.float() * s_ref
        t_ref = triton.testing.do_bench(ref, warmup=10, rep=50)
        row = f"{name} M={M:6d}: ref {t_ref * 1e3:7.1f} us |"
        for cfg in CFGS:
            try:
                def fused():
                    h = gate_up_silu(a8, b_kn, sa1, sb, bias, cfg)
                    return C.quant_fp8_rows(h, qnt_mode=3)
                q_f, s_f = fused()
                deq_f = q_f.float() * s_f
                err = ((deq_f - deq_ref).norm() / deq_ref.norm()).item()
                diff = (q_f.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean().item()
                t_f = triton.testing.do_bench(fused, warmup=10, rep=50)
                row += f" {cfg[0]}x{cfg[1]}x{cfg[2]}w{cfg[3]}s{cfg[4]} {t_f * 1e3:7.1f} us ({t_ref / t_f:4.2f}x, err {err:.1e}, codes {diff:.1%}) |"
            except Exception as e:
                row += f" {cfg} failed {type(e).__name__} |"
        print(row, flush=True)
