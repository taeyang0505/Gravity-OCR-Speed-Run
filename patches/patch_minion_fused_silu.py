"""SS_FUSED_SILU_VIT=1 (team taktak, 2026-11-21; default off): the ViT MLP's gate_up GEMM and SiLU*up in one Triton kernel
(taeyang/track2/fp16acc/fused_gate_up_silu.py, copied into gravity_ocr/ as ss_fused_silu.py), then the winner's
quant_fp8_rows; replaces gate_up = mm_f16(...) + silu_mul_quant_fp8(gate_up) in the ViT block forward of
gravity_ocr/fused_fp8.py. Only on the FP16-accumulate path (the weights already power-of-2 rescaled by the winner's code).

Measured alone on the L40S (bench_fused_gate_up_silu.py), ViT shapes: 1.02-1.41x faster than the reference pair
(1.19x at M = 16384, the common packed-batch size, where the output is bit-identical); at other M the gate / up values
differ from the CUTLASS FP16-accumulate GEMM in summation order only (7% of FP8 codes move by one step).

Usage: python patch_minion_fused_silu.py <team dir> <fused_gate_up_silu.py>
"""
import shutil
import sys
from pathlib import Path

team, kernel = Path(sys.argv[1]), sys.argv[2]
shutil.copy(kernel, team / "gravity_ocr" / "ss_fused_silu.py")
p = team / "gravity_ocr" / "fused_fp8.py"
s = p.read_text()
if "SS_FUSED_SILU_VIT" in s:
    print("already patched", p)
    sys.exit(0)
old = '''            gate_up = lin_fn(mlp.gate_up_proj, q8, s, x.dtype)
            q8, s = C.silu_mul_quant_fp8(gate_up, qnt_mode=smq_qnt)
            mlp_out = lin_fn(mlp.down_proj, q8, s, x.dtype)'''
new = '''            if use_f16 and _SS_FUSED_SILU_VIT and (_ss_fsilu.WARM or not torch.cuda.is_current_stream_capturing()):
                # team taktak: gate_up GEMM + SiLU*up fused (gravity_ocr/ss_fused_silu.py), then per-row FP8 quant.
                # The first launch (Triton compile) never happens inside a CUDA-graph capture.
                gu = mlp.gate_up_proj
                h = _ss_fsilu.gate_up_silu(q8, gu.weight, s, gu.weight_scale, gu.bias)
                _ss_fsilu.WARM = True
                q8, s = C.quant_fp8_rows(h, qnt_mode=smq_qnt)
            else:
                gate_up = lin_fn(mlp.gate_up_proj, q8, s, x.dtype)
                q8, s = C.silu_mul_quant_fp8(gate_up, qnt_mode=smq_qnt)
            mlp_out = lin_fn(mlp.down_proj, q8, s, x.dtype)'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
hook = "def _down_ok(layer) -> bool:\n"
assert s.count(hook) == 1
s = s.replace(hook, '''# team taktak (2026-11-21): SS_FUSED_SILU_VIT=1 -> ViT gate_up GEMM + SiLU*up in one kernel (default off)
_SS_FUSED_SILU_VIT = os.environ.get("SS_FUSED_SILU_VIT", "0") == "1"
if _SS_FUSED_SILU_VIT:
    from gravity_ocr import ss_fused_silu as _ss_fsilu


''' + hook, 1)
if "\nimport os\n" not in s:
    s = "import os\n" + s
p.write_text(s)
print("patched", p)
