#!/bin/bash
# pack_minion_submission.sh <name> <TY_SETTINGS file> (team taktak, 2026-11-22): the verified M3 delta package
# (~/subs/unz_m3d) + our default-off patches + ty_launch.py, switches from TY_SETTINGS -> ~/subs/<name>/ and <name>.zip
set -euo pipefail
name=$1; settings=$2
out=$HOME/subs/$name
rm -rf "$out" "$HOME/subs/$name.zip"; cp -r "$HOME/subs/unz_m3d" "$out"
cd "$HOME/minion_port"
python3 patch_minion_formula_glyph.py "$out"
python3 patch_minion_vit_drop_async.py "$out"
python3 patch_minion_vit_drop_tasks.py "$out"
python3 patch_minion_fused_silu.py "$out" "$HOME/fp16acc/fused_gate_up_silu.py"
python3 patch_minion_lanes.py "$out"
cp ty_launch.py "$out/ty_launch.py"; cp "$settings" "$out/TY_SETTINGS"
echo "serve: python3 ty_launch.py" > "$out/team.yaml"
cat >> "$out/NOTICE.md" <<'NOTE'

Changes by team taktak (2026-11-22), all default-off switches turned on in `TY_SETTINGS` (read by `ty_launch.py`):
4. `gravity_ocr/textscale.py`, glm4v processor: formula crops get the glyph-height downscale too (TRACK2_FORMULA_GLYPH=14).
5. `gravity_ocr/vit_prefetch.py`, ViT graph runner: blank 2x2-group ViT skip with the mask staged ahead on a prep stream
   (no host sync), table crops excluded (SS_VIT_DROP=1, SS_VIT_DROP_TABLE=0).
6. `gravity_ocr/fused_fp8.py` + `gravity_ocr/ss_fused_silu.py`: ViT MLP gate_up GEMM and SiLU*up in one Triton kernel
   (SS_FUSED_SILU_VIT=1).
7. `serve.py`: TRACK2_GLYPH_MIN_PATCHES=256 (their switch) and LANES: two SGLang servers on the GPU under CUDA MPS,
   tables (draft 33, below-normal MPS priority) and text + formula (draft 13); per-lane shm ring path
   (TRACK2_SHM_RING_PATH in the glm4v processor and glm_ocr.py).
NOTE
cd "$out"
for f in serve.py ty_launch.py gravity_ocr/*.py gravityocr/sglang/python/sglang/srt/multimodal/processors/glm4v.py \
         gravityocr/sglang/python/sglang/srt/models/glm_ocr.py gravityocr/sglang/python/sglang/srt/multimodal/glm_vit_graph_runner.py; do
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f"
done
find . -name __pycache__ -prune -exec rm -rf {} +
zip -qr "../$name.zip" . && ls -la "../$name.zip" && echo PACK_OK
