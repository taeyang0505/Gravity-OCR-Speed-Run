#!/bin/bash
# pack_minion_submission.sh <name> <TY_SETTINGS file> [base dir] (team taktak, 2026-11-22): the verified M3 delta package
# (~/subs/unz_m3d) + our default-off patches + ty_launch.py, switches from TY_SETTINGS -> ~/subs/<name>/ and <name>.zip
set -euo pipefail
name=$1; settings=$2
out=$HOME/subs/$name
base=${3:-$HOME/subs/unz_m3d}
rm -rf "$out" "$HOME/subs/$name.zip"; cp -r "$base" "$out"
cd "$HOME/minion_port"
python3 patch_minion_formula_glyph.py "$out"
python3 patch_minion_vit_drop_async.py "$out"
python3 patch_minion_vit_drop_tasks.py "$out"
python3 patch_minion_fused_silu.py "$out" "$HOME/fp16acc/fused_gate_up_silu.py"
python3 patch_minion_lanes.py "$out"
python3 patch_minion_lanes_seq.py "$out"
python3 patch_minion_lanes_env.py "$out"
python3 patch_minion_table_split.py "$out"
python3 patch_minion_text_split.py "$out"
cp ty_launch.py "$out/ty_launch.py"; cp "$settings" "$out/TY_SETTINGS"
echo "serve: python3 ty_launch.py" > "$out/team.yaml"
grep -q "Changes by team taktak (2026-11-22), all default-off" "$out/NOTICE.md" || cat >> "$out/NOTICE.md" <<'NOTE'

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
grep -q "LANE_SEQUENTIAL" "$out/NOTICE.md" || cat >> "$out/NOTICE.md" <<'NOTE'
8. `serve.py`: the lane servers start one after the other (TRACK2_LANE_SEQUENTIAL, default on with LANES), so that their
   KV pools are sized without racing for GPU memory; per-lane environment (TRACK2_LANE<j>_<VAR>); optional splitting of
   the longest tables at a full-width rule and of the longest text crops at a gap between lines into two requests whose
   outputs are joined (TRACK2_TABLE_SPLIT, TRACK2_TEXT_SPLIT, off unless set in TY_SETTINGS).
NOTE
cd "$out"
for f in serve.py ty_launch.py gravity_ocr/*.py gravityocr/sglang/python/sglang/srt/multimodal/processors/glm4v.py \
         gravityocr/sglang/python/sglang/srt/models/glm_ocr.py gravityocr/sglang/python/sglang/srt/multimodal/glm_vit_graph_runner.py; do
  python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f"
done
find . -name __pycache__ -prune -exec rm -rf {} +
zip -qr "../$name.zip" . && ls -la "../$name.zip" && echo PACK_OK
