# ty_m3: Track 2 winning stack + team taktak changes

Base: the Track 2 winning team's submission v17, github.com/sungu-che/minion @ c0f921ef6829f6a957047c30332c48298cb4aa75
(Apache License 2.0, copy in `license`). Their serving stack, kernels and preprocessing are used unchanged; the native
libraries in `gravity_ocr/_native/` were built from their sources with their `tools/build_kernels.py` and
`tools/build_draft_attn.py` on an L40S (CUDA 12.8, the track2-base:v1 toolchain), and the SGLang overlay with their
`tools/build_overlay.py`.

Changes by team taktak (2026-11-21):
1. `weights_delta/model_delta.safetensors`: the 118 tensors (language model + lm_head, 1.35 GB) in which our draft-tuned
   weights of submission 278 differ from the stock GravityOCR-KO weights; `serve.py` assembles the full model folder from
   the stock weights of the image plus these tensors before the server starts (untimed), to fit the submission storage.
2. `team.yaml`: `TRACK2_TEXT_GLYPH=11 TRACK2_TABLE_GLYPH=11` (their glyph-height target, 12 by default, lowered to 11).

3. `serve.py`: `_merge_weights_delta()` added (the step above); nothing else in their launcher changed.

Measured on a RunPod L40S with the dev set (3 fresh processes, median; quality by the kit's scoring program):
| | wall | text / formula / table |
|---|---:|---|
| their v17 as published | 2.35 s | 97.17 / 95.12 / 98.96 |
| + our draft-tuned weights | 2.29 s | 97.16 / 95.20 / 99.12 |
| + glyph target 11 (this package) | 2.21 s | 97.12 / 95.19 / 98.62 |

Changes by team taktak (2026-11-22), all default-off switches turned on in `TY_SETTINGS` (read by `ty_launch.py`):
4. `gravity_ocr/textscale.py`, glm4v processor: formula crops get the glyph-height downscale too (TRACK2_FORMULA_GLYPH=14).
5. `gravity_ocr/vit_prefetch.py`, ViT graph runner: blank 2x2-group ViT skip with the mask staged ahead on a prep stream
   (no host sync), table crops excluded (SS_VIT_DROP=1, SS_VIT_DROP_TABLE=0).
6. `gravity_ocr/fused_fp8.py` + `gravity_ocr/ss_fused_silu.py`: ViT MLP gate_up GEMM and SiLU*up in one Triton kernel
   (SS_FUSED_SILU_VIT=1).
7. `serve.py`: TRACK2_GLYPH_MIN_PATCHES=256 (their switch) and LANES: two SGLang servers on the GPU under CUDA MPS,
   tables (draft 33, below-normal MPS priority) and text + formula (draft 13); per-lane shm ring path
   (TRACK2_SHM_RING_PATH in the glm4v processor and glm_ocr.py).
8. `serve.py`: the lane servers start one after the other (TRACK2_LANE_SEQUENTIAL, default on with LANES), so that their
   KV pools are sized without racing for GPU memory; per-lane environment (TRACK2_LANE<j>_<VAR>); optional splitting of
   the longest tables at a full-width rule and of the longest text crops at a gap between lines into two requests whose
   outputs are joined (TRACK2_TABLE_SPLIT, TRACK2_TEXT_SPLIT, off unless set in TY_SETTINGS).
9. `TY_SETTINGS`: text-lane draft length 11 (was 13) and up to 200 running requests (was 160); lane memory fractions 0.36 / 0.55 for the sequential start.
10. `TY_SETTINGS`: table glyph-height target 10 px (was 11).
11. `TY_SETTINGS`: the table lane runs its ViT prefetch stream at priority -10 (inside its below-normal MPS client), so
    table images are encoded ahead of the table decode rounds.
12. `TY_SETTINGS`: the text and formula lane packs ViT prefetch batches of at most 8192 patches (was 16384).
