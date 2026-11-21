#!/usr/bin/env bash
# Track 2 launcher for GravityOCR on one L40S (Ada sm_89, 48 GB, 8 vCPU).
#   - ViT prefetch: packed ViT batches on a low-priority stream, decode on a high-priority one (gravity_ocr/vit_prefetch.py)
#   - MAXREQ=96 with CUDA graphs up to bs 96; FA2 varlen ViT attention; font-size-aware downscale of large text crops
#   - TRACK2_SHM_GPU_MM=1: 56-byte descriptors + page-locked shm ring -> GPU resize + fused patchify / normalize
#   - TRACK2_FAST_LM_HEAD=1: 16K-row draft lm_head + exact a0 + full-vocab BF16 verify, argmax on the GPU
#   - TRACK2_OWN_LAUNCH=1: start SGLang directly (the image's serve.sh, with the same settings, refuses some server flags,
#     e.g. --quantization); TRACK2_SGLANG_ARGS appends server flags in that mode.
set -euo pipefail

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PORT="${PORT:-30000}"
export MODE="${MODE:-self-spec}"
export FAST="${FAST:-1}"
export MAXREQ="${MAXREQ:-160}"
export GRAPHBS="${GRAPHBS:-160}"
export DRAFT="${DRAFT:-25}"
export TRACK2_PREFILL_COALESCE="${TRACK2_PREFILL_COALESCE:-16}"
export TRACK2_PREFILL_MAX_WAIT_S="${TRACK2_PREFILL_MAX_WAIT_S:-0.02}"
export TRACK2_VIT_FA2="${TRACK2_VIT_FA2:-1}"
export TRACK2_VIT_PREFETCH="${TRACK2_VIT_PREFETCH:-1}"
export MEMFRAC="${MEMFRAC:-0.82}"
export SS_GLM_VIT_GRAPH_MAXM="${SS_GLM_VIT_GRAPH_MAXM:-2048}"
export SS_GLM_VIT_GRAPH_CHECK="${SS_GLM_VIT_GRAPH_CHECK:-0}"
export TRACK2_FAST_MM="${TRACK2_FAST_MM:-1}"
export TRACK2_SHM_GPU_MM="${TRACK2_SHM_GPU_MM:-1}"
export TRACK2_FOLD_PATCH_EMBED="${TRACK2_FOLD_PATCH_EMBED:-1}"
export TRACK2_MM_WORKERS="${TRACK2_MM_WORKERS:-4}"
export TRACK2_TEXT_DOWNSCALE="${TRACK2_TEXT_DOWNSCALE:-1}"
# v14 size policy: glyph-body-height normalization (12 px target for text & table, 12 px floor upscale for tiny text;
# pitch rule fallback at 64 px / 4096 patches; table safety cap 4096) + margin_crop (28 px grid snap)
export TRACK2_TEXT_PITCH="${TRACK2_TEXT_PITCH:-64}"
export TRACK2_TEXT_MIN_PATCHES="${TRACK2_TEXT_MIN_PATCHES:-4096}"
export TRACK2_TEXT_GLYPH="${TRACK2_TEXT_GLYPH:-12}"
export TRACK2_TABLE_GLYPH="${TRACK2_TABLE_GLYPH:-12}"
export TRACK2_TEXT_GLYPH_UP="${TRACK2_TEXT_GLYPH_UP:-12}"
export TRACK2_TABLE_GLYPH_UP="${TRACK2_TABLE_GLYPH_UP:-0}"
export TRACK2_TABLE_MAX_PATCHES="${TRACK2_TABLE_MAX_PATCHES:-4096}"
export TRACK2_MARGIN_CROP="${TRACK2_MARGIN_CROP:-1}"
export TRACK2_KV_NOSPLIT="${TRACK2_KV_NOSPLIT:-1}"
export TRACK2_ATTN_NOFILL="${TRACK2_ATTN_NOFILL:-1}"
export TRACK2_JIT_STABLE="${TRACK2_JIT_STABLE:-1}"
export TRACK2_VIS_QKV_CUDA="${TRACK2_VIS_QKV_CUDA:-1}"
export TRACK2_FAST_LM_HEAD="${TRACK2_FAST_LM_HEAD:-1}"
export TRACK2_FP8_LM_HEAD="${TRACK2_FP8_LM_HEAD:-both}"
export TRACK2_FP8_A0="${TRACK2_FP8_A0:-1}"
export TRACK2_FP8_LM_HEAD_F16="${TRACK2_FP8_LM_HEAD_F16:-1}"
export TRACK2_FUSED_MROPE_KV="${TRACK2_FUSED_MROPE_KV:-1}"
export TRACK2_FP8_GEMM_GU="${TRACK2_FP8_GEMM_GU:-1}"
export TRACK2_F16ACC="${TRACK2_F16ACC:-all}"
export TRACK2_CUTLASS_F16="${TRACK2_CUTLASS_F16:-1}"
export TRACK2_VIT_HIDE_LAUNCH="${TRACK2_VIT_HIDE_LAUNCH:-1}"
export TRACK2_EXTEND_HIDE="${TRACK2_EXTEND_HIDE:-1}"
export TRACK2_RECV_HIDE="${TRACK2_RECV_HIDE:-2}"
export TRACK2_RECV_LIMIT="${TRACK2_RECV_LIMIT:-24}"
export SGLANG_MM_SKIP_COMPUTE_HASH="${SGLANG_MM_SKIP_COMPUTE_HASH:-1}"
# uvicorn closes idle keep-alive sockets after 5 s by default; the client's pre-opened sockets must survive the warmup
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-3600}"
export TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-1}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:---chunked-prefill-size 16384 --schedule-conservativeness 0.3 --log-level warning}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TRACK2_SLICE_DIR="${TRACK2_SLICE_DIR:-${REPO_ROOT}/vocab_slices}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Default: own launch with online FP8 (W8A8: per-channel weights, per-token dynamic activations; LLM + ViT linears,
# lm_head stays BF16). TRACK2_OWN_LAUNCH=0 runs the image's serve.sh (BF16, no extra server flags).
export TRACK2_OWN_LAUNCH="${TRACK2_OWN_LAUNCH:-1}"
export TRACK2_SGLANG_ARGS="${TRACK2_SGLANG_ARGS---quantization fp8}"

STOCK_SERVE="${TRIDA_STACK:-/opt/trida-stack}/inference/gravityocr/serve.sh"
if [[ "${TRACK2_OWN_LAUNCH}" != 1 && -f "${STOCK_SERVE}" ]]; then
    exec bash "${STOCK_SERVE}" "$@"
fi

# ---- own launch: the stock serve.sh's self-spec / FAST=1 settings, then extra server flags ------------------------
VBIN="${VBIN:-${SGLANG_SRC:-/opt/sglang}/.venv-sglang/bin}"
CKPT="${CKPT:-${GRAVITYOCR_WEIGHTS:-/opt/gravityocr/weights}}"
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PYTHONSAFEPATH=1
export PATH="$VBIN:$CUDA_HOME_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME="$CUDA_HOME_DIR" HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset PYTHONOPTIMIZE
export SS_GLM_PV_BF16="${SS_GLM_PV_BF16:-1}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
export SELFSPEC_FULLDRAFT="${SELFSPEC_FULLDRAFT:-1}" SELFSPEC_POS_OFFSET="${SELFSPEC_POS_OFFSET:-1}"
export SELFSPEC_FASTATTN="${SELFSPEC_FASTATTN:-1}" SELFSPEC_MROPE_FIX="${SELFSPEC_MROPE_FIX:-1}" \
       SELFSPEC_FAST_COMMIT="${SELFSPEC_FAST_COMMIT:-1}" SELFSPEC_HOSTLEN="${SELFSPEC_HOSTLEN:-1}" \
       SELFSPEC_DEV_POS="${SELFSPEC_DEV_POS:-1}" SELFSPEC_DRAFT_AHEAD="${SELFSPEC_DRAFT_AHEAD:-1}"
[ -f "$CKPT/block_diffusion.json" ] || export SELFSPEC_MASK_ID="${SELFSPEC_MASK_ID:-59282}"
_t2_args="${TRACK2_SGLANG_ARGS:-}"
read -r -a EXTRA <<< "${EXTRA_ARGS} ${_t2_args//,/ }" || true   # TRACK2_SGLANG_ARGS: space- or comma-separated
echo "[serve] own launch MODE=$MODE FAST=$FAST CKPT=$CKPT extra=${EXTRA[*]}"
exec "$VBIN/python" -m sglang.launch_server \
    --model-path "$CKPT" --served-model-name glm-ocr --host 127.0.0.1 --port "$PORT" \
    --attention-backend flashinfer --mem-fraction-static "$MEMFRAC" --context-length "${CONTEXT:-16384}" \
    --tokenizer-worker-num "$TOKENIZER_WORKERS" --max-running-requests "$MAXREQ" --cuda-graph-max-bs "$GRAPHBS" \
    --speculative-algorithm SELFSPEC_DIFFUSION --speculative-num-draft-tokens "${DRAFT:-33}" --page-size 1 \
    "${EXTRA[@]}"
