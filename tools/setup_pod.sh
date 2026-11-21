#!/bin/bash
# setup_pod.sh -- the track2-base:v1 serving stack installed natively on a RunPod L40S pod (no docker inside a pod).
# Same steps and pins as the kit's track2-base/Dockerfile: trida-stack @79ed00c (SGLang v0.5.10 + GravityOCR patch, uv lock),
# GravityOCR-KO weights @6c749d4. Needs a driver for CUDA 12.8 (>= 570). About 15-20 min, most of it the uv install.
# Then: team folder ~/my-team, dev set ~/dev-set, warm-up set ~/dev-warm, tools ~/tools (copied by push_to_pod.sh),
# runs: bash ~/tools/devrun_pod.sh <name> KEY=VAL ...
set -euo pipefail
TRIDA_STACK_COMMIT=79ed00c72c99ad71edd301302a75bc15ada8e927
GRAVITYOCR_REPO=trillionlabs/GravityOCR-KO
GRAVITYOCR_REVISION=6c749d417879037da8a1f6af4cf9aaa7632308bc
export SGLANG_SRC=/opt/sglang TRIDA_STACK=/opt/trida-stack GRAVITYOCR_WEIGHTS=/opt/gravityocr/weights
export UV_PYTHON_INSTALL_DIR=/opt/uv/python CUDA_HOME=/usr/local/cuda-12.8   # the pinned env needs a 12.x toolkit

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver $drv"; [ "${drv%%.*}" -ge 570 ] || { echo "driver < 570: CUDA 12.8 wheels will not run; pick a CUDA 12.8 host"; exit 1; }

apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential ninja-build \
    cmake libgl1 libglib2.0-0 libnuma1 >/dev/null
# RunPod images ship a CUDA 13 toolkit at /usr/local/cuda; FlashInfer / Triton JIT need the 12.x one beside it (NVIDIA apt repo)
[ -x $CUDA_HOME/bin/nvcc ] || apt-get install -y -qq --no-install-recommends cuda-toolkit-12-8 >/dev/null
$CUDA_HOME/bin/nvcc --version | tail -2
command -v uv >/dev/null && uv --version | grep -q 0.11.27 || curl -LsSf https://astral.sh/uv/0.11.27/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

if [ ! -d $TRIDA_STACK/.git ]; then
  git init -q $TRIDA_STACK && cd $TRIDA_STACK
  git fetch -q --depth 1 https://github.com/trillion-labs/trida-stack.git $TRIDA_STACK_COMMIT && git checkout -q $TRIDA_STACK_COMMIT
fi
[ -x $SGLANG_SRC/.venv-sglang/bin/python ] || bash $TRIDA_STACK/inference/gravityocr/sglang/setup.sh
export PATH=$SGLANG_SRC/.venv-sglang/bin:$CUDA_HOME/bin:$PATH
[ -f $GRAVITYOCR_WEIGHTS/model.safetensors ] || HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$GRAVITYOCR_REPO', revision='$GRAVITYOCR_REVISION', local_dir='$GRAVITYOCR_WEIGHTS')"
python -c "import torch, sglang, transformers; print('torch', torch.__version__, 'sglang', sglang.__version__, \
'transformers', transformers.__version__, 'cuda ok', torch.cuda.is_available())"
cat > /etc/profile.d/track2.sh <<EOF
export SGLANG_SRC=$SGLANG_SRC TRIDA_STACK=$TRIDA_STACK GRAVITYOCR_WEIGHTS=$GRAVITYOCR_WEIGHTS HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME=$CUDA_HOME PATH=$SGLANG_SRC/.venv-sglang/bin:$CUDA_HOME/bin:\$PATH
EOF
echo SETUP_OK
