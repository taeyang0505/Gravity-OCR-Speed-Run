#!/bin/bash
# setup_scorer.sh -- the official scorer environment of track2-base:v1 on the pod (the Dockerfile's GRAVITYOCR_EVAL step),
# plus ~/starting_kit (scoring program). Run after setup_pod.sh. CPU and network only.
set -euo pipefail
source /etc/profile.d/track2.sh
export GRAVITYOCR_EVAL=/opt/gravityocr-eval
# the metrics run pdflatex as uid 65534 through the eval venv: its interpreter must not live under /root (as in the image)
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
[ -x $GRAVITYOCR_EVAL/venv/bin/python ] || EVAL_HOME=$GRAVITYOCR_EVAL SKIP_SGLANG=1 bash $TRIDA_STACK/benchmark/gravityocr/setup.sh
mkdir -p $GRAVITYOCR_EVAL/bin
{ echo "#!$GRAVITYOCR_EVAL/venv/bin/python"; sed -n '/^import re, sys$/,/^PY$/p' $TRIDA_STACK/benchmark/gravityocr/run.sh | sed '$d'; } \
  > $GRAVITYOCR_EVAL/bin/magick
chmod 755 $GRAVITYOCR_EVAL/bin/magick && grep -q "get_pixmap" $GRAVITYOCR_EVAL/bin/magick
grep -q GRAVITYOCR_EVAL /etc/profile.d/track2.sh || echo "export GRAVITYOCR_EVAL=$GRAVITYOCR_EVAL" >> /etc/profile.d/track2.sh
echo SCORER_OK
