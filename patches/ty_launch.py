#!/usr/bin/env python3
"""team taktak launcher (2026-11-22): sets this submission's switches, then runs the winner-derived serve.py unchanged.

Every switch below is off in serve.py and the patched modules unless set here; an environment variable that is already
set wins (os.environ.setdefault), so the harness or a test can override any of them. The values are written into
TY_SETTINGS by the packaging step (pack_minion_submission.sh) from the measured configuration.
"""
import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
for line in (HERE / "TY_SETTINGS").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
runpy.run_path(str(HERE / "serve.py"), run_name="__main__")
