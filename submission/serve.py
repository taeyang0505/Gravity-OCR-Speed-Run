#!/usr/bin/env python3
"""Track 2 (Gravity-OCR Speed Run) Zero-Thread Async + Pure-BF16 GPU-Accelerated Server (V3).

Architecture verified on NVIDIA L40S (Ada sm_89, 8-vCPU) + track2-base:v1:
1. Single-threaded `asyncio` HTTP/1.1 Keep-Alive connection pool (pre-warmed TCP_NODELAY sockets):
   eliminates 96 OS worker threads & CPU context-switching contention against SGLang's GPU Scheduler.
2. Zero-decode `/dev/shm/track2_crops/c_{idx}.png` symlinks + `/dev/shm/track2_raw_chw.bin` zero-pickle
   mmap ring buffer + GPU bicubic resize & `gk_chw_u8_patchify_norm_bf16` + Folded PatchEmbed.
3. Native BF16 CUDA `sm_89` fused CogViT QKV+RMSNorm+RoPE (`gk_vis_qkv_sdpa_bf16`) + 16K Draft /
   Full-BF16 Verify `lm_head` (`gk_bf16_argmax_cs`, zero 751 MB FP32 logits cast).
4. Header-only (~4ms) LPT scheduling (`table -> text -> formula`, descending `W * H`) + full untimed
   `dev-set` warmup so zero Triton/FlashInfer JIT compilation occurs during timed runs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WORK = os.environ["TRACK2_DIR"]
DEV = os.environ.get("TRACK2_DEV_DIR", "")
SGLANG = os.environ.get("SGLANG_SRC", "/opt/sglang")
TRIDA_GOCR = os.path.join(os.environ.get("TRIDA_STACK", "/opt/trida-stack"), "inference", "gravityocr")

PORT = int(os.environ.get("PORT", "30000"))
URL = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 4096

STAGE = int(os.environ.get("TRACK2_STAGE", "3"))
CONCURRENCY = int(os.environ.get("TRACK2_CONCURRENCY", os.environ.get("CONCURRENCY", "160" if STAGE >= 1 else "48")))
# In-flight HTTP requests. The server still runs at most MAXREQ (= CONCURRENCY) at once; sending every crop up front
# lets the tokenizer decode all images while the GPU works and keeps SGLang's waiting queue full, so prefill batches are
# large (efficient ViT / extend) instead of one crop per freed slot.
CLIENT_CONC = int(os.environ.get("TRACK2_CLIENT_CONC", "1024" if STAGE >= 1 else str(CONCURRENCY)))
SEND_FIRST = int(os.environ.get("TRACK2_SEND_FIRST", "8"))        # crops sent before the first pause
SEND_WAVE = max(1, int(os.environ.get("TRACK2_SEND_WAVE", "16")))  # crops per wave after that
SEND_GAP_S = float(os.environ.get("TRACK2_SEND_GAP_MS", "6")) / 1e3
# "generate": SGLang's /generate with the chat template pre-applied (cheap per request); "chat": /v1/chat/completions
API = os.environ.get("TRACK2_API", "generate" if STAGE >= 1 else "chat")
# GravityOCR's chat template rendered for [image, text] (identical to what the OpenAI layer sends to the tokenizer)
PROMPT_TEMPLATE = "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>{}<|assistant|>\n"
USE_SHM_IPC = STAGE >= 1 and os.environ.get("TRACK2_SHM_IPC", "1") == "1"

SHM_DIR = Path("/dev/shm/track2_crops")
if USE_SHM_IPC:
    try:
        SHM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        USE_SHM_IPC = False

IPC_MODE = "shm_symlink" if USE_SHM_IPC else "png_b64"
# file:// straight to the job folder when the path is plain ASCII (no symlink syscalls); 0 = always symlink into /dev/shm
DIRECT_FILE_URL = os.environ.get("TRACK2_DIRECT_FILE_URL", "1") == "1"
# fsync of predictions.jsonl before DONE: not needed for other processes to read the file
FSYNC_PREDICTIONS = os.environ.get("TRACK2_FSYNC_PREDICTIONS", "0") == "1"
# Request ordering: "dens" (longest predicted decode rounds / area^beta, output-neutral), "rounds", "area", "text"
ORDER = os.environ.get("TRACK2_ORDER", "v2f:0.05")
ORDER_BETA = float(os.environ.get("TRACK2_ORDER_BETA", "0.25"))
TEXT_BOOST = float(os.environ.get("TRACK2_TEXT_BOOST", "1.0"))
FORMULA_BOOST = float(os.environ.get("TRACK2_FORMULA_BOOST", "1.0"))
_JOB_NO = 0

# team taktak (2026-11-21): LANES, one SGLang server per lane (see taeyang/track2/minion_port/patch_minion_lanes.py).
# Default off: TRACK2_LANES unset keeps the single server.
LANES = [[t.strip() for t in part.split(",") if t.strip()]
         for part in os.environ.get("TRACK2_LANES", "").split(";") if part.strip()]
LANE_PORT_STEP = int(os.environ.get("TRACK2_LANE_PORT_STEP", "100"))
LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"
# team taktak (2026-11-22): start the lane servers one after the other (patch_minion_lanes_seq.py)
LANE_SEQUENTIAL = os.environ.get("TRACK2_LANE_SEQUENTIAL", "1") == "1"
_LANE_KEYS = {"DRAFT": "DRAFT", "MAXREQ": "MAXREQ", "GRAPHBS": "GRAPHBS", "MEMFRAC": "MEMFRAC",
              "MM_WORKERS": "TRACK2_MM_WORKERS", "VIT_PF_PATCHES": "TRACK2_VIT_PF_PATCHES",
              "MPS_PRIO": "CUDA_MPS_CLIENT_PRIORITY", "MPS_SM_PCT": "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}
_MPS_ENV: dict = {}
# team taktak (2026-11-22): TRACK2_TABLE_SPLIT, see taeyang/track2/minion_port/patch_minion_table_split.py
TABLE_SPLIT = int(os.environ.get("TRACK2_TABLE_SPLIT", "0") or 0)
TABLE_SPLIT_MINH = int(os.environ.get("TRACK2_TABLE_SPLIT_MINH", "200") or 200)
TABLE_SPLIT_RULE = float(os.environ.get("TRACK2_TABLE_SPLIT_RULE", "0.95") or 0.95)
TEXT_SPLIT = int(os.environ.get("TRACK2_TEXT_SPLIT", "0") or 0)          # team taktak (patch_minion_text_split.py)
TEXT_SPLIT_MINLINES = int(os.environ.get("TRACK2_TEXT_SPLIT_MINLINES", "5") or 5)


def _split_text(p: dict):
    """two half-crop requests cut at a blank gap between text lines near the middle, or None"""
    import numpy as np
    path = p.get("img_path")
    if not path:
        return None
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    g = np.asarray(rgb.convert("L")).astype(np.int16)
    H, W = g.shape
    ink = (np.abs(g - int(np.median(g))) > 60).any(axis=1)
    d = np.diff(np.concatenate([[1], ink.astype(np.int8), [1]]))
    st, en = np.flatnonzero(d == -1), np.flatnonzero(d == 1)          # blank runs [st, en)
    gaps = [(a, b) for a, b in zip(st, en) if a > 0 and b < H and b - a >= 2]
    if len(gaps) + 1 < TEXT_SPLIT_MINLINES:
        return None
    mid = [(a + b) // 2 for a, b in gaps if 0.3 * H <= (a + b) / 2.0 <= 0.7 * H]
    if not mid:
        return None
    y = int(min(mid, key=lambda v: abs(v - H / 2.0)))
    out = []
    for k, (a, b) in enumerate(((0, y), (y, H))):
        fn = SHM_DIR / f"xs_{abs(hash(p['id'])) % 10**12}_{k}.png"
        rgb.crop((0, a, W, b)).save(fn, compress_level=1)
        q = dict(p)
        q.update(id=f"{p['id']}#xs{k}", image_url="file://" + str(fn), h=b - a, _split_of=p["id"])
        out.append(q)
    return out
_SPLIT_POOL = None


def _split_pool():
    global _SPLIT_POOL
    if _SPLIT_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _SPLIT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="t2split")
    return _SPLIT_POOL


def _split_table(p: dict):
    """two half-table requests cut at a full-width rule near the middle, or None"""
    import numpy as np
    path = p.get("img_path")
    if not path:
        return None
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    g = np.asarray(rgb.convert("L")).astype(np.int16)
    H, W = g.shape
    if H < TABLE_SPLIT_MINH:
        return None
    # rule rows: off the background colour (light grey lines count) across the table's whole width, in runs of
    # at most 3 px (a filled row band, e.g. zebra striping or a coloured header, holds text and is never cut)
    ink = np.abs(g - int(np.median(g))) > 8
    cols = np.flatnonzero(ink.any(axis=0))
    if cols.size < 2:
        return None
    full = ink[:, int(cols[0]):int(cols[-1]) + 1].mean(axis=1) >= TABLE_SPLIT_RULE
    d = np.diff(np.concatenate([[0], full.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    thin = [(a + b - 1) // 2 for a, b in zip(st, en) if b - a <= 3]
    mid = [y for y in thin if 0.3 * H <= y <= 0.7 * H]
    if not mid:
        return None
    y = int(min(mid, key=lambda v: abs(v - H / 2.0)))
    out = []
    for k, (a, b) in enumerate(((0, min(H, y + 2)), (max(0, y - 1), H))):
        fn = SHM_DIR / f"ts_{abs(hash(p['id'])) % 10**12}_{k}.png"
        rgb.crop((0, a, W, b)).save(fn, compress_level=1)
        q = dict(p)
        q.update(id=f"{p['id']}#ts{k}", image_url="file://" + str(fn), h=b - a, _split_of=p["id"])
        out.append(q)
    return out


def _merge_table_html(parts: list) -> str:
    """rows of the top part, then rows of the bottom part, in one <table>"""
    if not parts:
        return ""
    out = parts[0].rstrip()
    for nxt in parts[1:]:
        a = re.sub(r"(</tbody>)?\s*</table>\s*$", "", out)
        b = re.sub(r"^\s*<table[^>]*>\s*(<tbody>)?", "", nxt.strip())
        if a == out or b == nxt.strip():       # not two tables: keep both texts
            out = out + nxt
        else:
            out = a + b
    return out
_SGLANG_PY = None


def lane_of_task(task: str) -> int:
    for j, tasks in enumerate(LANES):
        if task in tasks:
            return j
    return len(LANES) - 1


def lane_env(j: int) -> dict:
    out = {}
    pre = f"TRACK2_LANE{j}_"   # team taktak: TRACK2_LANE<j>_<VAR>=value sets VAR for lane j only (patch_minion_lanes_env.py)
    for k, v in os.environ.items():
        if k.startswith(pre) and len(k) > len(pre):
            out[k[len(pre):]] = v
    for k, var in _LANE_KEYS.items():
        vals = [v.strip() for v in os.environ.get(f"TRACK2_LANE_{k}", "").split(",")]
        if j < len(vals) and vals[j]:
            out[var] = vals[j]
    return out


def lane_ports() -> list:
    return [PORT + j * LANE_PORT_STEP for j in range(len(LANES))] if LANES else [PORT]


def start_mps() -> None:
    """CUDA MPS daemon with pipe / log dirs under $HOME (writable by the unprivileged eval user)."""
    pipe, logd = os.path.join(os.environ["HOME"], "mps-pipe"), os.path.join(os.environ["HOME"], "mps-log")
    os.makedirs(pipe, exist_ok=True)
    os.makedirs(logd, exist_ok=True)
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=logd)
    try:
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True, timeout=20)
        log(f"MPS daemon: rc={r.returncode} {r.stderr.strip()[:120]}")
        if r.returncode == 0:
            _MPS_ENV.update(CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_MPS_LOG_DIRECTORY=logd)
    except Exception as e:
        log(f"MPS daemon not started ({e!r}): the servers will time-share the GPU")


def stop_mps() -> None:
    if not _MPS_ENV:
        return
    try:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=dict(os.environ, **_MPS_ENV),
                       capture_output=True, text=True, timeout=20)
    except Exception:
        pass


class LanePools:
    """One AsyncKeepAlivePool per lane; a request goes to the lane named at the end of its rid ("...l<lane>")."""

    def __init__(self, pools: list) -> None:
        self.pools = pools
        self.max_conns = sum(p.max_conns for p in pools)
        try:   # one pool's prewarm already holds 512 sockets: lift the soft open-file limit up to the hard one
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            want = hard if hard != resource.RLIM_INFINITY else 65536
            if soft != resource.RLIM_INFINITY and soft < want:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except Exception as e:
            log(f"open-file limit unchanged ({e!r})")

    @property
    def n_retry(self) -> int:
        return sum(p.n_retry for p in self.pools)

    @property
    def n_fail(self) -> int:
        return sum(p.n_fail for p in self.pools)

    async def prewarm(self, n: int | None = None) -> None:
        per = max(1, (n or 512) // len(self.pools))   # the single-server total, shared out
        for p in self.pools:
            await p.prewarm(per)

    async def close_all(self) -> None:
        for p in self.pools:
            await p.close_all()

    async def post_chat(self, image_url: str, prompt: str, rid: str | None = None) -> str:
        j = 0
        if rid and "l" in rid[3:]:
            try:
                j = int(rid.rsplit("l", 1)[1])
            except ValueError:
                j = 0
        return await self.pools[j].post_chat(image_url, prompt, rid)

TASK_PROMPTS = {
    "table": "Table Recognition:",
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
}
TASK_BUCKET_ORDER = {"table": 0, "text": 1, "formula": 2}

FENCE_HEAD_RE = re.compile(r"^\s*```(?:html|markdown|md|latex|tex|text)?\s*\n?", re.IGNORECASE)
FENCE_TAIL_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)

try:  # reference-convention output normalization (gravity_ocr/postfix.py)
    from gravity_ocr import postfix as _postfix
except Exception:
    _postfix = None

try:  # multi-layer run log (gravity_ocr/t2log.py), client layer L1
    from gravity_ocr import t2log as _t2log
    if _t2log.LEVEL < 1:
        _t2log = None
except Exception:
    _t2log = None


def log(msg: str) -> None:
    print(f"[track2-minion] {msg}", flush=True)


def own(name: str, stock: str | None) -> str | None:
    mine = str(HERE / name)
    return mine if os.path.exists(mine) else stock


def build_sglang() -> str:
    """Rebuild SGLang Python tree in $HOME/sglang-build (untimed) with submission overlay on top."""
    stock = os.path.join(TRIDA_GOCR, "sglang")
    base_file = own("gravityocr/sglang/BASE_COMMIT.txt", os.path.join(stock, "BASE_COMMIT.txt"))
    patch = own("gravityocr/sglang/gravityocr_selfspec.patch", os.path.join(stock, "gravityocr_selfspec.patch"))
    assert base_file is not None and patch is not None
    base = open(base_file, encoding="utf-8").read().split()[0]
    out = os.path.join(os.environ["HOME"], "sglang-build")
    subprocess.run(["rm", "-rf", out], check=True)
    os.makedirs(out, exist_ok=True)
    subprocess.run(f"git --git-dir={SGLANG}/.git archive {base} python/sglang | tar x -C {out}", shell=True, check=True)
    subprocess.run(["git", "apply", "--include=python/sglang/*", patch], cwd=out, check=True)
    version = os.path.join(SGLANG, "python", "sglang", "_version.py")
    if os.path.exists(version):
        subprocess.run(["cp", version, os.path.join(out, "python", "sglang")], check=True)
    for files in (os.path.join(stock, "python"), own("gravityocr/sglang/python", None)):
        if files and os.path.isdir(files):
            subprocess.run(["cp", "-r", files + "/.", os.path.join(out, "python")], check=True)
    log(f"built SGLang: base {base[:7]} + patch {patch} + python/ overlay")
    return os.path.join(out, "python")


def start_server(lane: int | None = None) -> subprocess.Popen:
    global _SGLANG_PY
    env = dict(os.environ)
    env.update({
        "VBIN": os.path.join(SGLANG, ".venv-sglang", "bin"),
        "CKPT": own("weights", os.environ.get("GRAVITYOCR_WEIGHTS", "/opt/gravityocr/weights")) or "/opt/gravityocr/weights",
        "MODE": os.environ.get("MODE", "self-spec"),
        "FAST": os.environ.get("FAST", "1"),
        "PORT": str(PORT),
        "SKIP_PREFLIGHT": "1",
        "HF_HOME": os.path.join(os.environ["HOME"], ".cache", "huggingface"),
        "TOKENIZER_WORKERS": os.environ.get("TOKENIZER_WORKERS", "1"),
        "MAXREQ": os.environ.get("MAXREQ", str(CONCURRENCY)),
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING", "EAGER"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "TRACK2_STAGE": str(STAGE),
        "TRACK2_RESCUE": os.environ.get("TRACK2_RESCUE", "1"),
        "TRACK2_SLICE_DIR": os.environ.get("TRACK2_SLICE_DIR", str(HERE / "vocab_slices")),
    })
    if lane is not None:   # team taktak: LANES
        env["PORT"] = str(PORT + lane * LANE_PORT_STEP)
        env.update(lane_env(lane))
        env["TRACK2_SHM_RING_PATH"] = f"/dev/shm/track2_raw_chw_l{lane}.bin"
        env.update(_MPS_ENV)

    py_paths: list[str] = []
    sglang_label = "base image"
    if own("gravityocr/sglang", None):
        if _SGLANG_PY is None:          # team taktak: build once (a second build would wipe the first server's tree)
            _SGLANG_PY = build_sglang()
        sglang_py = _SGLANG_PY
        py_paths.append(sglang_py)
        sglang_label = sglang_py
    py_paths.append(str(HERE))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    script = own("gravityocr/serve.sh", os.path.join(TRIDA_GOCR, "serve.sh"))
    assert script is not None
    log(
        f"starting {script} lane={lane} PORT={env['PORT']} DRAFT={env.get('DRAFT', '-')} MODE={env['MODE']} FAST={env['FAST']} MAXREQ={env['MAXREQ']} "
        f"STAGE={STAGE} CKPT={env['CKPT']} sglang={sglang_label}"
    )
    return subprocess.Popen(["bash", script], env=env, start_new_session=True)


def wait_ready(server: subprocess.Popen, timeout: int = 270, url: str = URL) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit(f"[track2-minion] server exited early with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    sys.exit("[track2-minion] server not ready in time")


def sanitize_output(text: str, task_key: str) -> str:
    if not text or STAGE == 0:
        return text or ""
    if "```" in text:
        cleaned = FENCE_HEAD_RE.sub("", text.strip())
        text = FENCE_TAIL_RE.sub("", cleaned).strip()
    return _postfix.fix(text, task_key) if _postfix is not None else text


def detect_task(item: dict[str, Any]) -> str:
    raw = f"{item.get('task', '')} {item.get('prompt', '')}".lower()
    if "table" in raw:
        return "table"
    if "formula" in raw:
        return "formula"
    return "text"


class AsyncKeepAlivePool:
    """Zero-thread persistent HTTP/1.1 Keep-Alive client pool over loopback TCP_NODELAY sockets."""

    def __init__(self, host: str, port: int, max_conns: int) -> None:
        self.host = host
        self.port = port
        self.max_conns = max_conns
        self._sem = asyncio.Semaphore(max_conns)
        self._idle: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        self.n_retry = 0   # requests re-sent on a fresh socket (stale keep-alive socket)
        self.n_fail = 0    # requests that failed on every attempt

    async def _open_conn(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection(self.host, self.port, limit=4 * 1024 * 1024)
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        return reader, writer

    async def prewarm(self, n: int | None = None) -> None:
        target = min(self.max_conns, n or self.max_conns)
        while len(self._idle) < target:
            try:
                self._idle.append(await self._open_conn())
            except Exception:
                break

    async def close_all(self) -> None:
        while self._idle:
            _, w = self._idle.pop()
            try:
                w.close()
            except Exception:
                pass

    @staticmethod
    async def _read_http_response(reader: asyncio.StreamReader) -> tuple[int, bytes, bool]:
        hdr_bytes = await reader.readuntil(b"\r\n\r\n")
        lines = hdr_bytes.decode("latin-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        clen: int | None = None
        chunked = False
        keep_alive = True
        for h in lines[1:]:
            if not h or ":" not in h:
                continue
            k, v = h.split(":", 1)
            kl = k.strip().lower()
            vl = v.strip().lower()
            if kl == "content-length":
                clen = int(vl)
            elif kl == "transfer-encoding" and "chunked" in vl:
                chunked = True
            elif kl == "connection" and "close" in vl:
                keep_alive = False

        if clen is not None:
            body = await reader.readexactly(clen) if clen > 0 else b""
        elif chunked:
            parts: list[bytes] = []
            while True:
                sz_line = await reader.readuntil(b"\r\n")
                sz = int(sz_line.split(b";", 1)[0].strip(), 16)
                if sz == 0:
                    await reader.readexactly(2)
                    break
                parts.append(await reader.readexactly(sz))
                await reader.readexactly(2)
            body = b"".join(parts)
        else:
            body = await reader.read()
            keep_alive = False
        return status, body, keep_alive

    async def post_chat(self, image_url: str, prompt: str, rid: str | None = None) -> str:
        if API == "generate":
            # SGLang's native endpoint with the chat template already applied (the exact string the OpenAI layer
            # renders for one image + one text part): the tokenizer manager skips request validation and Jinja
            # rendering, ~2 ms of GIL-holding Python per request that delayed every crop's hand-off to the GPU.
            body = json.dumps({
                "text": PROMPT_TEMPLATE.format(prompt),
                "image_data": image_url,
                **({"rid": rid} if rid else {}),
                "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
            }).encode("utf-8")
            path = "/generate"
        else:
            body = json.dumps({
                "model": "glm-ocr",
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            }).encode("utf-8")
            path = "/v1/chat/completions"
        req_Prefix = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: keep-alive\r\n\r\n"
        ).encode("latin-1")
        wire = req_Prefix + body

        async with self._sem:
            for attempt in range(3):
                # A pooled keep-alive socket may have been closed by the server while idle (uvicorn's keep-alive
                # timeout): skip sockets that already saw EOF, and retry on a FRESH socket (never another pooled one).
                conn = None
                if attempt == 0:
                    while self._idle:
                        r_, w_ = self._idle.pop()
                        if r_.at_eof() or w_.is_closing():
                            try:
                                w_.close()
                            except Exception:
                                pass
                            continue
                        conn = (r_, w_)
                        break
                if conn is None:
                    conn = await self._open_conn()
                reader, writer = conn
                try:
                    writer.write(wire)
                    await writer.drain()
                    status, raw, keep_alive = await self._read_http_response(reader)
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    try:
                        writer.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        self.n_fail += 1
                        raise
                    self.n_retry += 1
                    continue
                if keep_alive:
                    self._idle.append((reader, writer))
                else:
                    writer.close()
                if status != 200:
                    raise RuntimeError(f"HTTP {status}: {raw[:200]!r}")
                obj = json.loads(raw)
                if API == "generate":
                    return obj.get("text") or ""
                return obj["choices"][0]["message"]["content"] or ""
        return ""


def cleanup_shm() -> None:
    if not SHM_DIR.is_dir():
        return
    for p in SHM_DIR.iterdir():
        try:
            p.unlink()
        except OSError:
            pass


_PLAIN_PATH = re.compile(r"^[A-Za-z0-9_./\\-]+$")


def image_size_fast(path: str) -> tuple[int, int]:
    """(width, height) from the first bytes of a PNG / GIF / BMP file; (0, 0) when the format needs a real parser."""
    try:
        with open(path, "rb") as f:
            b = f.read(32)
    except OSError:
        return 0, 0
    if len(b) >= 24 and b[:8] == b"\x89PNG\r\n\x1a\n" and b[12:16] == b"IHDR":
        return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    if len(b) >= 10 and b[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(b[6:8], "little"), int.from_bytes(b[8:10], "little")
    if len(b) >= 26 and b[:2] == b"BM":
        return int.from_bytes(b[18:22], "little", signed=True), abs(int.from_bytes(b[22:26], "little", signed=True))
    return 0, 0


def build_image_url(img_path: Path, idx: int, mode: str) -> str:
    if mode == "shm_symlink" and DIRECT_FILE_URL and _PLAIN_PATH.match(str(img_path)):
        # the symlink only gave the server a path it can open: a plain absolute path is that already
        return f"file://{img_path}"
    if mode == "shm_symlink":
        ext = img_path.suffix.lower() or ".png"
        link = SHM_DIR / f"c_{idx}{ext}"
        try:
            link.unlink()
        except FileNotFoundError:
            pass
        os.symlink(str(img_path), str(link))
        return f"file://{link}"
    if mode == "shm_rawpath":
        ext = img_path.suffix.lower() or ".png"
        link = SHM_DIR / f"c_{idx}{ext}"
        try:
            link.unlink()
        except FileNotFoundError:
            pass
        os.symlink(str(img_path), str(link))
        return str(link)
    return "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")


_ROUNDS_DEFAULT = {"table": 17.0, "text": 5.0, "formula": 3.0}


def predict_rounds(task_key: str, w: int, h: int, fsize: int) -> float:
    """Decode rounds of a crop predicted from header dims + PNG byte size (no pixel decode)."""
    if w <= 0 or h <= 0 or fsize <= 0:
        return _ROUNDS_DEFAULT.get(task_key, 5.0)
    lf, lw, lh = math.log(fsize), math.log(w), math.log(h)
    if task_key == "table":
        y = 0.1674 + 0.2405 * lf
    elif task_key == "formula":
        y = -0.5223 + 0.4775 * lw - 0.1927 * lh
    else:
        y = -0.2399 + 0.4344 * lf - 0.3644 * lw
    return max(0.0, math.expm1(min(y, 6.0)))


def predict_rounds_v2(task_key: str, w: int, h: int, fsize: int) -> float:
    """Monotonic 2-feature (log fsize + log area) output-length proxy (5-fold CV on dev; non-negative weights)."""
    if w <= 0 or h <= 0 or fsize <= 0:
        return _ROUNDS_DEFAULT.get(task_key, 5.0) * 40.0
    lf = math.log(fsize)
    la = math.log(w * h)
    if task_key == "table":
        y = 3.28 + 0.20 * lf + 0.08 * la
    elif task_key == "formula":
        return math.exp(min(-1.23 + 0.50 * la, 9.0)) * FORMULA_BOOST
    else:
        return math.exp(min(-2.61 + 0.37 * lf + 0.29 * la, 9.0)) * TEXT_BOOST
    return math.exp(min(y, 9.0))


def order_items(prepared: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    beta = ORDER_BETA
    if ":" in order:
        order, _, b = order.partition(":")
        try:
            beta = float(b)
        except ValueError:
            order = "area"
    if order in ("v2", "v2f"):
        # Monotonic LPT order; "v2f" puts SEND_FIRST small-area high-work crops first so ViT finishes in <8ms at t=0,
        # then orders the rest by descending predicted work / area^beta (default beta=0.05 -> zero big-table tail).
        def lpt_v2(x: dict[str, Any], b_: float) -> float:
            return -x["pred2"] / (max(x["area"], 1) ** b_)

        if order == "v2f" and len(prepared) > SEND_FIRST:
            by_fast = sorted(prepared, key=lambda x: lpt_v2(x, 0.35))
            head = by_fast[:SEND_FIRST]
            head_ids = {id(x) for x in head}
            tail = sorted((x for x in prepared if id(x) not in head_ids), key=lambda x: lpt_v2(x, beta))
            return head + tail
        return sorted(prepared, key=lambda x: lpt_v2(x, beta))
    if order in ("rounds", "rounds_fl", "dens", "dens_fl", "text"):
        if order.startswith("rounds"):
            beta = 0.0

        def lpt(x: dict[str, Any]) -> float:
            return -x["pred"] / (max(x["area"], 1) ** beta)

        if order == "text":
            base = order_items(prepared, "area")
            texts = iter(sorted((x for x in base if x["bucket_rank"] == 1), key=lpt))
            return [next(texts) if x["bucket_rank"] == 1 else x for x in base]
        if order.endswith("_fl"):
            return sorted(prepared, key=lambda x: (x["bucket_rank"] == 2, lpt(x)))
        return sorted(prepared, key=lpt)
    tsign = -1 if os.environ.get("TRACK2_TABLE_ORDER", "asc") == "desc" else 1
    return sorted(prepared, key=lambda x: (x["bucket_rank"], (tsign if x["bucket_rank"] == 0 else -1) * x["area"]))


def prepare_job_items(items: list[dict[str, Any]], folder: Path, mode: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    root = folder.resolve()  # once: the per-item resolve() was a syscall chain per crop
    for idx, item in enumerate(items):
        cid = item["id"]
        task_key = detect_task(item)
        prompt_str = item.get("prompt") or TASK_PROMPTS[task_key]
        img_path = root / item["image"]
        area = 0
        nbytes = 0
        w = h = 0
        if STAGE >= 1:
            try:
                nbytes = os.stat(img_path).st_size
            except OSError:
                nbytes = 0
            w, h = image_size_fast(str(img_path))
            if not w or not h:
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size  # header-only, no pixel decode
                except Exception:
                    w = h = 0
            area = w * h if w and h else nbytes
        image_url = build_image_url(img_path, idx, mode)
        prepared.append({
            "id": cid,
            "task_key": task_key,
            "bucket_rank": TASK_BUCKET_ORDER[task_key],
            "area": area,
            "bytes": nbytes,
            "w": w,
            "h": h,
            "pred": predict_rounds(task_key, w, h, nbytes),
            "pred2": predict_rounds_v2(task_key, w, h, nbytes),
            "prompt": prompt_str,
            "image_url": image_url,
            "img_path": str(img_path),   # team taktak: TRACK2_TABLE_SPLIT reads the pixels of a few tables
        })
    if STAGE >= 1:
        prepared = order_items(prepared, ORDER)
    return prepared


async def infer_crop_async(pool: AsyncKeepAlivePool, prep: dict[str, Any]) -> tuple[str, str]:
    cid = prep["id"]
    task_key = prep["task_key"]
    try:
        raw = await pool.post_chat(prep["image_url"], prep["prompt"], prep.get("rid"))
        out = sanitize_output(raw, task_key)
        prep["_fixed"] = out != raw
        return cid, out
    except Exception as e:
        prep["_failed"] = f"{type(e).__name__}: {str(e)[:120]}"
        log(f"crop {cid} failed: {e}")
        return cid, ""


async def probe_ipc_mode_async(pool: AsyncKeepAlivePool, sample_item: dict[str, Any], folder: Path) -> str:
    if STAGE == 0 or not USE_SHM_IPC:
        return "png_b64"
    img_path = (folder / sample_item["image"]).resolve()
    prompt_str = sample_item.get("prompt") or TASK_PROMPTS[detect_task(sample_item)]
    for mode in ("shm_symlink", "shm_rawpath", "png_b64"):
        try:
            url = build_image_url(img_path, 0, mode)
            _ = await pool.post_chat(url, prompt_str)
            cleanup_shm()
            log(f"IPC transport selected: {mode}")
            return mode
        except Exception as e:
            log(f"IPC probe {mode} rejected ({e}); trying fallback...")
            cleanup_shm()
    return "png_b64"


_AFTER_DONE: list = []   # work deferred until the job's DONE marker exists (shm cleanup, logs)


def run_after_done() -> None:
    while _AFTER_DONE:
        fn = _AFTER_DONE.pop(0)
        try:
            fn()
        except Exception as e:  # logging / cleanup must never break the serving loop
            log(f"post-job step failed: {e}")


async def run_manifest_async(
    folder: Path,
    pool: AsyncKeepAlivePool,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    manifest_path = folder / "manifest.jsonl"
    items = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        items = items[:limit]
    if not items:
        return []

    t0 = time.perf_counter()
    scheduled = prepare_job_items(items, folder, IPC_MODE)
    t_prep = time.perf_counter()

    # Paced hand-off: SGLang's tokenizer manager runs one asyncio loop. Flooding it with every request at once makes it
    # parse all of them before the first image decode completes (~1 ms of Python each), so the GPU idles for ~0.4 s.
    # Small waves keep that loop responsive: the first crops reach the scheduler within a few ms while the rest stream in.
    prof_dir = os.environ.get("T2_PROF")
    t_send: dict[str, float] = {}
    t_done: dict[str, float] = {}
    retry0, fail0 = pool.n_retry, pool.n_fail

    async def one(p: dict[str, Any]) -> tuple[str, str]:
        t_send[p["id"]] = time.perf_counter()
        r = await infer_crop_async(pool, p)
        t_done[p["id"]] = time.perf_counter()
        return r

    tasks = []
    first = SEND_FIRST if STAGE >= 1 else len(scheduled)
    global _JOB_NO
    _JOB_NO += 1
    lane_rank = [0] * max(1, len(LANES))
    # team taktak: TRACK2_TABLE_SPLIT -- the chosen tables are split on worker threads from here on; ranks are spaced by
    # 2 so that both halves keep the original crop's place in the scheduler's queue
    step = 2 if (TABLE_SPLIT > 0 or TEXT_SPLIT > 0) else 1
    split_futs = {}
    if TEXT_SPLIT > 0:   # team taktak: TRACK2_TEXT_SPLIT (patch_minion_text_split.py)
        _texts = sorted((q for q in scheduled if q["task_key"] == "text"), key=lambda q: -q.get("pred2", 0.0))[:TEXT_SPLIT]
        _loop = asyncio.get_running_loop()
        for q in _texts:
            split_futs[id(q)] = _loop.run_in_executor(_split_pool(), _split_text, q)
    if TABLE_SPLIT > 0:
        tabs = sorted((q for q in scheduled if q["task_key"] == "table" and q.get("h", 0) >= TABLE_SPLIT_MINH),
                      key=lambda q: -q.get("pred2", 0.0))[:TABLE_SPLIT]
        _loop = asyncio.get_running_loop()
        for q in tabs:
            split_futs[id(q)] = _loop.run_in_executor(_split_pool(), _split_table, q)
    split_of = {}
    for i, p in enumerate(scheduled):
        # rank in the request id: the scheduler keeps its waiting queue (and the ViT prefetch queue) in this order,
        # whatever order the tokenizer manager finishes the requests in (gravity_ocr/vit_prefetch.py)
        if LANES:   # team taktak: rank within the crop's lane, lane at the end of the rid (LanePools routes on it)
            j = lane_of_task(p["task_key"])
            r = lane_rank[j]
            lane_rank[j] += step
            mk = (lambda rr, j=j: f"t2r{rr:05d}j{_JOB_NO}l{j}")
        else:
            r = i * step
            mk = (lambda rr: f"t2r{rr:05d}j{_JOB_NO}")
        parts = None
        if id(p) in split_futs:
            try:
                parts = await split_futs[id(p)]
            except Exception as e:
                log(f"table split of {p['id']} failed: {e!r}")
        if parts:
            for k, q in enumerate(parts):
                q["rid"] = mk(r + k)
                split_of[q["id"]] = p["id"]
                tasks.append(asyncio.ensure_future(one(q)))
        else:
            p["rid"] = mk(r)
            tasks.append(asyncio.ensure_future(one(p)))
        if SEND_GAP_S > 0 and i + 1 >= first and (i + 1 - first) % SEND_WAVE == 0 and i + 1 < len(scheduled):
            await asyncio.sleep(SEND_GAP_S)
    pairs = await asyncio.gather(*tasks)
    results_map = dict(pairs)
    if split_of:   # team taktak: join the halves back into one table
        halves = {}
        for hid, cid in split_of.items():
            halves.setdefault(cid, []).append(hid)
        for cid, hids in halves.items():
            _parts = [results_map.pop(h, "") for h in sorted(hids)]
            if any("#xs" in h for h in hids):   # text halves: one paragraph
                results_map[cid] = " ".join(x.strip() for x in _parts if x.strip())
            else:
                results_map[cid] = _merge_table_html(_parts)
            ts = [t_send[h] for h in hids if h in t_send]
            td = [t_done[h] for h in hids if h in t_done]
            if ts and td:   # the client log shows the crop as finished when its last half finished
                t_send[cid], t_done[cid] = min(ts), max(td)
    t_infer = time.perf_counter()
    if prof_dir:
        try:
            with open(os.path.join(prof_dir, f"client_{folder.name}.jsonl"), "w", encoding="utf-8") as f:
                for p in scheduled:
                    cid = p["id"]
                    f.write(json.dumps({"id": cid, "task": p["task_key"], "area": p["area"],
                                        "send": round(t_send.get(cid, 0) - t_prep, 4),
                                        "done": round(t_done.get(cid, 0) - t_prep, 4),
                                        "chars": len(results_map.get(cid, ""))}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    retries, fails = pool.n_retry - retry0, pool.n_fail - fail0

    def _finish() -> None:
        cleanup_shm()
        log(
            f"manifest {folder.name}: {len(items)} crops in {t_infer - t0:.3f}s "
            f"(prep={t_prep - t0:.3f}s, infer={t_infer - t_prep:.3f}s, ipc={IPC_MODE}, async_conns={pool.max_conns})"
        )
        if _t2log is not None:
            _log_client_job(folder.name, scheduled, results_map, t0, t_prep, t_infer, t_send, t_done, retries, fails)

    _AFTER_DONE.append(_finish)
    return [(m["id"], results_map.get(m["id"], "")) for m in items]


def _log_client_job(name, scheduled, results_map, t0, t_prep, t_infer, t_send, t_done, retries, fails) -> None:
    """L1 (client) of the multi-layer run log: one summary line per job + per-request records (TRACK2_LOG >= 2)."""
    rows = []
    for rank, p in enumerate(scheduled):
        cid = p["id"]
        rows.append({"id": cid, "rank": rank, "task": p["task_key"], "area": p["area"],
                     "send": round(t_send.get(cid, t_prep), 5), "done": round(t_done.get(cid, t_infer), 5),
                     "chars": len(results_map.get(cid, "")), "fixed": bool(p.get("_fixed")),
                     "failed": p.get("_failed", "")})
        _t2log.record("L1", {"job": name, **rows[-1]})
    by_task = {}
    for t in ("table", "text", "formula"):
        d = [r["done"] - t_prep for r in rows if r["task"] == t]
        if d:
            by_task[t] = {"n": len(d), "done_s_p50_p90_max": _t2log.q(d, 0.5, 0.9, 1.0)}
    last = sorted(rows, key=lambda r: -r["done"])[:5]
    _t2log.emit("L1", {
        "job": name, "crops": len(rows), "t_start": round(t0, 4), "t_sent0": round(t_prep, 4),
        "wall_s": round(t_infer - t0, 4), "prep_s": round(t_prep - t0, 4),
        "send_span_s": round(max(t_send.values()) - t_prep, 4) if t_send else None,
        "empty": sum(1 for r in rows if not r["chars"]), "failed": sum(1 for r in rows if r["failed"]),
        "retries": retries, "socket_failures": fails, "normalized": sum(1 for r in rows if r["fixed"]),
        "by_task": by_task,
        "last": [[r["id"][:40], r["task"], r["rank"], round(r["done"] - t_prep, 3)] for r in last],
    })
    _t2log.flush()


async def warmup_async(pool: AsyncKeepAlivePool) -> None:
    global IPC_MODE
    dev_dir = Path(DEV) if DEV else None
    if not dev_dir or not (dev_dir / "manifest.jsonl").exists():
        await pool.prewarm(512)
        return
    items = [json.loads(l) for l in (dev_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not items:
        await pool.prewarm(512)
        return
    t0 = time.monotonic()
    IPC_MODE = await probe_ipc_mode_async(pool, items[0], dev_dir)
    await pool.prewarm(512)
    # Warm ALL dev crops (untimed!) so every Triton/FlashInfer/ViT shape is JIT-compiled before READY
    await run_manifest_async(dev_dir, pool, limit=None)
    run_after_done()
    log(f"untimed full warmup ({len(items)} dev crops) finished in {time.monotonic() - t0:.1f}s")


async def async_main() -> None:
    if LANES:   # team taktak
        pool = LanePools([AsyncKeepAlivePool("127.0.0.1", p, CLIENT_CONC) for p in lane_ports()])
    else:
        pool = AsyncKeepAlivePool("127.0.0.1", PORT, CLIENT_CONC)
    try:
        await warmup_async(pool)
        try:
            from gravity_ocr import gcguard as _gcguard
            _gcguard.freeze("client")
        except Exception as e:
            log(f"gc guard not applied: {e}")
        Path(WORK, "READY").touch()
        log("READY signaled; polling jobs...")

        jobs_root = Path(WORK) / "jobs"
        seen: set[str] = set()
        stop_file = Path(WORK) / "STOP"
        while not stop_file.exists():
            if jobs_root.is_dir():
                entries = sorted(
                    os.listdir(jobs_root),
                    key=lambda s: int(s.split("-")[0]) if s.split("-")[0].isdigit() else 0,
                )
                for name in entries:
                    if name in seen:
                        continue
                    job_dir = jobs_root / name
                    if not (job_dir / "manifest.jsonl").exists() or (job_dir / "DONE").exists():
                        continue
                    results = await run_manifest_async(job_dir, pool)
                    pred_path = job_dir / "predictions.jsonl"
                    out_lines = [json.dumps({"id": cid, "text": text}, ensure_ascii=False) for cid, text in results]
                    with open(pred_path, "w", encoding="utf-8") as f:
                        if out_lines:
                            f.write("\n".join(out_lines) + "\n")
                        if FSYNC_PREDICTIONS:
                            f.flush()
                            os.fsync(f.fileno())
                    # closed file: its content is visible to every reader; DONE is what stops the harness clock
                    (job_dir / "DONE").touch()
                    seen.add(name)
                    run_after_done()
            await asyncio.sleep(0.0005)
    finally:
        await pool.close_all()


def _merge_weights_delta() -> None:
    """team taktak: weights_delta/model_delta.safetensors holds only the tensors that differ from the stock GravityOCR-KO
    weights (our draft-tuned language model, 1.35 GB), to fit the submission storage. The full model folder is assembled here,
    before the server starts (untimed load), from the stock weights of the image."""
    import shutil
    import tempfile
    delta = HERE / "weights_delta" / "model_delta.safetensors"
    if not delta.is_file() or (HERE / "weights").exists():
        return
    t0 = time.time()
    stock = Path(os.environ.get("GRAVITYOCR_WEIGHTS", "/opt/gravityocr/weights"))
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors = {}
    with safe_open(str(stock / "model.safetensors"), "pt") as s:
        meta = s.metadata() or {"format": "pt"}
        for k in s.keys():
            tensors[k] = s.get_tensor(k)
    n = 0
    with safe_open(str(delta), "pt") as d:
        for k in d.keys():
            t = d.get_tensor(k)
            if k not in tensors or tensors[k].shape != t.shape or tensors[k].dtype != t.dtype:
                raise RuntimeError(f"weights_delta: {k} does not match the stock weights")
            tensors[k] = t
            n += 1
    for base in (WORK, tempfile.gettempdir()):
        out = Path(base) / "weights_merged"
        try:
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            for f in stock.iterdir():
                if f.is_file() and f.name != "model.safetensors":
                    shutil.copy2(f, out / f.name)
            save_file(tensors, str(out / "model.safetensors"), metadata=meta)
            break
        except OSError as e:
            log(f"weights: could not write {out} ({e!r}); trying the next place")
    else:
        raise RuntimeError("weights_delta: nowhere to write the merged weights")
    os.environ["GRAVITYOCR_WEIGHTS"] = str(out)
    log(f"weights: stock + {n} tensors of weights_delta -> {out} ({time.time() - t0:.1f} s)")


def main() -> None:
    _merge_weights_delta()
    if LANES:   # team taktak: one server per lane, all on this GPU
        if LANE_MPS:
            start_mps()
        servers = []
        for j in range(len(LANES)):
            servers.append(start_server(j))
            if LANE_SEQUENTIAL and j < len(LANES) - 1:   # this lane's KV pool is sized before the next one starts
                wait_ready(servers[-1], url=f"http://127.0.0.1:{lane_ports()[j]}")
                log(f"lane {j} ready; starting lane {j + 1}")
    else:
        servers = [start_server()]

    def _kill_all() -> None:
        for server in servers:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _stop_server(*_: Any) -> None:
        _kill_all()
        stop_mps()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop_server)
    signal.signal(signal.SIGINT, _stop_server)

    try:
        for server, port in zip(servers, lane_ports()):
            wait_ready(server, url=f"http://127.0.0.1:{port}")
        asyncio.run(async_main())
    finally:
        cleanup_shm()
        _kill_all()
        stop_mps()


if __name__ == "__main__":
    main()
