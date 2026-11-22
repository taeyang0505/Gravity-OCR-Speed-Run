"""LANES for the Track 2 winning team's launcher (team taktak, 2026-11-21): one SGLang server per lane on the same L40S
(CUDA MPS), each lane with its own draft length. Default OFF (TRACK2_LANES unset: one server, unchanged behaviour).

Why: on M3 (winner v17 + our weights + glyph 11) the job ends when the longest requests end. The longest table needs
46 decode rounds at DRAFT=25 (it accepts the whole window 77% of the time, so it is window-limited), and every round of
a full batch costs ~40-45 ms. Text and formula crops accept ~8-13 tokens per round, so their 25-token windows are mostly
wasted compute. One draft length cannot serve both; a server per lane can (tables: long window, few requests;
text + formula: short window, many requests). Same idea as our own LANES launcher (taeyang/track2/team/serve.py).

  TRACK2_LANES="table;text,formula"       lanes in order; a task not listed goes to the last lane
  TRACK2_LANE_DRAFT="33,17"               per-lane --speculative-num-draft-tokens   (comma lists in lane order;
  TRACK2_LANE_MAXREQ="64,160"             per-lane max running requests            an empty entry keeps the default)
  TRACK2_LANE_GRAPHBS="64,160"            per-lane CUDA-graph max batch size
  TRACK2_LANE_MEMFRAC="0.38,0.40"         per-lane --mem-fraction-static
  TRACK2_LANE_MM_WORKERS="2,4"            per-lane multimodal processor workers
  TRACK2_LANE_VIT_PF_PATCHES="16384,16384"  per-lane ViT prefetch batch size
  TRACK2_LANE_MPS_PRIO="1,0"              per-lane CUDA_MPS_CLIENT_PRIORITY (0 normal, 1 below normal)
  TRACK2_LANE_MPS_SM_PCT="40,100"         per-lane CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
  TRACK2_LANE_PORT_STEP=100               lane i listens on PORT + i * step
  TRACK2_LANE_MPS=1                       start a CUDA MPS daemon (pipe / log dirs under $HOME) so the servers run
                                          kernels concurrently; without it they time-share the GPU

Each lane's server gets its own shm ring (TRACK2_SHM_RING_PATH, read by the patched glm4v.py / glm_ocr.py), the client
sends every crop to its lane's server, and the global LPT order is kept within each lane (rank = position in the lane).

Usage: python patch_minion_lanes.py <team dir>   (patches serve.py and the two shm-ring users in place)
"""
import os
import sys

team = sys.argv[1]
if "lane_of_task" in open(os.path.join(team, "serve.py")).read():
    print("already patched", team)
    sys.exit(0)


def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:80])
        s = s.replace(old, new, 1)
    open(path, "w").write(s)
    print("patched", path)


# ---------------------------------------------------------------- the shm ring path becomes per server (env)
sg = os.path.join(team, "gravityocr/sglang/python/sglang/srt")
patch(os.path.join(sg, "multimodal/processors/glm4v.py"),
      [('SHM_RING_PATH = "/dev/shm/track2_raw_chw.bin"',
        'SHM_RING_PATH = os.environ.get("TRACK2_SHM_RING_PATH", "/dev/shm/track2_raw_chw.bin")  # team taktak: per lane')])
patch(os.path.join(sg, "models/glm_ocr.py"),
      [('_T2_SHM_RING_PATH = "/dev/shm/track2_raw_chw.bin"',
        '_T2_SHM_RING_PATH = __import__("os").environ.get("TRACK2_SHM_RING_PATH", "/dev/shm/track2_raw_chw.bin")  # team taktak')])

# ---------------------------------------------------------------- serve.py
LANES_CFG = '''_JOB_NO = 0

# team taktak (2026-11-21): LANES, one SGLang server per lane (see taeyang/track2/minion_port/patch_minion_lanes.py).
# Default off: TRACK2_LANES unset keeps the single server.
LANES = [[t.strip() for t in part.split(",") if t.strip()]
         for part in os.environ.get("TRACK2_LANES", "").split(";") if part.strip()]
LANE_PORT_STEP = int(os.environ.get("TRACK2_LANE_PORT_STEP", "100"))
LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"
_LANE_KEYS = {"DRAFT": "DRAFT", "MAXREQ": "MAXREQ", "GRAPHBS": "GRAPHBS", "MEMFRAC": "MEMFRAC",
              "MM_WORKERS": "TRACK2_MM_WORKERS", "VIT_PF_PATCHES": "TRACK2_VIT_PF_PATCHES",
              "MPS_PRIO": "CUDA_MPS_CLIENT_PRIORITY", "MPS_SM_PCT": "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}
_MPS_ENV: dict = {}
_SGLANG_PY = None


def lane_of_task(task: str) -> int:
    for j, tasks in enumerate(LANES):
        if task in tasks:
            return j
    return len(LANES) - 1


def lane_env(j: int) -> dict:
    out = {}
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
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\\n", env=dict(os.environ, **_MPS_ENV),
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
'''

START_OLD = '''def start_server() -> subprocess.Popen:
    env = dict(os.environ)'''
START_NEW = '''def start_server(lane: int | None = None) -> subprocess.Popen:
    global _SGLANG_PY
    env = dict(os.environ)'''

START_OLD2 = '''        "TRACK2_SLICE_DIR": os.environ.get("TRACK2_SLICE_DIR", str(HERE / "vocab_slices")),
    })
'''
START_NEW2 = '''        "TRACK2_SLICE_DIR": os.environ.get("TRACK2_SLICE_DIR", str(HERE / "vocab_slices")),
    })
    if lane is not None:   # team taktak: LANES
        env["PORT"] = str(PORT + lane * LANE_PORT_STEP)
        env.update(lane_env(lane))
        env["TRACK2_SHM_RING_PATH"] = f"/dev/shm/track2_raw_chw_l{lane}.bin"
        env.update(_MPS_ENV)
'''

BUILD_OLD = '''    if own("gravityocr/sglang", None):
        sglang_py = build_sglang()
        py_paths.append(sglang_py)'''
BUILD_NEW = '''    if own("gravityocr/sglang", None):
        if _SGLANG_PY is None:          # team taktak: build once (a second build would wipe the first server's tree)
            _SGLANG_PY = build_sglang()
        sglang_py = _SGLANG_PY
        py_paths.append(sglang_py)'''

LOG_OLD = '''        f"starting {script} MODE={env['MODE']} FAST={env['FAST']} MAXREQ={env['MAXREQ']} "'''
LOG_NEW = '''        f"starting {script} lane={lane} PORT={env['PORT']} DRAFT={env.get('DRAFT', '-')} MODE={env['MODE']} FAST={env['FAST']} MAXREQ={env['MAXREQ']} "'''

WAIT_OLD = '''def wait_ready(server: subprocess.Popen, timeout: int = 270) -> None:
    deadline = time.monotonic() + timeout'''
WAIT_NEW = '''def wait_ready(server: subprocess.Popen, timeout: int = 270, url: str = URL) -> None:
    deadline = time.monotonic() + timeout'''
WAIT_OLD2 = '''            with urllib.request.urlopen(f"{URL}/v1/models", timeout=3) as r:'''
WAIT_NEW2 = '''            with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:'''

RID_OLD = '''    for i, p in enumerate(scheduled):
        # rank in the request id: the scheduler keeps its waiting queue (and the ViT prefetch queue) in this order,
        # whatever order the tokenizer manager finishes the requests in (gravity_ocr/vit_prefetch.py)
        p["rid"] = f"t2r{i:05d}j{_JOB_NO}"'''
RID_NEW = '''    lane_rank = [0] * max(1, len(LANES))
    for i, p in enumerate(scheduled):
        # rank in the request id: the scheduler keeps its waiting queue (and the ViT prefetch queue) in this order,
        # whatever order the tokenizer manager finishes the requests in (gravity_ocr/vit_prefetch.py)
        if LANES:   # team taktak: rank within the crop's lane, lane at the end of the rid (LanePools routes on it)
            j = lane_of_task(p["task_key"])
            p["rid"] = f"t2r{lane_rank[j]:05d}j{_JOB_NO}l{j}"
            lane_rank[j] += 1
        else:
            p["rid"] = f"t2r{i:05d}j{_JOB_NO}"'''

POOL_OLD = '''    pool = AsyncKeepAlivePool("127.0.0.1", PORT, CLIENT_CONC)
    try:
        await warmup_async(pool)'''
POOL_NEW = '''    if LANES:   # team taktak
        pool = LanePools([AsyncKeepAlivePool("127.0.0.1", p, CLIENT_CONC) for p in lane_ports()])
    else:
        pool = AsyncKeepAlivePool("127.0.0.1", PORT, CLIENT_CONC)
    try:
        await warmup_async(pool)'''

MAIN_OLD = '''def main() -> None:
    _merge_weights_delta()
    server = start_server()

    def _stop_server(*_: Any) -> None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop_server)
    signal.signal(signal.SIGINT, _stop_server)

    try:
        wait_ready(server)
        asyncio.run(async_main())
    finally:
        cleanup_shm()
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
'''
MAIN_NEW = '''def main() -> None:
    _merge_weights_delta()
    if LANES:   # team taktak: one server per lane, all on this GPU
        if LANE_MPS:
            start_mps()
        servers = [start_server(j) for j in range(len(LANES))]
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
'''

patch(os.path.join(team, "serve.py"), [
    ("_JOB_NO = 0\n", LANES_CFG),
    (START_OLD, START_NEW), (START_OLD2, START_NEW2), (BUILD_OLD, BUILD_NEW), (LOG_OLD, LOG_NEW),
    (WAIT_OLD, WAIT_NEW), (WAIT_OLD2, WAIT_NEW2), (RID_OLD, RID_NEW), (POOL_OLD, POOL_NEW), (MAIN_OLD, MAIN_NEW),
])
