"""Track 2 multi-layer run log (client -> tokenizer manager -> scheduler -> ViT prefetch -> GPU).

Levels (TRACK2_LOG, default 1):
  0  off
  1  one JSON summary line per layer per job on stdout ("[t2log L<n>] {...}"): the official harness keeps stdout in
     output/team_process_*.log, so `python3 tools/t2log_report.py <that log>` works on local_run.py / eval outputs.
  2  + per-item JSONL files in TRACK2_LOG_DIR (default $HOME/t2logs): L1 requests, L2 images, L3 scheduler iterations,
     L4 ViT batches (GPU time from CUDA events, no synchronization).
  3  + L5: a CUDA-only torch.profiler trace of each job in TRACK2_LOG_DIR (heavy; diagnostic runs only).

Layers:
  L1 client (serve.py)        per request: rank, task, image area, send / done time, output chars, retries
  L2 tokenizer manager        per image: decode ms, downscale factor, ViT grid, queue latency (processors/glm4v.py)
  L3 scheduler                per iteration: mode, batch size, run / result times, queue sizes, tokens committed
  L4 ViT prefetch             per packed batch: images, patches, host launch ms, GPU ms; on-demand encodes, waits
  L5 GPU                      CUDA kernel trace (level 3)

All timestamps are time.perf_counter() (CLOCK_MONOTONIC: shared by every process of the container), so the report
tool can put the layers on one time axis. Logging never synchronizes the GPU at levels 1-2.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

LEVEL = int(os.environ.get("TRACK2_LOG", "1") or "0")
_DIR = os.environ.get("TRACK2_LOG_DIR") or os.path.join(os.environ.get("HOME", "/tmp"), "t2logs")
_LOCK = threading.Lock()
_FILES: Dict[str, Any] = {}


def now() -> float:
    return time.perf_counter()


def emit(layer: str, summary: Dict[str, Any]) -> None:
    """One summary line on stdout (level >= 1)."""
    if LEVEL >= 1:
        print(f"[t2log {layer}] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


def record(layer: str, rec: Dict[str, Any]) -> None:
    """One per-item JSON line in <dir>/<layer>_<pid>.jsonl (level >= 2)."""
    if LEVEL < 2:
        return
    line = json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK:
        f = _FILES.get(layer)
        if f is None:
            os.makedirs(_DIR, exist_ok=True)
            f = _FILES[layer] = open(os.path.join(_DIR, f"{layer}_{os.getpid()}.jsonl"), "a", encoding="utf-8")
        f.write(line)


def flush() -> None:
    with _LOCK:
        for f in _FILES.values():
            try:
                f.flush()
            except Exception:
                pass


def q(vals: List[float], *ps: float) -> List[Optional[float]]:
    if not vals:
        return [None for _ in ps]
    s = sorted(vals)
    return [round(s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))], 4) for p in ps]


# ---------------------------------------------------------------------------------------------- L2 tokenizer manager
class _TmLayer:
    """Burst accounting in the tokenizer-manager process; a daemon thread emits the summary 0.5 s after the last image."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []
        self.last = 0.0
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def add(self, rec: Dict[str, Any]) -> None:
        record("L2", rec)
        with self.lock:
            self.items.append(rec)
            self.last = now()
            if self.thread is None:
                self.thread = threading.Thread(target=self._watch, daemon=True)
                self.thread.start()

    def _watch(self) -> None:
        while True:
            time.sleep(0.25)
            with self.lock:
                if not self.items or now() - self.last < 0.5:
                    continue
                items, self.items = self.items, []
            lat = [r["out"] - r["in"] for r in items]
            dec = [r["decode_ms"] for r in items]
            emit("L2", {
                "images": len(items),
                "first_in": round(min(r["in"] for r in items), 4),
                "last_out": round(max(r["out"] for r in items), 4),
                "latency_ms_p50_p90_max": [None if v is None else round(1e3 * v, 1) for v in q(lat, 0.5, 0.9, 1.0)],
                "decode_ms_p50_p90_max": q(dec, 0.5, 0.9, 1.0),
                "downscaled": sum(1 for r in items if r.get("scale", 1.0) < 1.0),
                "patches": sum(r.get("patches", 0) for r in items),
                "by_task": {t: sum(1 for r in items if r.get("task") == t) for t in ("text", "table", "formula", "")},
            })
            flush()


TM = _TmLayer()


# ---------------------------------------------------------------------------------------------- L3 / L4 scheduler side
class _SchedLayer:
    """Per-job accounting in the scheduler process (job = from the first batch after idle to the next idle)."""

    def __init__(self) -> None:
        self.reset()
        self.vit_batches: List[Dict[str, Any]] = []
        self.vit_pending_events: List[Any] = []
        self.ondemand = 0
        self.wait_first = []
        self.trace = None

    def reset(self) -> None:
        self.active = False
        self.its: List[Dict[str, Any]] = []
        self.recv = 0
        self.t_first = None
        self.t_last = None
        self.commit_hist: Dict[int, int] = {}  # level >= 2: tokens committed per request-round -> count

    # L4 hooks (vit_prefetch.py)
    def vit_batch(self, n_img: int, patches: int, launch_ms: float, ev0, ev1, ondemand: bool) -> None:
        rec = {"t": round(now(), 4), "images": n_img, "patches": patches, "launch_ms": round(launch_ms, 2),
               "ondemand": bool(ondemand)}
        self.vit_batches.append(rec)
        self.vit_pending_events.append((rec, ev0, ev1))
        if ondemand:
            self.ondemand += n_img

    def vit_wait(self, ms: float) -> None:
        self.wait_first.append(ms)

    # L3 hooks (scheduler wrappers)
    def iteration(self, st: Dict[str, Any]) -> None:
        if not self.active:
            self.active = True
            self.t_first = st["t0"]
            if LEVEL >= 3:
                self._trace_start()
        self.t_last = st["t2"]
        self.its.append(st)
        record("L3", st)

    def idle(self) -> None:
        if not self.active:
            return
        self._summary()
        self.reset()

    def _ahead_delta(self) -> Dict[str, Any]:
        """Self-spec draft-ahead counters of this job (the worker's _ahead_stats): launched / used / discarded, and why a
        round ran without a pre-launched draft (skip_sched: a prefill admission was due; no_precheck: a request could finish)."""
        try:
            dw = getattr(getattr(self, "sched", None), "draft_worker", None)
            cur = dict(getattr(dw, "_ahead_stats", None) or {})
            if not cur:
                return {}
            prev = getattr(self, "_ahead_prev", None) or {}
            self._ahead_prev = cur
            return {"draft_ahead": {k: int(v) - int(prev.get(k, 0)) for k, v in cur.items() if k != "freed_slots"}}
        except Exception:
            return {}

    def _trace_start(self) -> None:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
            self.trace = profile(activities=[ProfilerActivity.CUDA], record_shapes=False)
            self.trace.__enter__()
        except Exception as e:  # pragma: no cover
            print(f"[t2log L5] trace start failed: {e}", flush=True)
            self.trace = None

    def _trace_stop(self) -> Optional[Dict[str, Any]]:
        if self.trace is None:
            return None
        import torch
        torch.cuda.synchronize()
        p, self.trace = self.trace, None
        p.__exit__(None, None, None)
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"L5_trace_{os.getpid()}_{int(time.time())}.json")
        p.export_chrome_trace(path)
        return {"trace": path}

    def _summary(self) -> None:
        its = self.its
        ext = [r for r in its if r["mode"] == "EXTEND"]
        dec = [r for r in its if r["mode"] != "EXTEND"]
        gaps = [b["t0"] - a["t2"] for a, b in zip(its, its[1:])]

        def tot(rs, a, b):
            return round(sum(r[b] - r[a] for r in rs), 4)

        commit = sum(r.get("commit", 0) for r in dec)
        req_rounds = sum(r["bs"] for r in dec)
        alloc = {}
        try:  # CUDA caching-allocator stalls (free-all + retry, fresh cudaMalloc) during the job: deltas since last job
            import torch
            ms = torch.cuda.memory_stats()
            cur = {k: int(ms.get(k, 0)) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")}
            prev = getattr(self, "_alloc_prev", None) or {k: 0 for k in cur}
            alloc = {k.replace("num_", ""): cur[k] - prev[k] for k in cur}
            self._alloc_prev = cur
        except Exception:
            pass
        emit("L3", {
            "t_first": round(self.t_first, 4), "t_last": round(self.t_last, 4),
            "span_s": round(self.t_last - self.t_first, 4), "iterations": len(its),
            "prefill": {"n": len(ext), "reqs": sum(r["bs"] for r in ext), "tokens": sum(r.get("ext_tok", 0) for r in ext),
                        "run_s": tot(ext, "t0", "t1"), "result_s": tot(ext, "t1", "t2")},
            "decode": {"n": len(dec), "request_rounds": req_rounds, "tokens": commit,
                       "tokens_per_request_round": round(commit / max(req_rounds, 1), 2),
                       "mean_bs": round(req_rounds / max(len(dec), 1), 1),
                       "run_s": tot(dec, "t0", "t1"), "result_s": tot(dec, "t1", "t2")},
            "gaps_s": round(sum(max(0.0, g) for g in gaps), 4),
            "gap_ms_p50_p90_max": [None if v is None else round(1e3 * v, 1) for v in q(gaps, 0.5, 0.9, 1.0)],
            "finished": sum(r.get("fin", 0) for r in dec) + sum(r.get("fin", 0) for r in ext),
            **({"commit_hist": dict(sorted(self.commit_hist.items()))} if self.commit_hist else {}),
            **({"cuda_alloc": alloc} if alloc else {}),
            **self._ahead_delta(),
        })
        # L4: resolve the batch GPU times (events of finished batches; no synchronization)
        if self.vit_batches:
            gpu = []
            for rec, e0, e1 in self.vit_pending_events:
                try:
                    if e1.query():
                        rec["gpu_ms"] = round(e0.elapsed_time(e1), 2)
                        gpu.append(rec["gpu_ms"])
                except Exception:
                    pass
                record("L4", rec)
            emit("L4", {
                "batches": len(self.vit_batches), "images": sum(b["images"] for b in self.vit_batches),
                "patches": sum(b["patches"] for b in self.vit_batches),
                "host_launch_ms": round(sum(b["launch_ms"] for b in self.vit_batches), 1),
                "gpu_ms_sum": round(sum(gpu), 1), "gpu_ms_p50_max": q(gpu, 0.5, 1.0),
                "first_launch": self.vit_batches[0]["t"], "last_launch": self.vit_batches[-1]["t"],
                "ondemand_images": self.ondemand, "wait_first": len(self.wait_first),
                "wait_first_ms": round(sum(self.wait_first), 1),
            })
            self.vit_batches, self.vit_pending_events, self.ondemand, self.wait_first = [], [], 0, []
        t5 = self._trace_stop() if LEVEL >= 3 else None
        if t5:
            emit("L5", t5)
        flush()


SCHED = _SchedLayer()
_INSTALLED = False


def install_scheduler() -> None:
    """Wrap SGLang's Scheduler (run_batch / process_batch_result / idle) for L3; called from models/glm_ocr.py."""
    global _INSTALLED
    if _INSTALLED or LEVEL < 1:
        return
    _INSTALLED = True
    from sglang.srt.managers import scheduler as S

    Sch = S.Scheduler
    orig_run, orig_proc, orig_idle = Sch.run_batch, Sch.process_batch_result, Sch.self_check_during_idle

    def run_batch(self, batch, *a, **k):
        st = {"t0": now(), "mode": batch.forward_mode.name, "bs": batch.batch_size(), "wq": len(self.waiting_queue),
              "run": len(self.running_batch.reqs) if self.running_batch is not None else 0}
        SCHED.sched = self
        try:
            if batch.forward_mode.is_extend():
                st["ext_tok"] = int(sum(getattr(r, "extend_input_len", 0) or 0 for r in batch.reqs))
            lens = [len(r.output_ids) for r in batch.reqs]
            st["_out0"] = int(sum(lens))
            if LEVEL >= 2 and not batch.forward_mode.is_extend():
                st["_lens0"] = lens  # per-request tokens committed this round (acceptance histogram)
        except Exception:
            pass
        res = orig_run(self, batch, *a, **k)
        st["t1"] = now()
        self._t2log_st = st
        return res

    def process_batch_result(self, batch, result):
        r = orig_proc(self, batch, result)
        st = getattr(self, "_t2log_st", None)
        if st is not None:
            self._t2log_st = None
            st["t2"] = now()
            try:
                st["commit"] = int(sum(len(x.output_ids) for x in batch.reqs)) - st.pop("_out0", 0)
                st["fin"] = int(sum(1 for x in batch.reqs if x.finished()))
                lens0 = st.pop("_lens0", None)
                if lens0 is not None and len(lens0) == len(batch.reqs):
                    h = SCHED.commit_hist
                    for x, n0 in zip(batch.reqs, lens0):
                        c = len(x.output_ids) - n0
                        h[c] = h.get(c, 0) + 1
            except Exception:
                st.pop("_out0", None)
                st.pop("_lens0", None)
            for k in ("t0", "t1", "t2"):
                st[k] = round(st[k], 5)
            SCHED.iteration(st)
        return r

    def self_check_during_idle(self):
        SCHED.idle()
        return orig_idle(self)

    Sch.run_batch, Sch.process_batch_result, Sch.self_check_during_idle = run_batch, process_batch_result, self_check_during_idle
    print(f"[t2log] level {LEVEL}: scheduler layer installed (pid {os.getpid()}, dir {_DIR})", flush=True)
