"""Track 2: GLM-OCR ViT prefetch on a low-priority CUDA stream (SGLang scheduler process only).

Why: SGLang runs the vision tower inside every prefill forward, once per request
(mm_utils._get_chunked_embedding_by_item): each small crop pays its own graph replay / eager pass, and every prefill
stalls the decode batch for the whole ViT time (~45% of a run's GPU time on the dev set).

What:
  * The scheduler calls tick() once per event-loop iteration. Every newly queued request's image descriptors (the
    shm-ring descriptors written by the tokenizer manager, see processors/glm4v.py) are submitted in waiting-queue
    order and encoded in large packed batches (<= TRACK2_VIT_PF_PATCHES patches) on a dedicated LOW-priority stream.
    The schedule stream is created HIGH-priority (scheduler.py), so decode kernels win the SMs and the ViT fills the
    gaps between them (CPU bookkeeping between rounds, attention phases, small kernels).
  * get_image_feature() (models/glm_ocr.py) asks get_embeddings(items): the current stream waits on the batch's event
    and gets the stored embedding. An image that was never submitted is encoded on demand, on the same ViT stream.
  * Every ViT launch (graph replays included) runs on that ONE stream, from the scheduler thread, in submission order,
    so the graph runner's static input buffers are never shared across streams.
Embeddings are identical in math to the per-request path (packed FA2 varlen vs per-image SDPA: bf16 rounding only).

Env: TRACK2_VIT_PREFETCH (1 = on), TRACK2_VIT_PF_PATCHES (patches per packed batch), TRACK2_VIT_PF_IMAGES.
"""
from __future__ import annotations

import collections
import os
import time
from typing import Callable, List, Optional

import torch

ENABLED = os.environ.get("TRACK2_VIT_PREFETCH", "1") == "1"
_TRACE = os.environ.get("T2_PROF", "")  # diagnostic: per-tick timing lines (profiling runs only)
BATCH_PATCHES = int(os.environ.get("TRACK2_VIT_PF_PATCHES", "16384"))
BATCH_IMAGES = int(os.environ.get("TRACK2_VIT_PF_IMAGES", "64"))
# Batches queued on the ViT stream at once. Queuing everything fills the driver's launch queue: cudaLaunchKernel then
# blocks the scheduler thread (and every decode launch) until the GPU drains it.
MAX_INFLIGHT = int(os.environ.get("TRACK2_VIT_PF_INFLIGHT", "2"))
RANK_ORDER = os.environ.get("TRACK2_RANK_ORDER", "1") == "1"
# Pacing (0 = off): while the LM is decoding and at least this many waiting requests already have their image encoded
# (or in flight), no new ViT batch is launched. The ViT then stays just ahead of the LM's admissions instead of racing
# to the end of the queue, so its kernels are left to fill the LM's host-bound gaps late in the job as well.
READY_TARGET = int(os.environ.get("TRACK2_VIT_PF_READY_TARGET", "0"))
SCHED_PRIORITY = int(os.environ.get("TRACK2_SCHED_STREAM_PRIORITY", "-10"))
# Hide ViT host_launch (~9-11 ms/batch) behind DECODE's event.synchronize() wait on the same scheduler thread
HIDE_LAUNCH = os.environ.get("TRACK2_VIT_HIDE_LAUNCH", "1") == "1"
# Hide ZMQ recv_pyobj() unpickling (+ optional process_input_requests) behind DECODE's event.synchronize() wait:
# 0 = off, 1 = recv_pyobj into sched._t2_recv_buf while not event.query(), 2 = also process TokenizedGenerateReqInput + submit ViT
RECV_HIDE = int(os.environ.get("TRACK2_RECV_HIDE", "2"))
# team taktak (2026-11-21), SS_VIT_DROP_ASYNC: blank 2x2-group ViT skip with the mask staged ahead (see
# taeyang/track2/minion_port/patch_minion_vit_drop_async.py). Default off.
DROP = os.environ.get("SS_VIT_DROP", "0") == "1"
DROP_THR = float(os.environ.get("SS_VIT_DROP_THR", "0.03") or 0.03)
STAGE_DEPTH = max(1, int(os.environ.get("SS_VIT_DROP_STAGE", "2") or 2))
DROP_TABLE = os.environ.get("SS_VIT_DROP_TABLE", "1") == "1"   # 0: table crops are never skipped
_MASK_CAP = 1 << 16

_PF: Optional["VitPrefetcher"] = None
_SCHED = None

try:  # multi-layer run log, ViT layer (L4)
    from gravity_ocr import t2log as _LOG
    if _LOG.LEVEL < 1:
        _LOG = None
except Exception:
    _LOG = None


def req_rank(req) -> int:
    """the client's scheduling rank, carried in the request id ("t2r<rank:05d>j<job>"); unranked requests go last."""
    rid = getattr(req, "rid", None)
    if isinstance(rid, str) and rid.startswith("t2r"):
        try:
            return int(rid[3:8])
        except ValueError:
            pass
    return 1 << 30


def is_desc(item) -> bool:
    f = getattr(item, "feature", None)
    return (
        isinstance(f, torch.Tensor)
        and f.dtype == torch.int64
        and f.dim() == 2
        and f.shape[0] == 1
        and f.shape[1] == 7
        and int(f[0, 0]) == -99
    )


def desc(item):
    _, off, nbytes, h, w, gh, gw = (int(v) for v in item.feature[0].tolist())
    return off, nbytes, h, w, gh, gw


class VitPrefetcher:
    def __init__(self, model, preprocess: Callable, log: Callable[[str], None] = print):
        """model: GlmOcrForConditionalGeneration (uses model.visual); preprocess(items) -> (pixel_values, grid_thw)
        enqueues the H2D copy / resize / patchify of the items on the CURRENT stream."""
        self.model = model
        self.preprocess = preprocess
        self.log = log
        # default 0 = the lowest priority (decode wins); TRACK2_VIT_STREAM_PRIORITY < 0 raises it
        self.stream = torch.cuda.Stream(device=model.visual.device,
                                        priority=int(os.environ.get("TRACK2_VIT_STREAM_PRIORITY", "0")))
        self.pending: collections.deque = collections.deque()
        self.inflight: collections.deque = collections.deque()  # events of launched, possibly unfinished batches
        self._ondemand_now = False
        self.n_batches = 0
        self.n_items = 0
        self.n_ondemand = 0
        self.n_patches = 0
        self.t_launch = 0.0
        self.n_paced = 0  # ticks on which the ready-pool target held a launch back
        self.gc_done = False
        # SS_VIT_DROP_ASYNC (team taktak): prep stream + staged batches + page-locked mask ring
        self.staged: collections.deque = collections.deque()
        self.n_staged_nohint = 0
        if DROP:
            self.pstream = torch.cuda.Stream(device=model.visual.device,
                                             priority=int(os.environ.get("TRACK2_VIT_STREAM_PRIORITY", "0")))
            self._mask_ring = [torch.empty((_MASK_CAP,), dtype=torch.uint8, pin_memory=True) for _ in range(STAGE_DEPTH + 3)]
            self._mask_idx = 0

    # ------------------------------------------------------------------ submission (scheduler thread)
    def submit_reqs(self, reqs) -> None:
        added = False
        for req in reqs:
            if getattr(req, "_t2_pf", False):
                continue
            req._t2_pf = True
            mm = getattr(req, "multimodal_inputs", None)
            items = getattr(mm, "mm_items", None) if mm is not None else None
            if not items:
                continue
            rank = req_rank(req)
            for it in items:
                if getattr(it, "_t2_state", 0) == 0 and is_desc(it):
                    it._t2_state = 1  # queued
                    it._t2_rank = rank
                    it._t2_table = "Table Recognition" in (getattr(req, "origin_input_text", "") or "")
                    self.pending.append(it)
                    added = True
        if added and len(self.pending) > 1:
            # encode in the client's rank order, not in the order the tokenizer manager happened to finish requests
            self.pending = collections.deque(sorted(self.pending, key=lambda it: it._t2_rank))

    def reap(self) -> None:
        while self.inflight and self.inflight[0].query():
            self.inflight.popleft()

    def pump(self, max_batches: int = 0) -> int:
        """Keep up to MAX_INFLIGHT packed batches queued on the ViT stream (non-blocking: completion is polled)."""
        if DROP:
            return self._pump_staged(max_batches)
        self.reap()
        launched_n = 0
        while self.pending and len(self.inflight) < MAX_INFLIGHT:
            if max_batches > 0 and launched_n >= max_batches:
                break
            batch, patches = [], 0
            while self.pending and len(batch) < BATCH_IMAGES:
                it = self.pending[0]
                gh, gw = desc(it)[4:6]
                m = gh * gw
                if batch and patches + m > BATCH_PATCHES:
                    break
                batch.append(self.pending.popleft())
                patches += m
            self.inflight.append(self._encode(batch))
            launched_n += 1
        return launched_n

    # ------------------------------------------------------------------ SS_VIT_DROP_ASYNC (team taktak)
    def _take_batch(self) -> List:
        batch, patches = [], 0
        while self.pending and len(batch) < BATCH_IMAGES:
            it = self.pending[0]
            gh, gw = desc(it)[4:6]
            m = gh * gw
            if batch and patches + m > BATCH_PATCHES:
                break
            batch.append(self.pending.popleft())
            patches += m
        return batch

    def _stage(self, items: List) -> dict:
        """preprocess on the prep stream + blank-group mask -> page-locked host buffer, then an event (no host wait)"""
        t0 = time.perf_counter()
        with torch.cuda.stream(self.pstream):
            pixel_values, grid_thw = self.preprocess(items)
            M = int(pixel_values.shape[0])
            G = M // 4
            buf = None
            if M % 4 == 0 and 0 < G <= _MASK_CAP and pixel_values.dim() == 2 and pixel_values.shape[1] == 588:
                v = pixel_values.view(M, 3, 196).float()
                blank = (v.std(dim=-1) < DROP_THR).all(dim=1).view(G, 4).all(dim=1).to(torch.uint8)
                buf = self._mask_ring[self._mask_idx % len(self._mask_ring)]
                self._mask_idx += 1
                buf[:G].copy_(blank, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.pstream)
        for it in items:
            it._t2_staged = True
        self.t_launch += time.perf_counter() - t0
        return {"items": items, "pv": pixel_values, "grid": grid_thw, "G": G, "buf": buf, "ev": ev}

    def _pump_staged(self, max_batches: int = 0) -> int:
        self.reap()
        launched_n = 0
        while self.pending and len(self.staged) < STAGE_DEPTH:
            self.staged.append(self._stage(self._take_batch()))
        while self.staged and len(self.inflight) < MAX_INFLIGHT:
            if max_batches > 0 and launched_n >= max_batches:
                break
            st = self.staged[0]
            if not st["ev"].query():
                if self.inflight:
                    break               # the ViT stream still has work: launch on a later tick
                st["ev"].synchronize()  # the ViT stream is idle: a ~1 ms wait for the prep kernels
            self.staged.popleft()
            self.inflight.append(self._launch_staged(st, hint=True))
            launched_n += 1
            while self.pending and len(self.staged) < STAGE_DEPTH:
                self.staged.append(self._stage(self._take_batch()))
        return launched_n

    def _launch_staged(self, st: dict, hint: bool) -> None:
        mask = None
        if hint and st["buf"] is not None and st["ev"].query():
            import numpy as _np
            mask = st["buf"][: st["G"]].numpy().astype(_np.bool_)     # copied: the ring slot is reused later
            if not DROP_TABLE:
                o = 0
                for it in st["items"]:
                    gh, gw = desc(it)[4:6]
                    n = (gh * gw) // 4
                    if getattr(it, "_t2_table", False):
                        mask[o:o + n] = False
                    o += n
        elif hint:
            self.n_staged_nohint += 1
        self.stream.wait_event(st["ev"])
        st["pv"].record_stream(self.stream)
        for it in st["items"]:
            it._t2_staged = False
        return self._encode(st["items"], pre=(st["pv"], st["grid"]), drop_hint=mask)

    def _flush_staged_for(self, ids) -> None:
        """an on-demand request for staged items: launch their staged batches now (the hint only if already on the host)"""
        keep = collections.deque()
        while self.staged:
            st = self.staged.popleft()
            if any(id(it) in ids for it in st["items"]):
                self.inflight.append(self._launch_staged(st, hint=True))
            else:
                keep.append(st)
        self.staged = keep

    # ------------------------------------------------------------------ encoding (ViT stream)
    def _encode(self, items: List, pre=None, drop_hint=None) -> None:
        # No wait on the schedule stream: the ViT reads only weights (loaded at startup), the page-locked shm ring and
        # its own stream's buffers (the caching allocator never hands this stream a block freed by another stream).
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record(self.stream)
            if pre is None:
                pixel_values, grid_thw = self.preprocess(items)
            else:
                pixel_values, grid_thw = pre
            _runner = getattr(self.model.visual, "_ss_vit", None) if drop_hint is not None else None
            if _runner is not None:
                _runner._t2_drop_hint = drop_hint
            try:
                out = self.model.visual(pixel_values, grid_thw=grid_thw)
            finally:
                if _runner is not None:
                    _runner._t2_drop_hint = None
            counts = [int(g[1] * g[2]) // 4 for g in grid_thw.tolist()] if grid_thw.dim() == 2 else [out.shape[0]]
            parts = torch.split(out, counts, dim=0) if len(items) > 1 else (out,)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(self.stream)
        for it, e in zip(items, parts):
            it._t2_emb = e
            it._t2_ev = ev
            it._t2_state = 2  # encoded (launched)
        self.n_batches += 1
        self.n_items += len(items)
        self.n_patches += sum(int(g[1] * g[2]) for g in grid_thw.tolist())
        dt = time.perf_counter() - t0
        self.t_launch += dt
        if _LOG is not None:
            _LOG.SCHED.vit_batch(len(items), sum(int(g[1] * g[2]) for g in grid_thw.tolist()), 1e3 * dt, ev0, ev,
                                 self._ondemand_now)
        if self.n_batches in (1, 10, 100) or self.n_batches % 1000 == 0:
            self.log(f"[track2-vit-prefetch] batches={self.n_batches} images={self.n_items} patches={self.n_patches} "
                     f"on_demand={self.n_ondemand} host_launch={self.t_launch:.3f}s")
        return ev

    # ------------------------------------------------------------------ consumption (prefill forward)
    def get_embeddings(self, items: List) -> torch.Tensor:
        if DROP and self.staged:
            sid = {id(it) for it in items if getattr(it, "_t2_staged", False)}
            if sid:
                self._flush_staged_for(sid)
        todo = [it for it in items if getattr(it, "_t2_state", 0) != 2]
        if todo:
            ids = {id(it) for it in todo}
            if self.pending:
                self.pending = collections.deque(it for it in self.pending if id(it) not in ids)
            self.n_ondemand += len(todo)
            self._ondemand_now = True
            try:
                for i in range(0, len(todo), BATCH_IMAGES):
                    self.inflight.append(self._encode(todo[i : i + BATCH_IMAGES]))
            finally:
                self._ondemand_now = False
        cur = torch.cuda.current_stream(self.model.visual.device)
        seen = set()
        embs = []
        for it in items:
            ev = it._t2_ev
            if id(ev) not in seen:
                seen.add(id(ev))
                cur.wait_event(ev)
            e = it._t2_emb
            e.record_stream(cur)
            embs.append(e)
        return embs[0] if len(embs) == 1 else torch.cat(embs, dim=0)

    def run_on_stream(self, fn: Callable, *args):
        """Any other ViT call (non-descriptor items): same stream, the caller's stream waits for it."""
        cur = torch.cuda.current_stream(self.model.visual.device)
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            out = fn(*args)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        cur.wait_event(ev)
        if isinstance(out, torch.Tensor):
            out.record_stream(cur)
        return out


def ready(req) -> bool:
    """True when every descriptor image of the request has been encoded (non-blocking cudaEventQuery)."""
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None) if mm is not None else None
    if not items:
        return True
    for it in items:
        st = getattr(it, "_t2_state", 0)
        if st == 0:
            continue  # not a prefetched item (never submitted): computed in the forward as before
        if st != 2 or not it._t2_ev.query():
            return False
    return True


def wait_first(req) -> None:
    """Nothing is running: block until the first waiting request's ViT batch is done (it is admitted next)."""
    mm = getattr(req, "multimodal_inputs", None)
    for it in (getattr(mm, "mm_items", None) or []) if mm is not None else []:
        if getattr(it, "_t2_state", 0) == 2:
            it._t2_ev.synchronize()


def launched(req) -> bool:
    """True when every descriptor image of the request has been launched on the ViT stream (done or in flight)."""
    mm = getattr(req, "multimodal_inputs", None)
    items = getattr(mm, "mm_items", None) if mm is not None else None
    if not items:
        return True
    return all(getattr(it, "_t2_state", 0) in (0, 2) for it in items)


def register(pf: VitPrefetcher) -> None:
    global _PF
    _PF = pf


def register_sched(sched) -> None:
    global _SCHED
    _SCHED = sched


def get() -> Optional[VitPrefetcher]:
    return _PF


def pump_behind_decode(event=None, max_batches: int = 1) -> int:
    """Called on the scheduler thread right between self.event.record() and self.event.synchronize() so ViT
    host_launch (~9ms) and ZMQ recv_pyobj() unpickling run in parallel with the GPU's 10-55 ms decode round."""
    pf = _PF
    launched = 0
    if pf is not None and HIDE_LAUNCH and (pf.pending or pf.staged):
        if event is None or not event.query():
            launched = pf.pump(max_batches=max_batches)
    if RECV_HIDE > 0 and _SCHED is not None and event is not None:
        sched = _SCHED
        if sched.pp_rank == 0 and sched.attn_tp_rank == 0 and sched.attn_cp_rank == 0:
            import zmq
            from sglang.srt.managers.mm_utils import unwrap_shm_features
            sock = sched.recv_from_tokenizer
            buf = sched._t2_recv_buf
            n_got = 0
            while not event.query():
                got_any = False
                for _ in range(4):
                    try:
                        r = sock.recv_pyobj(zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    unwrap_shm_features(r)
                    got_any = True
                    n_got += 1
                    if RECV_HIDE >= 2 and hasattr(r, "input_ids") and hasattr(r, "rid"):
                        sched.process_input_requests([r])
                    else:
                        buf.append(r)
                if not got_any:
                    break
            if RECV_HIDE >= 2 and n_got > 0 and pf is not None and not event.query():
                wq = sched.waiting_queue
                if wq:
                    if RANK_ORDER and len(wq) > 1:
                        wq.sort(key=req_rank)
                    pf.submit_reqs(wq)
                if HIDE_LAUNCH and (pf.pending or pf.staged) and launched < max_batches and not event.query():
                    launched += pf.pump(max_batches=max_batches - launched)
    return launched


def tick(sched) -> None:
    """Scheduler hook (event loop, after process_input_requests): submit newly queued requests, launch their ViT."""
    pf = _PF
    if pf is None:
        return
    if getattr(pf.model, "_t2_capture_deferred", False):
        pf.model._t2_capture_if_deferred()  # quantized ViT: capture now (before the first request, untimed)
    t0 = time.perf_counter() if _TRACE else 0.0
    wq = getattr(sched, "waiting_queue", None)
    n0 = pf.n_items
    if wq and not pf.gc_done:
        pf.gc_done = True  # startup (load, graph capture) is over: no cyclic-GC pauses inside jobs from here on
        try:
            from gravity_ocr import gcguard
            gcguard.freeze("scheduler")
        except Exception as e:
            pf.log(f"[track2-gc] scheduler: not applied ({e})")
    if wq:
        if RANK_ORDER and len(wq) > 1:
            # admission follows the client's rank (FCFS without priority scheduling never reorders the queue itself)
            wq.sort(key=req_rank)
        pf.submit_reqs(wq)
    if pf.pending or pf.staged:
        if (READY_TARGET > 0 and wq and sched.running_batch.reqs
                and sum(1 for r in wq if launched(r)) >= READY_TARGET):
            pf.reap()
            pf.n_paced += 1  # enough encoded requests are waiting for LM slots: leave the GPU to the decode for now
        elif HIDE_LAUNCH and sched.running_batch.reqs:
            # Defer ViT kernel launch (~9ms) by ~0.3ms until right before decode/extend's GPU wait, where the GPU is
            # already busy and the scheduler thread would otherwise sleep idle! Even when wq[0] is not yet launched,
            # synchronous pump() cannot make ready(wq[0]) True 0.1ms later, so deferring always wins.
            pf.reap()
        else:
            pf.pump()
    t1 = time.perf_counter() if _TRACE else 0.0
    waited = False
    if wq and sched.chunked_req is None and not sched.running_batch.reqs and not ready(wq[0]):
        tw = time.perf_counter()
        wait_first(wq[0])  # idle GPU queue otherwise: nothing to decode until the first embeddings exist
        waited = True
        if _LOG is not None:
            _LOG.SCHED.vit_wait(1e3 * (time.perf_counter() - tw))
    if _TRACE:
        t2 = time.perf_counter()
        if pf.n_items != n0 or waited or (t2 - t0) > 0.005:
            pf.log(f"[t2-vit-tick] t={t0:.4f} launched={pf.n_items - n0} launch_ms={1e3 * (t1 - t0):.1f} "
                   f"wait_first={int(waited)} wait_ms={1e3 * (t2 - t1):.1f} wq={len(wq) if wq else 0} "
                   f"run={len(sched.running_batch.reqs)}")
