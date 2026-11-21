# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# New file in the GravityOCR self-speculative decoding patch for SGLang
# (sgl-project/sglang @ 1519acf37c23f2189adb93f57ca9cd2db1bebf18, tag v0.5.10; Apache-2.0).
# Design notes: https://github.com/trillion-labs/GravityOCR
#
# Portions Copyright 2023-2024 SGLang Team, licensed under the Apache License, Version 2.0 (the notice of
# sglang/srt/models/glm_ocr.py): GlmVitGraphRunner._body is adapted from GlmOcrVisionModel.forward in that file
# (SGLang v0.5.10).
"""Exact-M CUDA graphs for the GLM-OCR ViT (GlmOcrVisionModel). Off unless SS_GLM_VIT_GRAPH_MAXM > 0; the GravityOCR
launchers set it to 2048 under FAST=1, their default.

Settings, parsed strictly by vit_graph_settings() (a malformed value stops the server at startup with a ValueError that
names the variable) and printed once when GLM-OCR loads ("[ss-vit-graph] settings: ..."):
  SS_GLM_VIT_GRAPH_MAXM   largest patch count per image that gets a graph: decimal digits only; unset, empty or 0 = off;
                          rounded down to a multiple of 4 (the settings line says so); at most 2x the ViT's rotary table
                          (16384 for GLM-OCR: the largest graph is captured with the grid (1, 2, MAXM / 2)).
  SS_GLM_VIT_EAGER_LIST   1 (default) or 0: how calls without a graph run (below).
  SS_GLM_VIT_GRAPH_CHECK  1 (default) or 0: the startup self-check (below).
  Empty counts as unset; anything else (e.g. "2k", " 8", "-4", "true") is an error.

Why: a GLM crop's ViT call is hundreds of eager launches plus a per-block cu_seqlens H2D copy and a per-block
`.max().item()` sync (FA3), so its host time is many times its GPU time.

What (all in this file; wired in by models/glm_ocr.py and the list dispatch in layers/attention/vision.py):
  * capture at the end of GlmOcrForConditionalGeneration.load_weights (setup()), under no_grad (see capture() for why not
    inference_mode), with the default dtype set to float32 (the serving context; load_weights runs inside
    set_default_torch_dtype(bf16)): every M = 4k <= MAXM (every patch count reachable with even h and w), largest first,
    into ONE shared pool. Static inputs are [:M] views of max-size buffers (pixel rows, cos, sin) plus a per-graph
    cu_seqlens = [0, M]. A graph is keyed on M alone: the only (h, w)-dependent tensors are cos / sin, which are graph inputs.
  * a capture failure (out of memory, a CUDA or capture error) is not fatal: setup() releases every graph captured so far,
    the static buffers and the pool, prints "[ss-vit-graph] OFF: capture failed (...)", and the model keeps its original
    eager forward. Only when the CUDA context itself is unusable afterwards is the original error raised.
  * replay only for one image with t == 1, M captured, x has M rows (FA3 backend and no DP encoder are checked at capture).
    The output is cloned right after the replay (graphs share one pool, so a later replay can overwrite it).
  * everything else (M > MAXM, several images, t > 1) runs the eager path with cu_seqlens passed once as the FA3 list
    [cu_gpu, max_seqlen] (no per-block H2D copy and no per-block .item() sync). SS_GLM_VIT_EAGER_LIST=0 keeps the
    original eager forward for those calls instead.
  * one caller thread: run() serves the thread that built the runner (see run()); a call from any other thread takes the
    original eager forward, which shares nothing with the graphs.
  * self-check (SS_GLM_VIT_GRAPH_CHECK=1): every M right after its capture (seeded random input, replay vs the ORIGINAL
    eager forward, torch.equal), then all M again in shuffled order after all captures, plus a few grids that differ from
    the capture grid and a few eager-list fallback shapes. A mismatching M is dropped (-> eager); a mismatching eager-list
    shape turns the eager-list fallback off.
  * startup lines "[ss-vit-graph] ..." (settings; graph count, seconds, GB, dropped M) and a usage line after the 1st,
    10th, 100th and every 1000th call.
"""
import os
import random
import re
import threading
import time
import traceback
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import torch

MAXM_VAR = "SS_GLM_VIT_GRAPH_MAXM"
# team taktak (2026-11-21) SS_VIT_DROP_ASYNC: blank 2x2-group skip driven by a host mask staged by the ViT prefetch
_SS_VIT_DROP_KEEP = max(1, int(os.environ.get("SS_VIT_DROP_KEEP", "8") or 8))
_SS_VIT_DROP_MIN = max(1, int(os.environ.get("SS_VIT_DROP_MIN", "1") or 1))
_SS_DROP_PIN_RING = []
_SS_DROP_PIN_IDX = 0
_SS_DROP_PIN_CAP = 1 << 17
EAGER_LIST_VAR = "SS_GLM_VIT_EAGER_LIST"
CHECK_VAR = "SS_GLM_VIT_GRAPH_CHECK"


class VitGraphSettings(NamedTuple):
    maxm: int  # graphs for M = 4, 8, .., maxm; 0 = off
    requested: int  # SS_GLM_VIT_GRAPH_MAXM as given (maxm is it rounded down to a multiple of 4)
    eager_list: bool
    check: bool


def _parse_maxm(env: Mapping[str, str]) -> int:
    raw = env.get(MAXM_VAR, "")
    if raw == "":
        return 0
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError(
            f"{MAXM_VAR}={raw!r}: expected the largest patch count per image as decimal digits (e.g. 2048); "
            f"unset, empty or 0 turns the GLM-OCR ViT CUDA graphs off"
        )
    return int(raw)


def _parse_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "")
    if raw == "":
        return default
    if raw not in ("0", "1"):
        raise ValueError(f"{name}={raw!r}: expected 0 or 1 (unset or empty = {int(default)})")
    return raw == "1"


def glm_vit_graph_maxm(env: Optional[Mapping[str, str]] = None) -> int:
    """SS_GLM_VIT_GRAPH_MAXM rounded down to a multiple of 4 (0 = off); ValueError if malformed. Reads only that variable
    (layers/attention/vision.py calls it at import for every vision model, so the two flags are not validated there)."""
    m = _parse_maxm(os.environ if env is None else env)
    return m - m % 4


def vit_graph_settings(env: Optional[Mapping[str, str]] = None) -> VitGraphSettings:
    """All three settings, strictly parsed (ValueError naming the variable on a malformed value)."""
    env = os.environ if env is None else env
    requested = _parse_maxm(env)
    return VitGraphSettings(
        maxm=requested - requested % 4,
        requested=requested,
        eager_list=_parse_flag(env, EAGER_LIST_VAR, True),
        check=_parse_flag(env, CHECK_VAR, True),
    )


def describe_settings(s: VitGraphSettings) -> str:
    if s.maxm <= 0:
        why = "unset or 0" if s.requested == 0 else f"{s.requested} rounds down to 0"
        return f"{MAXM_VAR} {why} -> off (the original eager ViT forward)"
    note = "" if s.maxm == s.requested else f" (rounded down from {s.requested} to a multiple of 4)"
    return (f"{MAXM_VAR}={s.maxm}{note} -> graphs for M = 4..{s.maxm} step 4 ({s.maxm // 4}); "
            f"{EAGER_LIST_VAR}={int(s.eager_list)}; {CHECK_VAR}={int(s.check)}")


_settings_reported = False


def report_settings(s: VitGraphSettings) -> None:
    """Print the parsed settings once per process."""
    global _settings_reported
    if not _settings_reported:
        _settings_reported = True
        _p(f"settings: {describe_settings(s)}")


def _p(msg: str) -> None:  # print: the in-process Engine logs at level "error", so logger.info would be invisible
    print(f"[ss-vit-graph] {msg}", flush=True)


def _mem() -> Tuple[int, int]:
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _drop_frames(exc: BaseException) -> None:
    """Clear the locals of the finished frames an exception (and its cause / context chain) keeps alive: a half-captured
    graph, its output and the side stream would otherwise stay referenced, and their memory unreleasable, for as long as
    the exception object lives."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)  # skips frames still executing (the caller's)
        exc = exc.__cause__ or exc.__context__


_USAGE_AT = frozenset((1, 10, 100))



import numpy as _t2_np

_T2_POS_CACHE = {}
_T2_POS_PIN_RING = []
_T2_POS_PIN_IDX = 0
_T2_POS_PIN_CAP = 65536
_T2_I32_PIN_RING = []
_T2_I32_PIN_IDX = 0
_T2_I32_PIN_CAP = 512


def _t2_h2d_i32(t: torch.Tensor, device) -> torch.Tensor:
    global _T2_I32_PIN_IDX
    n = int(t.numel())
    if n <= _T2_I32_PIN_CAP:
        if not _T2_I32_PIN_RING:
            for _ in range(8):
                _T2_I32_PIN_RING.append(torch.empty((_T2_I32_PIN_CAP,), dtype=torch.int32, pin_memory=True))
        buf = _T2_I32_PIN_RING[_T2_I32_PIN_IDX & 7]
        _T2_I32_PIN_IDX += 1
        buf[:n].copy_(t)
        return buf[:n].to(device, non_blocking=True)
    return t.pin_memory().to(device, non_blocking=True)


def _t2_rot_pos_emb(vis, grid_thw):
    """Glm4vVisionModel.rot_pos_emb (same values and order) with reusable page-locked staging ring (no per-batch cudaHostAlloc)."""
    global _T2_POS_PIN_IDX
    g = grid_thw.cpu() if grid_thw.is_cuda else grid_thw
    gl = [tuple(int(v) for v in row) for row in g.tolist()]
    sm = vis.spatial_merge_size
    parts = []
    for t, h, w in gl:
        p = _T2_POS_CACHE.get((h, w))
        if p is None:
            r = _t2_np.arange(h // sm)[:, None, None, None] * sm + _t2_np.arange(sm)[None, None, :, None]
            c = _t2_np.arange(w // sm)[None, :, None, None] * sm + _t2_np.arange(sm)[None, None, None, :]
            r = _t2_np.broadcast_to(r, (h // sm, w // sm, sm, sm)).reshape(-1)
            c = _t2_np.broadcast_to(c, (h // sm, w // sm, sm, sm)).reshape(-1)
            p = _t2_np.ascontiguousarray(_t2_np.stack([r, c], axis=-1).astype(_t2_np.int64))
            if len(_T2_POS_CACHE) < 8192:
                _T2_POS_CACHE[(h, w)] = p
        parts.append(p if t == 1 else _t2_np.tile(p, (t, 1)))
    pos = parts[0] if len(parts) == 1 else _t2_np.concatenate(parts, axis=0)
    n = int(pos.shape[0])
    if n <= _T2_POS_PIN_CAP:
        if not _T2_POS_PIN_RING:
            for _ in range(4):
                _T2_POS_PIN_RING.append(torch.empty((_T2_POS_PIN_CAP, 2), dtype=torch.int64, pin_memory=True))
        buf = _T2_POS_PIN_RING[_T2_POS_PIN_IDX & 3]
        _T2_POS_PIN_IDX += 1
        buf[:n].copy_(torch.from_numpy(pos))
        pos_ids = buf[:n].to(vis.device, non_blocking=True)
    else:
        pos_ids = torch.from_numpy(pos).pin_memory().to(vis.device, non_blocking=True)
    cos, sin = vis.rotary_pos_emb.get_cos_sin(max(max(h, w) for _, h, w in gl))
    return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1), pos_ids


class GlmVitGraphRunner:
    def __init__(self, vis, settings: VitGraphSettings):
        self.vis = vis
        self.maxm = settings.maxm
        # graphs[M] = (graph, its output, its cu_seqlens): see _capture_all for why cu_seqlens stays in the tuple
        self.graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        self.eager_list_ok = settings.eager_list
        self.check = settings.check
        self.n_replay = 0
        self.n_eager_list = 0
        self.n_eager_orig = 0
        self.n_other_thread = 0
        self.dropped: List[int] = []
        self.X = self.C = self.S = None
        self.pool = None
        # The thread that builds the runner (load_weights) is the one that captures and, in SGLang, the one that runs
        # every model forward afterwards (see run()).
        self._owner = threading.get_ident()

    # ------------------------------------------------------------------ the forward body (== GlmOcrVisionModel.forward)
    def _body(self, x, cu_list, cos, sin):
        # KEEP IN SYNC with GlmOcrVisionModel._ss_forward_orig (models/glm_ocr.py, upstream's GlmOcrVisionModel.forward)
        # from `x = self.patch_embed(x)` to the end, minus the prologue the callers prepare (dtype / device cast, rotary
        # cos / sin, cu_seqlens). Guards: the startup self-check (torch.equal against _ss_forward_orig for every captured
        # M and the eager-list shapes) and an AST comparison in trida's inference/gravityocr/test_gravityocr_vit_graph.py.
        vis = self.vis
        x = vis.patch_embed(x)
        x = x.unsqueeze(1)
        for blk in vis.blocks:
            x = blk(x, cu_seqlens=cu_list, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin)
        x = vis.post_layernorm(x)
        x = x.view(-1, vis.spatial_merge_size, vis.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = vis.downsample(x).view(-1, vis.out_hidden_size)
        x = vis.merger(x)
        return x

    def _rot(self, grid_thw):
        cos, sin, _ = _t2_rot_pos_emb(self.vis, grid_thw)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)

    def _eager_list(self, x, grid_thw):
        vis = self.vis
        x = x.to(device=vis.device, dtype=vis.dtype)  # patch_embed runs inside _body
        cos, sin = self._rot(grid_thw)
        g = grid_thw.cpu() if grid_thw.is_cuda else grid_thw
        cu = torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu = torch.cat([cu.new_zeros(1), cu])
        seq_lens = cu[1:] - cu[:-1]
        max_seqlen = int(seq_lens.max())
        # [TRACK2-PINNED-H2D] reusable pinned staging ring (no per-batch cudaHostAlloc on the scheduler thread)
        cu_gpu = _t2_h2d_i32(cu, vis.device)
        if cu.numel() == 2:
            return self._body(x, [cu_gpu, max_seqlen], cos, sin)
        return self._body(x, [cu_gpu, _t2_h2d_i32(seq_lens, vis.device), max_seqlen], cos, sin)

    def _drop_run_host(self, x, grid_thw, blank):
        """the blank-group skip from a host mask (no device -> host sync); None when it does not apply"""
        global _SS_DROP_PIN_IDX
        import numpy as np
        vis = self.vis
        dev = vis.device
        M, D = x.shape
        if M % 4 or D not in (1176, 588) or grid_thw.is_cuda:
            return None
        gl = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        if any(t != 1 for t, _, _ in gl):
            return None
        per = np.array([h * w // 4 for _, h, w in gl], dtype=np.int64)
        G = M // 4
        if int(per.sum()) != G or blank.shape[0] != G:
            return None
        starts = np.concatenate([[0], np.cumsum(per)[:-1]])
        img = np.repeat(np.arange(len(per)), per)
        cb = np.cumsum(blank.astype(np.int64))
        base = np.where(starts > 0, cb[np.maximum(starts - 1, 0)], 0)
        rank = cb - 1 - base[img]
        keep = (~blank) | (rank % _SS_VIT_DROP_KEEP == 0)
        n_skip = int(G - keep.sum())
        if n_skip < _SS_VIT_DROP_MIN:
            return None
        kg = np.flatnonzero(keep).astype(np.int32)
        sk = np.flatnonzero(~keep).astype(np.int32)
        ar = np.arange(G, dtype=np.int64)
        lastk = np.maximum.accumulate(np.where(blank & keep, ar, -1))
        src = lastk[sk].astype(np.int32)                      # >= 0: an image's first blank group is always kept
        rows = (kg[:, None] * 4 + np.arange(4, dtype=np.int32)[None, :]).reshape(-1).astype(np.int32)
        kept_per_img = np.bincount(img[kg], minlength=len(per)).astype(np.int64) * 4
        cu = np.concatenate([[0], np.cumsum(kept_per_img)]).astype(np.int32)
        seq = (cu[1:] - cu[:-1]).astype(np.int32)
        max_seqlen = int(seq.max())
        packed = np.concatenate([rows, kg, sk, src, cu, seq]).astype(np.int32)
        n = packed.shape[0]
        if n <= _SS_DROP_PIN_CAP:
            if not _SS_DROP_PIN_RING:
                for _ in range(8):
                    _SS_DROP_PIN_RING.append(torch.empty((_SS_DROP_PIN_CAP,), dtype=torch.int32, pin_memory=True))
            buf = _SS_DROP_PIN_RING[_SS_DROP_PIN_IDX & 7]
            _SS_DROP_PIN_IDX += 1
            buf[:n].copy_(torch.from_numpy(packed))
            d = buf[:n].to(dev, non_blocking=True)
        else:
            d = torch.from_numpy(packed).pin_memory().to(dev, non_blocking=True)
        o = 0
        rows_t = d[o:o + rows.shape[0]]; o += rows.shape[0]
        kg_t = d[o:o + kg.shape[0]]; o += kg.shape[0]
        sk_t = d[o:o + sk.shape[0]]; o += sk.shape[0]
        src_t = d[o:o + src.shape[0]]; o += src.shape[0]
        cu_t = d[o:o + cu.shape[0]]; o += cu.shape[0]
        seq_t = d[o:o + seq.shape[0]]
        xd = x.to(device=dev, dtype=vis.dtype)
        cos, sin = self._rot(grid_thw)
        x_k = xd.index_select(0, rows_t)
        cu_list = [cu_t, max_seqlen] if len(per) == 1 else [cu_t, seq_t, max_seqlen]
        y_k = self._body(x_k, cu_list, cos.index_select(0, rows_t), sin.index_select(0, rows_t))
        y = torch.empty((G, y_k.shape[-1]), device=dev, dtype=y_k.dtype)
        y.index_copy_(0, kg_t.long(), y_k)
        if sk.shape[0]:
            y.index_copy_(0, sk_t.long(), y.index_select(0, src_t))
        self.n_drop = getattr(self, "n_drop", 0) + 1
        self.n_drop_groups = getattr(self, "n_drop_groups", 0) + n_skip
        self.n_drop_total = getattr(self, "n_drop_total", 0) + G
        if self.n_drop in (1, 10, 100, 1000):
            _p(f"[ss-vit-drop-async] call {self.n_drop}: skipped {self.n_drop_groups} of {self.n_drop_total} merge groups "
               f"so far (keep every {_SS_VIT_DROP_KEEP}th blank)")
        return y

    def _eager_orig(self, x, grid_thw):
        return self.vis._ss_forward_orig(x, grid_thw)

    # ------------------------------------------------------------------ serving entry point
    def run(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # Single caller. Replays share per-process state: the static inputs X / C / S, one graph pool and each graph's
        # output tensor, so two concurrent run() calls could overwrite each other's inputs or outputs. In SGLang the ViT
        # is reached only from the model forward (GlmOcrVisionModel.forward <- Glm4vForConditionalGeneration
        # .get_image_feature <- the multimodal embedding step of the forward), which runs on the scheduler process's
        # event-loop thread: run_scheduler_process builds the Scheduler (model load, load_weights, this capture) and then
        # runs its event loop on the same thread, for the normal, overlap and speculative (v1 / v2) loops alike. DP
        # replicas are separate processes with their own runner, the DP encoder (mm_enable_dp_encoder) gets no runner
        # (models/glm_ocr.py), and the scheduler's other threads never call the model. A call from another thread is
        # served by the original eager forward, which shares nothing with the graphs.
        if threading.get_ident() != self._owner:
            return self._run_other_thread(x, grid_thw)
        _hint = getattr(self, "_t2_drop_hint", None)
        if _hint is not None:
            self._t2_drop_hint = None
            y = self._drop_run_host(x, grid_thw, _hint)
            if y is not None:
                return y
        n = self.n_replay + self.n_eager_list + self.n_eager_orig
        if n and (n % 1000 == 0 or n in _USAGE_AT):
            _p(f"usage: calls={n} replay={self.n_replay} eager_list={self.n_eager_list} eager_orig={self.n_eager_orig}"
               f"{f' (other thread {self.n_other_thread})' if self.n_other_thread else ''}")
        if self.graphs and grid_thw.shape[0] == 1:
            t, h, w = (int(v) for v in grid_thw[0].tolist())
            M = t * h * w
            if t == 1 and M in self.graphs and x.shape[0] == M:
                self.n_replay += 1
                return self._replay(x, grid_thw, M)
        if self.eager_list_ok:
            self.n_eager_list += 1
            return self._eager_list(x, grid_thw)
        self.n_eager_orig += 1
        return self._eager_orig(x, grid_thw)

    def _run_other_thread(self, x, grid_thw):
        if not self.n_other_thread:
            _p(f"run() called from thread {threading.get_ident()}, not the capture thread {self._owner}: such calls use "
               f"the original eager forward (the graphs' static buffers serve one thread)")
        self.n_other_thread += 1
        self.n_eager_orig += 1
        return self._eager_orig(x, grid_thw)

    def _load_static(self, M, x, grid_thw):
        cos, sin, _ = _t2_rot_pos_emb(self.vis, grid_thw)
        d = cos.shape[-1]
        if x.shape[-1] == 1176 and self.X.shape[-1] == 588:
            x = x.view(M, 3, 2, 196)[:, :, 0].reshape(M, 588).contiguous()
        self.X[:M].copy_(x)
        self.C[:M, :d].copy_(cos)
        self.C[:M, d:].copy_(cos)
        self.S[:M, :d].copy_(sin)
        self.S[:M, d:].copy_(sin)

    def _replay(self, x, grid_thw, M):
        # single caller only (see run()): the static inputs and the graph's output are shared per process
        gr, out = self.graphs[M][:2]
        self._load_static(M, x, grid_thw)
        gr.replay()
        return out.clone()

    # ------------------------------------------------------------------ startup: capture + self-check, with fallback
    def setup(self) -> bool:
        """Capture (and self-check) the graphs at startup. True: route the model's forward through run(). False: keep
        the original eager forward, with every graph, buffer and pool of a failed capture released.

        A malformed or impossible setting raises ValueError before anything is allocated. A failure while capturing (out
        of memory, a CUDA or capture error) is not fatal, because the graphs only change speed: everything captured so
        far is released and the eager forward stays. If the device cannot even be synchronised after the failure (the
        CUDA context is unusable, e.g. after a device-side assert), the original error is raised."""
        self._check_maxm()
        mem0 = _mem()
        failed = None
        try:
            self.capture()
        except Exception as e:
            failed = e
            _drop_frames(e)
        if failed is not None:
            # Released only here, after the except block: inside it the interpreter's handled-exception state and the
            # traceback still reference capture()'s frames, so synchronize + empty_cache there could free nothing.
            msg = (str(failed).strip().splitlines() or [""])[0][:400]
            try:
                self.release()
            except Exception:
                raise failed  # the CUDA context is unusable: not recoverable here
            mem1 = _mem()
            _p(f"OFF: capture failed ({type(failed).__name__}: {msg}); released the partial graphs and buffers "
               f"(mem {(mem1[0] - mem0[0]) / 1e9:+.2f} GB alloc / {(mem1[1] - mem0[1]) / 1e9:+.2f} GB reserved vs before "
               f"the capture); using the original eager forward")
            return False
        if self.graphs or self.eager_list_ok:
            return True
        if self.X is not None:  # captured, but nothing usable passed the self-check (capture() reported the other OFFs)
            _p("OFF: no graph passed the self-check and the eager-list fallback is off; using the original eager forward")
            self.release()
        return False

    def _check_maxm(self) -> None:
        # The largest graph is captured with the grid (1, 2, MAXM // 2): its rotary positions run to MAXM // 2 - 1, and
        # rot_pos_emb indexes a table of rotary_pos_emb.cos_sin_cache.shape[0] positions (8192 for GLM-OCR), so a larger
        # MAXM would index past the table on the device.
        rows = int(self.vis.rotary_pos_emb.cos_sin_cache.shape[0])
        if self.maxm // 2 > rows:
            raise ValueError(f"{MAXM_VAR}={self.maxm}: at most {2 * rows} (2x the ViT's {rows}-position rotary table; the "
                             f"largest graph is captured with the grid (1, 2, {MAXM_VAR} / 2))")

    def release(self) -> None:
        """Drop every graph, its output and cu_seqlens, the static buffers and the graph pool, and hand their memory back
        to the device."""
        self.graphs.clear()
        self.X = self.C = self.S = None
        self.pool = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _fa3_backend_name(self) -> Optional[str]:
        # [TRACK2-L40S-VIT-GRAPH]
        """Allow both FA3 (Hopper) and patched VisionTritonAttention (L40S sm_89 SDPA+Triton list path)."""
        from sglang.srt.layers.attention.vision import VisionFlash3Attention, VisionTritonAttention

        for b in self.vis.blocks:
            if not isinstance(b.attn.qkv_backend, (VisionFlash3Attention, VisionTritonAttention)):
                return type(b.attn.qkv_backend).__name__
        return None

    def capture(self) -> None:
        backend = self._fa3_backend_name()
        if backend is not None:
            _p(f"OFF: ViT attention backend is {backend}, graphs need fa3")
            self.eager_list_ok = False
            return
        if self.maxm < 4:
            _p("OFF: MAXM < 4")
            return
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)  # = the serving context (load_weights runs under set_default_torch_dtype)
        try:
            # no_grad, NOT inference_mode: the first CUDA-graph capture of the process lazily creates the CUDA
            # generator's graph-state tensors; created under inference_mode they become inference tensors and the
            # LLM's CudaGraphRunner capture (outside inference_mode) then dies with "Inplace update to inference
            # tensor outside InferenceMode". The kernels do not depend on the mode.
            with torch.no_grad():
                self._capture_all(self.vis.device)
        finally:
            torch.set_default_dtype(prev_dtype)

    def _capture_all(self, dev):
        vis = self.vis
        t_all = time.perf_counter()
        d_in = vis.patch_embed.proj.weight.numel() // vis.patch_embed.hidden_size
        if os.environ.get("TRACK2_FOLD_PATCH_EMBED", "1") == "1" and d_in == 1176:
            _pw = vis.patch_embed.proj.weight
            if _pw.ndim == 5 and _pw.shape[2] == 2:
                _wf = (_pw[:, :, 0] + _pw[:, :, 1]).reshape(vis.patch_embed.hidden_size, 588).contiguous()
                _pb = vis.patch_embed.proj.bias
                _orig_pe_fwd = vis.patch_embed.forward
                def _folded_pe_fwd(_x, _wf=_wf, _pb=_pb, _orig=_orig_pe_fwd):
                    if _x.shape[-1] == 588:
                        return torch.nn.functional.linear(_x, _wf, _pb)
                    return _orig(_x)
                vis.patch_embed.forward = _folded_pe_fwd
                d_in = 588
                _p("folded GlmOcrVisionPatchEmbed Conv3d [1176] -> F.linear [588] (2x smaller)")
        # Warm the eager path twice before any capture with one small image: 26x26 = 676 patches, a typical crop. It
        # only compiles the dynamic-shape rotary kernel and creates the cuBLAS handles, which must not happen inside a
        # capture; the size itself does not matter.
        g0 = torch.tensor([[1, 26, 26]])
        for _ in range(2):
            vis._ss_forward_orig(torch.zeros(676, d_in, device=dev, dtype=vis.dtype), g0)
        torch.cuda.synchronize()
        cos0, _ = self._rot(torch.tensor([[1, 2, self.maxm // 2]]))  # only for the rotary width and dtype
        rot_d = cos0.shape[-1]
        mem0 = _mem()
        self.X = torch.zeros(self.maxm, d_in, device=dev, dtype=vis.dtype)
        self.C = torch.zeros(self.maxm, rot_d, device=dev, dtype=cos0.dtype)
        self.S = torch.zeros(self.maxm, rot_d, device=dev, dtype=cos0.dtype)
        self.pool = torch.cuda.graph_pool_handle()
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        Ms = [4 * k for k in range(self.maxm // 4, 0, -1)]  # largest -> smallest
        refs: Dict[int, torch.Tensor] = {}
        t_cap = 0.0
        t_chk = 0.0
        for M in Ms:
            a = time.perf_counter()
            # Capture grid (1, 2, M // 2): the smallest valid h (h and w must be even for the 2x2 merge) makes every
            # M = 4k capturable. Any grid with the same M replays this graph: cos / sin are graph inputs.
            g = torch.tensor([[1, 2, M // 2]])
            cos, sin = self._rot(g)
            xs, cs, ss = self.X[:M], self.C[:M], self.S[:M]
            cs.copy_(cos)
            ss.copy_(sin)
            cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
            side.wait_stream(cur)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                self._body(xs, [cu, M], cs, ss)  # warm run of this M (Triton specialisation, cuBLAS heuristics)
                side.synchronize()
                # capture_begin/end directly: torch.cuda.graph() would add gc.collect() + empty_cache() per graph
                gr.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    out = self._body(xs, [cu, M], cs, ss)
                finally:
                    gr.capture_end()
            cur.wait_stream(side)
            torch.cuda.synchronize()
            # Keep every graph and its output referenced (they live in the shared pool), and also cu: the captured FA3
            # kernels read cu_seqlens from this tensor's device memory at every replay, so it must outlive the graph.
            self.graphs[M] = (gr, out, cu)
            t_cap += time.perf_counter() - a
            if self.check:
                a = time.perf_counter()
                rnd = self._rand(M, d_in, dev)
                ref = vis._ss_forward_orig(rnd, g)
                y = self._replay(rnd, g, M)
                if not torch.equal(y, ref):
                    self.dropped.append(M)
                else:
                    refs[M] = ref
                t_chk += time.perf_counter() - a
        for M in self.dropped:
            self.graphs.pop(M, None)
        n_shuf = n_grid = n_el = 0
        bad_shuf, bad_grid, bad_el = [], [], []
        if self.check:
            a = time.perf_counter()
            order = list(refs.keys())
            random.Random(1).shuffle(order)
            for M in order:  # arbitrary replay order after ALL captures (shared-pool aliasing)
                g = torch.tensor([[1, 2, M // 2]])
                y = self._replay(self._rand(M, d_in, dev), g, M)
                n_shuf += 1
                if not torch.equal(y, refs[M]):
                    bad_shuf.append(M)
            refs.clear()
            rng = random.Random(2)
            cand = []
            for M in (16, 144, 360, 576, 1024, 1536, 2048):
                if M in self.graphs:
                    hs = [h for h in range(4, 200, 2) if M % h == 0 and (M // h) % 2 == 0]
                    if hs:
                        h = rng.choice(hs)
                        cand.append((M, h, M // h))
            for (M, h, w) in cand:  # same M, grid != capture grid
                g = torch.tensor([[1, h, w]])
                rnd = self._rand(M, d_in, dev)
                ref = vis._ss_forward_orig(rnd, g)
                y = self._replay(rnd, g, M)
                n_grid += 1
                if not torch.equal(y, ref):
                    bad_grid.append((M, h, w))
            if self.eager_list_ok:
                # Eager-list fallback shapes: 46x46 = 2116, 50x60 = 3000 and 64x64 = 4096 patches are above the default
                # MAXM 2048 (calls that get no graph); 2x1100 = 2200 is the extreme aspect ratio (a single merge row,
                # rotary positions far beyond any captured grid).
                for (h, w) in ((46, 46), (50, 60), (64, 64), (2, 1100)):
                    g = torch.tensor([[1, h, w]])
                    rnd = self._rand(h * w, d_in, dev)
                    ref = vis._ss_forward_orig(rnd, g)
                    y = self._eager_list(rnd, g)
                    n_el += 1
                    if not torch.equal(y, ref):
                        bad_el.append((h, w))
                # Two images in one call (600 + 256 patches): the multi-image cu_seqlens list and its max_seqlen.
                gm = torch.tensor([[1, 20, 30], [1, 16, 16]])
                rnd = self._rand(856, d_in, dev)
                ref = vis._ss_forward_orig(rnd, gm)
                y = self._eager_list(rnd, gm)
                n_el += 1
                if not torch.equal(y, ref):
                    bad_el.append("2img")
                if bad_el:
                    self.eager_list_ok = False
            t_chk += time.perf_counter() - a
            for M in bad_shuf:
                self.graphs.pop(M, None)
                self.dropped.append(M)
            for (M, h, w) in bad_grid:
                self.graphs.pop(M, None)
                self.dropped.append(M)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem1 = _mem()
        _p(f"ON: {len(self.graphs)} graphs (M 4..{self.maxm}, step 4) in {time.perf_counter() - t_all:.1f} s "
           f"(capture {t_cap:.1f} s, self-check {t_chk:.1f} s, check={int(self.check)}), mem +{(mem1[0] - mem0[0]) / 1e9:.2f} GB alloc "
           f"/ +{(mem1[1] - mem0[1]) / 1e9:.2f} GB reserved; dropped M: {sorted(set(self.dropped))}; "
           f"shuffled {n_shuf} bad {bad_shuf}; other-grid {n_grid} bad {bad_grid}; "
           f"eager_list={'on' if self.eager_list_ok else 'off'} (checked {n_el}, bad {bad_el}); load {os.getloadavg()[0]:.0f}")

    def _rand(self, M, d_in, dev):
        gen = torch.Generator(device=dev).manual_seed(1000 + M)
        return torch.randn(M, d_in, device=dev, dtype=self.vis.dtype, generator=gen)
