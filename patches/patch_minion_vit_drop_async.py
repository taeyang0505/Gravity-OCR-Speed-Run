"""Blank 2x2-group ViT skip WITHOUT host syncs, for the Track 2 winning team's ViT prefetch (team taktak, 2026-11-21).
Default OFF (SS_VIT_DROP=1 turns it on).

Our first port (patch_minion_vit_drop.py) computed the blank mask on the GPU and read it back inside the ViT call:
every read-back (.cpu(), nonzero, int(max)) made the scheduler thread wait for the low-priority ViT stream, behind all
queued ViT work, and the run got 10% slower. Here the mask reaches the host before the ViT call, asynchronously:

  stage  (vit_prefetch.py, pump): a packed batch is preprocessed (shm ring -> resize -> patchify) on a separate "prep"
         stream one to two batches ahead; a few small kernels mark each 2x2 merge group whose four patches are flat
         (per-channel std of the normalised pixels < SS_VIT_DROP_THR, default 0.03 ~ 2 grey levels) and copy that mask
         to page-locked memory, then an event is recorded.
  launch (pump, when a ViT slot is free and the event has completed): the ViT stream waits for the prep event on the
         GPU (no host wait), and the runner gets the host mask as a hint.
  run    (glm_vit_graph_runner.py): with a hint, the kept groups (all non-blank groups, plus every SS_VIT_DROP_KEEP-th
         blank group of each image, default 8, the first one always) run the 24 blocks with their own rotary positions
         and per-image varlen attention; a skipped group's output is a copy of the nearest kept blank group before it in
         the same image. All index arrays are built on the host from the hint and go to the GPU in one async copy from
         a page-locked ring. The number and order of image tokens the language model receives are unchanged.

Same rule and defaults as our measured submissions (ty_s16 .. ty_s31). On-demand encodes (an image the prefetch had not
launched yet when its prefill came) take the original path. SS_VIT_DROP_MIN (default 1): a batch with fewer skippable
groups runs the original path.

Usage: python patch_minion_vit_drop_async.py <team dir>   (patches gravity_ocr/vit_prefetch.py and the ViT runner in place)
"""
import os
import sys

team = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    if "_SS_VIT_DROP" in s or "SS_VIT_DROP_ASYNC" in s:
        print("already patched", path)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:80])
        s = s.replace(old, new, 1)
    open(path, "w").write(s)
    print("patched", path)


# ============================================================================ vit_prefetch.py
VP = os.path.join(team, "gravity_ocr/vit_prefetch.py")

VP_CFG_OLD = 'RECV_HIDE = int(os.environ.get("TRACK2_RECV_HIDE", "2"))\n'
VP_CFG_NEW = VP_CFG_OLD + '''# team taktak (2026-11-21), SS_VIT_DROP_ASYNC: blank 2x2-group ViT skip with the mask staged ahead (see
# taeyang/track2/minion_port/patch_minion_vit_drop_async.py). Default off.
DROP = os.environ.get("SS_VIT_DROP", "0") == "1"
DROP_THR = float(os.environ.get("SS_VIT_DROP_THR", "0.03") or 0.03)
STAGE_DEPTH = max(1, int(os.environ.get("SS_VIT_DROP_STAGE", "2") or 2))
_MASK_CAP = 1 << 16
'''

VP_INIT_OLD = '''        self.n_paced = 0  # ticks on which the ready-pool target held a launch back
        self.gc_done = False
'''
VP_INIT_NEW = VP_INIT_OLD + '''        # SS_VIT_DROP_ASYNC (team taktak): prep stream + staged batches + page-locked mask ring
        self.staged: collections.deque = collections.deque()
        self.n_staged_nohint = 0
        if DROP:
            self.pstream = torch.cuda.Stream(device=model.visual.device,
                                             priority=int(os.environ.get("TRACK2_VIT_STREAM_PRIORITY", "0")))
            self._mask_ring = [torch.empty((_MASK_CAP,), dtype=torch.uint8, pin_memory=True) for _ in range(STAGE_DEPTH + 3)]
            self._mask_idx = 0
'''

VP_PUMP_OLD = '''    def pump(self, max_batches: int = 0) -> int:
        """Keep up to MAX_INFLIGHT packed batches queued on the ViT stream (non-blocking: completion is polled)."""
        self.reap()
        launched_n = 0
'''
VP_PUMP_NEW = '''    def pump(self, max_batches: int = 0) -> int:
        """Keep up to MAX_INFLIGHT packed batches queued on the ViT stream (non-blocking: completion is polled)."""
        if DROP:
            return self._pump_staged(max_batches)
        self.reap()
        launched_n = 0
'''

VP_ENCODE_OLD = '''    # ------------------------------------------------------------------ encoding (ViT stream)
    def _encode(self, items: List) -> None:'''
VP_ENCODE_NEW = '''    # ------------------------------------------------------------------ SS_VIT_DROP_ASYNC (team taktak)
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
    def _encode(self, items: List, pre=None, drop_hint=None) -> None:'''

VP_PRE_OLD = '''            pixel_values, grid_thw = self.preprocess(items)
            out = self.model.visual(pixel_values, grid_thw=grid_thw)'''
VP_PRE_NEW = '''            if pre is None:
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
                    _runner._t2_drop_hint = None'''

VP_GET_OLD = '''        todo = [it for it in items if getattr(it, "_t2_state", 0) != 2]
        if todo:'''
VP_GET_NEW = '''        if DROP and self.staged:
            sid = {id(it) for it in items if getattr(it, "_t2_staged", False)}
            if sid:
                self._flush_staged_for(sid)
        todo = [it for it in items if getattr(it, "_t2_state", 0) != 2]
        if todo:'''

# staged batches must keep being launched after the pending queue is empty
VP_T1_OLD = "    if pf is not None and HIDE_LAUNCH and pf.pending:\n"
VP_T1_NEW = "    if pf is not None and HIDE_LAUNCH and (pf.pending or pf.staged):\n"
VP_T2_OLD = "                if HIDE_LAUNCH and pf.pending and launched < max_batches and not event.query():\n"
VP_T2_NEW = "                if HIDE_LAUNCH and (pf.pending or pf.staged) and launched < max_batches and not event.query():\n"
VP_T3_OLD = "    if pf.pending:\n        if (READY_TARGET > 0"
VP_T3_NEW = "    if pf.pending or pf.staged:\n        if (READY_TARGET > 0"

patch(VP, [(VP_CFG_OLD, VP_CFG_NEW), (VP_INIT_OLD, VP_INIT_NEW), (VP_PUMP_OLD, VP_PUMP_NEW), (VP_ENCODE_OLD, VP_ENCODE_NEW),
           (VP_PRE_OLD, VP_PRE_NEW), (VP_GET_OLD, VP_GET_NEW), (VP_T1_OLD, VP_T1_NEW), (VP_T2_OLD, VP_T2_NEW),
           (VP_T3_OLD, VP_T3_NEW)])

# ============================================================================ glm_vit_graph_runner.py
RN = os.path.join(team, "gravityocr/sglang/python/sglang/srt/multimodal/glm_vit_graph_runner.py")

RN_CFG_OLD = 'MAXM_VAR = "SS_GLM_VIT_GRAPH_MAXM"\n'
RN_CFG_NEW = RN_CFG_OLD + '''# team taktak (2026-11-21) SS_VIT_DROP_ASYNC: blank 2x2-group skip driven by a host mask staged by the ViT prefetch
_SS_VIT_DROP_KEEP = max(1, int(os.environ.get("SS_VIT_DROP_KEEP", "8") or 8))
_SS_VIT_DROP_MIN = max(1, int(os.environ.get("SS_VIT_DROP_MIN", "1") or 1))
_SS_DROP_PIN_RING = []
_SS_DROP_PIN_IDX = 0
_SS_DROP_PIN_CAP = 1 << 17
'''

RN_RUN_OLD = '''        if threading.get_ident() != self._owner:
            return self._run_other_thread(x, grid_thw)
'''
RN_RUN_NEW = RN_RUN_OLD + '''        _hint = getattr(self, "_t2_drop_hint", None)
        if _hint is not None:
            self._t2_drop_hint = None
            y = self._drop_run_host(x, grid_thw, _hint)
            if y is not None:
                return y
'''

RN_BODY_OLD = '''    def _eager_orig(self, x, grid_thw):
'''
RN_BODY_NEW = '''    def _drop_run_host(self, x, grid_thw, blank):
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
'''

patch(RN, [(RN_CFG_OLD, RN_CFG_NEW), (RN_RUN_OLD, RN_RUN_NEW), (RN_BODY_OLD, RN_BODY_NEW)])
