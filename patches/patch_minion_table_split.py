"""TRACK2_TABLE_SPLIT (team taktak, 2026-11-22; default 0 = off): the N tables with the largest predicted decode work are
cut at a full-width horizontal rule near their middle and read as two requests; the two HTML outputs are joined back
into one table (rows of the top part, then rows of the bottom part).

Why: with the LANES setup (patch_minion_lanes.py) every request that finishes last is a long table (the longest dev table
is 1,428 tokens = 46 decode rounds at ~40 ms each). Splitting halves its rounds; the work stays the same.

  TRACK2_TABLE_SPLIT=<N>             how many tables may be split per job (the largest by the launcher's work prediction)
  TRACK2_TABLE_SPLIT_MINH=<px>       only tables at least this tall (default 200)
  TRACK2_TABLE_SPLIT_RULE=<f>        a row counts as a full-width rule when this fraction of the table's width is dark
                                     (default 0.95: a cell spanning the cut would break the line, so it is never cut)

The cut row is the rule closest to the middle within 30-70% of the height; a table without such a rule is sent whole.
The split runs on a worker thread from the moment the job is prepared; both halves take the original crop's place in the
queue (request ranks are spaced by 2). Needs patch_minion_lanes.py (applied first).

Usage: python patch_minion_table_split.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "serve.py")
s = open(p).read()
if "TABLE_SPLIT" in s:
    print("already patched", p)
    sys.exit(0)
assert "lane_of_task" in s, "apply patch_minion_lanes.py first"

CFG_OLD = "_MPS_ENV: dict = {}\n"
CFG_NEW = CFG_OLD + '''# team taktak (2026-11-22): TRACK2_TABLE_SPLIT, see taeyang/track2/minion_port/patch_minion_table_split.py
TABLE_SPLIT = int(os.environ.get("TRACK2_TABLE_SPLIT", "0") or 0)
TABLE_SPLIT_MINH = int(os.environ.get("TRACK2_TABLE_SPLIT_MINH", "200") or 200)
TABLE_SPLIT_RULE = float(os.environ.get("TRACK2_TABLE_SPLIT_RULE", "0.95") or 0.95)
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
        a = re.sub(r"(</tbody>)?\\s*</table>\\s*$", "", out)
        b = re.sub(r"^\\s*<table[^>]*>\\s*(<tbody>)?", "", nxt.strip())
        if a == out or b == nxt.strip():       # not two tables: keep both texts
            out = out + nxt
        else:
            out = a + b
    return out
'''

PREP_OLD = '''            "prompt": prompt_str,
            "image_url": image_url,
        })'''
PREP_NEW = '''            "prompt": prompt_str,
            "image_url": image_url,
            "img_path": str(img_path),   # team taktak: TRACK2_TABLE_SPLIT reads the pixels of a few tables
        })'''

LOOP_OLD = '''    lane_rank = [0] * max(1, len(LANES))
    for i, p in enumerate(scheduled):
        # rank in the request id: the scheduler keeps its waiting queue (and the ViT prefetch queue) in this order,
        # whatever order the tokenizer manager finishes the requests in (gravity_ocr/vit_prefetch.py)
        if LANES:   # team taktak: rank within the crop's lane, lane at the end of the rid (LanePools routes on it)
            j = lane_of_task(p["task_key"])
            p["rid"] = f"t2r{lane_rank[j]:05d}j{_JOB_NO}l{j}"
            lane_rank[j] += 1
        else:
            p["rid"] = f"t2r{i:05d}j{_JOB_NO}"
        tasks.append(asyncio.ensure_future(one(p)))
        if SEND_GAP_S > 0 and i + 1 >= first and (i + 1 - first) % SEND_WAVE == 0 and i + 1 < len(scheduled):
            await asyncio.sleep(SEND_GAP_S)
    pairs = await asyncio.gather(*tasks)
    results_map = dict(pairs)'''
LOOP_NEW = '''    lane_rank = [0] * max(1, len(LANES))
    # team taktak: TRACK2_TABLE_SPLIT -- the chosen tables are split on worker threads from here on; ranks are spaced by
    # 2 so that both halves keep the original crop's place in the scheduler's queue
    step = 2 if TABLE_SPLIT > 0 else 1
    split_futs = {}
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
            results_map[cid] = _merge_table_html([results_map.pop(h, "") for h in sorted(hids)])
            ts = [t_send[h] for h in hids if h in t_send]
            td = [t_done[h] for h in hids if h in t_done]
            if ts and td:   # the client log shows the crop as finished when its last half finished
                t_send[cid], t_done[cid] = min(ts), max(td)'''

for old, new in ((CFG_OLD, CFG_NEW), (PREP_OLD, PREP_NEW), (LOOP_OLD, LOOP_NEW)):
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
