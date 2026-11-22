"""TRACK2_TEXT_SPLIT (team taktak, 2026-11-22; default 0 = off): the N text crops with the largest predicted decode work
are cut at a blank gap between two text lines near their middle and read as two requests; the two transcriptions are
joined with one space (the model writes a paragraph's lines as one line). Apply after patch_minion_table_split.py.

Why: Korean text accepts only ~8 draft tokens per round, so a long paragraph (e.g. 14 handwritten lines, 300 tokens)
needs ~36 rounds and can be the last request of the job. Two halves read in parallel need about half the rounds each.

  TRACK2_TEXT_SPLIT=<N>              how many text crops may be split per job (largest predicted work first)
  TRACK2_TEXT_SPLIT_MINLINES=<n>     only crops with at least n text lines (default 5: a half keeps enough lines for the
                                     glyph-size estimate that sets its resolution)

The cut is the blank row run (no ink across the whole width, >= 2 px) nearest the middle within 30-70% of the height.

Usage: python patch_minion_text_split.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "serve.py")
s = open(p).read()
if "TEXT_SPLIT" in s:
    print("already patched", p)
    sys.exit(0)
assert "TABLE_SPLIT" in s, "apply patch_minion_table_split.py first"

CFG_OLD = 'TABLE_SPLIT_RULE = float(os.environ.get("TRACK2_TABLE_SPLIT_RULE", "0.95") or 0.95)\n'
CFG_NEW = CFG_OLD + '''TEXT_SPLIT = int(os.environ.get("TRACK2_TEXT_SPLIT", "0") or 0)          # team taktak (patch_minion_text_split.py)
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
'''

SEL_OLD = '''    step = 2 if TABLE_SPLIT > 0 else 1
    split_futs = {}
    if TABLE_SPLIT > 0:'''
SEL_NEW = '''    step = 2 if (TABLE_SPLIT > 0 or TEXT_SPLIT > 0) else 1
    split_futs = {}
    if TEXT_SPLIT > 0:   # team taktak: TRACK2_TEXT_SPLIT (patch_minion_text_split.py)
        _texts = sorted((q for q in scheduled if q["task_key"] == "text"), key=lambda q: -q.get("pred2", 0.0))[:TEXT_SPLIT]
        _loop = asyncio.get_running_loop()
        for q in _texts:
            split_futs[id(q)] = _loop.run_in_executor(_split_pool(), _split_text, q)
    if TABLE_SPLIT > 0:'''

MERGE_OLD = '''            results_map[cid] = _merge_table_html([results_map.pop(h, "") for h in sorted(hids)])
'''
MERGE_NEW = '''            _parts = [results_map.pop(h, "") for h in sorted(hids)]
            if any("#xs" in h for h in hids):   # text halves: one paragraph
                results_map[cid] = " ".join(x.strip() for x in _parts if x.strip())
            else:
                results_map[cid] = _merge_table_html(_parts)
'''

for old, new in ((CFG_OLD, CFG_NEW), (SEL_OLD, SEL_NEW), (MERGE_OLD, MERGE_NEW)):
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
