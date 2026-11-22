"""TRACK2_LANE_TABLE_MOVE=N (team taktak, 2026-11-22; default 0 = off): the N tables with the smallest predicted decode
work are sent to the text lane instead of the table lane. Apply after patch_minion_lanes.py (and the split patches if used).

Why: with ty_m5 / ty_m6 the table lane (63 tables, below-normal MPS priority) finishes last while the text lane has slack;
fewer patches on the table lane (table glyph 10) was the largest single gain. Short tables decode in a few rounds even with
the text lane's short draft, so moving them balances the two lanes' work without touching the long tables.

Usage: python patch_minion_lanes_move.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "serve.py")
s = open(p).read()
if "LANE_TABLE_MOVE" in s:
    print("already patched", p)
    sys.exit(0)
assert "lane_of_task" in s, "apply patch_minion_lanes.py first"
pairs = [
    ('LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"\n',
     'LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"\n'
     '# team taktak (2026-11-22): the N smallest tables go to the text lane (patch_minion_lanes_move.py)\n'
     'LANE_TABLE_MOVE = int(os.environ.get("TRACK2_LANE_TABLE_MOVE", "0") or 0)\n'),
    ('''    split_of = {}
    for i, p in enumerate(scheduled):''',
     '''    split_of = {}
    moved = set()
    if LANES and LANE_TABLE_MOVE > 0:   # team taktak: TRACK2_LANE_TABLE_MOVE
        _tabs = sorted((q for q in scheduled if q["task_key"] == "table"), key=lambda q: q.get("pred2", 0.0))
        moved = {id(q) for q in _tabs[:LANE_TABLE_MOVE]}
    for i, p in enumerate(scheduled):'''),
    ('''            j = lane_of_task(p["task_key"])
            r = lane_rank[j]''',
     '''            j = lane_of_task("text") if id(p) in moved else lane_of_task(p["task_key"])
            r = lane_rank[j]'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
