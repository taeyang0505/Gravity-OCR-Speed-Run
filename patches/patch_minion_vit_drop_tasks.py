"""SS_VIT_DROP_TABLE=0 (team taktak, 2026-11-21): the async blank-group ViT skip (patch_minion_vit_drop_async.py, apply
that first) leaves table crops alone. Measured on M3 + skip for every task: 2.13 s instead of 2.18 s, text and formula
unchanged, table -0.9 points on dev (99.03 -> 98.14), held-out and unseen tables down as well. Empty table cells and
thin rules are where the skipped groups sit, so tables are excluded and text / formula crops keep the skip.

A crop is a table when its request text carries the table prompt ("Table Recognition"); its merge groups are cleared
from the host mask before the ViT call. Default 1 (tables skipped too, the measured behaviour of the first version).

Usage: python patch_minion_vit_drop_tasks.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "gravity_ocr/vit_prefetch.py")
s = open(p).read()
if "SS_VIT_DROP_TABLE" in s:
    print("already patched", p)
    sys.exit(0)
assert "SS_VIT_DROP_ASYNC" in s, "apply patch_minion_vit_drop_async.py first"
pairs = [
    ('STAGE_DEPTH = max(1, int(os.environ.get("SS_VIT_DROP_STAGE", "2") or 2))\n',
     'STAGE_DEPTH = max(1, int(os.environ.get("SS_VIT_DROP_STAGE", "2") or 2))\n'
     'DROP_TABLE = os.environ.get("SS_VIT_DROP_TABLE", "1") == "1"   # 0: table crops are never skipped\n'),
    ('''                    it._t2_rank = rank
''', '''                    it._t2_rank = rank
                    it._t2_table = "Table Recognition" in (getattr(req, "origin_input_text", "") or "")
'''),
    ('''            mask = st["buf"][: st["G"]].numpy().astype(_np.bool_)     # copied: the ring slot is reused later
''', '''            mask = st["buf"][: st["G"]].numpy().astype(_np.bool_)     # copied: the ring slot is reused later
            if not DROP_TABLE:
                o = 0
                for it in st["items"]:
                    gh, gw = desc(it)[4:6]
                    n = (gh * gw) // 4
                    if getattr(it, "_t2_table", False):
                        mask[o:o + n] = False
                    o += n
'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
