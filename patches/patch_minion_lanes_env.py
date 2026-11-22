"""Per-lane environment, any variable (team taktak, 2026-11-22). Apply after patch_minion_lanes.py.

  TRACK2_LANE<j>_<VAR>=<value>   sets <VAR> for lane j's server only, e.g. TRACK2_LANE0_TRACK2_PREFILL_COALESCE=4
                                 (the table lane admits prefills in smaller groups) or
                                 TRACK2_LANE1_TRACK2_VIT_PF_INFLIGHT=3. Unset: nothing changes.

Usage: python patch_minion_lanes_env.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "serve.py")
s = open(p).read()
if "TRACK2_LANE{j}_" in s:
    print("already patched", p)
    sys.exit(0)
old = '''def lane_env(j: int) -> dict:
    out = {}
'''
new = '''def lane_env(j: int) -> dict:
    out = {}
    pre = f"TRACK2_LANE{j}_"   # team taktak: TRACK2_LANE<j>_<VAR>=value sets VAR for lane j only (patch_minion_lanes_env.py)
    for k, v in os.environ.items():
        if k.startswith(pre) and len(k) > len(pre):
            out[k[len(pre):]] = v
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
