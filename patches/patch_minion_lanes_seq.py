"""TRACK2_LANE_SEQUENTIAL (team taktak, 2026-11-22; default 1 when LANES is on): start the lane servers one after the
other instead of all at once. Apply after patch_minion_lanes.py.

Why: SGLang sizes its KV pool from the GPU memory that is free when the process starts and again after its weights are
loaded (rest = available - total_at_start * (1 - mem_fraction_static)). Two servers started together race: when one has
already allocated its KV pool, the other's rest goes negative and it dies with "Not enough memory" (seen once in
about 18 lane runs on a Brev L40S: the server exited before READY). Started one after the other, each server sizes its pool from
what the previous ones left, so the outcome no longer depends on timing. The second lane's --mem-fraction-static then
applies to the memory left after the first server (TRACK2_LANE_MEMFRAC "0.36,0.55" gives the text lane about the pool
it had when both started together). Startup is about one server load longer (untimed; well inside the 300 s limit).

Usage: python patch_minion_lanes_seq.py <team dir>
"""
import os
import sys

team = sys.argv[1]
p = os.path.join(team, "serve.py")
s = open(p).read()
if "LANE_SEQUENTIAL" in s:
    print("already patched", p)
    sys.exit(0)
assert "lane_of_task" in s, "apply patch_minion_lanes.py first"
pairs = [
    ('LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"\n',
     'LANE_MPS = os.environ.get("TRACK2_LANE_MPS", "1") == "1"\n'
     '# team taktak (2026-11-22): start the lane servers one after the other (patch_minion_lanes_seq.py)\n'
     'LANE_SEQUENTIAL = os.environ.get("TRACK2_LANE_SEQUENTIAL", "1") == "1"\n'),
    ('''        servers = [start_server(j) for j in range(len(LANES))]
''', '''        servers = []
        for j in range(len(LANES)):
            servers.append(start_server(j))
            if LANE_SEQUENTIAL and j < len(LANES) - 1:   # this lane's KV pool is sized before the next one starts
                wait_ready(servers[-1], url=f"http://127.0.0.1:{lane_ports()[j]}")
                log(f"lane {j} ready; starting lane {j + 1}")
'''),
]
for old, new in pairs:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
