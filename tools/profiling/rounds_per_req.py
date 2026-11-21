"""rounds_per_req.py <t2log dir> <out.json> [job]: per-request admission time and decode rounds, from the winner's TRACK2_LOG=2 records.
Prefill admits waiting requests in rank order (the scheduler keeps the waiting queue in rank order), so the k-th admitted
request is rank k. Rounds = DECODE iterations between its admission and its completion."""
import json, glob, sys, bisect, collections
d, out = sys.argv[1], sys.argv[2]; jobname = sys.argv[3] if len(sys.argv) > 3 else "1-test"
L1 = [json.loads(l) for f in glob.glob(d + "/L1_*.jsonl") for l in open(f)]
L3 = [json.loads(l) for f in glob.glob(d + "/L3_*.jsonl") for l in open(f)]
job = sorted([r for r in L1 if r["job"] == jobname], key=lambda r: r["rank"])
T0 = min(r["send"] for r in job); T1 = max(r["done"] for r in job)
it = [x for x in L3 if T0 - 0.01 <= x["t0"] <= T1 + 0.01]
adm = []                       # admission time of the k-th admitted request
for x in it:
    if x["mode"] != "DECODE":
        adm += [x["t1"]] * x["bs"]
dec = [x["t0"] for x in it if x["mode"] == "DECODE"]
res = {}
for r in job:
    a = adm[r["rank"]] if r["rank"] < len(adm) else None
    n = bisect.bisect_right(dec, r["done"]) - bisect.bisect_left(dec, a) if a else None
    res[r["id"]] = {"task": r["task"], "rank": r["rank"], "admit": round(a - T0, 3) if a else None, "done": round(r["done"] - T0, 3), "rounds": n}
json.dump(res, open(out, "w"))
by = collections.defaultdict(list)
for k, v in res.items():
    by[v["task"]].append(v["rounds"])
for t, v in by.items():
    v.sort(); print(t, "n", len(v), "rounds p50", v[len(v) // 2], "p90", v[int(len(v) * .9)], "max", v[-1], "sum", sum(v))
top = sorted(res.items(), key=lambda kv: -kv[1]["rounds"])[:15]
for k, v in top:
    print("  %-34s %-8s rank %3d admit %.3f done %.3f rounds %d" % (k[:34], v["task"], v["rank"], v["admit"], v["done"], v["rounds"]))
