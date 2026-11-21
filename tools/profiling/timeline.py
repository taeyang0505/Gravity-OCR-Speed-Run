"""timeline.py <t2log dir> [job]: 100 ms timeline of a job from the winner's TRACK2_LOG=2 records (L1 requests, L3 iterations, L4 ViT)."""
import json, glob, collections, sys
d = sys.argv[1]; jobname = sys.argv[2] if len(sys.argv) > 2 else "1-test"
L1 = [json.loads(l) for f in glob.glob(d + "/L1_*.jsonl") for l in open(f)]
L3 = [json.loads(l) for f in glob.glob(d + "/L3_*.jsonl") for l in open(f)]
L4 = [json.loads(l) for f in glob.glob(d + "/L4_*.jsonl") for l in open(f)]
job = [r for r in L1 if r["job"] == jobname]
T0 = min(r["send"] for r in job); T1 = max(r["done"] for r in job)
print("job span %.3f s, requests %d" % (T1 - T0, len(job)))
it = [x for x in L3 if T0 - 0.05 <= x["t0"] <= T1 + 0.01]
vb = [x for x in L4 if T0 - 0.05 <= x["t"] <= T1]
print("iterations", len(it), "vit batches", len(vb), "vit patches", sum(v["patches"] for v in vb))
B = 0.1; nb = int((T1 - T0) / B) + 1
agg = [collections.Counter() for _ in range(nb)]
for x in it:
    a = agg[min(int((x["t0"] - T0) / B), nb - 1)]
    a["it"] += 1; a["wq"] = max(a["wq"], x["wq"]); a["run"] = max(a["run"], x["run"])
    if x["mode"] != "DECODE":
        a["pf"] += 1; a["pf_tok"] += x.get("ext_tok", 0); a["pf_req"] += x["bs"]
    else:
        a["dec"] += 1; a["dec_ms"] += (x["t1"] - x["t0"]) * 1000; a["bs"] += x["bs"]
    a["commit"] += x.get("commit", 0); a["fin"] += x.get("fin", 0)
for v in vb:
    a = agg[min(int((v["t"] - T0) / B), nb - 1)]; a["vit_p"] += v["patches"]; a["vit_img"] += v["images"]
fc = vc = pc = 0
print("  t(s) iters dec ms/dec mean_bs | admitted(cum) pf_tok | run_max wq_max | commit fin(cum) | vit_img(cum) vit_patches")
for i, a in enumerate(agg):
    fc += a["fin"]; vc += a["vit_img"]; pc += a["pf_req"]
    md = a["dec_ms"] / a["dec"] if a["dec"] else 0; mb = a["bs"] / a["dec"] if a["dec"] else 0
    print("  %4.1f %4d %4d %6.1f %6.1f | %5d %6d | %4d %4d | %6d %4d | %4d %6d" % (
        i * B, a["it"], a["dec"], md, mb, pc, a["pf_tok"], a["run"], a["wq"], a["commit"], fc, vc, a["vit_p"]))
for r in sorted(job, key=lambda r: -r["done"])[:10]:
    print("  last %-32s %-8s rank %3d done %.3f" % (r["id"][:32], r["task"], r["rank"], r["done"] - T0))
