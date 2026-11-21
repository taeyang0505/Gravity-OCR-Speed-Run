#!/usr/bin/env python3
"""summarize.py <run dir> [<run dir> ...]  -- one line per run: wall times, load time, GPU use during the timed job, CPU time,
and how many predictions equal the reference run's (~/runs/e1_ref/r1 by default, REF=<predictions.jsonl> to change)."""
import calendar, glob, json, os, sys, time


def gpu_rows(path):
    out = []
    for l in open(path):
        p = [x.strip() for x in l.split(",")]
        if len(p) < 4:
            continue
        try:
            ts = calendar.timegm(time.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S")) + float("0." + p[0].split(".")[1])
            out.append((ts, float(p[1]), float(p[2]), float(p[3])))
        except (ValueError, IndexError):
            pass
    return out


def preds(path):
    return {json.loads(l)["id"]: json.loads(l)["text"] for l in open(path) if l.strip()}


ref_path = os.environ.get("REF", os.path.expanduser("~/runs/e1_ref/r1/predictions_1.jsonl"))
ref = preds(ref_path) if os.path.exists(ref_path) else None
tz = time.timezone if time.localtime().tm_isdst == 0 else time.altzone      # nvidia-smi prints local time
for base in sys.argv[1:]:
    walls, loads, util, mem, power, cpu, same, empty = [], [], [], [], [], [], [], []
    for d in sorted(glob.glob(base + "/r*")):
        try:
            t = json.load(open(d + "/timing.json"))
        except Exception:
            continue
        walls.append(t["walls"][0]); loads.append(t["load_s"])
        j = (t.get("jobs") or [{}])[0]
        if j.get("cpu_s") is not None:
            cpu.append(j["cpu_s"])
        if os.path.exists(d + "/gpu.csv") and j.get("start_epoch"):
            rows = [r for r in gpu_rows(d + "/gpu.csv") if j["start_epoch"] <= r[0] + tz <= j["end_epoch"]]
            allr = gpu_rows(d + "/gpu.csv")
            if rows:
                util.append(sum(r[1] for r in rows) / len(rows)); power.append(sum(r[3] for r in rows) / len(rows))
            if allr:
                mem.append(max(r[2] for r in allr))
        if os.path.exists(d + "/predictions_1.jsonl"):
            p = preds(d + "/predictions_1.jsonl")
            empty.append(sum(1 for v in p.values() if not v))
            if ref is not None:
                same.append(sum(1 for k, v in p.items() if ref.get(k) == v))
    if not walls:
        print(f"RESULT {os.path.basename(base)}: no finished run"); continue
    w = sorted(walls); med = w[len(w) // 2]
    m = lambda xs, f="%.0f": (f % (sum(xs) / len(xs))) if xs else "-"
    energy = ("%.0f" % (sum(p * x for p, x in zip(power, walls)) / len(power))) if power and len(power) == len(walls) else "-"
    print(f"RESULT {os.path.basename(base)}: wall {' '.join('%.2f' % x for x in walls)} | median {med:.2f} s | load {max(loads):.0f} s | "
          f"GPU util {m(util)}% power {m(power)} W energy {energy} J peak mem {m(mem)} MiB | CPU {m(cpu, '%.1f')} core-s | "
          f"same as ref {'/'.join(map(str, same)) if same else '-'} of 402 | empty {'/'.join(map(str, empty))}")
