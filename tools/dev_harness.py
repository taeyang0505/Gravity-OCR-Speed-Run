#!/usr/bin/env python3
"""Fast development stand-in for the Track 2 harness (2026-11-19). Runs INSIDE the track2-base:v1 container.

Starts the team's serve command like the organisers' harness (cwd = team folder, $TRACK2_DIR, $TRACK2_DEV_DIR, $HOME),
waits for READY (load time is reported, limit 300 s), then hands over the dev set as a job N times in the SAME process and
reports each wall time (job folder appears -> DONE mtime, as the real harness measures it). Predictions and the team log are
kept in --out. No scoring here: compare the predictions with a reference run (compare_preds.py) or score separately.
The official check stays `python3 local_run.py` (fresh process per run, unprivileged user, scorer).

  python3 dev_harness.py --team /team --dev /dev-set --out /out [--jobs 3] [--env KEY=VAL ...]
"""
import argparse, json, os, shutil, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--team", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--warm", default=None, help="folder given to the team as $TRACK2_DEV_DIR (default: --dev). Use a set whose "
                "images differ from --dev (make_warm_set.py): on the eval server the warm-up images are not the timed ones.")
ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait after READY before the job (profiler attach)")
ap.add_argument("--env", nargs="*", default=[])
a = ap.parse_args()
work = "/tmp/track2"
shutil.rmtree(work, ignore_errors=True)
os.makedirs(os.path.join(work, "jobs")); os.makedirs(os.path.join(work, "home")); os.makedirs(a.out, exist_ok=True)
serve = [l.split(":", 1)[1].strip() for l in open(os.path.join(a.team, "team.yaml")) if l.startswith("serve:")][0]
env = dict(os.environ, PYTHONUNBUFFERED="1", TRACK2_DIR=work, TRACK2_DEV_DIR=a.warm or a.dev, HOME=os.path.join(work, "home"),
           TMPDIR=os.path.join(work, "home"))
env.update(kv.split("=", 1) for kv in a.env)
logf = open(os.path.join(a.out, "team.log"), "w")
t0 = time.time()
proc = subprocess.Popen(serve, shell=True, cwd=a.team, env=env, stdout=logf, stderr=subprocess.STDOUT)
while not os.path.exists(os.path.join(work, "READY")):
    if proc.poll() is not None:
        sys.exit(f"team process exited with code {proc.returncode} before READY; see {a.out}/team.log")
    if time.time() - t0 > 600:
        proc.kill(); sys.exit("no READY after 600 s")
    time.sleep(0.05)
load = time.time() - t0
print(f"READY after {load:.1f} s (limit 300 s)", flush=True)
time.sleep(a.delay)
def cpu_usec():      # CPU time used by this container so far (cgroup v2)
    try:
        return int(next(l.split()[1] for l in open("/sys/fs/cgroup/cpu.stat") if l.startswith("usage_usec")))
    except Exception:
        return None


walls, jobs = [], []
for n in range(1, a.jobs + 1):
    hidden = os.path.join(work, f".staging-{n}")
    shutil.copytree(a.dev, hidden, ignore=shutil.ignore_patterns("targets.jsonl"))
    job = os.path.join(work, "jobs", f"{n}-test")
    c0 = cpu_usec()
    t0 = time.time_ns()
    os.rename(hidden, job)
    done = os.path.join(job, "DONE")
    while not os.path.exists(done):
        if proc.poll() is not None:
            sys.exit(f"team process exited with code {proc.returncode} during job {n}")
        time.sleep(0.002)
    wall = (os.stat(done).st_mtime_ns - t0) / 1e9
    walls.append(wall)
    c1 = cpu_usec()
    jobs.append({"start_epoch": t0 / 1e9, "end_epoch": os.stat(done).st_mtime_ns / 1e9, "wall": wall,
                 "cpu_s": None if c0 is None or c1 is None else (c1 - c0) / 1e6})
    shutil.copy(os.path.join(job, "predictions.jsonl"), os.path.join(a.out, f"predictions_{n}.jsonl"))
    print(f"job {n}: wall {wall:.3f} s", flush=True)
    time.sleep(1.0)
open(os.path.join(work, "STOP"), "w").close()
try:
    proc.wait(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()
subprocess.run("pkill -9 -f sglang 2>/dev/null; true", shell=True)
json.dump({"load_s": load, "walls": walls, "median": sorted(walls)[len(walls) // 2], "jobs": jobs},
          open(os.path.join(a.out, "timing.json"), "w"))
print(f"walls {['%.3f' % w for w in walls]}  median {sorted(walls)[len(walls) // 2]:.3f} s  load {load:.1f} s", flush=True)
