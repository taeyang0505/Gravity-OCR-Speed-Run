#!/bin/bash
# score.sh <run name>  -- official Track 2 scorer (the kit's scoring program, in the competition image) on the predictions of
# ~/runs/<name>/r*/predictions_1.jsonl: each fresh-process run counts as one timed test run, r1 doubles as the dev run.
# Prints the leaderboard columns (quality, per-task lowest run, eligible) and keeps scores.json / details.json in
# ~/runs/<name>/score/. Takes the GPU lock so that scoring (CPU heavy) never overlaps a timed run.
set -uo pipefail
name=$1; base=$HOME/runs/$name; w=$base/score_work
sudo rm -rf "$w" "$base/score"; mkdir -p "$w/res" "$w/ref" "$w/out" "$w/program"
cp -r "$HOME/starting_kit/programs/scoring_program/." "$w/program/"
python3 - "$base" "$w" <<'PY'
import glob, json, os, shutil, sys
base, w = sys.argv[1], sys.argv[2]
walls = []
runs = [d for d in sorted(glob.glob(base + "/r*/")) if os.path.exists(d + "predictions_1.jsonl")]   # failed runs left out
for i, d in enumerate(runs, 1):
    shutil.copy(d + "predictions_1.jsonl", f"{w}/res/predictions_test_{i}.jsonl")
    walls.append(json.load(open(d + "timing.json"))["walls"][0])
shutil.copy(runs[0] + "predictions_1.jsonl", f"{w}/res/predictions_dev.jsonl")
med = sorted(walls)[len(walls) // 2]
json.dump({"wall_time": med, "wall_times": walls, "test_runs": len(walls), "dev_wall_time": walls[0]}, open(f"{w}/res/metadata.json", "w"))
# DEVSET=<dir>: score against that crop set's targets (a subset of dev made by make_subsets.py); default ~/dev-set
targets = [json.loads(l) for l in open(os.path.join(os.environ.get("DEVSET") or os.path.expanduser("~/dev-set"), "targets.jsonl")) if l.strip()]
with open(f"{w}/ref/reference.jsonl", "w") as out:
    for split in ("test", "dev"):
        for t in targets:
            out.write(json.dumps({**t, "split": split}, ensure_ascii=False) + "\n")
PY
G=${GPU:-all}   # GPU=<n>: the lock and the 8 CPUs of GPU n (see devrun.sh)
if [ "$G" = all ]; then LOCK=$HOME/gpu.lock; CPUS=(); else LOCK=$HOME/gpu$G.lock; CPUS=(--cpuset-cpus "${CPUSET:-$((G * 8))-$((G * 8 + 7))}"); fi
exec 9>"$LOCK"; flock 9
docker run --rm -v "$w":/w track2-base:v1 chown -R 0:0 /w >/dev/null 2>&1
docker run --rm --network none "${CPUS[@]}" -v "$w/program":/app/program -v "$w/ref":/app/input/ref:ro -v "$w/res":/app/input/res:ro \
  -v "$w/out":/app/output -w /app/program track2-base:v1 python3 scoring.py > "$base/score.log" 2>&1
rc=$?
docker run --rm -v "$w":/w --entrypoint sh track2-base:v1 -c "chown -R $(id -u):$(id -g) /w && chmod -R u+rwX /w" >/dev/null 2>&1
mkdir -p "$base/score"; cp "$w/out/"*.json "$base/score/" 2>/dev/null; rm -rf "$w"
if [ $rc -ne 0 ] || [ ! -f "$base/score/scores.json" ]; then echo "SCORE $name: scorer failed (see $base/score.log)"; exit 1; fi
python3 - "$name" "$base/score/scores.json" "$base/score.log" <<'PY'
import json, re, sys
s = json.load(open(sys.argv[2]))
m = re.search(r"test quality per run: (\[.*?\])", open(sys.argv[3]).read())
print(f"SCORE {sys.argv[1]}: text {s['text']:.2f} formula {s['formula']:.2f} table {s['table']:.2f} | quality {s['quality']:.2f} "
      f"(per run {m.group(1) if m else '?'}) | eligible {s['eligible']} | missing {s['missing']}")
PY
