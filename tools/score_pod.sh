#!/bin/bash
# score_pod.sh <run name> -- score.sh without docker: the kit's scoring program on ~/runs/<name>/r*/predictions_1.jsonl.
# DEVSET=<dir> scores against that set's targets (default ~/dev-set). Prints the SCORE line; keeps details.json in
# ~/runs/<name>/score/ (hold_score.py reads it for the 100 held-out crops).
set -uo pipefail
source /etc/profile.d/track2.sh
name=$1; base=$HOME/runs/$name; w=$base/score_work
rm -rf "$w" "$base/score"; mkdir -p "$w/res" "$w/ref" "$w/out"
python3 - "$base" "$w" <<'PY'
import glob, json, os, shutil, sys
base, w = sys.argv[1], sys.argv[2]
walls = []
runs = sorted(glob.glob(base + "/r*/"))
for i, d in enumerate(runs, 1):
    shutil.copy(d + "predictions_1.jsonl", f"{w}/res/predictions_test_{i}.jsonl")
    walls.append(json.load(open(d + "timing.json"))["walls"][0])
shutil.copy(runs[0] + "predictions_1.jsonl", f"{w}/res/predictions_dev.jsonl")
json.dump({"wall_time": sorted(walls)[len(walls) // 2], "wall_times": walls, "test_runs": len(walls), "dev_wall_time": walls[0]},
          open(f"{w}/res/metadata.json", "w"))
targets = [json.loads(l) for l in open(os.path.join(os.environ.get("DEVSET") or os.path.expanduser("~/dev-set"), "targets.jsonl")) if l.strip()]
with open(f"{w}/ref/reference.jsonl", "w") as out:
    for split in ("test", "dev"):
        for t in targets:
            out.write(json.dumps({**t, "split": split}, ensure_ascii=False) + "\n")
PY
prog=$HOME/starting_kit/programs/scoring_program
( cd "$prog" && TRACK2_REF="$w/ref/reference.jsonl" TRACK2_RES="$w/res" TRACK2_SCORE_OUT="$w/out" python3 scoring.py ) > "$base/score.log" 2>&1
mkdir -p "$base/score"; cp "$w/out/"*.json "$base/score/" 2>/dev/null
[ -f "$base/score/scores.json" ] || { echo "SCORE $name: scorer failed (see $base/score.log)"; exit 1; }
python3 - "$name" "$base/score/scores.json" "$base/score.log" <<'PY'
import json, re, sys
s = json.load(open(sys.argv[2]))
m = re.search(r"test quality per run: (\[.*?\])", open(sys.argv[3]).read())
print(f"SCORE {sys.argv[1]}: text {s['text']:.2f} formula {s['formula']:.2f} table {s['table']:.2f} | quality {s['quality']:.2f} "
      f"(per run {m.group(1) if m else '?'}) | eligible {s['eligible']} | missing {s['missing']}")
PY
