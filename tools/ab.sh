#!/bin/bash
# ab.sh <rounds> <tag> "<label>|<team dir>|KEY=VAL KEY=VAL" ... : interleaved timing (one fresh-process run per config per
# round, configs alternated every round), so that drifts and occasional slow runs hit every config alike.
# Appends "tag label round wall" to ~/ab.tsv and prints per-label median / mean / min at the end.
rounds=$1; tag=$2; shift 2
for r in $(seq 1 "$rounds"); do
  for spec in "$@"; do
    IFS='|' read -r label team envs <<< "$spec"
    n="${tag}_${label}_r$r"
    TEAM=$team REPS=1 bash ~/tools/devrun.sh "$n" $envs > /dev/null 2>&1
    w=$(python3 -c "import json;print(json.load(open('$HOME/runs/$n/r1/timing.json'))['walls'][0])" 2>/dev/null || echo NA)
    echo "$tag $label $r $w" >> ~/ab.tsv
    echo "$tag $label $r $w"
  done
done
python3 - "$tag" <<'PY'
import sys, statistics, collections
tag = sys.argv[1]; d = collections.defaultdict(list); bad = collections.Counter()
for l in open(__import__("os").path.expanduser("~/ab.tsv")):
    t, lab, r, w = l.split()
    if t != tag: continue
    if w == "NA": bad[lab] += 1; continue
    d[lab].append(float(w))
for lab, v in d.items():
    v.sort()
    print(f"AB {tag} {lab}: n={len(v)} fail={bad[lab]} median={statistics.median(v):.3f} mean={statistics.mean(v):.3f} min={v[0]:.3f} all={' '.join(f'{x:.2f}' for x in v)}")
PY
