#!/bin/bash
# devrun_pod.sh <name> [KEY=VAL ...] -- devrun.sh without docker, for a pod set up by setup_pod.sh.
# REPS fresh processes (default 3), one timed job each (dev_harness.py), warm-up on ~/dev-warm. Results in ~/runs/<name>/r<i>/.
# TEAM=<dir> picks the team folder (default ~/my-team). One run at a time (flock ~/gpu.lock).
set -uo pipefail
source /etc/profile.d/track2.sh
[ -e "$HOME/runs/SKIP_$1" ] && { echo "skipped $1"; exit 0; }
name=$1; shift
base=$HOME/runs/$name; rm -rf "$base"; mkdir -p "$base"
team=$HOME/runs/.team_$name; rm -rf "$team"; cp -r "${TEAM:-$HOME/my-team}" "$team"
exec 9>"$HOME/gpu.lock"; flock 9
for i in $(seq 1 "${REPS:-3}"); do
  mkdir -p "$base/r$i"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 200 > "$base/r$i/gpu.csv" 2>/dev/null &
  smi=$!
  python3 "$HOME/tools/dev_harness.py" --team "$team" --dev "${DEVSET:-$HOME/dev-set}" --warm "$HOME/dev-warm" --out "$base/r$i" \
    --jobs 1 --delay "${DELAY:-0}" --env "$@" 2>&1 | sed -u "s/^/[r$i] /"
  kill $smi 2>/dev/null
done
rm -rf "$team"
echo "$*" > "$base/settings.txt"
python3 "$HOME/tools/summarize.py" "$base"
