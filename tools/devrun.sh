#!/bin/bash
# devrun.sh <name> [KEY=VAL ...]  -- run a team folder in the competition image with the fast dev harness: REPS fresh
# processes (default 3), one timed job each, warm-up on ~/dev-warm (images that are not the timed ones).
# Results in ~/runs/<name>/r<i>/ (predictions_1.jsonl, team.log, timing.json, gpu.csv). TEAM=<dir> picks the team folder
# (default ~/my-team). One run at a time on this GPU: everything goes through the lock ~/gpu.lock (flock), so runs started
# by different people queue up instead of disturbing each other's timing.
# DEVSET=<dir>: time another crop set than ~/dev-set (e.g. ~/dev-noodb: dev without the 50 'odb' text crops, closer to the
# size mix of the hidden test set; timing only, quality is always scored on the full dev set).
# Network off and /dev/shm 16g as on the eval server; runs as root, unlike the real harness -> final check: local_run.py.
set -uo pipefail
name=$1; shift
base=$HOME/runs/$name; rm -rf "$base"; mkdir -p "$base"
team=$HOME/runs/.team_$name; rm -rf "$team"; cp -r "${TEAM:-$HOME/my-team}" "$team"
# GPU=<n> (machines with several GPUs): this run uses GPU n only, 8 CPUs of its own (CPUSET, default n*8..n*8+7) and the
# lock ~/gpu<n>.lock, so one run per GPU can go at the same time. Unset: the whole machine, ~/gpu.lock.
G=${GPU:-all}
if [ "$G" = all ]; then LOCK=$HOME/gpu.lock; DGPU=(--device nvidia.com/gpu=all); SMI=()
else LOCK=$HOME/gpu$G.lock; DGPU=(--gpus "device=$G" --cpuset-cpus "${CPUSET:-$((G * 8))-$((G * 8 + 7))}"); SMI=(-i "$G"); fi
exec 9>"$LOCK"; flock 9
for i in $(seq 1 "${REPS:-3}"); do
  mkdir -p "$base/r$i"
  nvidia-smi "${SMI[@]}" --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -lms 200 > "$base/r$i/gpu.csv" 2>/dev/null &
  smi=$!
  docker run --rm --network none --shm-size=16g "${DGPU[@]}" \
    -v "$team":/team:ro -v "${DEVSET:-$HOME/dev-set}":/dev-set:ro -v "$HOME/dev-warm":/dev-warm:ro -v "$base/r$i":/out \
    -v "$HOME/tools":/tools:ro --entrypoint python3 track2-base:v1 /tools/dev_harness.py --team /team --dev /dev-set \
    --warm /dev-warm --out /out --jobs 1 --delay "${DELAY:-0}" --env "$@" 2>&1 | sed -u "s/^/[r$i] /"
  kill $smi 2>/dev/null
done
sudo chown -R "$(id -u):$(id -g)" "$base" 2>/dev/null || true
rm -rf "$team"
echo "$*" > "$base/settings.txt"
python3 "$HOME/tools/summarize.py" "$base"
