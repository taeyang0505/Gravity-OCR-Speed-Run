#!/bin/bash
# prof_run.sh <name> <team dir> [KEY=VAL ...]: one fresh-process timed job; nsys collects only during the timed job
# (nsys launch ... ; nsys start before the job, nsys stop right after DONE, while the server is still alive).
set -uo pipefail
source /etc/profile.d/track2.sh
name=$1; src=$2; shift 2
out=$HOME/prof/$name; rm -rf $out; mkdir -p $out
team=$HOME/runs/.team_prof_$name; rm -rf $team; cp -r $src $team
serve=$(grep '^serve:' $src/team.yaml | cut -d: -f2- | sed 's/^ *//')
envpart=${serve%%python3 serve.py*}
echo "serve: ${envpart}nsys launch --session-new=t2prof -t cuda --cuda-graph-trace=node --trace-fork-before-exec=true python3 serve.py" > $team/team.yaml
cat $team/team.yaml
python3 - "$out" <<'PY'
import sys, re
out = sys.argv[1]
s = open("/root/tools/dev_harness.py").read()
s = s.replace("proc.wait(timeout=30)", "proc.wait(timeout=120)")
s = re.sub(r'^subprocess.run\("pkill.*$', "pass", s, flags=re.M)
s = s.replace("    os.rename(hidden, job)\n",
  f"    subprocess.run(['nsys', 'start', '--session=t2prof', '-o', '{out}/rep', '--force-overwrite=true'], check=True, env=env)\n"
  "    time.sleep(1.0)\n    c0 = cpu_usec(); t0 = time.time_ns()\n    os.rename(hidden, job)\n")
s = s.replace("    wall = (os.stat(done).st_mtime_ns - t0) / 1e9\n",
  "    wall = (os.stat(done).st_mtime_ns - t0) / 1e9\n    time.sleep(0.3)\n"
  "    subprocess.run(['nsys', 'stop', '--session=t2prof'], check=True, env=env)\n"
  f"    print('nsys sessions:', subprocess.run(['nsys', 'sessions', 'list'], env=env, capture_output=True, text=True).stdout, flush=True)\n"
  f"    for _ in range(600):\n        if os.path.exists('{out}/rep.nsys-rep'): break\n        time.sleep(0.5)\n"
  f"    time.sleep(5)\n")
open("/root/tools/dev_harness_prof.py", "w").write(s)
PY
grep -n "nsys" $HOME/tools/dev_harness_prof.py
exec 9>"$HOME/gpu.lock"; flock 9
python3 $HOME/tools/dev_harness_prof.py --team $team --dev $HOME/dev-set --warm $HOME/dev-warm --out $out --jobs 1 --env "$@" 2>&1 | tail -6
pkill -9 -f sglang 2>/dev/null
nsys shutdown --session=t2prof >/dev/null 2>&1
ls -la $out
nsys export --type sqlite -o $out/rep.sqlite $out/rep.nsys-rep > /dev/null 2>&1 && echo EXPORT_OK
rm -rf $team
