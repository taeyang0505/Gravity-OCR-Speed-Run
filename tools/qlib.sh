#!/bin/bash
# qlib.sh (team taktak, Brev L40S, docker): source it in a queue script.
#   T <name> <team dir> <reps> [KEY=VAL ...]   timing only (devrun.sh in track2-base:v1) + the 'last' list
#   Q <name> <team dir> [KEY=VAL ...]          quality of runs/<name> (score.sh + held-out) + an unseen-350 run and score
#   W <tmux session>...                        wait until those sessions are gone
W() { for s in "$@"; do while tmux has-session -t "$s" 2>/dev/null; do sleep 5; done; done; }
T() { n=$1; team=$2; reps=$3; shift 3
  TEAM=$team REPS=$reps bash ~/tools/devrun.sh "$n" "$@" 2>&1 | grep -E "RESULT|Traceback|Error|exited" | cut -c1-260
  grep -h -o '"last":.*' ~/runs/$n/r1/team.log 2>/dev/null | tail -1 | cut -c1-200; }
Q() { n=$1; team=$2; shift 2
  bash ~/tools/score.sh "$n" | cut -c1-220
  python3 ~/tools/hold_score.py ~/runs/$n/score/details.json | sed "s/.*| hold:/   hold:/" | head -1
  TEAM=$team REPS=1 DEVSET=$HOME/unseen_r7 bash ~/tools/devrun.sh "${n}_un" "$@" > /dev/null 2>&1
  DEVSET=$HOME/unseen_r7 bash ~/tools/score.sh "${n}_un" | sed "s/^SCORE/UNSEEN/" | cut -c1-200; }
