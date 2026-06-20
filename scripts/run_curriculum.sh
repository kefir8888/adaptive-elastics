#!/bin/bash
# SAFE bounding/running curriculum (the FIXED version of the driver that wedged the box).
# Warm-starts from the hopper, ramps forward speed. Follows README "GPU-box safety".
# BEFORE launching, start an INDEPENDENT watchdog so a misbehaving driver can't run away:
#   nohup bash -c 'sleep ${BUDGET:-10800}; pkill -9 -f run_curriculum; pkill -9 -f pea-train; sudo poweroff' >~/watchdog.log 2>&1 &
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
LOG=$HOME/curriculum.log
BUDGET_S=${1:-10800}   # default 3h
START=$(date +%s); DEADLINE=$((START + BUDGET_S))
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "git reset -> origin/main + verify import; deadline in ${BUDGET_S}s"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED, abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ # cfg restore budget tag -> sets STAGE_DIR; returns 0 only on a REAL success; guards busy-spin
  local cfg=$1 restore=$2 budget=$3 tag=$4 t0; t0=$(date +%s)
  [ $(date +%s) -ge $DEADLINE ] && { say "DEADLINE, skip $tag"; return 2; }
  say "START $tag (restore=${restore:-scratch} budget=$budget)"
  nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --restore "$restore" --num_timesteps "$budget" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag — sleep 60 (anti-thrash)"; sleep 60; return 1; fi
  return 0
}
HOPPER=$(latest g1_hop_s1); [ -z "$HOPPER" ] && { say "no hopper to warm-start from, abort"; exit 1; }
stage configs/g1_bound_r1.yaml "$HOPPER" 100000000 r1 || { say "R1 failed, abort"; exit 1; }
PREV=$STAGE_DIR; i=0
while [ $(date +%s) -lt $DEADLINE ] && [ $i -lt 30 ]; do
  i=$((i+1))
  stage configs/g1_bound_r2.yaml "$PREV" 100000000 "r2_$i" || break
  PREV=$STAGE_DIR
done
echo "$PREV" > "$HOME/BEST_RUN"; touch "$HOME/CURRICULUM_DONE"; say "CURRICULUM DONE best=$PREV"
