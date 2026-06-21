#!/bin/bash
# RUNNING: knee-pogo spring on the s4 bounding gait, 2-stage ramp k64 -> 127.7, warm-chained from s4.
# After stage 1 (k64) ENDs, GATE: rsync + hop_failure_diag at cmd 0.7,0,0 -> must still move forward
# (~0.4 m/s) and survive a seed; if it collapses to standing/falls, kill (pkill -f run_bound_spring).
# Usage: run_bound_spring.sh <S4_RUN_DIR>
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S4=${1:?usage: $0 <S4_RUN_DIR>}
LOG=$HOME/bound_spring.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -d "$S4/checkpoints" ] || { say "S4 missing: $S4 — abort"; exit 1; }
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ local k=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (k=$k steps=$steps restore=$restore)"
  nice -n 19 ionice -c3 uv run pea-train --config configs/g1_bound_spring.yaml \
    --spring_k "$k" --num_timesteps "$steps" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
stage 64    40000000 "$S4"  bound_spring_k64 || exit 1; K64=$STAGE_DIR
say "*** k64 done ($K64) — GATE NOW: rsync + hop_failure_diag cmd 0.7,0,0; kill if it stops moving/collapses ***"
stage 127.7 60000000 "$K64" bound_spring_k128 || exit 1; K128=$STAGE_DIR
printf '%s\n' "$K128" > "$HOME/BOUND_SPRING_DONE"; say "RAMP DONE: spring=$K128"
