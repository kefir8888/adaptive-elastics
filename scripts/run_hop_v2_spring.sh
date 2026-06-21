#!/bin/bash
# OPTION A: knee-pogo spring on the v2 base, staged ramp k40 -> 75 -> 95.8, warm-chained from v2.
# After the k40 stage ENDs, GATE: rsync + hop_spring_survival.py on the k40 dir. If the spring
# collapses v2 (falls full-episode / no hop on nominal), kill this driver (pkill -f run_hop_v2_spring)
# and skip to Option B. If it survives, let the ramp finish.
# Usage: run_hop_v2_spring.sh <V2_RUN_DIR>
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
V2=${1:?usage: $0 <V2_RUN_DIR>}
LOG=$HOME/hop_v2_spring.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -d "$V2/checkpoints" ] || { say "V2 missing: $V2 — abort"; exit 1; }
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ local k=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (k=$k steps=$steps restore=$restore)"
  nice -n 19 ionice -c3 uv run pea-train --config configs/g1_hop_v2_spring.yaml \
    --spring_k "$k" --num_timesteps "$steps" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
stage 40   40000000 "$V2"  v2spring_k40 || exit 1; K40=$STAGE_DIR
say "*** k40 done ($K40) — GATE NOW: rsync + hop_spring_survival.py; kill this driver if it collapses v2 ***"
stage 75   40000000 "$K40" v2spring_k75 || exit 1; K75=$STAGE_DIR
stage 95.8 60000000 "$K75" v2spring_k96 || exit 1; K96=$STAGE_DIR
printf '%s\n' "$K96" > "$HOME/HOP_V2_SPRING_DONE"; say "RAMP DONE: spring=$K96"
