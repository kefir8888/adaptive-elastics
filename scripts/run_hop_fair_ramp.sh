#!/bin/bash
# FAIR comparison — SPRING staged ramp ONLY (run on box A, in parallel with the baseline on box B).
# k40 -> [GATE] -> k75 -> k93.3, warm-start-chained from s2; centered DR (in the config).
# Usage: run_hop_fair_ramp.sh <S2_RUN_DIR>
#
# *** SUPERVISE THE k=40 GATE ***: this driver runs the FULL ramp, but after the k=40 stage ENDs
# you (the agent) should rsync that run off-box and run `hop_spring_survival.py <k40_dir>` on the
# Mac — if the centered-DR spring still FALLS on nominal (<full episode / no hop), KILL this driver
# (pkill -f run_hop_fair_ramp) before it wastes k75+k93. If k40 PASSES, let it run to k93.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S2=${1:?usage: $0 <S2_RUN_DIR>}
LOG=$HOME/hop_fair_ramp.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -d "$S2/checkpoints" ] || { say "S2 missing: $S2 — abort"; exit 1; }
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ local k=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (k=$k steps=$steps restore=$restore, centered DR)"
  nice -n 19 ionice -c3 uv run pea-train --config configs/g1_hop_fair_spring.yaml \
    --spring_k "$k" --num_timesteps "$steps" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
stage 40   40000000 "$S2"  fair_spring_k40 || exit 1; K40=$STAGE_DIR
say "*** k40 done ($K40) — GATE NOW: rsync + hop_spring_survival.py; kill this driver if it still falls ***"
stage 75   40000000 "$K40" fair_spring_k75 || exit 1; K75=$STAGE_DIR
stage 93.3 60000000 "$K75" fair_spring_k93 || exit 1; K93=$STAGE_DIR
printf '%s\n' "$K93" > "$HOME/HOP_FAIR_RAMP_DONE"; say "RAMP DONE: spring=$K93"
