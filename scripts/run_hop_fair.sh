#!/bin/bash
# FAIR hop-spring energy comparison on the CLEAN (non-spinning) base. Warm-starts BOTH arms
# from clean s2 (the in-place apex hopper), at MATCHED apex 0.12 + cadence 1.8 Hz:
#   - baseline (no spring): one warm-start run.
#   - spring (knee pogo k=93.3): staged stiffness ramp k 40 -> 75 -> 93.3 (warm-start-chained),
#     because the full-strength inject destabilizes (the prior campaign's validated recipe).
# Energy is OFF in training (post-hoc); compare with scripts/hop_energy_compare.py.
# Usage: run_hop_fair.sh <S2_RUN_DIR>
# Same box-safety scaffolding as the curriculum drivers: nice-19 ionice-c3, real-exit-code +
# min-duration + run-dir guards, sleep-on-failure. Launch detached; keep the && watchdog up.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S2=${1:?usage: $0 <S2_RUN_DIR>}
LOG=$HOME/hop_fair.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -d "$S2/checkpoints" ] || { say "S2 checkpoint missing: $S2 — abort"; exit 1; }
say "FAIR comparison warm-started from s2=$S2"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ local cfg=$1 extra=$2 steps=$3 restore=$4 tag=$5 t0; t0=$(date +%s)
  say "START $tag (steps=$steps restore=${restore:-scratch} ${extra})"
  nice -n 19 ionice -c3 uv run pea-train --config "$cfg" $extra --num_timesteps "$steps" \
    --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag (rc=$rc dur=$dur) — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
# Baseline (no spring), warm-start from s2.
stage configs/g1_hop_fair_baseline.yaml "" 80000000 "$S2" fair_baseline || exit 1; BASE=$STAGE_DIR
# Spring staged stiffness ramp (k 40 -> 75 -> 93.3), warm-start from s2.
stage configs/g1_hop_fair_spring.yaml "--spring_k 40"   40000000 "$S2"  fair_spring_k40 || exit 1; K40=$STAGE_DIR
stage configs/g1_hop_fair_spring.yaml "--spring_k 75"   40000000 "$K40" fair_spring_k75 || exit 1; K75=$STAGE_DIR
stage configs/g1_hop_fair_spring.yaml "--spring_k 93.3" 60000000 "$K75" fair_spring_k93 || exit 1; K93=$STAGE_DIR
say "FAIR done: baseline=$BASE  spring(k93)=$K93"
printf '%s\n%s\n' "$BASE" "$K93" > "$HOME/HOP_FAIR_DONE"
say "HOP FAIR DONE — compare: uv run python scripts/hop_energy_compare.py $BASE $K93"
