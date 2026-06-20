#!/bin/bash
# SAFE driver: STAGED stiffness ramp for the hop spring arm (NO domain randomization).
# Warm-start-chains fixed-k runs  S1(k=0) -> k=40 -> k=75 -> k=106.1  so the policy
# migrates gradually to full strength and ends TUNED to k=106 (vs the per-episode-DR
# robustness compromise), on the fast fixed-k SpringWrapper path. The final k=106 run dir
# is the comparison spring arm (its saved config.yaml has k_dr off, k=106.1) read directly
# by scripts/hop_energy_compare.py against the existing baseline.
#
# Follows README "GPU-box safety": LOW priority (sshd never starves), real-exit-code +
# min-duration + run-dir guards, sleep-on-failure. Launch detached, watch the first eval,
# keep the independent watchdog up. Reuses the existing S1 elicit checkpoint on the box.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
LOG=$HOME/spring_ramp.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "git reset -> origin/main + verify import"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — stale/broken code, abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ # k steps restore tag -> sets STAGE_DIR; returns 0 only on a REAL (>120s, dir-created) success
  local k=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (k=$k steps=$steps restore=${restore:-scratch})"
  nice -n 19 ionice -c3 uv run pea-train --config configs/g1_hop_spring_fixed.yaml \
    --spring_k "$k" --num_timesteps "$steps" --restore "$restore" \
    --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag (rc=$rc dur=$dur) — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
S1=$(latest g1_hop_s1)
[ -n "$S1" ] && [ -d "$S1/checkpoints" ] || { say "S1 checkpoint missing ($S1) — abort"; exit 1; }
say "S1=$S1"
stage 40    45000000 "$S1"  ramp_k40  || exit 1; K40=$STAGE_DIR
stage 75    45000000 "$K40" ramp_k75  || exit 1; K75=$STAGE_DIR
stage 106.1 70000000 "$K75" ramp_k106 || exit 1; K106=$STAGE_DIR
say "ramp done: k40=$K40  k75=$K75  k106=$K106"
printf '%s\n' "$K106" > "$HOME/HOP_RAMP_DONE"
say "HOP RAMP DONE — compare:  uv run python scripts/hop_energy_compare.py <baseline> $K106"
