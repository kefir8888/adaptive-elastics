#!/bin/bash
# SAFE driver: DIRECTIONAL-JUMP curriculum. Warm-start from the in-place hopper (S1),
# then introduce a commanded velocity SLOWLY in two stages (D0 small band -> D1 final
# low-speed band), commanding BOTH linear (x,y) and angular (yaw) velocity with boosted
# tracking for controllability (NOT high speed). Config-only on G1JoystickHop.
# Follows README "GPU-box safety": LOW priority, real-exit-code + min-duration + run-dir
# guards, sleep-on-failure. Launch detached; keep the independent watchdog up.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
LOG=$HOME/dir_jumps.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "git reset -> origin/main + verify import"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ # cfg steps restore tag -> sets STAGE_DIR; returns 0 only on a REAL (>120s, dir) success
  local cfg=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (steps=$steps restore=${restore:-scratch})"
  nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --num_timesteps "$steps" \
    --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
S1=$(latest g1_hop_s1)
[ -n "$S1" ] && [ -d "$S1/checkpoints" ] || { say "S1 checkpoint missing ($S1) — abort"; exit 1; }
say "S1=$S1"
stage configs/g1_hop_dir_d0.yaml 45000000 "$S1"  dir_d0 || exit 1; D0=$STAGE_DIR
stage configs/g1_hop_dir_d1.yaml 45000000 "$D0"  dir_d1 || exit 1; D1=$STAGE_DIR
say "dir-jumps done: d0=$D0  d1=$D1"
printf '%s\n' "$D1" > "$HOME/DIR_JUMPS_DONE"; say "DIR JUMPS DONE (final policy: $D1)"
