#!/bin/bash
# SAFE driver for the apex-controlled FAIR hop spring comparison (2026-06-20 re-run).
# Trains S1 (elicit) if absent, then warm-starts the apex-capped baseline + spring arms.
# Follows README "GPU-box safety": LOW priority (sshd never starves), real-exit-code +
# min-duration + run-dir guards, sleep-on-failure. Launch detached, watch the first eval,
# and have the independent watchdog up (see the header of run_curriculum.sh).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
LOG=$HOME/hop_compare.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "git reset -> origin/main + verify import"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — stale/broken code, abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ # cfg restore tag -> sets STAGE_DIR; returns 0 only on a REAL (>120s, dir-created) success
  local cfg=$1 restore=$2 tag=$3 t0; t0=$(date +%s)
  say "START $tag (restore=${restore:-scratch})"
  if [ -n "$restore" ]; then
    nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  else
    nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  fi
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag (rc=$rc dur=$dur) — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
HOPPER=$(latest g1_hop_s1)
if [ -z "$HOPPER" ] || [ ! -d "$HOPPER/checkpoints" ]; then
  stage configs/g1_hop_s1.yaml "" g1_hop_s1 || exit 1; HOPPER=$STAGE_DIR
fi
say "hopper=$HOPPER"
stage configs/g1_hop_baseline.yaml "$HOPPER" fair_base  || exit 1; BASE_DIR=$STAGE_DIR
stage configs/g1_hop_spring.yaml   "$HOPPER" fair_spring || exit 1; SPR_DIR=$STAGE_DIR
say "baseline=$BASE_DIR"; say "spring=$SPR_DIR"
# Record BOTH run dirs (distinct tags, no ambiguous `latest fair`) for the manual energy compare:
#   uv run python scripts/hop_energy_compare.py <baseline> <spring>
printf '%s\n%s\n' "$BASE_DIR" "$SPR_DIR" > "$HOME/HOP_COMPARE_DONE"; say "HOP COMPARE DONE"
