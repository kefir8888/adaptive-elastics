#!/bin/bash
# SAFE driver: the CLEAN (non-spinning) hop curriculum (2026-06-21).
# Re-elicits a NON-SPINNING hop FROM SCRATCH (stage 1: strong heading hold + leg_symmetry),
# then warm-start-chains: height control (s2) -> velocity tracking (s3) -> alternating-foot
# support exchange with swing-foot clearance (s4). Fixes the pervasive +2-3.8 rad/s spin that
# contaminated the whole 2026-06-20 hop lineage (inherited from a diagonal-stance S1).
#
# *** GATE STAGE 1 HARD ***: before trusting s2-s4, confirm s1 has |yaw| < 0.15 rad/s at zero
# command (no spin), ~0 net drift, symmetric stance, full survival. If it still spins, STOP and
# debug — do not build downstream on a spinning base. (Measure with the sample_command-patch
# yaw probe, not the raw rollout command override.)
#
# Follows README "GPU-box safety": nice-19, real-exit-code + min-duration + run-dir guards,
# sleep-on-failure. Launch detached, keep the && fail-safe watchdog up.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
LOG=$HOME/clean_curric.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "git reset -> origin/main + verify import"
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ # cfg steps restore tag -> sets STAGE_DIR; returns 0 only on a REAL (>120s, dir-created) success
  local cfg=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (steps=$steps restore=${restore:-scratch})"
  if [ -n "$restore" ]; then
    nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --num_timesteps "$steps" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  else
    nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --num_timesteps "$steps" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  fi
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag (rc=$rc dur=$dur) — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
stage configs/g1_clean_s1.yaml       200000000 ""     clean_s1 || exit 1; S1=$STAGE_DIR
stage configs/g1_clean_s2_height.yaml 100000000 "$S1" clean_s2 || exit 1; S2=$STAGE_DIR
stage configs/g1_clean_s3_vel.yaml    100000000 "$S2" clean_s3 || exit 1; S3=$STAGE_DIR
stage configs/g1_clean_s4_bound.yaml  140000000 "$S3" clean_s4 || exit 1; S4=$STAGE_DIR
say "clean curriculum done: s1=$S1  s2=$S2  s3=$S3  s4=$S4"
printf '%s\n%s\n%s\n%s\n' "$S1" "$S2" "$S3" "$S4" > "$HOME/CLEAN_CURRIC_DONE"
say "CLEAN CURRICULUM DONE — gate s1 on |yaw|<0.15 before trusting s2-s4"
