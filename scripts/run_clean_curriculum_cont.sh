#!/bin/bash
# CONTINUATION driver: WARM-START the clean curriculum from a pre-trained s1 (NO s1 retrain).
# Usage: run_clean_curriculum_cont.sh <S1_RUN_DIR>
#
# s1 (the non-spinning clean hop) was trained 2026-06-21 and validated (yaw 0.016 rad/s); it is
# uploaded to the box, and this driver chains the rest:
#   (relaxed gate sanity on s1) -> s2 height (--restore s1) -> s3 velocity -> s4 bounding.
# The relaxed gate (yaw + survival; drift is a benign soft diagnostic) doubles as a RESTORE check:
# it loads the uploaded s1 checkpoint and confirms it is non-spinning before spending s2's budget.
#
# Same box-safety scaffolding as run_clean_curriculum.sh: nice-19 ionice-c3, real-exit-code +
# min-duration + run-dir guards, sleep-on-failure. Launch detached; keep the && fail-safe watchdog up.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S1=${1:?usage: $0 <S1_RUN_DIR>}
LOG=$HOME/clean_curric.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "CONT from s1=$S1"
[ -d "$S1" ] || { say "S1 dir missing: $S1 — abort"; exit 1; }
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }

# Restore-sanity + relaxed gate on the uploaded s1: confirms the checkpoint LOADS and s1 is
# NON-SPINNING (yaw+survival) before committing s2's budget. Drift is reported, not gated.
GATE=$HOME/clean_s1_gate.log
say "GATE s1 (relaxed: yaw+survival) -> $GATE"
if timeout 1200 nice -n 19 uv run python scripts/hop_yaw_gate.py --run "$S1" \
     --command 0 0 0 --steps 600 --seeds 0 1 2 >"$GATE" 2>&1; then
  say "S1 GATE PASSED -> $(grep -E '\|mean yaw\||PASS' "$GATE" | tr '\n' ' ')"
else
  say "S1 GATE FAILED (rc=$?) — abort (bad upload or unexpected spin). Inspect $GATE"; exit 2
fi

latest(){ ls -dt "$HOME"/runs/*"$1"* 2>/dev/null | head -1; }
STAGE_DIR=""
stage(){ local cfg=$1 steps=$2 restore=$3 tag=$4 t0; t0=$(date +%s)
  say "START $tag (steps=$steps restore=${restore:-scratch})"
  nice -n 19 ionice -c3 uv run pea-train --config "$cfg" --num_timesteps "$steps" --restore "$restore" --output_dir "$HOME/runs" --suffix "$tag" >>"$LOG" 2>&1
  local rc=$? dur; dur=$(( $(date +%s) - t0 )); STAGE_DIR=$(latest "$tag")
  say "END $tag rc=$rc dur=${dur}s dir=$STAGE_DIR"
  if [ $rc -ne 0 ] || [ $dur -lt 120 ] || [ -z "$STAGE_DIR" ]; then say "FAIL $tag (rc=$rc dur=$dur) — sleep 60, abort"; sleep 60; return 1; fi
  return 0
}
stage configs/g1_clean_s2_height.yaml 100000000 "$S1" clean_s2 || exit 1; S2=$STAGE_DIR
stage configs/g1_clean_s3_vel.yaml    100000000 "$S2" clean_s3 || exit 1; S3=$STAGE_DIR
stage configs/g1_clean_s4_bound.yaml  140000000 "$S3" clean_s4 || exit 1; S4=$STAGE_DIR
say "clean curriculum (cont) done: s2=$S2  s3=$S3  s4=$S4"
printf '%s\n%s\n%s\n' "$S2" "$S3" "$S4" > "$HOME/CLEAN_CURRIC_DONE"
say "CONT CURRICULUM DONE (warm-started from s1=$S1)"
