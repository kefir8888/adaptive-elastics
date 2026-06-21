#!/bin/bash
# FAIR comparison — BASELINE arm ONLY (run on box B, in parallel with the ramp on box A).
# Warm-start from s2; centered DR (in the config). Usage: run_hop_fair_baseline.sh <S2_RUN_DIR>
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S2=${1:?usage: $0 <S2_RUN_DIR>}
LOG=$HOME/hop_fair_baseline.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -d "$S2/checkpoints" ] || { say "S2 missing: $S2 — abort"; exit 1; }
git fetch origin >>"$LOG" 2>&1 && git reset --hard origin/main >>"$LOG" 2>&1 || { say "git reset FAILED"; exit 1; }
nice -n 19 uv run python -c "from pea.env import make_env" >>"$LOG" 2>&1 || { say "import FAILED — abort"; exit 1; }
t0=$(date +%s); say "START fair_baseline (80M, restore=$S2, centered DR)"
nice -n 19 ionice -c3 uv run pea-train --config configs/g1_hop_fair_baseline.yaml \
  --num_timesteps 80000000 --restore "$S2" --output_dir "$HOME/runs" --suffix fair_baseline >>"$LOG" 2>&1
rc=$?; dur=$(( $(date +%s) - t0 )); D=$(ls -dt "$HOME"/runs/*fair_baseline* 2>/dev/null | head -1)
say "END fair_baseline rc=$rc dur=${dur}s dir=$D"
if [ $rc -eq 0 ] && [ $dur -ge 120 ] && [ -n "$D" ]; then
  printf '%s\n' "$D" > "$HOME/HOP_FAIR_BASELINE_DONE"; say "BASELINE DONE: $D"
else say "FAIL (rc=$rc dur=$dur)"; exit 1; fi
