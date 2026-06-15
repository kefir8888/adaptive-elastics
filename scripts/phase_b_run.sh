#!/usr/bin/env bash
# Phase B for the Go1 load-carrying program. Detached; survives disconnect; does NOT halt.
#   1. 2nd seed @ 6 kg      (is the energy win seed-luck?)
#   2. 15 kg CURRICULUM     (warm-start from the 6 kg checkpoints via --restore; raised tau0 cap)
#   3. ROUGH terrain @ 6 kg (does the win survive rough ground?)
# Order = highest-value first (rough has the most env-compat risk). Each stage trains
# baseline + adaptive, then runs the capacity/energy-vs-load sweep (with forward speed) on both.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/adaptive-elastics" || exit 1
RUNS="$HOME/runs"; mkdir -p "$RUNS"
ts(){ date "+%F %T"; }
say(){ echo "[$(ts)] $*"; }

train(){   # $1=config $2=suffix [$3=restore_run_dir]
  local extra=""; [ -n "${3:-}" ] && extra="--restore $3"
  say "TRAIN $(basename "$1") suffix=$2 $extra"
  uv run pea-train --config "$1" --output_dir "$RUNS" --suffix "$2" $extra \
      >> "$HOME/train_${2}_$(basename "$1" .yaml).log" 2>&1
  say "TRAIN $(basename "$1") exit=$?"
}
find_run(){ ls -dt "$RUNS"/*_"$1" 2>/dev/null | head -1; }
track(){ uv run python - "$1" <<'PY'
import json,sys,os
try:
    d=json.loads(open(os.path.join(sys.argv[1],"metrics.jsonl")).read().strip().split("\n")[-1])
    print(round(d.get("eval/episode_reward/tracking_lin_vel",0.0),1))
except Exception: print(0.0)
PY
}
evalrun(){  # $1=run_dir $2=logname
  [ -z "$1" ] && { say "SKIP eval ($2): no run dir (train failed)"; return; }
  say "EVAL $2 -> $1"
  uv run python scripts/go1_capacity.py "$1" 1500 > "$HOME/capacity_$2.log" 2>&1
  say "EVAL $2 done"
}

say "================ PHASE B START ($(git rev-parse --short HEAD)) ================"
P6B=$(find_run go1_baseline_payload_p6); P6A=$(find_run spring_go1_adaptive_p6)
say "existing 6kg policies: baseline=$P6B  adaptive=$P6A"

# ---- 1. 2nd seed @ 6 kg ----
train configs/go1_baseline_payload_p6_s2.yaml p6s2
B=$(find_run go1_baseline_payload_p6s2); say "seed2 baseline track=$(track "$B")"
train configs/spring_go1_adaptive_p6_s2.yaml p6s2
A=$(find_run spring_go1_adaptive_p6s2); say "seed2 adaptive track=$(track "$A")"
evalrun "$B" baseline_p6s2; evalrun "$A" adaptive_p6s2

# ---- 2. 15 kg curriculum (warm-start from 6 kg) ----
if [ -n "$P6B" ]; then
  train configs/go1_baseline_payload_p15.yaml curr15 "$P6B"
  B=$(find_run go1_baseline_payload_curr15); say "curr15 baseline track=$(track "$B")"
  evalrun "$B" baseline_curr15
else say "SKIP curr15 baseline: no 6kg checkpoint to restore"; fi
if [ -n "$P6A" ]; then
  train configs/spring_go1_adaptive_p15.yaml curr15 "$P6A"
  A=$(find_run spring_go1_adaptive_curr15); say "curr15 adaptive track=$(track "$A")"
  evalrun "$A" adaptive_curr15
else say "SKIP curr15 adaptive: no 6kg checkpoint to restore"; fi

# ---- 3. rough terrain @ 6 kg ----
train configs/go1_baseline_rough_p6.yaml r6
B=$(find_run go1_baseline_rough_r6); say "rough baseline track=$(track "$B")"
train configs/spring_go1_adaptive_rough_p6.yaml r6
A=$(find_run spring_go1_adaptive_rough_r6); say "rough adaptive track=$(track "$A")"
evalrun "$B" baseline_rough; evalrun "$A" adaptive_rough

say "================ PHASE B DONE ================"
for L in baseline_p6s2 adaptive_p6s2 baseline_curr15 adaptive_curr15 baseline_rough adaptive_rough; do
  echo "===== capacity_$L ====="; cat "$HOME/capacity_$L.log" 2>/dev/null || echo "(none)"
done
say "Box left UP. rsync ~/runs + ~/capacity_*.log, then halt."
