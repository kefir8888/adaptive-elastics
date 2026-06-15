#!/usr/bin/env bash
# Autonomous Go1 smaller-payload retrain + capacity/energy-vs-load eval.
# Designed to run DETACHED (nohup) and survive an SSH/internet disconnect.
#
# Plan:
#   1. Train baseline (no spring) at 10 kg payload DR.
#   2. GATE on eval tracking_lin_vel: if the policy COLLAPSED to standing (< 550, the failure
#      we saw at 25 kg), auto-fall back to a SAFE 6 kg and retrain the baseline.
#   3. Train the adaptive per-leg-preload policy at the chosen payload.
#   4. Run the capacity sweep (reports forward SPEED + electrical-vs-load) on both.
# Does NOT halt the box — results live in ~/runs for rsync; box lifecycle is managed externally.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/adaptive-elastics" || exit 1
RUNS="$HOME/runs"; mkdir -p "$RUNS"
GATE=550                                   # tracking_lin_vel below this == collapsed to standing
ts(){ date "+%F %T"; }
say(){ echo "[$(ts)] $*"; }

train(){   # $1=config  $2=suffix
  say "TRAIN $(basename "$1") suffix=$2"
  uv run pea-train --config "$1" --output_dir "$RUNS" --suffix "$2" \
      >> "$HOME/train_${2}_$(basename "$1" .yaml).log" 2>&1
  say "TRAIN $(basename "$1") exit=$?"
}
find_run(){ ls -dt "$RUNS"/*_"$1" 2>/dev/null | head -1; }   # $1 = name_suffix
track(){   # $1=run_dir -> tracking_lin_vel (0.0 on any error)
  uv run python - "$1" <<'PY'
import json,sys,os
try:
    d=json.loads(open(os.path.join(sys.argv[1],"metrics.jsonl")).read().strip().split("\n")[-1])
    print(d.get("eval/episode_reward/tracking_lin_vel",0.0))
except Exception: print(0.0)
PY
}

say "================ PIPELINE START ($(git rev-parse --short HEAD)) ================"

# ---- 1. baseline @ 10 kg ----
train configs/go1_baseline_payload.yaml p10
BASE=$(find_run go1_baseline_payload_p10)
[ -z "$BASE" ] && { say "FATAL: no baseline run dir (10kg train failed) — see ~/train_p10_*.log"; exit 1; }
TL=$(track "$BASE"); TL=${TL:-0}
say "baseline@10 dir=$BASE tracking_lin_vel=$TL"

PAY=10; SCFG=configs/spring_go1_adaptive.yaml; SSFX=p10
# ---- 2. gate: collapsed? fall back to 6 kg ----
if [ "$(awk "BEGIN{print ($TL<$GATE)?1:0}")" = "1" ]; then
  say "!! 10kg COLLAPSED (tracking_lin_vel=$TL < $GATE) -> FALLBACK to 6 kg"
  PAY=6; SCFG=configs/spring_go1_adaptive_p6.yaml; SSFX=p6
  train configs/go1_baseline_payload_p6.yaml p6
  BASE=$(find_run go1_baseline_payload_p6)
  [ -z "$BASE" ] && { say "FATAL: no baseline run dir (6kg train failed)"; exit 1; }
  TL=$(track "$BASE"); say "baseline@6 dir=$BASE tracking_lin_vel=${TL:-0}"
fi

# ---- 3. adaptive spring @ chosen payload ----
train "$SCFG" "$SSFX"
ADAPT=$(find_run "spring_go1_adaptive_$SSFX")
[ -z "$ADAPT" ] && { say "FATAL: no adaptive run dir (train failed)"; exit 1; }
say "adaptive@$PAY dir=$ADAPT tracking_lin_vel=$(track "$ADAPT")"

# ---- 4. capacity + energy-vs-load (with forward speed) ----
say "EVAL baseline @$PAY"
uv run python scripts/go1_capacity.py "$BASE"  1500 > "$HOME/capacity_baseline_p${PAY}.log" 2>&1
say "EVAL adaptive @$PAY"
uv run python scripts/go1_capacity.py "$ADAPT" 1500 > "$HOME/capacity_adaptive_p${PAY}.log" 2>&1

say "================ PIPELINE DONE — payload=${PAY}kg ================"
say "baseline=$BASE"
say "adaptive=$ADAPT"
echo "------- capacity BASELINE (p${PAY}) -------"; cat "$HOME/capacity_baseline_p${PAY}.log"
echo "------- capacity ADAPTIVE (p${PAY}) -------"; cat "$HOME/capacity_adaptive_p${PAY}.log"
say "Box left UP. To retrieve: rsync -avz ubuntu@<ip>:~/runs/ <local>/  then halt."
