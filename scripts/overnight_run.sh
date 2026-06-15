#!/usr/bin/env bash
# Autonomous ~5h NON-BLOCKING schedule (user offline). Runs AFTER the G1 Stage-1 jog finishes.
# DEFERRED (needs a human check, NOT run here): the G1 flight gate / Stage 2.
# Non-blocking dog work only, in priority order:
#   A. 3rd seed @6kg (completes the 3-seed flat result)
#   B. rough redo on EASIER terrain (2.5cm, warm-start from flat) -- logs the less-rough result
#   C. capacity-to-failure ladder, seed 1 (warm-start curr15 -> 20 -> 25 -> 30 until it can't walk)
#   D. bring seeds 2/3 to the 15kg curriculum (prep for their ladders later)
# Detached; survives disconnect; does NOT halt. Results -> ~/runs + ~/cap_*.log; log -> ~/overnight.log.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/adaptive-elastics" || exit 1
RUNS="$HOME/runs"; GATE=650
ts(){ date "+%F %T"; }; say(){ echo "[$(ts)] $*"; }
train(){ local x=""; [ -n "${3:-}" ] && x="--restore $3"; say "TRAIN $(basename "$1") sfx=$2 $x"
  uv run pea-train --config "$1" --output_dir "$RUNS" --suffix "$2" $x >> "$HOME/tr_${2}.log" 2>&1; say "  TRAIN $2 exit=$?"; }
find_run(){ ls -dt "$RUNS"/*_"$1" 2>/dev/null | head -1; }
track(){ uv run python - "$1" <<'PY'
import json,sys,os
try: print(round(json.loads(open(os.path.join(sys.argv[1],"metrics.jsonl")).read().strip().split("\n")[-1]).get("eval/episode_reward/tracking_lin_vel",0.0),1))
except Exception: print(0.0)
PY
}
evalcap(){ [ -z "${1:-}" ] && { say "SKIP eval $2 (no run dir)"; return; }; say "EVAL $2"; uv run python scripts/go1_capacity.py "$1" 1500 > "$HOME/cap_$2.log" 2>&1; }
mkcfg(){ local o="/tmp/$(basename "$1" .yaml)_P$2.yaml"; sed -E "s/payload_max_kg: .*/payload_max_kg: $2.0/" "$1" > "$o"; [ -n "${3:-}" ] && sed -i -E "s/tau0: .*/tau0: $3.0/" "$o"; echo "$o"; }
ladder(){  # $1=base_config $2=start_run $3=tag(bl|sp) $4=is_spring(1/0)
  local cur="$2" maxw=15
  [ -z "$cur" ] && { say "$3 ladder: no start policy on box, skipping"; return; }
  for P in 20 25 30; do
    local tau0=""; [ "$4" = "1" ] && tau0=$((P*4/5))
    local cfg; cfg=$(mkcfg "$1" "$P" "$tau0")
    train "$cfg" "${3}_p$P" "$cur"
    local r; r=$(find_run "${3}_p$P")
    [ -z "$r" ] && { say "$3 @${P}kg: no run dir (train failed) -> stop ladder at ${maxw}kg"; break; }
    local tl; tl=$(track "$r"); say "$3 @${P}kg tracking_lin_vel=$tl (gate $GATE)"
    if awk "BEGIN{exit !(${tl:-0} < $GATE)}"; then say "$3 FAILED to walk @${P}kg -> MAX WALKABLE ${maxw}kg"; break; fi
    maxw=$P; cur="$r"; evalcap "$r" "${3}_p$P"
  done
  say "$3 MAX WALKABLE = ${maxw}kg"
}

say "===== OVERNIGHT START ($(git rev-parse --short HEAD)) ====="
say "waiting for G1 Stage-1 jog (S2 flight gate DEFERRED for human check)..."
while pgrep -f "pea-train.*s1jog" >/dev/null 2>&1; do sleep 120; done
say "S1 done; GPU free. Beginning non-blocking dog schedule."

# A. 3rd seed @6kg
train configs/go1_baseline_payload_p6_s3.yaml p6s3
train configs/spring_go1_adaptive_p6_s3.yaml p6s3
evalcap "$(find_run go1_baseline_payload_p6s3)" baseline_p6s3
evalcap "$(find_run spring_go1_adaptive_p6s3)" adaptive_p6s3
say "===== A done: 3rd seed ====="

# B. rough redo (easier 2.5cm terrain, warm-start flat 6kg)
train configs/go1_baseline_rough_half.yaml rh "$(find_run go1_baseline_payload_p6)"
train configs/spring_go1_adaptive_rough_half.yaml rh "$(find_run spring_go1_adaptive_p6)"
evalcap "$(find_run go1_baseline_rough_half_rh)" baseline_roughhalf
evalcap "$(find_run spring_go1_adaptive_rough_half_rh)" adaptive_roughhalf
say "===== B done: rough redo (easier terrain) ====="

# C. capacity-to-failure ladder, seed 1
say "--- C baseline ladder ---"; ladder configs/go1_baseline_payload_p15.yaml "$(find_run go1_baseline_payload_curr15)" bl 0
say "--- C spring ladder ---";   ladder configs/spring_go1_adaptive_p15.yaml "$(find_run spring_go1_adaptive_curr15)" sp 1
say "===== C done: capacity-to-failure (seed 1) ====="

# D. bring seeds 2/3 to curr15 (prep)
B2=$(find_run go1_baseline_payload_p6s2); A2=$(find_run spring_go1_adaptive_p6s2)
B3=$(find_run go1_baseline_payload_p6s3); A3=$(find_run spring_go1_adaptive_p6s3)
[ -n "$B2" ] && train configs/go1_baseline_payload_p15.yaml s2c15 "$B2"
[ -n "$A2" ] && train configs/spring_go1_adaptive_p15.yaml s2c15 "$A2"
[ -n "$B3" ] && train configs/go1_baseline_payload_p15.yaml s3c15 "$B3"
[ -n "$A3" ] && train configs/spring_go1_adaptive_p15.yaml s3c15 "$A3"
say "===== D done: seeds 2/3 -> curr15 (prep) ====="

say "===== OVERNIGHT DONE ====="
for L in baseline_p6s3 adaptive_p6s3 baseline_roughhalf adaptive_roughhalf bl_p20 bl_p25 bl_p30 sp_p20 sp_p25 sp_p30; do
  [ -f "$HOME/cap_$L.log" ] && { echo "===== cap_$L ====="; cat "$HOME/cap_$L.log"; }
done
say "Box left UP. rsync ~/runs + ~/cap_*.log + ~/overnight.log, then halt."
