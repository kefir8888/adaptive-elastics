#!/usr/bin/env bash
# Staged Go1 "dog-running" experiment, run detached on the GPU box.
#   cd ~/adaptive-elastics && nohup bash scripts/run_dog_pipeline.sh > ~/run_all.log 2>&1 &
# Stages (each warm-starts from the previous):
#   S0 walker (from scratch) -> gate on reward -> S1 trot -> S2 run+flight
#   -> FLIGHT GATE (a real all-feet-off window?) -> if yes: S4a no-spring control + S4b spring.
# It does NOT power off the box: the driver pulls results, then powers off (so results are never
# lost to a mid-sync shutdown). A separate hard backstop power-off guards against a runaway.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${PEA_RUNS_DIR:-$HOME/runs}"; mkdir -p "$RUNS"
SUMMARY="$RUNS/SUMMARY.txt"; : > "$SUMMARY"
say(){ echo "[$(date -u +%H:%M:%S)] PIPE: $*"; }
note(){ echo "$*" >> "$SUMMARY"; }

last_reward(){ tail -n 8 "$1/metrics.jsonl" 2>/dev/null | python3 -c '
import sys, json
ls = [l for l in sys.stdin if l.strip()]
print(json.loads(ls[-1]).get("eval/episode_reward", "nan") if ls else "nan")' 2>/dev/null || echo nan; }

run_stage(){ # tag config [restore_dir]; sets STAGE_DIR/STAGE_RC/STAGE_REW
  local tag="$1" cfg="$2" restore="${3:-}"
  local slog="$RUNS/$tag.train.log"
  say "START $tag cfg=$cfg ${restore:+restore=$restore}"
  local a=(--config "$cfg" --output_dir "$RUNS" --suffix "$tag")
  [ -n "$restore" ] && a+=(--restore "$restore")
  uv run pea-train "${a[@]}" > "$slog" 2>&1; local rc=$?
  local dir; dir=$(grep -m1 'run dir:' "$slog" | sed 's/.*run dir:[[:space:]]*//' | tr -d '[:space:]')
  local rew; rew=$(last_reward "$dir")
  say "END $tag rc=$rc reward=$rew dir=$dir"
  note "$tag rc=$rc reward=$rew dir=$dir"
  STAGE_DIR="$dir"; STAGE_RC=$rc; STAGE_REW=$rew
}

trap 'say "PIPELINE EXIT"; touch "$RUNS/PIPELINE_DONE"' EXIT
say "PIPELINE START $(hostname) $(date -u)"; note "started $(date -u) on $(hostname)"

# --- S0: flat walker (from scratch) ---
run_stage s0_walker configs/go1_run_s0_walker.yaml
S0="$STAGE_DIR"
if ! python3 -c "import sys; sys.exit(0 if float('$STAGE_REW') > 8 else 1)" 2>/dev/null; then
  say "GATE S0 FAIL: reward=$STAGE_REW <= 8 (walker not learning) -> STOP"; note "STOP: S0 gate (reward=$STAGE_REW)"; exit 1
fi
say "GATE S0 OK reward=$STAGE_REW"

# --- S1: fast trot (warm-start S0) ---
run_stage s1_trot configs/go1_run_s1.yaml "$S0"; S1="$STAGE_DIR"

# --- S2: run + flight (warm-start S1) ---
run_stage s2_run configs/go1_run_s2.yaml "$S1"; S2="$STAGE_DIR"

# --- FLIGHT GATE: probe S2 for a real all-feet-off window ---
say "FLIGHT PROBE $S2"
uv run python scripts/go1_run_probe.py "$S2" 700 > "$RUNS/s2_flight_probe.log" 2>&1 || true
cat "$RUNS/s2_flight_probe.log" || true
note "--- flight probe ---"; grep -E 'cmd .* m/s|true_flight' "$RUNS/s2_flight_probe.log" >> "$SUMMARY" 2>/dev/null || true

if grep -q 'true_flight True' "$RUNS/s2_flight_probe.log" 2>/dev/null; then
  say "FLIGHT GATE OK: a real all-feet-off window emerged -> running the spring stages"
  note "FLIGHT: YES -> S4a + S4b"
  # S4a: matched NO-SPRING control (S2 + 120M, identical to S4b except no spring)
  run_stage s4a_control configs/go1_run_s2.yaml "$S2"
  # S4b: per-leg adaptive preload spring (the almost-constant spring; same recipe as walking)
  run_stage s4b_spring configs/go1_run_spring_preload.yaml "$S2"
  note "DONE: S0-S2 + S4a/S4b complete"
  say "PIPELINE COMPLETE (with spring stages)"
else
  say "FLIGHT GATE: no true all-feet-off window -> it is a fast trot. Per design, STOP (this collapses to the known walking result); no spring compute spent."
  note "FLIGHT: NO -> stopped after S2 (fast trot)"
  say "PIPELINE COMPLETE (stopped at flight gate)"
fi
