#!/usr/bin/env bash
# Staged Go1 "dog-running" experiment with the forward-command CURRICULUM, run detached on the box.
#   cd ~/adaptive-elastics && nohup bash scripts/run_dog_pipeline.sh > ~/run_all.log 2>&1 &
# Sequence (each warm-starts from the previous):
#   S0 walker (scratch) -> C1 [1.2-1.8] -> C2 [1.8-2.5] -> C3 [2.5-3.2]+flight tweaks
#   -> FLIGHT GATE -> if a real all-feet-off window: S4a no-spring control + S4b preload spring.
# Gates: S0 on reward; C1/C2 on achieved forward speed (constant-command probe); C3 on flight.
# Does NOT power off the box (the driver pulls results then halts, so nothing is lost to a mid-sync shutdown).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${PEA_RUNS_DIR:-$HOME/runs}"; mkdir -p "$RUNS"
SUMMARY="$RUNS/SUMMARY.txt"; : > "$SUMMARY"
say(){ echo "[$(date -u +%H:%M:%S)] PIPE: $*"; }
note(){ echo "$*" >> "$SUMMARY"; }

last_reward(){ tail -n 8 "$1/metrics.jsonl" 2>/dev/null | python3 -c '
import sys, json
ls = [l for l in sys.stdin if l.strip()]
print(json.loads(ls[-1]).get("eval/episode_reward", "nan") if ls else "nan")' 2>/dev/null || echo nan; }

run_stage(){ # tag config [restore_dir]; sets STAGE_DIR/STAGE_REW
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
  STAGE_DIR="$dir"; STAGE_REW="$rew"
}

probe_speed(){ # dir tag cmd  -> echoes achieved forward speed (m/s) at a constant command
  uv run python scripts/go1_run_probe.py "$1" 700 "$3" > "$RUNS/$2.probe.log" 2>&1 || true
  grep -oE 'speed [0-9.]+' "$RUNS/$2.probe.log" | head -1 | awk '{print $2+0}'
}

ge(){ python3 -c "import sys; sys.exit(0 if float('$1') >= float('$2') else 1)" 2>/dev/null; }

trap 'say "PIPELINE EXIT"; touch "$RUNS/PIPELINE_DONE"' EXIT
say "PIPELINE START $(hostname) $(date -u)"; note "started $(date -u) on $(hostname)"

# --- S0: flat walker (from scratch) ---
run_stage s0_walker configs/go1_run_s0_walker.yaml; S0="$STAGE_DIR"
ge "$STAGE_REW" 8 || { say "GATE S0 FAIL reward=$STAGE_REW <=8 -> STOP"; note "STOP: S0 gate ($STAGE_REW)"; exit 1; }
say "GATE S0 OK reward=$STAGE_REW"

# --- C1: forced-forward jog [1.2,1.8] ---
run_stage c1 configs/go1_run_c1.yaml "$S0"; C1="$STAGE_DIR"
SP=$(probe_speed "$C1" c1 1.5); say "C1 achieved speed ${SP} m/s at cmd 1.5"; note "C1 speed=$SP"
ge "${SP:-0}" 1.0 || { say "GATE C1 FAIL speed=$SP <1.0 (not running forward) -> STOP"; note "STOP: C1 speed $SP"; exit 1; }

# --- C2: forced-forward run [1.8,2.5] ---
run_stage c2 configs/go1_run_c2.yaml "$C1"; C2="$STAGE_DIR"
SP=$(probe_speed "$C2" c2 2.2); say "C2 achieved speed ${SP} m/s at cmd 2.2"; note "C2 speed=$SP"
ge "${SP:-0}" 1.5 || { say "GATE C2 FAIL speed=$SP <1.5 -> STOP"; note "STOP: C2 speed $SP"; exit 1; }

# --- C3: fast run + flight tweaks [2.5,3.2] ---
run_stage c3 configs/go1_run_c3.yaml "$C2"; C3="$STAGE_DIR"

# --- FLIGHT GATE: probe C3 across running speeds ---
say "FLIGHT PROBE $C3"
uv run python scripts/go1_run_probe.py "$C3" 700 2.6,3.0,3.2 > "$RUNS/c3_flight_probe.log" 2>&1 || true
cat "$RUNS/c3_flight_probe.log" || true
note "--- C3 flight probe ---"; grep -E 'cmd .* m/s|true_flight' "$RUNS/c3_flight_probe.log" >> "$SUMMARY" 2>/dev/null || true

if grep -q 'true_flight True' "$RUNS/c3_flight_probe.log" 2>/dev/null; then
  say "FLIGHT GATE OK: a real all-feet-off window emerged -> running the spring stages"
  note "FLIGHT: YES -> S4a + S4b"
  run_stage s4a_control configs/go1_run_c3.yaml "$C3"           # matched NO-SPRING control (C3 settings, +120M)
  run_stage s4b_spring  configs/go1_run_spring_preload.yaml "$C3"  # per-leg adaptive preload (the almost-constant spring)
  note "DONE: curriculum + S4a/S4b complete"
  say "PIPELINE COMPLETE (with spring stages)"
else
  say "FLIGHT GATE: no true all-feet-off window -> running but grounded (or still slow). Stop; report; decide on the spring."
  note "FLIGHT: NO -> stopped after C3"
  say "PIPELINE COMPLETE (stopped at flight gate)"
fi
