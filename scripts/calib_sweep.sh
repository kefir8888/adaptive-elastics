#!/usr/bin/env bash
# Energy-weight calibration sweep (Milestone 4 prerequisite).
# Runs the NO-SPRING condition at several total-electrical penalty weights so we
# can pick the largest weight that keeps velocity tracking intact while the
# energy term still drops. Spring is OFF so this isolates the weight's effect.
# Short runs (not the full 200 M) — enough to read the tracking-vs-energy trade.
# Run detached on the GPU box:  nohup bash scripts/calib_sweep.sh > ~/calib.log 2>&1 &
set -euo pipefail
cd "$HOME/adaptive-elastics"
STEPS=80000000
WEIGHTS="-1e-4 -2.5e-4 -5e-4 -7.5e-4 -1e-3"
for w in $WEIGHTS; do
  echo "=== calibration: energy_reward_weight=$w, $STEPS steps, spring OFF ==="
  # NOTE: --energy-weight=$w (equals form) — argparse reads a leading-'-' value
  # only with '='; scientific notation like -5e-4 is not matched as a number.
  ~/.local/bin/uv run pea-train \
    --config configs/walk_baseline.yaml \
    --energy-weight="$w" \
    --num_timesteps "$STEPS" \
    --output_dir "$HOME/runs" \
    --suffix "calib_${w}"
done
echo "=== calibration sweep complete ==="
