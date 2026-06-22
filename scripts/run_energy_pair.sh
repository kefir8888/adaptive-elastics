#!/bin/bash
# Re-run BOTH energy-objective bounding arms back-to-back with centered_dr=true (the 2026-06-22 fix
# for the spring falling 0/3 at nominal). No git reset here -- the centered_dr configs are scp'd in
# and a reset would clobber them; the CODE is unchanged at f1c2be9 (already has pea/randomize.py).
# Runs detached, low priority so sshd never starves. Monitor metrics.jsonl / box.py status.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$HOME/adaptive-elastics" || exit 1
S4="$HOME/runs/2026-06-21_g1_clean_s4_bound_clean_s4"
LOG="$HOME/energy_pair.log"
echo "[$(date)] PAIR START (centered_dr=true)" >> "$LOG"
for cfg in g1_bound_energy_baseline g1_bound_energy_spring; do
  echo "[$(date)] START $cfg" >> "$LOG"
  nice -n 19 ionice -c3 uv run pea-train --config "configs/$cfg.yaml" \
    --restore "$S4" --output_dir "$HOME/runs" --suffix run >> "$LOG" 2>&1
  echo "[$(date)] END $cfg rc=$?" >> "$LOG"
done
touch "$HOME/ENERGY_PAIR_DONE"
echo "[$(date)] PAIR DONE" >> "$LOG"
