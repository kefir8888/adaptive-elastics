#!/usr/bin/env bash
# Assemble the gravity-compensation reporting bundle (outputs/gravity_compensation) from the
# per-experiment raw outputs, with descriptive names. Run after galaxea_lift.py / limx_roll.py
# (with --video) and gravcomp_table.py (which writes 05_energy_savings_table.md into the folder).
set -euo pipefail
cd "$(dirname "$0")/.."
DST=outputs/gravity_compensation
RAW=$DST/raw
mkdir -p "$DST"
cp "$RAW/galaxea/galaxea_lift.mp4"  "$DST/01_galaxea_coordinated_upright_lift.mp4"
cp "$RAW/galaxea/galaxea_taus.png"  "$DST/02_galaxea_torque_vs_angle_fitted_springs.png"
cp "$RAW/limx/limx_roll_spring.mp4" "$DST/03_limx_w1_wheeled_roll.mp4"
cp "$RAW/limx/limx_taus.png"        "$DST/04_limx_w1_knee_torque_vs_angle.png"
echo "report assembled in $DST/ :"
ls -1 "$DST"
