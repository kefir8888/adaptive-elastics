#!/usr/bin/env bash
# Assemble the gravity-compensation reporting bundle (outputs/gravity_compensation) from the
# per-experiment raw outputs, with descriptive names. Run after galaxea_lift.py / limx_roll.py
# (with --video) and gravcomp_table.py (which writes 05_energy_savings_table.md into the folder).
set -euo pipefail
cd "$(dirname "$0")/.."
DST=outputs/gravity_compensation
RAW=$DST/raw
mkdir -p "$DST"
cp "$RAW/galaxea/galaxea_lift.mp4"          "$DST/01_galaxea_coordinated_upright_lift.mp4"
cp "$RAW/galaxea/galaxea_taus.png"          "$DST/02_galaxea_torque_vs_angle_fitted_springs.png"
cp "$RAW/limx/limx_roll_spring.mp4"         "$DST/03_limx_w1_wheeled_roll.mp4"
cp "$RAW/limx/limx_taus.png"                "$DST/04_limx_w1_knee_torque_over_time.png"
cp "$RAW/galaxea/galaxea_taus_kneeonly.png" "$DST/06_galaxea_spring_middle_joint_only.png"
[ -f "$RAW/galaxea/galaxea_lift_kneeonly.mp4" ] && cp "$RAW/galaxea/galaxea_lift_kneeonly.mp4" "$DST/06_galaxea_lift_middle_joint_only.mp4"
rm -f "$DST/04_limx_w1_knee_torque_vs_angle.png"   # superseded by the over-time plot
# Compact, loopable, HIGH-RES GIFs (trimmed to a few representative cycles). Full 256-color
# palette + bayer (ordered) dither: stable across frames (no dither "flicker") and compresses
# far better than error-diffusion; diff_mode=rectangle skips static regions. Slow clips trade
# frame-rate for resolution to stay ~5-7 MB at 720 px (the long phases clip drops to 600 px).
gif () {  # in_mp4 out_gif ss(s) t(s) fps width
  ffmpeg -y -loglevel error -ss "$3" -t "$4" -i "$1" \
    -filter_complex "fps=$5,scale=$6:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$2"
}
gif "$DST/01_galaxea_coordinated_upright_lift.mp4" "$DST/01_galaxea_coordinated_upright_lift.gif" 0 7.6 10 720
gif "$DST/03_limx_w1_wheeled_roll.mp4"             "$DST/03_limx_w1_wheeled_roll.gif"             2 7   8  720
[ -f "$DST/06_galaxea_lift_middle_joint_only.mp4" ] && gif "$DST/06_galaxea_lift_middle_joint_only.mp4" "$DST/06_galaxea_lift_middle_joint_only.gif" 0 7.6 10 720

# cyclic forward-reach experiment (scripts/galaxea_reach.py)
if [ -f "$RAW/galaxea_reach/reach.mp4" ]; then
  cp "$RAW/galaxea_reach/reach.mp4"                  "$DST/07_galaxea_forward_reach.mp4"
  cp "$RAW/galaxea_reach/reach_knee_torque_angle.png" "$DST/08_galaxea_reach_knee_torque_angle.png"
  # high-res gif, trimmed to ~2 reach cycles (the reach is slow)
  gif "$RAW/galaxea_reach/reach.mp4" "$DST/07_galaxea_forward_reach.gif" 0 16 9 720
fi

# LimX 4-phase body-height roll + adaptive knee spring (scripts/limx_phases.py)
if [ -f "$RAW/limx_phases/phases.mp4" ]; then
  cp "$RAW/limx_phases/phases.mp4"            "$DST/09_limx_w1_height_phases_adaptive.mp4"
  cp "$RAW/limx_phases/phases_timeplots.png"  "$DST/10_limx_w1_height_phases_timeplots.png"
  # long clip (~20 s): lower fps + 600 px to keep size moderate at higher resolution
  gif "$RAW/limx_phases/phases.mp4" "$DST/09_limx_w1_height_phases_adaptive.gif" 0 22 4 600
fi

# Galaxea forward-lean with a linear spring on the bottom torso joint (scripts/galaxea_lean.py)
if [ -f "$RAW/galaxea_lean/lean.mp4" ]; then
  cp "$RAW/galaxea_lean/lean.mp4"              "$DST/11_galaxea_lean_bottom_joint_spring.mp4"
  cp "$RAW/galaxea_lean/lean_torque_angle.png" "$DST/12_galaxea_lean_bottom_joint_torque_angle.png"
  gif "$RAW/galaxea_lean/lean.mp4" "$DST/11_galaxea_lean_bottom_joint_spring.gif" 0 11 9 720
fi

# presentation-quality colored turntables + stills (scripts/pretty_render.py).
# These are presentation-only renders (no measurement) -- NOT experiment results -- so they
# go into _superseded/ rather than the top-level deliverable bundle (which holds only results).
if [ -d "$RAW/pretty" ]; then
  mkdir -p "$DST/_superseded"
  for r in "galaxea:13" "dog:14"; do
    name=${r%%:*}; num=${r##*:}
    [ -f "$RAW/pretty/${name}_turntable.mp4" ] || continue
    cp "$RAW/pretty/${name}_pretty.png"      "$DST/_superseded/${num}_${name}_rendered.png"
    cp "$RAW/pretty/${name}_turntable.mp4"   "$DST/_superseded/${num}_${name}_turntable.mp4"
    gif "$RAW/pretty/${name}_turntable.mp4" "$DST/_superseded/${num}_${name}_turntable.gif" 0 6 10 720
  done
fi

# Galaxea 5 'free' reaching motions, bottom-joint spring (scripts/galaxea_free.py) -> subfolder
if [ -d "$RAW/galaxea_free" ]; then
  mkdir -p "$DST/free_reaches"
  for opt in forward reach_down reach_up side_twist free_scan; do
    [ -f "$RAW/galaxea_free/$opt.mp4" ] || continue
    cp "$RAW/galaxea_free/${opt}_torque_angle.png" "$DST/free_reaches/${opt}_torque_angle.png"
    gif "$RAW/galaxea_free/$opt.mp4" "$DST/free_reaches/$opt.gif" 0 11 9 720
  done
fi

echo "report assembled in $DST/ :"
ls -1 "$DST"
