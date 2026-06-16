# Training run checkpoint inventory

This file catalogs every training run produced by this project, along with its location and
the question it was designed to answer. It exists so any run can be found quickly and its
purpose understood without reading code or logs.

**Backup rule:** Every completed run must be mirrored to the Google Drive `pea_runs` folder
(path resolved by `pea.config.resolve_runs_dir()`). Do this immediately after the GPU box
finishes training, before deleting the box. The `runs/` subdirectory on Drive mirrors the
`outputs/runs/` layout used for the load-carrying program runs. Runs that are Drive-only
(no local copy) are marked accordingly.

**Spring types used below:**
- `none` — no spring; pure motor-driven baseline.
- `constant` — a fixed torque applied at every step regardless of joint angle (a preload).
- `linear` — torque proportional to how far the joint is from an equilibrium angle: τ = k · (θ − θ₀).
- `preload_dr` — like `constant`, but the preload magnitude is randomised each episode in proportion to the payload; this is the "adaptive preload" mechanism for load-carrying.

**Reward** shown is the final evaluation episode reward (dimensionless sum of weighted
sub-terms; higher is better within the same robot/environment). Not comparable across
robots or reward configurations.

---

## G1 walking (Unitree G1 humanoid, flat terrain)

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-11_baseline | G1 | Very first smoke test of the training pipeline; 0 steps completed (crashed at init). | none | 1 | 0 | −5.92 (step 0 eval) | Drive only | No |
| 2026-06-11_baseline_2 | G1 | Early Colab T4 baseline run; stopped at ~106 M steps (Colab disconnect); checkpoints present but no final policy_params export. | none | 1 | 106 | 9.07 | Drive only | Yes (checkpoints/) |
| 2026-06-11_baseline_h100 | G1 | First full 200 M-step G1 flat-walk baseline on H100; establishes the reference walk policy for post-hoc spring analysis. | none | 1 | 202 | 12.46 | Drive only | Yes |
| 2026-06-13_baseline_gate_smoke | G1 | Smoke test confirming the electrical-energy reward term (energy_reward_weight = −5×10⁻⁴) can be added without breaking training. | none | 1 | 0.03 | −16.22 | Drive only | Yes |
| 2026-06-13_spring_hip_linear_smoke | G1 | Smoke test of the hip-pitch linear spring (k = 68 N·m/rad, θ₀ = −0.29 rad) with energy penalty; verifies the spring code path runs end-to-end. | linear, hip_pitch, k=68, θ₀=−0.29 rad | 1 | 0.03 | −4.55 | Drive only | Yes |
| 2026-06-10_baseline_smoke | G1 | Very first local smoke test of the full pipeline on Mac CPU; confirms env + policy + metrics all execute. | none | 1 | 0.03 | −4.43 | Drive+local | Yes |
| 2026-06-14_walk_baseline_smoke | G1 | Smoke test of the final walk_baseline config (with action-rate, dof-acc, and torque penalties + energy reward weight −2.5×10⁻⁴). | none | 1 | 0.03 | −7.67 | Drive+local | Yes |
| 2026-06-14_walk_baseline_calib_-5e-4 | G1 | Calibration run to check that a stronger energy reward weight (−5×10⁻⁴) trains stably; stopped at ~81 M steps. | none | 1 | 81 | 3.17 | Drive+local | Yes |
| 2026-06-14_walk_baseline_full200 | G1 | **Primary G1 no-spring baseline** for the in-loop spring comparison; full 200 M-step run with the final reward config. | none | 1 | 121 | 9.76 | Drive+local | Yes |
| 2026-06-14_spring_walk_spring_full200 | G1 | **Primary G1 hip-pitch linear spring run** paired against walk_baseline_full200; tests whether a hand-tuned in-loop spring reduces electrical energy during walking. | linear, hip_pitch, k=66.36 N·m/rad, θ₀=−0.273 rad | 1 | 121 | 9.37 | Drive+local | Yes |

---

## G1 running (Unitree G1 humanoid, flat terrain, high-speed joystick)

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-15_g1_run_baseline_run1 | G1 | Baseline running policy trained with the joystick speed range widened to 0–3 m/s; establishes whether the G1 can learn a running gait at all. | none | 1 | 202 | 5.91 | Drive+local | Yes |
| 2026-06-15_g1_run_s1_jog_s1jog | G1 | Alternative running config with joystick range 0–1.6 m/s and reward weights tuned for jogging (higher feet_air_time, step 1 reward shaping); 150 M-step cap. | none | 1 | 153 | 1.32 | Drive+local | Yes |

---

## Go1 walking — no-spring baselines (Go1 quadruped, flat terrain)

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-11_spring_constant_smoke | Go1* | Smoke test of a constant knee preload (τ₀ = −12 N·m) on the Go1 env; verifies the constant spring code path works. *(Config says G1 env — may be a copy-paste artefact from before the Go1 branch.)* | constant, knee, τ₀=−12 N·m | 1 | 0.03 | −4.65 | Drive only | Yes |
| 2026-06-14_go1_baseline_go1base | Go1 | **Go1 flat-walk no-spring reference** (0 kg payload range); paired with go1spring to isolate the spring effect at zero load. | none | 1 | 206 | 28.23 | Drive+local | Yes |
| 2026-06-14_spring_go1_go1spring | Go1 | **Go1 constant knee preload** (τ₀ = 3.5 N·m on calf joint) at zero payload; first in-loop spring result on Go1, confirms the low-gear ohmic savings. | constant, calf, τ₀=3.5 N·m | 1 | 206 | 28.17 | Drive+local | Yes |

---

## Go1 load-carrying — no-spring baselines

Each run trains a single blind policy (the controller does not observe the payload) on a uniform random payload in [0, payload_max_kg] kg; this is the control arm for the adaptive-preload comparison.

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-15_go1_baseline_payload_p6 | Go1 | Baseline, payload range 0–6 kg, seed 1; primary low-load reference. | none | 1 | 206 | 26.21 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_p6s2 | Go1 | Baseline, 0–6 kg, seed 2; second seed for statistical reliability. | none | 2 | 206 | 26.10 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_p6s3 | Go1 | Baseline, 0–6 kg, seed 3; third seed. | none | 3 | 206 | 26.44 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_p10 | Go1 | Baseline, 0–10 kg; checks how performance degrades with increasing load ceiling. | none | 1 | 206 | 20.88 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_bl_p20 | Go1 | Baseline, 0–20 kg; heavier load tier. | none | 1 | 206 | 23.38 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_bl_p25 | Go1 | Baseline, 0–25 kg; heavier load tier. | none | 1 | 206 | 21.48 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_bl_p30 | Go1 | Baseline, 0–30 kg; heavier load tier. | none | 1 | 206 | 20.20 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_p25 | Go1 | Baseline, 0–25 kg, extended to 310 M steps to check late-stage convergence. | none | 1 | 310 | 18.16 | Drive+local | Yes |
| 2026-06-15_go1_baseline_payload_curr15 | Go1 | Baseline, 0–15 kg; mid-range load tier, seed 1. | none | 1 | 206 | 24.62 | Drive+local | Yes |
| 2026-06-15_go1_baseline_rough_half_rh | Go1 | Baseline, 0–6 kg, **rough terrain** (bump height halved to 2.5 cm); tests robustness of load-carrying on uneven ground with an easier terrain curriculum. | none | 1 | 206 | 25.34 | Drive+local | Yes |
| 2026-06-15_go1_baseline_rough_r6 | Go1 | Baseline, 0–6 kg, **rough terrain** (full 5 cm bumps); harder terrain reference. | none | 1 | 206 | 18.45 | Drive+local | Yes |
| 2026-06-16_go1_baseline_payload_s2c15 | Go1 | Baseline, 0–15 kg, seed 2; second seed for 15 kg tier. | none | 2 | 206 | 24.94 | Drive+local | Yes |
| 2026-06-16_go1_baseline_payload_s3c15 | Go1 | Baseline, 0–15 kg, seed 3; stopped before completing (no policy_params exported). | none | 3 | 184 | 24.77 | Local only | No |

---

## Go1 load-carrying — adaptive preload spring

Each run uses `preload_dr` spring: a constant calf-joint preload whose magnitude is randomised each episode proportionally to the payload drawn that episode (so the spring is heavier when the robot carries more). The controller still does not observe the payload.

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-15_spring_go1_adaptive_p6 | Go1 | **Adaptive preload, 0–6 kg, seed 1**; primary result for low-load spring benefit. τ₀_max = 8 N·m. | preload_dr, calf, τ₀=8 N·m | 1 | 206 | 26.84 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_p6s2 | Go1 | Adaptive preload, 0–6 kg, seed 2. τ₀_max = 8 N·m. | preload_dr, calf, τ₀=8 N·m | 2 | 206 | 26.19 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_p6s3 | Go1 | Adaptive preload, 0–6 kg, seed 3. τ₀_max = 8 N·m. | preload_dr, calf, τ₀=8 N·m | 3 | 206 | 26.00 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_curr15 | Go1 | Adaptive preload, 0–15 kg; τ₀_max = 14 N·m. | preload_dr, calf, τ₀=14 N·m | 1 | 206 | 25.12 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_sp_p20 | Go1 | Adaptive preload, 0–20 kg; τ₀_max = 16 N·m. | preload_dr, calf, τ₀=16 N·m | 1 | 206 | 23.96 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_sp_p25 | Go1 | Adaptive preload, 0–25 kg; τ₀_max = 20 N·m. | preload_dr, calf, τ₀=20 N·m | 1 | 206 | 22.26 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_adapt | Go1 | Adaptive preload, 0–25 kg, extended run to 310 M steps; τ₀_max = 15 N·m. | preload_dr, calf, τ₀=15 N·m | 1 | 310 | 18.21 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_rough_half_rh | Go1 | Adaptive preload, 0–6 kg, rough terrain half-height (2.5 cm); tests spring on uneven ground. τ₀_max = 8 N·m. | preload_dr, calf, τ₀=8 N·m | 1 | 206 | 25.87 | Drive+local | Yes |
| 2026-06-15_spring_go1_adaptive_rough_r6 | Go1 | Adaptive preload, 0–6 kg, rough terrain full-height (5 cm). τ₀_max = 8 N·m. | preload_dr, calf, τ₀=8 N·m | 1 | 206 | 18.52 | Drive+local | Yes |
| 2026-06-16_spring_go1_adaptive_s2c15 | Go1 | Adaptive preload, 0–15 kg, seed 2; τ₀_max = 14 N·m. | preload_dr, calf, τ₀=14 N·m | 1 | 206 | 24.04 | Drive+local | Yes |
| 2026-06-16_spring_go1_adaptive_sp_p30 | Go1 | Adaptive preload, 0–30 kg; τ₀_max = 24 N·m; heaviest load tier tested. | preload_dr, calf, τ₀=24 N·m | 1 | 206 | 20.72 | Drive+local | Yes |

---

## Smoke / aborted (low value)

These runs produced no usable policy or were terminated before reaching meaningful training progress.

| Run folder | Robot | Experiment + purpose | Spring setup | Seed | Steps (M) | Final eval reward | Location | Has checkpoint? |
|---|---|---|---|---|---|---|---|---|
| 2026-06-11_baseline | G1 | Pipeline crash at step 0; no training occurred. | none | 1 | 0 | −5.92 (init eval) | Drive only | No |
| 2026-06-16_go1_baseline_payload_s3c15 | Go1 | Baseline 0–15 kg seed 3; terminated early (no policy_params file exported). Metrics only. | none | 3 | 184 | 24.77 | Local only | No |
