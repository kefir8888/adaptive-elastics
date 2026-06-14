# Constant Registry — metric-affecting constants to proofread at wrap-up

**Purpose.** A living checklist of every numerical constant that participates in a
reported metric (electrical power, ohmic loss, cost of transport, braking/recoverable
energy, peak/RMS torque, saturation verdict, actuation share, any % reduction). At the
end of the investigation, each row gets verified before publication. Keep this current:
when a constant is confirmed against a primary source, mark it; when a value changes,
update the row and re-run any number it feeds.

**How to read provenance:** MEASURED | DATASHEET | PROXY-ESTIMATE | PLACEHOLDER |
DERIVED-FROM-MODEL | CONVENTION/ASSUMPTION. A **KNOWN-SUSPECT** flag means we already
have reason to believe the value is wrong.

> **Quantified sensitivity for the #1 risk (R/Kt², added 2026-06-14).** Re-running the
> post-hoc hip-spring saving across the corrected R/Kt² band (`scripts/recalc_rkt2.py`,
> same baseline trajectory, no-regen) gives: whole-body saving **−2.9 % @ 0.0025 →
> −5.0 % @ 0.02 → −8.3 % @ 0.05**; whole-body ohmic share **3.9 % → 25.9 % → 48.3 %**;
> hip-pitch joint saving **−10.7 % → −28.4 %**. This is the SENSITIVITY; for WHERE the
> truth sits, the exhaustive search (`docs/g1_motor_constants.md`, 2026-06-14) places
> **0.05 BEYOND the plausible band.** With the corrected Go2 anchor + motor-size scaling
> (the G1 7520 is larger than the Go2 motor → lower R/Kt²), the defensible joint-side
> band is **0.001–0.020** (BEST ≈ 0.0024 knee / 0.0059 hip-pitch — essentially the code
> value). So the real walking headline is **−2 % to −5 %, best −2.9 %**, and the code
> 0.0025 is the OPTIMISTIC EDGE, not wrong by 10×. Only a bench 7520 measurement collapses
> the band; the relative spring-vs-no-spring % is R/Kt²-invariant regardless.

---

## Method & scope note

Every file in scope was read in full (`src/pea/{energy,metrics,experiment,env,config,springs,policy}.py`,
all of `scripts/`, and the `configs/*.yaml` that supply spring/reward parameters consumed
by reported metrics). A constant is listed if a wrong value would bias any reported number.
Pure plumbing (PPO/network hyperparameters, smoke-test sizes, XLA memory fraction, render
dims, rollout RNG seeds) is excluded — it does not enter a reported physical number.
`pea-train`/`pea-rollout`/`pea-analyze`/`pea-sweep` are thin shims in `scripts/`; the real
logic lives in `src/pea/`.

**Highest-risk constants (most load-bearing AND least verified) — proofread first:**

1. **`R/Kt²` for the G1 (≈ 0.0025)** — from `Kt=2.3`, `R=0.013` in `energy.py:46`. Sets the
   *entire* ohmic channel, hence every ohmic-% and CoT number. PROXY-ESTIMATE; the
   exhaustive search (`docs/g1_motor_constants.md`) gives a defensible joint-side band
   **0.001–0.020** with BEST ≈ the code value (motor-size scaling corrected the earlier
   "8–19× optimistic" alarm). Still NO external G1 source. **#1 publication risk** — only
   a bench 7520 measurement collapses the band.
2. **`energy.MOTORS['go2']` `Kt=0.26`, `R=0.30`** (`energy.py:56`) — `Kt=0.26` is
   motor-side (teardown), not joint-side as the field contract requires (joint-side ≈ 1.62,
   ×6.22); `R` is a self-declared PLACEHOLDER superseded by the measured 0.44/0.66.
3. **Using the 7520-22.5 Kt/R for *all* actuated DoFs** — hip-yaw (7520-14.3) and ankle
   (5020) differ; the G1 hip-pitch gear may be 14.3:1, not 22.5:1, shifting joint-side Kt
   and the per-joint ohmic split.
4. **`OMEGA_NOLOAD_EST = 20.0`** (`energy.py:105`) — drives the torque- vs speed-limited
   verdict (the whole B1 jump-height conclusion); a limit, not a true no-load speed, with a
   linear back-EMF rolloff assumed on top.

---

## 1. Motor electrical constants

| Symbol | Value (units) | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| `G1_KNEE.kt` (Kt, joint-side) | 2.3 N·m/A | `energy.py:46` | ohmic, elec power, CoT, ohmic %, all spring %, energy reward | PROXY-ESTIMATE (gear-scaled Go2 + peak-torque routes; 2.0–2.7) | Real 7520-22.5 datasheet/teardown. **KNOWN-SUSPECT** via R/Kt². |
| `G1_KNEE.r` (R, winding) | 0.013 Ω | `energy.py:46` | same as Kt | PROXY-ESTIMATE (0.009–0.025) | Measured phase resistance. **KNOWN-SUSPECT**. |
| **R/Kt² (derived)** | ≈ 0.00246 Ω/(N·m/A)² | from `energy.py:46` (doc `:38`) | the only Kt/R combination affecting headline ohmic-% | DERIVED from two estimates (0.0020–0.0032) | **KNOWN-SUSPECT: believed ~8–19× optimistic.** Re-derive from datasheet; report CoT as a band. |
| `MOTORS['g1']` | = `G1_KNEE` | `energy.py:55` | sweep energy when `--robot g1` | same as G1_KNEE | as above |
| `MOTORS['go2'].kt` | 0.26 N·m/A | `energy.py:56` | Go2 cross-robot energy | PROXY-ESTIMATE | **KNOWN-SUSPECT: motor-side, must gear-scale ×6.22 → ≈1.62.** |
| `MOTORS['go2'].r` | 0.30 Ω | `energy.py:56` | Go2 cross-robot energy | PLACEHOLDER | Superseded by measured 0.44 line / 0.66 phase. |
| `KT, R` (probe locals) | 2.3, 0.013 | `probe_speed_hold.py:25` | probe elec/ohmic/spring table | hardcoded copy of `G1_KNEE` | Drift risk — import `energy.G1_KNEE` instead. |
| `KT, R` (motor_budget) | = `energy.G1_KNEE` | `motor_budget.py:28` | per-joint budget, hip share, hip-spring saving | imported | inherits G1_KNEE status |
| `KT, R` (power_compare) | = `energy.G1_KNEE` | `power_compare.py:25` | whole-body N-vs-M watts, regen inflation | imported | inherits G1_KNEE status |

## 2. Gear ratios & motor limits (torque/speed envelope)

| Symbol | Value (units) | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| Gear ratio (7520-22.5) | 22.5:1 | `energy.py:34,55` (comment) | rationale for joint-side Kt scaling | DATASHEET-claimed | **KNOWN-SUSPECT for hip-pitch: may be 14.3:1.** Confirm per-joint. |
| `G1_LIMITS.tau_peak` | 139.0 N·m (knee) | `energy.py:106` | `tau_available()`, jump torque-limit | DERIVED-FROM-MODEL (matches `jnt_actfrcrange`) | AUTHORITATIVE; spot-check URDF effort. |
| `OMEGA_NOLOAD_EST` | 20.0 rad/s | `energy.py:105` | `saturation()` fallback; torque-vs-speed verdict | PROXY-ESTIMATE (= URDF knee limit) | **KNOWN-SUSPECT as no-load proxy** — it is the limit, not no-load. |
| `G1_JOINT_VEL` hip | 32.0 rad/s | `energy.py:98` | `saturation()` hip speed wall | DATASHEET (Unitree URDF) | Re-confirm URDF revision (g1_23dof). |
| `G1_JOINT_VEL` knee | 20.0 rad/s | `energy.py:99` | `saturation()` knee wall; B1 jump-height verdict | DATASHEET (URDF) | Load-bearing for the knee-speed-limited claim. |
| `G1_JOINT_VEL` ankle | 30.0 rad/s | `energy.py:99` | ankle speed wall | DATASHEET (URDF) | as above |
| `tau_peak` (per joint, live) | `jnt_actfrcrange[jid][1]` | `metrics.py:151`; `env.py:117` | saturation %, peak-vs-cap | DERIVED-FROM-MODEL (knee 139, hip 88, ankle 50) | AUTHORITATIVE; verify model version. |
| back-EMF rolloff form | `tau_peak·clip(1−|ω|/ω_noload,0,1)` | `energy.py:111–112` | `tau_available` | CONVENTION (linear back-EMF) | Validate against a real torque-speed curve. |

## 3. Energy-model assumptions

| Symbol | Value | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| `regen` (no-regeneration) | `False` | `energy.py:128,131`; clamp in `metrics.py:86`, `env.py:68`, `motor_budget.py:33`, `power_compare.py:64`, `probe_speed_hold.py:90` | which power model is reported; the braking-recovery win is no-regen-dependent | CONVENTION (back-EMF < bus, not a verified spec) | Report ~24 % regen sensitivity; note MIT Cheetah / Optimus exceptions. |
| Per-DoF clamp `max(P,0)` | per actuator, not net bus | same locations | no-regen elec energy (per-DoF clamp inflates vs net-bus) | CONVENTION | Confirm per-DoF (not net) is intended. |
| Elec power formula | `τ·ω + (τ/Kt)²·R` | `energy.py:130`, `metrics.py:85`, `env.py:68` | every electrical number | CONVENTION (no iron/friction/drive losses) | Note omitted loss channels. |

## 4. Mass & geometry

| Symbol | Value | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| `G` (gravity) | 9.81 m/s² | `energy.py:25` | CoT denominator | CONVENTION | fine |
| robot mass | `mj_model.body_mass.sum()` | `rollout.py:101`, `experiment.py:106`, `metrics.py:202` | CoT denominator | DERIVED-FROM-MODEL | Confirm Menagerie G1 mass ≈ real ~35 kg. |
| distance d | `‖qpos[-1,:2]−qpos[skip,:2]‖` | `metrics.py:203`, `analyze.py:88` | CoT, mean speed | DERIVED (planar, heading-randomized) | `analyze.py:116` uses index 0 not skip for header distance — minor inconsistency. |
| base-height z index | qpos index 2 | `metrics.py:116` | jump-height metric | CONVENTION (free-base z) | OK for floating-base G1. |

## 5. Time & sampling

| Symbol | Value | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| `dt` (control timestep) | `env.dt` (~0.02 s, 50 Hz) | `rollout.py:101`, `experiment.py:105` | energy integral, CoT, mean watts | DERIVED-FROM-MODEL | Confirm 50 Hz (`env.py:84`); energy scales linearly with dt. |
| `TRANSIENT_S` (trim) | 2.0 s | `experiment.py:40`, `analyze.py:27`, `motor_budget.py:27`, `power_compare.py:24`, `probe_speed_hold.py:26` | which samples enter steady-state metrics | CONVENTION | Confirm 2 s removes the transient; five copies — drift risk. |
| rollout steps | 400 / 600 | `experiment.py:46,195`; `rollout.py:21` | window length → averaging stability | DEFAULT | Ensure post-trim window long enough for stable RMS. |

## 6. Reward & spring parameters

| Symbol | Value (units) | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| `energy_reward_weight` | −2.5e-4 | `config.py:60`; `baseline_gate.yaml:13`, `spring_hip_linear.yaml`, `run_baseline.yaml:43` | in-loop objective; **MUST be identical** across conditions | PLACEHOLDER (set by `calib_sweep.sh`) | Confirm calibration; verify byte-identical between matched conditions. |
| calib weight grid | −1e-4 … −1e-3 | `calib_sweep.sh:11` | which weight is chosen | DEFAULT | informs the choice |
| spring `k` (hip linear) | 68.0 N·m/rad | `spring_hip_linear.yaml:14` | post-hoc & in-loop hip-spring saving | DERIVED (k≥0 LS fit, R²≈0.51–0.60) | Re-fit; confirm offline optimum valid in-loop. |
| spring `theta0` | −0.29 rad | `spring_hip_linear.yaml:15`; `experiment.py:190` | same | DERIVED (LS fit) | as above |
| spring `tau0` (constant) | −12.0 N·m | `spring_constant.yaml:12` | constant-element post-hoc saving | DERIVED (degenerate k=0 fit) | Re-confirm offset. |
| semiparabolic `k`,`p1`,`p2` | 42.5, −0.69, 0.11 | `spring_semiparabolic.yaml:17–19` | realizability (k_eff=2k(p2−p1)=68, θ0=(p1+p2)/2=−0.29) | DERIVED | Verify identity & band covers gait. |
| `run_baseline.yaml` reward scales | action_rate −0.01, torques −1e-4, feet_air_time 4.0, base_height_target 0.78, … | `run_baseline.yaml:13–37` | gait shape → indirectly every running-arm energy metric | CONVENTION/PLACEHOLDER (self-flagged TODO) | Spring arm must reuse byte-identically; several flagged to verify. |
| `k_min` (fit clamp) | 0.0 | `metrics.py:64` | fitted spring k floor | CONVENTION | OK |

## 7. Magic numbers & defaults

| Symbol | Value | Location | Feeds | Provenance | Verify at wrap-up |
|---|---|---|---|---|---|
| free-base DoF offset | `6` (`[...,6:]`, `>=6`) | `env.py:66`; `metrics.py:30`; `motor_budget.py:62`; `power_compare.py:42`; `probe_speed_hold.py:51` | which DoFs count as actuated → every whole-body sum, energy reward, actuation share | CONVENTION (floating-base 6-DoF) | Correct only for a single free joint at index 0; load-bearing for EVERY whole-body number. |
| saturation power window | 95th pct (top-5 %) | `metrics.py:155` | torque-vs-speed binding verdict | CONVENTION | Sensitivity to cutoff (90/95/99). |
| saturation statistic | median over hi window | `metrics.py:156` | binding verdict | CONVENTION | OK (robust-vs-max documented). |
| epsilon floors | 1e-9 | `metrics.py:77,94,...`; `experiment.py:165` | division guards | CONVENTION | harmless unless a true value ≈ 1e-9. |
| RNG seed (rollout/sweep) | PRNGKey(0) | `experiment.py:89`; `rollout.py:63` | the single trajectory all metrics use | CONVENTION | Single-seed → no variance estimate before publishing % deltas. |
| baseline trajectory path | `…/2026-06-11_baseline_h100` | `motor_budget.py:23`, `power_compare.py:20`, `probe_speed_hold.py:21`, `recalc_rkt2.py` | source trajectory for all one-off numbers | CONVENTION (hardcoded) | Ensure all scripts point at the canonical current baseline. |

---

### Cross-cutting flags for the proofread pass
- **Duplicated literals that can silently drift:** Kt/R copied as bare `2.3, 0.013` in
  `probe_speed_hold.py:25`; `TRANSIENT_S=2.0` in five files; the `>=6` free-base offset and
  the no-regen clamp reimplemented in 4–6 places. Any correction to Kt/R or the trim window
  must be applied everywhere or the consolidated `metrics.py` path and the one-off scripts
  will diverge.
- **Most load-bearing yet least-verified number is `R/Kt²`** (suspected 8–19× optimistic).
  The `[...,6:]`/`>=6` offset and the no-regen flag are the most load-bearing *assumptions*
  (they define what "whole-body electrical" means).
