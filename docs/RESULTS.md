# Results — Parallel-Elastic Knee Efficiency Study

Running record of experimental results, accumulating across milestones. Terse
session notes live in `JOURNAL.md`; this file holds the numbers, methods, and
caveats in enough detail to write up or reproduce later. Newest milestone last.

Headline so far: a walking G1 baseline is trained and verified; the knee work
loop is **offset-dominated, not stiffness-dominated**, which redirects the
spring design from a torsion spring to a **preloaded constant-torque element**;
the optimistic post-hoc bound is **−16.1 % total knee electrical energy** and
**−41.5 % / −35.8 % knee copper loss** on the fixed baseline gait.

---

## Setup common to all results

- **Env:** `G1JoystickFlatTerrain` (MuJoCo Playground 0.2.0), `impl=jax`.
  Control 50 Hz, physics 500 Hz (10 substeps), 29 actuated DoF, total model
  mass 33.3 kg.
- **Control architecture (verified from the model):** the policy outputs a
  position *target* per joint, `q* = q_default + 0.5·a`, `a ∈ [−1,1]`; the
  actuator applies `τ = kp·(q*−q)` with knee `kp = 75` (no actuator kd; joint
  damping is in the XML). So the logged `qfrc_actuator` *is* the motor torque
  the energy model needs.
- **Knee joints:** `left_knee_joint` (qpos 10, dof 9), `right_knee_joint`
  (qpos 16, dof 15).
- **Energy model** (`src/pea/energy.py`): per actuator
  `P = τ·ω + (τ/Kt)²·R`; negative work is **dissipated, not regenerated**
  (`P ← max(P,0)`) per the project assumption. Motor constants are
  **PLACEHOLDERS** (`Kt = 1.0 N·m/A`, `R = 0.05 Ω`, joint-side): absolute
  watts/joules are not trustworthy, but (a) copper-loss **percentages** are
  Kt/R-independent and (b) baseline-vs-spring comparisons at identical
  constants are valid. Real G1 actuator Kt/R is an open TODO before any
  headline electrical % is published.

---

## Milestone 1 — Baseline trained and replays locally  ✅

**Training** (immers.cloud H100 PCIe 80 GB, 1× GPU, driven over SSH):
- Config `configs/baseline.yaml` (no spring), 200 M env-steps, brax PPO with
  Playground's recommended G1 hyperparameters (8192 envs, batch 256, etc.).
- Wall time **57 min**; steady throughput **~71 k env-steps/s** (≈47 k while
  XLA autotuned the first ~20 M steps). Final eval **episode reward 12.46**;
  reward still climbing at 170 M, so the full 200 M budget is justified for all
  comparison runs.
- Cost ≈ **325 ₽** (~£3) of GPU time for the training itself.
- Run folder: `pea_runs/2026-06-11_baseline_h100/` (config, 20 orbax
  checkpoints, `policy_params`, `metrics.jsonl`).

**Cross-hardware reproducibility:** a Colab T4 ran the *same* config in
parallel (~7–10 k steps/s, abandoned at ~75 M). Its reward-vs-steps curve
overlaps the H100's almost exactly — the learning dynamics are hardware
independent, only wall-clock differs (~7 h vs ~1 h). Plot:
`outputs/reward_curves.png`.

**Local replay (Mac CPU, the M1 acceptance test):** the trained policy walks
**10.76 m in 12.0 s = 0.90 m/s** against a 1.0 m/s forward command, no fall.
Video `pea_runs/2026-06-11_baseline_h100/video.mp4`. (Distance is planar
displacement — initial heading is randomized at reset, so world-frame x alone
misreads a good walk as "backwards"; fixed in rollout/analyze.)

---

## Milestone 2 — Knee gait logged  ✅

From the 12 s baseline rollout (`trajectory.npz`), steady walking (2 s
transient dropped):

| knee  | RMS torque | negative work (12 s) | mean copper loss* |
|-------|-----------:|---------------------:|------------------:|
| left  | 18.9 N·m   | −92.9 J              | 17.8 W            |
| right | 20.5 N·m   | −110.3 J             | 21.0 W            |

\*placeholder Kt/R. **~8–9 W of negative work per knee is being absorbed by the
motors and burned as heat** — precisely the energy a parallel spring can
capture and return. The thesis has a visible target.

**Work-loop finding (the design pivot).** Plotting knee torque vs knee angle
(`outputs/knee_work_loops.png`) shows the loop is **offset-dominated**: the
flexed-knee gait carries a roughly constant ~−12 N·m gravity-support torque,
not a torque that grows linearly with deflection. Consequently the best
*passive* (k ≥ 0) linear spring fitted by least squares **degenerates to
k = 0** — i.e. the optimal simple element is a **preloaded constant-torque
spring**, not a torsion spring. This inverted the original linear-spring
hypothesis and produced `configs/spring_constant.yaml` (`τ₀ = −12 N·m`) as the
lead Milestone-4 candidate.

---

## Milestone 3 — Post-hoc spring subtraction (optimistic bound)  ✅

Method (`pea-analyze --run <run> --spring configs/spring_constant.yaml`):
subtract `τ_spring(θ)` from the recorded motor torque at every logged timestep
on the **unchanged** baseline gait, recompute electrical power per knee. This
is an **upper bound by construction** — it assumes the robot walks identically
with the spring attached, which it would not.

Constant −12 N·m element, no-regen, placeholder Kt/R:

| knee  | electrical          | copper loss        |
|-------|---------------------|--------------------|
| left  | 413.8 → 342.2 J (−17.3 %) | 175.5 → 102.7 J (**−41.5 %**) |
| right | 522.7 → 443.2 J (−15.2 %) | 212.1 → 136.2 J (**−35.8 %**) |
| **total** | **936.5 → 785.5 J (−16.1 %)** | |

Reading: copper loss (the quadratic, torque-squared term) drops 36–42 %, as
expected when a constant offload shaves the torque the motor must supply; the
total electrical figure is smaller because mechanical `τ·ω` work is unchanged
on a fixed gait. **These are optimistic; the credible number requires
retraining (M4).**

---

## Milestone 4 — In-loop retraining  ◻ ready, not yet run

- **Spring injection implemented** (`SpringWrapper` in `src/pea/env.py`): adds
  `τ_spring(θ)` at the knee DoFs through `qfrc_applied` (external generalized
  force), *beside* the motors — deliberately not through the actuators, so
  `qfrc_actuator` keeps meaning "motor torque" and the energy model stays
  honest. Torque is sampled at the control boundary, held across substeps.
  Delegates everything else, so Playground's brax training wrapper and jit/vmap
  compose with it transparently.
- **Verified under jit:** −12 N·m lands on both knee DoFs; a full CPU
  smoke-train (`pea-train --config configs/spring_constant.yaml --smoke`)
  passes end to end through the brax pipeline.
- **To run:** one command on a fresh H100 box,
  `pea-train --config configs/spring_constant.yaml --output_dir ~/runs`
  (~1 h, ~350 ₽), then best-vs-best comparison against the baseline on cost of
  transport and copper loss.

---

## Open items / risks

1. **Real G1 knee Kt, R** — needed before any absolute electrical % is
   published; sets the copper-vs-mechanical blend in the headline number.
2. **Swing-phase cost of an always-engaged preload** — a constant extension
   torque helps in stance but fights the motor in swing and may hurt foot
   clearance. Visible in the data; only in-loop retraining (M4) reveals whether
   the gait can absorb it. (This is why some hardware PEAs add a clutch.)
3. **No-regen assumption** — favourable to the spring by design; if the G1
   actually regenerates, the negative-work win shrinks. Worth a sensitivity
   pass.
4. **Single baseline seed** — for a credible M4 result, train ≥2–3 seeds per
   arm so the comparison is best-vs-best, not single-sample.
