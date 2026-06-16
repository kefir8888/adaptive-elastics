# Negative results — Parallel-Elastic Efficiency Study

The load-bearing output of the study so far. On a high-geared commercial humanoid
(Unitree G1), targeted parallel elasticity for walking efficiency **does not pay**,
and we can say precisely why. **The counterpoint:** on a *low-gear quadruped* (Go1) it
*does* pay — a constant knee preload cuts cost of transport **−14 to −27% in 3 of 4
conditions, growing with load** (3 seeds; seed 2 a weak −3 to −8% outlier), with **no
stability cost at low-to-mid load but survival degrading above ~7.5 kg** (see the resolved
item at the end, and `RESULTS.md`); **gear ratio is the crux.** Each entry is tagged **EXPERIMENTAL** (trained
policies + measured energy) or **REASONED** (from real specs / actuator physics /
literature).

**Conventions.** Electrical energy = mechanical + ohmic per actuated DoF, no
regeneration: `P = max(τ·ω + (τ/Kt)²·R, 0)`. Estimated G1 constants (R/Kt² ≈
0.0025); the spring-vs-no-spring **relative %** is R/Kt²-invariant, absolute watts
are a band. **Post-hoc** = subtract `τ_spring(θ)` from the recorded motor torque on
the *fixed* baseline gait (optimistic upper bound). **In-loop** = inject the spring
and *retrain*, so the gait adapts (the credible test). All G1 numbers are single-seed
(multi-seed deferred); robustness checks noted per item.

---

## NR-1 — A linear parallel spring at the G1 KNEE is degenerate (offset-dominated)
**EXPERIMENTAL** (work-loop analysis, Milestone 2).
The knee carries a near-constant gravity-support torque across the gait, so its
torque–angle work loop is **offset-dominated**, not sign-changing. A least-squares
linear-spring fit collapses to **k ≈ 0** — a linear restoring spring captures almost
none of the knee torque. The knee cannot host a useful *linear* parallel spring;
only a constant-torque (preload) element fits, and that was **never validated
in-loop**. This is *why* the spring target was moved off the knee.
- Metric: linear fit `k → 0` (clamped). A post-hoc *constant* element gave −16.1%
  per-knee total electrical / −36–42% per-knee ohmic on the *fixed* gait (optimistic,
  placeholder constants, per-joint not whole-body, never retrained).

## NR-2 — A linear parallel spring at the G1 HIP-PITCH fails IN-LOOP — the headline
**EXPERIMENTAL** (matched in-loop retrain). The central result.
Hip-pitch is the AC joint where a linear spring *does* fit (post-hoc R² ≈ 0.57,
k = 66 N·m/rad, θ₀ = −0.27). **Post-hoc it looked like a win: −3.84% whole-body
electrical** on the fixed gait. But when the gait is free to adapt (in-loop retrain,
matched: same 80M init, +120M, same energy weight, differ *only* by the spring), the
result **reverses**:
- whole-body electrical **151.6 → 162.8 W = +7.4% WORSE** (4-seed mean; single worst
  deterministic seed +20%, 148→178 W); **CoT +8.4%**.
- **stability degrades**: baseline survives 4/4 reset-seed rollouts, spring 3/4; the
  deterministic spring rollout falls at 10.4 s (baseline does not).
- mechanism (indicative, single rollout): the spring *does* absorb hip-pitch braking
  (9.8 → 4.0 W) but the motor pays **more fighting the always-on spring in the driving
  phase** (hip-pitch positive-work electrical 44.8 → 55.7 W). Net negative.

**Robustness:** a *fresh-from-scratch* spring policy is no better (rules out
warm-start bias); a **2×2 cross-condition test** confirms the retraining genuinely
adapts — the spring-trained policy beats a spring-*blind* one in the spring world
(survives 3/4 vs 2/4, 162.8 vs 165.6 W) — so this is a **real physical effect, not a
training/plumbing artifact**. The spring is also verified present in the *training*
env, and the spring joint is in the policy's observation.

## NR-3 — The post-hoc bound can have the WRONG SIGN (methodological)
**EXPERIMENTAL** corollary of NR-2.
The optimistic post-hoc fixed-gait analysis said **−3.84% (improvement)**; the in-loop
truth is **+7.4% (worse)**. So post-hoc spring-subtraction is not merely an optimistic
*upper bound* — for an **always-engaged** spring it can be qualitatively wrong (flip
sign), because it ignores the gait's forced adaptation to the always-on torque.
Implication: post-hoc parallel-spring analyses that never retrain may overstate, or
**invert**, the real benefit. This is the strongest possible vindication of the
project's "post-hoc alone is not sufficient" thesis.

## NR-4 — The ohmic (quadratic-copper) lever is gear-killed on the G1 (~4% of budget)
**EXPERIMENTAL.**
The motivating mechanism — copper loss ∝ τ², offload torque, cut heat quadratically —
is **nearly absent on a high-geared humanoid**. On the G1 (knee/hip 22.5:1) ohmic loss
is only **~4% of the motor electrical budget (6.3 W of 148.7 W)**; the other ~96% is
mechanical work the spring cannot remove. So even the best-case post-hoc win (~3%) is
small *and does not come from the advertised quadratic channel*.

## NR-5 — The small post-hoc win is entirely no-regeneration-dependent
**EXPERIMENTAL.**
The ~3% post-hoc saving is braking energy the spring intercepts that the motor would
otherwise dissipate. Under a drivetrain that **regenerates**, the saving → **~0%**.
The spring is a passive *substitute for regeneration*, not a copper-loss reducer; its
win lives entirely in the **+43 W (+32%)** no-regen braking tax and vanishes if braking
energy can be recovered.

## NR-6 — An energy-reward penalty is not a control lever on the G1 walk
**EXPERIMENTAL** (5-weight calibration sweep).
Sweeping the total-electrical reward weight across a **10× range (−1e-4 … −1e-3)** left
measured walking electrical power **flat (153–159 W, no trend)** at fixed speed. The
walking budget is mechanical-work-dominated, so the policy cannot shed energy by gait
changes without walking slower (which tracking forbids). You **cannot train a
meaningfully-more-efficient G1 walker via an energy penalty**.

## NR-7 — A parallel spring cannot raise G1 jump HEIGHT (knee is speed-limited)
**REASONED** (model `jnt_actfrcrange` + Unitree G1 URDF velocity limits).
The G1 knee is **speed-limited** (20 rad/s; the walker already uses 52–67%). A parallel
spring adds *force, not speed*, so it cannot beat the speed wall a max jump hits at the
knee. Jump height points **off the G1** (series compliance or a low-gear platform).

## NR-8 — The clutch cannot rescue the parallel spring in RUNNING
**REASONED** (actuator physics).
The diagnosed failure (NR-2) is the always-on spring fighting the drive phase. The
obvious fix — gate the spring by gait phase — is **not realizable in running**: a
running stance is ~100–150 ms and no real clutch toggles on/off cleanly within it,
under load, every stride. The project's dead-zone clutch is *passive and
angle-triggered* (a between-**mode** switch, not a within-stride phase modulator), so
it cannot do braking-vs-drive gating. Running's elastic energy wants a **series**
element (passive bounce, no switching), not a parallel spring plus clutch.

## NR-9 — The adjustable-spring mechanism is not novel
**REASONED** (literature; `mechanism.md` / `related_work.md`).
The dual half-parabola → tunable linear spring is prior art (Hurst 2004 AMASC;
Migliore 2005; the group's own Belov/Osokin 2024). Only **full passive disengagement**
(the dead-zone clutch) survives as a mechanism-level novelty; the rest of the
contribution is integration (parallel + in-loop RL co-adaptation + commercial humanoid
+ electrical accounting).

---

## Caveat that makes the picture worse, not better
The energy model **omits iron (core) loss** (hysteresis + eddy, ∝ motor *speed*). A
parallel spring offloads *torque, not speed*, so it cannot cut iron loss; including it
would enlarge the denominator and **dilute** the (already negative) result. **Quantified
upper bound:** G1 whole-body iron loss likely **~25–35 W (14–20% of the ~178 W budget)**,
worst case ~58–65 W (33–37%); Go1 negligible (~3–5 W, <5%), because the G1's high gear
spins its rotor ~10× faster than the Go1's and eddy loss grows with frequency squared.
Effect on the headlines: G1 walking offline −2.9% → ~−2.1 to −2.4%; Go1 −14 to −27% →
~−13.3 to −26.8% (essentially unchanged). It is an estimate (a no-load spin-down test would
pin it), and **all conclusions survive it**. The G1 numbers above are therefore, if anything,
slightly optimistic. (See `RESULTS.md` and the `energy.py` scope note.)

## What is NOT yet a negative result (open / under test)
- **Go1 quadruped (low gear 6.33:1): RESOLVED — POSITIVE.** Ohmic is **54%** of the budget;
  the calf is also offset-dominated (NR-1), so a **constant knee preload** is the buildable
  optimum, and in-loop it HELD/IMPROVED the offline estimate instead of reversing (post-hoc
  −14.9%). The validated load-carrying WALKING program cuts **cost of transport −14 to −27%
  in 3 of 4 conditions, growing with load** (3 seeds; seed 2 a weak −3 to −8% outlier), with
  **no stability cost at low-to-mid load but survival degrading above ~7.5 kg** (per-seed table
  in `RESULTS.md`). **NR-4/5/6 are G1-specific and do NOT transfer to low gear.** This is the
  payoff that gives the negatives their meaning.
- **A per-stride stance/swing clutch on WALKING** (feasible, ~1 Hz, Collins-2015 style):
  could rescue the hip spring (NR-2) by freeing it in swing; untested.
- **A constant-torque element at the G1 knee** (NR-1): post-hoc only, never in-loop.
