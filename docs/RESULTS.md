# Results — Parallel-Elastic Efficiency Study

Running record of experimental results, accumulating across milestones. Terse
session notes live in `JOURNAL.md`; this file holds the numbers, methods, and
caveats in enough detail to write up or reproduce later. Newest milestone last.

**Headline — the arc.** Targeted parallel elasticity does **not** buy walking efficiency
on a high-geared commercial humanoid (G1), but **does** on a low-gear quadruped (Go1).
**The gear ratio is the crux.**

- **G1 (humanoid, knee 22.5:1, hip-pitch 14.3:1).** The knee work loop is offset-dominated (a passive
  linear spring degenerates to k=0), so the target moved to the **hip-pitch** (a buildable
  linear spring there captures ~51–60 % of mean-square torque). Post-hoc it looked like a
  win (**−3.84 % whole-body**), but the **in-loop retrain REVERSED it to +7.4 % WORSE**, and
  less stable (3/4 vs 4/4 survival). Ohmic is only **~4 %** of the budget; the small post-hoc
  win was no-regen braking recovery (**~0 % under regeneration**). → nine negative results,
  catalogued in **`docs/negative_results.md`**.
- **Go1 (quadruped, 6.33:1).** Ohmic is **54 %** of the budget — the τ² lever is finally
  armed. The calf is *also* offset-dominated (linear spring null, +0.0 %), so the right
  passive element is a **constant knee preload** (τ₀≈3.5 N·m, all four calves). Post-hoc
  **−14.9 %**, and the **in-loop retrain HELD/IMPROVED it** instead of reversing (as the G1
  did). The gait adapts to *exploit* the offload instead of fighting it. The load-carrying
  WALKING program (below) is the validated headline: **cost of transport −14 to −27 % in 3 of
  4 conditions, growing with payload, 3 seeds** (seed 2 a weak −3 to −8 % outlier), with **no
  stability cost at low-to-mid load but survival degrading above ~7.5 kg**. **The one positive
  result.**

The whole difference is **gearing + element kind**: low gear arms the ohmic (τ²) lever, and
a *constant* preload matches the offset-dominated knee work loop and is gait-compatible (no
swing-phase fighting, unlike the G1's angle-dependent linear spring). G1 post-hoc detail is
retained in the milestone sections below.

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
  (`P ← max(P,0)`) per the project assumption. Motor constants for the **Milestone 1–3 post-hoc numbers below** are
  **PLACEHOLDERS** (`Kt = 1.0 N·m/A`, `R = 0.05 Ω`, joint-side); the energy model now
  ships **estimated** G1 constants (`Kt ≈ 2.3`, `R ≈ 0.013`, `R/Kt² ≈ 0.0025`, see
  *Motor constants estimated* below). Either way absolute
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

**Local replay (Mac CPU, the Milestone 1 acceptance test):** the trained policy walks
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
retraining (Milestone 4).**

---

## Milestone 4 — In-loop GO/NO-GO gate (hip-pitch)  ✅ DONE — NEGATIVE (+7.4 %)

**Result (the project's central negative result).** The in-loop hip-pitch linear spring was
trained (matched retrain: same 80 M init, +120 M more steps, same −5e-4 energy weight in the
reward, differing *only* by the spring) and it made G1 walking **+7.4 % WORSE**, not better:
whole-body electrical **151.6 → 162.8 W**, **CoT +8.4 %**, survival **3/4 vs 4/4** for the
matched no-spring baseline. This **reverses** the optimistic offline (post-hoc) estimate of
**−3.84 %** — the clearest evidence in the project that an offline analysis on the recorded
gait can flip sign once the policy is allowed to retrain. The always-on spring absorbs
hip-pitch braking (9.8 → 4.0 W) but the motor then fights it through the drive phase. The
effect is robust: a fresh-from-scratch spring run is no better, and a 2×2 cross-condition
check confirms the retrained spring policy beats a spring-blind one, so it is a real
co-adaptation outcome, not a bug. Catalogued in `docs/negative_results.md` (NR-2/NR-3). The
design and arms that produced this gate are documented below.

Scope pivoted from the knee constant element to a **hip-pitch linear spring**
(see Milestone 2: the knee work loop is offset-dominated, so a passive linear
spring degenerates to k=0 there; the hip-pitch is the AC joint where a buildable
linear spring captures ~51–60 % of mean-square torque). The knee constant
−12 N·m element (`configs/spring_constant.yaml`) remains as the earlier
post-hoc lead; it is superseded as the in-loop candidate.

- **Spring injection implemented** (`SpringWrapper` in `src/pea/env.py`): adds
  `τ_spring(θ)` at the target joint's DoFs (now **hip-pitch**, `cfg.spring.joint`)
  through `qfrc_applied` (external generalized force), *beside* the motors —
  deliberately not through the actuators, so `qfrc_actuator` keeps meaning "motor
  torque" and the energy model stays honest. Torque is sampled at the control
  boundary, held across substeps. Delegates everything else, so Playground's brax
  training wrapper and jit/vmap compose with it transparently.
- **Gate arms (matched, identical reward):** SPRING = `configs/spring_hip_linear.yaml`
  (linear, **k = 68 N·m/rad, θ₀ = −0.29 rad**, the offline hip fit), NO-SPRING =
  `configs/baseline_gate.yaml`. Both carry the same total-electrical penalty in
  the reward (`energy_reward_weight`, placeholder −2.5e-4, to be fixed by the
  calibration sweep). The linear spring is exactly the realizable dual-semi-
  parabolic mechanism within its linear band (`docs/mechanism.md`).
- **Verified under jit + CPU smoke-train:** the hip-pitch spring lands on both
  hip-pitch DoFs and a full CPU smoke-train passes end to end through the brax
  pipeline for both arms (`2026-06-13_spring_hip_linear_smoke`,
  `2026-06-13_baseline_gate_smoke`).
- **How it was run** (immers.cloud H100, rubles): the calibration sweep
  (`scripts/calib_sweep.sh`) picked the energy weight, then
  `pea-train --config configs/spring_hip_linear.yaml` and
  `pea-train --config configs/baseline_gate.yaml` at the matched weight (~1 h each,
  ~350 ₽), followed by the best-vs-best comparison on ohmic loss, cost of transport, and
  total electrical power. Run folders: `2026-06-14_spring_walk_spring_full200` (spring) vs
  `2026-06-14_walk_baseline_full200` (matched no-spring) — see `docs/checkpoints.md`.

---

## Go1 load-carrying walking program  ✅ DONE — POSITIVE

This is the project's **headline positive result**. On the low-gear Go1 quadruped (gear
6.33:1, ohmic ~54 % of the motor electrical budget), an **adaptive per-leg constant knee
preload** — a near-constant offload torque whose magnitude each leg sets for itself from its
own measured knee torque, with no load sensor and no payload observation — is compared against
a **matched no-spring control** (same training, same payload range, the spring is the only
difference). Full mechanism and design in `docs/load_program.md`.

**Headline metric = cost of transport (CoT)** = electrical watts ÷ forward speed (m/s). It is
the only fair metric here because the spring lets the policy walk faster, so raw watts confound
speed; watts alone must always be **speed-matched** before they mean anything. Per-seed CoT
reduction vs the matched no-spring baseline, at three payloads:

| condition | @0 kg | @2.5 kg | @5 kg |
|---|--:|--:|--:|
| seed 1 | −16.6 % | −19.5 % | −22.8 % |
| seed 2 (weak outlier) | −3.4 % | −8.3 % | −6.5 % |
| seed 3 | −13.9 % | −20.1 % | −26.7 % |
| curriculum | −16.8 % | −20.4 % | −22.0 % |

**Reconciled statement:** CoT **−14 to −27 % in 3 of 4 conditions, growing with load; seed 2
is a weak −3 to −8 % outlier.** (Earlier drafts quoted three different bands — −16.7/−19.7,
−17/−20, −14/−27 — now reconciled from the local capacity logs to this single CoT-per-seed-
per-payload definition.) Whole-body electrical at no load: **−9.5 % (seed 1), +0.8 % (seed 2),
−5.6 % (seed 3)**. Figure: `outputs/figures/cot_vs_load.png`.

**Stability cost (reported honestly).** The adaptive policy **loses survival at high load** —
down to ~870–1260 of 1500 simulation steps at roughly ≥7.5 kg payload — while the **matched
no-spring baseline holds 1500/1500**. The honest claim is therefore: **no stability cost at
low-to-mid load; survival degrades above ~7.5 kg.** The energy win is a low-to-mid-load result,
not a free lunch at every load.

**Capacity realism.** The plain MJX simulation lets the Go1 "walk" at up to 30 kg, but the real
Go1 (~12 kg robot) carries ~5–10 kg at most; the sim enforces peak torque but not continuous
(thermal) torque, structure, or balance. The physically meaningful study is **0–6 (10) kg**,
where the result above lives; 15–30 kg numbers are sim-only. See `docs/load_program.md`.

## Motor constants estimated (2026-06-13) — and a sobering implication

Estimated for the 7520-22.5 actuator (knee gear 22.5:1), joint-side,
from a deep web search: **Kt ≈ 2.3 N·m/A (2.0–2.7), R ≈ 0.013 Ω (0.009–0.025)**;
gear-invariant **R/Kt² ≈ 0.0025** is the load-bearing quantity. Estimates only
(no datasheet/hardware) — report cost of transport as a band, lead with the
Kt/R-independent ohmic-% reduction. Now in `energy.py` (replacing placeholders).
(Gear note: the **hip-pitch is 14.3:1**, not 22.5:1 — per `docs/robot_inventory.md`;
the dispute over the exact hip-pitch ratio is **unresolved** and swings that joint's
ohmic coefficient ~2.5×, but ohmic stays **~4 %** of the budget either way, so it does
not change any conclusion here.)

> **Retraction (supersedes an earlier alarm).** An earlier note worried that the
> code's `R/Kt² ≈ 0.0025` might be "~10× too optimistic / untrustworthy." That alarm
> is **withdrawn** — it was a **100× arithmetic error** on our side, not a real
> mismatch. The code value 0.0025 sits at the **optimistic edge of a defensible band
> (0.001–0.020)**, so it is fine to use. The standing caveat is the honest one: all
> motor constants (Kt, R) are **estimates** for both robots, so the **relative**
> spring-vs-no-spring percentage is constant-invariant and safe, but any **absolute**
> watt or CoT number is a **band, not a publication-grade figure**.

**Implication (important).** With realistic constants, ohmic loss is only **~4 %
of total electrical energy** (the placeholders made it ~48 %). The 22.5:1 gearing
lets the motor make 120 N·m at ~50 A, so I²R heating is small. So the original
"copper loss ∝ τ², cut quadratically" motivation is true but applies to a small
slice of the budget for a geared humanoid.

**Iron (core) loss caveat — quantified.** The energy model above counts only mechanical
work (τ·ω) and ohmic heating ((τ/Kt)²·R). It **omits iron loss** — the heat lost in the
motor's magnetic core (hysteresis + eddy currents). Iron loss grows with **motor speed**,
not torque, so a torque-offloading spring **cannot reduce it**; including it therefore
**dilutes** the percentage savings (a fixed saving spread over a larger total). Estimated
upper bounds: **G1 whole-body iron loss likely ~25–35 W (14–20 % of the ~178 W budget),
worst case ~58–65 W (33–37 %); Go1 negligible (~3–5 W, < 5 %).** The G1 is hit far harder
because its high gear spins the rotor about 10× faster than the Go1's, and eddy-current loss
grows with frequency squared. Effect on the headlines: the G1 walking offline saving shifts
from −2.9 % to about **−2.1 to −2.4 %**; the Go1 −14 to −27 % shifts to about **−13.3 to
−26.8 % (essentially unchanged)**. This is an **estimate** — a single no-load spin-down test
of a real motor would pin it. **Conclusion: every headline conclusion survives a full
iron-loss accounting** (the G1 stays a small, negative-after-retrain lever; the Go1 stays a
large positive one).

Offline (post-hoc) hip-pitch linear spring on the baseline, realistic constants:
**ohmic −51…−59 % (large, small base); total electrical at the hip joint −7…−14 %**.
That larger total reduction comes from the spring offloading *positive mechanical
work* under no-regeneration, not from ohmic. Two consequences:
1. The headline total-electrical CoT reduction will be **more modest than the
   −16 % the placeholder implied**, concentrated at the hip, smaller whole-body.
2. It is **sensitive to the no-regeneration assumption** (a "replace costly
   positive motor work" effect) — physically right for the geared G1, but also
   the assumption most favourable to the spring; say so.
Framing: lead with the ohmic-% reduction (large, Kt/R-independent); report the
total-electrical CoT reduction honestly as modest and assumption-sensitive.

## Motor budget, actuation share, and direct N→M power (2026-06-14)

Measured on the baseline trajectory (`scripts/motor_budget.py`,
`power_compare.py`, `probe_speed_hold.py`), 10 s steady walk, estimated Kt/R,
no-regen:

- **Whole-body motor electrical ≈ 178 W**; mechanical ~96 %, **ohmic ~4 %** (69 J
  of 1785 J). Per-joint share: right knee 20.6 %, left knee 16.1 %, **right
  hip-pitch 13.8 %, left hip-pitch 13.5 % (both = 27.3 %)**, shoulders ~13 %.
- **Direct N→M (hip-pitch linear spring, post-hoc, fixed gait):** hip-pitch
  motors **48.7 → 43.5 W** (−5.2 W, −10.7 %); whole-body **178.5 → 173.3 W**
  (−5.2 W, −2.9 %). All of it is hip-pitch (the post-hoc only touches those DoFs).
- **No-regeneration tax:** with regen the bill is 135.5 W; no-regen is 178.5 W —
  the **+43 W** gap is braking work dumped as heat (**+32 %** over the regen bill, i.e. **~24 %** of the no-regen bill — the single ~24 % figure used elsewhere). With the spring the
  regen-view bill barely moves (135.5 → 135.6 W), so the entire 5.2 W saving is
  braking recovery, **not** ohmic. The spring acts as a passive substitute for
  regeneration; its win is as no-regen-dependent as that implies.
- **Actuation share of total robot power** (research workflow): G1 battery 421 Wh
  (~210 W mixed-use average); steady walking ~250–350 W (cf. Cassie ~300 W,
  ANYmal ~280 W). House load (Jetson Orin NX 10–25 W + Livox 6.5 W + RealSense
  ~2 W + standby) ≈ 40–50 W → **actuation ≈ 80–90 % while walking** (majority is
  house load when standing). The spring's ~5 W motor saving is **~2 % of
  whole-robot power**, ~0 % under regen.
- **Speed/hold probes:** standing → walking-tuned spring saves ~0 W; faster
  walking (1.23 m/s) → ohmic share flat ~3.8 %, braking 46→63 W, spring gain
  5→6 W. Speed alone does not arm the copper lever; running needs its own policy.

Strategic consequence: the six-direction map in **`docs/directions.md`** — TRY
in-loop G1 → running G1/H1 → quadrupeds (zero-shot conditioned) → DecART; SKIP
static manipulation.

## Open items / risks

1. **Real G1 knee Kt, R** — needed before any absolute electrical % is
   published; sets the copper-vs-mechanical blend in the headline number.
2. **Swing-phase cost of an always-engaged preload** — a constant extension
   torque helps in stance but fights the motor in swing and may hurt foot
   clearance. Visible in the data; only in-loop retraining (Milestone 4) reveals whether
   the gait can absorb it. (This is why some hardware PEAs add a clutch.)
3. **No-regen assumption** — now JUSTIFIED for the G1 (back-EMF below the ~48 V bus
   at locomotion speeds → no battery recovery without a boost converter the driver
   lacks; documented regenerative-resistor pattern). NOT a verified spec; the spring's
   entire win is this dissipated braking, so report the ~24 % regen-vs-no-regen
   sensitivity. Exceptions: MIT Cheetah 2013, reportedly Tesla Optimus. See
   `docs/running_program.md`. Gold standard: a hardware braking test.
4. **Single baseline seed** — for a credible Milestone 4 result, train ≥2–3 seeds per
   arm so the comparison is best-vs-best, not single-sample.
5. **Baseline is ENERGY-NAIVE (the key confound).** Every number above is measured
   against the `2026-06-11` walker, trained with the energy term at ZERO — a policy
   that never tried to save energy. The proper control is an **energy-aware baseline
   (no spring, electrical penalty on)**, never trained. The spring's real value is
   `CoT(energy-aware+spring) − CoT(energy-aware, no spring)`, not vs the naive walker;
   an energy-aware policy will already trim torque (and likely self-de-chatter), so
   these post-hoc %s are UPPER BOUNDS and the marginal in-loop number could be
   smaller. Train the energy-aware baseline FIRST (see `docs/running_program.md`
   Milestone 1b).
