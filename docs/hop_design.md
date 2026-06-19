# G1 continuous-hopping for energy — design (2026-06-19)

The Part-2 explosive-move track, restarted on the humanoid G1. Decision (user,
2026-06-19): test a parallel knee/ankle spring on **continuous, two-footed, in-place
hopping** (jump → land → jump → …) for **electrical-energy** benefit. This is the
**repetitive** corner of Part 2: unlike a one-shot jump, a sustained hop has
cross-cycle energy recovery, so the spring's two channels (recover landing-braking
energy + offload push-off torque) both apply every cycle.

## Why hopping (and why this is the defensible spring case)

- **Repetitive ⇒ the energy story is real.** A single max jump is one-shot (no
  recovery; cost-of-transport undefined). Continuous hopping recovers braking energy
  each landing — exactly the channel the no-regen G1 dumps as heat and the spring can
  intercept. Landing-braking dominates the hop energy budget, so the spring's win is
  **larger and cleaner than walking's ~3 %**.
- **Landing dodges the speed wall.** Part 2's B1 result stands: a parallel spring
  **cannot raise G1 jump HEIGHT** because the knee is speed-limited (20 rad/s; the
  walker already uses 52–67 %), and a parallel spring adds force, not speed
  (`docs/negative_results.md` NR-7). But **landing load is set by impact velocity,
  not motor speed**, so the cap does not bind there. We therefore target **energy and
  peak load (B2)**, not height — the strongest explosive story on the stock G1
  (`docs/directions.md`).
- **Reverses the old "hopping dropped" note** (`docs/running_program.md`,
  2026-06-14): that was about a *Raibert one-leg hopper*. This is a full humanoid
  doing *both-feet* hops — a different, legitimate task.

## Metric — energy per hop, NOT cost of transport

In-place hopping has **zero forward speed**, so CoT (W ÷ m/s) is undefined. Headline:
**electrical energy per hop (J/hop)**, or **mean electrical power at a matched hop
height + cadence**. Both arms must hop at the **same apex and cadence** (a controlled
mechanical task) so the comparison isolates the spring. Electrical model is the
project standard: per actuated DoF `P = max(τ·ω + (τ/Kt)²·R, 0)` (mechanical + ohmic,
no-regen), τ the **motor** torque (`qfrc_actuator`), so the passive spring (injected
via `qfrc_applied`) is correctly excluded. **Regen sensitivity matters more here than
for walking**: the hop win is braking-recovery-dominated, so report no-regen **and**
the regen lower bound (keep negative `τ·ω`) — the gap is the regen-dependent share.

## The environment subclass — `G1JoystickHop` (`src/pea/g1_hop_env.py`)

A minimal subclass of the Playground G1 walk env, same pattern as `g1_run_env.py`.
The stock walk env forbids a two-footed hop in three config-unreachable ways, all
overridden here: the **anti-phase gait clock** (`feet_phase`, rewards stepping), the
**stand-still objective** (`stand_still`, punishes the crouch at zero command), and
the **air-time clip** (caps reward at a fast-walk stride). Changes:

- **Synchronous clock** — `reset` sets `phase = [0, 0]` (both feet in phase → one
  whole-body bounce, not a step) and draws the rate from a hop range (1.5–2.2 Hz).
  The policy observes `phase`, so it locks onto whatever cadence runs → fixing the
  cadence fixes the task.
- **Air-time un-capped** (0.5 → 0.8 s); walk-only terms neutralized in the default
  config (`feet_phase`, `stand_still`, `joint_deviation_knee` → 0; `pose` → −0.05);
  `max_contact_force` raised 500 → 2000 N (the walking value would crush hop landings).
- **In-place command** (`lin_vel_* = [0, 0]`): `tracking_lin_vel/ang_vel` then reward
  not drifting. A forward hop is reachable by widening `lin_vel_x`.
- **Five added reward terms** (the design that defeats the stand-still trap):
  | term | form | role |
  |---|---|---|
  | `hop_rhythm` | `2·(s·both_air + (1−s)·both_down) − 1`, signed [−1,1] | **backbone** — periodic flight is the task; grounded-during-flight is a penalty, so standing nets ~0 (no free floor), a hop in phase scores ~+1 |
  | `hop_push` | `(≥1 foot down)·clip(pelvis v_z, 0, 2)` | **dense bootstrap** — the only continuous gradient across the crouch→push-off, where airborne terms have none |
  | `hop_height` | `both_air · (apex above standing)`; maximize, or track `hop_height_target` | amplitude (the matched-task knob) |
  | `hop_sync` | `1 − |c_L − c_R|` | forbid asymmetric skips (two-footed, not a step) |
  | `hop_flight` | `both_air` | reward leaving the ground at all |

### The key design problem: the stand-still trap

At zero command the stock env rewards standing still (~1.75/step from tracking
alone). A naive hop reward loses to "just stand," and worse, the path to hopping runs
*downhill* first (the crouch pays nothing, height has zero gradient until already
airborne) — a reward **barrier**, not a slope (found in adversarial review,
2026-06-19). Solved by: (1) the **signed** `hop_rhythm` (removes standing's free
floor and penalizes being grounded when the clock says fly), and (2) the dense
`hop_push` (rewards generating upward velocity *while still on the ground*, bridging
stand → crouch → push → fly). Verified term-by-term that a synchronized hop now
strictly out-scores standing and every intermediate state is uphill.

## Curriculum (staged, each gated; cheap probe before every GPU run)

0. **S1 — elicit** (`configs/g1_hop_s1.yaml`): from scratch, maximize height, sampled
   cadence, softened termination (−50), smoothness on, **no spring, no energy
   penalty**. GATE at eval: a **real both-feet-airborne window each cycle**
   (`min over cycle of (c_L + c_R == 0)`), a **steady cadence**, and **survival** the
   full episode. Failure modes + knobs below.
1. **S2 — matched-task arms** (fork BOTH from the S1 checkpoint):
   - **Baseline** (`configs/g1_hop_baseline.yaml`): no spring, **fixed cadence
     (`hop_freq`) + fixed apex (`hop_height_target`)**, energy penalty **ON**,
     termination −100.
   - **Spring** (`configs/g1_hop_spring.yaml`): byte-identical to the baseline except
     the spring block. Do **not** warm-start the spring arm from the no-spring policy
     (anchors the gait, hides co-adaptation).
   - Compare **J/hop** at the matched apex+cadence, with survival, ≥3 seeds/arm.

## The spring — a pogo-stick element

The hop knee/ankle load is offset-dominated (extensor-sign throughout), and the
*work* changes sign across the bounce (absorb at landing, return at push-off). The
right element is a **one-sided stiffness spring** (`springs.py` kind
`one_sided_linear`) that engages as the joint flexes (landing/crouch) and returns the
stored energy on extension (push-off) — a pogo stick. Fit `(theta_engage, engage_sign,
k)` from a **baseline-hop work-loop**; **plot ALL leg joints — the ANKLE is a strong
candidate** (Achilles analogue; weaker 50 N·m actuator over a long arc → a spring's
% share is larger) — and let the data choose the joint. The spring config currently
holds `k = 0` placeholders; **fill them before training the spring arm**.

## Validity checklist (the watchdog)

- **Parity**: `g1_hop_baseline.yaml` and `g1_hop_spring.yaml` must be byte-identical
  except `spring.kind`. S1 is **elicit-only**, *not* a comparison arm (it differs in
  termination, torque penalty, energy weight, cadence — using it as the baseline is
  invalid).
- **Calibrate `hop_height_target`** from the S1 rollout (the mean apex actually
  reached), so both arms are asked for an apex they can hit — else the comparison
  becomes "who can reach 0.20 m" not "who is more efficient at 0.20 m".
- **Eval at the trained condition**: the eval env fixes the same `hop_freq`; verify
  the achieved apex matches between arms (±2 cm) or normalize energy to equal apex.
- **Energy honesty**: confirmed — spring torque is `qfrc_applied`, energy uses
  `qfrc_actuator`; the `torques` penalty also reads `actuator_force`, so the spring
  is not double-counted.
- **Seeds ≥3** per arm (the Go1 program's seed 2 was a weak outlier); report spread.
- **Regen sensitivity** reported (see Metric).

## Failure modes → knobs (for S1)

- **Never leaves the ground / just bobs** → raise `hop_flight`, `hop_push`,
  `hop_rhythm`; soften `termination` further; raise `max_foot_height`.
- **One big jump then stands** (no repetition) → the signed `hop_rhythm` already
  penalizes the long stand; if it persists, fix `hop_freq` so repeated hops are
  required, or add a per-landing cadence reward.
- **Asymmetric skip / hopscotch** → raise `hop_sync`; `hop_height`/`hop_flight`
  already require BOTH feet airborne, so amplitude is sync-gated.
- **Slamming landings** → tighten `contact_force` / lower `max_contact_force` (but
  remember impact is the task; do not over-suppress it before the spring exists).

## Status

Env subclass built and validated locally (reset/step + full PPO `--smoke` pipeline,
2026-06-19). Configs: `g1_hop_s1` (run first), `g1_hop_baseline` + `g1_hop_spring`
(matched pair, params TBD from the work-loop). **Next: train S1 on the GPU, gate on a
real flight window, then build the work-loop and the matched comparison.**
