# Getting a G1-class humanoid to RUN (flight phase) in RL sim — research

*Written 2026-06-16. Question: is RL humanoid running a known-hard problem, or did we make
simple mistakes? We use brax PPO on MuJoCo Playground's `G1JoystickFlatTerrain`; two attempts
failed (a `[0,3]` m/s from-scratch collapse to a 0.85 m/s never-fall walk; and a
curriculum + reward-redesign that destabilised it). Companion to `docs/g1_running_design.md`
(our staged design) and `docs/running_program.md`.*

## Is it hard

Yes — but "hard" in a specific, well-documented way. The literature is unanimous on two points:

1. **A fast WALK is easy; a true RUN with a flight phase is a distinct, harder regime.** Walking
   (single/double support, no aerial phase) is the standard joystick result and trains in minutes
   on GPU. Running requires a *both-feet-airborne* window, short stance (~0.2 s), and an active
   push-off — a different attractor that velocity-tracking reward alone does **not** reliably
   reach. Essentially every group that reports running adds *running-specific machinery* (below);
   nobody reports it falling out of a plain walk reward by just raising the command ceiling.

2. **The named failure mode is exactly ours.** "RL policies struggle to acquire high-speed
   behaviors when such commands are introduced prematurely … gradually increasing the range of
   command velocities over time leads to more stable learning" (curriculum survey). Conservative
   collapse — *"becoming overly conservative and failing to progress when relying on reward
   shaping alone"* — is a recognised PPO-on-humanoid pathology, not a bug unique to us. A wide
   command range from scratch plus a large fall penalty is the textbook way to land in the
   slow-but-safe local optimum we hit.

So: running is genuinely a harder problem than walking, **and** our specific collapse is a
standard, named pitfall. Both are true at once. It is achievable for a G1 — see below — but not
with a config-only tweak of the stock walk env.

**It IS known-achievable for a G1-class robot.** Real, deployed results:
- **Unitree G1, 3.3 m/s with a genuine flight phase** ("Chasing Autonomy", arXiv 2603.25902) —
  RL **guided by retargeted human running references**, not pure reward shaping.
- **Gait-conditioned curriculum** (arXiv 2505.20619) — G1, PPO, Isaac Gym, a single recurrent
  policy doing stand/walk/run up to a **4.0 m/s** target, run mode gated by the **Froude number
  Fr = v²/(g·l) > 0.5**.
- AMP character control (arXiv 2104.02180): a mocap locomotion prior yields speed-dependent
  gaits — walk ~1 m/s, jog ~2.5 m/s, **run ~4.5 m/s** — emergent from the motion prior.

## How people do it

Recurring ingredients across the running-humanoid papers (in rough order of how load-bearing):

- **Reference motion / imitation is the single most reliable lever for a *clean* flight phase.**
  - **Motion retargeting + RL** (Chasing Autonomy): optimise a periodic reference from one human
    running clip, then track it. Their flight phase comes largely from the reference, not from
    hand-tuned rewards.
  - **DeepMimic** (imitation): tracking-error reward against a mocap clip; reproduces dynamic,
    aerial skills with high fidelity. Needs a clip and phase alignment.
  - **AMP / adversarial motion priors** (arXiv 2104.02180): a discriminator learns the "looks like
    running" reward from a mocap *distribution* (no per-step phase alignment). Gives natural
    speed-dependent gaits and **emergent flight** at high commanded speed. ~39 M samples / ~6 min
    on 4096 envs. This is the most popular middle ground: style from data, task from reward.
  - Demonstration-free / reduced-order-model-guided RL exists (arXiv 2509.19023) but is the harder
    road; most running results lean on *some* motion data.

- **Explicit running rewards (when going reference-free)** — the gait-conditioned curriculum paper
  spells out the minimal set that *induces* a flight phase from reward alone:
  - **alternating single-leg contact + flight** (reward the both-airborne window),
  - **push-off**: reward strong vertical+forward velocity at toe-off,
  - **short contact / stance-time penalty**: penalise prolonged stance to force dynamic running,
  - a **feet-swing-height** term.
  These are *additive to* velocity tracking and only switch on in run mode.

- **A periodic gait clock / phase variable** in the observation (sine–cosine phase), and rewards
  keyed to it (stance vs. flight indicators). This is standard in MJX/Brax humanoid work too — the
  REEM-C Brax+MJX paper (arXiv 2407.05148, **our exact stack**) uses a periodic clock signal plus
  `I_stance`/`I_flight` foot-contact/-velocity indicators. Note: *that* paper deliberately tunes
  the clock for **walking** (0.35 s double + 0.75 s single support, top **1.0 m/s**, no flight) —
  which is the point: the clock is a knob; **how you set it decides walk vs. run.** The Playground
  G1 env's clock is set for walking, and (per our finding) the run-relevant literals are hardcoded.

- **A velocity curriculum**, not a wide range from scratch. Grid/adaptive curricula sample command
  velocity from a distribution that *expands only when the agent earns reward* (e.g. start
  `[-1,1]`, widen toward `[-3,3]`), and some **shrink lateral/yaw range as forward speed rises**
  and **shorten the gait-cycle time as commanded speed rises** (stride-frequency curriculum). The
  Froude-gated approach is one principled version of "only ask for a run once a fast walk exists."

- **Termination shaping (DeepMimic-style), tuned soft.** Terminate on torso/non-foot ground
  contact (+ base-height / orientation gates). The *magnitude* matters: a large flat fall penalty
  with a wide command makes "walk slowly, never fall" dominate. An **alive/per-step bonus** plus a
  *moderate* termination penalty is the more common modern choice (e.g. the "15-minute" minimal
  recipe, arXiv 2512.01996).

- **Symmetry / mirror loss** (auxiliary *loss*, not reward) for natural, symmetric gait and a
  smaller search space — but it is **a polish lever, not a flight-enabler**: shown beneficial only
  *within* a curriculum, "no significant improvement over vanilla PPO" without one (arXiv 1801.08093).
  Do not reach for it to fix a collapse.

- **Simulator.** Isaac Gym dominates the *running* humanoid literature (4096–8192 envs). MJX/Brax
  (our stack) is proven for humanoid locomotion (Playground G1/H1/Berkeley/T1 joystick **walking**
  with sim-to-real; REEM-C walking) but published **running** humanoid results on MJX are thin —
  most flight-phase results are Isaac Gym. MJX contact is solver-based/softer and a known tuning
  axis (stiffness/damping/armature); it is not a blocker but it is less battle-tested for the
  high-impact, short-stance contact of running. This is a real, if secondary, headwind for us.

## Our mistakes vs standard pitfalls

| Our symptom | Standard pitfall? | Verdict |
|---|---|---|
| `[0,3]` m/s **from scratch** → collapse to 0.85 m/s never-fall walk | **Yes** — "commands introduced prematurely"; the canonical fix is a velocity curriculum that widens only on success. | Self-inflicted, textbook. |
| Large `termination -100` with a wide command → policy plays it safe | **Yes** — conservative-collapse local optimum; modern recipes use a moderate termination + an alive bonus. | Self-inflicted. Our S1 softening to −50 is the right direction. |
| Curriculum + reward-redesign "destabilised it" | **Partly** — changing many reward terms at once across a stage boundary without warm-start/regression-gating is a known way to lose a working gait; staged warm-starts with a single dominant change per stage are the discipline. | Mostly self-inflicted (process), but see next row. |
| No flight phase ever appeared | **Structural, NOT just a mistake.** Our own audit (`g1_running_design.md`) found the Playground walk env *mathematically forbids* a double-float: `feet_phase` pays +1/step for keeping exactly one foot planted, `_reward_feet_air_time` hard-clips at 0.3 s, and `gait_freq`/`threshold_max` are **hardcoded literals, not config fields**. | **Real env limitation.** Config-only tuning caps at a fast walk. A small env subclass is required — this is correct and matches the literature (running needs running-specific machinery, not a re-weight). |
| Tried to get flight from **reward shaping alone, no reference motion** | **The harder road by design.** Every clean-flight G1 result used reference motion (retarget/DeepMimic/AMP). Reward-only running is *possible* (Froude-gated push-off/short-contact rewards) but is the lower-probability path and needs the env subclass first. | Strategic gap, not a bug. |

Net: **most of our failures are standard, self-inflicted pitfalls** (wide-range-from-scratch,
heavy fall penalty, too-many-changes-at-once), layered on **one real structural limit** of the
Playground walk env (the gait clock + air-time cap forbid a flight phase config-only). We did not
do anything exotic-wrong; we hit the two most common walls in sequence.

## Minimal recipe to try

Two tracks. **Track A (reward-only)** is cheaper and stays on-method (we want emergent gaits for
the spring study, and avoiding mocap keeps the comparison clean). **Track B (reference motion)** is
the higher-probability route to a *fast, clean* run if Track A plateaus.

**Track A — reward-only, staged (matches `g1_running_design.md`):**
1. **Velocity curriculum, never wide-from-scratch.** Start `[0, 1.0]`, widen the *upper* bound only
   after the policy tracks the current ceiling (gate on `tracking_lin_vel`). Don't jump to 3 m/s.
2. **Soft termination + alive bonus.** Termination ≈ −20 to −50 (not −100) during run-eliciting
   stages; add/keep a per-step alive bonus; restore a stiffer penalty only to consolidate.
3. **Build the env subclass first** (the hardcoded literals make this unavoidable): un-cap
   `_reward_feet_air_time` (threshold_max ≈ 0.8 s), widen `gait_freq` sampling (~1.6–2.2 Hz), and
   **down-weight `feet_phase` 1.0 → ~0.3** so a double-float stops being forbidden.
4. **Add the three running rewards, velocity-gated** (only above ~1.5 m/s, each clipped small):
   `flight_bonus = +k·(1−cL)·(1−cR)·I[vx>1.5]`, a **short-stance / contact-time** penalty, and a
   **push-off** reward (vertical+forward foot velocity at toe-off). Keep energy/torque penalties
   **off** while eliciting the run; restore them only in the final consolidation stage.
5. **Warm-start every stage; change ONE dominant thing per stage; regression-gate** on the previous
   competence before widening. (This directly fixes attempt-2's "redesign destabilised it.")
6. **Eval the flight phase explicitly:** log `min-over-gait of (cL+cR==0)` to confirm a real
   both-feet-airborne window AND single-leg *alternating* contact (guards the bunny-hop degeneracy).
7. Optional polish once it runs: a **mirror/symmetry loss** inside a curriculum for a cleaner gait —
   not before, and not as a collapse fix.

**Track B — reference motion (if A plateaus at a fast bounding walk, or we want > ~2.5 m/s clean):**
Add a single retargeted running clip and an **AMP-style discriminator** (style from the clip
distribution, task from velocity-tracking). This is the most reliable published path to G1 flight
and decouples "looks like running" from hand-tuned reward weights. Cost: a mocap clip + retarget +
a discriminator network; it muddies the "emergent, no-prior" cleanliness slightly but is the proven
fast route. For the spring study, keep the prior **byte-identical** across spring/no-spring arms.

## Verdict

**Not a mystery; not just our incompetence — both.** RL humanoid running with a flight phase is a
*genuinely harder regime than walking* and is a **known-achievable** target for a G1 (real results
to 3.3 m/s, sim to 4+ m/s), but **only with running-specific machinery**: reference motion (the
most reliable) and/or velocity-gated push-off + short-stance + flight rewards under a proper
velocity curriculum and soft termination — none of which exist in the stock Playground walk env.

Our two failures are the **two most common pitfalls in the field, in textbook order**:
(1) wide command from scratch + heavy fall penalty → conservative slow-walk collapse;
(2) too many reward changes at once across a stage without warm-start/gating → lost the gait.
Layered on **one real structural limit** we already diagnosed: the Playground walk env's gait
clock + air-time cap *forbid* a flight phase config-only, so an env subclass is mandatory.

**Recommendation:** keep the staged plan, but (a) build the env subclass before expecting any
flight, (b) enforce curriculum + warm-start + single-change-per-stage + regression-gating
discipline, and (c) treat **reference-motion / AMP as the planned fallback**, not a last resort —
it is the highest-probability route to a fast, clean run and is how the field actually does it. If
> 2 m/s with clean alternating flight does not appear within ~1–2 staged reward-only runs after the
subclass, switch to Track B rather than tuning weights indefinitely. Caveat: published *running*
humanoids are overwhelmingly Isaac Gym; MJX/Brax is proven for humanoid *walking* but less so for
the high-impact short-stance contact of running — budget some MJX contact-tuning time, and treat a
persistent reward-only plateau as possibly a sim-contact issue, not only a reward issue.

## Sources
- Chasing Autonomy — G1 running 3.3 m/s, retargeted human refs + control-guided RL: https://arxiv.org/abs/2603.25902
- Gait-Conditioned RL with Multi-Phase Curriculum (G1, PPO, Isaac Gym, Froude>0.5 run gate, push-off/short-contact/flight rewards, to 4.0 m/s): https://arxiv.org/html/2505.20619v3
- AMP: Adversarial Motion Priors (emergent speed-dependent gaits, run ~4.5 m/s): https://arxiv.org/pdf/2104.02180
- DeepMimic (imitation, aerial skills): https://github.com/xbpeng/DeepMimic
- Learning Velocity-based Humanoid Locomotion: Brax + MJX (our stack; periodic clock + stance/flight indicators; REEM-C walking, 1.0 m/s): https://arxiv.org/html/2407.05148v1
- MuJoCo Playground (G1/H1/Berkeley/T1 joystick walking + sim-to-real): https://playground.mujoco.org/assets/playground_technical_report.pdf
- Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (minimal <10-term reward, alive bonus, DeepMimic-style termination): https://arxiv.org/html/2512.01996v1
- Berkeley Humanoid (research platform, PPO, DeepMimic-style termination): https://arxiv.org/pdf/2407.21781
- Learning Symmetric and Low-Energy Locomotion (mirror loss only helps within a curriculum): https://arxiv.org/pdf/1801.08093
- Velocity-command curriculum guidance (premature wide command harms high-speed learning): https://arxiv.org/pdf/2410.10438
- Achieving Stable High-Speed Locomotion for Humanoid Robots with Deep RL: https://arxiv.org/pdf/2409.16611
