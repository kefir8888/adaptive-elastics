# Running-efficiency program (Part 1, applied to running)

The experimental program for the **G1 running-for-efficiency** task and a parallel
**Go2 quadruped** efficiency experiment. Reflects the 2026-06-14 decisions. Reward
weights and some recipe details marked **[SOTA]** are filled from the running-RL
and regen/quadruped research workflows (in progress at time of writing).

## Decisions (2026-06-14)

- **Task = RUNNING for EFFICIENCY**, not speed. The knee is speed-limited (20 rad/s
  ceiling), so "run faster / jump higher" is not the parallel-spring lever; the
  lever is **electrical energy** (the spring's braking-energy recovery channel is
  *larger* in running than walking). Metric: electrical cost of transport / total
  electrical power, no-regeneration.
- **No-regeneration: JUSTIFIED for the G1** (verification `wf_924da954`), on three
  grounds: (1) **back-EMF physics** — at locomotion joint speeds the back-EMF is well
  below the ~48 V bus, so returning current needs a boost converter commercial drivers
  lack; (2) the documented **regenerative-resistor** pattern (braking shunted to heat);
  (3) no public evidence of bidirectional power electronics in the Unitree driver.
  Engineering judgment, NOT a verified G1 spec — gold standard is a hardware test
  (command a braking torque, see if the DC-bus voltage rises). **Exceptions** (so do
  NOT claim "no humanoid regenerates"): MIT Cheetah 2013 (custom bidirectional
  converter — the "dog") and Tesla Optimus (one 2026 source, gravity-assisted arm
  motion). **Sensitivity is real:** G1 walking 178.5 W no-regen vs 135.5 W regen
  (~24 %), and the spring's braking-recovery win largely **vanishes under true regen**
  — report no-regen as the model WITH this sensitivity, and confirm before publishing.
- **Hopping: DROPPED** — we work with a full humanoid, not a Raibert one-leg hopper.
- **Quadruped (Go2): ADDED** as a parallel track — low gear (6.22:1) → ohmic
  dominates *and* big braking → a bigger, cleaner efficiency win; differentiate from
  Bjelonic 2023 / PIL 2026.
- **Step 2 is a hard GO/NO-GO gate** — the per-joint post-hoc estimate must show a
  gain materially above walking's ~3 % whole-body or we stop. Later steps run only
  if the preliminary tests show something.
- **Adaptive (spring co-designed with the controller): LATER** — that is maxing out
  the tech once the effect is shown. For now: a fixed spring, just demonstrate the
  effect exists.
- **Speed / load ranges: tested at the END** (the robustness envelope).
- **Spring-target joints = the PITCH trio** (hip-pitch, knee-pitch, ankle-pitch) as
  primary candidates — **plot all leg joints, let the data choose.** The knee's
  speed limit excludes it for height but NOT for efficiency, so it is back in play.

## Pipeline (each step gated)

1. **Train a clean baseline runner** (milestone — below).
2. **Per-joint post-hoc efficiency estimate** (all leg joints, pitch trio focus) +
   **spring-stiffness sweep** → **GO/NO-GO gate**.
3. **Fixed-spring in-loop gate** (matched baseline, identical reward, ≥2–3 seeds).
4. *(later)* **Adaptive / co-designed spring** across a speed/load envelope.

---

## Milestone 1 — the clean runner

### What the env already provides (verified from `registry.get_default_config`)
- `G1JoystickFlatTerrain`, 50 Hz control. Command `lin_vel_x` defaults to **[-1, 1]
  m/s** — too slow to run, and commanding 2.0 falls (out-of-distribution). Gait
  machinery is present: `feet_air_time` (2.0), `feet_phase` (1.0),
  `tracking_lin_vel` (1.0), `orientation` (-2.0), `termination` (-100).
- **Smoothness penalties exist but are ZERO** (`action_rate`=0, `dof_acc`=0,
  `torques`=0) — **this is the cause of the ~55 %-reversal sawtooth.** Enabling
  `action_rate` (negative) fixes it.
- Config now supports `env_overrides` (top-level keys, e.g. the speed range) and
  `reward_scales` (e.g. enabling `action_rate`) — `src/pea/config.py`, `env.py`.

### Reward design  — concrete, from the running-RL SOTA workflow `wf_c9783c03`
**The full reward lives in `configs/run_baseline.yaml`.** Highlights:
- **Anti-chatter (THE fix):** the smoothness trio was all 0.0 by default — that IS
  the sawtooth. Enable `action_rate=-0.01` (primary), `dof_acc=-2.5e-7`,
  `torques=-1e-4`. Optional 10 Hz IIR low-pass on actions (a wrapper) for sim2real.
  Acceptance: control-reversal rate well below ~55 %, peak/RMS torque near ~1.4.
- **Speed:** `env_overrides.lin_vel_x = [0, 2.5]` m/s (default [-1,1] falls >1);
  optionally curriculum up.
- **Flight phase:** `feet_air_time` 2→4, keep `feet_phase` 1.0, enable
  `lin_vel_z=-0.2` (kill hop-in-place), `feet_clearance=-0.3`. If a hopping/skipping
  degeneracy appears, add the **velocity-gated flight bonus** (a small wrapper):
  `+0.5·(1−c_L)(1−c_R)·I[v_x>1.5]` (clip ≤0.3) and `double_stance_penalty −0.3`.
- **Landing/stability:** `contact_force=-0.01` (tighten to -0.02 / max 400 N to land
  with flexed knees), `ang_vel_xy=-0.3`, `orientation=-2.0`, `base_height` enabled.
- **Energy:** `ElectricalRewardWrapper` (total electrical, no-regen) — NOT
  Playground's built-in `energy` (mechanical-only; undercounts the spring). Identical
  weight both arms; calibrate to ≤15 % of mean tracking reward.

### Training recipe (from SOTA)
- **≥3 seeds** per arm (5 credible); report mean±std CoT + best-vs-best. Fixed seed
  set recorded in config.
- **DR on, identical across arms**; tighten push magnitude upper bound to ~1.5 m/s².
- **Budget:** flat ~300 M steps (~85 min/H100), then optional rough finetune ~100 M.
  Episode length 1000 (20 s).
- **Warm-start** the runner from the flat walker (same protocol both arms); validate
  the headline with a from-scratch seed. **Do NOT** warm-start the spring arm from the
  no-spring runner — it anchors the gait and hides co-adaptation.
- **Rough terrain (optional):** flat→rough finetune only (never from scratch); Perlin
  heightfield amplitude 0.05–0.08 m, octaves 2, freq 0.05–0.1 m⁻¹; NO stairs (G1 is
  blind). Throughput ~47k→~30k steps/s.

### Validity guards (the watchdog)
- The smoothness penalty MUST be on before any energy number is trusted. **The
  sawtooth was diluting the spring's measured benefit** — a de-chatter proxy
  (low-pass the recorded walking torque) drops the baseline 178.5 → ~110–132 W
  (the chattery gait wastes ~26–40 %, mostly via the no-regen clamp *rectifying*
  the high-frequency torque) and roughly **doubles the spring's % saving**
  (−2.9 % → ~−6 %). So the gate MUST be run on the smooth baseline (`run_baseline`
  has `action_rate` on), and chattery-baseline energy numbers are not comparable.
  Caveat: this is a filtering proxy — a retrained smooth policy may also do *less*
  braking (less for the spring to recover), so the in-loop number could differ.
- **Matched arms:** identical reward (including `action_rate` and the energy weight)
  for spring vs no-spring.
- Report the energy as a band over the Kt/R uncertainty; lead with no-regen.

### What we need to run (the GPU checklist)
- `configs/run_baseline.yaml` (this milestone) — widened speed range + smoothness +
  energy weight. (Spring arm reuses it with a spring block + identical weights.)
- A calibration of the energy weight (as for walking) so it is ~7–12 % of the
  tracking reward.
- Box bootstrap via `scripts/gpu_box_setup.sh`; ~2–3 seeds.

---

## Milestone 1b — the ENERGY-AWARE baseline is the real control (do this first)

**Every spring number to date is measured against an ENERGY-NAIVE walker**
(`energy_reward_weight=0`, Playground `energy` scale 0) — a policy that never tried
to save energy. That is the wrong reference. The proper control is an **energy-aware
baseline (no spring, electrical penalty ON)**, and the spring's value is
`CoT(energy-aware + spring) − CoT(energy-aware, no spring)` — NOT vs the naive walker.

- An energy-aware policy reduces torque *actively* (posture, smoothness, less
  braking) and will likely **self-de-chatter** (the energy penalty and `action_rate`
  push the same way; the rectified chatter burns energy). So the energy-aware
  baseline is probably much leaner than the naive 178 W walker (the de-chatter proxy
  ~110–132 W is a hint) — the spring competes against THAT.
- This can shrink the spring's marginal value (if policy and spring offload the same
  torque) or leave it (if the spring cancels an irreducible gravity component the
  policy can't). Only the energy-aware in-loop comparison resolves it.
- **Action:** train the energy-aware baseline first (the calibration sweep does this:
  short no-spring runs at several energy weights → pick the weight, measure the
  naive→energy-aware saving, confirm self-de-chatter). `run_baseline.yaml` is already
  energy-aware + smooth. The naive-baseline post-hoc numbers are UPPER BOUNDS only.

## Milestone 2 — per-joint post-hoc estimate + GO/NO-GO gate

- On the baseline running trajectory, with `pea-sweep` / `pea.metrics`: per-joint
  **electrical energy, recoverable braking energy (no-regen), ohmic share** for ALL
  leg joints, the **pitch trio** highlighted. Verify L/R symmetry before reducing to
  one leg.
- **Spring-stiffness sweep** (`pea-sweep` grid) per candidate joint → extract the
  post-hoc optimum `(k, θ₀)` from the data, then confirm with a small grid.
- **GATE:** proceed to Milestone 3 only if the running post-hoc whole-body saving is
  materially above walking's ~3 % (target ≳ 6–8 %). Otherwise stop / write the
  negative result. **Plot all joints.**

## Milestone 3 — fixed-spring in-loop gate
- Best joint(s) + `(k, θ₀)` from Milestone 2. Matched no-spring baseline, identical
  reward, ≥2–3 seeds. Compare CoT / total electrical / ohmic best-vs-best, and
  report the post-hoc-vs-in-loop delta (the methodological contribution; the
  walking→running gradient is the paper's spine).

## Milestone 4 *(later)* — adaptive / co-designed spring
- One policy conditioned on `(K_eff, θ₀)` and the operating envelope (speed ±
  slope/load), generalising zero-shot; outer optimiser picks the spring. The
  dead-zone clutch gives weak dominance ("spring off" always reachable).

---

## Parallel track — Go2 quadruped efficiency

**Params** (GO-M8010-6, teardown / PIL 2026): gear **6.22:1**, Kt ≈ **0.26 N·m/A**
(teardown, high-conf), **R UNMEASURED** (placeholder 0.30 Ω in `energy.MOTORS['go2']`
— a real teardown R is the blocker for absolute numbers), peak torque **23.5 N·m**,
mass 15 kg, 12 DoF (3/leg), ~231 Wh, 1–2 h. Load-bearing ratio **R/Kt² ≈ 4.4** vs
G1's 0.0025 → ohmic is **39–76 %** of the motor budget (vs G1 ~4 %): the
copper-offload lever is fully armed.

**Differentiator vs prior art** (the cell is partly occupied, so this matters):
- Bjelonic 2023 (ANYmal, 100:1) uses a **torque-square proxy**; PIL 2026 (Go2 sim)
  uses **positive mechanical power** — **neither uses battery-electrical** (ohmic +
  negative work), which on a low-gear quadruped **systematically undercounts** the
  spring. Headline = **total electrical CoT** (`τω + (τ/Kt)²R`, no-regen) — a claim
  neither makes. Lead with the **ohmic-% reduction** (Kt/R-independent) until R is
  measured; total electrical CoT as a band.
- **Zero-shot spring-conditioned policy** (no per-condition retrain): one policy
  conditioned on `(K_eff, θ₀, slope, payload)` generalising zero-shot — vs Bjelonic's
  BO+retrain and PIL's per-design distillation.
- Always-engaged spring + the post-hoc-vs-in-loop contrast.

**Pipeline:** same as the G1 (baseline trot/run → per-joint post-hoc gate →
fixed-spring in-loop → later conditioned). Platform: MuJoCo Playground ships a
Go1/Go2 quadruped env; reuse `pea-sweep --robot go2`. **Blocker for absolute
numbers: source/measure the Go2 winding resistance R.**

## Open questions in flight (workflows)
- No-regeneration verification across modern humanoids.
- Running-RL reward / smoothness / aerial / DR / rough-terrain SOTA.
- Go2 actuator params + differentiated experiment design.
