# Go1 (dog) running knee-spring — experiment design (2026-06-16 design workflow)

## Verdict: the Go1 can likely RUN (far more permissive than the G1) — but a true FLIGHT phase is the risk.
The Go1 joystick env has **none of the G1's anti-flight structure**: no `feet_phase` gait clock, soft
termination (`-1` vs the G1's `-100`, firing only on torso flip — a level-torso aerial phase does NOT
terminate), fully emergent contact, and `feet_air_time +0.1` already rewards longer swings.

**KEY MECHANISM FIX:** the Go1 speed knob is **`command_config.a`** (the x/y/yaw command AMPLITUDE, stock
`a[0]=1.5` m/s), **NOT `lin_vel_x`** (that is a G1 field). Raise `a[0]` to ~3.0–3.5 m/s to push trot → gallop.
Trim the yaw/lateral amplitudes so the policy spends capacity on straight-line speed.

**Adversarial caveat (HIGH severity):** speed is reachable, but a genuine all-feet-off **flight window is NOT
guaranteed** — the default may be a grounded fast trot. **Gate on a measured all-feet-off fraction** before
spending spring compute. Without flight, the "running" result collapses back to the known walking win.

## Plan (warm-started; no reward surgery — the explicit lesson from the two G1 failures)
1. **S1 warm-start trot** — restore the flat Go1 walker, raise command to ~2.2 m/s, stock rewards. ~80–150 M.
2. **S2 run + flight** — restore S1, command ~3.0–3.5, **LIGHT** flight tweaks: `feet_air_time 0.1→0.4`,
   `lin_vel_z -0.5→-0.25` (halve, do NOT zero), relax `feet_height/feet_clearance/max_foot_height` caps.
   **Leave `termination` at −1; keep `energy` on** (zeroing energy + softening termination is exactly what
   destabilized G1 attempt 2). **Gate on the measured flight fraction.** ~120 M.
3. **S3 post-hoc work-loop** (local CPU) — build the calf work-loop from the S2 run trajectory; the **offset
   sets the constant preload**, the **braking lobe sets the stiffness**; decide preload vs stiffness here.
4. **S4 run + almost-constant per-leg adaptive preload — the spring we try first.** The exact recipe that gave the
   walking −14 to −27 %: a low-stiffness, heavily pre-wound spring (≈ constant torque over the calf's small
   range), one per leg, slowly re-tensioned per-leg by the adaptive controller from each leg's own measured
   load — passive within a stride. Manufacturable and clutch-free. Config `configs/go1_run_spring_preload.yaml`.
5. **S5 run + one-sided linear stiffness — DEFERRED.** Run only if the S3 work-loop shows the constant preload
   leaves braking energy unrecovered (decision rule below). Built but parked: `springs.one_sided_linear` +
   `configs/go1_run_spring_onesided.yaml` (placeholders await S3). NOT a required condition.
6. **S6 second seeds** for the spring we settle on (match the 2-seed Go1 walking standard).

## Spring choice — lead with the almost-constant spring; decide the rest from the data (re-synced 2026-06-16)
We did walking with the almost-constant (per-leg adaptive) preload and it worked; **we now apply the same
recipe to running.** Manufacturing preference order: **almost-constant preload → linear spring → clutch**. A
clutch or any other within-stride engagement mechanism is a large risk/complexity factor and is a **last
resort**, used only if the data forces it. (Note: per-leg *adaptive preload* is a slow servo re-tensioning the
spring between conditions — passive within a stride — which is a far smaller risk than a clutch.) The spring
TYPE is an OUTPUT of the baseline work-loop, not an assumption: running adds a flexion→extension energy
exchange a flat torque cannot store, so a stiffness spring *might* do better — but we try the simple, buildable
spring first and escalate only if the data says so.

Decision rule from the S3 calf work-loop (calf torque vs calf angle over one stride), pre-registered:
- mostly a flat **offset** (support torque ~independent of angle) → **almost-constant preload** (the walking case);
- torque rises ~**linearly** with angle (a clear restoring slope) → **linear spring**;
- a large energy-absorbing region whose torque opposes what an always-on passive spring would give in swing
  (a passive spring would FIGHT the leg for part of the cycle) → **clutch needed**;
- anything else (e.g. energy stored only on flexion) → **else** — revisit then (a one-sided stiffness `k`
  ~50–70 % of the braking work, one-sided so it does not fight swing — the G1 reversal failure mode).

## Loads — the +2.5 / +5 kg extension (agreed 2026-06-16)
Get the no-load RUN working first (S1–S5). Then extend to LOAD-CARRYING running, mirroring the validated
walking load study: retrain the no-spring and spring arms with **payload DR (0–5 kg)** + the adaptive per-leg
preload, and evaluate at **0, 2.5, 5 kg**. This tests whether the running-spring energy win (and the
load-adaptive preload) holds at load — the natural unification of the running study with the walking
load-carrying result. Keep payloads **≤5 kg** (the real Go1's realistic range; do NOT repeat the 30 kg
sim-fantasy). Report CoT per load with per-seed spread and the survival/stability cost.

## Parity (valid comparison)
The electrical-reward wrapper stays ON and **byte-identical** across arms (`energy_reward_weight -1e-4`,
`energy_motor go1_knee`); same schedule, tweaks, and seeds; the spring is the ONLY between-condition difference.
Keep `tracking_lin_vel` at 1.0 so forward speed must be tracked (not just flight) — else a degenerate
bound/hop inflates the spring's apparent benefit. Compare on **cost of transport** (speed-matched).

## Three fixes required before the spring arms run (adversarial panel)
1. **Fix the command mechanism** — use `command_config.a`, pin forward, trim yaw/lateral (closes both the
   wide-range collapse risk and the speed-match confound).
2. **Gate on the all-feet-off fraction** — do not assume flight; verify it emerges at the commanded speed.
3. **Include the stiffness arm** — a constant preload alone misses the braking-recovery channel that is the
   whole point of the running spring.

*(Design produced by a 7-agent workflow: feasibility/gait, spring-design, reward/curriculum lenses →
synthesis → 3 adversarial checks. ~423k tokens, ~22 min.)*
