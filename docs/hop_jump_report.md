# G1 Hopping & Jumping — Part-2 Report (2026-06-20)

*Live document — updated at each milestone during the autonomous run. Numbers marked
**[TBD]** are pending a training/analysis step still in progress.*

## Summary

Part-2 (explosive moves) restarted on the Unitree G1 with **continuous two-footed
hopping for energy**, then extended toward **directional jumps** and an
**alternating-foot bounding** gait (the future running baseline). Headline so far:

- A parallel **pogo knee spring** lets the G1 hop **~33 % higher for ~7–10 % less
  electrical energy** (≈ **−30 % per unit hop height**). Ohmic loss is only ~1.5 % of
  the budget, so the benefit is the **mechanical braking-energy-recovery** channel —
  **gear-independent**, unlike the (negative) walking result where gearing was the crux.
- At a **matched apex (0.13 m) + cadence (1.9 Hz)** — fairness gate **FAIR** (apex Δ 0.3 cm) —
  the spring cuts electrical energy **−4.4 % (no-regen) / −4.5 % (regen)** per hop. Modest but
  **clean, real, and gear-independent** (ohmic ~1.5 %; the saving is the mechanical channel).
- Directional jumps: **trained but NOT controllable** — the policy hops but spins ~2.7 rad/s
  and ignores the command (the inherited asymmetric-stance yaw bias; the *deferred* leg-symmetry
  fix is implicated). Alternating-foot bounding: **partial success** — a stable forward-running gait
  with a strong flight phase (48 % airborne), going **straight** (no spin); clean foot alternation
  ambiguous (video clarifies). Goes straight where directional spun *because the forward command
  breaks the symmetry*.

## 1. In-place hop spring — energy study

### Setup
- Env `G1JoystickHop`: synchronous gait clock (both feet in phase), in-place command,
  hop reward terms (signed `hop_rhythm`, dense `hop_push`, `hop_height`, `hop_sync`,
  `hop_flight`). Spring = `one_sided_linear` pogo on the knee (k=106 N·m/rad), injected
  as `qfrc_applied` (passive) so the electrical energy (from `qfrc_actuator`, the motor
  torque) honestly excludes it.
- Metric: **electrical energy per hop (J/hop)** at a MATCHED apex + cadence (1.9 Hz),
  measured post-hoc (`P = max(τ·ω + (τ/Kt)²·R, 0)`, no-regen + regen band). NOT CoT
  (zero forward speed).

### The instability problem and the fix (the methodological story)
Injecting the full-strength spring onto a no-spring policy **destabilized** the gait and
made it overshoot the apex. Survival across 3 deterministic eval seeds:

| Spring policy | Survival | Apex (m) |
|---|---|---|
| Full-strength inject (original) | 1/3 | ~0.13 (overshoots 0.09 target) |
| Per-episode stiffness randomization (k_dr U(0,106)) | 2/3 | ~0.13 |
| **Staged stiffness ramp (k 40→75→106, warm-start-chained)** | **3/3** | **~0.129** |

The **staged ramp** (gradually raising stiffness so the policy migrates to full strength
and ends tuned to it) fixed the instability cleanly — a stable, k=106-tuned spring policy.

### Energy result (apex-confounded measurements — consistent direction)
Every spring policy *prefers* to fly to ~0.13 m rather than sit at 0.09 (the flight rewards
make height cheap once the spring supplies it), so the raw J/hop is apex-confounded; the
fairness gate flags it. The consistent pattern across policies:
- Spring hops **~33 % higher** (0.13 vs 0.097 m) for **~7–10 % less raw energy** →
  **~−30 % per unit hop height**.
- **Ohmic share ~1.5 %** → the win is mechanical (braking-energy recovery), gear-independent.

### Clean apex-matched comparison (Option 1: match at the spring's natural ~0.13 m)
Rather than fight the spring down to 0.09, both arms are pinned at the spring's natural
apex (~0.13). Spring@0.13 (staged ramp, 3/3 survival) vs baseline@0.13.

**Result — CLEAN POSITIVE (fairness gate FAIR, apex Δ 0.3 cm, cadence Δ 0.01 Hz):**

| metric (matched apex 0.13 m, cadence 1.9 Hz) | baseline | spring | delta |
|---|---|---|---|
| apex (m above standing) | 0.134 | 0.131 | Δ 0.3 cm → **FAIR** |
| **E/hop, no-regen (J)** | 498 | 477 | **−4.4 %** |
| **E/hop, regen (J)** | 428 | 408 | **−4.5 %** |
| mean power, no-regen (W) | 952 | 908 | −4.6 % |
| ohmic share | 1.6 % | 1.5 % | — |

The parallel pogo knee spring cuts hop electrical energy **~4.4 %** at a matched apex + cadence.
Ohmic is only ~1.5 %, so the saving is **mechanical** (reduced net motor work — push-off assist +
braking recovery), **gear-independent** — the opposite of the (negative) walking result. **Regen ≈
no-regen**, so the result is robust to the regen assumption (not a pure braking-recovery artifact).
This is the **first clean positive Part-2 result on the stock G1**, and it required the staged
stiffness ramp to obtain a stable, apex-matched spring policy. Magnitude is modest — consistent
with the G1's high gearing making the ohmic channel tiny, so the spring's only room is mechanical.
Caveats: single seed; energy objective OFF (post-hoc); Kt/R estimated.

**Energy partition (where the power goes — both arms split near-identically):**

| Component | % of battery draw (no-regen) | In our model? |
|---|---|---|
| Mechanical work (positive τ·ω) | **~98.5 %** | yes |
| Ohmic / copper (I²R) | **~1.5 %** | yes |
| Eddy + hysteresis (iron/core) | — (real, ~few %) | NO |
| Gearbox / transmission friction | — (real, ~10–30 %) | NO |
| Inverter / electronics | — (small) | NO |

Braking energy absorbed at landing ≈ **14.5 % of draw** = the **regen ceiling**, but the no-regen
G1 dumps it as heat → **~0 recovered in practice** (regen ≈ no-regen confirms this). So hop energy
is **~98 % mechanical work**, ohmic is negligible (killed by high gearing → low motor current), and
the spring's only lever is the **mechanical** channel (push-off work) — the inverse of the walking
result, where the ohmic-based claim died to gearing. Our model omits gearbox friction (the dominant
real loss); both arms share it, so the −4.4 % *relative* delta holds, but absolute watts are optimistic.

## 2. Directional jumps — TRAINED but NOT controllable (negative / partial)
Config-only curriculum on `G1JoystickHop`: command lin (x,y) + yaw, slow low-speed bands
(D0 lin ±0.15 / yaw ±0.3 → D1 lin ±0.3 / yaw ±0.5), boosted tracking, warm-start from S1.

**Training looked good** — reward −0.5 → ~70; `tracking_lin_vel` +839, `tracking_ang_vel` +204.
**But behavioral validation FAILED.** With the command correctly forced (verified it reaches the
policy observation — obs indices 11/114 shift by exactly the command delta; `state.info['command']`
correct), the deterministic policy **spins ~+2.7 rad/s regardless of command** (yaw− → +2.7, still
→ +2.7) and barely translates (~0.12 m/s vs commanded 0.3). It hops with flight but **does not track
commanded velocity, especially yaw.**

| forced command | mean yaw rate | net displacement / 8 s |
|---|---|---|
| forward (vx 0.3) | +3.5 rad/s | ~0.8 m (drifting) |
| yaw + (0.6) | +2.8 rad/s | ~0.9 m |
| yaw − (−0.6) | +2.7 rad/s (wrong sign!) | ~1.1 m |
| still (0,0,0) | +2.7 rad/s | ~1.0 m |

**Likely cause:** the inherited **asymmetric / diagonal stance** from S1 (the leg-splay noted earlier
and *deliberately not fixed*) creates a yaw bias the directional policy amplified into a constant
spin; the command can't override it. The high in-training tracking reward is misleading — earned in
the stochastic eval / dominated by linear tracking, while the deterministic gait spins. **The rigorous
validation caught what the reward alone would have falsely passed.**

**Recommended fix (for review):** (1) add the deferred **leg-symmetry reward** (foot fore-aft + lateral
symmetry) — directly targets the spin's root cause; (2) raise `tracking_ang_vel` weight; (3) consider
fixing S1's stance symmetry before warm-starting, or not warm-starting from S1. Video
`hop_directional.mp4` shows the uncontrolled spinning hop.

## 3. Alternating-foot bounding (running baseline) — PARTIAL SUCCESS
`G1JoystickBound`: anti-phase clock, flight machinery retained, forward command 0.5–1.2 m/s,
warm-start from S1. Trained cleanly (flight reward 18→414, feet_phase 6→211, fwd-tracking 20→1023).

**Behavioral validation (forced 0.8 m/s forward command):**

| metric | value | reading |
|---|---|---|
| survival | 600/600 (12 s) | stable |
| forward speed | 0.55 m/s (cmd 0.8) | real forward motion (undershoots) |
| mean yaw rate | +0.02 rad/s | **STRAIGHT — no spin** |
| flight (both feet airborne) | **48 %** of the time | strong flight phase |
| foot L/R height corr | +0.09 (~0) | alternation **ambiguous** (below) |

A **stable, forward-moving, flight-heavy gait that goes straight** — a real dynamic running-ish gait
and a solid running baseline. The one ambiguity is clean foot **alternation**: the L/R foot-height
correlation is ~0 (not the −1 of a clean anti-phase bound), consistent with EITHER a running gait
(alternating stance + heavy flight) OR a forward-traveling two-footed hop. `bounding.mp4` clarifies
the gait type; the strong `feet_phase` reward (211) indicates alternation was developed.

**Notable — this explains item 2.** Bounding goes STRAIGHT (yaw +0.02) while directional **spun**
(yaw +2.7). The only difference is the **forward command**: it gives a clear movement direction that
breaks the symmetry, whereas directional at low/zero speed was under-determined → the asymmetric
stance spun it. A clear task direction is itself a fix for the spin (matches the turning-cures-splay
observation). So directional jumps likely need either the leg-symmetry reward **or** a minimum
commanded speed to stay controllable.

## Methods & caveats
- Energy is measured POST-HOC with the energy objective OFF (the matched apex+cadence pins
  the task). A deployment-credible follow-up needs the calibrated **quadratic energy penalty
  ON** in both arms (planned).
- Single seed (seed 1) per arm so far; ≥3 seeds + the regen-sensitivity band are needed for a
  publishable claim.
- Motor Kt/R are estimates; lead with the % deltas, not absolute watts.

## Artifacts
- Videos (in `outputs/`): `hop_s1_new_hd/slowmo`, `hop_baseline_apexpinned_current`,
  `hop_baseline_vs_spring_current`, `hop_baseline_vs_spring_dr_k106`,
  **`hop_baseline_vs_spring_0p13_matched`** (the clean FAIR result — both arms 600/600 at 0.13 m),
  `hop_directional` (the spinning hop — item-2 failure), `bounding` (forward running gait — item 3).
- Run dirs (rsync'd to `outputs/hop_runs/`): S1, baseline@0.09, spring (orig/DR/ramp@0.13),
  baseline@0.13, directional (dir_d0/d1), bounding — all ✓ backed up locally.
