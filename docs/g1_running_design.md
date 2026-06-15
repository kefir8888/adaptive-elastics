# G1 Running Curriculum — design (from the 2026-06-15 design workflow)

## Goal
Get the Unitree G1 (29-DoF, MuJoCo Playground `G1JoystickFlatTerrain`, brax PPO) to **truly run
— >2 m/s with a flight phase** — so a parallel **knee** spring can be tested for running energy
(braking-energy recovery at impact + push-off torque offload). The mechanism needs a real
flight/impact; a fast walk does not exercise it.

## Attempt 1 (FAILED)
`[0,3]` from scratch + walker reward → collapsed to a **0.85 m/s never-fall walk**, falls ≥2 m/s.
Diagnosis: too-wide range from scratch + `termination -100` → policy plays it safe (slow, never fall).

## Structural finding (the key result)
**The Playground G1 walk env mathematically forbids a flight phase, and it's only partly config-reachable:**
- `feet_phase` (joystick.py ~L779) pays **+1/step for keeping exactly one foot planted** (anti-phase
  gait clock at `gait_freq=U(1.25,1.5)` Hz) — full weight forbids a double-float.
- `_reward_feet_air_time` **hard-clips** the per-contact bonus at `threshold_max−threshold_min = 0.3 s`.
- `threshold_max` and `gait_freq` are **hardcoded literals, NOT config fields** (`reward_scales` + nested
  `reward_config` keys like `tracking_sigma`, `max_foot_height` ARE reachable; these are not).
- ⇒ **config-only caps at a fast walk.** A true run needs a small env subclass (Stage 2).

## Adversarial verdict
- Stage 1 (jog) → **low risk**, ~1.4–1.6 m/s, fixes the collapse.
- True run >2 m/s + flight → **HIGH risk**; most likely a fast bounding walk. >2 m/s may be infeasible here.
- ⇒ **Stage and gate; fail fast.** Don't build the subclass or burn S2/S3 until the jog is solid.

## Curriculum (3 stages, each ~1 h, warm-started)
| Stage | lin_vel_x | key reward changes | steps | from |
|--|--|--|--|--|
| **S1 jog** | [0, 1.6] | tracking_lin_vel 1.5, tracking_sigma 0.5, termination −50, feet_phase 0.5, feet_air_time 4, max_foot_height 0.22, lin_vel_z −0.2, torques −2e-5, **energy 0** | 150 M | scratch |
| **S2 run+flight** | [0.8, 2.6] | as S1 + feet_phase **0.3**, feet_slip −0.5, **+ env subclass** | 200 M | S1 |
| **S3 consolidate+energy** | [1.2, 3.0] | termination **−100**, torques −1e-4, **energy −2.5e-4** (the comparison reward) | 150 M | S2 |

## Stage-2 env subclass spec (build only if S1 jogs well)
A small, version-controlled G1 subclass, **byte-identical across the eventual spring/no-spring arms**:
1. Override `_reward_feet_air_time` with `threshold_max ≈ 0.8 s` (un-cap stride/air time).
2. Widen `gait_freq` reset sampling to ~1.6–2.2 Hz (one line in `reset()`).
3. Add velocity-gated terms: `flight_bonus = +0.5·(1−cL)·(1−cR)·I[vx>1.5]` (clip ≤0.3/step) and
   `double_stance_penalty = −0.3·I[cL∧cR]·I[vx>0.5]`.
4. Highest-leverage single change: down-weight `feet_phase` **1.0→0.3** + the air-time un-cap together.

## Failure modes + guards
- **Plateau at fast walk** (air-time cap/gait clock) → the subclass; at eval **log min-over-gait of
  (cL+cR==0)** to confirm a real both-feet-airborne window, not just rising air-time.
- **Hop/bunny-hop degeneracy** (games air-time without forward run) → lin_vel_z −0.2, command FLOOR
  0.8/1.2, feet_phase 0.3 (keeps L/R alternation), the velocity-gated flight bonus; eval must verify
  ~2 m/s AND single-leg **alternating** flight.
- **Never-fall collapse recurs** → tracking_sigma 0.5 (keep speed gradient), termination −50 in S1/S2,
  warm-start with top command only +0.5–1.0 m/s above current competence.
- **Energy/torque suppress push-off** → energy 0 + torques −2e-5 in S1/S2; restore in S3.

## Reward parity (for the spring comparison)
Only the **final S3 reward** is the comparison reward and must be **byte-identical** between the
no-spring and spring arms (same termination −100, energy −2.5e-4, subclass, all weights). Both arms
**fork from the same S2 checkpoint**; the spring is the ONLY between-condition difference. S1/S2
bootstrap weights may differ from S3 since both arms share them identically. (Consider forking the
spring arm at S2, not only S3, so the comparison doesn't understate the spring — document the choice.)
