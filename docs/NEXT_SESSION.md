# NEXT SESSION — PIVOT to a real alternating-leg RUN on G1JoystickRun (updated 2026-06-22)

## Why the pivot (read first)
The energy-objective study was being run on **`G1JoystickBound`**, but that env is built on the *hop* env
and carries synchronous-jump terms (`hop_push`/`hop_flight`/`hop_height`) that **overpower its anti-phase
clock** — so the trained gait is a **two-footed JUMP, not running**: knee L/R correlation **+0.69** (in-phase),
left hip swings **47°** vs right **76°** (left leg under-driven). User confirmed from video: "it's jumping,
the left leg never gets forward." Diagnosis tool: `scripts/gait_check.py`.

**Fix = use the right env, not imitation.** `G1JoystickRun` ([src/pea/g1_run_env.py](../src/pea/g1_run_env.py))
is built on the *walk* env: anti-phase gait clock, **no synchronous-hop terms**, + un-capped air-time, faster
clock, a velocity-gated flight bonus. It alternates legs by construction. Imitation (LocoMuJoCo/AMP — data is
public: `openhe/g1-retargeted-motions`, `robfiras/loco-mujoco` ships AMP+DeepMimic+datasets in JAX) is the
ESCALATION only if the reward-shaped runner still looks unclean.

## ⚠️ DECIDE BEFORE ANY GPU SPEND — the G1 spring result is PRE-FLAGGED NEGATIVE
A 9-agent running-plan analysis flags that "parallel knee spring helps G1 RUNNING" collides with this project's
OWN negative results:
- **NR-7:** the knee is SPEED-limited (~20 rad/s) → a parallel spring can only cut landing-braking ohmic (the one
  gear-independent channel); it cannot help the speed-capped push-off.
- **NR-8:** a parallel knee spring + passive dead-zone clutch can't be rescued in running; running wants a SERIES element.

So the EXPECTED G1 outcome is **neutral / braking-recovery-only** — frame it as a CONFIRMATION of NR-8, not a win.
**Decide before spending on the spring arm (Stage 3):** confirmation-of-the-negative suffices → proceed on G1; a
positive WIN required → REDIRECT to the **Go1** (low-gear, ohmic ~54%, documented-positive braking-recovery) or a
SERIES element. Do not escalate approaches chasing a pre-flagged-negative G1 outcome. (Stage 1 — eliciting a clean
alternating RUNNER — is worthwhile either way and is a reusable result.)

## STAGE 1 — elicit the runner (READY, smoke-passed)
Config: **[configs/g1_run_s1.yaml](../configs/g1_run_s1.yaml)** — `G1JoystickRun`, forward 0.5–1.2 m/s,
`centered_dr: true`, **energy OFF, no spring** (clean gait first), 150 M steps, from scratch. Smoke-passed
locally (policy saved). Launch on a fresh box:
```
cd ~/adaptive-elastics && export PATH=$HOME/.local/bin:$PATH && nohup nice -n 19 ionice -c3 \
  uv run pea-train --config configs/g1_run_s1.yaml --output_dir ~/runs --suffix s1 \
  > ~/run_s1.log 2>&1 < /dev/null & echo $!
```
**GATE (the test of "is it actually running"):** pull the run, then
`env -u PYTHONPATH uv run python scripts/gait_check.py <run> 0.8,0,0 1 600` — require **knee L/R correlation
NEGATIVE (antiphase)** and **symmetric L/R hip ranges** (vs the bound run's +0.69 / 47°-vs-76°), plus forward
speed ≈ command and nominal survival (`scripts/hop_failure_diag.py`). Render: `scripts/render_hop.py` with
`PEA_RENDER_CMD=0.8,0,0`.

## STAGE 2 — refit the knee spring from the RUN gait
The bound-gait pogo (k=127.7, θ_engage 0.734) will NOT transfer to a running gait. Re-fit from stage-1's
knee work-loop (`scripts/hop_spring_prep.py` / work-loop tooling) → new k, θ_engage.

## STAGE 3 — parity baseline + spring → CoT
Two byte-identical `G1JoystickRun` configs except the spring (energy_reward_weight **−1e-4** — the
running-dominant weight; −1e-3 suppressed forward speed), `centered_dr: true`, warm from stage-1.
Compare CoT at matched achieved speed (`scripts/hop_energy_compare.py <base> <spring> 800 0.8,0,0`):
report J/m no-regen + regen, ohmic-vs-mechanical, both arms ≥2/3 survival.

## Caveats / known risks
- A true FLIGHT phase on G1JoystickRun was historically hard (the suspended running track over-corrected to
  standing). We do NOT require flight — an alternating jog at 0.5–1.2 m/s satisfies "running" and trains more
  readily. Push speed only after a stable alternating gait exists.
- **DON'Ts (workflow-verified against the code):** do NOT patch `G1JoystickBound`; do NOT raise `feet_phase` to
  0.6 (joystick.py rewards foot-HEIGHT, masked at low speed, and FIGHTS flight); do NOT reuse `g1_hop_env`
  `leg_symmetry` (it is a fore_aft+lateral asymmetry penalty → fights the alternation; s4 disables it); do NOT
  warm-start the s4 Bound checkpoint (it encodes the synchronous jump). If stage-1 is asymmetric, use **mirror
  DATA augmentation**, not a symmetry loss.
- `gait_check.py` now prints a **GATE VERDICT** (antiphase knees corr<−0.2 / hip-swing-gap <10° / |vx|>0.4 /
  survival). Knee peak/RMS angular velocity vs the ~20 rad/s ceiling is a **DIAG** (NR-7 headroom), not a gate;
  do NOT gate on flight (NR-7 makes it a stretch). The Stage-3 BASELINE must also clear |vx|>0.4 so CoT is
  matched-speed (the s4 base stalled ~0.1 m/s — the unmatched-speed confound).

## Box / infra (unchanged, hard-won today)
- USER provisions the immers box + pastes IP; `scripts/box.py` drives it (status/launch/pull/watch); arm
  `scripts/box_safety_arm.sh` BUT note the idle dead-man proved UNRELIABLE today (box idled ~2 h, never fired)
  — **console-DELETE is the only proven billing stop.** Top action item: obtain an immers API destroy token.
- Watchdog must tolerate long VPN outages (training is detached/nohup; 25-min unreachable was just VPN).
- Incident post-mortem: `docs/incident_2026-06-22_overnight_billing.md`. All bound-run data + videos are local
  in `outputs/clean_curriculum/fair_centered/`.
