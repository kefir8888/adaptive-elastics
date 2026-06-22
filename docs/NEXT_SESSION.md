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
- If stage-1 still shows asymmetry, add a left/right symmetry reward (a `leg_symmetry`-style term exists in
  `g1_hop_env.py`, body-frame foot symmetry — port it).

## Box / infra (unchanged, hard-won today)
- USER provisions the immers box + pastes IP; `scripts/box.py` drives it (status/launch/pull/watch); arm
  `scripts/box_safety_arm.sh` BUT note the idle dead-man proved UNRELIABLE today (box idled ~2 h, never fired)
  — **console-DELETE is the only proven billing stop.** Top action item: obtain an immers API destroy token.
- Watchdog must tolerate long VPN outages (training is detached/nohup; 25-min unreachable was just VPN).
- Incident post-mortem: `docs/incident_2026-06-22_overnight_billing.md`. All bound-run data + videos are local
  in `outputs/clean_curriculum/fair_centered/`.
