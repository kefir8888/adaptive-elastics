# NEXT SESSION — run the dog-running knee-spring experiment (handoff)

> **Fresh-agent entry point.** Read order: **this file → `README.md` → `docs/JOURNAL.md` →
> `docs/dog_running_design.md`**. Everything is committed on `main` (current). The GPU box is OFF.

## Goal
Test a parallel **KNEE (calf) spring for Go1 RUNNING energy** — the committed direction. The Go1 is the
low-gear platform where springs pay; running adds a braking-energy-recovery channel a constant preload can't
capture (so a one-sided *stiffness* spring may beat the walking constant-preload).

## Infrastructure — immers.cloud, rubles (decided 2026-06-16)
- **Why:** crypto-from-Russia is KYC-blocked everywhere (Vast → BitPay/Crypto.com; Copperx, Coinbase, etc. all
  exclude Russia under US sanctions). immers takes **rubles, zero friction**. The crypto savings (~$5–10) aren't
  worth it — this experiment is **~$18–20 total** (short Go1 runs), so cost is a non-issue.
- **The USER provisions the box** (immers, ruble-funded): an **A100** (or H100), an **Ubuntu+CUDA** image,
  **~50 GB disk**, and pastes the SSH (ip + port + key/password).
- **The AGENT then drives it** and **DESTROYS it when done** — short runs; do NOT leave it billing (the last
  campaign overran to ~13 h / ~4600 ₽; sync after every run, kill the box after).
- **Bootstrap:** `curl -fsSL https://raw.githubusercontent.com/kefir8888/adaptive-elastics/main/scripts/gpu_box_setup.sh | bash`
  (clones `main` — now current — installs uv, `uv sync --extra cuda`, verifies JAX-on-GPU).

## The experiment (staged; full detail in `dog_running_design.md`)
> **The warm-start walker checkpoint was lost with the deleted box → Stage 0 retrains a walker first.**

0. **Walker** — train a flat Go1 walker from scratch (`configs/go1_baseline_payload.yaml` with `payload_max_kg: 0`,
   or the stock walker), stock rewards, ~200 M. Needed as the S1 warm-start source.
1. **S1 trot** — warm-start the walker, `configs/go1_run_s1.yaml` (`command_config.a=[2.2,…]`). GATE: stable trot.
2. **S2 run+flight** — warm-start S1, `configs/go1_run_s2.yaml` (`a=[3.2,…]`, light flight tweaks; NO termination
   softening, NO energy zeroing — those killed the G1).
   - **⚠ BUILD NEEDED:** a **flight-fraction metric** — min-over-gait of `(contact_L + contact_R == 0)`. GATE on a
     real all-feet-off window before spending on the spring arms.
   - If **no flight** emerges → it's a fast trot; report that honestly (it collapses to the walking win) and stop.
3. **S3 work-loop** (local CPU, no GPU) — roll out S2, build the calf work-loop; **offset → constant preload**,
   **braking lobe → stiffness**. Decide preload vs stiffness here.
4. **S4 run + constant preload** — the defensible arm. BUILD a `go1_run_spring_preload.yaml` (preload_dr + the
   adaptive controller in `src/pea/control.py`).
5. **S5 run + one-sided stiffness** — the braking-recovery arm.
   - **⚠ BUILD NEEDED:** a **one-sided linear-stiffness spring kind** in `src/pea/springs.py` (store on flexion,
     return on extension; `k` modest, strictly one-sided so it doesn't fight swing — the G1 reversal failure mode).
6. **S6 second seeds** for the headline arm(s) (match the 2-seed walking standard).

## Load extension (+2.5 / +5 kg) — after the no-load run works
Retrain the no-spring + spring arms with **payload DR (0–5 kg)** + the adaptive per-leg preload; **eval at
0 / 2.5 / 5 kg**. Keep **≤5 kg** (real Go1 limit — do NOT repeat the 30 kg sim-fantasy). Report CoT per load
with per-seed spread + survival/stability.

## Methodology discipline (README has the full list — follow it)
1. Cheap feasibility probe before every full RL run. 2. One change per stage; warm-start + gate. 3. Sync after
every run; **destroy the box after**. 4. Report **cost of transport (W ÷ m/s)**, per-seed spread, and the
stability/survival cost — never raw watts. 5. Eval AT the trained condition; measure forward **speed**, not
`tracking_lin_vel`. 6. Check physical realism before choosing ranges.

## Key facts (don't re-derive)
- **Gearing is the crux:** G1 (high gear, ohmic ~4 %) spring **NEGATIVE**; Go1 (low gear, ohmic ~54 %) **POSITIVE**.
- **Walking result (validated, 3 seeds):** adaptive knee preload cuts **CoT −14 to −27 %** (seed 2 a weak −3 to
  −8 % outlier), growing with load, with a high-load **stability cost** (survival drops ≥7.5 kg).
- **Adaptive controller** = `src/pea/control.py` `AdaptivePreloadController` (per-leg 15 s-EMA → clipped-proportional
  preload). The spring injects via `qfrc_applied`; motor torque (`qfrc_actuator`) **excludes** it → energy honest.
- **Go1 speed knob = `command_config.a`** (NOT `lin_vel_x` — that's a G1 field). The Go1 env is **flight-permissive**
  (no gait clock, soft termination −1) → **no env subclass needed** (unlike the G1).
- **G1 running is parked** (long-shot; needs an env subclass + ideally reference-motion/AMP).
- Results/eval tooling: `scripts/go1_capacity.py` (capacity + CoT sweep), `scripts/render_walk.py` (videos),
  `scripts/g1_run_probe.py` (speed/gait probe — adapt for the flight-fraction metric).
