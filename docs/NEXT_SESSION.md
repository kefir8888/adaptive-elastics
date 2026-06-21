# NEXT SESSION — energy-objective running-spring series (updated 2026-06-22)

**Goal:** a CLEAN, defensible answer to *"does a parallel knee spring reduce the electrical cost-of-transport
(CoT) of G1 running?"* Tonight's bounding runs gave a PROMISING but confounded signal (full-strength spring made
the bound RUN — vx 0.31-0.82 vs baseline ~0.1 — at comparable E/hop → CoT several-fold lower). The confounds to
fix: (1) energy was NOT in the objective, (2) cadence was effectively fixed, (3) the spring arm trained ~+100 M
steps vs the baseline, and the baseline barely translated. Read first: top of `docs/JOURNAL.md` (2026-06-22),
`outputs/clean_curriculum/fair_centered/running_spring_results.md`, and the README "GPU-box safety".

## Where things stand
- Box DESTROYED. Local warm-start: `outputs/clean_curriculum/2026-06-21_g1_clean_s4_bound_clean_s4` (s4 bound, 37 MB).
- Spring fit (s4 bound work-loop): knee one_sided_linear, theta_engage 0.734, k=127.7; knee braking -59.6 W.
- IN-PLACE jump spring = NEGATIVE (collapses the hop, 0/3) — do NOT revisit unless adding a true clutch.
- Infra ready: `g1_bound_spring.yaml`, `run_bound_spring.sh`, `cadence_sweep.py`, command-aware
  `hop_failure_diag`/`hop_energy_compare`/`hop_spring_prep`, `hop_stay` anchor, `ElectricalRewardWrapper` (env.py).

## Design principles (fix all 3 confounds)
- **Energy IN the objective** (both arms): set `energy_reward_weight` so the gait optimizes electrical energy and
  EXPLOITS the spring. CALIBRATE it first (energy term ≈ 15-25 % of total reward at the baseline operating point).
- **Parity:** baseline & spring byte-identical except the spring block; SAME warm-start (s4), SAME steps, SAME
  energy weight. Kills the +100 M confound.
- **Free cadence** (bound env samples 1.6-2.6; do NOT set `hop_freq`): each arm finds its resonant-efficient cadence.
- **Compare CoT at a MATCHED ACHIEVED forward speed**, both arms ≥2/3 stable (or DR-averaged). Pick the target
  speed = whatever the baseline reliably achieves after A2.
- **Report no-regen AND regen** (~24 % sensitivity) + ohmic-vs-mechanical breakdown (% of braking ceiling recovered).

## Phases + ~time (core fits one ≤4 h GPU session)
| phase | step | where | ~time |
|---|---|---|---|
| 0  | provision box, bootstrap (`gpu_box_setup.sh`), upload s4 | box | 15 min |
| A1 | calibrate `energy_reward_weight` (energy ≈ 15-25 % of reward) | local | 20 min |
| A2 | train energy-objective BASELINE (no spring, free cadence, target speed, warm s4) | box | ~60 min |
| A3 | gate: runs stably at target speed (≥2/3); baseline CoT + cadence | local | 15 min |
| B1 | train SPRING arm (parity recipe + knee pogo; staged k if needed) | box | ~70 min |
| B2 | gate: survival; spring CoT + cadence shift | local | 15 min |
| C1 | compare CoT at matched speed (no-regen+regen, mechanical breakdown, % braking recovered) | local | 20 min |
| C2 | render baseline‖spring + write-up | local | 25 min |
| —  | rsync + wrap + DESTROY box | — | 20 min |
Core ≈ 3.5-4 h (GPU ~2.2 h = the two training runs).

## Pre-registered decision
Spring WINS iff CoT (no-regen, matched achieved speed, both ≥2/3 stable) drops vs the PARITY baseline. Report the
%, regen sensitivity, the cadence each arm chose, and the realized braking-recovery fraction.

## Contingencies / Session 2 (only if core is positive)
- Baseline won't translate/stabilize → lower target speed or rebalance reward, re-run (+~60 min).
- Spring still over-hops/destabilizes even with energy objective → the no-clutch limitation (a real finding);
  then test a TRUE CLUTCH (engage in stance only) or sweep lower k.
- Stiffness sweep k∈{60,90,127,160} to co-optimize (+~2.5 h); ≥3 seeds for significance.

## Box runbook + safety (unchanged)
Bootstrap → `git reset --hard origin/main` → verify `from pea.env import make_env` + HEAD. Launch detached at
`nice -19 ionice -c3`. Retry-ssh (`-o IPQoS=none`). Mac helpers `/tmp/bx`, `/tmp/bxpull`, `/tmp/bxpush` (repoint IP).
Pre-spend gate before launching. rsync each run; DESTROY the box when done (≤4 h GPU). nc -z probes can mislead — use
direct ssh to test reachability.
