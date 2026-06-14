# CLAUDE.md — Parallel-Elastic Efficiency Study

## Project goal
Two parts, **in sequence — finish Part 1 before starting Part 2.**

### Part 1 — Energy efficiency (current focus)
Test whether adding a tunable **parallel elastic** spring (target now **hip-pitch**,
not the knee — see `docs/RESULTS.md`) on a humanoid (Unitree G1) reduces the
*electrical* energy of walking, by offloading motor torque. Headline metric: cost
of transport / total electrical power.

Mechanism / why this should work:
- Motor copper loss scales with torque squared: `P_loss ≈ (τ/Kt)² · R`. Offloading
  torque to a passive spring cuts heat **quadratically**, not linearly.
- Two distinct wins: (1) lower RMS torque → lower copper loss; (2) energy recovery —
  the spring stores energy the motor would otherwise dissipate (braking) and returns it.
- **Parallel**, not series: the goal is torque offloading, not force control. A parallel
  spring sits beside the motor and avoids the large-force-bandwidth penalty of series
  compliance.
- *Reality check — the study's MAIN FINDING (2026-06-14/15): **gearing is the crux**.*
  On the high-geared G1 (22.5:1) ohmic is only ~4 % of the budget, and the **in-loop**
  spring is actually **WORSE** for walking (+7 %, reversing the optimistic post-hoc −3.8 %;
  the always-on spring fights the gait, no clutch) — **nine negative results** in
  `docs/negative_results.md`. But on the **LOW-gear Go1 quadruped** (6.33:1, ohmic **54 %**)
  a **CONSTANT knee preload** cuts **−17 to −20 % in-loop with no stability cost** (2 seeds)
  — the one positive result. **ACTIVE direction:** the **Go1 load-carrying program** — an
  adaptive *per-leg* knee preload that scales with payload, one blind load-robust controller;
  see **`docs/load_program.md`**. Direction map: `docs/directions.md`.

### Part 2 — Explosive moves (after Part 1)
Test whether the **same adaptive elastic** helps EXPLOSIVE moves: vertical **jump
height**, broad-**jump distance**, **drop-landing** (jumping down), explosive
**sprint** start. Unlike Part 1 these may be **one-shot** (a single max jump — no
cross-cycle energy recovery, cost-of-transport does not apply) or **repetitive**
(continuous hopping — recovery applies). This is a genuine **bifurcation** from
Part 1 (repetitive-motion efficiency); see `docs/directions.md`. Part 2 itself
splits into two sub-cases with opposite verdicts on the high-geared G1:
- **(B1) Performance — jump HEIGHT / top speed.** Which architecture helps is set
  ENTIRELY by which wall takeoff hits, because a **series** spring can never add
  torque (force through it equals the motor force; it only amplifies *speed/power*
  by storing energy then releasing it fast) and a **parallel** spring can never
  beat the speed wall (it only adds *force*). So: **torque-limited takeoff →
  parallel helps; speed-limited takeoff → series helps and parallel does nothing.**
  **RESOLVED from the real specs** (model jnt_actfrcrange + Unitree G1 URDF velocity
  limits): the **knee is SPEED-limited** (139 N·m but only **20 rad/s**, and the
  walker already uses ~52–67 % of that speed; a jump needs the knee faster still),
  while the **hip is TORQUE-limited** (88 N·m, only ~11 % of its 32 rad/s). Since
  the knee is the jump extensor and it is speed-capped, **a parallel spring cannot
  raise G1 jump HEIGHT** — height points off the G1 (series compliance or a low-gear
  platform). The hip-torque case and the efficiency/landing case (B2) remain valid
  for a parallel spring. (Confirm with a max-jump policy; the walking data already
  pins the knee near its speed ceiling.)
- **(B2) Efficiency / peak-load of explosive moves — the defensible parallel case,
  LEAD with this.** Even with height capped, the spring cuts the **energy and peak
  load** of each push-off and landing. Parallel is the correct architecture here,
  and the gear speed-cap does NOT apply to **landing** (load is set by impact
  velocity, not motor speed). Jump/landing torques (≥139 N·m) dwarf walking, so the
  ohmic (∝ τ²) and braking-recovery channels are far larger than the ~3 % walking
  lever — this is where the spring most plausibly pays on the stock G1.

Metric: performance (height/distance, takeoff velocity, top speed, peak power) for
B1; energy + peak load for B2. The tunable spring + dead-zone clutch is the
between-conditions knob (engage for explosive, disengage for precise/efficient).
Gearing is the crux for BOTH parts; maximum jump HEIGHT may point off the G1 (a
low-gear quadruped or a DecART-style leg-length / series element).

## Session ritual
Project memory lives on disk, not in any single conversation. Work in short, task-scoped
sessions and start fresh ones freely (a new session reloads this file automatically).
- **Start** a session: skim `docs/JOURNAL.md` → "Current state".
- **End** a session: run `/wrap` (`.claude/commands/wrap.md`) to update `docs/JOURNAL.md`
  and, if anything changed, this file.

## Experiment design (staged — do in this order)
1. **Baseline**: train a G1 flat-ground "walk straight" policy with **no spring**.
2. **Log**: roll the policy out, record knee angle θ and knee torque τ over the gait.
3. **Post-hoc analysis** (fast, optimistic): subtract `τ_spring(θ)` from the recorded
   motor torque on the *same* trajectory and recompute electrical losses. Upper bound —
   it assumes the gait doesn't change. Use it only to sanity-check that the effect exists.
4. **In-loop** (the credible result): inject `τ_spring(θ)` into the simulation, **retrain**
   the policy so the gait adapts, then compare best-without-spring vs best-with-spring.
   The post-hoc number alone is NOT a sufficient result.

## Metrics — measure ELECTRICAL energy, not mechanical work
- Copper loss `P = (τ/Kt)² · R`. Need approximate motor constants (Kt, R) for the G1
  actuators; approximate is fine for a *relative* comparison.
- Negative work is **dissipated, not regenerated** (no-regen) — JUSTIFIED for the G1
  by back-EMF physics (at locomotion joint speeds the back-EMF is below the ~48 V bus,
  so returning current needs a boost converter commercial drivers lack) + the
  documented regenerative-resistor pattern. NOT a verified spec; exceptions exist (MIT
  Cheetah 2013, reportedly Tesla Optimus). The spring's win lives precisely in this
  dissipated braking, so it is **no-regen-dependent** — report the ~24 % regen-vs-no-regen
  sensitivity. Full treatment: `docs/running_program.md`.
- Headline metric: **cost of transport (CoT)**. Report % reduction in CoT and copper loss.

## Tech stack
- **Simulator: MuJoCo, end to end.** MuJoCo Playground (ships a Unitree G1 flat-walk env)
  + MJX for GPU training; the same MJX env on CPU JAX for local rollout/analysis.
- **Training: rented immers.cloud H100 PCIe** (~342 ₽/hr, per-second billing, SSH —
  Claude drives it end to end: bootstrap via `scripts/gpu_box_setup.sh`, train, rsync
  results into the Drive folder, then DELETE the server, not stop). Measured: ~47k
  env-steps/s with DR → ~70 min per 200M-step run. Colab T4 notebook remains as free
  fallback (~9k steps/s, ~7 h, checkpoint-resume across disconnects).
- **Analysis/inference: local MacBook Pro**, CPU MuJoCo + **CPU JAX** (only GPU/Metal JAX
  is dead; CPU JAX runs fine on Mac).
- **Language: Python.**

## Repo structure (three layers)
- `src/pea/` — shared core, imported by everything:
  - `env.py` (G1 walking env; optionally injects spring torque)
  - `springs.py` (`τ_spring(θ)`: linear + nonlinear/tunable)
  - `energy.py` (copper-loss model + cost of transport)
  - `policy.py` (network def + load/save), `config.py`
- `scripts/` — `train.py` (Colab), `rollout.py` (local), `analyze.py` (local)
- `configs/` — no-spring (`baseline.yaml`, `walk_baseline.yaml`) + spring arms (`spring_hip_linear.yaml`, `spring_linear.yaml`, `spring_constant.yaml`, `spring_semiparabolic.yaml`)
- `notebooks/colab_train.ipynb` — thin runner: pip-install repo, mount Drive, call train.py
- `outputs/` — gitignored; one folder per run (config, checkpoint, metrics, trajectory)

## Critical design rules
- The spring lives in `src/pea/springs.py` and MUST be callable two ways: as an in-sim
  torque (in-loop training/rollout) AND as a standalone `τ_spring(θ)` function (post-hoc).
  This single decision lets the same code serve both halves of the experiment.
- The spring is **config-selected, not baked into `train.py`** — the same script produces
  the baseline and spring-active runs by swapping a config (keeps them on identical paths).
- All real logic lives in versioned `.py` files; the Colab notebook is only a thin runner
  that `pip install`s the repo. Never let the notebook hold logic.

## Workflow / data flow (no manual downloads)
Edit in VS Code → `git push` to GitHub → Colab `pip install`s the repo + runs `train.py`
→ writes a run folder (config, checkpoint, metrics, trajectory) to **Google Drive** →
"Google Drive for desktop" mirrors it to the Mac → `analyze.py` reads the synced folder.

## Constraints & gotchas — what to AVOID
- **Always `impl: jax`** (a RunConfig field): Playground 0.2.0 defaults the env to
  MuJoCo Warp, which is broken on Mac and was never validated for this project.
- **jax pinned `<0.10`**: brax 0.14.2 (latest) calls `jax.device_put_replicated`,
  removed in jax 0.10. Revisit when brax releases.
- **On the Mac, prefix every venv command with `env -u PYTHONPATH`** — the shell
  profile sources a ROS 2 workspace whose PYTHONPATH shadows venv numpy/jax.
- **Drive mount is locale-named** («Мой диск», not "My Drive") — `config.py`
  handles it; don't hardcode the English name anywhere.
- **GPU box over VPN**: SSH banner-exchange timeouts = flaky VPN exit; bypass the
  box IP (`route add -host <ip> <gateway>`) or switch VPN location. Always launch
  remote training detached (`nohup … &`) so it survives drops; monitor
  `metrics.jsonl` (per-eval append), not stdout (block-buffered under nohup).
- **NO Isaac Sim / Isaac Lab** — they need an NVIDIA RTX GPU + CUDA; dev machine is a
  MacBook Pro (no CUDA). This is the reason we're MuJoCo-only.
- **Don't rely on jax-metal / GPU MJX on the Mac** — jax-metal is abandoned. CPU only locally.
- **Colab free tier**: T4 GPU (allocation is a lottery), **ephemeral disk** — always mount
  Drive and save checkpoints there or you lose the run. MJX JIT compile ≈ 1–3 min/session.
- **Don't** put code on Drive (loses version control) or model binaries in Git (bloats repo).
- **Don't** train on Mac CPU except to smoke-test the pipeline (too slow for a 29-DoF humanoid).
- The spring is **always engaged**, so it resists the motor in phases where its torque is
  unwanted. Tuning (rate, equilibrium angle, nonlinearity) is a real optimization — ideally
  co-optimize spring params with the policy. (This is why some hardware PEAs add a clutch.)
- First demo: one sensibly hand-tuned spring curve is enough; parameter sweeps come later.

## Background / framing
The G1 ships with stiff quasi-direct-drive actuators and **no springs**, making it a clean
testbed for adding parallel elasticity. (Even Agility's commercial Digit V4 appears to have
dropped the Cassie-lineage leg leaf springs for a rigid leg — distal passive compliance pays
off most in running/hopping, less for precise manipulation.) The project is essentially
testing whether *targeted parallel elasticity is worth the added mechanism and control
complexity* for walking efficiency.

## Session ritual (end of every work session)
Keep the project's memory on disk, not in chat. Before closing a session:
1. Append a dated entry to `docs/JOURNAL.md` — what you did, what you decided and why,
   what's open/broken, and the single next step.
2. If any decision, convention, constraint, or file-structure choice changed, update this
   `CLAUDE.md` to match.

Claude: when a session is wrapping up, or the user signals they're done, proactively draft
the `docs/JOURNAL.md` entry and point out any updates this file needs.
