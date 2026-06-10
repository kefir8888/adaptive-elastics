# CLAUDE.md — Parallel-Elastic Knee Efficiency Study

## Project goal
Test whether adding a tunable **parallel elastic** spring at the knee of a humanoid
(Unitree G1) reduces the *electrical* energy of walking, by offloading motor torque.

Mechanism / why this should work:
- Motor copper loss scales with torque squared: `P_loss ≈ (τ/Kt)² · R`. Offloading
  torque to a passive spring cuts heat **quadratically**, not linearly.
- Two distinct wins: (1) lower RMS torque → lower copper loss; (2) energy recovery —
  the spring stores energy the motor would otherwise dissipate (braking) and returns it.
- **Parallel**, not series: the goal is torque offloading, not force control. A parallel
  spring sits beside the motor and avoids the large-force-bandwidth penalty of series
  compliance.

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
- Decide whether negative work is recovered (regen) or dissipated. Assume **dissipated**
  unless the G1 actually regenerates — the spring wins precisely where the motor would
  otherwise burn negative work as heat.
- Headline metric: **cost of transport (CoT)**. Report % reduction in CoT and copper loss.

## Tech stack
- **Simulator: MuJoCo, end to end.** MuJoCo Playground (ships a Unitree G1 flat-walk env)
  + MJX for GPU training; plain CPU MuJoCo for rollout and analysis.
- **Training: Google Colab (T4)** to start; rent a cloud NVIDIA GPU for iteration/sweeps.
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
- `configs/` — `baseline.yaml`, `spring_linear.yaml`, `spring_nonlinear.yaml`
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
