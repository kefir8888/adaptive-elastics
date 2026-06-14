# adaptive-elastics

Does a tunable **parallel elastic spring at the knee** reduce the *electrical*
energy of walking for a Unitree G1 humanoid? Motor copper loss scales with
torque squared (`P ≈ (τ/Kt)²·R`), so offloading torque to a passive spring cuts
heat quadratically — if the gait can exploit it. MuJoCo end to end; metrics are
copper loss and cost of transport, not mechanical work. See `CLAUDE.md` for the
full experiment design.

## Current state (2026-06-14)

Milestones 1–3 done; the G1-walk in-loop gate is prepared (CPU-validated, needs GPU).
Detail in `docs/JOURNAL.md` (start here), `RESULTS.md`, `PLAN.md`, `mechanism.md`,
`related_work.md`, and the three docs written this session: **`directions.md`**
(six-direction map), **`running_program.md`** (the next focus), **`taxonomy.md`**
(cross-morphology study).

**Meaningful outcomes of the 2026-06-14 dialogue:**
- **The G1-walk energy win is small and gear-limited.** With real-ish constants
  ohmic is only ~4 % of the motor budget; the post-hoc hip-spring saves ~3 %
  whole-body and **~0 % under regeneration** — the win is braking-energy recovery,
  not the quadratic copper term.
- **Torque-vs-speed resolved from the real Unitree specs** (jnt_actfrcrange + G1
  URDF): **knee is SPEED-limited** (139 N·m / 20 rad/s), **hip is TORQUE-limited**
  (88 N·m / 32 rad/s). So a parallel spring **cannot raise G1 jump height** (knee
  speed-capped → series/low-gear for height); it helps **hip torque + efficiency/
  landing**. The biped knee wants a constant-torque element, not a torsion spring.
- **Two value axes:** efficiency (energy/CoT/ohmic) AND performance/durability
  (peak power, jump/sprint, and **gearbox wear** = peak+RMS torque).
- **Two validity confounds caught:** (1) the baseline policy *chatters* (a default-
  zero `action_rate` penalty → ~55 % sawtooth) which inflates energy and dilutes
  the spring %; fix = enable `action_rate`. (2) The baseline was **energy-naive** —
  the spring must be measured against an **energy-AWARE** baseline, never the naive
  walker. Both raise the bar for the gate.
- **No-regeneration justified for the G1** (back-EMF below the bus at locomotion
  speeds); exceptions exist (MIT Cheetah, reportedly Optimus); ~24 % sensitivity.
- **Next focus: G1 running for EFFICIENCY** (`running_program.md`) — bigger braking
  energy than walking, where the spring should pay more.
- **Cross-morphology study mapped** (`taxonomy.md`): the spring's benefit **shifts
  with gear** — energy at low gear (Berkeley 9:1, Go1/Barkour 6:1), wear at high
  gear (Spot, ANYmal), G1 the middle. Biped → hip-pitch; quadruped → thigh+knee.
  Kinematics: serial (Berkeley, H1) · parallel ankle (G1, Booster) · full parallel
  (Cassie, DecART). A comprehensive study is **~£160–210** on ready Playground envs.

Spring (walk gate): **hip-pitch** linear `k=68 N·m/rad, θ0=-0.29` vs matched no-spring.
New tooling: `pea-sweep` (multi-joint, energy + wear), `metrics.{saturation,
fit_linear_spring}`, the motor torque–speed envelope (`energy.G1_LIMITS/G1_JOINT_VEL`),
and `configs/run_baseline.yaml` (the energy-aware, smooth runner).

### To run Milestone 4 on a GPU machine

```sh
# on the box (Ubuntu+CUDA): bootstrap, then
curl -fsSL https://raw.githubusercontent.com/kefir8888/adaptive-elastics/main/scripts/gpu_box_setup.sh | bash
# 1) calibration: 5 short no-spring runs at different energy weights
nohup bash scripts/calib_sweep.sh > ~/calib.log 2>&1 &
# 2) gate (after picking the weight W from calibration), spring vs no-spring:
pea-train --config configs/spring_hip_linear.yaml --energy-weight=W --output_dir ~/runs
pea-train --config configs/baseline_gate.yaml     --energy-weight=W --output_dir ~/runs
# then rsync ~/runs into the Drive pea_runs folder and, locally:
#   pea-rollout --run <run> ; pea-analyze --run <run>   (reports ohmic, CoT, total power)
```

## Setup (local, macOS)

```sh
uv sync          # creates .venv with Python 3.12 + pinned deps
```

**Gotcha:** if your shell exports `PYTHONPATH` (e.g. a sourced ROS 2 workspace),
it leaks foreign site-packages into the venv and breaks numpy/jax imports.
Run project commands with it cleared:

```sh
env -u PYTHONPATH uv run pea-train --config configs/baseline.yaml --smoke
```

## Workflow

```
edit (VS Code) → git push → Colab: notebooks/colab_train.ipynb (pip installs
repo, runs pea-train) → run folder on Google Drive → Drive for Desktop syncs
to Mac → pea-rollout / pea-analyze locally
```

- `pea-train --config configs/baseline.yaml` — train (Colab GPU; `--smoke` for
  a tiny CPU pipeline test; `--restore <run>` to resume)
- `pea-rollout --run <run_dir> --video --viewer` — replay locally on CPU,
  writes `trajectory.npz`
- `pea-analyze --run <run_dir>` — knee angle/torque/work-loop plots and stats

Run folders land in `$PEA_RUNS_DIR`, else Drive (`pea_runs/`), else `outputs/`.

## Layout

- `src/pea/` — all logic: env, springs, energy model, policy IO, train/rollout/analyze
- `configs/` — one YAML per experiment arm (baseline / linear / nonlinear spring)
- `scripts/` — thin shims onto `src/pea` entry points
- `notebooks/colab_train.ipynb` — thin Colab runner, no logic
- `docs/JOURNAL.md` — session-by-session project memory
