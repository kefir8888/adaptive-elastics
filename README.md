# adaptive-elastics

Does a tunable **parallel elastic spring at the knee** reduce the *electrical*
energy of walking for a Unitree G1 humanoid? Motor copper loss scales with
torque squared (`P ≈ (τ/Kt)²·R`), so offloading torque to a passive spring cuts
heat quadratically — if the gait can exploit it. MuJoCo end to end; metrics are
copper loss and cost of transport, not mechanical work. See `CLAUDE.md` for the
full experiment design.

## Current state (2026-06-13)

Milestones 1–3 done (baseline trained, knee/hip logged, offline spring analysis).
**Milestone 4 (in-loop comparison) is prepared and CPU-validated; it needs a GPU
machine to run.** Full detail in `docs/PLAN.md`, `docs/RESULTS.md`, `docs/JOURNAL.md`
(start here), `docs/mechanism.md`, `docs/related_work.md`.

Key decisions: spring target = **hip-pitch**, linear spring `k=68 N·m/rad,
θ0=-0.29 rad` (`configs/spring_hip_linear.yaml`); matched no-spring
(`configs/baseline_gate.yaml`). Reward and headline metric = **total electrical
power** (mechanical + ohmic, no regeneration); we also report the Kt/R-independent
ohmic-loss percentage and the cost of transport. Energy-weight placeholder
`-2.5e-4`, to be fixed by calibration. Kt/R are estimates (no hardware), reported
as a band.

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
