# adaptive-elastics

Does a tunable **parallel elastic spring at the knee** reduce the *electrical*
energy of walking for a Unitree G1 humanoid? Motor copper loss scales with
torque squared (`P ≈ (τ/Kt)²·R`), so offloading torque to a passive spring cuts
heat quadratically — if the gait can exploit it. MuJoCo end to end; metrics are
copper loss and cost of transport, not mechanical work. See `CLAUDE.md` for the
full experiment design.

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
