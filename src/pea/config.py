"""Run configuration.

One YAML config fully describes a run (baseline or spring-active); the same
train/rollout/analyze code path consumes either. The resolved config is copied
into every run folder for provenance.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class SpringConfig:
    """Parallel knee spring. kind='none' disables it (baseline)."""

    kind: str = "none"  # "none" | "constant" | "linear" | "semiparabolic"
    joint: str = "knee"  # target joint, matched by substring: "knee" | "hip_pitch"
    k: float = 0.0      # linear stiffness [N*m/rad], or per-element quadratic
                        # stiffness [N*m/rad^2] when kind="semiparabolic"
    theta0: float = 0.0  # equilibrium joint angle, rad (linear)
    tau0: float = 0.0   # constant preload torque, N*m (constant only)
    p1: float = 0.0     # lower onset, rad (semiparabolic only)
    p2: float = 0.0     # upper onset, rad (semiparabolic only)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    name: str
    env_name: str = "G1JoystickFlatTerrain"
    # MJX implementation. Playground 0.2.0 defaults to 'warp', which is broken
    # on Mac (no CUDA); 'jax' runs on both Mac CPU and Colab GPU.
    impl: str = "jax"
    seed: int = 1
    # None -> use MuJoCo Playground's recommended value for the env.
    num_timesteps: int | None = None
    domain_randomization: bool = True
    spring: SpringConfig = dataclasses.field(default_factory=SpringConfig)
    # Overrides merged onto Playground's recommended brax PPO params.
    ppo: dict = dataclasses.field(default_factory=dict)
    # Overrides merged onto the env's reward weights (reward_config.scales).
    # The spring and no-spring conditions MUST share identical reward_scales for
    # the in-loop comparison to be fair (the default G1 reward has energy=0).
    reward_scales: dict = dataclasses.field(default_factory=dict)


def load_config(path: str | pathlib.Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    spring = SpringConfig(**(raw.pop("spring", None) or {}))
    return RunConfig(spring=spring, **raw)


def save_config(cfg: RunConfig, path: str | pathlib.Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)


def resolve_runs_dir() -> pathlib.Path:
    """Where run folders live, by priority:

    1. $PEA_RUNS_DIR
    2. Colab: /content/drive/MyDrive/pea_runs (requires Drive mounted)
    3. Mac with Google Drive for Desktop: <My Drive>/pea_runs
    4. ./outputs (local fallback, gitignored)
    """
    if env := os.environ.get("PEA_RUNS_DIR"):
        return pathlib.Path(env)
    colab_drive = pathlib.Path("/content/drive/MyDrive")
    if colab_drive.is_dir():
        return colab_drive / "pea_runs"
    # Drive for Desktop localizes the folder name ("My Drive", "Мой диск", …);
    # fall back to the only visible top-level dir in the account mount.
    for account in pathlib.Path.home().glob("Library/CloudStorage/GoogleDrive-*"):
        for name in ("My Drive", "Мой диск"):
            if (account / name).is_dir():
                return account / name / "pea_runs"
        visible = [
            d for d in account.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if len(visible) == 1:
            return visible[0] / "pea_runs"
    return pathlib.Path("outputs")


def new_run_dir(
    cfg: RunConfig,
    root: str | pathlib.Path | None = None,
    suffix: str = "",
) -> pathlib.Path:
    root = pathlib.Path(root) if root else resolve_runs_dir()
    date = datetime.date.today().isoformat()
    name = f"{date}_{cfg.name}" + (f"_{suffix}" if suffix else "")
    run_dir = root / name
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = root / f"{name}_{n}"
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    return run_dir
