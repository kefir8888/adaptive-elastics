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
                        # | "one_sided_linear" | "preload_dr"
    joint: str = "knee"  # target joint, matched by substring: "knee" | "hip_pitch"
                         # | "calf" (Go1 knee)
    k: float = 0.0      # linear stiffness [N*m/rad], or per-element quadratic
                        # stiffness [N*m/rad^2] when kind="semiparabolic"
    theta0: float = 0.0  # equilibrium joint angle, rad (linear)
    tau0: float = 0.0   # constant preload torque, N*m (constant only);
                        # for kind="preload_dr" it is the MAX of the U(0,tau0) draw
    p1: float = 0.0     # lower onset, rad (semiparabolic only)
    p2: float = 0.0     # upper onset, rad (semiparabolic only)
    # one_sided_linear fields (see springs.SpringSpec):
    theta_engage: float = 0.0  # engage angle, rad (one_sided_linear only)
    engage_sign: float = 1.0   # which side engages (one_sided_linear only);
                               # +1 = the side where theta grows past theta_engage
    # Optional per-episode randomization of the one_sided_linear stiffness k
    # (DR over k). DEFAULT OFF: the experiment sets k from the S3 work-loop
    # analysis, so a single fixed k is the default; DR over k is an OPTIONAL
    # robustness knob, not the headline condition. When k_dr=True the per-episode
    # k is drawn U(k_dr_min, k_dr_max); k above is then ignored.
    k_dr: bool = False         # enable stiffness DR (one_sided_linear only)
    k_dr_min: float = 0.0      # low end of the U(k_dr_min, k_dr_max) draw
    k_dr_max: float = 0.0      # high end of the U(k_dr_min, k_dr_max) draw


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
    # e.g. {action_rate: -0.01} to enable the (default-zero) smoothness penalty
    # that suppresses control chatter; {feet_air_time: 4.0} to push a running gait.
    reward_scales: dict = dataclasses.field(default_factory=dict)
    # Overrides merged onto TOP-LEVEL env config keys (not reward scales), e.g.
    # {lin_vel_x: [0.0, 3.0]} to widen the joystick speed range for running (the
    # default is [-1, 1] m/s — commanding faster is out-of-distribution and falls).
    env_overrides: dict = dataclasses.field(default_factory=dict)
    # Weight on a TOTAL-ELECTRICAL energy penalty added to the reward each step:
    # per actuated DoF, max(tau*omega + (tau/Kt)^2*R, 0) (mechanical + ohmic,
    # no regeneration) — the SAME quantity we evaluate, so reward and metric are
    # aligned (unlike Playground's built-in `energy`, which is mechanical only
    # and omits the ohmic channel where the spring helps most). 0 disables it.
    # MUST be identical across the spring and no-spring conditions. Negative.
    energy_reward_weight: float = 0.0
    # Which energy.MOTORS[...] constants the electrical penalty uses (joint-side Kt,
    # winding R). Robot-specific: "g1" (default, geared humanoid) or "go1_knee" (Go1
    # quadruped, low gear -> ohmic actually bites). Eval uses the same lookup.
    energy_motor: str = "g1"
    # Payload domain randomization (Go1): add a box of +U(0, payload_max_kg) kg to the
    # trunk each episode (0 disables -> stock per-env DR). The controller does NOT observe
    # it (real-world realism); one blind policy learns load-robust walking. See payload.py.
    payload_max_kg: float = 0.0
    # Scales the rough-terrain heightfield elevation (hfield_size[:,2]). 1.0 = stock
    # (Go1 rough = 5 cm max bumps); 0.5 halves all bumps (2.5 cm) for an easier-terrain
    # start / roughness curriculum. No-op on flat terrain (no hfields). Eval must match.
    terrain_height_scale: float = 1.0
    # Forward-running command override (Go1). Empty dict (default) = stock behaviour:
    # the joystick samples a SYMMETRIC velocity command (vx in [-1.5, 1.5], mean zero,
    # sometimes zeroed), so the policy is never required to run forward and learns to
    # stand. When this dict is non-empty, make_env replaces the env's sample_command
    # with one that always commands FORWARD motion. Expected keys (all floats):
    #   vx_min, vx_max  -- forward speed range (BOTH positive); vx ~ U(vx_min, vx_max)
    #   vy_max          -- sideways speed magnitude; vy ~ U(-vy_max, vy_max)
    #   vyaw_max        -- turn-rate magnitude;     vyaw ~ U(-vyaw_max, vyaw_max)
    # No axis is ever zeroed, so every episode demands forward running.
    command_forward: dict = dataclasses.field(default_factory=dict)


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
