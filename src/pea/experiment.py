"""Generalized experiment harness: sweep ROBOT x TASK x SPRING and tabulate
energy + performance metrics.

One entry point (`pea-sweep`) replaces the one-off analysis scripts. It rolls a
trained policy out at one or more task commands (stand / walk / run / …) and, for
each, sweeps a grid of post-hoc spring settings (stiffness, equilibrium, joint),
scoring every cell with `pea.metrics` (electrical no-regen/regen, ohmic %, braking
power, cost of transport, AND peak power / top speed / jump height). Results print
as a table and optionally write to CSV.

Design choices that make it general:
  * Robot-agnostic: the env (hence model + joints) comes from the run's config;
    motor constants are selected by --robot (pea.energy.MOTORS) or --kt/--r.
  * Task-agnostic: --tasks gives name:vx,vy,yaw commands; standing is 0,0,0.
  * Spring-agnostic: --joint/--kind/--k/--theta0 expand to a SpringConfig grid,
    applied POST-HOC on the rolled-out gait (cheap, no GPU). The SAME grid feeds
    an in-loop run by writing each cell to a config — see configs/.
  * Tolerant config load: stale fields in a run's config.yaml (e.g. a legacy
    cubic-spring `k3`) are dropped; the env is always built spring-free so
    qfrc_actuator is the pure motor torque the post-hoc spring subtracts from.

This is post-hoc (fixed-gait, optimistic) — the credible number is in-loop
retraining. The harness is the screening tool that decides which cells are worth
GPU time, and the same metrics scorer is reused on the in-loop run outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import pathlib

import numpy as np
import yaml

from pea import energy, metrics, springs
from pea.config import RunConfig, SpringConfig

TRANSIENT_S = 2.0


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    command: tuple[float, float, float]  # (vx, vy, yaw_rate)
    steps: int = 400


def _clean_run_config(run_dir: pathlib.Path) -> RunConfig:
    """Build a SPRING-FREE RunConfig from a run folder, tolerating stale fields.

    We want a clean env (no in-sim spring, no energy reward) so the logged
    qfrc_actuator is the pure motor torque; the post-hoc spring is subtracted in
    analysis. Preserves env_name/seed/impl so the right robot/task is loaded.
    """
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_fields = {f.name for f in dataclasses.fields(RunConfig)}
    raw = {k: v for k, v in raw.items() if k in run_fields}
    raw["spring"] = SpringConfig(kind="none")
    raw["energy_reward_weight"] = 0.0
    return RunConfig(**raw)


def load_clean_policy(run_dir: pathlib.Path):
    """(env, policy, mj_model) — clean (spring-free) env + the run's trained policy."""
    from pea import policy as policy_lib
    from pea.env import make_env

    cfg = _clean_run_config(run_dir)
    env = make_env(cfg)
    params = run_dir / "policy_params"
    if params.exists():
        policy = policy_lib.load_policy(env, cfg, params, deterministic=True)
    else:
        ckpt = policy_lib.latest_checkpoint(run_dir)
        policy = policy_lib.load_policy_from_checkpoint(env, cfg, ckpt, deterministic=True)
    return env, policy, env.mj_model


def rollout(env, policy, command, steps: int) -> dict:
    """Roll the policy out at a fixed joystick command; return a trajectory dict."""
    import jax
    import jax.numpy as jnp

    jit_reset, jit_step, jit_policy = (
        jax.jit(env.reset), jax.jit(env.step), jax.jit(policy))
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    cmd = jnp.array(command, dtype=jnp.float32)
    qpos, qvel, qfrc = [], [], []
    for _ in range(steps):
        if "command" in state.info:
            state.info["command"] = cmd
        rng, ar = jax.random.split(rng)
        action, _ = jit_policy(state.obs, ar)
        state = jit_step(state, action)
        if bool(state.done):
            break
        qpos.append(np.asarray(state.data.qpos))
        qvel.append(np.asarray(state.data.qvel))
        qfrc.append(np.asarray(state.data.qfrc_actuator))
    return dict(
        qpos=np.stack(qpos), qvel=np.stack(qvel), qfrc=np.stack(qfrc),
        dt=float(env.dt), total_mass=float(env.mj_model.body_mass.sum()),
        n=len(qpos),
    )


def _trim(traj: dict, transient_s: float) -> dict:
    skip = min(int(transient_s / traj["dt"]), max(traj["n"] - 10, 0))
    return {**traj, "qpos": traj["qpos"][skip:], "qvel": traj["qvel"][skip:],
            "qfrc": traj["qfrc"][skip:], "n": traj["n"] - skip}


def spring_grid(joints, kind: str, ks, theta0s) -> list[SpringConfig]:
    """Expand JOINTS x stiffness x equilibrium into SpringConfig cells, so one
    sweep compares several candidate joints (e.g. the pitch trio hip/knee/ankle)."""
    if isinstance(joints, str):
        joints = [joints]
    cells = []
    for joint in joints:
        for k, t0 in itertools.product(ks, theta0s):
            cells.append(SpringConfig(kind=kind, joint=joint, k=float(k), theta0=float(t0)))
    return cells


def run_sweep(run_dir, tasks, grid, kt, r, transient_s=TRANSIENT_S):
    """Roll out each task and score baseline + every spring cell. Returns rows."""
    env, policy, model = load_clean_policy(pathlib.Path(run_dir))
    act = metrics.actuated_dof_adrs(model)
    rows = []
    for task in tasks:
        traj = _trim(rollout(env, policy, task.command, task.steps), transient_s)
        base = metrics.evaluate(traj, model, act, kt, r, spec=None,
                                joint_substr=grid[0].joint if grid else None)
        rows.append(_row(task, None, base, base))
        for scfg in grid:
            spec = springs.from_config(scfg)
            rec = metrics.evaluate(traj, model, act, kt, r, spec=spec,
                                   joint_substr=scfg.joint)
            rows.append(_row(task, scfg, rec, base))
    return rows


def _row(task: Task, scfg, rec: dict, base: dict) -> dict:
    return {
        "task": task.name,
        "v_cmd": task.command[0],
        "joint": (scfg.joint if scfg else "-"),
        "kind": (scfg.kind if scfg else "none"),
        "k": (scfg.k if scfg else 0.0),
        "theta0": (scfg.theta0 if scfg else 0.0),
        "v_act": round(rec["mean_speed"], 3),
        "elec_W": round(rec["elec_W"], 1),
        "ohmic_pct": round(rec["ohmic_pct"], 1),
        "braking_W": round(rec["braking_W"], 1),
        "jbrake_W": round(rec.get("joint_braking_W", float("nan")), 2),
        "peak_mech_W": round(rec["peak_mech_W"], 0),
        "peak_tau": round(rec["peak_abs_torque"], 1),
        "rms_tau": round(rec["rms_torque"], 1),
        "max_z": round(rec["max_base_z"], 3),
        "cot": round(rec["cot"], 3),
        "dW_vs_base": round(rec["elec_W"] - base["elec_W"], 2),
        "dpct": round(100 * (rec["elec_W"] / max(base["elec_W"], 1e-9) - 1), 1),
    }


def _parse_tasks(specs: list[str], steps: int) -> list[Task]:
    tasks = []
    for s in specs:
        name, _, vec = s.partition(":")
        cmd = tuple(float(x) for x in vec.split(",")) if vec else (0.0, 0.0, 0.0)
        cmd = (cmd + (0.0, 0.0, 0.0))[:3]
        tasks.append(Task(name=name, command=cmd, steps=steps))
    return tasks


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep robot x task x spring; tabulate "
                                "energy + performance metrics.")
    p.add_argument("--run", required=True, help="run folder (trained policy + config)")
    p.add_argument("--tasks", nargs="+", default=["walk:1,0,0"],
                   help="name:vx,vy,yaw commands, e.g. stand:0,0,0 walk:1,0,0 run:2,0,0")
    p.add_argument("--joint", nargs="+", default=["hip_pitch", "knee", "ankle_pitch"],
                   help="spring target joint(s) (substring); default = the pitch trio")
    p.add_argument("--kind", default="linear", choices=["linear", "constant", "semiparabolic"])
    p.add_argument("--k", nargs="+", type=float, default=[40, 68, 100],
                   help="stiffness sweep values [N*m/rad]")
    p.add_argument("--theta0", nargs="+", type=float, default=[-0.29],
                   help="equilibrium-angle sweep values [rad]")
    p.add_argument("--robot", default="g1", help="motor-constant preset (pea.energy.MOTORS)")
    p.add_argument("--kt", type=float, default=None, help="override joint-side Kt [N*m/A]")
    p.add_argument("--r", type=float, default=None, help="override winding R [Ohm]")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--out", default=None, help="optional CSV path")
    args = p.parse_args()

    mc = energy.motor_constants(args.robot)
    kt = args.kt if args.kt is not None else mc.kt
    r = args.r if args.r is not None else mc.r
    tasks = _parse_tasks(args.tasks, args.steps)
    grid = spring_grid(args.joint, args.kind, args.k, args.theta0)
    rows = run_sweep(args.run, tasks, grid, kt, r)

    cols = ["task", "v_act", "joint", "k", "theta0", "elec_W", "ohmic_pct",
            "jbrake_W", "peak_tau", "rms_tau", "cot", "dW_vs_base", "dpct"]
    print(f"\nrobot={args.robot} Kt={kt} R={r}  (post-hoc, no-regen; "
          f"dW/dpct vs the task's no-spring baseline)")
    print("  ".join(f"{c:>10s}" for c in cols))
    for row in rows:
        print("  ".join(f"{row[c]:>10}" for c in cols))

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
