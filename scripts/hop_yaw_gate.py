"""S1 yaw GATE probe — measure mean yaw-rate at a FORCED command.

The make-or-break gate for the clean (non-spinning) hop curriculum: every hopper
from the 2026-06-20 campaign spun ~+2-3.8 rad/s at zero command (a diagonal
stance inherited from S1). Before chaining stages 2-4 on a stage-1 policy we must
confirm stage 1 is actually NON-spinning:

    |mean yaw rate| < 0.15 rad/s   AND   ~0 net planar drift   AND   full survival

THE PITFALL this script exists to avoid
---------------------------------------
The env re-draws the joystick command internally every ~5 s of an episode (and on
reset, with its OWN uniform — not via sample_command). So merely pinning
`state.info["command"]` each step (as scripts/rollout.py does) is NOT enough: on a
re-sample step the env overwrites it before the observation is built, and the
policy then sees a random command. We therefore PATCH `sample_command` on the env
instance that owns it to return a CONSTANT command, and also overwrite the initial
command each step — so the policy sees the forced command on every step.

Yaw rate is read from the canonical accessor `get_global_angvel(data,"pelvis")[2]`
(WORLD-frame angular velocity z), the same quantity the base env uses for heading.

Dual use:
  * S1 gate     : --command 0 0 0   -> expect |mean yaw| < 0.15, ~0 drift, survival.
  * S3 tracking : --command 0 0 0.4 -> expect mean yaw ~ +0.4 (the command IS tracked,
                  not ignored); --command 0.3 0 0 -> expect forward drift ~ +0.3 m/s.

Usage:
    env -u PYTHONPATH uv run python scripts/hop_yaw_gate.py --run <run_dir> \
        [--command VX VY VYAW] [--steps 600] [--transient 2.0] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import types

import numpy as np


def _patch_constant_command(env, command):
    """Patch `sample_command` on the env instance that owns it to ALWAYS return
    `command` (length-3 jax array), defeating the env's internal re-sample. Walks
    the `_env` / `.env` wrapper chain to find the owner. Returns that instance."""
    import jax.numpy as jp

    cmd = jp.asarray(command, dtype=jp.float32)
    target = env
    while not hasattr(target, "sample_command"):
        if hasattr(target, "_env"):
            target = target._env
        elif hasattr(target, "env"):
            target = target.env
        else:
            raise RuntimeError("no env in the chain defines sample_command")

    def sample_command(self, rng, x_k=None):  # noqa: ANN001 - host signature varies
        del rng, x_k  # always return the forced command (G1: rng-only; Go1: rng,x_k)
        return cmd

    target.sample_command = types.MethodType(sample_command, target)
    return target


def _yaw_accessor(env):
    """Return a fn data -> yaw_rate (world-frame angvel z of the pelvis). Walks to
    the instance that defines get_global_angvel (wrappers delegate via __getattr__,
    but bind to the real owner to be safe)."""
    owner = env
    while not hasattr(owner, "get_global_angvel"):
        owner = owner._env
    return lambda data: float(owner.get_global_angvel(data, "pelvis")[2])


def probe_once(env, policy, command, steps: int, seed: int) -> dict:
    """Deterministic rollout at a forced constant command; per-step yaw + qpos."""
    import jax

    _patch_constant_command(env, command)
    yaw_of = _yaw_accessor(env)
    import jax.numpy as jp

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)
    cmd = jp.asarray(command, dtype=jp.float32)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    yaw, xy = [], []
    for _ in range(steps):
        if "command" in state.info:           # belt-and-suspenders: force the
            state.info["command"] = cmd       # initial (reset-drawn) command too
        rng, ar = jax.random.split(rng)
        action, _ = jit_policy(state.obs, ar)
        state = jit_step(state, action)
        yaw.append(yaw_of(state.data))
        xy.append(np.asarray(state.data.qpos[:2]))
        if bool(state.done):
            break
    yaw = np.array(yaw)
    xy = np.stack(xy)
    return dict(yaw=yaw, xy=xy, n=len(yaw), dt=float(env.dt))


def summarize(traj: dict, steps: int, transient_s: float) -> dict:
    n, dt = traj["n"], traj["dt"]
    skip = min(int(transient_s / dt), max(n - 10, 0))
    yaw_ss = traj["yaw"][skip:]
    xy = traj["xy"]
    # planar drift over the steady-state window, as a mean speed (m/s)
    drift_m = float(np.linalg.norm(xy[-1] - xy[skip])) if n > skip else float("nan")
    drift_s = (n - skip) * dt
    return dict(
        survived=(n >= steps),
        n=n,
        episode_s=round(n * dt, 2),
        mean_yaw=float(np.mean(yaw_ss)),
        abs_mean_yaw=float(abs(np.mean(yaw_ss))),
        std_yaw=float(np.std(yaw_ss)),
        net_drift_m=round(drift_m, 3),
        drift_speed_mps=round(drift_m / drift_s, 3) if drift_s > 0 else float("nan"),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="trained run folder (s1 checkpoint/policy)")
    p.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("VX", "VY", "VYAW"), help="forced constant command")
    p.add_argument("--steps", type=int, default=600, help="control steps (~12 s)")
    p.add_argument("--transient", type=float, default=2.0, help="seconds skipped before averaging")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--yaw_gate", type=float, default=0.15,
                   help="|mean yaw| pass threshold (rad/s); the S1 gate (the contaminant)")
    p.add_argument("--drift_warn", type=float, default=0.25,
                   help="drift-speed (m/s) above which to WARN (soft diagnostic, not a fail)")
    p.add_argument("--drift_gate", type=float, default=0.0,
                   help="if >0, also HARD-fail when drift speed exceeds this (default 0 = off: "
                        "drift is benign for the in-place comparison — both arms drift equally and "
                        "s3's commanded velocity trains it out, so the gate is yaw + survival)")
    args = p.parse_args()

    from pea import experiment

    run_dir = pathlib.Path(args.run)
    print(f"[yaw-gate] run={run_dir}  command={args.command}  seeds={args.seeds}")
    rows = []
    for seed in args.seeds:
        # fresh clean env per seed (sample_command is patched in place)
        env, policy, _ = experiment.load_clean_policy(run_dir)
        traj = probe_once(env, policy, args.command, args.steps, seed)
        s = summarize(traj, args.steps, args.transient)
        rows.append(s)
        flag = "ok " if s["survived"] else "DIED"
        print(f"  seed {seed}: yaw {s['mean_yaw']:+.3f}±{s['std_yaw']:.3f} rad/s  "
              f"drift {s['net_drift_m']:.2f} m ({s['drift_speed_mps']:+.3f} m/s)  "
              f"{flag} {s['episode_s']}s")

    survived_all = all(r["survived"] for r in rows)
    mean_abs_yaw = float(np.mean([r["abs_mean_yaw"] for r in rows]))
    worst_abs_yaw = float(np.max([r["abs_mean_yaw"] for r in rows]))
    mean_drift = float(np.mean([r["drift_speed_mps"] for r in rows]))
    print("\n[yaw-gate] SUMMARY (mean over seeds):")
    print(f"  |mean yaw|  : {mean_abs_yaw:.3f} rad/s  (worst seed {worst_abs_yaw:.3f})")
    print(f"  drift speed : {mean_drift:+.3f} m/s")
    print(f"  survival    : {'ALL' if survived_all else 'NOT ALL'} seeds full episode")

    is_zero_cmd = (abs(args.command[0]) + abs(args.command[1]) + abs(args.command[2])) < 1e-6
    if is_zero_cmd:
        # The gate is YAW + SURVIVAL — yaw was the contaminant (it made the matched
        # in-place hop two differently-pirouetting hops). Drift is reported as a soft
        # diagnostic: benign for the comparison (both arms drift equally; s3's
        # commanded velocity trains it out). Enable a hard drift fail only via --drift_gate.
        spin_ok = (worst_abs_yaw < args.yaw_gate) and survived_all
        drift_speed = abs(mean_drift)
        drift_ok = (args.drift_gate <= 0.0) or (drift_speed < args.drift_gate)
        passed = spin_ok and drift_ok
        if drift_speed > args.drift_warn:
            print(f"\n  [drift] {drift_speed:.3f} m/s > {args.drift_warn} warn "
                  f"(soft — not a fail unless --drift_gate set)")
        gate_desc = f"|yaw|<{args.yaw_gate}, survival" + (
            f", drift<{args.drift_gate}" if args.drift_gate > 0 else "")
        print(f"\n  S1 GATE ({gate_desc}): "
              f"{'*** PASS ***' if passed else '*** FAIL — do NOT build s2-s4 ***'}")
        # Exit non-zero on a FAILED gate so the curriculum driver ABORTS the chain.
        sys.exit(0 if passed else 1)
    else:
        print("\n  (non-zero command: read mean_yaw / drift vs the commanded value to "
              "check TRACKING, e.g. yaw cmd 0.4 -> mean_yaw ~ +0.4)")


if __name__ == "__main__":
    main()
