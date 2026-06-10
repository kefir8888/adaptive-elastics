"""Roll out a trained policy locally (CPU MJX) and log the trajectory.

Uses the same MJX env as training — identical physics, no re-implementation.
A single-env 10-20 s rollout is fine on CPU JAX. Writes trajectory.npz with
everything the analysis needs (knee angle/torque included); optionally renders
a video and/or replays interactively in the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder (synced from Drive)")
    p.add_argument("--steps", type=int, default=600, help="control steps (~12 s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("VX", "VY", "YAW_RATE"),
        help="fixed joystick command for a straight walk (m/s, m/s, rad/s)",
    )
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--video", action="store_true", help="write video.mp4 to run dir")
    p.add_argument("--viewer", action="store_true", help="interactive replay")
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco

    from pea import config as cfg_lib
    from pea import policy as policy_lib
    from pea.env import knee_joints, make_env

    run_dir = pathlib.Path(args.run)
    cfg = cfg_lib.load_config(run_dir / "config.yaml")
    env = make_env(cfg)
    policy = policy_lib.load_policy(
        env, cfg, run_dir / "policy_params",
        deterministic=not args.stochastic,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    command = jnp.array(args.command, dtype=jnp.float32)

    qpos, qvel, ctrl, qfrc, reward = [], [], [], [], []
    print(f"[pea-rollout] {args.steps} steps, command {args.command} (JIT first)…")
    t0 = time.monotonic()
    for i in range(args.steps):
        # Pin the joystick command so the env's resampling can't change it.
        if "command" in state.info:
            state.info["command"] = command
        rng, act_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)
        qpos.append(np.asarray(state.data.qpos))
        qvel.append(np.asarray(state.data.qvel))
        ctrl.append(np.asarray(state.data.ctrl))
        qfrc.append(np.asarray(state.data.qfrc_actuator))
        reward.append(float(state.reward))
        if bool(state.done):
            print(f"[pea-rollout] episode terminated (fall?) at step {i + 1}")
            break
    n = len(qpos)
    print(f"[pea-rollout] {n} steps in {time.monotonic() - t0:.1f} s wall")

    mj_model = env.mj_model
    knees = knee_joints(mj_model)
    out = run_dir / "trajectory.npz"
    np.savez_compressed(
        out,
        time=np.arange(n) * env.dt,
        qpos=np.stack(qpos),
        qvel=np.stack(qvel),
        ctrl=np.stack(ctrl),
        qfrc_actuator=np.stack(qfrc),
        reward=np.array(reward),
        command=np.array(args.command),
        dt=env.dt,
        total_mass=float(mj_model.body_mass.sum()),
        knee_names=np.array(list(knees)),
        knee_qpos_adr=np.array([v["qpos_adr"] for v in knees.values()]),
        knee_dof_adr=np.array([v["dof_adr"] for v in knees.values()]),
    )
    dist = qpos[-1][0] - qpos[0][0]
    print(f"[pea-rollout] walked {dist:.2f} m in {n * env.dt:.1f} s; saved {out}")

    if args.video:
        render_video(mj_model, qpos, env.dt, run_dir / "video.mp4")
    if args.viewer:
        replay_viewer(mj_model, qpos, env.dt)


def _camera(mj_model) -> str | int:
    import mujoco

    cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    return "track" if cam_id >= 0 else -1


def render_video(mj_model, qpos_hist, dt: float, path: pathlib.Path) -> None:
    import mediapy
    import mujoco

    data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    cam = _camera(mj_model)
    frames = []
    for qpos in qpos_hist:
        data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    renderer.close()
    mediapy.write_video(str(path), frames, fps=1.0 / dt)
    print(f"[pea-rollout] video saved to {path}")


def replay_viewer(mj_model, qpos_hist, dt: float) -> None:
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos_hist[0]
    mujoco.mj_forward(mj_model, data)
    print("[pea-rollout] viewer replay — close the window to exit")
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        while viewer.is_running():
            for qpos in qpos_hist:
                if not viewer.is_running():
                    break
                step_start = time.monotonic()
                data.qpos[:] = qpos
                mujoco.mj_forward(mj_model, data)
                viewer.sync()
                leftover = dt - (time.monotonic() - step_start)
                if leftover > 0:
                    time.sleep(leftover)


if __name__ == "__main__":
    main()
