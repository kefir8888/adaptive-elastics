"""Render N walking rollouts of a Go1 policy at a fixed payload, concatenated back-to-back
into one mp4 (a short black gap between clips). Camera tracks the dog. Adaptive preload applied
if the run uses preload_dr.

Usage: render_walk.py <run_dir> <payload_kg> <n_clips> <steps_per_clip> <out.mp4>
"""
import sys
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from pea import config as cfg_lib, energy, metrics, policy as policy_lib
from pea.control import AdaptivePreloadController
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
PAY = float(sys.argv[2])
N = int(sys.argv[3])
STEPS = int(sys.argv[4])
OUT = sys.argv[5]

cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
go1 = env
while hasattr(go1, "_env"):
    go1 = go1._env
base = float(go1._mjx_model.body_mass[1])
go1._mjx_model = go1._mjx_model.replace(body_mass=go1._mjx_model.body_mass.at[1].set(base + PAY))
mj = env.mj_model
if cfg.terrain_height_scale != 1.0 and mj.nhfield > 0:
    mj.hfield_size[:, 2] *= cfg.terrain_height_scale  # match render terrain to scaled dynamics
_, _, da = metrics.joint_addrs(mj, "calf")
da = np.array(da)
dt = float(env.dt)
cmd = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
ADAPT = cfg.spring.kind == "preload_dr"
TMAX = float(cfg.spring.tau0) if ADAPT else 0.0
nleg = len(da)
jr, js, jp = jax.jit(env.reset), jax.jit(env.step), jax.jit(pol)

H, W = 480, 640
renderer = mujoco.Renderer(mj, H, W)
mjd = mujoco.MjData(mj)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 2.2, -18.0, 130.0
black = np.zeros((H, W, 3), np.uint8)
frames = []
print(f"# {run.name}  payload {PAY}kg  adaptive={ADAPT}", flush=True)

for vid in range(N):
    rng = jax.random.PRNGKey(vid)
    st = jr(rng)
    ctrl = AdaptivePreloadController(nleg, dt, TMAX) if ADAPT else None
    nf = 0
    for i in range(STEPS):
        if ADAPT:
            st.info["preload_tau0"] = jnp.asarray(ctrl.tau0)
        if "command" in st.info:
            st.info["command"] = cmd
        rng, ar = jax.random.split(rng)
        a, _ = jp(st.obs, ar)
        st = js(st, a)
        if ADAPT:
            ctrl.update(np.asarray(st.data.qfrc_actuator)[da])
        mjd.qpos[:] = np.asarray(st.data.qpos)
        mjd.qvel[:] = np.asarray(st.data.qvel)
        mujoco.mj_forward(mj, mjd)
        cam.lookat[:] = mjd.qpos[:3]
        renderer.update_scene(mjd, camera=cam)
        frames.append(renderer.render().copy())
        nf += 1
        if bool(st.done):
            break
    frames.extend([black] * 8)  # short separator between clips
    print(f"  clip {vid+1}/{N}: {nf} frames", flush=True)

import subprocess
fps = int(round(1 / dt))
proc = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-an",
     "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
    stdin=subprocess.PIPE,
)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
proc.stdin.close()
proc.wait()
print(f"WROTE {OUT}  {len(frames)} frames", flush=True)
