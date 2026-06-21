"""Render G1 hopping rollouts from a run's latest checkpoint (or final params).

Side view with a fixed camera height so the vertical bounce is visible (tracking
only x,y; z pinned). Command is zeroed (in-place hop). Reuses experiment.rollout +
load_clean_policy (which falls back to the latest checkpoint mid-training).

Usage: render_hop.py <run_dir> <steps_per_clip> <n_clips> <out.mp4> [WxH] [slowmo]
  WxH     : frame size, e.g. 1920x1080 (default 640x480)
  slowmo  : integer playback slowdown, e.g. 4 = 4x slow (default 1)
"""
import os
import sys
import pathlib
import subprocess

import numpy as np
import mujoco

from pea import config as cfg_lib, experiment, policy as policy_lib
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OUT = sys.argv[4] if len(sys.argv) > 4 else "hop.mp4"
W, H = (int(x) for x in (sys.argv[5].split("x") if len(sys.argv) > 5 else ("640", "480")))
SLOWMO = int(sys.argv[6]) if len(sys.argv) > 6 else 1
# Optional joystick command (vx,vy,vyaw) via env var, e.g. PEA_RENDER_CMD="0.3,0,0" to show a
# forward-moving stage. Default zero = in-place hop. Clips are < 500 steps, so the env's internal
# command re-sample (step>500) never fires and experiment.rollout's per-step command pin holds.
CMD = tuple(float(x) for x in os.environ.get("PEA_RENDER_CMD", "0,0,0").split(","))

# Load with the FULL config (spring ACTIVE if the run has one), so a spring policy
# rolls out with the spring it was trained with — not the spring-free clean env.
cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
mj = env.mj_model
if (run / "policy_params").exists():
    pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
else:
    pol = policy_lib.load_policy_from_checkpoint(
        env, cfg, policy_lib.latest_checkpoint(run), deterministic=True)
dt = float(env.dt)
renderer = mujoco.Renderer(mj, H, W)
mjd = mujoco.MjData(mj)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# Side view; fixed lookat height so the up/down hop is not cancelled by tracking.
cam.distance, cam.elevation, cam.azimuth = 3.4, -6.0, 90.0
CAM_Z = 0.75
black = np.zeros((H, W, 3), np.uint8)
frames = []


def _render(i, st):
    mjd.qpos[:] = np.asarray(st.data.qpos)
    mjd.qvel[:] = np.asarray(st.data.qvel)
    mujoco.mj_forward(mj, mjd)
    cam.lookat[0] = float(mjd.qpos[0])
    cam.lookat[1] = float(mjd.qpos[1])
    cam.lookat[2] = CAM_Z
    renderer.update_scene(mjd, camera=cam)
    frames.append(renderer.render().copy())


print(f"command = {CMD}", flush=True)
for vid in range(N):
    traj = experiment.rollout(env, pol, CMD, STEPS, seed=vid,
                              callback=_render, record_terminal=True)
    frames.extend([black] * 8)  # short separator between clips
    print(f"clip {vid + 1}/{N}: {traj['n']} steps", flush=True)

fps = max(1, int(round(1 / dt)) // SLOWMO)  # lower playback fps => slow motion
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
