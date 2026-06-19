"""Render G1 hopping rollouts from a run's latest checkpoint (or final params).

Side view with a fixed camera height so the vertical bounce is visible (tracking
only x,y; z pinned). Command is zeroed (in-place hop). Reuses experiment.rollout +
load_clean_policy (which falls back to the latest checkpoint mid-training).

Usage: render_hop.py <run_dir> <steps_per_clip> <n_clips> <out.mp4> [WxH] [slowmo]
  WxH     : frame size, e.g. 1920x1080 (default 640x480)
  slowmo  : integer playback slowdown, e.g. 4 = 4x slow (default 1)
"""
import sys
import pathlib
import subprocess

import numpy as np
import mujoco

from pea import experiment

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OUT = sys.argv[4] if len(sys.argv) > 4 else "hop.mp4"
W, H = (int(x) for x in (sys.argv[5].split("x") if len(sys.argv) > 5 else ("640", "480")))
SLOWMO = int(sys.argv[6]) if len(sys.argv) > 6 else 1

env, pol, mj = experiment.load_clean_policy(run)
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


for vid in range(N):
    traj = experiment.rollout(env, pol, (0.0, 0.0, 0.0), STEPS, seed=vid,
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
