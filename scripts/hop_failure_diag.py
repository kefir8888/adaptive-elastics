"""Diagnose WHY a hop policy falls on the nominal model: deterministic vs stochastic
survival, achieved apex, and the dominant failure signal at the end of the episode
(body tilt vs yaw-spin vs horizontal drift). Free local triage before spending box time.

Usage: hop_failure_diag.py <run_dir> [steps] [n_seeds]
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from pea import config as cfg_lib, experiment, policy as policy_lib
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# Optional command "vx,vy,vyaw" (default in-place 0,0,0). For a FORWARD gait pass e.g. "0.5,0,0";
# rollout pins this command, so "drift" then reads as forward PROGRESS and vx_mean ~ the command.
CMD = tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0.0, 0.0, 0.0)

cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
dt = float(env.dt)
b = env
while hasattr(b, "_env"):
    b = b._env
H = float(b._init_q[2])


def tilt_deg(quat):
    # angle of body z-axis from world up, from wxyz quaternion
    w, x, y, z = quat
    # world-z component of body-z axis
    cz = 1.0 - 2.0 * (x * x + y * y)
    cz = max(-1.0, min(1.0, cz))
    return np.degrees(np.arccos(cz))


def diag(det: bool):
    pol = (policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=det)
           if (run / "policy_params").exists()
           else policy_lib.load_policy_from_checkpoint(env, cfg, policy_lib.latest_checkpoint(run), deterministic=det))
    print(f"  [{'deterministic' if det else 'stochastic'}]")
    for seed in range(NSEED):
        tr = experiment.rollout(env, pol, CMD, STEPS, seed=seed)
        n = tr["n"]
        qpos = tr["qpos"]; qvel = tr["qvel"]
        apex = float(np.max(qpos[:, 2]) - H) if n > 10 else float("nan")
        # state near the end (last recorded frame)
        last = min(n - 1, len(qpos) - 1)
        til = tilt_deg(qpos[last, 3:7])
        yaw_rate = float(qvel[last, 5])           # body yaw rate (free-joint angular vel z)
        drift = float(np.hypot(qpos[last, 0] - qpos[0, 0], qpos[last, 1] - qpos[0, 1]))
        vx_mean = float(np.mean(qvel[:n, 0])) if n > 0 else float("nan")   # base forward speed (m/s)
        # peak tilt over the run (catches the topple onset)
        tilts = np.array([tilt_deg(q) for q in qpos[:n, 3:7]])
        peak_til = float(np.max(tilts)) if n > 0 else float("nan")
        mean_yaw = float(np.mean(np.abs(qvel[:n, 5]))) if n > 0 else float("nan")
        verdict = "SURVIVED" if n >= STEPS else "FELL"
        print(f"    seed {seed}: {n:4d} steps ({n*dt:5.1f}s) {verdict}  apex {apex:+.3f}  vx {vx_mean:+.2f}m/s  "
              f"endtilt {til:5.1f}deg peaktilt {peak_til:5.1f}deg  endyaw {yaw_rate:+.2f} meanyaw {mean_yaw:.2f}  drift {drift:.2f}m")


print(f"FAILURE DIAG {run.name} (spring={cfg.spring.kind} k={cfg.spring.k}) cmd={CMD}, H={H:.3f}")
diag(det=True)
diag(det=False)
