"""Diagnose whether a G1 bound/run policy ALTERNATES legs (running) or moves them
together (two-footed jump), and whether one leg is stuck. Rolls out a surviving seed
at a forward command and reports, per leg, the hip-pitch / knee range of motion and the
LEFT-vs-RIGHT phase relationship (correlation: +1 = in-phase = jump; -1 = antiphase = run).

Usage: gait_check.py <run_dir> [cmd vx,vy,vyaw] [seed] [steps]
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from pea import config as cfg_lib, experiment, policy as policy_lib
from pea.env import make_env, joints_by_substring

run = pathlib.Path(sys.argv[1])
CMD = tuple(float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("0.7", "0", "0")))
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 600

cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
mj = env.mj_model

def adr(sub):
    j = joints_by_substring(mj, sub)
    (name, info), = j.items()
    return name, info["qpos_adr"]

legs = {}
for side in ("left", "right"):
    for j in ("hip_pitch", "knee", "ankle_pitch"):
        try:
            legs[f"{side}_{j}"] = adr(f"{side}_{j}")[1]
        except Exception:
            pass

tr = experiment.rollout(env, pol, CMD, STEPS, seed=SEED)
n = tr["n"]; qpos = tr["qpos"][:n]; qvel = tr["qvel"][:n]
dt = float(env.dt)
print(f"GAIT CHECK {run.name}  cmd={CMD} seed={SEED}  n={n} ({n*dt:.1f}s)  survived={n>=STEPS}")
print(f"  base forward speed vx_mean = {np.mean(qvel[:,0]):+.3f} m/s   drift_x = {qpos[-1,0]-qpos[0,0]:+.2f} m")
print(f"  {'joint':18s} {'range(deg)':>11s} {'mean(deg)':>10s}")
series = {}
for name, a in legs.items():
    ang = np.degrees(qpos[:, a])
    series[name] = ang
    print(f"  {name:18s} {ang.max()-ang.min():11.1f} {ang.mean():10.1f}")

def phase(a, b):
    a = a - a.mean(); b = b - b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

print("\n  LEFT-vs-RIGHT correlation (+1 in-phase=JUMP, -1 antiphase=RUN):")
for j in ("hip_pitch", "knee", "ankle_pitch"):
    l, r = f"left_{j}", f"right_{j}"
    if l in series and r in series:
        print(f"    {j:12s} corr = {phase(series[l], series[r]):+.2f}   "
              f"L range {series[l].max()-series[l].min():5.1f}deg  R range {series[r].max()-series[r].min():5.1f}deg")


# --- STAGE-1 GATE VERDICT (acceptance criteria for "is it actually running?") ---
def _rng(name):
    return (series[name].max() - series[name].min()) if name in series else float("nan")

knee_corr = (phase(series["left_knee"], series["right_knee"])
             if ("left_knee" in series and "right_knee" in series) else float("nan"))
hip_gap = abs(_rng("left_hip_pitch") - _rng("right_hip_pitch"))
vx = float(np.mean(qvel[:, 0]))
# knee peak/RMS angular velocity (rad/s) vs the ~20 rad/s actuator ceiling (NR-7: knee is SPEED-limited)
knee_dof = [v["dof_adr"] for v in joints_by_substring(mj, "knee").values()]
knee_w = np.abs(qvel[:, knee_dof]) if knee_dof else np.zeros((max(n, 1), 1))
print("\n  GATE VERDICT (Stage 1 — a real run must PASS the first four):")
print(f"    knee antiphase : corr {knee_corr:+.2f}  -> {'PASS (alternating)' if knee_corr < -0.2 else 'FAIL (in-phase = jump)'}")
print(f"    hip symmetry   : swing gap {hip_gap:4.1f} deg -> {'PASS' if hip_gap < 10 else 'FAIL (asymmetric, a leg under-driven)'}")
print(f"    forward speed  : vx {vx:+.2f} m/s -> {'PASS' if abs(vx) > 0.4 else 'FAIL (<0.4, not matched-speed)'}  (sign must match cmd)")
print(f"    survival       : {'PASS' if n >= STEPS else 'FAIL (fell)'}")
print(f"    knee speed DIAG: peak {float(knee_w.max()):4.1f} rad/s  RMS {float(np.sqrt((knee_w**2).mean())):4.1f}  "
      f"(NR-7 actuator ceiling ~20 rad/s; report headroom, do NOT gate)")
