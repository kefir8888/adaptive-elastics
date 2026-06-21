"""Torque-vs-angle work loops for ALL 6 G1 leg joints during in-place hopping,
with CONSTANT and LINEAR passive-spring fits per joint and a feasibility call.

For each leg joint (pooled left+right) it plots motor torque (qfrc_actuator) vs
angle (qpos) over a steady in-place hop, then fits:
  * CONSTANT spring   tau0 = mean(tau)            -- offloads the DC/offset torque
  * LINEAR  spring    tau = -k*(theta - theta0)   -- least-squares line through the loop
and recommends which is FEASIBLE as a *passive* element:
  - angle range < 0.08 rad      -> CONSTANT  (joint barely moves; a stiffness is ill-defined)
  - else if linear R^2 > 0.5 and k > 0 (restoring) -> LINEAR
  - else                        -> CONSTANT  (weak angle-torque slope, or non-restoring)
A passive linear spring needs k>0 (restoring); a positive slope would need negative
stiffness, so it falls back to a constant preload.

Usage: hop_workloop_6joint.py <run_dir> [steps] [out_plot.png]
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

from pea import experiment
from pea.env import joints_by_substring

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = sys.argv[3] if len(sys.argv) > 3 else "outputs/clean_curriculum/hop_workloop_6joint.png"
TRANSIENT_S = 2.0
RANGE_MIN = 0.08   # rad: below this a joint barely moves -> stiffness ill-defined
R2_MIN = 0.5       # linear feasible only if it explains this much variance

JOINTS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]

env, pol, mj = experiment.load_clean_policy(run)
dt = float(env.dt)
qpos, qvel, qfrc = [], [], []
for seed in (0, 1, 2):
    tr = experiment.rollout(env, pol, (0.0, 0.0, 0.0), STEPS, seed=seed)
    if tr["n"] > 0:
        qpos.append(tr["qpos"]); qvel.append(tr["qvel"]); qfrc.append(tr["qfrc"])
qpos = np.concatenate(qpos); qvel = np.concatenate(qvel); qfrc = np.concatenate(qfrc)
skip = int(TRANSIENT_S / dt)
qpos, qvel, qfrc = qpos[skip:], qvel[skip:], qfrc[skip:]
T = len(qpos)
print(f"rollout {T} steps ({T*dt:.1f} s) after {TRANSIENT_S}s transient — {run.name}")

rows, data = [], {}
for jsub in JOINTS:
    try:
        joints = joints_by_substring(mj, jsub)
    except RuntimeError:
        print(f"[{jsub}] not found, skipping"); continue
    th = np.concatenate([qpos[:, j["qpos_adr"]] for j in joints.values()])
    ta = np.concatenate([qfrc[:, j["dof_adr"]] for j in joints.values()])
    om = np.concatenate([qvel[:, j["dof_adr"]] for j in joints.values()])
    rng = float(th.max() - th.min())
    mean_tau = float(np.mean(ta)); rms = float(np.sqrt(np.mean(ta ** 2)))
    ss_tot = float(np.sum((ta - mean_tau) ** 2)) + 1e-12
    a, b = np.polyfit(th, ta, 1)                 # tau ~ a*theta + b
    r2 = 1.0 - float(np.sum((ta - (a * th + b)) ** 2)) / ss_tot
    k = -float(a); theta0 = (-b / a) if abs(a) > 1e-9 else 0.0   # tau = -k(theta-theta0)
    brake = float(np.mean(np.minimum(ta * om, 0.0)))  # mean braking (recoverable) power, W
    if rng < RANGE_MIN:
        rec, why = "constant", f"range {rng:.3f} rad < {RANGE_MIN} (barely moves)"
    elif r2 > R2_MIN and k > 0:
        rec, why = "linear", f"R^2={r2:.2f}, k={k:.0f}>0 restoring"
    else:
        rec, why = "constant", (f"linear R^2={r2:.2f} weak" if k > 0 else f"k={k:.0f}<0 non-restoring")
    rows.append((jsub, rng, mean_tau, rms, r2, k, theta0, brake, rec, why))
    data[jsub] = dict(th=th, ta=ta, a=a, b=b, k=k, theta0=theta0, mean=mean_tau, r2=r2, rec=rec)

print("\n=== per-joint torque-vs-angle fits (pooled L+R, in-place hop) ===")
hdr = f"{'joint':12}{'θrange':>8}{'mean τ':>8}{'RMS τ':>7}{'linR²':>7}{'k':>8}{'θ0':>7}{'brakeW':>8}  recommend"
print(hdr); print("-" * len(hdr))
for (j, rng, mt, rms, r2, k, t0, br, rec, why) in rows:
    print(f"{j:12}{rng:8.3f}{mt:8.1f}{rms:7.1f}{r2:7.2f}{k:8.1f}{t0:7.2f}{br:8.2f}  {rec.upper():8} ({why})")
print("\nLEGEND: τ=motor torque (N·m), θ=joint angle (rad), k=N·m/rad, brakeW=mean negative "
      "(recoverable) joint power (W). CONSTANT→preload τ0; LINEAR→spring k about θ0.")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = len(rows)
fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
for ax, (j, rng, mt, rms, r2, k, t0, br, rec, why) in zip(axes, rows):
    d = data[j]
    ax.scatter(d["th"], d["ta"], s=2, alpha=0.25, color="0.6")
    tg = np.linspace(d["th"].min(), d["th"].max(), 100)
    ax.axhline(d["mean"], color="tab:blue", ls="--", lw=1.6, label=f"const τ0={d['mean']:.0f}")
    lin_c = "tab:red" if rec == "linear" else "tab:orange"
    ax.plot(tg, d["a"] * tg + d["b"], color=lin_c, lw=1.8,
            label=f"linear k={d['k']:.0f} R²={d['r2']:.2f}")
    ax.set_title(f"{j}   →  {rec.upper()}", fontsize=10, weight="bold")
    ax.set_xlabel("θ [rad]"); ax.set_ylabel("τ [N·m]"); ax.legend(fontsize=7, loc="best")
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle(f"G1 in-place hop — motor torque vs joint angle, constant & linear spring fits "
             f"({run.name})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print(f"\nsaved {OUT}")
