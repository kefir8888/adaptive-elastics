"""Prepare the G1-hop spring comparison from a trained no-spring hopper.

Rolls out the policy in-place, then:
  1. CALIBRATES the matched task: mean hop apex (above standing) -> hop_height_target,
     and hop cadence (Hz) -> hop_freq, so the no-spring and spring arms hop the SAME.
  2. Builds the knee AND ankle_pitch torque-angle WORK-LOOP, reports negative
     (braking) work per joint -- the energy a pogo spring could recover.
  3. FITS a one-sided-linear (pogo) spring per joint: engage_sign + theta_engage
     (flexion onset of the loaded/stance branch) + k (least-squares stiffness over
     the engaged, loaded samples), and recommends the joint with the larger offload.

Prints config-ready values and saves a work-loop plot. Derives all sign
conventions FROM THE DATA (no hardcoded joint signs).

Usage: hop_spring_prep.py <run_dir> [steps] [out_plot.png]
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

from pea import experiment, energy, metrics
from pea.env import joints_by_substring

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = sys.argv[3] if len(sys.argv) > 3 else "outputs/hop_workloop.png"
TRANSIENT_S = 2.0

env, pol, mj = experiment.load_clean_policy(run)
dt = float(env.dt)
# Roll out in place (zero command). Concatenate a couple of seeds for more cycles.
qpos, qvel, qfrc = [], [], []
for seed in (0, 1):
    tr = experiment.rollout(env, pol, (0.0, 0.0, 0.0), STEPS, seed=seed)
    if tr["n"] > 0:
        qpos.append(tr["qpos"]); qvel.append(tr["qvel"]); qfrc.append(tr["qfrc"])
qpos = np.concatenate(qpos); qvel = np.concatenate(qvel); qfrc = np.concatenate(qfrc)
skip = int(TRANSIENT_S / dt)
qpos, qvel, qfrc = qpos[skip:], qvel[skip:], qfrc[skip:]
T = len(qpos)
print(f"rollout: {T} steps ({T*dt:.1f} s) after {TRANSIENT_S}s transient")

# --- 1. apex + cadence ------------------------------------------------------
H_STAND = float(env.base_env._init_q[2] if hasattr(env, "base_env") else env._init_q[2])
bh = qpos[:, 2]
try:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(bh, distance=int(0.25 / dt), prominence=0.01)
except Exception:
    peaks = np.array([i for i in range(1, T - 1) if bh[i] > bh[i-1] and bh[i] >= bh[i+1]])
apexes = bh[peaks] - H_STAND if len(peaks) else np.array([bh.max() - H_STAND])
mean_apex = float(np.mean(apexes[apexes > 0])) if np.any(apexes > 0) else float(apexes.mean())
cadence = len(peaks) / (T * dt) if len(peaks) else float("nan")
contact_z = 0.05  # rough foot-clearance proxy not available; use base-height oscillation
print(f"\n=== TASK CALIBRATION ===")
print(f"standing pelvis z      : {H_STAND:.3f} m")
print(f"mean hop apex (above)  : {mean_apex:.3f} m   (peaks={len(peaks)}, "
      f"range {apexes.min():.3f}..{apexes.max():.3f})")
print(f"hop cadence            : {cadence:.2f} Hz")
print(f"  -> reward_config.hop_height_target: {mean_apex:.2f}")
print(f"  -> reward_config.hop_freq         : {cadence:.1f}")

# --- 2 & 3. per-joint work-loop + pogo fit ----------------------------------
kt, r = energy.G1_KNEE.kt, energy.G1_KNEE.r
results = {}
for jsub in ("knee", "ankle_pitch"):
    try:
        joints = joints_by_substring(mj, jsub)
    except RuntimeError:
        print(f"\n[{jsub}] not found, skipping"); continue
    # average the two legs in the fit but report per joint
    legs = []
    for name, j in joints.items():
        theta = qpos[:, j["qpos_adr"]]
        tau = qfrc[:, j["dof_adr"]]
        omega = qvel[:, j["dof_adr"]]
        neg_work = float(np.sum(np.minimum(tau * omega, 0.0)) * dt) / (T * dt)  # W (mean braking)
        rms = float(np.sqrt(np.mean(tau**2)))
        legs.append((name, theta, tau, omega, neg_work, rms))
    # Pool both legs for a robust fit.
    th = np.concatenate([l[1] for l in legs])
    ta = np.concatenate([l[2] for l in legs])
    # Flexion = the theta direction associated with the deep-crouch (high extensor
    # torque). Loaded/stance samples: top 40% by |tau|. engage_sign points toward
    # the flexed (loaded) theta extreme.
    loaded = np.abs(ta) >= np.percentile(np.abs(ta), 60)
    th_load = th[loaded]; ta_load = ta[loaded]
    # extensor torque sign on the loaded set (the dominant sign there)
    tau_sign = np.sign(np.mean(ta_load)) or 1.0
    # flexion extreme = where extensor torque is largest -> theta there
    th_flexed = float(np.mean(th_load[np.argsort(-tau_sign * ta_load)[:max(1, len(ta_load)//5)]]))
    th_light = float(np.percentile(th_load, 15) if th_flexed > np.median(th_load)
                     else np.percentile(th_load, 85))
    engage_sign = 1.0 if th_flexed > th_light else -1.0
    theta_engage = th_light
    d = th_load - theta_engage
    eng = engage_sign * d > 0
    # tau = -k*d on engaged side -> k = -<tau*d>/<d^2> over engaged, loaded samples
    if np.any(eng) and np.var(d[eng]) > 1e-9:
        k = float(-np.sum(ta_load[eng] * d[eng]) / np.sum(d[eng]**2))
    else:
        k = 0.0
    k = max(k, 0.0)
    neg_work_total = sum(l[4] for l in legs)
    rms_mean = np.mean([l[5] for l in legs])
    results[jsub] = dict(theta_engage=theta_engage, engage_sign=engage_sign, k=k,
                         neg_work=neg_work_total, rms=rms_mean, legs=legs)
    print(f"\n=== {jsub} ===")
    print(f"  mean braking power (both legs): {neg_work_total:7.2f} W   RMS torque {rms_mean:5.1f} N*m")
    print(f"  POGO FIT -> theta_engage {theta_engage:+.3f} rad  engage_sign {engage_sign:+.0f}  k {k:6.1f} N*m/rad")

# recommend joint with the larger recoverable braking
if results:
    best = max(results, key=lambda j: abs(results[j]["neg_work"]))
    rb = results[best]
    print(f"\n=== RECOMMENDED SPRING JOINT: {best} ===")
    print(f"  spring.kind: one_sided_linear")
    print(f"  spring.joint: {best if best != 'ankle_pitch' else 'ankle_pitch'}")
    print(f"  spring.theta_engage: {rb['theta_engage']:.3f}")
    print(f"  spring.engage_sign: {rb['engage_sign']:.0f}")
    print(f"  spring.k: {rb['k']:.1f}")

# --- plot work loops --------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
joints_plot = [j for j in ("knee", "ankle_pitch") if j in results]
fig, axes = plt.subplots(len(joints_plot), 2, figsize=(11, 4 * len(joints_plot)), squeeze=False)
for i, jsub in enumerate(joints_plot):
    res = results[jsub]
    for name, theta, tau, omega, nw, rms in res["legs"]:
        axes[i, 0].plot(np.arange(len(theta)) * dt, theta, lw=0.6, label=name)
        axes[i, 1].plot(theta, tau, lw=0.5, alpha=0.7)
    te, es, k = res["theta_engage"], res["engage_sign"], res["k"]
    tg = np.linspace(min(res["legs"][0][1]), max(res["legs"][0][1]), 100)
    spr = -k * (tg - te) * ((es * (tg - te)) > 0)
    axes[i, 1].plot(tg, spr, "r--", lw=1.5, label=f"pogo k={k:.0f}")
    axes[i, 0].set(title=f"{jsub}: angle", xlabel="t [s]", ylabel="θ [rad]"); axes[i, 0].legend(fontsize=7)
    axes[i, 1].set(title=f"{jsub}: work loop + pogo fit", xlabel="θ [rad]", ylabel="τ [N·m]"); axes[i, 1].legend(fontsize=7)
fig.tight_layout()
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print(f"\nsaved {OUT}")
