"""Energy-vs-cadence sweep for a trained hop/bound policy. Tests whether the energy-optimal
cadence differs (e.g. a spring's RESONANT frequency) -- the fixed-cadence matched comparison
can handicap a spring forced off its efficient point. For each commanded cadence it overrides
reward_config.hop_freq, rolls out deterministically at CMD, and reports electrical energy
(no-regen, the spring's recovery regime), energy-per-metre (CoT proxy), achieved apex/speed and
survival. Run it on the no-spring AND spring arms and compare the curves' minima.

Usage: cadence_sweep.py <run_dir> "<vx,vy,vyaw>" "<f1,f2,...>" [steps]
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from pea import config as cfg_lib, energy, experiment, policy as policy_lib
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
CMD = tuple(float(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0.0, 0.0, 0.0)
FREQS = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 800
TRANSIENT_S = 2.0


def measure_at(freq: float):
    cfg = cfg_lib.load_config(run / "config.yaml")
    cfg.env_overrides["reward_config.hop_freq"] = freq      # FIX the gait clock at this cadence
    env = make_env(cfg)                                     # spring ACTIVE if cfg has one
    pol = (policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
           if (run / "policy_params").exists()
           else policy_lib.load_policy_from_checkpoint(env, cfg, policy_lib.latest_checkpoint(run), deterministic=True))
    dt = float(env.dt)
    qpos, qvel, qfrc, survived = [], [], [], 0
    for seed in (0, 1, 2):
        tr = experiment.rollout(env, pol, CMD, STEPS, seed=seed)
        survived += int(tr["n"] >= STEPS)
        if tr["n"] > int(TRANSIENT_S / dt) + 20:
            s = int(TRANSIENT_S / dt)
            qpos.append(tr["qpos"][s:]); qvel.append(tr["qvel"][s:]); qfrc.append(tr["qfrc"][s:])
    if not qpos:
        return dict(freq=freq, ok=0, e_nr=float("nan"), e_per_m=float("nan"),
                    apex=float("nan"), vx=float("nan"), survived=survived)
    qpos = np.concatenate(qpos); qvel = np.concatenate(qvel); qfrc = np.concatenate(qfrc)
    T = len(qpos); act = slice(6, None)
    tau = qfrc[:, act]; om = qvel[:, act]
    kt, r = energy.G1_KNEE.kt, energy.G1_KNEE.r
    p_dof = tau * om + (tau / kt) ** 2 * r
    p_noregen = np.maximum(p_dof, 0.0).sum(axis=1)
    dur = T * dt
    b = env
    while hasattr(b, "_env"):
        b = b._env
    H = float(b._init_q[2])
    bh = qpos[:, 2]
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(bh, distance=int(0.25 / dt), prominence=0.01)
    except Exception:
        peaks = np.array([i for i in range(1, T - 1) if bh[i] > bh[i - 1] and bh[i] >= bh[i + 1]])
    n_hops = max(len(peaks), 1)
    pos_apex = (bh[peaks] - H)[bh[peaks] - H > 0] if len(peaks) else np.array([])
    apex = float(np.mean(pos_apex)) if len(pos_apex) else float("nan")
    E_nr = float(p_noregen.sum() * dt)
    e_nr = E_nr / n_hops
    vx = float(np.mean(qvel[:, 0]))
    e_per_m = (E_nr / dur) / vx if abs(vx) > 1e-3 else float("nan")
    cad = n_hops / dur
    return dict(freq=freq, ok=1, e_nr=e_nr, e_per_m=e_per_m, apex=apex, vx=vx,
                cad=cad, survived=survived)


cfg0 = cfg_lib.load_config(run / "config.yaml")
print(f"\nCADENCE SWEEP {run.name} (spring={cfg0.spring.kind} k={cfg0.spring.k}) cmd={CMD}")
print(f"{'cmd_freq':>8} {'act_cad':>8} {'surv':>5} {'apex':>7} {'vx':>7} {'E/hop_nr(J)':>12} {'E/m(J/m)':>10}")
print("-" * 64)
for f in FREQS:
    m = measure_at(f)
    if not m["ok"]:
        print(f"{f:8.2f} {'--':>8} {m['survived']:>3}/3   (no steady window)")
        continue
    print(f"{f:8.2f} {m['cad']:8.2f} {m['survived']:>3}/3 {m['apex']:7.3f} {m['vx']:+7.2f} "
          f"{m['e_nr']:12.1f} {m['e_per_m']:10.1f}")
