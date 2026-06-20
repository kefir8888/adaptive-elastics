"""Compare electrical energy-per-hop of two trained G1 hop policies at the matched
fixed task (cadence + apex fixed by the env). The SPRING arm is rolled out WITH its
spring active (make_env from its own config), and energy is computed from MOTOR
torque (qfrc_actuator), which excludes the passive spring -> honest offload.

Reports, per run: achieved apex + cadence (to confirm the task is matched),
electrical energy per hop (no-regen AND regen), and ohmic share. Then the
spring-vs-baseline % reduction.

Usage: hop_energy_compare.py <baseline_run_dir> <spring_run_dir> [steps]
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

from pea import config as cfg_lib, energy, experiment, policy as policy_lib
from pea.env import make_env

STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
TRANSIENT_S = 2.0


def measure(run_dir: str):
    run = pathlib.Path(run_dir)
    cfg = cfg_lib.load_config(run / "config.yaml")
    env = make_env(cfg)                      # spring ACTIVE if cfg has one
    if (run / "policy_params").exists():
        pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
    else:
        pol = policy_lib.load_policy_from_checkpoint(
            env, cfg, policy_lib.latest_checkpoint(run), deterministic=True)
    dt = float(env.dt)
    qpos, qvel, qfrc = [], [], []
    for seed in (0, 1, 2):
        tr = experiment.rollout(env, pol, (0.0, 0.0, 0.0), STEPS, seed=seed)
        if tr["n"] > int(TRANSIENT_S / dt) + 20:
            s = int(TRANSIENT_S / dt)
            qpos.append(tr["qpos"][s:]); qvel.append(tr["qvel"][s:]); qfrc.append(tr["qfrc"][s:])
    qpos = np.concatenate(qpos); qvel = np.concatenate(qvel); qfrc = np.concatenate(qfrc)
    T = len(qpos)
    # actuated DoFs = past the 6 free-base DoFs
    act = slice(6, None)
    tau = qfrc[:, act]; om = qvel[:, act]
    kt, r = energy.G1_KNEE.kt, energy.G1_KNEE.r
    p_dof = tau * om + (tau / kt) ** 2 * r           # per-DoF electrical power
    p_noregen = np.maximum(p_dof, 0.0).sum(axis=1)   # no-regen: clamp each DoF >=0
    p_regen = p_dof.sum(axis=1)                      # regen: allow negative net
    ohmic = ((tau / kt) ** 2 * r).sum(axis=1)
    dur = T * dt
    E_noregen = float(p_noregen.sum() * dt)
    E_regen = float(max(p_regen.sum(), 0.0) * dt)
    E_ohmic = float(ohmic.sum() * dt)
    # hop count from base-height peaks
    H = float(env.base_env._init_q[2] if hasattr(env, "base_env") else env._init_q[2])
    bh = qpos[:, 2]
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(bh, distance=int(0.25 / dt), prominence=0.01)
    except Exception:
        peaks = np.array([i for i in range(1, T-1) if bh[i] > bh[i-1] and bh[i] >= bh[i+1]])
    n_hops = max(len(peaks), 1)
    apex = float(np.mean((bh[peaks] - H)[bh[peaks] - H > 0])) if len(peaks) else float("nan")
    cadence = n_hops / dur
    return dict(E_noregen_per_hop=E_noregen / n_hops, E_regen_per_hop=E_regen / n_hops,
                ohmic_share=E_ohmic / max(E_noregen, 1e-9), apex=apex, cadence=cadence,
                n_hops=n_hops, mean_power_noregen=E_noregen / dur, dur=dur)


base = measure(sys.argv[1]); spr = measure(sys.argv[2])
print("\n=== MATCHED-TASK HOP ENERGY COMPARISON (estimated Kt/R; lead with the %) ===")
hdr = "metric                         baseline      spring     delta"
print(hdr); print("-" * len(hdr))
def row(label, b, s, pct=True, unit=""):
    d = (100 * (s / b - 1)) if (pct and b) else (s - b)
    print(f"{label:28s} {b:10.3f}{unit}  {s:9.3f}{unit}  {d:+7.1f}{'%' if pct else ''}")
row("apex (m above standing)", base["apex"], spr["apex"], pct=False)
row("cadence (Hz)", base["cadence"], spr["cadence"], pct=False)
row("E/hop no-regen (J)", base["E_noregen_per_hop"], spr["E_noregen_per_hop"])
row("E/hop regen (J)", base["E_regen_per_hop"], spr["E_regen_per_hop"])
row("mean power no-regen (W)", base["mean_power_noregen"], spr["mean_power_noregen"])
print(f"\nohmic share: baseline {base['ohmic_share']*100:.1f}%  spring {spr['ohmic_share']*100:.1f}%")
print("NOTE: apex+cadence must match for the energy delta to be valid; if they differ, "
      "the comparison is confounded (renormalize or retrain).")
