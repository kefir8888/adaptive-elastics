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
    pos_apex = (bh[peaks] - H)[bh[peaks] - H > 0] if len(peaks) else np.array([])
    apex = float(np.mean(pos_apex)) if len(pos_apex) else float("nan")
    apex_std = float(np.std(pos_apex)) if len(pos_apex) else float("nan")
    cadence = n_hops / dur
    e_nr = E_noregen / n_hops
    # apex-normalized energy (J per m of apex): divides out a residual apex gap that survives the
    # fairness tolerance, so a small height difference cannot masquerade as an energy delta.
    e_nr_per_m = e_nr / apex if (apex == apex and apex > 1e-3) else float("nan")
    return dict(E_noregen_per_hop=e_nr, E_regen_per_hop=E_regen / n_hops,
                ohmic_share=E_ohmic / max(E_noregen, 1e-9), apex=apex, apex_std=apex_std,
                n_pos_peaks=int(len(pos_apex)), E_noregen_per_m=e_nr_per_m,
                cadence=cadence, n_hops=n_hops, mean_power_noregen=E_noregen / dur, dur=dur)


base = measure(sys.argv[1]); spr = measure(sys.argv[2])
print("\n=== MATCHED-TASK HOP ENERGY COMPARISON (estimated Kt/R; lead with the %) ===")
hdr = "metric                         baseline      spring     delta"
print(hdr); print("-" * len(hdr))
def row(label, b, s, pct=True, unit=""):
    d = (100 * (s / b - 1)) if (pct and b) else (s - b)
    print(f"{label:28s} {b:10.3f}{unit}  {s:9.3f}{unit}  {d:+7.1f}{'%' if pct else ''}")
row("apex (m above standing)", base["apex"], spr["apex"], pct=False)
row("apex std (m, within-arm)", base["apex_std"], spr["apex_std"], pct=False)
row("cadence (Hz)", base["cadence"], spr["cadence"], pct=False)
row("E/hop no-regen (J)", base["E_noregen_per_hop"], spr["E_noregen_per_hop"])
row("E/hop regen (J)", base["E_regen_per_hop"], spr["E_regen_per_hop"])
row("E/hop per m apex (J/m)", base["E_noregen_per_m"], spr["E_noregen_per_m"])
row("mean power no-regen (W)", base["mean_power_noregen"], spr["mean_power_noregen"])
print(f"\nohmic share: baseline {base['ohmic_share']*100:.1f}%  spring {spr['ohmic_share']*100:.1f}%")
print(f"positive-apex peaks: baseline {base['n_pos_peaks']}  spring {spr['n_pos_peaks']}")

# Degenerate-policy guard: a collapsed-to-standing arm has no positive-apex hop -> apex is NaN and the
# gate would silently read CONFOUNDED. Flag it explicitly as a FAILED elicitation, not a fair comparison.
if not (base["n_pos_peaks"] > 0 and spr["n_pos_peaks"] > 0):
    print("\n*** DEGENERATE: an arm produced no positive-apex hops (collapsed policy?) — "
          "failed elicitation, NOT a valid energy comparison. ***")

# Explicit fairness gate: the apex MUST match (cadence is physics-fixed by hop_freq).
# TIGHTENED 0.015 -> 0.005 m: the old 1.5 cm tolerance was ~17% of the 0.09 m apex — large enough that a
# residual height gap could masquerade as an energy delta (the 2026-06-20 confound). 5 mm is ~5% of apex;
# the J/m row above is the secondary, apex-normalized cross-check.
APEX_TOL = 0.005  # metres
d_apex = abs(spr["apex"] - base["apex"])
d_cad = abs(spr["cadence"] - base["cadence"])
fair = d_apex <= APEX_TOL and d_cad <= 0.15
print(f"\nFAIRNESS GATE: apex match Δ={d_apex*100:.1f} cm (tol {APEX_TOL*100:.1f}), "
      f"cadence Δ={d_cad:.2f} Hz -> {'FAIR — energy delta is valid' if fair else 'CONFOUNDED — apex/cadence differ, energy delta NOT valid (use the J/m row or re-train with a tighter apex cap)'}")
