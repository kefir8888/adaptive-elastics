"""Energy-per-hop comparison UNDER DOMAIN RANDOMIZATION — the condition the policies
were trained AND evaluated in (the spring arm is nominally fragile; with DR it is the
122-reward gait from the box). Both arms are rolled out on the SAME set of DR samples
(identical friction/mass/armature/pose perturbations per index), so the per-sample
delta is fair; we report over the samples where BOTH arms survive.

Usage: hop_energy_compare_dr.py <baseline_dir> <spring_dir> [N_samples] [steps]
"""
from __future__ import annotations
import os, sys, pathlib
import numpy as np
import jax
from scipy.signal import find_peaks
from pea import config as cfg_lib, energy, experiment, policy as policy_lib
from pea.env import make_env
from mujoco_playground import registry

N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 700
TRANSIENT_S = 2.0
DET = os.environ.get("PEA_DET", "1") == "1"
KT, R = energy.G1_KNEE.kt, energy.G1_KNEE.r

def base_of(env):
    b = env
    while hasattr(b, "_env"):
        b = b._env
    return b

def measure_dr(run_dir, rngs):
    run = pathlib.Path(run_dir); cfg = cfg_lib.load_config(run / "config.yaml")
    env = make_env(cfg)
    pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=DET)
    base = base_of(env); m0 = base._mjx_model
    batched, _ = registry.get_domain_randomizer(cfg.env_name)(m0, rngs)
    dt = float(env.dt); H = float(base._init_q[2])
    out = []
    for i in range(len(rngs)):
        base._mjx_model = m0.tree_replace({k: getattr(batched, k)[i] for k in
            ("pair_friction", "dof_frictionloss", "dof_armature", "body_mass", "qpos0")})
        tr = experiment.rollout(env, pol, (0., 0., 0.), STEPS, seed=i)
        n = tr["n"]
        if n <= int(TRANSIENT_S / dt) + 20:
            out.append(None); continue
        s = int(TRANSIENT_S / dt)
        qpos, qvel, qfrc = tr["qpos"][s:], tr["qvel"][s:], tr["qfrc"][s:]
        T = len(qpos); act = slice(6, None)
        tau, om = qfrc[:, act], qvel[:, act]
        p = tau * om + (tau / KT) ** 2 * R
        Enr = float(np.maximum(p, 0).sum(1).sum() * dt)
        Ere = float(max(p.sum(1).sum(), 0.0) * dt)
        Eoh = float(((tau / KT) ** 2 * R).sum(1).sum() * dt)
        bh = qpos[:, 2]; peaks, _ = find_peaks(bh, distance=int(0.25 / dt), prominence=0.01)
        nh = max(len(peaks), 1); pos = (bh[peaks] - H)[bh[peaks] - H > 0] if len(peaks) else np.array([])
        out.append(dict(Enr=Enr / nh, Ere=Ere / nh, ohmic=Eoh / max(Enr, 1e-9),
                        apex=float(np.mean(pos)) if len(pos) else float("nan"),
                        cad=nh / (T * dt), surv=n * dt))
    base._mjx_model = m0
    return out

rngs = jax.random.split(jax.random.PRNGKey(0), N)
print(f"rolling out {N} DR samples x {STEPS} steps per arm...", flush=True)
b = measure_dr(sys.argv[1], rngs); s = measure_dr(sys.argv[2], rngs)
print(f"\nsurvival: baseline {sum(x is not None for x in b)}/{N}   spring {sum(x is not None for x in s)}/{N}")
both = [i for i in range(N) if b[i] and s[i]]
print(f"samples where BOTH survived: {len(both)}/{N}  (indices {both})")
if not both:
    print("*** no shared survivors — cannot compare ***"); sys.exit(0)
def mean(res, idx, k): return float(np.mean([res[i][k] for i in idx]))
print("\n=== MATCHED-TASK HOP ENERGY UNDER DR (paired DR samples; estimated Kt/R) ===")
hdr = "metric                       baseline      spring     delta"; print(hdr); print("-" * len(hdr))
def row(lbl, bv, sv, pct=True, u=""):
    d = (100 * (sv / bv - 1)) if (pct and bv) else (sv - bv)
    print(f"{lbl:26s} {bv:10.3f}{u} {sv:10.3f}{u} {d:+7.1f}{'%' if pct else ''}")
row("apex (m above standing)", mean(b, both, "apex"), mean(s, both, "apex"), pct=False)
row("cadence (Hz)", mean(b, both, "cad"), mean(s, both, "cad"), pct=False)
row("E/hop no-regen (J)", mean(b, both, "Enr"), mean(s, both, "Enr"))
row("E/hop regen (J)", mean(b, both, "Ere"), mean(s, both, "Ere"))
print(f"\nohmic share: baseline {mean(b,both,'ohmic')*100:.1f}%  spring {mean(s,both,'ohmic')*100:.1f}%")
da = abs(mean(s, both, "apex") - mean(b, both, "apex")); dc = abs(mean(s, both, "cad") - mean(b, both, "cad"))
print(f"\nFAIRNESS GATE: apex Δ={da*100:.1f} cm (tol 0.5), cadence Δ={dc:.2f} Hz -> "
      f"{'FAIR' if da<=0.005 and dc<=0.15 else 'CONFOUNDED (apex/cadence differ)'}")
