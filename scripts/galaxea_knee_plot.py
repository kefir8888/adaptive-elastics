"""Single-panel plot: a SINGLE parallel spring on the Galaxea R1 knee (torso_joint2) --
the dominant torso joint in the coordinated upright lift. Shows required motor torque vs
joint angle, the gravity torque, the energy-optimal fitted linear spring, and the residual
motor torque after the spring (which hovers near zero), with the knee electrical-energy
reduction annotated.

  python scripts/galaxea_knee_plot.py [--config configs/galaxea_lift.yaml]
"""
import argparse
import importlib.util
import pathlib

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pea import energy as E, gravcomp
from pea.springs import tau_spring

KNEE = "torso_joint2"   # the middle torso joint -- Galaxea's "knee"


def _load_lift():
    """Import scripts/galaxea_lift.py by path (no scripts/ package) to reuse its physics."""
    p = pathlib.Path(__file__).parent / "galaxea_lift.py"
    spec = importlib.util.spec_from_file_location("galaxea_lift", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/galaxea_lift.yaml")
    ap.add_argument("--out", default="outputs/gravity_compensation/raw/galaxea/galaxea_knee_single.png")
    args = ap.parse_args()

    G = _load_lift()
    cfg = G.load_cfg(args.config)
    m = G.build_reduced(cfg, with_assets=False); d = mujoco.MjData(m)
    motor = E.motor_constants(cfg.motor)

    z = G.z_cycle(cfg)
    q = gravcomp.vertical_lift_ik(m, d, "torso_link4", z, q_init=cfg.ik_seed)
    tau, grav, qd = gravcomp.multi_joint_torque(m, d, q, cfg.dt)

    k = G.TORSO_JOINTS.index(KNEE)
    th, t, g, w = q[:, k], tau[:, k], grav[:, k], qd[:, k]
    spec, e_with, e_base = gravcomp.fit_linear_spring_per_joint(th, t, w, motor, cfg.dt, cfg.regen)
    T = len(th) * cfg.dt
    resid = t - tau_spring(th, spec)
    red = 100 * (e_with - e_base) / e_base
    order = np.argsort(th)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(th[order], t[order], ".", ms=3, color="tab:blue", label="required motor torque (no spring)")
    ax.plot(th[order], g[order], "-", lw=1.5, color="gray", label="gravity torque")
    ax.plot(th[order], tau_spring(th[order], spec), "-", lw=2.5, color="tab:red",
            label=f"fitted single spring  (k={spec.k:.0f} N*m/rad, $\\theta_0$={spec.theta0:.2f} rad)")
    ax.plot(th[order], resid[order], ".", ms=2.5, color="tab:green",
            label="motor torque WITH spring (residual)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("knee (torso_joint2) angle (rad)")
    ax.set_ylabel("torque (N*m)")
    ax.set_title(f"Galaxea R1 knee — single parallel spring\n"
                 f"knee motor power {e_base / T:.1f} W $\\rightarrow$ {e_with / T:.1f} W  ({red:+.0f}%), "
                 f"peak |torque| {np.abs(t).max():.0f} N*m")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3); fig.tight_layout()

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"knee {KNEE}: k={spec.k:.0f} N*m/rad, theta0={spec.theta0:.2f} rad; "
          f"power {e_base / T:.1f} -> {e_with / T:.1f} W ({red:+.0f}%); peak |tau| {np.abs(t).max():.0f} N*m")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
