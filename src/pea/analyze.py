"""Knee-focused gait analysis from a logged trajectory (Milestone 1 scope).

Reads trajectory.npz only — no env or jax dependency. Plots knee angle, knee
torque, and the torque-angle work loop (the plot a spring curve gets fitted
to), and prints summary stats. Full electrical CoT comparison lands in M2/M3.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from pea import energy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder with trajectory.npz")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run)
    tr = np.load(run_dir / "trajectory.npz")
    t = tr["time"]
    dt = float(tr["dt"])
    # planar displacement — initial heading is randomized at reset
    dist = float(np.linalg.norm(tr["qpos"][-1, :2] - tr["qpos"][0, :2]))
    print(f"duration {t[-1]:.1f} s   distance {dist:.2f} m   "
          f"mean speed {dist / max(t[-1], 1e-9):.2f} m/s")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(s) for s in tr["knee_names"]]
    fig, axes = plt.subplots(len(names), 3, figsize=(14, 4 * len(names)),
                             squeeze=False)
    for i, name in enumerate(names):
        theta = tr["qpos"][:, tr["knee_qpos_adr"][i]]
        tau = tr["qfrc_actuator"][:, tr["knee_dof_adr"][i]]
        omega = tr["qvel"][:, tr["knee_dof_adr"][i]]
        rms_tau = float(np.sqrt(np.mean(tau**2)))
        p_cu = energy.copper_loss(tau, energy.G1_KNEE.kt, energy.G1_KNEE.r)
        neg_work = float(np.sum(np.minimum(tau * omega, 0.0)) * dt)
        print(f"{name}: RMS torque {rms_tau:.1f} N*m   "
              f"mean copper loss {np.mean(p_cu):.1f} W (PLACEHOLDER Kt,R)   "
              f"negative work {neg_work:.1f} J")

        axes[i, 0].plot(t, theta)
        axes[i, 0].set(title=f"{name}: angle", xlabel="t [s]", ylabel="θ [rad]")
        axes[i, 1].plot(t, tau)
        axes[i, 1].set(title=f"{name}: motor torque", xlabel="t [s]",
                       ylabel="τ [N·m]")
        axes[i, 2].plot(theta, tau, lw=0.8)
        axes[i, 2].set(title=f"{name}: work loop", xlabel="θ [rad]",
                       ylabel="τ [N·m]")
    fig.tight_layout()
    out = run_dir / "knee_gait.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
