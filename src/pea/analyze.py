"""Knee-focused gait analysis from a logged trajectory.

Reads trajectory.npz only — no env or jax dependency. Plots knee angle, knee
torque, and the torque-angle work loop (the plot a spring curve gets fitted
to), and prints summary stats.

--spring <config.yaml> runs the offline (post-hoc) analysis: subtract
tau_spring(theta) from the recorded motor torque on the same trajectory and
recompute electrical power per joint. Optimistic upper bound — assumes the gait
does not change. The credible comparison is retraining with the spring included.

Always prints a whole-body cost of transport (electrical energy / m g d), under
the no-regeneration assumption (primary) with a regeneration value alongside for
reference.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from pea import energy, springs
from pea.config import load_config

TRANSIENT_S = 2.0  # settle time dropped from all stats


def posthoc(tr, spring_cfg_path: str) -> None:
    spec = springs.from_config(load_config(spring_cfg_path).spring)
    if spec is None:
        raise SystemExit("config has spring.kind=none — nothing to subtract")
    dt = float(tr["dt"])
    skip = int(TRANSIENT_S / dt)
    kt, r = energy.G1_KNEE.kt, energy.G1_KNEE.r
    print(f"\nPOST-HOC spring offload ({spec}) — optimistic bound, gait fixed:")
    tot_base, tot_spring = 0.0, 0.0
    for name, qa, da in zip(
        tr["knee_names"], tr["knee_qpos_adr"], tr["knee_dof_adr"]
    ):
        theta = tr["qpos"][skip:, qa]
        omega = tr["qvel"][skip:, da]
        tau = tr["qfrc_actuator"][skip:, da]
        tau_new = tau - springs.tau_spring(theta, spec)
        e_base = energy.energy(energy.electrical_power(tau, omega, kt, r), dt)
        e_new = energy.energy(energy.electrical_power(tau_new, omega, kt, r), dt)
        cu_base = energy.energy(energy.copper_loss(tau, kt, r), dt)
        cu_new = energy.energy(energy.copper_loss(tau_new, kt, r), dt)
        tot_base += e_base
        tot_spring += e_new
        print(
            f"  {name}: electrical {e_base:7.1f} -> {e_new:7.1f} J "
            f"({100 * (e_new / e_base - 1):+5.1f}%)   "
            f"copper {cu_base:6.1f} -> {cu_new:6.1f} J "
            f"({100 * (cu_new / cu_base - 1):+5.1f}%)"
        )
    print(
        f"  TOTAL joint electrical: {tot_base:.1f} -> {tot_spring:.1f} J "
        f"({100 * (tot_spring / tot_base - 1):+.1f}%)   "
        f"[Kt={kt}, R={r} PLACEHOLDER; no-regen]"
    )


def cost_of_transport_report(tr) -> None:
    """Whole-body cost of transport from the recorded trajectory.

    Sums electrical power over all actuated DoFs. No-regeneration clamps each
    actuator's electrical power to >= 0 individually (a motor cannot recharge
    the battery); the regeneration value allows negative power to net across
    the bus and is reported only as a reference bound. Placeholder Kt/R, so the
    ABSOLUTE CoT is not trustworthy; the spring-vs-no-spring RATIO and the ohmic
    percentage are the defensible quantities.
    """
    dt = float(tr["dt"])
    skip = int(TRANSIENT_S / dt)
    kt, r = energy.G1_KNEE.kt, energy.G1_KNEE.r
    tau = tr["qfrc_actuator"][skip:]            # (T, ndof)
    omega = tr["qvel"][skip:]
    mech = tau * omega                          # mechanical power per DoF
    ohm = energy.ohmic_power(tau, kt, r)        # Joule heating per DoF
    e_mech = float(np.sum(mech) * dt)
    e_ohm = float(np.sum(ohm) * dt)
    # No regeneration: each actuator's electrical power is clamped >= 0 (a geared
    # G1 motor does not recharge the battery during braking). This is the sole
    # reported model — see docs/related_work.md / RESULTS.md.
    e_elec = float(np.sum(np.maximum(mech + ohm, 0.0)) * dt)
    dist = float(np.linalg.norm(tr["qpos"][-1, :2] - tr["qpos"][skip, :2]))
    mass = float(tr["total_mass"])
    denom = mass * energy.G * max(dist, 1e-9)
    print(
        f"\nWHOLE-BODY energy over {len(tau)*dt:.1f} s, {dist:.2f} m "
        f"(mass {mass:.1f} kg), no regeneration:\n"
        f"  mechanical work {e_mech:7.1f} J   ohmic (Joule) {e_ohm:7.1f} J\n"
        f"  electrical {e_elec:7.1f} J  ->  CoT {e_elec / denom:.3f}\n"
        f"  [Kt={kt}, R={r} PLACEHOLDER — absolute CoT not trustworthy; "
        f"use spring-vs-no-spring ratios]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run folder with trajectory.npz")
    p.add_argument(
        "--spring",
        default=None,
        help="spring config YAML for post-hoc offload analysis (Milestone 3)",
    )
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

    cost_of_transport_report(tr)

    if args.spring:
        posthoc(tr, args.spring)


if __name__ == "__main__":
    main()
