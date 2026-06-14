"""Recalculate the hip-pitch spring's energy saving across a band of R/Kt^2.

The robot inventory (docs/robot_inventory.md) flags the team's R/Kt^2 ~= 0.0025 as
~8-19x optimistic; the corrected Go2-proxy estimate is ~0.02 (knee) to ~0.05
(hip-pitch, lower-geared). Ohmic power = tau^2 * (R/Kt^2), so the ohmic channel
scales LINEARLY with this single ratio while the mechanical (tau*omega) channel is
unchanged. This re-runs the whole-body and hip-pitch no-regen power decomposition
plus the post-hoc hip-linear-spring saving at several R/Kt^2 values, on the SAME
2026-06-11 baseline trajectory used for the original numbers. Estimate-grade input;
read the band, not a single figure.
"""

from __future__ import annotations

import pathlib

import numpy as np

from pea import config as cfg_lib
from pea import springs
from pea.env import joints_by_substring, make_env

RUN = pathlib.Path(
    "/Users/elijah/Library/CloudStorage/GoogleDrive-kefir8888@gmail.com/"
    "Мой диск/pea_runs/2026-06-11_baseline_h100"
)
TRANSIENT_S = 2.0
# R/Kt^2 [Ohm/(N*m/A)^2]: current team value, then the corrected band.
CVALS = [0.0025, 0.01, 0.02, 0.05]


def main():
    tr = np.load(RUN / "trajectory.npz", allow_pickle=True)
    dt = float(tr["dt"])
    skip = int(TRANSIENT_S / dt)
    qvel = tr["qvel"][skip:]
    tau = tr["qfrc_actuator"][skip:]
    dur = len(tau) * dt

    cfg = cfg_lib.load_config("configs/baseline_gate.yaml")
    model = make_env(cfg).mj_model
    act = [int(model.jnt_dofadr[j]) for j in range(model.njnt)
           if int(model.jnt_dofadr[j]) >= 6]
    hip = joints_by_substring(model, "hip_pitch")
    hip_dof = [v["dof_adr"] for v in hip.values()]
    hip_qpos = [v["qpos_adr"] for v in hip.values()]

    spec = springs.from_config(
        cfg_lib.load_config("configs/spring_hip_linear.yaml").spring)
    th = tr["qpos"][skip:][:, hip_qpos]
    tau_hip_spring = tau[:, hip_dof] - springs.tau_spring(th, spec)
    tau_wb_spring = tau[:, act].copy()
    hip_pos_in_act = [act.index(d) for d in hip_dof]
    tau_wb_spring[:, hip_pos_in_act] = tau_hip_spring

    def block(t, w, c):
        mech = t * w
        ohm = t ** 2 * c
        s = dt / dur
        return dict(
            noregen=float(np.maximum(mech + ohm, 0).sum() * s),
            regen=float((mech + ohm).sum() * s),
            ohm=float(ohm.sum() * s),
        )

    print(f"\nbaseline {RUN.name}   window {dur:.0f} s   "
          f"(post-hoc, no-regen, fixed gait — optimistic bound)\n")
    head = (f"{'R/Kt^2':>8} | {'WB noreg':>9} {'+spring':>8} {'saved W':>8} "
            f"{'saved %':>8} {'WB ohm%':>8} | {'HP noreg':>9} {'+spring':>8} "
            f"{'saved W':>8} {'saved %':>8} {'HP ohm%':>8}")
    print(head)
    print("-" * len(head))
    for c in CVALS:
        wbN = block(tau[:, act], qvel[:, act], c)
        wbM = block(tau_wb_spring, qvel[:, act], c)
        hpN = block(tau[:, hip_dof], qvel[:, hip_dof], c)
        hpM = block(tau_hip_spring, qvel[:, hip_dof], c)
        wb_saved = wbN["noregen"] - wbM["noregen"]
        hp_saved = hpN["noregen"] - hpM["noregen"]
        tag = "  <-- current" if c == 0.0025 else (
            "  <-- hip est" if c == 0.05 else "")
        print(f"{c:>8.4f} | {wbN['noregen']:>9.1f} {wbM['noregen']:>8.1f} "
              f"{wb_saved:>8.1f} {100*wb_saved/wbN['noregen']:>7.1f}% "
              f"{100*wbN['ohm']/wbN['noregen']:>7.1f}% | "
              f"{hpN['noregen']:>9.1f} {hpM['noregen']:>8.1f} {hp_saved:>8.1f} "
              f"{100*hp_saved/hpN['noregen']:>7.1f}% "
              f"{100*hpN['ohm']/hpN['noregen']:>7.1f}%{tag}")
    print("\nWB = whole body (29 actuated DoF, single R/Kt^2 applied to all — an "
          "approximation;\n     real knee~0.02 / hip-pitch~0.05 differ). "
          "HP = the two hip-pitch motors (spring target, est ~0.05).")
    print("Mechanical (tau*omega) saving is R/Kt^2-invariant; only the ohmic "
          "channel and\nthe no-regen-clamp coupling move with R/Kt^2.\n")


if __name__ == "__main__":
    main()
