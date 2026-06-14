"""Direct power comparison: no-spring (N W) vs hip-spring (M W), baseline gait.

Prints mean power in WATTS over the steady window, decomposed into mechanical
(signed / positive / negative-braking) and ohmic, under BOTH regen and no-regen,
for (a) the whole-body actuators and (b) the two hip-pitch motors. The spring
case subtracts tau_spring(theta) from the hip-pitch motor torque on the fixed
baseline gait (post-hoc optimistic bound). Sim with ESTIMATED Kt/R.
"""

from __future__ import annotations

import pathlib

import numpy as np

from pea import config as cfg_lib
from pea import energy, springs
from pea.env import joints_by_substring, make_env

RUN = pathlib.Path(
    "/Users/elijah/Library/CloudStorage/GoogleDrive-kefir8888@gmail.com/"
    "Мой диск/pea_runs/2026-06-11_baseline_h100"
)
TRANSIENT_S = 2.0
KT, R = energy.G1_KNEE.kt, energy.G1_KNEE.r


def main():
    tr = np.load(RUN / "trajectory.npz", allow_pickle=True)
    dt = float(tr["dt"])
    skip = int(TRANSIENT_S / dt)
    qvel = tr["qvel"][skip:]
    tau = tr["qfrc_actuator"][skip:]
    T = len(tau)
    dur = T * dt

    cfg = cfg_lib.load_config("configs/baseline_gate.yaml")
    model = make_env(cfg).mj_model
    import mujoco

    act = [int(model.jnt_dofadr[j]) for j in range(model.njnt)
           if int(model.jnt_dofadr[j]) >= 6]
    hip = joints_by_substring(model, "hip_pitch")
    hip_dof = [v["dof_adr"] for v in hip.values()]
    hip_qpos = [v["qpos_adr"] for v in hip.values()]

    spec = springs.from_config(
        cfg_lib.load_config("configs/spring_hip_linear.yaml").spring)

    def block(dofs, tau_mod=None):
        """Mean watts for a set of DoFs. tau_mod optionally replaces torque."""
        t = tau[:, dofs].copy()
        w = qvel[:, dofs]
        if tau_mod is not None:
            t = tau_mod
        mech = t * w                       # (T, k) instantaneous W
        ohm = (t / KT) ** 2 * R
        s = dt / dur                       # sum(power)*dt/dur = mean W over window
        mech_W = mech.sum() * s            # signed mechanical, mean W
        pos_W = np.maximum(mech, 0).sum() * s
        neg_W = np.minimum(mech, 0).sum() * s
        ohm_W = ohm.sum() * s
        elec_regen = (mech + ohm).sum() * s               # allow recharge
        elec_noregen = np.maximum(mech + ohm, 0).sum() * s  # per-DoF clamp >=0
        return dict(mech=mech_W, pos=pos_W, neg=neg_W, ohm=ohm_W,
                    regen=elec_regen, noregen=elec_noregen)

    # hip torque with spring subtracted
    th = tr["qpos"][skip:][:, hip_qpos]
    tau_hip_spring = tau[:, hip_dof] - springs.tau_spring(th, spec)
    # whole-body with spring = baseline torques but hip-pitch replaced
    tau_wb_spring = tau[:, act].copy()
    hip_pos_in_act = [act.index(d) for d in hip_dof]
    tau_wb_spring[:, hip_pos_in_act] = tau_hip_spring

    wb0 = block(act)
    wbS = block(act, tau_mod=tau_wb_spring)
    hp0 = block(hip_dof)
    hpS = block(hip_dof, tau_mod=tau_hip_spring)

    def show(name, d):
        print(f"  {name:24s} mech(signed) {d['mech']:6.1f}  "
              f"[+{d['pos']:6.1f} / {d['neg']:6.1f}]  ohmic {d['ohm']:4.1f}  "
              f"| regen {d['regen']:6.1f}  no-regen {d['noregen']:6.1f}")

    print(f"\nMEAN POWER over {dur:.0f} s steady walk (W); Kt={KT}, R={R} "
          f"(ESTIMATED). 'no-regen' is the reported model.\n")
    print("WHOLE-BODY actuators (29 DoF):")
    show("no spring (N)", wb0)
    show("hip spring (M)", wbS)
    print(f"    -> whole-body no-regen:  N = {wb0['noregen']:.1f} W  ->  "
          f"M = {wbS['noregen']:.1f} W   (saved {wb0['noregen']-wbS['noregen']:.1f} W, "
          f"{100*(wbS['noregen']/wb0['noregen']-1):+.1f}%)")
    print("\nHIP-PITCH motors (2 DoF):")
    show("no spring (N)", hp0)
    show("hip spring (M)", hpS)
    print(f"    -> hip-pitch no-regen:   N = {hp0['noregen']:.1f} W  ->  "
          f"M = {hpS['noregen']:.1f} W   (saved {hp0['noregen']-hpS['noregen']:.1f} W, "
          f"{100*(hpS['noregen']/hp0['noregen']-1):+.1f}%)")
    print(f"\nNO-REGEN INFLATION (whole-body): regen {wb0['regen']:.1f} W -> "
          f"no-regen {wb0['noregen']:.1f} W  (+{wb0['noregen']-wb0['regen']:.1f} W, "
          f"{100*(wb0['noregen']/wb0['regen']-1):+.0f}%)")
    print(f"  negative (braking) mechanical power being dissipated: "
          f"{-wb0['neg']:.1f} W")


if __name__ == "__main__":
    main()
