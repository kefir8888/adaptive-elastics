"""One-off: per-joint electrical breakdown + hip-pitch share + hip-spring savings.

Answers, from the real 2026-06-11 H100 baseline trajectory:
  Q2  share of the two hip-pitch motors in total motor electrical consumption,
  Q3  post-hoc electrical saving from the hip-pitch linear spring (k=68, th0=-0.29),
      reported both at the hip joints and as a fraction of total motor energy.
Also prints the whole-body mechanical-vs-ohmic split (context for Q1).

No-regeneration, clamped per DoF (max(P,0) each actuator), matching the project
convention in energy.py / env.ElectricalRewardWrapper.
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


def per_dof_elec(tau, omega, dt, regen=False):
    """Electrical energy per DoF over the window, no-regen clamp per actuator."""
    mech = tau * omega
    ohm = (tau / KT) ** 2 * R
    p = mech + ohm
    if not regen:
        p = np.maximum(p, 0.0)
    return p.sum(axis=0) * dt, mech.sum(axis=0) * dt, ohm.sum(axis=0) * dt


def main():
    tr = np.load(RUN / "trajectory.npz", allow_pickle=True)
    dt = float(tr["dt"])
    skip = int(TRANSIENT_S / dt)
    qvel = tr["qvel"][skip:]            # (T, 35)
    tau = tr["qfrc_actuator"][skip:]    # (T, 35)
    T = len(tau)
    dur = T * dt

    # Build dof_adr -> joint name map for every actuated joint (dof_adr >= 6).
    # Use a clean config (the run's own config.yaml has a stale cubic-spring
    # field); the G1 model is identical regardless of spring/energy settings.
    cfg = cfg_lib.load_config("configs/baseline_gate.yaml")
    model = make_env(cfg).mj_model
    import mujoco

    dof2name = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(mujoco.mjtObj.mjOBJ_JOINT, j) if False else \
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_dofadr[j])
        if adr >= 6 and name:
            dof2name[adr] = name

    e_dof, mech_dof, ohm_dof = per_dof_elec(tau, qvel, dt)

    # ---- whole-body totals (actuated DoFs only) ----
    act = sorted(dof2name)
    tot_elec = e_dof[act].sum()
    tot_mech_signed = mech_dof[act].sum()
    tot_mech_pos = np.maximum(tr["qfrc_actuator"][skip:][:, act]
                              * qvel[:, act], 0.0).sum() * dt
    tot_ohm = ohm_dof[act].sum()
    print(f"\n=== Baseline whole-body motor budget  ({dur:.1f} s steady walk, "
          f"Kt={KT}, R={R}) ===")
    print(f"  total motor electrical (no-regen, per-DoF clamp): {tot_elec:8.1f} J"
          f"  -> mean {tot_elec/dur:6.1f} W")
    print(f"  of which mechanical (clamped +work): {tot_mech_pos:8.1f} J  "
          f"({100*tot_mech_pos/tot_elec:4.1f}%)")
    print(f"          ohmic  (I^2 R, always >=0):  {tot_ohm:8.1f} J  "
          f"({100*tot_ohm/tot_elec:4.1f}%)")
    print(f"  [signed mechanical work (regen view): {tot_mech_signed:8.1f} J]")

    # ---- per-joint breakdown, sorted by share ----
    print(f"\n=== Per-joint electrical share of motor consumption ===")
    rows = sorted(((dof2name[a], e_dof[a]) for a in act),
                  key=lambda kv: -kv[1])
    for name, e in rows:
        print(f"  {name:28s} {e:8.1f} J   {100*e/tot_elec:5.1f}%")

    # ---- hip-pitch share (Q2) ----
    hip = joints_by_substring(model, "hip_pitch")
    hip_adrs = [v["dof_adr"] for v in hip.values()]
    e_hip = e_dof[hip_adrs].sum()
    print(f"\n=== Q2: hip-pitch share ===")
    for nm, v in hip.items():
        print(f"  {nm:28s} {e_dof[v['dof_adr']]:8.1f} J  "
              f"{100*e_dof[v['dof_adr']]/tot_elec:5.1f}%")
    print(f"  BOTH hip-pitch / total motor electrical: "
          f"{e_hip:.1f} / {tot_elec:.1f} J = {100*e_hip/tot_elec:.1f}%")

    # ---- hip-pitch post-hoc spring saving (Q3) ----
    spec = springs.from_config(
        cfg_lib.load_config("configs/spring_hip_linear.yaml").spring)
    print(f"\n=== Q3: hip-pitch linear spring post-hoc ({spec}) — gait FIXED, "
          f"optimistic bound ===")
    qpos = tr["qpos"][skip:]
    e_hip_new_total = 0.0
    ohm_hip_base = ohm_hip_new = 0.0
    for nm, v in hip.items():
        th = qpos[:, v["qpos_adr"]]
        om = qvel[:, v["dof_adr"]]
        t0 = tr["qfrc_actuator"][skip:][:, v["dof_adr"]]
        t1 = t0 - springs.tau_spring(th, spec)
        eb, _, ob = per_dof_elec(t0[:, None], om[:, None], dt)
        en, _, on = per_dof_elec(t1[:, None], om[:, None], dt)
        e_hip_new_total += en[0]
        ohm_hip_base += ob[0]
        ohm_hip_new += on[0]
        print(f"  {nm:28s} elec {eb[0]:7.1f} -> {en[0]:7.1f} J "
              f"({100*(en[0]/eb[0]-1):+5.1f}%)   "
              f"ohmic {ob[0]:6.2f} -> {on[0]:6.2f} J "
              f"({100*(on[0]/ob[0]-1):+5.1f}%)")
    saved = e_hip - e_hip_new_total
    print(f"  hip-pitch electrical: {e_hip:.1f} -> {e_hip_new_total:.1f} J  "
          f"(saved {saved:.1f} J, {100*saved/e_hip:+.1f}% of hip-pitch)")
    print(f"  hip-pitch ohmic:      {ohm_hip_base:.2f} -> {ohm_hip_new:.2f} J  "
          f"({100*(ohm_hip_new/ohm_hip_base-1):+.1f}%)")
    print(f"  saving as fraction of TOTAL motor electrical: "
          f"{100*saved/tot_elec:.2f}%   (optimistic, fixed gait)")


if __name__ == "__main__":
    main()
