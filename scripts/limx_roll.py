"""LimX W1 (WL_P311D) wheeled-quadruped rolling-energy experiment (physically grounded).

Realistic torso mass + rolling resistance + sourced low-gear-QDD motor constants. The legs
hold a bent stance (constant-sign knee load) while the wheels roll the robot forward 20 s;
a CONSTANT per-leg knee (calf/KFE) preload spring offloads the stance holding torque.

Wheel transport is billed ANALYTICALLY (Crr*m*g*distance + wheel ohmic), NOT from the
velocity-servo actuator force (which is inflated by contact micro-slip -- see docs).
Reports per-motor + whole-robot (all servos + onboard computer) electrical energy and CoT,
saves a torque-vs-angle plot, a results JSON, and an annotated video. No regen.

  python scripts/limx_roll.py [--config configs/limx_roll.yaml] [--video out.mp4]
"""
import argparse
import dataclasses
import json
import pathlib

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from pea import energy as E, render_util
from pea.urdf_loader import limx_dir, make_spec

LEGS = ["LF", "RF", "LH", "RH"]
G = 9.81


@dataclasses.dataclass
class Cfg:
    urdf: str = "wheellegged/WL_P311D/urdf/robot.urdf"
    torso_mass_kg: float = 17.0
    crr: float = 0.015
    wheel_radius: float = 0.127
    wheel_speed: float = 6.0
    settle_s: float = 2.0
    drive_s: float = 18.0
    dt: float = 0.002
    knee_motor: str = "limx_knee"
    wheel_motor: str = "limx_knee"
    regen: bool = False
    knee_preload: float | None = None
    compute_w: float = 150.0
    compute_sweep: tuple = (0.0, 50.0, 150.0, 300.0)


def load_cfg(path):
    cfg = Cfg()
    if path:
        for k, v in (yaml.safe_load(pathlib.Path(path).read_text()) or {}).items():
            setattr(cfg, k, v)
    return cfg


def build_model(cfg, with_assets=True):
    spec = make_spec(limx_dir() / cfg.urdf,
                     compiler_attrs='fusestatic="false" balanceinertia="true" discardvisual="false"')
    spec.option.timestep = cfg.dt
    spec.body("base").add_freejoint()
    if with_assets:
        render_util.add_scene_assets(spec, floor=True)
    else:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, 0.05], friction=[1.0, 0.01, 0.001])

    def add_pos(joint, kp, kd):
        a = spec.add_actuator(); a.name = joint + "_p"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT; a.target = joint
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0] = 0; a.biasprm[1] = -kp; a.biasprm[2] = -kd

    def add_vel(joint, kv):
        a = spec.add_actuator(); a.name = joint + "_v"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT; a.target = joint
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED; a.gainprm[0] = kv
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0] = 0; a.biasprm[1] = 0; a.biasprm[2] = -kv

    for leg in LEGS:
        add_pos(leg + "_HAA", 250, 6); add_pos(leg + "_HFE", 300, 8)
        add_pos(leg + "_KFE", 300, 8); add_vel(leg + "_WHL", 4.0)
    m = spec.compile()
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    m.body_inertia[bid] *= cfg.torso_mass_kg / float(m.body_mass[bid])
    m.body_mass[bid] = cfg.torso_mass_kg
    return m


def aid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def jdof(m, n): return m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
def jqp(m, n): return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
def bpos(m, d, n): return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()


def find_stance(m, d, target_drop=0.30):
    stance = {}
    for leg in LEGS:
        best, bc = None, 1e9
        for hfe in np.linspace(-1.4, 1.4, 29):
            for kfe in np.linspace(0.1, 2.0, 25):
                mujoco.mj_resetData(m, d); d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 0.6
                d.qpos[jqp(m, leg + "_HFE")] = hfe; d.qpos[jqp(m, leg + "_KFE")] = kfe
                mujoco.mj_forward(m, d)
                r = bpos(m, d, leg + "_wheel") - bpos(m, d, leg + "_hip")
                c = r[0] ** 2 + (r[2] + target_drop) ** 2
                if c < bc:
                    bc, best = c, (0.0, hfe, kfe)
        stance[leg] = best
    return stance


def roll(m, cfg, stance, knee_preload, frames=None):
    """One rollout. Returns (log, distance, Mtot). log[j] = list of (tau, omega, angle)."""
    d = mujoco.MjData(m)
    Mtot = float(m.body_mass.sum())
    tau_roll = cfg.crr * (Mtot * G / 4.0) * cfg.wheel_radius
    d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 1.0
    for leg in LEGS:
        ha, hf, kf = stance[leg]
        d.qpos[jqp(m, leg + "_HAA")] = ha; d.qpos[jqp(m, leg + "_HFE")] = hf
        d.qpos[jqp(m, leg + "_KFE")] = kf
    mujoco.mj_forward(m, d)
    wz = min(bpos(m, d, leg + "_wheel")[2] for leg in LEGS)
    d.qpos[2] = 1.0 - (wz - cfg.wheel_radius)
    ctrl0 = np.zeros(m.nu)
    for leg in LEGS:
        ha, hf, kf = stance[leg]
        for j, v in zip(("HAA", "HFE", "KFE"), (ha, hf, kf)):
            ctrl0[aid(m, f"{leg}_{j}_p")] = v
    d.ctrl[:] = ctrl0

    leg_joints = [f"{leg}_{j}" for leg in LEGS for j in ("HAA", "HFE", "KFE")]
    wheel_joints = [f"{leg}_WHL" for leg in LEGS]
    act = {j: aid(m, j + ("_v" if j.endswith("WHL") else "_p")) for j in leg_joints + wheel_joints}
    dof = {j: jdof(m, j) for j in leg_joints + wheel_joints}
    qpa = {j: jqp(m, j) for j in leg_joints}
    kfe_dof = {leg: jdof(m, leg + "_KFE") for leg in LEGS}
    whl_dof = {leg: jdof(m, leg + "_WHL") for leg in LEGS}

    n_settle = int(cfg.settle_s / cfg.dt); n_drive = int(cfg.drive_s / cfg.dt)
    log = {j: [] for j in leg_joints + wheel_joints}
    x0 = x_end = None; e_live = 0.0
    if frames is not None:
        kmc, wmc = frames["knee_mc"], frames["wheel_mc"]

    for i in range(n_settle + n_drive):
        drive = i >= n_settle
        for leg in LEGS:
            d.ctrl[aid(m, leg + "_WHL_v")] = cfg.wheel_speed if drive else 0.0
        d.qfrc_applied[:] = 0.0
        for leg in LEGS:
            d.qfrc_applied[whl_dof[leg]] = -np.sign(d.qvel[whl_dof[leg]]) * tau_roll
            d.qfrc_applied[kfe_dof[leg]] = knee_preload.get(leg, 0.0)
        mujoco.mj_step(m, d)
        if drive:
            if x0 is None:
                x0 = float(d.qpos[0])
            for j in leg_joints:
                log[j].append((float(d.actuator_force[act[j]]), float(d.qvel[dof[j]]), float(d.qpos[qpa[j]])))
            for j in wheel_joints:
                log[j].append((float(d.actuator_force[act[j]]), float(d.qvel[dof[j]]), 0.0))
            x_end = float(d.qpos[0])
            if frames is not None:
                P = sum(float(E.electrical_power(d.actuator_force[act[j]], d.qvel[dof[j]], kmc.kt, kmc.r, regen=cfg.regen)) for j in leg_joints)
                P += 4 * float(E.electrical_power(tau_roll, cfg.wheel_speed, wmc.kt, wmc.r, regen=cfg.regen))
                e_live += P * cfg.dt
                if (i - n_settle) % frames["stride"] == 0:
                    frames["cam"].lookat[:] = d.qpos[:3]
                    frames["renderer"].update_scene(d, camera=frames["cam"])
                    fr = frames["renderer"].render().copy()
                    lines = ["LimX W1  -  wheeled roll (knee gravity-comp spring)",
                             f"no spring:    {frames['e0']:5.0f} J    {frames['p0']:4.0f} W",
                             f"with spring:  {e_live:5.0f} J    {P:4.0f} W",
                             f"total electrical energy  {frames['pct']:+.0f}%"]
                    frames["buf"].append(render_util.overlay_text(
                        fr, lines, colors=[(235,235,235),(255,120,120),(120,235,140),(255,235,140)]))
    return log, (x_end - x0), Mtot


def energy_of(log, cfg, knee_mc, wheel_mc, dist, Mtot):
    """Electrical energy (J). Legs from logged actuator force; wheels billed ANALYTICALLY
    (rolling resistance Crr*m*g*distance + wheel ohmic), avoiding velocity-servo slip."""
    def jE(j, mc):
        a = np.array(log[j]); return E.energy(E.electrical_power(a[:, 0], a[:, 1], mc.kt, mc.r, regen=cfg.regen), cfg.dt)
    knee = sum(jE(leg + "_KFE", knee_mc) for leg in LEGS)
    hipE = sum(jE(leg + j, knee_mc) for leg in LEGS for j in ("_HAA", "_HFE"))
    tau_roll = cfg.crr * (Mtot * G / 4.0) * cfg.wheel_radius
    wheel = cfg.crr * Mtot * G * dist + 4 * float(E.ohmic_power(tau_roll, wheel_mc.kt, wheel_mc.r)) * cfg.drive_s
    return dict(knee=knee, hip=hipE, wheel=wheel, total=knee + hipE + wheel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--video", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    m = build_model(cfg, with_assets=False); d0 = mujoco.MjData(m)
    knee_mc = E.motor_constants(cfg.knee_motor); wheel_mc = E.motor_constants(cfg.wheel_motor)
    stance = find_stance(m, d0)

    log0, dist0, Mtot = roll(m, cfg, stance, {leg: 0.0 for leg in LEGS})
    mk = {leg: float(np.mean([t for t, _, _ in log0[leg + "_KFE"]])) for leg in LEGS}
    e0 = energy_of(log0, cfg, knee_mc, wheel_mc, dist0, Mtot)
    preload = mk if cfg.knee_preload is None else {leg: float(cfg.knee_preload) for leg in LEGS}
    log1, dist1, _ = roll(m, cfg, stance, preload)
    e1 = energy_of(log1, cfg, knee_mc, wheel_mc, dist1, Mtot)
    T = cfg.drive_s
    def cot(Ej, dist): return Ej / (Mtot * G * dist) if dist > 1e-6 else float("nan")

    print(f"=== LimX W1 rolling energy (torso {cfg.torso_mass_kg}kg, total {Mtot:.1f}kg, Crr {cfg.crr}, "
          f"knee R/Kt^2={knee_mc.r/knee_mc.kt**2:.3f}, regen={cfg.regen}) ===")
    print(f"drive {T:.0f}s, {dist0:.2f} m ({dist0/T:.2f} m/s); "
          f"baseline knee torque per leg: " + " ".join(f"{leg} {mk[leg]:+.1f}" for leg in LEGS) + " N*m")
    print(f"  {'energy (J)':22s} {'knee':>9} {'hip(8 jt)':>10} {'wheel':>8} {'TOTAL':>9} {'CoT':>8}")
    print(f"  {'baseline (no spring)':22s} {e0['knee']:9.1f} {e0['hip']:10.1f} {e0['wheel']:8.1f} {e0['total']:9.1f} {cot(e0['total'],dist0):8.4f}")
    print(f"  {'with knee spring':22s} {e1['knee']:9.1f} {e1['hip']:10.1f} {e1['wheel']:8.1f} {e1['total']:9.1f} {cot(e1['total'],dist1):8.4f}")
    print(f"  knee-motor saving {100*(e1['knee']-e0['knee'])/e0['knee']:+.1f}%   "
          f"all-servo saving {100*(e1['total']-e0['total'])/e0['total']:+.1f}%")
    print(f"\n  whole-robot (all servos + computer), saving = {e0['total']-e1['total']:.0f} J:")
    print(f"  {'compute (W)':>12} {'overall base(J)':>16} {'overall spring(J)':>18} {'overall red':>12}")
    for w in cfg.compute_sweep:
        ob = e0['total'] + w * T; os_ = e1['total'] + w * T
        tag = "  <-- headline" if abs(w - cfg.compute_w) < 1e-6 else ""
        print(f"  {w:12.0f} {ob:16.0f} {os_:18.0f} {100*(os_-ob)/ob:+11.1f}%{tag}")

    # torque-vs-angle plot (per leg knee) + constant preload spring
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for leg in LEGS:
        a = np.array(log0[leg + "_KFE"])
        ax.plot(a[:, 2], a[:, 0], ".", ms=2, label=f"{leg} knee torque")
    for leg in LEGS:
        ax.axhline(preload[leg], ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("knee (KFE) angle (rad)"); ax.set_ylabel("motor torque (N*m)")
    ax.set_title(f"LimX W1 rolling: knee torque vs angle + constant preload spring "
                 f"(knee {100*(e1['knee']-e0['knee'])/e0['knee']:+.0f}%)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    pathlib.Path("outputs/gravity_compensation/raw/limx").mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/gravity_compensation/raw/limx/limx_taus.png", dpi=130); plt.close(fig)
    print("\nWROTE outputs/gravity_compensation/raw/limx/limx_taus.png")

    results = dict(robot="LimX W1 (wheeled roll)", target_label="4 knees (KFE)", duration_s=T,
                   target_base_J=e0['knee'], target_spring_J=e1['knee'],
                   other_servo_J=e0['hip'] + e0['wheel'], distance_m=dist0,
                   all_servo_base_J=e0['total'], all_servo_spring_J=e1['total'],
                   cot_base=cot(e0['total'], dist0), cot_spring=cot(e1['total'], dist1), regen=cfg.regen)
    pathlib.Path("outputs/gravity_compensation/raw/limx/limx_results.json").write_text(json.dumps(results, indent=2))
    print("WROTE outputs/gravity_compensation/raw/limx/limx_results.json")

    if args.video:
        m_vid = build_model(cfg, with_assets=True)   # textured scene (skybox + checker floor)
        W, H = 1280, 720; render_util.set_quality(m_vid, W, H)
        cam = render_util.free_camera(3.6, -16.0, 120.0)
        fr = {"renderer": mujoco.Renderer(m_vid, H, W), "cam": cam, "stride": max(1, round((1/30)/cfg.dt)),
              "buf": [], "knee_mc": knee_mc, "wheel_mc": wheel_mc, "e0": e0['total'],
              "p0": e0['total']/T, "pct": 100*(e1['total']-e0['total'])/e0['total']}
        roll(m_vid, cfg, stance, preload, frames=fr)
        render_util.write_mp4(fr["buf"], args.video, 30, W, H)


if __name__ == "__main__":
    main()
