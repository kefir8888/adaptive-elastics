"""Galaxea R1 COORDINATED UPRIGHT LIFT gravity-compensation energy experiment.

The upper body (~27 kg of torso + arms) is translated VERTICALLY while kept LEVEL
(coordinated 3-joint torso motion solved by IK), going up and down on a fixed cycle.
We compute the ELECTRICAL energy per torso joint with/without a parallel linear torsion
spring (one per joint), the arms' constant holding power, and the whole-robot total
(all servos + a fixed onboard computer). Saves a tau-vs-angle + fitted-spring plot, a
results JSON, and an annotated video.

  python scripts/galaxea_lift.py [--config configs/galaxea_lift.yaml] [--video out.mp4]
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

from pea import energy as E, gravcomp, render_util
from pea.springs import tau_spring
from pea.urdf_loader import fiveages_dir, make_spec, subtree_mass

TORSO_JOINTS = ["torso_joint1", "torso_joint2", "torso_joint3"]
ARM_JOINTS = [f"{s}_joint{i}" for s in ("left", "right") for i in range(1, 7)]


def _spring_str(spec):
    """Short kind-aware descriptor for the table/plot legend."""
    if spec is None:
        return "(none)"
    if spec.kind == "constant":
        return f"const t0={spec.tau0:+.0f}"
    return f"linear k={spec.k:.0f}"


@dataclasses.dataclass
class Cfg:
    urdf: str = "humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf"
    z_top: float = 0.95
    z_bot: float = 0.62
    cycle_s: float = 2.5
    total_s: float = 20.0
    dwell_frac: float = 0.15      # fraction of each cycle held at top & bottom
    dt: float = 0.004
    ik_seed: tuple = (0.4, -0.8, -0.4)   # bent seed (upright pose is an IK singularity)
    motor: str = "galaxea_torso"
    arm_motor: str = "galaxea_arm"
    regen: bool = False
    spring_joints: tuple = ("torso_joint1", "torso_joint2", "torso_joint3")  # which joints get a spring
    label: str = "Galaxea R1 (coordinated upright lift)"   # robot/scenario label for the table
    tag: str = ""                                          # output-filename suffix for this scenario
    compute_w: float = 150.0
    compute_sweep: tuple = (0.0, 50.0, 150.0, 300.0)


def load_cfg(path):
    cfg = Cfg()
    if path:
        for k, v in (yaml.safe_load(pathlib.Path(path).read_text()) or {}).items():
            setattr(cfg, k, v)
    return cfg


def build_reduced(cfg, with_assets):
    """Reduced 3-DOF torso model (arms/head/wheels welded as rigid mass)."""
    spec = make_spec(fiveages_dir() / cfg.urdf,
                     compiler_attrs='fusestatic="false" discardvisual="false"')
    for j in list(spec.joints):
        if j.name not in set(TORSO_JOINTS):
            spec.delete(j)
    if with_assets:
        render_util.add_scene_assets(spec, floor=True)
    m = spec.compile()
    return m


def z_cycle(cfg):
    """Min-jerk up/down: hold(top)-lower-hold(bot)-raise, repeated; height z(t)."""
    dwell = cfg.cycle_s * cfg.dwell_frac
    move = (cfg.cycle_s - 2 * dwell) / 2.0
    th, om, al = gravcomp.minjerk_cycle(cfg.z_top, cfg.z_bot, dwell, move,
                                        max(1, round(cfg.total_s / cfg.cycle_s)), cfg.dt)
    return th  # z path (treat 'theta' channel as the height target)


def arm_holding_power(cfg):
    """Constant arms power: the arms hold a fixed (level) pose, so their gravity
    holding torque -> ohmic is a constant background. Computed on the FULL model."""
    spec = make_spec(fiveages_dir() / cfg.urdf, compiler_attrs="")
    m = spec.compile(); d = mujoco.MjData(m)
    # nominal: torso mid-lift, arms at mid-range (off their limits)
    for j in range(m.njnt):
        if m.jnt_type[j] in (2, 3) and m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]; d.qpos[m.jnt_qposadr[j]] = 0.5 * (lo + hi)
    mujoco.mj_forward(m, d)
    mc = E.motor_constants(cfg.arm_motor)
    P = 0.0
    for jn in ARM_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        tau = float(d.qfrc_bias[m.jnt_dofadr[jid]])
        P += float(E.ohmic_power(tau, mc.kt, mc.r))   # holding: omega=0 -> ohmic only
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    m = build_reduced(cfg, with_assets=False); d = mujoco.MjData(m)
    motor = E.motor_constants(cfg.motor)
    up_mass = subtree_mass(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link1"))

    z = z_cycle(cfg)
    q = gravcomp.vertical_lift_ik(m, d, "torso_link4", z, q_init=cfg.ik_seed)   # (N x 3)
    tau, grav, qd = gravcomp.multi_joint_torque(m, d, q, cfg.dt)  # (N x 3) each
    N = q.shape[0]; T = N * cfg.dt

    # per-joint spring fit + energy (a joint NOT in spring_joints keeps its full motor load)
    springs, e_base_j, e_spr_j = [], [], []
    for k in range(3):
        e_base = gravcomp.lift_energy(tau[:, k], qd[:, k], motor, cfg.dt, cfg.regen).total
        if TORSO_JOINTS[k] in cfg.spring_joints:
            spec, e_with, e_base = gravcomp.fit_spring_per_joint(
                q[:, k], tau[:, k], qd[:, k], motor, cfg.dt, cfg.regen)
        else:
            spec, e_with = None, e_base          # no spring on this joint
        springs.append(spec); e_base_j.append(e_base); e_spr_j.append(e_with)
    target_base = sum(e_base_j); target_spring = sum(e_spr_j)

    arms_P = arm_holding_power(cfg); arms_E = arms_P * T
    print(f"=== {cfg.label} (motor {cfg.motor} R/Kt^2={motor.r/motor.kt**2:.2e}, regen={cfg.regen}) ===")
    print(f"vertical lift {cfg.z_top}->{cfg.z_bot} m (level), cycle {cfg.cycle_s}s, {T:.1f}s total, "
          f"upper body {up_mass:.1f} kg; spring on: {', '.join(cfg.spring_joints)}")
    print(f"peak |tau| per joint: {np.round(np.abs(tau).max(0),0)} N*m\n")
    print(f"  {'torso joint':14s} {'spring':>14} {'avg P base(W)':>14} {'avg P spring(W)':>16} {'red':>7}")
    for k, jn in enumerate(TORSO_JOINTS):
        print(f"  {jn:14s} {_spring_str(springs[k]):>14} {e_base_j[k]/T:14.1f} {e_spr_j[k]/T:16.1f} "
              f"{100*(e_spr_j[k]-e_base_j[k])/max(e_base_j[k],1e-9):+6.0f}%")
    print(f"  {'TARGET (3 torso)':14s} {'':9} {target_base/T:14.1f} {target_spring/T:16.1f} "
          f"{100*(target_spring-target_base)/target_base:+6.1f}%")
    ob = target_base + arms_E + cfg.compute_w * T; os_ = target_spring + arms_E + cfg.compute_w * T
    print(f"\n  arms {arms_P:.1f} W; whole-robot @{cfg.compute_w:.0f} W computer: {100*(os_-ob)/ob:+.1f}%")

    # ---- plot: tau vs angle (+ fitted spring where present), per joint ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for k, (ax, jn) in enumerate(zip(axes, TORSO_JOINTS)):
        order = np.argsort(q[:, k])
        ax.plot(q[order, k], tau[order, k], ".", ms=2, color="tab:blue", label="required torque")
        ax.plot(q[order, k], grav[order, k], "-", lw=1, color="gray", alpha=0.7, label="gravity")
        if springs[k] is not None:
            ax.plot(q[order, k], tau_spring(q[order, k], springs[k]), "-", lw=2, color="tab:red",
                    label=f"fitted spring ({_spring_str(springs[k])})")
        else:
            ax.plot([], [], " ", label="no spring")
        ax.set_title(jn); ax.set_xlabel("joint angle (rad)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        if k == 0:
            ax.set_ylabel("torque (N*m)")
    fig.suptitle(f"{cfg.label}: torque vs angle + fitted parallel spring "
                 f"(target {100*(target_spring-target_base)/target_base:+.0f}%)")
    fig.tight_layout()
    pathlib.Path("outputs/gravity_compensation/raw/galaxea").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"outputs/gravity_compensation/raw/galaxea/galaxea_taus{cfg.tag}.png", dpi=130); plt.close(fig)
    print(f"\nWROTE outputs/gravity_compensation/raw/galaxea/galaxea_taus{cfg.tag}.png")

    results = dict(robot=cfg.label, target_label="3 torso joints",
                   duration_s=T, target_base_J=target_base, target_spring_J=target_spring,
                   other_servo_J=arms_E, other_servo_W=arms_P,
                   all_servo_base_J=target_base + arms_E, all_servo_spring_J=target_spring + arms_E,
                   regen=cfg.regen)
    pathlib.Path(f"outputs/gravity_compensation/raw/galaxea/galaxea_results{cfg.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"WROTE outputs/gravity_compensation/raw/galaxea/galaxea_results{cfg.tag}.json")

    if args.video:
        _render(cfg, z, q, tau, qd, springs, motor, target_base, target_spring, args.video)


def _render(cfg, z, q, tau, qd, springs, motor, e_base, e_spring, out):
    W, H = 1280, 720
    m = build_reduced(cfg, with_assets=True); d = mujoco.MjData(m)
    render_util.apply_palette(m, "galaxea")
    # place the floor at the robot's base of support
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    qa = np.array([m.jnt_qposadr[j] for j in range(m.njnt)])
    d.qpos[qa] = q[0]; mujoco.mj_forward(m, d)
    if fid >= 0:
        m.geom_pos[fid, 2] = float(d.geom_xpos[:, 2].min()) - 0.02
    render_util.set_quality(m, W, H)
    # per-step power for the live meter (baseline vs spring), summed over the 3 joints
    def _ts(k):  # spring torque on joint k (0 where no spring)
        return tau_spring(q[:, k], springs[k]) if springs[k] is not None else 0.0
    pb = sum(E.electrical_power(tau[:, k], qd[:, k], motor.kt, motor.r, regen=cfg.regen) for k in range(3))
    ps = sum(E.electrical_power(tau[:, k] - _ts(k), qd[:, k],
             motor.kt, motor.r, regen=cfg.regen) for k in range(3))
    eb = np.cumsum(pb) * cfg.dt; es = np.cumsum(ps) * cfg.dt
    pct = 100 * (e_spring - e_base) / e_base
    renderer = mujoco.Renderer(m, H, W)
    cam = render_util.free_camera(3.0, -10.0, 90.0, lookat=(0, 0, 0.8))
    track = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link4")
    frames = []; stride = max(1, round((1 / 30) / cfg.dt))
    for i in range(0, len(q), stride):
        d.qpos[qa] = q[i]; mujoco.mj_forward(m, d)
        cam.lookat[:] = [0, 0, d.xpos[track][2]]
        renderer.update_scene(d, camera=cam)
        fr = renderer.render().copy()
        lines = ["Galaxea R1  -  coordinated upright lift",
                 f"no spring:    {eb[i]:5.0f} J    {pb[i]:4.0f} W",
                 f"with spring:  {es[i]:5.0f} J    {ps[i]:4.0f} W",
                 f"torso-motor energy  {pct:+.0f}%"]
        cols = [(235, 235, 235), (255, 120, 120), (120, 235, 140), (255, 235, 140)]
        frames.append(render_util.overlay_text(fr, lines, colors=cols))
    render_util.write_mp4(frames, out, 30, W, H)


if __name__ == "__main__":
    main()
