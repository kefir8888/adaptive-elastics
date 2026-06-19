"""Galaxea R1 cyclic forward LEAN about the BOTTOM torso joint (torso_joint1), with a
parallel LINEAR torsion spring fitted to that joint.

The upper body (~36 kg) leans forward and back about the bottom joint (a slow, safe waist
lean); the bottom joint carries the leaning body, so its gravity torque tracks its angle
-- a clean target for a linear spring tau = -k (theta - theta0). We fit the energy-optimal
(k, theta0), then report/visualize the electrical-energy saving (no spring vs spring) with
a live meter over the video, plus the torque-vs-angle + fitted-spring plot.

  python scripts/galaxea_lean.py [--config configs/galaxea_lean.yaml] [--video out.mp4]
"""
import argparse
import dataclasses
import pathlib

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from pea import energy as E, gravcomp, render_util
from pea.springs import tau_spring
from pea.urdf_loader import fiveages_dir, make_spec

TORSO_JOINTS = ["torso_joint1", "torso_joint2", "torso_joint3"]
JOINT = "torso_joint1"   # the BOTTOM torso joint (base/waist) -- carries the leaning body


@dataclasses.dataclass
class Cfg:
    urdf: str = "humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf"
    theta_up: float = 0.05     # rad, near-upright
    theta_fwd: float = 0.65    # rad, leaned forward (~37 deg)
    move_s: float = 4.0        # s each way (slow, safe)
    dwell_s: float = 0.8
    n_cycles: int = 3
    dt: float = 0.004
    motor: str = "galaxea_torso"
    regen: bool = False


def load_cfg(path):
    cfg = Cfg()
    if path:
        for k, v in (yaml.safe_load(pathlib.Path(path).read_text()) or {}).items():
            setattr(cfg, k, v)
    return cfg


def build_reduced(cfg, with_assets):
    spec = make_spec(fiveages_dir() / cfg.urdf,
                     compiler_attrs='fusestatic="false" discardvisual="false"')
    for j in list(spec.joints):
        if j.name not in set(TORSO_JOINTS):
            spec.delete(j)
    if with_assets:
        render_util.add_scene_assets(spec, floor=True)
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    m = build_reduced(cfg, with_assets=False); d = mujoco.MjData(m)
    motor = E.motor_constants(cfg.motor)
    # hold the upper body rigid (torso_joint2/3 = 0 -> straight column) while joint1 leans
    qnom = np.zeros(m.nv)

    theta, omega, alpha = gravcomp.minjerk_cycle(cfg.theta_up, cfg.theta_fwd,
                                                 cfg.dwell_s, cfg.move_s, cfg.n_cycles, cfg.dt)
    tau, grav, _ = gravcomp.rigid_lift_torque(m, d, JOINT, theta, omega, alpha, qnom=qnom)
    N = len(theta); T = N * cfg.dt

    spec_spring, e_spring, e_base = gravcomp.fit_linear_spring_per_joint(
        theta, tau, omega, motor, cfg.dt, cfg.regen)
    base = gravcomp.lift_energy(tau, omega, motor, cfg.dt, cfg.regen)
    with_spring = gravcomp.motor_energy_with_spring(tau, theta, omega, spec_spring, motor, cfg.dt, cfg.regen)

    print(f"=== Galaxea R1 forward LEAN about bottom joint {JOINT} ({cfg.n_cycles}x, {T:.1f}s) ===")
    print(f"lean {cfg.theta_up}->{cfg.theta_fwd} rad, {cfg.move_s}s each way; peak |tau| {np.abs(tau).max():.0f} N*m")
    print(f"best linear spring: k={spec_spring.k:.0f} N*m/rad, theta0={spec_spring.theta0:.2f} rad")
    print(f"bottom-joint electrical energy: no spring {base.total:.0f} J ({base.total/T:.1f} W) -> "
          f"with spring {with_spring.total:.0f} J ({with_spring.total/T:.1f} W)  "
          f"({100*(with_spring.total-base.total)/base.total:+.0f}%)")

    # ---- torque-vs-angle + fitted spring ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = np.argsort(theta)
    ax.plot(theta, tau, ".", ms=2.5, color="tab:blue", label="required motor torque (3 cycles)")
    ax.plot(theta[order], grav[order], "-", lw=1.5, color="gray", label="gravity torque")
    ax.plot(theta[order], tau_spring(theta[order], spec_spring), "-", lw=2, color="tab:red",
            label=f"fitted linear spring (k={spec_spring.k:.0f}, theta0={spec_spring.theta0:.2f})")
    ax.set_xlabel(f"bottom joint ({JOINT}) angle (rad)"); ax.set_ylabel("torque (N*m)")
    ax.set_title(f"Galaxea R1 forward lean — bottom-joint torque vs angle + fitted linear spring "
                 f"(energy {100*(with_spring.total-base.total)/base.total:+.0f}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    out_dir = pathlib.Path("outputs/gravity_compensation/raw/galaxea_lean"); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "lean_torque_angle.png", dpi=130); plt.close(fig)
    print(f"WROTE {out_dir}/lean_torque_angle.png")

    if args.video:
        _render(cfg, theta, omega, tau, spec_spring, motor, base.total, with_spring.total, args.video)


def _render(cfg, theta, omega, tau, spring, motor, e_base, e_spring, out):
    W, H = 960, 540
    m = build_reduced(cfg, with_assets=True); d = mujoco.MjData(m)
    render_util.apply_palette(m, "galaxea")
    qa = np.array([m.jnt_qposadr[j] for j in range(m.njnt)])
    qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT)]
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[qa] = 0.0; d.qpos[qadr] = theta[0]; mujoco.mj_forward(m, d)
    if fid >= 0:
        m.geom_pos[fid, 2] = float(d.geom_xpos[:, 2].min()) - 0.02
    render_util.set_quality(m, W, H)
    pb = E.electrical_power(tau, omega, motor.kt, motor.r, regen=cfg.regen)
    ps = E.electrical_power(tau - tau_spring(theta, spring), omega, motor.kt, motor.r, regen=cfg.regen)
    eb = np.cumsum(pb) * cfg.dt; es = np.cumsum(ps) * cfg.dt
    pct = 100 * (e_spring - e_base) / e_base
    renderer = mujoco.Renderer(m, H, W)
    cam = render_util.free_camera(3.2, -8.0, 90.0, lookat=(0.1, 0, 0.9))
    track = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link4")
    frames = []; stride = max(1, round((1 / 30) / cfg.dt))
    for i in range(0, len(theta), stride):
        d.qpos[qa] = 0.0; d.qpos[qadr] = theta[i]; mujoco.mj_forward(m, d)
        cam.lookat[:] = [0.1, 0, d.xpos[track][2]]
        renderer.update_scene(d, camera=cam)
        fr = renderer.render().copy()
        lines = ["Galaxea R1  -  forward lean (spring on bottom joint)",
                 f"no spring:    {eb[i]:5.0f} J    {pb[i]:4.0f} W",
                 f"with spring:  {es[i]:5.0f} J    {ps[i]:4.0f} W",
                 f"bottom-joint energy  {pct:+.0f}%"]
        cols = [(235, 235, 235), (255, 120, 120), (120, 235, 140), (255, 235, 140)]
        frames.append(render_util.overlay_text(fr, lines, size=20, colors=cols))
    render_util.write_mp4(frames, out, 30, W, H)


if __name__ == "__main__":
    main()
