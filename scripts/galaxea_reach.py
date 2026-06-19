"""Galaxea R1 cyclic forward-reach + slight incline (slow, safe, sustainable torso motion).

The upper body reaches FORWARD (~0.2 m at roughly constant height) and inclines slightly,
then returns — repeated n_cycles. Slow minimum-jerk trajectory: low velocity/acceleration,
joints kept well inside limits. Collects the torque-vs-angle characteristic of the "knee"
(the middle torso joint, torso_joint2) and renders a video.

  python scripts/galaxea_reach.py [--config configs/galaxea_reach.yaml] [--video out.mp4]
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
from pea.urdf_loader import fiveages_dir, make_spec

TORSO_JOINTS = ["torso_joint1", "torso_joint2", "torso_joint3"]
KNEE = 1   # torso_joint2 = the "knee" (middle torso joint)


@dataclasses.dataclass
class Cfg:
    urdf: str = "humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf"
    neutral_q: tuple = (0.2, -0.4, -0.2)   # neutral torso pose (slightly bent, body ~vertical)
    reach_dx: float = 0.14                 # m forward reach (kept so the knee stays in its
                                           # smooth range -- bigger reach saturates the knee at 0)
    incline_rad: float = 0.10              # forward incline at full reach (~6 deg)
    move_s: float = 5.0                    # s each way (slow -> safe, low inertia)
    dwell_s: float = 1.0
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
    qa = np.array([m.jnt_qposadr[j] for j in range(m.njnt)])
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link4")
    d.qpos[qa] = cfg.neutral_q; mujoco.mj_forward(m, d)
    R = d.xmat[tb].reshape(3, 3)
    x0, z0, pit0 = d.xpos[tb][0], d.xpos[tb][2], float(np.arctan2(-R[2, 0], R[0, 0]))

    # cyclic reach parameter s(t): 0(neutral) -> 1(reached) -> 0, repeated; min-jerk (smooth).
    # Solve IK on a coarse s-grid then CUBIC-SPLINE-interpolate q(s) -> q(t): this yields a
    # C2-smooth joint trajectory (per-timestep IK jitter would otherwise inject torque spikes).
    from scipy.interpolate import CubicSpline
    s, _, _ = gravcomp.minjerk_cycle(0.0, 1.0, cfg.dwell_s, cfg.move_s, cfg.n_cycles, cfg.dt)
    s_grid = np.linspace(0.0, 1.0, 81)
    q_grid = gravcomp.track_ik(m, d, "torso_link4", x0 + cfg.reach_dx * s_grid,
                               np.full_like(s_grid, z0), pit0 + cfg.incline_rad * s_grid,
                               q_init=cfg.neutral_q)
    q = CubicSpline(s_grid, q_grid, axis=0)(s)
    tau, grav, qd = gravcomp.multi_joint_torque(m, d, q, cfg.dt)
    N = len(q); T = N * cfg.dt

    knee_ang, knee_tau, knee_w = q[:, KNEE], tau[:, KNEE], qd[:, KNEE]
    print(f"=== Galaxea R1 cyclic forward-reach + incline ({cfg.n_cycles}x, {T:.1f}s) ===")
    print(f"reach {cfg.reach_dx} m forward, incline {cfg.incline_rad} rad, {cfg.move_s}s each way; top z~{z0:.2f} m")
    print(f"knee (torso_joint2): angle [{knee_ang.min():+.2f},{knee_ang.max():+.2f}] rad, "
          f"torque [{knee_tau.min():.1f},{knee_tau.max():.1f}] N*m, peak |omega| {np.abs(knee_w).max():.2f} rad/s")
    print(f"peak |tau| all torso joints {np.round(np.abs(tau).max(0),0)} N*m (slow -> gravity-dominated, safe)")

    # ---- torque-vs-angle characteristic for the knee ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(knee_ang, knee_tau, ".", ms=2.5, color="tab:blue", label="required motor torque (3 cycles)")
    order = np.argsort(knee_ang)
    ax.plot(knee_ang[order], grav[order, KNEE], "-", lw=1.5, color="gray", label="gravity torque")
    ax.set_xlabel("knee (torso_joint2) angle (rad)"); ax.set_ylabel("knee motor torque (N*m)")
    ax.set_title(f"Galaxea R1 forward-reach — knee (torso_joint2) torque vs angle ({cfg.n_cycles} cycles)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    out_dir = pathlib.Path("outputs/gravity_compensation/raw/galaxea_reach"); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "reach_knee_torque_angle.png", dpi=130); plt.close(fig)
    print(f"WROTE {out_dir}/reach_knee_torque_angle.png")
    np.savez(out_dir / "reach_knee.npz", angle=knee_ang, torque=knee_tau, omega=knee_w, t=np.arange(N) * cfg.dt)

    if args.video:
        _render(cfg, q, knee_ang, knee_tau, args.video)


def _render(cfg, q, knee_ang, knee_tau, out):
    W, H = 960, 540                                   # lower res (gifs are heavy)
    m = build_reduced(cfg, with_assets=True); d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[j] for j in range(m.njnt)])
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[qa] = q[0]; mujoco.mj_forward(m, d)
    if fid >= 0:
        m.geom_pos[fid, 2] = float(d.geom_xpos[:, 2].min()) - 0.02
    render_util.set_quality(m, W, H)
    renderer = mujoco.Renderer(m, H, W)
    cam = render_util.free_camera(3.2, -8.0, 90.0, lookat=(0.1, 0, 0.9))   # side view shows the reach
    frames = []; stride = max(1, round((1 / 30) / cfg.dt))
    for i in range(0, len(q), stride):
        d.qpos[qa] = q[i]; mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        fr = renderer.render().copy()
        lines = ["Galaxea R1  -  cyclic forward reach + incline",
                 f"knee (torso_joint2) angle {knee_ang[i]:+.2f} rad",
                 f"knee torque {knee_tau[i]:+.0f} N*m"]
        frames.append(render_util.overlay_text(fr, lines, size=20,
                      colors=[(235, 235, 235), (150, 220, 255), (255, 220, 150)]))
    render_util.write_mp4(frames, out, 30, W, H)


if __name__ == "__main__":
    main()
