"""Galaxea R1 'free' torso reaching motions (5 options) with a parallel LINEAR spring on the
BOTTOM torso joint (torso_joint1) + electrical-energy compensation.

Each option is a smooth cyclic coordinated torso motion (direct joint schedules, no IK ->
robust): the body leans forward (bottom joint sweeps -> the spring target) plus a different
'free' component (knee bend, upward extension, yaw twist, ...), as if reaching for something.
For each: fit the energy-optimal linear spring tau=-k(theta-theta0) on the bottom joint,
compute its electrical energy with/without the spring, render a video with a live energy
meter, and plot the bottom-joint torque-vs-angle + fitted spring.

  python scripts/galaxea_free.py --option <name> [--video out.mp4]
  options: forward, reach_down, reach_up, side_twist, free_scan
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from pea import energy as E, gravcomp, render_util
from pea.springs import tau_spring
from pea.urdf_loader import fiveages_dir, make_spec

TORSO = ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]
BOTTOM = 0   # torso_joint1
NEUTRAL = np.array([0.10, 0.0, 0.0, 0.0])

# 5 reaching options: per-joint sweep amplitude [j1(pitch,bottom), j2(pitch,knee), j3(pitch), j4(yaw)]
OPTIONS = {
    "forward":    dict(amps=[0.55, 0.00, 0.00, 0.00], label="forward lean-reach"),
    "reach_down": dict(amps=[0.50, 0.65, 0.00, 0.00], label="reach down (bend + lean)"),
    "reach_up":   dict(amps=[0.45, -0.55, 0.40, 0.00], label="reach up (extend up + forward)"),
    "side_twist": dict(amps=[0.45, 0.00, 0.00, 0.85], label="side reach (lean + yaw twist)"),
    "free_scan":  dict(amps=[0.50, 0.35, -0.25, 0.65], label="free scan reach (combined)"),
}
DT, MOVE_S, DWELL_S, NCYC = 0.004, 4.0, 0.8, 3


def build_reduced(with_assets):
    spec = make_spec(fiveages_dir() / "humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf",
                     compiler_attrs='fusestatic="false" discardvisual="false"')
    for j in list(spec.joints):
        if j.name not in set(TORSO):
            spec.delete(j)
    if with_assets:
        render_util.add_scene_assets(spec, floor=True)
    return spec.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--option", required=True, choices=list(OPTIONS))
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    opt = OPTIONS[args.option]; amps = np.array(opt["amps"])
    motor = E.motor_constants("galaxea_torso")

    m = build_reduced(False); d = mujoco.MjData(m)
    s, _, _ = gravcomp.minjerk_cycle(0.0, 1.0, DWELL_S, MOVE_S, NCYC, DT)
    q = NEUTRAL[None, :] + amps[None, :] * s[:, None]        # (N, 4) coordinated motion
    tau, grav, qd = gravcomp.multi_joint_torque(m, d, q, DT)
    N = len(q); T = N * DT

    ba, bt, bw = q[:, BOTTOM], tau[:, BOTTOM], qd[:, BOTTOM]   # bottom-joint angle/torque/omega
    spring, e_spring, e_base = gravcomp.fit_linear_spring_per_joint(ba, bt, bw, motor, DT, regen=False)
    base = gravcomp.lift_energy(bt, bw, motor, DT, False)
    withs = gravcomp.motor_energy_with_spring(bt, ba, bw, spring, motor, DT, False)
    red = 100 * (withs.total - base.total) / base.total if base.total > 1e-9 else 0.0

    print(f"=== Galaxea R1 free reach: {opt['label']} ({NCYC}x, {T:.1f}s) ===")
    print(f"joint amplitudes (rad): {dict(zip(TORSO, amps))}")
    print(f"bottom joint (torso_joint1): angle [{ba.min():+.2f},{ba.max():+.2f}] rad, "
          f"torque [{bt.min():.0f},{bt.max():.0f}] N*m")
    print(f"best linear spring: k={spring.k:.0f} N*m/rad theta0={spring.theta0:.2f} rad")
    print(f"bottom-joint electrical energy: no spring {base.total:.0f} J ({base.total/T:.1f} W) -> "
          f"with spring {withs.total:.0f} J ({withs.total/T:.1f} W)  ({red:+.0f}%)")

    out_dir = pathlib.Path("outputs/gravity_compensation/raw/galaxea_free"); out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = np.argsort(ba)
    ax.plot(ba, bt, ".", ms=2.5, color="tab:blue", label="required motor torque")
    ax.plot(ba[order], grav[order, BOTTOM], "-", lw=1.3, color="gray", label="gravity torque")
    ax.plot(ba[order], tau_spring(ba[order], spring), "-", lw=2, color="tab:red",
            label=f"fitted linear spring (k={spring.k:.0f})")
    ax.set_xlabel("bottom joint (torso_joint1) angle (rad)"); ax.set_ylabel("torque (N*m)")
    ax.set_title(f"Galaxea free reach: {opt['label']} — bottom-joint torque vs angle ({red:+.0f}%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / f"{args.option}_torque_angle.png", dpi=130); plt.close(fig)
    print(f"WROTE {out_dir}/{args.option}_torque_angle.png")

    if args.video:
        _render(args.option, opt, q, ba, bt, bw, spring, motor, base.total, withs.total, red, args.video)


def _render(name, opt, q, ba, bt, bw, spring, motor, e_base, e_spring, red, out):
    W, H = 960, 540
    m = build_reduced(True); d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[j] for j in range(m.njnt)])
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[qa] = q[0]; mujoco.mj_forward(m, d)
    if fid >= 0:
        m.geom_pos[fid, 2] = float(d.geom_xpos[:, 2].min()) - 0.02
    render_util.set_quality(m, W, H)
    pb = E.electrical_power(bt, bw, motor.kt, motor.r, regen=False)
    ps = E.electrical_power(bt - tau_spring(ba, spring), bw, motor.kt, motor.r, regen=False)
    eb = np.cumsum(pb) * DT; es = np.cumsum(ps) * DT
    renderer = mujoco.Renderer(m, H, W)
    cam = render_util.free_camera(3.3, -12.0, 115.0, lookat=(0.05, 0, 0.9))
    track = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link4")
    frames = []; stride = max(1, round((1 / 30) / DT))
    for i in range(0, len(q), stride):
        d.qpos[qa] = q[i]; mujoco.mj_forward(m, d)
        cam.lookat[:] = [0.05, 0, d.xpos[track][2]]
        renderer.update_scene(d, camera=cam)
        fr = renderer.render().copy()
        lines = [f"Galaxea R1  -  {opt['label']}  (spring on bottom joint)",
                 f"no spring:    {eb[i]:5.0f} J    {pb[i]:4.0f} W",
                 f"with spring:  {es[i]:5.0f} J    {ps[i]:4.0f} W",
                 f"bottom-joint energy  {red:+.0f}%"]
        cols = [(235, 235, 235), (255, 120, 120), (120, 235, 140), (255, 235, 140)]
        frames.append(render_util.overlay_text(fr, lines, size=20, colors=cols))
    render_util.write_mp4(frames, out, 30, W, H)


if __name__ == "__main__":
    main()
