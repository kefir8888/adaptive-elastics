"""Presentation-quality renders: assign a per-part color palette (the URDF meshes carry no
materials into MuJoCo -> everything is flat grey) and render a clean turntable.

  python scripts/pretty_render.py --robot galaxea [--still out.png] [--video out.mp4]
  python scripts/pretty_render.py --robot dog      [--still out.png] [--video out.mp4]
"""
import argparse
import pathlib
import sys

import numpy as np
import mujoco

from pea import render_util
from pea.urdf_loader import fiveages_dir, limx_dir, make_spec

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

GAL_URDF = "humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf"


def _bodyname(m, g):
    return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").lower()


def color_galaxea(m):
    for g in range(m.ngeom):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == "floor":
            continue
        b = _bodyname(m, g)
        if "wheel" in b or "steer" in b:
            c = [0.05, 0.05, 0.06, 1]                 # black wheels/casters
        elif "base_link" in b:
            c = [0.10, 0.10, 0.12, 1]                 # dark mobile base
        elif "torso" in b:
            c = [0.83, 0.84, 0.88, 1]                 # silver/white torso panels
        elif "gripper" in b or "finger" in b:
            c = [0.16, 0.16, 0.18, 1]                 # dark grippers
        elif "head" in b:
            c = [0.04, 0.04, 0.05, 1]                 # black head
        elif "link" in b or "base" in b:
            c = [0.11, 0.11, 0.13, 1]                 # black arms
        else:
            c = [0.30, 0.30, 0.32, 1]
        m.geom_rgba[g] = c


def color_dog(m):
    for g in range(m.ngeom):
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) == "floor":
            continue
        b = _bodyname(m, g)
        if "wheel" in b:
            c = [0.05, 0.05, 0.06, 1]                 # black wheels
        elif "base" in b or "imu" in b:
            c = [0.12, 0.12, 0.14, 1]                 # dark body
        elif "calf" in b or "thigh" in b or "hip" in b:
            c = [0.85, 0.40, 0.05, 1]                 # bright metallic orange legs
        else:
            c = [0.20, 0.20, 0.22, 1]
        m.geom_rgba[g] = c


def build_galaxea():
    spec = make_spec(fiveages_dir() / GAL_URDF)
    render_util.add_scene_assets(spec, floor=True)
    m = spec.compile(); d = mujoco.MjData(m)
    # arms tucked near the body (shoulders in), torso upright
    def setj(name, v):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = v
    for s in ("left", "right"):
        setj(f"{s}_joint2", 0.35); setj(f"{s}_joint3", -0.55); setj(f"{s}_joint4", 0.0)
    color_galaxea(m)
    mujoco.mj_forward(m, d)
    return m, d, (0, 0, 0.95)


def build_dog():
    import limx_roll as L
    cfg = L.Cfg()
    m = L.build_model(cfg, with_assets=True); d = mujoco.MjData(m)
    stance = L.find_stance(m, mujoco.MjData(m), target_drop=0.30)
    d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 1.0
    for leg in L.LEGS:
        ha, hf, kf = stance[leg]
        d.qpos[L.jqp(m, leg + "_HAA")] = ha; d.qpos[L.jqp(m, leg + "_HFE")] = hf; d.qpos[L.jqp(m, leg + "_KFE")] = kf
    mujoco.mj_forward(m, d)
    wz = min(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg + "_wheel")][2] for leg in L.LEGS)
    d.qpos[2] = 1.0 - (wz - cfg.wheel_radius)
    color_dog(m)
    mujoco.mj_forward(m, d)
    return m, d, (0, 0, 0.30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=["galaxea", "dog"])
    ap.add_argument("--still", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    m, d, lookat = build_galaxea() if args.robot == "galaxea" else build_dog()
    W, H = 1280, 720
    render_util.set_quality(m, W, H, offsamples=8, shadowsize=8192)
    renderer = mujoco.Renderer(m, H, W)
    dist = 2.5 if args.robot == "galaxea" else 1.9
    cam = render_util.free_camera(dist, -10.0, 120.0, lookat=lookat)
    if args.still:
        renderer.update_scene(d, camera=cam)
        from PIL import Image
        Image.fromarray(renderer.render()).save(args.still)
        print("WROTE", args.still)
    if args.video:
        frames = []
        for k in range(180):                          # 6 s turntable at 30 fps
            cam.azimuth = 120 + 360 * k / 180
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        render_util.write_mp4(frames, args.video, 30, W, H)


if __name__ == "__main__":
    main()
