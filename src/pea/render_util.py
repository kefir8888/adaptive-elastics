"""Offscreen MuJoCo video helpers (gravity-comp / wheeled-robot experiments).

MuJoCo's built-in OpenGL renderer -> RGB frames -> ffmpeg (H.264). Used by
scripts/galaxea_lift.py and scripts/limx_roll.py to record the prescribed motion.
"""
from __future__ import annotations

import subprocess

import numpy as np
import mujoco


def add_scene_assets(spec, floor: bool = True) -> None:
    """Add a gradient skybox, a checker floor (optional), and even directional
    lighting to an MjSpec, so renders look clean at any robot position."""
    sky = spec.add_texture()
    sky.name = "sky"; sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.58, 0.72, 0.92]; sky.rgb2 = [0.18, 0.26, 0.42]  # horizon -> zenith
    sky.width = 512; sky.height = 512
    if floor:
        gtx = spec.add_texture()
        gtx.name = "grid"; gtx.type = mujoco.mjtTexture.mjTEXTURE_2D
        gtx.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        gtx.rgb1 = [0.26, 0.30, 0.35]; gtx.rgb2 = [0.34, 0.39, 0.45]
        gtx.width = 512; gtx.height = 512
        gmat = spec.add_material()
        gmat.name = "grid"; gmat.texrepeat = [8, 8]; gmat.reflectance = 0.15
        gmat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"  # int index, not enum
    if floor:
        fl = spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
            rgba=[0.5, 0.55, 0.6, 1], friction=[1.0, 0.01, 0.001])
        try:
            fl.material = "grid"
        except Exception:
            pass
    sun = spec.worldbody.add_light(pos=[2, 2, 4], dir=[-0.3, -0.3, -1])
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    sun.diffuse = [0.85, 0.85, 0.85]; sun.specular = [0.3, 0.3, 0.3]; sun.castshadow = True
    fill = spec.worldbody.add_light(pos=[-2, -1, 3], dir=[0.3, 0.2, -1])
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.diffuse = [0.3, 0.3, 0.35]; fill.castshadow = False


def set_quality(model, width: int, height: int, offsamples: int = 8,
                shadowsize: int = 8192) -> None:
    """High-quality offscreen settings; MUST be called before mujoco.Renderer."""
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    model.vis.quality.offsamples = offsamples
    model.vis.quality.shadowsize = shadowsize
    model.vis.headlight.ambient = [0.35, 0.35, 0.40]
    model.vis.headlight.diffuse = [0.45, 0.45, 0.45]
    model.vis.headlight.specular = [0.2, 0.2, 0.2]


def free_camera(distance: float, elevation: float, azimuth: float, lookat=(0, 0, 0)):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    cam.lookat[:] = lookat
    return cam


_FONT_PATHS = ["/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/Supplemental/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def overlay_text(frame, lines, org=(22, 18), size=24, line_gap=8,
                 colors=None, box=True):
    """Draw text lines on an RGB frame (numpy uint8). `lines` is a list of strings;
    `colors` an optional list of (r,g,b). Returns a new annotated array."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(np.ascontiguousarray(frame, np.uint8))
    draw = ImageDraw.Draw(img)
    font = None
    for fp in _FONT_PATHS:
        try:
            font = ImageFont.truetype(fp, size); break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    colors = colors or [(255, 255, 255)] * len(lines)
    if box:  # translucent dark panel behind the text for legibility
        h = org[1] * 2 + len(lines) * (size + line_gap)
        w = max(draw.textlength(s, font=font) for s in lines) + org[0] * 2
        panel = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 120))
        img.paste(Image.new("RGB", panel.size, (15, 15, 20)),
                  (org[0] - 12, org[1] - 12), panel)
    y = org[1]
    for s, c in zip(lines, colors):
        draw.text((org[0], y), s, fill=tuple(c), font=font)
        y += size + line_gap
    return np.asarray(img)


def write_mp4(frames, path: str, fps: int, width: int, height: int) -> None:
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path],
        stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    print(f"WROTE {path}  ({len(frames)} frames)")
