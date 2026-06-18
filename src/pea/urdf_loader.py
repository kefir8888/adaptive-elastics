"""Load external ROS/URDF robot descriptions into MuJoCo (CPU).

The gravity-compensation direction (2026-06-18) uses robots whose descriptions ship
as ROS2 URDF packages, not MuJoCo MJCF: the LimX W1 wheeled quadruped and the fiveages
mobile-manipulator humanoids (Galaxea R1, Airbot MMK2, ...). Two things must be fixed
before MuJoCo can load them:
  1. ROS `package://<pkg>/...` mesh URIs  ->  absolute paths.
  2. `.glb` visual meshes (MuJoCo decodes only STL/OBJ/MSH) -> `.obj`
     (scripts/setup_robots.sh pre-converts every .glb with trimesh).

The descriptions live in a gitignored, regenerable tree (default <repo>/external,
override with $PEA_ROBOTS); see scripts/setup_robots.sh.
"""
from __future__ import annotations

import functools
import os
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


def robots_root() -> pathlib.Path:
    """Root holding the cloned descriptions ($PEA_ROBOTS or <repo>/external)."""
    if env := os.environ.get("PEA_ROBOTS"):
        return pathlib.Path(env)
    return pathlib.Path(__file__).resolve().parents[2] / "external"


# Convenience paths to the two description trees set up by scripts/setup_robots.sh.
def fiveages_dir() -> pathlib.Path:
    return robots_root() / "fiveages_robot_descriptions"


def limx_dir() -> pathlib.Path:
    return robots_root() / "limx" / "robot-description"


@functools.lru_cache(maxsize=None)
def _pkg_index(search_root: str) -> dict[str, str]:
    """Map ROS package name -> directory, by scanning package.xml under search_root.
    Falls back to directory basename when a package.xml lacks/﹣mismatches its <name>."""
    index: dict[str, str] = {}
    root = pathlib.Path(search_root)
    for pkgxml in root.rglob("package.xml"):
        d = str(pkgxml.parent)
        index.setdefault(pkgxml.parent.name, d)  # by dir name
        try:
            name = ET.parse(pkgxml).getroot().findtext("name")
            if name:
                index.setdefault(name.strip(), d)  # by declared <name>
        except ET.ParseError:
            pass
    return index


def resolve_urdf(urdf_path, search_root=None) -> pathlib.Path:
    """Rewrite package:// + .glb in a URDF and return a MuJoCo-loadable .mjfixed.urdf.

    search_root: where to look up package:// names (defaults to the URDF's own tree
    under robots_root()). The fixed file is written next to the original.
    """
    urdf_path = pathlib.Path(urdf_path)
    txt = urdf_path.read_text()
    search_root = str(search_root) if search_root else _owning_tree(urdf_path)
    index = _pkg_index(search_root)
    missing = []
    for pkg in sorted(set(re.findall(r"package://([^/]+)/", txt))):
        d = index.get(pkg)
        if d is None:
            missing.append(pkg)
        else:
            txt = txt.replace(f"package://{pkg}/", d + "/")
    if missing:
        raise FileNotFoundError(
            f"unresolved ROS packages {missing} for {urdf_path.name}; "
            f"run scripts/setup_robots.sh (search_root={search_root})"
        )
    # MuJoCo can't decode GLB; setup_robots.sh pre-converted every .glb -> .obj.
    txt = txt.replace(".glb", ".obj").replace(".GLB", ".obj")
    fixed = urdf_path.with_suffix(".mjfixed.urdf")
    fixed.write_text(txt)
    return fixed


def _owning_tree(urdf_path: pathlib.Path) -> str:
    """The description tree a URDF belongs to (so package:// lookups stay scoped)."""
    p = urdf_path.resolve()
    root = robots_root().resolve()
    for parent in p.parents:
        if parent.parent == root:   # e.g. <external>/limx, <external>/fiveages_...
            return str(parent)
    return str(root)


def load_model(urdf_path, search_root=None, compiler_attrs="") -> mujoco.MjModel:
    """Resolve + load a URDF as an MjModel. compiler_attrs is injected into a
    <mujoco><compiler .../></mujoco> block (e.g. 'fusestatic="false"')."""
    fixed = resolve_urdf(urdf_path, search_root)
    if compiler_attrs:
        txt = fixed.read_text()
        block = f"<mujoco><compiler {compiler_attrs}/></mujoco>"
        txt = re.sub(r"(<robot\b[^>]*>)", r"\1\n  " + block, txt, count=1)
        fixed.write_text(txt)
    return mujoco.MjModel.from_xml_path(str(fixed))


def make_spec(urdf_path, search_root=None,
              compiler_attrs='fusestatic="false" discardvisual="false"'):
    """Resolve + return a mujoco.MjSpec (editable: add freejoint/ground/actuators).
    discardvisual=false is REQUIRED to keep visual meshes AND any added skybox/floor
    textures (URDF import defaults discardvisual=true, which drops them at compile)."""
    fixed = resolve_urdf(urdf_path, search_root)
    if compiler_attrs:
        txt = fixed.read_text()
        block = f"<mujoco><compiler {compiler_attrs}/></mujoco>"
        txt = re.sub(r"(<robot\b[^>]*>)", r"\1\n  " + block, txt, count=1)
        fixed.write_text(txt)
    return mujoco.MjSpec.from_file(str(fixed))


def subtree_mass(model: mujoco.MjModel, body_id: int) -> float:
    """Total mass of body_id and all its descendants (the load below a joint)."""
    total, stack = 0.0, [body_id]
    while stack:
        b = stack.pop()
        total += float(model.body_mass[b])
        stack += [c for c in range(model.nbody)
                  if model.body_parentid[c] == b and c != b]
    return total
