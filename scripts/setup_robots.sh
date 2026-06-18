#!/usr/bin/env bash
# Fetch + prepare the external robot descriptions for the gravity-compensation direction.
# Idempotent. Populates a gitignored tree (default <repo>/external, or $PEA_ROBOTS).
#   - fiveages mobile-manipulator humanoids (Galaxea R1, Airbot MMK2, ...)  [URDF/ROS2]
#   - LimX W1 wheeled quadruped (WL_P311D)                                   [URDF/ROS2]
#   - the fiveages `common` submodule (camera meshes) via HTTPS (the .gitmodules uses SSH)
#   - converts every .glb visual mesh -> .obj (MuJoCo decodes only STL/OBJ/MSH)
# Usage: scripts/setup_robots.sh
set -euo pipefail

ROOT="${PEA_ROBOTS:-$(cd "$(dirname "$0")/.." && pwd)/external}"
PY="${PEA_PY:-env -u PYTHONPATH $(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python}"
mkdir -p "$ROOT"
echo "robots root: $ROOT"

FA="$ROOT/fiveages_robot_descriptions"
if [ ! -d "$FA/.git" ]; then
  git clone --depth 1 https://github.com/fiveages-sim/robot_descriptions.git "$FA"
fi
# `common` submodule (sensor/camera meshes) — .gitmodules uses git@ SSH; clone via HTTPS
if [ -z "$(ls -A "$FA/common" 2>/dev/null)" ]; then
  rmdir "$FA/common" 2>/dev/null || true
  git clone --depth 1 https://github.com/fiveages-sim/robot-descriptions-common.git "$FA/common"
fi

LIMX="$ROOT/limx/robot-description"
if [ ! -d "$LIMX/.git" ]; then
  git clone --depth 1 https://github.com/limxdynamics/robot-description.git "$LIMX"
fi

echo "converting .glb -> .obj (trimesh) ..."
$PY - "$ROOT" <<'PYEOF'
import glob, os, sys, trimesh
root = sys.argv[1]
files = glob.glob(root + "/**/*.glb", recursive=True) + glob.glob(root + "/**/*.GLB", recursive=True)
ok = skip = fail = 0
for f in files:
    obj = os.path.splitext(f)[0] + ".obj"
    if os.path.exists(obj):
        skip += 1; continue
    try:
        m = trimesh.load(f, force="mesh"); m.export(obj); ok += 1
    except Exception as e:
        fail += 1; print("  FAIL", f, e)
print(f"glb->obj: converted={ok} skipped={skip} failed={fail}")
PYEOF
echo "done. key models:"
ls "$FA/humanoid/Galaxea/galaxea_r1_description/urdf/galaxea_r1.urdf"
ls "$LIMX/wheellegged/WL_P311D/urdf/robot.urdf"
