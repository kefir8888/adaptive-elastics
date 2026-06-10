"""G1 walking environment, loaded from MuJoCo Playground's registry.

The spring is config-selected here (never baked into train.py): kind='none'
returns the stock env; spring kinds will wrap the env to inject tau_spring at
the knee DOFs (Milestone 2).
"""

from __future__ import annotations

import mujoco
from mujoco_playground import registry

from pea import springs
from pea.config import RunConfig


def make_env(cfg: RunConfig):
    env = registry.load(cfg.env_name, config_overrides={"impl": cfg.impl})
    spec = springs.from_config(cfg.spring)
    if spec is not None:
        raise NotImplementedError(
            "in-sim spring injection lands in Milestone 2; "
            "use spring.kind='none' for now"
        )
    return env


def knee_joints(mj_model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """Locate knee joints by name; returns {joint_name: {id, qpos_adr, dof_adr}}.

    Matched by substring so it survives Menagerie naming tweaks
    (e.g. 'left_knee_joint' / 'right_knee_joint').
    """
    found = {}
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and "knee" in name:
            found[name] = {
                "id": j,
                "qpos_adr": int(mj_model.jnt_qposadr[j]),
                "dof_adr": int(mj_model.jnt_dofadr[j]),
            }
    if not found:
        raise RuntimeError("no knee joints found in model")
    return found
