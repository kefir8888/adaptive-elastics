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
        env = SpringWrapper(env, spec)
    return env


class SpringWrapper:
    """Injects tau_spring(theta) at the knee DoFs, in parallel with the motors.

    The torque goes through `qfrc_applied` (external generalized force), NOT
    through the actuators — so `qfrc_actuator` keeps meaning "motor torque"
    and the energy model stays honest. theta is sampled at the control
    boundary and the torque held constant across the substeps of one control
    step (50 Hz; tau(theta) is smooth, the error is negligible).

    Delegates everything else, so Playground's brax training wrapper and
    jit/vmap compose with it transparently.
    """

    def __init__(self, env, spec):
        import jax.numpy as jnp

        self._env = env
        self._spec = spec
        knees = knee_joints(env.mj_model)
        self._knee_qpos_adr = jnp.array([v["qpos_adr"] for v in knees.values()])
        self._knee_dof_adr = jnp.array([v["dof_adr"] for v in knees.values()])

    def step(self, state, action):
        from pea.springs import tau_spring

        theta = state.data.qpos[..., self._knee_qpos_adr]
        tau = tau_spring(theta, self._spec)
        qfrc = state.data.qfrc_applied.at[..., self._knee_dof_adr].set(tau)
        state = state.replace(data=state.data.replace(qfrc_applied=qfrc))
        return self._env.step(state, action)

    def __getattr__(self, name):
        return getattr(self._env, name)


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
