"""G1 walking environment, loaded from MuJoCo Playground's registry.

The spring is config-selected here (never baked into train.py): kind='none'
returns the stock env; any other kind wraps the env to inject tau_spring at the
target joint's DoFs (knee or hip_pitch, set by cfg.spring.joint). The env's
reward weights can also be overridden via cfg.reward_scales (e.g. to enable the
energy term, which the default G1 reward leaves at zero).
"""

from __future__ import annotations

import mujoco
from mujoco_playground import registry

from pea import springs
from pea.config import RunConfig


def make_env(cfg: RunConfig):
    env_cfg = registry.get_default_config(cfg.env_name)
    env_cfg.impl = cfg.impl  # 'jax', not the broken Warp default
    for key, value in cfg.reward_scales.items():
        env_cfg.reward_config.scales[key] = value
    env = registry.load(cfg.env_name, config=env_cfg)
    spec = springs.from_config(cfg.spring)
    if spec is not None:
        env = SpringWrapper(env, spec, cfg.spring.joint)
    return env


class SpringWrapper:
    """Injects tau_spring(theta) at the target joint's DoFs, in parallel with
    the motors.

    The torque goes through `qfrc_applied` (external generalized force), NOT
    through the actuators — so `qfrc_actuator` keeps meaning "motor torque" and
    the energy model stays honest. theta is sampled at the control boundary and
    the torque held constant across the substeps of one control step (50 Hz;
    tau(theta) is smooth, the error is negligible).

    Delegates everything else, so Playground's brax training wrapper and
    jit/vmap compose with it transparently.
    """

    def __init__(self, env, spec, joint_substr: str):
        import jax.numpy as jnp

        self._env = env
        self._spec = spec
        joints = joints_by_substring(env.mj_model, joint_substr)
        self._qpos_adr = jnp.array([v["qpos_adr"] for v in joints.values()])
        self._dof_adr = jnp.array([v["dof_adr"] for v in joints.values()])

    def step(self, state, action):
        from pea.springs import tau_spring

        theta = state.data.qpos[..., self._qpos_adr]
        tau = tau_spring(theta, self._spec)
        qfrc = state.data.qfrc_applied.at[..., self._dof_adr].set(tau)
        state = state.replace(data=state.data.replace(qfrc_applied=qfrc))
        return self._env.step(state, action)

    def __getattr__(self, name):
        return getattr(self._env, name)


def joints_by_substring(mj_model: mujoco.MjModel, substr: str) -> dict:
    """Locate joints whose name contains `substr`; returns
    {joint_name: {id, qpos_adr, dof_adr}}. Substring match survives Menagerie
    naming tweaks (e.g. 'left_hip_pitch_joint' / 'right_hip_pitch_joint').
    """
    found = {}
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and substr in name:
            found[name] = {
                "id": j,
                "qpos_adr": int(mj_model.jnt_qposadr[j]),
                "dof_adr": int(mj_model.jnt_dofadr[j]),
            }
    if not found:
        raise RuntimeError(f"no joints matching {substr!r} in model")
    return found


def knee_joints(mj_model: mujoco.MjModel) -> dict:
    """Backward-compatible helper used by rollout.py."""
    return joints_by_substring(mj_model, "knee")
