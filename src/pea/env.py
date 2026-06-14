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
    for key, value in cfg.env_overrides.items():  # e.g. lin_vel_x range for running
        if "." in key:  # nested, e.g. reward_config.base_height_target
            *parents, leaf = key.split(".")
            obj = env_cfg
            for parent in parents:
                obj = obj[parent]
            obj[leaf] = value
        else:
            env_cfg[key] = value
    env = registry.load(cfg.env_name, config=env_cfg)
    spec = springs.from_config(cfg.spring)
    if spec is not None:
        env = SpringWrapper(env, spec, cfg.spring.joint)
    if cfg.energy_reward_weight != 0.0:
        env = ElectricalRewardWrapper(env, cfg.energy_reward_weight)
    return env


class ElectricalRewardWrapper:
    """Adds a TOTAL-ELECTRICAL energy penalty to the reward, in both conditions.

    Per actuated DoF the electrical power is max(tau*omega + (tau/Kt)^2*R, 0) —
    mechanical plus ohmic, with no regeneration — the same quantity the
    evaluation cost of transport uses, so the training objective is aligned with
    the metric. tau is the MOTOR torque (qfrc_actuator), so in the spring
    condition the spring's contribution (injected via qfrc_applied) is correctly
    excluded from the motor's electrical cost. Kt, R are the placeholder
    constants in energy.G1_KNEE, identical across conditions.
    """

    def __init__(self, env, weight: float):
        from pea.energy import G1_KNEE

        self._env = env
        self._weight = float(weight)
        self._kt = G1_KNEE.kt
        self._r = G1_KNEE.r

    def step(self, state, action):
        import jax.numpy as jnp

        state = self._env.step(state, action)
        tau = state.data.qfrc_actuator[..., 6:]   # actuated DoFs (skip free base)
        omega = state.data.qvel[..., 6:]
        p_elec = jnp.maximum(tau * omega + (tau / self._kt) ** 2 * self._r, 0.0)
        penalty = self._weight * jnp.sum(p_elec, axis=-1)  # weight < 0 -> a cost
        return state.replace(reward=state.reward + penalty)

    def __getattr__(self, name):
        return getattr(self._env, name)


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


def joint_torque_limits(mj_model: mujoco.MjModel) -> dict:
    """Real per-joint actuator-torque limits the model already ENFORCES.

    These live in `jnt_actfrcrange` (joint-level, jnt_actfrclimited=True), NOT in
    `actuator_forcerange` (which the Playground G1 leaves [0,0]). Authoritative
    from the MuJoCo Menagerie G1: knee ±139, hip-pitch/yaw ±88, ankle-pitch ±50.
    So the TORQUE half of the motor envelope is present and active; only the
    velocity rolloff (back-EMF) is missing (MuJoCo has no joint speed limit) —
    diagnose speed-limiting post-hoc with `metrics.saturation()`.
    """
    out = {}
    for j in range(mj_model.njnt):
        if mj_model.jnt_actfrclimited[j]:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            out[name] = float(mj_model.jnt_actfrcrange[j][1])
    return out


def set_torque_limit(mj_model: mujoco.MjModel, tau_peak: float) -> mujoco.MjModel:
    """Override the joint torque ceiling to ±tau_peak on every actuated joint.

    Writes `jnt_actfrcrange` (the field MuJoCo enforces — an earlier version wrote
    the ignored `actuator_forcerange`). Rarely needed: the model already carries
    the real per-joint limits (see `joint_torque_limits`); use this only to test a
    different ceiling.
    """
    import numpy as np

    limited = mj_model.jnt_actfrclimited.astype(bool)
    mj_model.jnt_actfrcrange[limited] = np.array([-tau_peak, tau_peak])
    return mj_model


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
