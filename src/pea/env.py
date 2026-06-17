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


def scale_hfield(hfield_size, factor: float):
    """Scale rough-terrain bump heights by `factor`, IN PLACE on `hfield_size`.

    Single source of truth for the terrain-roughness scaling used by BOTH the
    physics model (make_env) and the render model (render_walk.py), which used to
    carry separate copies that agreed only by coincidence. heights =
    hfield_data * hfield_size[:,2], so scaling the elevation column (index 2)
    scales every bump. `hfield_size` is the model's (N,4) size array (a numpy
    array — either mj_model.hfield_size directly, or a writable np.array copy of
    the MJX model's). Mutates and returns it.
    """
    hfield_size[:, 2] *= factor
    return hfield_size


def make_env(cfg: RunConfig):
    # Importing g1_run_env registers the running env "G1JoystickRun" in the
    # Playground registry (side effect on import). Harmless if the config does not
    # use it; required before registry.load when env_name is the runner. Does not
    # touch G1JoystickFlatTerrain or the Go1 path.
    from pea import g1_run_env  # noqa: F401

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
    if cfg.command_forward:
        # The stock Go1 joystick samples a SYMMETRIC velocity command (forward speed
        # in [-1.5, 1.5], mean zero, sometimes zeroed). With a mean-zero, often-zero
        # forward command the policy is never required to run forward, so it just
        # learned to stand. Swap in a sampler that always commands forward motion so
        # the policy is forced to run. We bind it onto the actual env instance that
        # owns sample_command (registry.load returns it directly, but walk the
        # _env/.env chain just in case it is nested) BEFORE the trainer wraps and jits
        # the env, so the running env uses the forward command.
        bind_forward_command(env, cfg.command_forward)
    if cfg.terrain_height_scale != 1.0 and getattr(env, "_mjx_model", None) is not None \
            and int(env._mjx_model.nhfield) > 0:
        # Scale rough-terrain bump heights for an easier-terrain start / roughness
        # curriculum. mjx_model is a read-only property -> set the backing field
        # (same pattern as go1_capacity.py's payload-mass injection). The scaling
        # math lives in scale_hfield (shared with render_walk.py).
        import numpy as np
        m = env._mjx_model
        hs = np.array(m.hfield_size)            # static (numpy) field, not a traced jax array
        scale_hfield(hs, cfg.terrain_height_scale)
        env._mjx_model = m.replace(hfield_size=hs)
    if cfg.spring.kind == "preload_dr":
        # per-leg constant knee preload, randomized U(0, tau0) per episode (preload DR):
        # the policy learns robust-to-any-preload; the adaptive controller sets it at eval.
        env = PreloadDRWrapper(env, cfg.spring.joint, cfg.spring.tau0)
    elif cfg.spring.kind == "one_sided_linear" and cfg.spring.k_dr:
        # one-sided stiffness spring with the stiffness k randomized per episode
        # (OPTIONAL robustness knob; default OFF -> the branch below uses a fixed k).
        env = OneSidedStiffnessDRWrapper(env, springs.from_config(cfg.spring),
                                         cfg.spring.joint,
                                         cfg.spring.k_dr_min, cfg.spring.k_dr_max)
    else:
        spec = springs.from_config(cfg.spring)
        if spec is not None:
            env = SpringWrapper(env, spec, cfg.spring.joint)
    if cfg.energy_reward_weight != 0.0:
        env = ElectricalRewardWrapper(env, cfg.energy_reward_weight, cfg.energy_motor)
    return env


def make_forward_sampler(cmd: dict):
    """Build a replacement `sample_command(self, rng, x_k)` that always commands
    FORWARD motion.

    `cmd` carries vx_min, vx_max, vy_max, vyaw_max (floats). The returned function
    returns a length-3 jax array [vx, vy, vyaw]. Unlike the stock sampler it draws
    forward speed from a one-sided range U(vx_min, vx_max) (both positive, so
    always forward) and NEVER zeroes any axis, so every episode demands forward
    running.

    `x_k` (the current command) is accepted but OPTIONAL so this composes with two
    different host signatures: the Go1 walk env calls `sample_command(rng, x_k)`
    (it mixes toward the old command), while the G1 walk env calls
    `sample_command(rng)` with no current command. We ignore x_k either way (we
    re-draw a fresh forward command every time), so one sampler serves both robots.
    """
    import jax
    import jax.numpy as jp

    vx_min = float(cmd["vx_min"])
    vx_max = float(cmd["vx_max"])
    vy_max = float(cmd["vy_max"])
    vyaw_max = float(cmd["vyaw_max"])

    def sample_command(self, rng, x_k=None):
        # x_k (the current command) is intentionally unused and optional: the Go1
        # passes it, the G1 does not. Either way we re-draw a fresh forward command.
        del x_k
        vx_rng, vy_rng, vyaw_rng = jax.random.split(rng, 3)
        vx = jax.random.uniform(vx_rng, (), minval=vx_min, maxval=vx_max)
        vy = jax.random.uniform(vy_rng, (), minval=-vy_max, maxval=vy_max)
        vyaw = jax.random.uniform(vyaw_rng, (), minval=-vyaw_max, maxval=vyaw_max)
        return jp.array([vx, vy, vyaw])

    return sample_command


def bind_forward_command(env, cmd: dict):
    """Bind the forward-command sampler onto the env instance that owns
    `sample_command`.

    registry.load returns the joystick env directly, but to be safe we walk the
    `_env` / `.env` chain to find the instance that actually defines
    `sample_command` and patch THAT one (so the override survives if the loader
    ever returns a wrapped env). Patches in place; returns the patched instance.

    We patch TWO entry points:
    1. `sample_command` — the env re-draws the command here every ~5 s of an
       episode (and on done), so this is the main fix.
    2. `reset` — the env's reset draws the FIRST command of each episode with its
       own symmetric uniform (it does NOT call sample_command), so without this the
       first ~5 s of every episode would still command a random, possibly backward
       velocity. We let the original reset run, then overwrite the initial command
       with a forward draw so the episode is commanded forward from step one.
    """
    import types

    target = env
    # Walk inward until we find the instance that owns sample_command.
    while not hasattr(target, "sample_command"):
        if hasattr(target, "_env"):
            target = target._env
        elif hasattr(target, "env"):
            target = target.env
        else:
            raise RuntimeError("no env in the chain defines sample_command")
    sampler = make_forward_sampler(cmd)
    target.sample_command = types.MethodType(sampler, target)

    orig_reset = target.reset

    def forward_reset(self, rng):
        import jax

        rng, key = jax.random.split(rng)
        state = orig_reset(rng)
        # Replace the symmetric initial command with a forward one (x_k is unused).
        state.info["command"] = self.sample_command(key, state.info["command"])
        return state

    target.reset = types.MethodType(forward_reset, target)
    return target


class _EnvWrapper:
    """Mixin for the env wrappers: a `base_env` accessor that recurses through the
    `_env` chain to the innermost (base) env. Base case: the innermost env has no
    `_env`, so the loop stops and returns it. Replaces the
    `while hasattr(x, '_env'): x = x._env` idiom that scripts open-coded.
    """

    @property
    def base_env(self):
        env = self
        while hasattr(env, "_env"):
            env = env._env
        return env


class ElectricalRewardWrapper(_EnvWrapper):
    """Adds a TOTAL-ELECTRICAL energy penalty to the reward, in both conditions.

    Per actuated DoF the electrical power is max(tau*omega + (tau/Kt)^2*R, 0) —
    mechanical plus ohmic, with no regeneration — the same quantity the
    evaluation cost of transport uses, so the training objective is aligned with
    the metric. tau is the MOTOR torque (qfrc_actuator), so in the spring
    condition the spring's contribution (injected via qfrc_applied) is correctly
    excluded from the motor's electrical cost. Kt, R are the joint-side constants
    from energy.motor_constants(motor) (per cfg.energy_motor), identical across
    conditions.
    """

    def __init__(self, env, weight: float, motor: str = "g1"):
        import jax.numpy as jnp

        from pea.energy import motor_constants

        mc = motor_constants(motor)
        self._env = env
        self._weight = float(weight)
        self._kt = mc.kt
        self._r = mc.r
        # Same actuated-DoF rule as metrics.actuated_dof_adrs (DoFs past the 6
        # free-base DoFs), computed once. Identical to the old [..., 6:] slice on
        # G1/Go1 (so reward numbers are unchanged), but correct for a model with
        # an unactuated non-base joint.
        m = env.mj_model
        self._act = jnp.array(
            [int(m.jnt_dofadr[j]) for j in range(m.njnt)
             if int(m.jnt_dofadr[j]) >= 6]
        )

    def step(self, state, action):
        import jax.numpy as jnp

        state = self._env.step(state, action)
        tau = state.data.qfrc_actuator[..., self._act]   # actuated DoFs (skip free base)
        omega = state.data.qvel[..., self._act]
        p_elec = jnp.maximum(tau * omega + (tau / self._kt) ** 2 * self._r, 0.0)
        penalty = self._weight * jnp.sum(p_elec, axis=-1)  # weight < 0 -> a cost
        return state.replace(reward=state.reward + penalty)

    def __getattr__(self, name):
        return getattr(self._env, name)


class SpringWrapper(_EnvWrapper):
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


class PreloadDRWrapper(_EnvWrapper):
    """Per-leg constant knee preload, randomized U(0, max_tau0) per episode (preload DR).

    Trains a policy ROBUST to any preload; the adaptive controller (eval) sets the per-leg
    tau0 to track each knee's measured load. tau0 is a per-joint vector stored in
    state.info['preload_tau0'] (it persists — the env step mutates info in place). Applied as
    a constant torque at each target DoF via qfrc_applied, exactly like SpringWrapper(constant)
    but per-leg and randomized. qfrc_actuator stays the MOTOR torque, so the energy model
    correctly excludes the (passive) preload. At eval, overwrite state.info['preload_tau0']
    each step with the adaptive controller's value and this wrapper applies it.
    """

    def __init__(self, env, joint_substr: str, max_tau0: float):
        import jax.numpy as jnp

        self._env = env
        self._max = float(max_tau0)
        joints = joints_by_substring(env.mj_model, joint_substr)
        self._dof_adr = jnp.array([v["dof_adr"] for v in joints.values()])
        self._n = len(joints)

    def reset(self, rng):
        import jax

        rng, key = jax.random.split(rng)
        state = self._env.reset(rng)
        state.info["preload_tau0"] = jax.random.uniform(
            key, (self._n,), minval=0.0, maxval=self._max
        )
        return state

    def step(self, state, action):
        tau0 = state.info["preload_tau0"]
        qfrc = state.data.qfrc_applied.at[..., self._dof_adr].set(tau0)
        state = state.replace(data=state.data.replace(qfrc_applied=qfrc))
        return self._env.step(state, action)

    def __getattr__(self, name):
        return getattr(self._env, name)


class OneSidedStiffnessDRWrapper(_EnvWrapper):
    """One-sided linear-stiffness knee spring with the stiffness k randomized per episode.

    Mirrors PreloadDRWrapper, but for the one-sided stiffness spring (springs.py
    kind 'one_sided_linear') and over k instead of the constant preload tau0.
    Each reset draws one scalar k ~ U(k_min, k_max), stored in state.info['spring_k']
    (it persists -- the env step mutates info in place). Each step injects the
    one-sided torque tau = -k*(theta - theta_engage), active only on the engaging
    side, via qfrc_applied -- so qfrc_actuator stays the MOTOR torque and the
    energy model correctly excludes the (passive) spring.

    DEFAULT OFF: the experiment sets k from the S3 work-loop, so a single fixed k
    (the plain SpringWrapper path) is the default; this DR is an OPTIONAL
    robustness knob enabled with spring.k_dr=true.
    """

    def __init__(self, env, spec, joint_substr: str, k_min: float, k_max: float):
        import jax.numpy as jnp

        self._env = env
        self._spec = spec  # carries the STATIC theta_engage / engage_sign
        self._k_min = float(k_min)
        self._k_max = float(k_max)
        joints = joints_by_substring(env.mj_model, joint_substr)
        self._qpos_adr = jnp.array([v["qpos_adr"] for v in joints.values()])
        self._dof_adr = jnp.array([v["dof_adr"] for v in joints.values()])

    def reset(self, rng):
        import jax

        rng, key = jax.random.split(rng)
        state = self._env.reset(rng)
        state.info["spring_k"] = jax.random.uniform(
            key, (), minval=self._k_min, maxval=self._k_max
        )
        return state

    def step(self, state, action):
        theta = state.data.qpos[..., self._qpos_adr]
        k = state.info["spring_k"]
        d = theta - self._spec.theta_engage
        active = (self._spec.engage_sign * d > 0)
        tau = -k * d * active                       # traced k, static theta_engage/sign
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
