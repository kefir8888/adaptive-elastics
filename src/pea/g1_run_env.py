"""G1 RUNNING environment — a small subclass of the Playground G1 walk env that
ALLOWS a flight phase (both feet off the ground at once).

Why this file exists (verified against the Playground source, see
docs/g1_running_design.md): the stock G1 walk env
(`mujoco_playground/_src/locomotion/g1/joystick.py`, registered as
`G1JoystickFlatTerrain`) mathematically FORBIDS running, and the blocking pieces
are NOT reachable from config:

  1. `feet_phase` reward (weight 1.0) pays about +1 per step for keeping EXACTLY
     one foot planted, following an anti-phase gait clock fixed at
     `gait_freq = U(1.25, 1.5)` Hz. At full weight a both-feet-airborne moment is
     punished, so a flight phase never pays.
  2. `_reward_feet_air_time` hard-clips the per-step air-time bonus at
     `threshold_max - threshold_min = 0.5 - 0.2 = 0.3 s`, so there is no reward
     gradient for strides longer than a fast walk.
  3. Both the gait frequency and the air-time thresholds are hardcoded numeric
     literals in the env code, not fields of `reward_config` — so re-weighting
     alone (the only thing config can do) caps the policy at a fast walk.

This subclass overrides exactly those pieces and adds two velocity-gated reward
terms that actively pay for a flight phase, while changing NOTHING else, so it
stays as close to the walk env as possible:

  * un-cap the air-time bonus (threshold_max 0.5 s -> 0.8 s),
  * widen the gait-frequency reset sampling (1.25-1.5 Hz -> 1.6-2.2 Hz),
  * down-weight `feet_phase` (1.0 -> 0.3) so a double-float stops being forbidden
    (set in this env's default config, still reachable via reward_scales),
  * add `flight_bonus`  = +0.5 * (1 - contactL) * (1 - contactR), active only when
    forward speed > 1.5 m/s, clipped to <= 0.3 per step, and
  * add `double_stance` = -0.3 when BOTH feet are down AND forward speed > 0.5 m/s.

Selection: this env is registered under the new name `G1JoystickRun`. Set
`env_name: G1JoystickRun` in a run config and the existing make_env / train /
rollout flow loads it through `registry.load` unchanged. `G1JoystickFlatTerrain`
and the Go1 path are untouched.

The forward-command override (`command_forward` in the config, applied in
make_env via bind_forward_command) composes with this env unchanged: this
subclass inherits `sample_command` and `reset` from the walk env, and
bind_forward_command patches both on the instance after load — so a runner is
commanded forward, not the stock symmetric (mean-zero, sometimes-zeroed) command.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco import mjx

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize


# Registered env name. Use this as `env_name` in a run config to get the runner.
ENV_NAME = "G1JoystickRun"

# --- The exact running overrides, named here so they are easy to find/tune. ---
# Un-cap the air-time bonus. The walk env clips at threshold_max - threshold_min;
# raising threshold_max from 0.5 s to 0.8 s lets a long, running stride keep
# earning reward instead of saturating at a fast-walk stride.
AIR_TIME_THRESHOLD_MAX = 0.8  # seconds (stock walk env: 0.5)
AIR_TIME_THRESHOLD_MIN = 0.2  # seconds (unchanged from the walk env)

# Widen the gait-frequency reset sampling. A faster clock asks for shorter stance
# and a quicker leg cycle, which is what a run needs. The walk env uses U(1.25, 1.5).
GAIT_FREQ_MIN = 1.6  # Hz (stock walk env: 1.25)
GAIT_FREQ_MAX = 2.2  # Hz (stock walk env: 1.5)

# Velocity gates and weights for the two new running terms.
FLIGHT_BONUS_WEIGHT = 0.5     # reward per step when both feet are airborne (before clip)
FLIGHT_BONUS_CLIP = 0.3       # max flight_bonus per step (before reward_scales)
FLIGHT_SPEED_GATE = 1.5       # m/s forward speed above which the flight bonus turns on
DOUBLE_STANCE_PENALTY = 0.3   # magnitude of the both-feet-down penalty (sign is in the scale)
DOUBLE_STANCE_SPEED_GATE = 0.5  # m/s forward speed above which double-stance is penalized

# Down-weighted feet_phase for the runner (stock walk env: 1.0).
FEET_PHASE_WEIGHT_RUN = 0.3


def default_config() -> config_dict.ConfigDict:
    """Run-env default config: the stock G1 walk config, with the running tweaks
    that ARE config-reachable already baked in, plus the two new reward scales so
    they show up in the reward breakdown / metrics by default.

    Everything here is still overridable from a run config's reward_scales /
    env_overrides, exactly like the walk env.
    """
    cfg = g1_joystick.default_config()
    # Down-weight the foot-planted gait-clock term so a flight phase is allowed.
    cfg.reward_config.scales.feet_phase = FEET_PHASE_WEIGHT_RUN
    # Register the two new reward terms so the env builds their metrics keys and
    # they appear in the reward breakdown. flight_bonus is a reward (+), the
    # double-stance term is a cost (-).
    cfg.reward_config.scales.flight_bonus = 1.0
    cfg.reward_config.scales.double_stance = -1.0
    # Sensible running command range (the walk default is [-1, 1] m/s). Still
    # overridable via env_overrides, and bind_forward_command (command_forward)
    # replaces the sampler entirely when set.
    cfg.lin_vel_x = [0.0, 3.0]
    return cfg


class G1JoystickRun(g1_joystick.Joystick):
    """G1 joystick env that allows a flight phase. See module docstring."""

    def reset(self, rng: jax.Array) -> "mjx.Data":  # returns mjx_env.State
        # Reuse all of the walk env's reset (initial pose, command, push, phase),
        # then OVERWRITE just the gait-clock rate with a faster, running-range
        # frequency. The walk reset stores phase_dt = 2*pi*dt*gait_freq with
        # gait_freq ~ U(1.25, 1.5); we re-draw it from the running range so the
        # rest of reset is byte-identical to the walk env.
        state = super().reset(rng)
        rng2, key = jax.random.split(state.info["rng"])
        gait_freq = jax.random.uniform(
            key, (1,), minval=GAIT_FREQ_MIN, maxval=GAIT_FREQ_MAX
        )
        state.info["phase_dt"] = 2 * jp.pi * self.dt * gait_freq
        state.info["rng"] = rng2
        return state

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = AIR_TIME_THRESHOLD_MIN,
        threshold_max: float = AIR_TIME_THRESHOLD_MAX,
    ) -> jax.Array:
        # Same formula as the walk env, but with the higher threshold_max so the
        # per-step bonus is no longer clipped at a fast-walk stride length.
        del commands  # Unused, matches the walk env's signature.
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        return jp.sum(air_time)

    def _reward_flight_bonus(
        self, contact: jax.Array, fwd_speed: jax.Array
    ) -> jax.Array:
        # Pay only when BOTH feet are off the ground (a real flight phase) and only
        # once the robot is actually moving forward fast (so it cannot farm the
        # bonus by hopping in place). `contact` is the env's length-2 foot-contact
        # bool array [left, right]. Clipped small so flight does not dominate.
        both_airborne = (1.0 - contact[0]) * (1.0 - contact[1])
        gate = fwd_speed > FLIGHT_SPEED_GATE
        bonus = FLIGHT_BONUS_WEIGHT * both_airborne * gate
        return jp.clip(bonus, max=FLIGHT_BONUS_CLIP)

    def _cost_double_stance(
        self, contact: jax.Array, fwd_speed: jax.Array
    ) -> jax.Array:
        # Penalize keeping BOTH feet planted while moving forward — that is a slow,
        # double-support walk, the gait we are trying to leave. Returned as a
        # positive number; the negative reward scale turns it into a cost.
        both_down = contact[0] * contact[1]
        gate = fwd_speed > DOUBLE_STANCE_SPEED_GATE
        return DOUBLE_STANCE_PENALTY * both_down * gate

    def _get_reward(
        self,
        data: "mjx.Data",
        action: jax.Array,
        info,
        metrics,
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ):
        # Start from the walk env's full reward dict (so every existing term is
        # preserved unchanged), then add the two running terms. Forward speed is
        # the pelvis local linear velocity along x — the same accessor the
        # tracking reward uses.
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        fwd_speed = self.get_local_linvel(data, "pelvis")[0]
        rewards["flight_bonus"] = self._reward_flight_bonus(contact, fwd_speed)
        rewards["double_stance"] = self._cost_double_stance(contact, fwd_speed)
        return rewards


def register() -> None:
    """Register `G1JoystickRun` so registry.load / get_default_config find it.

    register_environment is the public Playground hook; it fills the env + config
    tables. It does NOT register a domain randomizer, so we also drop the standard
    G1 randomizer into the locomotion randomizer table — otherwise a run with
    `domain_randomization: true` would silently train this env WITHOUT DR (train.py
    looks the randomizer up by env_name). Idempotent: safe to call more than once.
    """
    locomotion.register_environment(ENV_NAME, G1JoystickRun, default_config)
    # Same DR function the stock G1 walk env uses (torso mass, friction, etc.).
    locomotion._randomizer[ENV_NAME] = g1_randomize.domain_randomize


# Register on import so `from pea import g1_run_env` is enough to make the env
# selectable by name. make_env imports this module (see pea/env.py) before
# resolving the config's env_name.
register()
