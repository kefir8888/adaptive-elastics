"""G1 alternating-foot BOUNDING / running env — "two feet interchangeably".

Built on the hopping env (`G1JoystickHop`), which already demonstrably elicits a
real flight phase. The move to running: the two feet push off INTERCHANGEABLY
(anti-phase gait clock, phase = [0, pi], like walking) with a flight phase between
footfalls, while the body travels FORWARD. The idea (user, 2026-06-20): start from
the proven two-footed jump and let the feet alternate -> a bounding run.

vs the hop env (synchronous, in-place) this:
  * resets the clock to ANTI-PHASE (phase = [0, pi]) so the feet alternate;
  * re-enables `feet_phase` (the base anti-phase foot-height clock) so alternation
    is rewarded, and turns OFF the synchronous-only terms (`hop_rhythm`, `hop_sync`);
  * keeps the flight machinery that made jumping work: `hop_push` (vertical push),
    `hop_flight` (a both-feet-airborne flight phase), un-capped air-time, and the
    neutralized stand-still / free crouch;
  * restores forward-velocity tracking and is driven by a FORWARD command
    (set `command_forward` in the run config) so it runs forward, not hops in place.

Warm-start from the two-footed hopper (same obs/action space) and curriculum the
forward speed up. Registered as `G1JoystickBound`.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco import mjx

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

from pea import g1_hop_env


ENV_NAME = "G1JoystickBound"

# Anti-phase bounding clock cadence (Hz). A running stride is quicker than a hop.
BOUND_FREQ_MIN = 1.6
BOUND_FREQ_MAX = 2.6

# Default weights (overridable via reward_scales).
FEET_PHASE_WEIGHT_BOUND = 0.6     # re-enable anti-phase alternation (hop env had 0)
TRACKING_LIN_VEL_WEIGHT_BOUND = 1.5  # forward-speed tracking matters now
HOP_HEIGHT_WEIGHT_BOUND = 1.0     # modest vertical (running travels, not hops high)
HOP_FLIGHT_WEIGHT_BOUND = 1.0     # reward a real flight phase
HOP_PUSH_WEIGHT_BOUND = 1.0       # keep the push-off bootstrap


def default_config() -> config_dict.ConfigDict:
    cfg = g1_hop_env.default_config()  # inherits the hop neutralizations + hop terms
    s = cfg.reward_config.scales
    # Synchronous-only hop terms OFF (they assume both feet share one clock).
    s.hop_rhythm = 0.0
    s.hop_sync = 0.0
    # Alternation ON (anti-phase foot-height clock from the base walk env).
    s.feet_phase = FEET_PHASE_WEIGHT_BOUND
    # Forward running shaping.
    s.tracking_lin_vel = TRACKING_LIN_VEL_WEIGHT_BOUND
    s.hop_height = HOP_HEIGHT_WEIGHT_BOUND
    s.hop_flight = HOP_FLIGHT_WEIGHT_BOUND
    s.hop_push = HOP_PUSH_WEIGHT_BOUND
    s.feet_slip = -0.5                # discourage foot scuffing at speed
    # Forward command range (the actual command is set by command_forward in the
    # run config; this is the fallback sampler range).
    cfg.lin_vel_x = [0.0, 2.5]
    cfg.lin_vel_y = [0.0, 0.0]
    cfg.ang_vel_yaw = [0.0, 0.0]
    return cfg


class G1JoystickBound(g1_hop_env.G1JoystickHop):
    """Alternating-foot forward bounding/running. See module docstring."""

    def reset(self, rng: jax.Array) -> "mjx.Data":
        # Hop reset sets a SYNCHRONOUS clock [0,0]; override to ANTI-PHASE [0,pi] so
        # the two feet alternate, and draw the cadence from the bounding range.
        state = super().reset(rng)
        rng2, key = jax.random.split(state.info["rng"])
        fixed = float(self._config.reward_config.hop_freq)
        if fixed > 0.0:
            gait_freq = jp.array([fixed])
        else:
            gait_freq = jax.random.uniform(
                key, (1,), minval=BOUND_FREQ_MIN, maxval=BOUND_FREQ_MAX
            )
        state.info["phase_dt"] = 2 * jp.pi * self.dt * gait_freq
        state.info["phase"] = jp.array([0.0, jp.pi])  # anti-phase: feet alternate
        state.info["rng"] = rng2
        return state


def register() -> None:
    locomotion.register_environment(ENV_NAME, G1JoystickBound, default_config)
    locomotion._randomizer[ENV_NAME] = g1_randomize.domain_randomize


register()
