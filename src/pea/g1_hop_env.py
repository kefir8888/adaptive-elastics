"""G1 continuous-HOPPING environment — a small subclass of the Playground G1 walk
env that elicits repeated, synchronized, two-footed vertical jumps
(crouch -> push -> flight -> land -> crouch -> ...), the explosive-but-REPETITIVE
task for Part 2.

Why this file exists
--------------------
The stock G1 walk env (`G1JoystickFlatTerrain`,
`mujoco_playground/_src/locomotion/g1/joystick.py`) is built around an ANTI-PHASE
walking gait clock and a "stand still at zero command" objective, BOTH of which
forbid a two-footed hop. The blocking pieces are not reachable from config:

  1. `feet_phase` (weight 1.0) tracks an anti-phase clock (`phase = [0, pi]`) that
     pays for keeping the two feet OUT of phase — i.e. for stepping, not jumping.
  2. `stand_still` (weight -1.0) penalizes ANY deviation from the default pose when
     the joystick command is ~0 — and an in-place hop needs a deep crouch at ~0
     horizontal command, so the crouch is punished.
  3. `_reward_feet_air_time` hard-clips the per-foot air-time bonus at
     threshold_max - threshold_min = 0.5 - 0.2 = 0.3 s, so a hop higher than a
     fast-walk stride earns no extra reward.
  4. Both the gait frequency and the air-time clip are hardcoded numeric literals,
     not fields of `reward_config`, so re-weighting alone cannot reach a hop.

This subclass keeps EVERYTHING else identical to the walk env and changes only the
pieces above:

  * synchronous gait clock — `reset` overwrites `phase = [0, 0]` (both feet IN
    phase) so the clock describes one whole-body bounce, not a step, and re-draws
    the clock rate from a hopping range (HOP_FREQ_MIN..MAX Hz). The policy observes
    `phase` (cos/sin), so it can lock onto whatever cadence the clock runs — which
    is how the matched-cadence energy comparison is controlled (fix the freq).
  * un-cap the air-time bonus (threshold_max 0.5 s -> 0.8 s), so a higher/longer
    hop keeps earning instead of saturating at a fast-walk stride.
  * in its default config, neutralize the walk-only terms that forbid a hop
    (`feet_phase` 1.0->0, `stand_still` -1.0->0, `joint_deviation_knee` -0.1->0 so
    the crouch is free) and lighten `pose` (-0.1 -> -0.05 so the legs may crouch).
  * add five hopping reward terms (registered so they show up in the breakdown):
      - `hop_rhythm` = 2*(s*both_air + (1-s)*both_down) - 1, SIGNED to [-1,1] with s
                       the synchronous clock in [0,1]. The BACKBONE: it makes a
                       periodic flight phase the task and breaks the stand-still
                       local optimum — being grounded during the flight half is a
                       PENALTY, so standing nets ~0 while a hop in phase scores ~+1.
      - `hop_push`   = (>=1 foot down) * clip(pelvis upward velocity, 0, VCAP): a
                       DENSE bootstrap rewarding the controllable push-off, giving a
                       continuous gradient stand -> crouch -> push -> fly where the
                       airborne-only terms below have none.
      - `hop_height` = (both airborne) * apex height above STANDING (self._init_q[2],
                       not the 0.5 nominal) — MAXIMIZE mode (`hop_height_target` <= 0)
                       or a tracking bump toward `hop_height_target` when set (the
                       matched-task mode for the spring-vs-no-spring energy compare).
      - `hop_sync`   = 1 - |contactL - contactR|: rewards the two feet being in the
                       SAME state (both up / both down) — a two-footed jump, not a
                       step or skip.
      - `hop_flight` = (both airborne): a bootstrap reward for leaving the ground at
                       all (the height term has ~zero gradient until the body has
                       already risen).

The command defaults to ZERO horizontal velocity (`lin_vel_* = [0, 0]`), so the
task is hopping IN PLACE: `tracking_lin_vel` / `tracking_ang_vel` then reward not
drifting while the hop terms reward bouncing. A forward hop is reachable by widening
`lin_vel_x` in a run config (the inherited command machinery is unchanged), but the
clean energy experiment is the in-place vertical bounce: maximal landing-braking
energy for the spring to recover, and no forward-locomotion confound.

METRIC NOTE: in-place hopping has no forward speed, so the headline is electrical
ENERGY PER HOP (or mean electrical power at a matched hop height + cadence), NOT
cost-of-transport — CoT (W / (m/s)) is undefined at zero forward speed. Fix the
cadence (HOP_FREQ_MIN == HOP_FREQ_MAX or one value) and `hop_height_target` so the
no-spring and spring arms perform the IDENTICAL mechanical task, then compare
electrical energy.

Selection: registered as `G1JoystickHop`; set `env_name: G1JoystickHop` in a run
config and the existing make_env / train / rollout flow loads it through
`registry.load` unchanged. The spring / energy wrappers in `pea.env.make_env`
compose with it transparently. For the spring arm, the knee one-sided stiffness
spring (`springs.py` kind `one_sided_linear`, engaging in deep flexion =
landing/crouch) is the natural pogo-stick element; fit it from a baseline-hop knee
work-loop — plot ALL leg joints (the ANKLE is also a strong hop-spring candidate)
and let the data choose.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from ml_collections import config_dict

import mujoco
from mujoco import mjx

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize


# Registered env name. Use this as `env_name` in a run config to get the hopper.
ENV_NAME = "G1JoystickHop"

# --- Hop clock --------------------------------------------------------------
# The synchronous gait clock sets the hop CADENCE (one bounce per clock period).
# A hop period of ~0.5-0.67 s (1.5-2.2 Hz) is a natural two-footed bounce. The
# policy observes the clock phase, so it locks onto whatever rate is sampled. For
# the matched-cadence energy comparison set HOP_FREQ_MIN == HOP_FREQ_MAX (or fix it
# in the eval) so both arms hop at the SAME frequency.
HOP_FREQ_MIN = 1.5  # Hz (walk env clock: 1.25)
HOP_FREQ_MAX = 2.2  # Hz (walk env clock: 1.5)

# --- Air-time un-cap (same idea as g1_run_env) ------------------------------
# Raise threshold_max from 0.5 s to 0.8 s so a long flight phase keeps earning the
# air-time bonus instead of saturating at a fast-walk stride.
AIR_TIME_THRESHOLD_MAX = 0.8  # seconds (stock walk env: 0.5)
AIR_TIME_THRESHOLD_MIN = 0.2  # seconds (unchanged from the walk env)

# --- Hop reward shape -------------------------------------------------------
# Apex reward cap (MAXIMIZE mode): height of the pelvis above the STANDING
# reference (self._init_q[2], the keyframe pelvis z ~ 0.75 m, NOT the nominal
# base_height_target of 0.5) is rewarded up to this many metres (a ~0.35 m bounce
# is already large for the G1; the knee is speed-limited, so do not expect more).
HOP_HEIGHT_CAP = 0.35      # metres above standing, clipped (maximize mode)
HOP_HEIGHT_VAR = 0.01      # Gaussian variance (sigma^2, m^2; sigma = 0.1 m) for
                           # apex-height tracking when hop_height_target is set.

# Upward push (the dense bootstrap that crosses the stand->hop reward barrier):
# reward positive pelvis vertical velocity WHILE a foot is still on the ground, so
# the gradient runs stand -> crouch -> push -> fly. A ~0.2 m apex needs a takeoff
# speed of sqrt(2 g h) ~ 2 m/s, so cap there.
HOP_PUSH_VCAP = 2.0        # m/s, upward pelvis velocity rewarded up to this

# Default weights for the added terms (overridable via reward_scales). Tuned from a
# term-by-term standing-vs-hopping analysis so a synchronized hop clearly beats
# standing AND every intermediate state (crouch, push) is uphill: rhythm is the
# (signed) backbone, push is the dense bootstrap, height drives amplitude, sync
# forbids asymmetric skips, flight rewards leaving the ground.
HOP_RHYTHM_WEIGHT = 2.0
HOP_PUSH_WEIGHT = 1.0
HOP_HEIGHT_WEIGHT = 4.0
HOP_SYNC_WEIGHT = 2.0
HOP_FLIGHT_WEIGHT = 1.5

# Walk-env terms that forbid a hop, neutralized / lightened in the hop default cfg.
FEET_PHASE_WEIGHT_HOP = 0.0          # kill the anti-phase walking gait clock
STAND_STILL_WEIGHT_HOP = 0.0         # do NOT punish the crouch at zero command
JOINT_DEVIATION_KNEE_WEIGHT_HOP = 0.0  # the crouch needs free knees
POSE_WEIGHT_HOP = -0.05              # lighten (was -0.1) so legs may crouch
FEET_AIR_TIME_WEIGHT_HOP = 3.0       # boost (was 2.0): reward longer flight
# Loosen position-holding: the robot should hop ROUGHLY in place but is only gently
# nudged to stay put, NOT strongly punished for drifting while it bounces (user
# intent: in-place, not rigid). Down-weighting tracking also reduces the reward for
# standing still, which helps escape the stand-still local optimum. A commanded
# hop-direction "joystick" will replace this soft hold later.
TRACKING_LIN_VEL_WEIGHT_HOP = 0.5    # gentle stay-local nudge (walk env: 1.0)
TRACKING_ANG_VEL_WEIGHT_HOP = 0.5    # gentle no-spin nudge (walk env: 0.75)
# Leg-symmetry penalty: punish a DIAGONAL (asymmetric) stance — the ROOT CAUSE of the
# inherited yaw spin (every hopper trained 2026-06-20 spun ~+2-3.8 rad/s except bounding).
# 0 by default; set negative (a penalty) in the clean-curriculum configs (stage 1+).
LEG_SYMMETRY_WEIGHT = 0.0


def default_config() -> config_dict.ConfigDict:
    """Hop-env default config: the stock G1 walk config with the structural hop
    changes baked in (the anti-hop terms neutralized, the four hop terms
    registered, the command zeroed for an in-place bounce). Reward WEIGHTS and the
    stage choices (smoothness, softened termination, energy penalty, a fixed
    cadence / height target for the comparison) are left to the run config, exactly
    like the walk and run envs.
    """
    cfg = g1_joystick.default_config()
    s = cfg.reward_config.scales
    # Neutralize the walk-only terms that structurally forbid a two-footed hop.
    s.feet_phase = FEET_PHASE_WEIGHT_HOP
    s.stand_still = STAND_STILL_WEIGHT_HOP
    s.joint_deviation_knee = JOINT_DEVIATION_KNEE_WEIGHT_HOP
    s.pose = POSE_WEIGHT_HOP
    s.feet_air_time = FEET_AIR_TIME_WEIGHT_HOP
    # Loose position-holding (see the constants): hop in place, but do not punish
    # drift hard. Identical across all hop arms (set here, not per-config) for parity.
    s.tracking_lin_vel = TRACKING_LIN_VEL_WEIGHT_HOP
    s.tracking_ang_vel = TRACKING_ANG_VEL_WEIGHT_HOP
    s.leg_symmetry = LEG_SYMMETRY_WEIGHT
    # Register the hop terms so the env builds their metrics keys and they show up
    # in the reward breakdown. All are rewards (+); hop_rhythm is SIGNED ([-1, 1]).
    s.hop_rhythm = HOP_RHYTHM_WEIGHT
    s.hop_push = HOP_PUSH_WEIGHT
    s.hop_height = HOP_HEIGHT_WEIGHT
    s.hop_sync = HOP_SYNC_WEIGHT
    s.hop_flight = HOP_FLIGHT_WEIGHT
    s.hop_overshoot = 0.0   # apex-cap penalty; OFF by default, negative in fair-compare configs
    # The base contact-force cap is 500 N, calibrated for a WALKING foot-strike; a
    # hop landing peaks far higher, so at 500 N the contact_force cost would crush
    # any real hop. Raise the cap; impact IS the hop task (and the very thing the
    # Part-2 spring absorbs), so do not over-penalize it before the spring exists.
    cfg.reward_config.max_contact_force = 2000.0
    # Apex-tracking target (metres above standing). 0 -> MAXIMIZE the hop height;
    # > 0 -> reward matching this apex (the matched-task mode for the comparison).
    cfg.reward_config.hop_height_target = 0.0
    # Tracking width (sigma^2, m^2) for the apex when hop_height_target is set; the
    # FAIR-comparison configs tighten this so the apex is pinned, not loosely tracked.
    cfg.reward_config.hop_height_var = HOP_HEIGHT_VAR
    # `hop_overshoot` (registered below) caps the apex at the target; default weight 0.
    # Hop cadence. 0 -> sample U(HOP_FREQ_MIN, HOP_FREQ_MAX) per episode (training);
    # > 0 -> FIX the clock at this frequency (Hz) so the no-spring and spring arms
    # hop at the SAME cadence for the matched-energy comparison.
    cfg.reward_config.hop_freq = 0.0
    # In-place hop: zero horizontal command. tracking_lin_vel/ang_vel then reward
    # not drifting. Widen lin_vel_x in a run config for a forward hop.
    cfg.lin_vel_x = [0.0, 0.0]
    cfg.lin_vel_y = [0.0, 0.0]
    cfg.ang_vel_yaw = [0.0, 0.0]
    return cfg


class G1JoystickHop(g1_joystick.Joystick):
    """G1 joystick env that elicits continuous two-footed hopping. See module docstring."""

    def reset(self, rng: jax.Array) -> "mjx.Data":  # returns mjx_env.State
        # Reuse all of the walk env's reset (pose, command, push, metrics), then
        # overwrite ONLY the gait clock: make it SYNCHRONOUS (phase = [0, 0] so both
        # feet share one bounce, not an anti-phase step) and re-draw its rate from
        # the hopping range. Everything else stays byte-identical to the walk reset.
        state = super().reset(rng)
        rng2, key = jax.random.split(state.info["rng"])
        fixed = float(self._config.reward_config.hop_freq)
        if fixed > 0.0:  # matched-cadence comparison: fix the hop frequency
            gait_freq = jp.array([fixed])
        else:            # training: sample a cadence; the policy reads it off `phase`
            gait_freq = jax.random.uniform(
                key, (1,), minval=HOP_FREQ_MIN, maxval=HOP_FREQ_MAX
            )
        state.info["phase_dt"] = 2 * jp.pi * self.dt * gait_freq
        # Synchronous clock: both feet in phase -> a whole-body hop. (Walk: [0, pi].)
        state.info["phase"] = jp.array([0.0, 0.0])
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
        # Same formula as the walk env, but with the higher threshold_max so a long
        # flight phase is no longer clipped at a fast-walk stride. In a hop BOTH feet
        # are airborne together, so both accumulate the same air_time and both pay on
        # the (simultaneous) landing -> this rewards higher/longer hops.
        del commands  # Unused, matches the walk env's signature.
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        return jp.sum(air_time)

    def _reward_hop_rhythm(
        self, contact: jax.Array, phase: jax.Array
    ) -> jax.Array:
        # The backbone term, SIGNED to [-1, 1]. s in [0,1] is the synchronous clock:
        # 0 at phase 0 (stance/load -> want both feet DOWN), 1 at phase pi (flight ->
        # want both feet UP). `2*(...) - 1` makes being grounded during the flight
        # half a PENALTY (not a free +0.5 floor): standing through the cycle nets ~0
        # instead of +0.5, so a periodic flight phase decisively out-scores standing.
        s = 0.5 * (1.0 - jp.cos(phase[0]))
        both_airborne = (1.0 - contact[0]) * (1.0 - contact[1])
        both_down = contact[0] * contact[1]
        return 2.0 * (s * both_airborne + (1.0 - s) * both_down) - 1.0

    def _reward_hop_push(self, data: "mjx.Data", contact: jax.Array) -> jax.Array:
        # Dense bootstrap across the stand->hop reward barrier: reward positive
        # pelvis vertical velocity WHILE at least one foot is still on the ground
        # (the controllable push, not the uncontrollable ballistic ascent). This is
        # what gives a continuous gradient from standing through the crouch and
        # push-off, where the airborne-only height/flight terms have none.
        vz = self.get_global_linvel(data, "pelvis")[2]
        on_ground = 1.0 - (1.0 - contact[0]) * (1.0 - contact[1])  # >=1 foot down
        return on_ground * jp.clip(vz, 0.0, HOP_PUSH_VCAP)

    def _reward_hop_height(
        self, contact: jax.Array, base_height: jax.Array
    ) -> jax.Array:
        # Apex amplitude, paid only while BOTH feet are airborne (you cannot earn it
        # by standing tall on the ground). `gain` is the pelvis height above the
        # STANDING reference (self._init_q[2], the keyframe pelvis z ~ 0.75 m — NOT
        # base_height_target=0.5, which is a disabled nominal and would hand the
        # robot height reward just for standing). MAXIMIZE mode (target <= 0): reward
        # the clipped gain. TRACK mode (target > 0): reward matching the commanded
        # apex — the matched mechanical task for the spring-vs-no-spring comparison.
        both_airborne = (1.0 - contact[0]) * (1.0 - contact[1])
        h_ref = self._init_q[2]
        target = float(self._config.reward_config.hop_height_target)
        var = float(self._config.reward_config.hop_height_var)  # tighten for a pinned apex
        gain = base_height - h_ref
        if target > 0.0:
            shaped = jp.exp(-jp.square(gain - target) / var)
        else:
            shaped = jp.clip(gain, 0.0, HOP_HEIGHT_CAP)
        return both_airborne * shaped

    def _reward_hop_overshoot(self, base_height: jax.Array) -> jax.Array:
        # Metres the pelvis rises ABOVE the commanded apex. With a negative scale this
        # CAPS the hop height at the target, so a spring (or any extra push capacity)
        # cannot be spent on hopping higher — it is forced into the matched task. This
        # is the height-control knob for a FAIR spring-vs-no-spring energy comparison.
        # Off (returns 0) in maximize mode (no commanded apex).
        target = float(self._config.reward_config.hop_height_target)
        if target <= 0.0:
            return jp.array(0.0)
        h_ref = self._init_q[2]
        return jp.maximum((base_height - h_ref) - target, 0.0)

    def _reward_hop_sync(self, contact: jax.Array) -> jax.Array:
        # Reward the two feet being in the SAME contact state (both up / both down):
        # a two-footed jump, not a step or skip. 1 when synchronized, 0 when split.
        cl = contact[0].astype(jp.float32)
        cr = contact[1].astype(jp.float32)
        return 1.0 - jp.abs(cl - cr)

    def _reward_hop_flight(self, contact: jax.Array) -> jax.Array:
        # Small bootstrap: pay for both feet off the ground at all. The height term
        # has ~zero gradient until the body has already risen, so this gets the
        # policy to leave the ground first.
        return (1.0 - contact[0]) * (1.0 - contact[1])

    def _reward_leg_symmetry(self, data: "mjx.Data") -> jax.Array:
        # Penalize a DIAGONAL / asymmetric stance — the ROOT CAUSE of the inherited yaw
        # spin (2026-06-21). In the PELVIS BODY frame the two feet should be symmetric:
        # same fore-aft (x) and mirror-image lateral (y_L = -y_R). A diagonal splay (one
        # foot fore, one aft) breaks the fore-aft term and yaws the body. Sign-free (body
        # frame; no joint mirror map). mj_name2id on the static model is constant-folded
        # under jit (zero runtime cost).
        pid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        feet = data.site_xpos[self._feet_site_id]       # (2,3) world: [left, right]
        rel = feet - data.xpos[pid]                      # relative to pelvis (world)
        body = rel @ data.xmat[pid].reshape(3, 3)        # -> pelvis body frame (rel @ R = R^T·rel)
        fore_aft_asym = jp.abs(body[0, 0] - body[1, 0])  # feet at same fore-aft -> 0
        lateral_asym = jp.abs(body[0, 1] + body[1, 1])   # mirror lateral (L=+, R=−) -> sum 0
        return fore_aft_asym + lateral_asym  # POSITIVE cost (metres of asymmetry); use a NEGATIVE scale

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
        # preserved — its scale, not its presence, is what we changed), then add the
        # four hop terms. base_height is the pelvis z (data.qpos[2]), the same value
        # the walk env's base_height cost uses.
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        base_height = data.qpos[2]
        rewards["hop_rhythm"] = self._reward_hop_rhythm(contact, info["phase"])
        rewards["hop_push"] = self._reward_hop_push(data, contact)
        rewards["hop_height"] = self._reward_hop_height(contact, base_height)
        rewards["hop_sync"] = self._reward_hop_sync(contact)
        rewards["hop_flight"] = self._reward_hop_flight(contact)
        rewards["hop_overshoot"] = self._reward_hop_overshoot(base_height)
        rewards["leg_symmetry"] = self._reward_leg_symmetry(data)
        return rewards


def register() -> None:
    """Register `G1JoystickHop` so registry.load / get_default_config find it, and
    drop the standard G1 randomizer into the locomotion randomizer table so a run
    with `domain_randomization: true` trains this env WITH DR (train.py looks the
    randomizer up by env_name). Idempotent: safe to call more than once.
    """
    locomotion.register_environment(ENV_NAME, G1JoystickHop, default_config)
    locomotion._randomizer[ENV_NAME] = g1_randomize.domain_randomize


# Register on import so `from pea import g1_hop_env` is enough to make the env
# selectable by name (make_env imports this module before resolving env_name).
register()
