"""Electrical energy model: copper loss + cost of transport.

We measure ELECTRICAL energy, not mechanical work (CLAUDE.md). Per actuator:

    P_elec = tau * qvel + (tau / Kt)^2 * R        [mechanical + copper loss]

During negative work the motor generates; unless the robot regenerates, that
power is dissipated, so consumed power is max(P_elec, 0) when regen=False.
The spring wins precisely in those phases, so the regen assumption matters —
default is NO regen (assumed dissipated) per CLAUDE.md.

Motor constants are gear-scaled ESTIMATES (not order-of-magnitude placeholders):
joint-side Kt ~= 2.3 N*m/A, R ~= 0.013 Ohm, load-bearing R/Kt^2 ~= 0.0025 (see
G1_KNEE below and docs/g1_motor_constants.md). No external G1 datasheet exists, so
absolute watts are a BAND not a point; the baseline-vs-spring comparison at identical
constants is Kt/R-invariant and is the trustworthy output.

SCOPE: models copper (ohmic) loss + mechanical power only. It OMITS iron (core) loss
(hysteresis + eddy-current), which grows with motor SPEED, not torque. A parallel
spring offloads TORQUE, so it cannot cut iron loss; adding iron loss would enlarge the
total-power denominator while leaving the spring's saving unchanged -- so our % savings
IGNORE a loss channel that would only dilute them (a conservative, one-sided bias). On
the high-geared G1 the motor turns ~gear x joint speed, so iron loss is not negligible
at the motor; we lack the 7520 core-loss coefficients to quantify it. Cf. arXiv:2506.12314
(copper + iron + mechanical decomposition).
"""

from __future__ import annotations

import dataclasses

import numpy as np

G = 9.81


@dataclasses.dataclass(frozen=True)
class MotorConstants:
    kt: float  # joint-side torque constant, N*m/A
    r: float   # phase/winding resistance, Ohm


# ESTIMATED constants for the G1 7520-22.5 actuator (hip-pitch AND knee share
# this motor), JOINT-SIDE, from a deep web search (2026-06-13). Two independent
# routes (gear-scaled Go2 GO-M8010-6 proxy; peak-torque/implied-current) agree:
#   Kt ~ 2.3 N*m/A (range 2.0-2.7);  R ~ 0.013 Ohm (range 0.009-0.025).
# The load-bearing, GEAR-INVARIANT quantity is R/Kt^2 ~ 0.0025 Ohm/(N*m/A)^2
# (band 0.0020-0.0032): ohmic loss = (tau/Kt)^2 * R depends only on this ratio,
# and only its identical reuse across the spring/no-spring conditions affects the
# headline % reductions. ESTIMATES (no datasheet/hardware) — do NOT publish
# absolute watts on these; report cost of transport as a band, and lead with the
# Kt/R-independent ohmic-% reduction. Other joints differ (hip-yaw 7520-14.3
# Kt~1.5; ankle 5020 Kt~1.4-1.7); using the 7520-22.5 value for all actuated
# DoFs is an estimate-grade simplification, fine for the relative comparison.
G1_KNEE = MotorConstants(kt=2.3, r=0.013)

# Per-robot joint-side motor constants, for cross-robot experiments (metrics.py /
# experiment.py). ESTIMATES — the load-bearing quantity is R/Kt^2; low-gear QDD
# robots (Go2 ~6:1) have a much larger R/Kt^2 than the geared G1, which is the
# whole point of the cross-robot sweep. Add rows as real data is found; override
# per run with --kt/--r. Go2: Kt_q~0.26 N*m/A (teardown); R unmeasured here, so
# the Go2 row is a placeholder until a winding resistance is sourced.
MOTORS: dict[str, MotorConstants] = {
    "g1": G1_KNEE,                          # 7520-22.5, hip-pitch/knee
    "go2": MotorConstants(kt=0.26, r=0.30),  # PLACEHOLDER R — confirm before use
    # Go1 GO-M8010-6 calf (knee): Kt datasheet joint-side (~0.64 at 6.33:1); R an
    # estimate (winding unpublished, ~0.05-0.3) -> R/Kt^2 ~ 0.29, ~120x the geared
    # G1 (0.0025). LOW gear ARMS the ohmic (tau^2) lever — the point of this track.
    "go1_knee": MotorConstants(kt=0.64, r=0.12),
    # --- gravity-compensation direction (mobile manipulators), 2026-06-18 ---
    # LimX W1 (WL_P311D) knee (KFE), ±60 N*m LOW-gear QDD. LimX publishes NO Kt/R;
    # ESTIMATE scaled from the GO-M8010-6 / CubeMars AK80 class to a 60 N*m / ~6.3:1
    # frame -> joint-side Kt~1.5, R~0.06 => R/Kt^2 ~ 0.027 (band 0.018-0.036).
    # Ohmic-significant (low gear) but less extreme than the smaller Go1 unit.
    "limx_knee": MotorConstants(kt=1.5, r=0.06),
    # Galaxea R1 torso-lift joint, HIGH-gear (harmonic/planetary ~100-160:1). Galaxea
    # publishes NO Kt/R; ESTIMATE from the comparable TQ-RoboDrive ILM85 frameless
    # torque-motor class (motor-side Kt 0.073, R 0.027) at gear ~160 -> joint-side
    # Kt~11.7, R~0.027 => R/Kt^2 ~ 0.0002 (band 0.0002-0.002). Ohmic-negligible (high
    # gear) — but gravity-comp still wins big via the gear-INDEPENDENT mechanical
    # (lift-energy recovery) channel; see docs and scripts/galaxea_lift.py.
    "galaxea_torso": MotorConstants(kt=11.7, r=0.027),
    # Galaxea R1 6-DoF ARM joint, high-gear (planetary/harmonic). ESTIMATE from the
    # TQ-RoboDrive ILM70 frameless class (motor-side Kt 0.054, R 0.055) at gear ~100 ->
    # joint-side Kt~5.4, R~0.055 => R/Kt^2 ~ 0.0019. Used for the arms' (constant)
    # holding power in the whole-robot energy accounting.
    "galaxea_arm": MotorConstants(kt=5.4, r=0.055),
}


def motor_constants(name: str) -> MotorConstants:
    """Look up named motor constants; raises with the known names if absent."""
    try:
        return MOTORS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown robot {name!r}; known: {sorted(MOTORS)}") from None


@dataclasses.dataclass(frozen=True)
class MotorLimits:
    """Joint-side torque-SPEED envelope (the thing the stock MuJoCo G1 omits).

    Available torque falls linearly with speed (back-EMF):
        tau_avail(omega) = tau_peak * clamp(1 - |omega|/omega_noload, 0, 1)
    so a push-off can saturate on TORQUE (low speed, |tau|->tau_peak; a parallel
    spring CAN add force) or on SPEED (|omega|->omega_noload; a parallel spring
    CANNOT raise takeoff velocity). This is the data needed to tell a jump's
    torque limit from its speed limit. ESTIMATES — do not publish on them.
    """

    tau_peak: float       # peak joint torque [N*m]
    omega_noload: float   # no-load joint speed [rad/s]


# G1 joint TORQUE limits are AUTHORITATIVE — read from the MuJoCo Menagerie model's
# `jnt_actfrcrange`, which MuJoCo ENFORCES (jnt_actfrclimited=True): knee ±139 N*m,
# hip-pitch/yaw ±88, ankle-pitch ±50. So the sim already caps torque (a max jump is
# torque-limited at these). What the model LACKS is the velocity rolloff (back-EMF):
# `omega_noload` was RESOLVED on 2026-06-14 from the official Unitree G1 URDF (see
# G1_JOINT_VEL below): the knee ceiling is 20 rad/s. The earlier omega_noload=25 rad/s
# was a placeholder, now superseded — NO Unitree *datasheet* value exists, but the URDF
# `<limit velocity=>` is authoritative. The sim knee already reaches ~13.4 rad/s just
# walking (~67% of 20), which is what makes a max jump SPEED-limited at the knee.
# metrics.saturation() reads the real per-joint torque cap from the model and pairs it
# with these URDF velocity limits.
# Per-joint max angular velocity (rad/s) — AUTHORITATIVE from the official Unitree
# G1 URDF (unitreerobotics/unitree_ros, g1_23dof.urdf, <limit velocity=>), fetched
# 2026-06-14. Its effort limits match the model's jnt_actfrcrange (knee 139, hip 88).
G1_JOINT_VEL = {"hip_pitch": 32.0, "hip_roll": 32.0, "hip_yaw": 32.0,
                "knee": 20.0, "ankle_pitch": 30.0, "ankle_roll": 30.0}
# Key consequence: the KNEE ceiling is LOW (20 rad/s) and the sim knee already
# reaches ~13.4 rad/s just walking (~67%), so the knee is SPEED-LIMITED — a parallel
# spring (adds only force) cannot raise knee takeoff speed; the hip (32 rad/s, ~11%
# used) is torque-limited. So a parallel spring is unlikely to raise G1 jump HEIGHT
# (knee-speed-capped); it still helps hip torque and the efficiency/landing case.
OMEGA_NOLOAD_EST = 20.0  # rad/s, knee (the binding leg joint); per-joint in G1_JOINT_VEL
G1_LIMITS = MotorLimits(tau_peak=139.0, omega_noload=G1_JOINT_VEL["knee"])  # knee


def tau_available(omega, lim: MotorLimits):
    """Torque the motor can still deliver at joint speed omega (elementwise)."""
    frac = 1.0 - np.abs(omega) / lim.omega_noload
    return lim.tau_peak * np.clip(frac, 0.0, 1.0)


def ohmic_power(tau, kt: float, r: float):
    """Ohmic loss (Joule heating, I^2 R) in the windings, W. Always >= 0.

    Grows with the square of joint torque; this is the term a parallel spring
    reduces disproportionately by offloading torque. Elementwise; numpy or jax.
    """
    return (tau / kt) ** 2 * r


# Backward-compatible alias; "ohmic_power" is the preferred name.
copper_loss = ohmic_power


def electrical_power(tau, qvel, kt: float, r: float, regen: bool = False):
    """Instantaneous electrical power drawn per actuator, W (elementwise)."""
    p = tau * qvel + copper_loss(tau, kt, r)
    if not regen:
        p = np.maximum(p, 0.0)
    return p


def energy(power, dt: float) -> float:
    """Integrate a power time series, J."""
    return float(np.sum(power) * dt)


def cost_of_transport(energy_j: float, mass_kg: float, distance_m: float) -> float:
    """Dimensionless CoT = E / (m g d). Headline metric."""
    return energy_j / (mass_kg * G * distance_m)
