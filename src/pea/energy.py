"""Electrical energy model: copper loss + cost of transport.

We measure ELECTRICAL energy, not mechanical work (CLAUDE.md). Per actuator:

    P_elec = tau * qvel + (tau / Kt)^2 * R        [mechanical + copper loss]

During negative work the motor generates; unless the robot regenerates, that
power is dissipated, so consumed power is max(P_elec, 0) when regen=False.
The spring wins precisely in those phases, so the regen assumption matters —
default is NO regen (assumed dissipated) per CLAUDE.md.

Motor constants are PLACEHOLDERS — order-of-magnitude values for a quasi-
direct-drive knee actuator, expressed at the JOINT side (Kt_joint = Kt_motor *
gear ratio; R is winding resistance). Absolute watts are not trustworthy;
baseline-vs-spring comparisons with identical constants are. Confirm against
real Unitree G1 actuator data before publishing energy numbers (Milestone 3).
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
# `omega_noload` is still unknown — a 2026-06-14 web chase found NO Unitree datasheet
# value (it circled back to our own estimates), and the menagerie carries no joint
# velocity limit. omega_noload=25 rad/s is a PLACEHOLDER (the sim knee already reaches
# ~13.4 rad/s walking, so the true ceiling exceeds that); it is the decisive unknown
# for whether a REAL jump is torque- or speed-limited. Next lead: the Unitree G1 URDF
# (`<limit velocity=>`) on GitHub, or a motor spin-down test. metrics.saturation()
# reads the real per-joint torque cap from the model and pairs it with this estimate.
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
