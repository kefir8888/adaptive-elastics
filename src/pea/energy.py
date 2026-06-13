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
