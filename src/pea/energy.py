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


# PLACEHOLDER values — see module docstring. TODO(M3): confirm for G1.
G1_KNEE = MotorConstants(kt=1.0, r=0.05)


def copper_loss(tau, kt: float, r: float):
    """Resistive heating power, W. Elementwise; numpy or jax."""
    return (tau / kt) ** 2 * r


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
