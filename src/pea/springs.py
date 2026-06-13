"""Parallel knee spring torque tau_spring(theta).

Design rule (CLAUDE.md): this module must serve BOTH halves of the experiment —
as an in-sim torque during training/rollout AND as a standalone function applied
post-hoc to logged trajectories. tau_spring is therefore a pure function of the
knee angle that works elementwise on numpy arrays and jax arrays alike (it only
uses arithmetic ufuncs), with spec parameters as static Python floats (safe
under jax.jit).

Sign convention: theta is the knee joint angle in the model's joint frame and
the returned torque is expressed in that same frame, i.e. directly addable to
the motor torque at the knee DOF. The spring restores toward theta0:
    tau = -k*(theta-theta0) - k3*(theta-theta0)^3
"""

from __future__ import annotations

import dataclasses

from pea.config import SpringConfig


@dataclasses.dataclass(frozen=True)
class SpringSpec:
    kind: str  # "constant" | "linear" | "semiparabolic"
    k: float = 0.0       # linear stiffness [N*m/rad], or per-element quadratic
                         # stiffness [N*m/rad^2] when kind="semiparabolic"
    theta0: float = 0.0  # equilibrium angle [rad] (linear)
    tau0: float = 0.0    # constant preload torque [N*m] (constant)
    p1: float = 0.0      # lower onset [rad] (semiparabolic)
    p2: float = 0.0      # upper onset [rad] (semiparabolic)


def from_config(cfg: SpringConfig) -> SpringSpec | None:
    """None means no spring (baseline run)."""
    if cfg.kind == "none":
        return None
    if cfg.kind not in ("constant", "linear", "semiparabolic"):
        raise ValueError(f"unknown spring kind: {cfg.kind!r}")
    return SpringSpec(
        kind=cfg.kind, k=cfg.k, theta0=cfg.theta0, tau0=cfg.tau0,
        p1=cfg.p1, p2=cfg.p2,
    )


def tau_spring(theta, spec: SpringSpec):
    """Spring torque at joint angle theta (rad). Elementwise; numpy or jax.

    - "constant": preloaded constant-torque element, tau = tau0.
    - "linear": tau = -k*(theta - theta0).
    - "semiparabolic": two opposed ONE-SIDED quadratic elements (zero on one
      side of the onset, quadratic on the other) — the realizable building
      block of the adjustable parallel spring (Hurst et al. 2004; Migliore et
      al. 2005, see docs/mechanism.md). Element A engages above p1, element B
      below p2; with p1 < p2 their overlap is exactly linear with stiffness
      2k(p2-p1) and equilibrium (p1+p2)/2, and the outside is hardening.
    """
    if spec.kind == "constant":
        return spec.tau0 + 0.0 * theta  # broadcast to theta's shape
    if spec.kind == "linear":
        return -spec.k * (theta - spec.theta0)
    # semiparabolic: one-sided clamp via (d > 0) multiply (numpy/jax agnostic)
    dA = theta - spec.p1
    dA = dA * (dA > 0)            # 0 at/below p1, positive above
    dB = spec.p2 - theta
    dB = dB * (dB > 0)            # 0 at/above p2, positive below
    return -spec.k * dA**2 + spec.k * dB**2
