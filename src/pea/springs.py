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
    kind: str  # "constant" | "linear" | "nonlinear"
    k: float = 0.0
    theta0: float = 0.0
    k3: float = 0.0
    tau0: float = 0.0  # constant preload torque, N*m (kind="constant")


def from_config(cfg: SpringConfig) -> SpringSpec | None:
    """None means no spring (baseline run)."""
    if cfg.kind == "none":
        return None
    if cfg.kind not in ("constant", "linear", "nonlinear"):
        raise ValueError(f"unknown spring kind: {cfg.kind!r}")
    return SpringSpec(
        kind=cfg.kind, k=cfg.k, theta0=cfg.theta0, k3=cfg.k3, tau0=cfg.tau0
    )


def tau_spring(theta, spec: SpringSpec):
    """Spring torque at knee angle theta (rad). Elementwise; numpy or jax."""
    if spec.kind == "constant":
        # Preloaded / constant-torque element. Motivated by the baseline knee
        # work loop: gravity-support torque in a flexed-knee gait is offset-
        # dominated, and the k>=0-constrained optimum degenerates to k=0.
        return spec.tau0 + 0.0 * theta  # broadcast to theta's shape
    d = theta - spec.theta0
    tau = -spec.k * d
    if spec.kind == "nonlinear":
        tau = tau - spec.k3 * d**3
    return tau
