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
    kind: str  # "constant" | "linear" | "semiparabolic" | "one_sided_linear"
    k: float = 0.0       # linear stiffness [N*m/rad], or per-element quadratic
                         # stiffness [N*m/rad^2] when kind="semiparabolic"
    theta0: float = 0.0  # equilibrium angle [rad] (linear)
    tau0: float = 0.0    # constant preload torque [N*m] (constant)
    p1: float = 0.0      # lower onset [rad] (semiparabolic)
    p2: float = 0.0      # upper onset [rad] (semiparabolic)
    # one_sided_linear: engages (gives a restoring torque) only on ONE side of
    # theta_engage and is exactly zero on the other side.
    theta_engage: float = 0.0  # engage angle [rad] (one_sided_linear)
    engage_sign: float = 1.0   # which side engages: the spring is active where
                               # engage_sign*(theta - theta_engage) > 0. The
                               # default (+1) is the FLEXION side of the Go1 calf
                               # (knee), where the joint angle GROWS as the knee
                               # bends under stance load; so the spring stores on
                               # flexion and returns on extension, and is silent
                               # during swing-leg flexion only if -1 is the swing
                               # direction -- pick the sign from the S3 work loop.


# Which SpringConfig fields each kind actually uses. A field set to a
# NON-DEFAULT value for a kind that ignores it is almost always a config bug
# (e.g. tau0 on a 'linear' spring does nothing) -> _check_fields raises.
# `kind`/`joint` are not spring-shape params and are never checked here.
_ALLOWED_FIELDS = {
    "constant": {"tau0"},
    "linear": {"k", "theta0"},
    "semiparabolic": {"k", "p1", "p2"},
    "one_sided_linear": {"k", "theta_engage", "engage_sign",
                         "k_dr", "k_dr_min", "k_dr_max"},
}


def _check_fields(cfg: SpringConfig, kind: str) -> None:
    """Raise ValueError if any field NOT applicable to `kind` is set to a
    non-default value (silently-ignored config). Compares against each field's
    dataclass default so e.g. engage_sign's non-zero default (1.0) is handled."""
    allowed = _ALLOWED_FIELDS[kind] | {"kind", "joint"}
    for f in dataclasses.fields(cfg):
        if f.name in allowed:
            continue
        if getattr(cfg, f.name) != f.default:
            raise ValueError(
                f"spring field {f.name!r}={getattr(cfg, f.name)!r} is not "
                f"applicable to kind={kind!r} (it would be silently ignored); "
                f"applicable fields: {sorted(_ALLOWED_FIELDS[kind])}"
            )


def from_config(cfg: SpringConfig) -> SpringSpec | None:
    """None means no spring (baseline run)."""
    if cfg.kind == "none":
        return None
    if cfg.kind not in ("constant", "linear", "semiparabolic", "one_sided_linear"):
        raise ValueError(f"unknown spring kind: {cfg.kind!r}")
    _check_fields(cfg, cfg.kind)
    return SpringSpec(
        kind=cfg.kind, k=cfg.k, theta0=cfg.theta0, tau0=cfg.tau0,
        p1=cfg.p1, p2=cfg.p2,
        theta_engage=cfg.theta_engage, engage_sign=cfg.engage_sign,
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
    - "one_sided_linear": a linear spring that engages on ONLY ONE side of
      theta_engage and is exactly zero on the other side. On the engaging side
      (where engage_sign*(theta - theta_engage) > 0) it restores toward
      theta_engage: tau = -k*(theta - theta_engage). The running knee spring:
      it stores energy as the knee bends past theta_engage in stance and
      returns it on extension, while staying silent on the other side so it
      does NOT fight the swing leg (the G1 reversal failure mode). It is
      continuous at theta_engage (both branches give 0 there).
    """
    if spec.kind == "constant":
        return spec.tau0 + 0.0 * theta  # broadcast to theta's shape
    if spec.kind == "linear":
        return -spec.k * (theta - spec.theta0)
    if spec.kind == "one_sided_linear":
        d = theta - spec.theta_engage
        # active (1.0) only on the engaging side, else 0 -- numpy/jax agnostic.
        active = (spec.engage_sign * d > 0)
        return -spec.k * d * active
    # semiparabolic: one-sided clamp via (d > 0) multiply (numpy/jax agnostic)
    dA = theta - spec.p1
    dA = dA * (dA > 0)            # 0 at/below p1, positive above
    dB = spec.p2 - theta
    dB = dB * (dB > 0)            # 0 at/above p2, positive below
    return -spec.k * dA**2 + spec.k * dB**2
