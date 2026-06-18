"""Gravity-compensation energy scenarios for a single lift joint (Part-1 spinoff).

A heavy load is raised/lowered about ONE joint that sees a constant-SIGN gravity load
(a torso lift, a shoulder, a knee in stance). Unlike cyclic locomotion, the load never
reverses, so a PERMANENT parallel spring (no clutch) is unambiguously helpful. We:
  1. generate a smooth up/down duty cycle (minimum-jerk: zero end velocity AND accel);
  2. get the required motor torque as I(theta)*alpha + tau_gravity(theta) -- the exact
     rigid-body dynamics for a single-axis rotation (no Coriolis/centrifugal on that DoF;
     we deliberately avoid mj_inverse, whose held-but-unwelded joints inject spurious
     multibody velocity cross-terms);
  3. compute ELECTRICAL energy with pea.energy (P = tau*omega + (tau/Kt)^2*R, no-regen);
  4. subtract a parallel spring tau_spring(theta) (pea.springs) so the motor supplies
     tau_req - tau_spring, and find the energy-optimal spring.

This module is robot-agnostic; scripts/galaxea_lift.py wires it to Galaxea R1.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import mujoco

from pea import energy as E
from pea.springs import SpringSpec, tau_spring


# ----------------------------------------------------------------------------- trajectory
def minjerk_cycle(top: float, bot: float, hold_s: float, move_s: float,
                  n_cycles: int, dt: float):
    """Up/down duty cycle: hold(top) - lower to bot - hold(bot) - raise to top - hold(top),
    repeated n_cycles. Returns (theta, omega, alpha) with analytic minimum-jerk segments
    (zero velocity AND acceleration at every segment end, so no torque spikes)."""
    def seg_hold(v, dur):
        n = int(round(dur / dt)); z = np.zeros(n)
        return np.full(n, v), z, z.copy()

    def seg_move(a, b, dur):
        n = int(round(dur / dt)); T = n * dt
        u = (np.arange(n) + 1) * dt / T
        th = a + (b - a) * (10 * u**3 - 15 * u**4 + 6 * u**5)
        om = (b - a) * (30 * u**2 - 60 * u**3 + 30 * u**4) / T
        al = (b - a) * (60 * u - 180 * u**2 + 120 * u**3) / T**2
        return th, om, al

    segs = []
    for _ in range(n_cycles):
        segs += [seg_hold(top, hold_s), seg_move(top, bot, move_s),
                 seg_hold(bot, hold_s), seg_move(bot, top, move_s),
                 seg_hold(top, hold_s)]
    theta = np.concatenate([s[0] for s in segs])
    omega = np.concatenate([s[1] for s in segs])
    alpha = np.concatenate([s[2] for s in segs])
    return theta, omega, alpha


# ----------------------------------------------------------------------------- dynamics
def rigid_lift_torque(model, data, joint_name, theta, omega, alpha, qnom=None):
    """Required motor torque along the prescribed motion for a single lift joint.

    tau_req = I(theta)*alpha + tau_gravity(theta), with I = mass-matrix diagonal at the
    joint and tau_gravity = qfrc_bias at zero velocity. The downstream body is rigid, so
    this is exact (single-axis rigid rotation has no velocity-dependent generalized force).
    qnom: full qpos for the HELD joints (defaults to each limited joint's range midpoint,
    so none rests on a URDF limit and injects a spurious constraint force).
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qadr = model.jnt_qposadr[jid]; dadr = model.jnt_dofadr[jid]
    if qnom is None:
        qnom = np.zeros(model.nq)
        for j in range(model.njnt):
            if model.jnt_type[j] in (2, 3) and model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                qnom[model.jnt_qposadr[j]] = 0.5 * (lo + hi)
    N = len(theta)
    tau_grav = np.zeros(N); Idiag = np.zeros(N)
    Mbuf = np.zeros((model.nv, model.nv))
    for i in range(N):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qnom; data.qpos[qadr] = theta[i]
        mujoco.mj_forward(model, data)
        tau_grav[i] = data.qfrc_bias[dadr]
        mujoco.mj_fullM(model, Mbuf, data.qM)
        Idiag[i] = Mbuf[dadr, dadr]
    tau_req = Idiag * alpha + tau_grav
    return tau_req, tau_grav, Idiag


# ----------------------------------------------------------------------------- energy
@dataclasses.dataclass(frozen=True)
class EnergyBreakdown:
    total: float       # J, electrical energy over the whole motion (no-regen by default)
    ohmic: float       # J, (tau/Kt)^2 R   integrated
    mech_pos: float    # J, max(tau*omega, 0) integrated (positive mechanical work)


def lift_energy(tau_motor, omega, motor: E.MotorConstants, dt: float,
                regen: bool = False) -> EnergyBreakdown:
    p = E.electrical_power(tau_motor, omega, motor.kt, motor.r, regen=regen)
    ohm = E.ohmic_power(tau_motor, motor.kt, motor.r)
    mech = np.maximum(tau_motor * omega, 0.0)
    return EnergyBreakdown(E.energy(p, dt), E.energy(ohm, dt), E.energy(mech, dt))


def motor_energy_with_spring(tau_req, theta, omega, spec: SpringSpec | None,
                             motor: E.MotorConstants, dt: float, regen: bool = False):
    """Electrical energy when a parallel spring supplies tau_spring(theta)."""
    tau_motor = tau_req if spec is None else tau_req - tau_spring(theta, spec)
    return lift_energy(tau_motor, omega, motor, dt, regen)


def best_linear_spring(tau_req, theta, omega, motor: E.MotorConstants, dt: float,
                       regen: bool = False, k_max: float = 500.0,
                       n_k: int = 51, n_t0: int = 49):
    """Grid-search the energy-optimal linear torsion spring tau=-k(theta-theta0)."""
    th_lo, th_hi = theta.min() - 0.6, theta.max() + 0.6
    best = (np.inf, None)
    for k in np.linspace(0, k_max, n_k):
        for t0 in np.linspace(th_lo, th_hi, n_t0):
            e = motor_energy_with_spring(
                tau_req, theta, omega, SpringSpec(kind="linear", k=k, theta0=t0),
                motor, dt, regen).total
            if e < best[0]:
                best = (e, SpringSpec(kind="linear", k=float(k), theta0=float(t0)))
    return best  # (energy_J, SpringSpec)


def ideal_cancel_spring_torque(tau_grav):
    """The perfect (nonlinear) gravity-cancelling spring torque: tau_spring = tau_grav."""
    return tau_grav


# ----------------------------------------------------------- coordinated upright lift
def reduced_model(urdf_path, keep_joints, search_root=None,
                  compiler_attrs='fusestatic="false" discardvisual="false"'):
    """A reduced MjModel keeping ONLY keep_joints; every other joint is deleted so its
    body welds rigidly to its parent (arms/head/wheels become rigid lumped mass). This
    makes multi-joint inverse dynamics on the kept chain clean (no held-but-unwelded
    joints injecting spurious multibody velocity cross-terms). fusestatic=false keeps
    the welded link bodies named (so IK can target e.g. torso_link4 -- otherwise the
    compiler fuses them away)."""
    from pea.urdf_loader import make_spec
    spec = make_spec(urdf_path, search_root, compiler_attrs=compiler_attrs)
    for j in list(spec.joints):
        if j.name not in set(keep_joints):
            spec.delete(j)
    return spec.compile()


def _joint_addrs(model):
    qa = np.array([model.jnt_qposadr[j] for j in range(model.njnt)])
    dofs = np.array([model.jnt_dofadr[j] for j in range(model.njnt)])
    return qa, dofs


def vertical_lift_ik(model, data, top_body, z_path, x0=None, pitch0=0.0,
                     q_init=None, iters=200, damp=1e-3, tol=1e-6):
    """Solve, for each target top-body height in z_path, the joint angles that keep the
    body LEVEL (pitch=pitch0) at fixed horizontal x0 -- a true vertical 'upright lift'.
    Damped-least-squares IK on the analytic body Jacobian (mj_jacBody); warm-started
    along the path. Returns q (len(z_path) x ndof).

    q_init: a SLIGHTLY-BENT seed. The fully-extended (upright) pose is a kinematic
    singularity (dz/dtheta = 0 -- no vertical leverage straight), so the IK must start
    bent or it cannot make vertical progress. x0 defaults to the seed's x."""
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, top_body)
    qa, _ = _joint_addrs(model)
    lo, hi = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))

    def fk(q):
        data.qpos[qa] = q
        mujoco.mj_forward(model, data)
        R = data.xmat[tb].reshape(3, 3)
        return data.xpos[tb].copy(), float(np.arctan2(-R[2, 0], R[0, 0]))

    seed = np.zeros(model.nv) if q_init is None else np.asarray(q_init, float)
    if x0 is None:
        x0 = fk(seed)[0][0]
    q = seed.copy(); out = []
    for zt in z_path:
        for _ in range(iters):
            pos, pitch = fk(q)
            e = np.array([pos[0] - x0, pos[2] - zt, pitch - pitch0])
            if np.abs(e).max() < tol:
                break
            mujoco.mj_jacBody(model, data, jacp, jacr, tb)
            J = np.vstack([jacp[0], jacp[2], jacr[1]])           # rows: x, z, pitch(about y)
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), -e)
            q = np.clip(q + dq, lo, hi)
        out.append(q.copy())
    return np.array(out)


def multi_joint_torque(model, data, q, dt):
    """Required motor torque per kept joint along a coordinated trajectory q(t), via the
    EXPLICIT equation of motion tau = M(q) qdd + qfrc_bias(q, qdot) on the reduced model
    (all DOFs are the moving joints). We do NOT use mj_inverse: it was verified (2026-06-18)
    to return a large SPURIOUS torque on the base joint for this model, while M qdd + bias
    matches the other joints exactly and gives the physically correct gravity-dominated
    base-joint torque. Velocities/accelerations are central differences of the smooth
    (min-jerk) q(t). Returns (tau, tau_grav, qvel), each (N x ndof)."""
    qa, dofs = _joint_addrs(model)
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    N = q.shape[0]; nd = len(dofs)
    tau = np.zeros((N, nd)); grav = np.zeros((N, nd))
    Mbuf = np.zeros((model.nv, model.nv))
    for i in range(N):
        mujoco.mj_resetData(model, data)
        data.qpos[qa] = q[i]; data.qvel[dofs] = qd[i]
        mujoco.mj_forward(model, data)            # fills qM (mass matrix) + qfrc_bias
        mujoco.mj_fullM(model, Mbuf, data.qM)
        tau[i] = (Mbuf @ qdd[i])[dofs] + data.qfrc_bias[dofs]   # M qdd + C qdot + g
        # gravity-only (qdot=0): the constant-sign load the spring targets
        mujoco.mj_resetData(model, data)
        data.qpos[qa] = q[i]; mujoco.mj_forward(model, data)
        grav[i] = data.qfrc_bias[dofs]
    return tau, grav, qd


def fit_linear_spring_per_joint(theta, tau_req, omega, motor: E.MotorConstants, dt: float,
                                regen: bool = False, k_max: float = 600.0):
    """Energy-optimal linear torsion spring tau=-k(theta-theta0) for ONE joint's
    (theta, tau_req, omega) series. Returns (SpringSpec, energy_with, energy_without)."""
    base = lift_energy(tau_req, omega, motor, dt, regen).total
    best = (np.inf, SpringSpec(kind="linear"))
    th_lo, th_hi = theta.min() - 0.6, theta.max() + 0.6
    for k in np.linspace(0, k_max, 61):
        for t0 in np.linspace(th_lo, th_hi, 49):
            spec = SpringSpec(kind="linear", k=float(k), theta0=float(t0))
            e = lift_energy(tau_req - tau_spring(theta, spec), omega, motor, dt, regen).total
            if e < best[0]:
                best = (e, spec)
    return best[1], best[0], base
