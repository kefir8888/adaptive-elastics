"""Energy AND performance metrics over a logged trajectory — robot/task/spring
agnostic.

Consolidates the one-off analysis scripts (motor_budget, power_compare,
probe_speed_hold) into reusable functions. Everything takes explicit DoF indices
and motor constants, so the same code scores ANY robot (G1, H1, Go2, DecART, …),
ANY task (stand / walk / run / jump), and ANY post-hoc spring.

Two metric families, because adaptive elastics buy two different things:
  * ENERGY   — electrical power (no-regen + regen), ohmic loss, braking power,
               cost of transport. "Does the spring save energy?"
  * PERFORMANCE — peak mechanical power, peak joint torque, top speed, jump
               height / takeoff velocity. "Does the spring let the robot run
               faster / jump higher?" (springs amplify peak power: store slowly,
               release fast, exceeding the motor's own peak — the catapult win.)

Mean powers are WATTS over the window: sum(power)*dt/duration = sum(power)/T.
"""

from __future__ import annotations

import numpy as np

from pea import energy, springs

G = energy.G


def actuated_dof_adrs(mj_model) -> list[int]:
    """DoF addresses of actuated joints (skip the 6 free-base DoFs)."""
    return [int(mj_model.jnt_dofadr[j]) for j in range(mj_model.njnt)
            if int(mj_model.jnt_dofadr[j]) >= 6]


def joint_addrs(mj_model, substr: str):
    """(names, qpos_adrs, dof_adrs) for joints whose name contains `substr`."""
    import mujoco

    names, q, d = [], [], []
    for j in range(mj_model.njnt):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and substr in nm:
            names.append(nm)
            q.append(int(mj_model.jnt_qposadr[j]))
            d.append(int(mj_model.jnt_dofadr[j]))
    if not names:
        raise ValueError(f"no joints matching {substr!r} in model")
    return names, q, d


def subtract_spring(tau_full, qpos, spec, qpos_adrs, dof_adrs):
    """Copy of tau_full with tau_spring(theta) subtracted at the target DoFs.

    Models a PARALLEL spring offloading the motor on a FIXED gait (post-hoc): the
    motor now supplies tau - tau_spring(theta). Robot/joint agnostic — pass the
    qpos/dof addresses of whichever joint carries the spring.
    """
    out = np.array(tau_full, copy=True)
    for qa, da in zip(qpos_adrs, dof_adrs):
        out[:, da] = out[:, da] - np.asarray(springs.tau_spring(qpos[:, qa], spec))
    return out


def fit_linear_spring(theta, tau, k_min: float = 0.0):
    """Least-squares linear spring tau_s = -k(theta-theta0) fitted to a joint's
    torque-angle loop — the post-hoc OPTIMUM that minimizes residual motor torque
    (the 'extract the value from the data' step, before a refining grid).

    Regress tau ~ a*theta + b; then k=-a (restoring slope), theta0=-b/a, with k
    clamped >= k_min for a passive spring. If the fit degenerates (k clamps to 0,
    i.e. an offset-dominated loop like the walking knee), theta0 falls back to the
    mean angle — that joint wants a CONSTANT element, not a linear one.
    """
    theta, tau = np.asarray(theta), np.asarray(tau)
    a, b = np.polyfit(theta, tau, 1)
    k = max(-float(a), k_min)
    theta0 = float(-b / a) if (k > k_min and abs(a) > 1e-9) else float(theta.mean())
    return k, theta0


def power_breakdown(tau, qvel, dt, kt, r) -> dict:
    """Mean watts over the window for a set of DoFs. tau, qvel: (T, k)."""
    mech = tau * qvel
    ohm = (tau / kt) ** 2 * r
    s = dt / (len(tau) * dt)  # = 1/T; sum(power)*dt/dur = mean W
    elec_noregen = float(np.maximum(mech + ohm, 0.0).sum() * s)
    return dict(
        mech_signed=float(mech.sum() * s),
        mech_pos=float(np.maximum(mech, 0).sum() * s),
        braking=float(-np.minimum(mech, 0).sum() * s),
        ohmic=float(ohm.sum() * s),
        elec_regen=float((mech + ohm).sum() * s),
        elec_noregen=elec_noregen,
        ohmic_share=float(ohm.sum() * s / max(elec_noregen, 1e-9)),
    )


def performance(traj, act, kt, r) -> dict:
    """Peak-power / speed / height metrics (the 'run faster, jump higher' axis)."""
    tau = traj["qfrc"][:, act]
    qvel = traj["qvel"][:, act]
    mech = tau * qvel
    # peak instantaneous positive mechanical power summed over actuated DoFs
    peak_mech_W = float(np.maximum(mech, 0).sum(axis=1).max())
    peak_abs_torque = float(np.abs(tau).max())
    qpos = traj["qpos"]
    # planar speed over the window; base height = floating-base z (qpos index 2)
    dt = float(traj["dt"])
    dur = max((len(qpos) - 1) * dt, 1e-9)
    mean_speed = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]) / dur)
    return dict(
        peak_mech_W=peak_mech_W,
        peak_abs_torque=peak_abs_torque,       # max tooth stress (gearbox wear)
        rms_torque=float(np.sqrt(np.mean(tau ** 2))),  # cyclic/thermal load (wear)
        mean_speed=mean_speed,
        max_base_z=float(qpos[:, 2].max()),
        takeoff_vz=float(traj["qvel"][:, 2].max()) if traj["qvel"].shape[1] > 2 else 0.0,
    )


def _vel_limit_for(name: str, vel_limits: dict, default: float) -> float:
    """Look up a joint's max angular velocity by substring match on its name."""
    for key, v in vel_limits.items():
        if key in name:
            return v
    return default


def saturation(traj, mj_model, joint_substr: str, vel_limits: dict | None = None) -> list[dict]:
    """Per joint, is the high-demand operating point against the TORQUE or SPEED
    wall? Answers "torque-limited or speed-limited?" — parallel elasticity helps
    iff torque-bound; series iff speed-bound.

    Both walls are now REAL (no estimate): the per-joint TORQUE cap is the model's
    enforced jnt_actfrcrange (knee 139, hip 88) and the SPEED ceiling is the Unitree
    G1 URDF velocity limit (energy.G1_JOINT_VEL: knee 20, hip 32 rad/s) by default.
    Uses ROBUST percentiles, not the raw max: RL control chatter leaves single-sample
    spikes (peak/RMS up to ~4 even in walking) that the max would misread as "near the
    limit"; the binding wall is judged from the MEDIAN operating point over the
    top-5%-power samples (the push-off window in a jump).
    """
    import mujoco

    if vel_limits is None:
        vel_limits = energy.G1_JOINT_VEL
    names, _, dadr = joint_addrs(mj_model, joint_substr)
    tau, w = traj["qfrc"], traj["qvel"]
    out = []
    for nm, d in zip(names, dadr):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        tau_peak = float(mj_model.jnt_actfrcrange[jid][1])  # real enforced cap
        omega_noload = _vel_limit_for(nm, vel_limits, energy.OMEGA_NOLOAD_EST)
        t, om = np.abs(tau[:, d]), np.abs(w[:, d])
        power = np.abs(tau[:, d] * w[:, d])
        hi = power >= np.percentile(power, 95)               # top-5% power window
        tau_frac = float(np.median(t[hi])) / tau_peak
        spd_frac = float(np.median(om[hi])) / omega_noload
        out.append(dict(
            joint=nm, tau_peak=tau_peak, omega_max=omega_noload,
            p99_tau_pct=round(100 * np.percentile(t, 99) / tau_peak, 1),
            p99_spd_pct=round(100 * np.percentile(om, 99) / omega_noload, 1),
            atP_tau_pct=round(100 * tau_frac, 1),
            atP_spd_pct=round(100 * spd_frac, 1),
            binding=("torque-bound" if tau_frac >= spd_frac else "speed-bound"),
        ))
    return out


def evaluate(traj, mj_model, act, kt, r, spec=None, joint_substr: str | None = None,
             mass: float | None = None) -> dict:
    """Full energy+performance record for a trajectory under an optional spring.

    traj: dict with qpos (T,nq), qvel (T,nv), qfrc (T,nv), dt, and (optional)
    total_mass. `spec` None = baseline (no spring). `joint_substr` selects the
    spring's target joint (and its energy share is reported); required if spec.
    """
    qpos, qvel, dt = traj["qpos"], traj["qvel"], float(traj["dt"])
    tau_full = traj["qfrc"]
    tq = td = []
    if spec is not None:
        if joint_substr is None:
            raise ValueError("joint_substr required when a spring spec is given")
        _, tq, td = joint_addrs(mj_model, joint_substr)
        tau_full = subtract_spring(tau_full, qpos, spec, tq, td)
    traj_eff = {**traj, "qfrc": tau_full}

    wb = power_breakdown(tau_full[:, act], qvel[:, act], dt, kt, r)
    perf = performance(traj_eff, act, kt, r)

    # target-joint share (uses the grid joint even at baseline, for reference).
    # joint_braking is the negative (braking) work this joint dissipates under
    # no-regen — the energy a spring can intercept, i.e. the gate metric.
    if joint_substr is not None:
        _, tq2, td2 = joint_addrs(mj_model, joint_substr)
        jt = power_breakdown(tau_full[:, td2], qvel[:, td2], dt, kt, r)
        joint_elec = jt["elec_noregen"]
        joint_share = joint_elec / max(wb["elec_noregen"], 1e-9)
        joint_braking = jt["braking"]
    else:
        joint_elec = joint_share = joint_braking = float("nan")

    m = mass if mass is not None else float(traj.get("total_mass", float("nan")))
    dist = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    elec_J = wb["elec_noregen"] * (len(qpos) * dt)
    cot = elec_J / (m * G * max(dist, 1e-9)) if m == m else float("nan")
    return dict(
        elec_W=wb["elec_noregen"], elec_regen_W=wb["elec_regen"],
        ohmic_W=wb["ohmic"], ohmic_pct=100 * wb["ohmic_share"],
        braking_W=wb["braking"], mech_W=wb["mech_signed"],
        cot=cot, joint_elec_W=joint_elec, joint_share_pct=100 * joint_share,
        joint_braking_W=joint_braking,
        **perf,
    )
