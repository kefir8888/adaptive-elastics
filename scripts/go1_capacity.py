"""Payload-capacity + energy curves for Go1 — baseline OR adaptive-preload spring.

Rolls out a trained policy at a sweep of FIXED payloads (real mass on the trunk) and
reports survival (capacity ceiling) + steady-window electrical per payload. If the run
uses the `preload_dr` spring, the per-leg ADAPTIVE controller runs in the loop: each
knee's motor torque is EMA-averaged (~15 s) and the passive preload tau0 ramps
(clipped-proportional, <=2 N*m/s, kp=0.2) to drive that mean toward ~0 (full compensation).
The policy (trained robust to any preload via DR) rides the slow preload change.

Usage: go1_capacity.py <run_dir> [steps]
"""
import sys
import pathlib
import numpy as np
import jax.numpy as jnp

from pea import config as cfg_lib, energy, experiment, metrics, policy as policy_lib
from pea.control import AdaptivePreloadController
from pea.env import make_env
from pea.payload import TORSO_BODY_ID

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
go1 = getattr(env, "base_env", env)         # innermost Go1 env (holds _mjx_model)
base = float(go1._mjx_model.body_mass[TORSO_BODY_ID])
mj = env.mj_model
act = metrics.actuated_dof_adrs(mj)
names, qa, da = metrics.joint_addrs(mj, "calf")
da = np.array(da)
mc = energy.motor_constants(cfg.energy_motor)
kt, r = mc.kt, mc.r
dt = float(env.dt)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
ADAPT = (cfg.spring.kind == "preload_dr")
TMAX = float(cfg.spring.tau0) if ADAPT else 0.0
n = len(da)
print(f"# base trunk {base:.2f} kg; mode={'ADAPTIVE preload' if ADAPT else 'baseline (no preload)'}; steps={STEPS}", flush=True)


def at_payload(P):
    go1._mjx_model = go1._mjx_model.replace(
        body_mass=go1._mjx_model.body_mass.at[TORSO_BODY_ID].set(base + P))
    ctrl = AdaptivePreloadController(n, dt, TMAX) if ADAPT else None
    # pre_step writes the slowly-ramping preload into state.info before each step;
    # callback feeds the per-leg motor torque (after offload) back to the controller.
    pre_step = (lambda i, st: st.info.__setitem__("preload_tau0", jnp.asarray(ctrl.tau0))) if ADAPT else None
    callback = (lambda i, st: ctrl.update(np.asarray(st.data.qfrc_actuator)[da])) if ADAPT else None
    traj = experiment.rollout(env, pol, (1.0, 0.0, 0.0), STEPS,
                              pre_step=pre_step, callback=callback, record_terminal=True)
    QV, QF, QP, nstep = traj["qvel"], traj["qfrc"], traj["qpos"], traj["n"]
    w0 = int(nstep * 0.5)                                     # steady window: last half (preload converged)
    if nstep > w0 + 100:
        QF2, QV2 = QF[w0:], QV[w0:]
        elec = metrics.power_breakdown(QF2[:, act], QV2[:, act], dt, kt, r)["elec_noregen"]
        ctau = float(np.mean(np.abs(QF2[:, da])))
        spd = float(np.linalg.norm(QP[-1][:2] - QP[w0][:2])) / max((nstep - 1 - w0) * dt, 1e-9)
    else:
        # too short to have a steady window: do NOT report a meaningless 0.00 speed.
        elec, ctau, spd = float("nan"), float("nan"), float("nan")
    extra = f"  conv_tau0 {np.round(ctrl.tau0, 1)}" if ADAPT else ""
    print(f"P={P:5.1f}kg: survived {nstep}/{STEPS}  speed {spd:.2f} m/s  elec {elec:6.1f} W  "
          f"mean|calf_mot| {ctau:5.1f}{extra}", flush=True)


for P in [0, 2.5, 5, 7.5, 10, 12.5, 15]:   # 0-10 in-distribution + 12.5/15 OOD (baseline degrades first)
    at_payload(P)
print("CAPACITY_DONE")
