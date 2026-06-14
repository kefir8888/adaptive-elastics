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
import jax
import jax.numpy as jnp

from pea import config as cfg_lib, energy, metrics, policy as policy_lib
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
go1 = env                                   # resolve the base Go1 env (holds _mjx_model)
while hasattr(go1, "_env"):
    go1 = go1._env
base = float(go1._mjx_model.body_mass[1])
mj = env.mj_model
act = metrics.actuated_dof_adrs(mj)
names, qa, da = metrics.joint_addrs(mj, "calf")
da = np.array(da)
mc = energy.motor_constants(cfg.energy_motor)
kt, r = mc.kt, mc.r
dt = float(env.dt)
cmd = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
ADAPT = (cfg.spring.kind == "preload_dr")
ALPHA = float(np.exp(-dt / 15.0))           # 15 s EMA
KP, RATE, ETARGET = 0.2, 2.0, 0.0
TMAX = float(cfg.spring.tau0) if ADAPT else 0.0
n = len(da)
print(f"# base trunk {base:.2f} kg; mode={'ADAPTIVE preload' if ADAPT else 'baseline (no preload)'}; steps={STEPS}", flush=True)


def at_payload(P):
    go1._mjx_model = go1._mjx_model.replace(
        body_mass=go1._mjx_model.body_mass.at[1].set(base + P))
    jr, js, jp = jax.jit(env.reset), jax.jit(env.step), jax.jit(pol)
    rng = jax.random.PRNGKey(0)
    st = jr(rng)
    tau0 = np.zeros(n)
    ema = np.zeros(n)
    QV, QF, QP, nstep = [], [], [], 0
    for i in range(STEPS):
        if ADAPT:
            st.info["preload_tau0"] = jnp.asarray(tau0)
        if "command" in st.info:
            st.info["command"] = cmd
        rng, ar = jax.random.split(rng)
        a, _ = jp(st.obs, ar)
        st = js(st, a)
        nstep += 1
        QV.append(np.asarray(st.data.qvel))
        QF.append(np.asarray(st.data.qfrc_actuator))
        QP.append(np.asarray(st.data.qpos))
        if ADAPT:
            cm = QF[-1][da]                                  # per-leg motor torque (after offload)
            ema = ALPHA * ema + (1 - ALPHA) * cm
            tau0 = np.clip(tau0 + np.clip(KP * (ema - ETARGET), -RATE, RATE) * dt, 0.0, TMAX)
        if bool(st.done):
            break
    w0 = int(nstep * 0.5)                                     # steady window: last half (preload converged)
    if nstep > w0 + 100:
        QF2, QV2 = np.stack(QF)[w0:], np.stack(QV)[w0:]
        elec = metrics.power_breakdown(QF2[:, act], QV2[:, act], dt, kt, r)["elec_noregen"]
        ctau = float(np.mean(np.abs(np.stack(QF)[w0:][:, da])))
    else:
        elec, ctau = float("nan"), float("nan")
    QPa = np.stack(QP)
    spd = float(np.linalg.norm(QPa[-1][:2] - QPa[w0][:2])) / max((nstep - w0) * dt, 1e-9)
    extra = f"  conv_tau0 {np.round(tau0, 1)}" if ADAPT else ""
    print(f"P={P:5.1f}kg: survived {nstep}/{STEPS}  speed {spd:.2f} m/s  elec {elec:6.1f} W  "
          f"mean|calf_mot| {ctau:5.1f}{extra}", flush=True)


for P in [0, 2.5, 5, 7.5, 10, 12.5, 15]:   # 0-10 in-distribution + 12.5/15 OOD (baseline degrades first)
    at_payload(P)
print("CAPACITY_DONE")
