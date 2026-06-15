"""Probe a G1 running policy: sweep commanded forward speed, report achieved speed +
survival + electrical power, AND the knee angle/torque stats over the steady gait — the
data that sets the parallel knee spring's equilibrium angle and stiffness.

Usage: g1_run_probe.py <run_dir> [steps]
"""
import sys
import pathlib
import numpy as np
import jax
import jax.numpy as jnp

from pea import config as cfg_lib, energy, metrics, policy as policy_lib
from pea.env import make_env, joints_by_substring

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
mj = env.mj_model
kn = joints_by_substring(mj, "knee")
qadr = np.array([v["qpos_adr"] for v in kn.values()])
dadr = np.array([v["dof_adr"] for v in kn.values()])
act = metrics.actuated_dof_adrs(mj)
mc = energy.motor_constants(cfg.energy_motor)
kt, r = mc.kt, mc.r
dt = float(env.dt)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
jr, js, jp = jax.jit(env.reset), jax.jit(env.step), jax.jit(pol)
print(f"# {run.name}  knee joints {list(kn.keys())}", flush=True)

for vx in [1.0, 1.5, 2.0, 2.5, 3.0]:
    cmd = jnp.array([vx, 0.0, 0.0], dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    st = jr(rng)
    QP, KQ, KT, QV, QF, n = [], [], [], [], [], 0
    for i in range(STEPS):
        if "command" in st.info:
            st.info["command"] = cmd
        rng, ar = jax.random.split(rng)
        a, _ = jp(st.obs, ar)
        st = js(st, a)
        n += 1
        QP.append(np.asarray(st.data.qpos))
        KQ.append(np.asarray(st.data.qpos[qadr]))
        KT.append(np.asarray(st.data.qfrc_actuator[dadr]))
        QV.append(np.asarray(st.data.qvel))
        QF.append(np.asarray(st.data.qfrc_actuator))
        if bool(st.done):
            break
    QP = np.stack(QP)
    w0 = n // 2
    spd = float(np.linalg.norm(QP[-1][:2] - QP[w0][:2]) / max((n - w0) * dt, 1e-9))
    kq = np.stack(KQ)[w0:]
    ktq = np.stack(KT)[w0:]
    elec = metrics.power_breakdown(
        np.stack(QF)[w0:][:, act], np.stack(QV)[w0:][:, act], dt, kt, r
    )["elec_noregen"]
    print(
        f"cmd {vx:.1f} m/s: survived {n}/{STEPS}  speed {spd:.2f} m/s  elec {elec:5.0f} W  "
        f"| knee angle mean {kq.mean():+.2f} range[{kq.min():+.2f},{kq.max():+.2f}] rad  "
        f"| knee |tau| mean {np.abs(ktq).mean():.1f} Nm",
        flush=True,
    )
print("PROBE_DONE")
