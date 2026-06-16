"""Probe a Go1 RUNNING policy: roll out at a commanded forward speed and report the
achieved forward speed, the flight fraction (all-feet-off), and a short gait summary
(knee/calf angle + torque, electrical power). This is the S2 GO/NO-GO gate: a true RUN
opens an all-feet-off window; a fast TROT keeps a foot down and collapses back to the
walking win.

Adapted from g1_run_probe.py for the Go1 (4-foot quadruped): the speed knob is the
joystick command (set directly in state.info here, the same way the trainer commands
it), the spring joint is the calf (Go1 knee), and foot contact comes from the env's
four '{geom}_floor_found' sensors.

Usage: go1_run_probe.py <run_dir> [steps] [cmd_speeds_comma_separated]
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
SPEEDS = ([float(x) for x in sys.argv[3].split(",")]
          if len(sys.argv) > 3 else [2.2, 2.6, 3.0, 3.2, 3.5])

cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)
mj = env.mj_model
kn = joints_by_substring(mj, "calf")             # Go1 knee == calf joint
qadr = np.array([v["qpos_adr"] for v in kn.values()])
dadr = np.array([v["dof_adr"] for v in kn.values()])
act = metrics.actuated_dof_adrs(mj)
# jnp array (not a Python list): jax forbids list-based multidimensional indexing.
contact_adr = jnp.array(metrics.go1_foot_contact_sensor_adrs(mj))
mc = energy.motor_constants(cfg.energy_motor)
kt, r = mc.kt, mc.r
dt = float(env.dt)
pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
jr, js, jp = jax.jit(env.reset), jax.jit(env.step), jax.jit(pol)
print(f"# {run.name}  calf joints {list(kn.keys())}  steps={STEPS}", flush=True)

for vx in SPEEDS:
    cmd = jnp.array([vx, 0.0, 0.0], dtype=jnp.float32)
    rng = jax.random.PRNGKey(0)
    st = jr(rng)
    QP, KQ, KT, QV, QF, CT, n = [], [], [], [], [], [], 0
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
        CT.append(np.asarray(st.data.sensordata[contact_adr]))
        if bool(st.done):
            break
    if not QP:                                   # fell on step 0 -> nothing to stack
        print(f"cmd {vx:.1f} m/s: survived 0/{STEPS}  (no steps; skipped)", flush=True)
        continue
    QP = np.stack(QP)
    w0 = n // 2                                  # steady window: last half
    # P0 fix: a window from index w0 to n-1 spans (n-1-w0) sample intervals, so the
    # duration is (n-1-w0)*dt -- matches metrics.performance (dur = (len-1)*dt).
    spd = float(np.linalg.norm(QP[-1][:2] - QP[w0][:2]) / max((n - 1 - w0) * dt, 1e-9))
    kq = np.stack(KQ)[w0:]
    ktq = np.stack(KT)[w0:]
    flight = metrics.flight_fraction(np.stack(CT)[w0:], min_grounded_run=2)
    elec = metrics.power_breakdown(
        np.stack(QF)[w0:][:, act], np.stack(QV)[w0:][:, act], dt, kt, r
    )["elec_noregen"]
    print(
        f"cmd {vx:.1f} m/s: survived {n}/{STEPS}  speed {spd:.2f} m/s  elec {elec:5.0f} W  "
        f"| flight {100*flight['flight_fraction']:.1f}%  feet_down_min {flight['n_feet_down_min']}  "
        f"max_flight_run {flight['max_flight_run']}  true_flight {flight['true_flight']}  "
        f"| calf angle mean {kq.mean():+.2f} range[{kq.min():+.2f},{kq.max():+.2f}] rad  "
        f"| calf |tau| mean {np.abs(ktq).mean():.1f} Nm",
        flush=True,
    )
print("PROBE_DONE")
