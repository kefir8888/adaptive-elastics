"""Probe a G1 running policy: sweep commanded forward speed, report achieved speed +
survival + electrical power, AND the knee angle/torque stats over the steady gait — the
data that sets the parallel knee spring's equilibrium angle and stiffness.

Usage: g1_run_probe.py <run_dir> [steps]
"""
import sys
import pathlib
import numpy as np

from pea import config as cfg_lib, energy, experiment, metrics, policy as policy_lib
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
print(f"# {run.name}  knee joints {list(kn.keys())}", flush=True)

for vx in [1.0, 1.5, 2.0, 2.5, 3.0]:
    # Per-step knee angle (KQ) and knee torque (KTAU) collected via the rollout
    # callback so they align index-for-index with the harness qpos/qvel/qfrc.
    KQ, KTAU = [], []

    def _probe(i, st, KQ=KQ, KTAU=KTAU):
        KQ.append(np.asarray(st.data.qpos[qadr]))
        KTAU.append(np.asarray(st.data.qfrc_actuator[dadr]))

    traj = experiment.rollout(env, pol, (vx, 0.0, 0.0), STEPS,
                              callback=_probe, record_terminal=True)
    QP, QV, QF, n = traj["qpos"], traj["qvel"], traj["qfrc"], traj["n"]
    w0 = n // 2
    # P0 fix: a window from index w0 to n-1 spans (n-1-w0) sample intervals, so its
    # duration is (n-1-w0)*dt -- matches metrics.performance (dur = (len-1)*dt).
    spd = float(np.linalg.norm(QP[-1][:2] - QP[w0][:2]) / max((n - 1 - w0) * dt, 1e-9))
    kq = np.stack(KQ)[w0:]
    ktq = np.stack(KTAU)[w0:]
    elec = metrics.power_breakdown(
        QF[w0:][:, act], QV[w0:][:, act], dt, kt, r
    )["elec_noregen"]
    print(
        f"cmd {vx:.1f} m/s: survived {n}/{STEPS}  speed {spd:.2f} m/s  elec {elec:5.0f} W  "
        f"| knee angle mean {kq.mean():+.2f} range[{kq.min():+.2f},{kq.max():+.2f}] rad  "
        f"| knee |tau| mean {np.abs(ktq).mean():.1f} Nm",
        flush=True,
    )
print("PROBE_DONE")
