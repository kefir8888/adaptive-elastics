"""Two no-training probes on the existing G1 baseline policy (Directions 6 & 4).

Rolls the trained baseline walker at several joystick commands WITHOUT retraining
and WITHOUT overwriting the canonical trajectory.npz, and for each command reports
the whole-body motor electrical power, ohmic share, braking (negative-work) power,
and the hip-pitch linear-spring post-hoc saving.

- command [0,0,0]  = STANDING/HOLDING probe (Direction 4: static load).
- command [1.5..],  = elevated-speed / RUNNING-like probe (Direction 6).

The walker was trained at 1.0 m/s, so it will not truly run; this is a lower
bound on how the ohmic share and braking energy move with speed on this platform.
"""

from __future__ import annotations

import pathlib

import numpy as np

RUN = pathlib.Path(
    "/Users/elijah/Library/CloudStorage/GoogleDrive-kefir8888@gmail.com/"
    "Мой диск/pea_runs/2026-06-11_baseline_h100"
)
KT, R = 2.3, 0.013
TRANSIENT_S = 2.0
STEPS = 400
COMMANDS = [
    ("stand/hold", [0.0, 0.0, 0.0]),
    ("walk 1.0", [1.0, 0.0, 0.0]),
    ("fast 1.5", [1.5, 0.0, 0.0]),
    ("fast 2.0", [2.0, 0.0, 0.0]),
]


def main():
    import jax
    import jax.numpy as jnp

    from pea import config as cfg_lib
    from pea import energy, springs
    from pea import policy as policy_lib
    from pea.env import joints_by_substring, make_env

    cfg = cfg_lib.load_config("configs/baseline.yaml")  # clean (run cfg has stale k3)
    env = make_env(cfg)
    policy = policy_lib.load_policy(env, cfg, RUN / "policy_params", deterministic=True)
    model = env.mj_model

    act = [int(model.jnt_dofadr[j]) for j in range(model.njnt)
           if int(model.jnt_dofadr[j]) >= 6]
    hip = joints_by_substring(model, "hip_pitch")
    hip_dof = [v["dof_adr"] for v in hip.values()]
    hip_qpos = [v["qpos_adr"] for v in hip.values()]
    spec = springs.from_config(
        cfg_lib.load_config("configs/spring_hip_linear.yaml").spring)

    jit_reset, jit_step, jit_policy = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    dt = float(env.dt)
    skip = int(TRANSIENT_S / dt)

    print(f"\nNo-training probes on the baseline walker (Kt={KT}, R={R}; no-regen).")
    print(f"{'command':12s} {'v_act':>6s} {'elec_W':>7s} {'ohmic%':>6s} "
          f"{'brake_W':>7s} {'hipSpring saving':>18s}")
    for label, cmd in COMMANDS:
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)
        command = jnp.array(cmd, dtype=jnp.float32)
        qpos, qvel, qfrc = [], [], []
        for _ in range(STEPS):
            if "command" in state.info:
                state.info["command"] = command
            rng, ar = jax.random.split(rng)
            action, _ = jit_policy(state.obs, ar)
            state = jit_step(state, action)
            if bool(state.done):
                break
            qpos.append(np.asarray(state.data.qpos))
            qvel.append(np.asarray(state.data.qvel))
            qfrc.append(np.asarray(state.data.qfrc_actuator))
        qpos = np.stack(qpos); qvel = np.stack(qvel); qfrc = np.stack(qfrc)
        n = len(qpos)
        s = max(min(skip, n - 10), 0)
        dur = (n - s) * dt
        v_act = float(np.linalg.norm(qpos[-1, :2] - qpos[s, :2]) / max(dur, 1e-9))

        tau = qfrc[s:][:, act]; w = qvel[s:][:, act]
        mech = tau * w; ohm = (tau / KT) ** 2 * R
        sc = dt / dur
        elec = np.maximum(mech + ohm, 0).sum() * sc
        ohm_share = 100 * ohm.sum() * sc / max(elec, 1e-9)
        brake = -np.minimum(mech, 0).sum() * sc

        # hip-pitch post-hoc spring
        th = qpos[s:][:, hip_qpos]
        t0 = qfrc[s:][:, hip_dof]
        t1 = t0 - springs.tau_spring(th, spec)
        def noregen(t, om):
            return np.maximum(t * om + (t / KT) ** 2 * R, 0).sum() * sc
        e_hip0 = noregen(t0, qvel[s:][:, hip_dof])
        e_hip1 = noregen(t1, qvel[s:][:, hip_dof])
        saved = e_hip0 - e_hip1
        pct = 100 * saved / max(e_hip0, 1e-9)

        print(f"{label:12s} {v_act:6.2f} {elec:7.1f} {ohm_share:6.1f} "
              f"{brake:7.1f}  {saved:5.1f} W ({pct:+4.1f}% hip)")


if __name__ == "__main__":
    main()
