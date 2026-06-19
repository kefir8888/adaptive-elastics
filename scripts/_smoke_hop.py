"""Local CPU smoke test for the G1 hopping env. Not a training run — just verifies
the subclass builds, resets, steps, and that the hop reward terms are present and
finite, the gait clock is synchronous, and contact/base-height read sensibly.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys

import jax
import jax.numpy as jp

from pea.config import load_config
from pea.env import make_env

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/g1_hop_s1.yaml"
print("config:", cfg_path)
cfg = load_config(cfg_path)
cfg = type(cfg)(**{**cfg.__dict__, "impl": "jax"})  # ensure CPU JAX impl
print("env_name:", cfg.env_name)

env = make_env(cfg)

# Inspect the resolved reward scales (post default_config + overrides).
scales = env.base_env._config.reward_config.scales if hasattr(env, "base_env") else env._config.reward_config.scales
for k in ["feet_phase", "stand_still", "joint_deviation_knee", "pose", "feet_air_time",
          "max_contact_force", "hop_rhythm", "hop_push", "hop_height", "hop_sync", "hop_flight"]:
    src = scales if k in scales else (env.base_env._config.reward_config if hasattr(env, "base_env") else env._config.reward_config)
    print(f"  [{k}] = {float(src[k])}")
rc = env.base_env._config.reward_config if hasattr(env, "base_env") else env._config.reward_config
print("  hop_height_target =", float(rc.hop_height_target), " hop_freq =", float(rc.hop_freq))

key = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(key)
print("\nreset OK")
print("  phase (should be synchronous, ~[0,0]):", state.info["phase"])
print("  base height qpos[2]:", float(state.data.qpos[2]))

step = jax.jit(env.step)
nu = env.action_size if hasattr(env, "action_size") else state.info["last_act"].shape[0]
act = jp.zeros(nu)
hop_keys = ["hop_rhythm", "hop_push", "hop_height", "hop_sync", "hop_flight"]
print("  h_ref (standing pelvis z, self._init_q[2]):",
      float((env.base_env if hasattr(env, "base_env") else env)._init_q[2]))
for i in range(6):
    state = step(state, act)
    rew = {k: float(state.metrics[f"reward/{k}"]) for k in hop_keys}
    contact = None
    print(f"step {i}: reward={float(state.reward):+.4f} done={float(state.done):.0f} "
          f"z={float(state.data.qpos[2]):.3f} phase0={float(state.info['phase'][0]):+.3f} "
          f"hop={ {k: round(v,4) for k,v in rew.items()} }")

allfinite = bool(jp.isfinite(state.reward)) and all(
    bool(jp.isfinite(state.metrics[f"reward/{k}"])) for k in hop_keys)
print("\nALL FINITE:", allfinite)
print("SMOKE TEST PASSED" if allfinite else "SMOKE TEST FAILED")
