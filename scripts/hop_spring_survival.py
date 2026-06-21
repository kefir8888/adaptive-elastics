"""NOMINAL-dynamics survival gate for a (spring) hop policy.

Rolls the policy out on the NOMINAL (un-randomized) model with the spring ACTIVE,
deterministically, and reports per-seed survival + apex. Use it RIGHT AFTER the k=40
ramp stage in the fair re-run: it confirms the centered-DR fix actually made the spring
nominally robust BEFORE spending the k=75/k=93 box time. The stock-DR spring fell in
~2 s here (0/3); a fixed one should survive the full episode and hop.

PASS (exit 0) = all seeds survive the full episode AND hop (apex > 0). Else exit 1.
Usage: hop_spring_survival.py <run_dir> [steps] [n_seeds]
"""
from __future__ import annotations
import sys
import pathlib
import numpy as np
from pea import config as cfg_lib, experiment, policy as policy_lib
from pea.env import make_env

run = pathlib.Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
NSEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3

cfg = cfg_lib.load_config(run / "config.yaml")
env = make_env(cfg)  # spring ACTIVE if the run has one
if (run / "policy_params").exists():
    pol = policy_lib.load_policy(env, cfg, run / "policy_params", deterministic=True)
else:
    pol = policy_lib.load_policy_from_checkpoint(
        env, cfg, policy_lib.latest_checkpoint(run), deterministic=True)
dt = float(env.dt)
b = env
while hasattr(b, "_env"):
    b = b._env
H = float(b._init_q[2])

print(f"NOMINAL survival of {run.name} (spring={cfg.spring.kind} k={cfg.spring.k}):")
ok = 0
for seed in range(NSEED):
    tr = experiment.rollout(env, pol, (0.0, 0.0, 0.0), STEPS, seed=seed)
    n = tr["n"]
    apex = float(np.max(tr["qpos"][:, 2]) - H) if n > 10 else float("nan")
    survived = n >= STEPS and apex > 0.0
    ok += survived
    print(f"  seed {seed}: {n} steps ({n*dt:.1f}s)  apex {apex:+.3f}  "
          f"{'SURVIVED+HOPS' if survived else ('FELL' if n < STEPS else 'no hop')}")
passed = ok == NSEED
print(f"\nNOMINAL SURVIVAL GATE: {ok}/{NSEED} -> "
      f"{'*** PASS *** (centered DR fixed it — continue the ramp)' if passed else '*** FAIL *** (still brittle — STOP, rethink)'}")
sys.exit(0 if passed else 1)
