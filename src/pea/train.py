"""Train a G1 walking policy with brax PPO.

The same script produces baseline and spring-active runs — the spring is
selected by the YAML config, never hardcoded here (CLAUDE.md design rule).
Runs on Colab GPU (real training) and local CPU (--smoke pipeline test).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import time

# Tiny-but-real PPO settings for the CPU pipeline test. Network architecture
# is deliberately NOT changed: policy.load_policy rebuilds nets from the
# env's recommended config, so smoke checkpoints must reload with it too.
SMOKE_PARAMS = dict(
    num_timesteps=30_000,
    num_envs=32,
    num_eval_envs=32,
    batch_size=32,
    num_minibatches=4,
    unroll_length=8,
    num_updates_per_batch=2,
    episode_length=200,
    num_evals=3,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML run config")
    p.add_argument(
        "--output_dir",
        default=None,
        help="runs root (default: $PEA_RUNS_DIR / Drive / ./outputs)",
    )
    p.add_argument("--suffix", default="", help="run-folder name suffix")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU run to validate the full pipeline before Colab",
    )
    p.add_argument(
        "--restore",
        default=None,
        help="run dir (or checkpoints dir) to resume from (restores weights; "
        "the step counter restarts — combine with --num_timesteps to train "
        "only the remainder)",
    )
    p.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="override config/recommended step budget",
    )
    p.add_argument(
        "--energy-weight",
        type=float,
        default=None,
        dest="energy_weight",
        help="override energy_reward_weight (total-electrical penalty weight; "
        "for the calibration sweep)",
    )
    p.add_argument(
        "--spring_k",
        type=float,
        default=None,
        help="override spring stiffness k and force k_dr off (staged stiffness "
        "ramp: warm-start-chain runs with increasing k so the policy migrates "
        "gradually to full strength)",
    )
    args = p.parse_args()

    # Must be set before jax initializes: the default 0.75 memory fraction
    # wastes 4 GB of a 16 GB T4. (--xla_gpu_triton_gemm_any was tried here
    # and removed: on driver-595/CUDA-13 boxes the triton autotuner fails
    # every config with xtile DEVICE_TYPE_INVALID and compile takes forever.)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

    import jax
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    from pea import config as cfg_lib
    from pea import policy as policy_lib
    from pea.env import make_env

    cfg = cfg_lib.load_config(args.config)
    if args.energy_weight is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, energy_reward_weight=args.energy_weight)
    if args.spring_k is not None:
        import dataclasses
        cfg = dataclasses.replace(
            cfg, spring=dataclasses.replace(cfg.spring, k=args.spring_k, k_dr=False))
        print(f"[pea-train] spring stiffness override: k={args.spring_k} (k_dr forced off)")
    suffix = "smoke" if args.smoke else args.suffix
    run_dir = cfg_lib.new_run_dir(cfg, root=args.output_dir, suffix=suffix)
    ckpt_dir = (run_dir / "checkpoints").resolve()
    print(f"[pea-train] run dir: {run_dir}")
    print(f"[pea-train] jax backend: {jax.default_backend()}")

    ppo_params = policy_lib.ppo_params_for(cfg)
    training_params = dict(ppo_params)
    training_params.pop("network_factory", None)
    if cfg.num_timesteps is not None:
        training_params["num_timesteps"] = cfg.num_timesteps
    if "network_factory" in cfg.ppo:
        raise ValueError(
            "ppo overrides must not change network architecture "
            "(breaks policy reload — see pea/policy.py)"
        )
    training_params.update(cfg.ppo)
    if args.smoke:
        training_params.update(SMOKE_PARAMS)
    if args.num_timesteps is not None:
        training_params["num_timesteps"] = args.num_timesteps
    if cfg.domain_randomization and not args.smoke:
        if cfg.payload_max_kg > 0.0:
            from pea import payload
            training_params["randomization_fn"] = payload.payload_randomizer(
                cfg.payload_max_kg
            )
        else:
            training_params["randomization_fn"] = registry.get_domain_randomizer(
                cfg.env_name
            )

    restore = (
        str(policy_lib.latest_checkpoint(pathlib.Path(args.restore).resolve()))
        if args.restore
        else None
    )
    if restore:
        print(f"[pea-train] resuming from {restore}")

    num_timesteps = training_params["num_timesteps"]
    t0 = time.monotonic()
    last = {"t": t0, "steps": 0}

    def progress(num_steps: int, metrics: dict) -> None:
        now = time.monotonic()
        row = {"num_steps": int(num_steps), "wall_time_s": now - t0}
        row.update({k: float(v) for k, v in metrics.items()})
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        reward = metrics.get("eval/episode_reward", float("nan"))
        dsteps, dt = num_steps - last["steps"], now - last["t"]
        if dsteps > 0 and dt > 0:
            sps = dsteps / dt
            eta_min = (num_timesteps - num_steps) / sps / 60
            print(
                f"step {num_steps:>12,}  reward {reward:8.3f}  "
                f"{sps:,.0f} env-steps/s  ETA {eta_min:.0f} min",
                flush=True,  # stdout is block-buffered under nohup
            )
        else:
            print(
                f"step {num_steps:>12,}  reward {reward:8.3f}  (JIT compiling)",
                flush=True,
            )
        last["t"], last["steps"] = now, num_steps

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=policy_lib.network_factory_for(cfg),
        seed=cfg.seed,
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=restore,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    _, params, _ = train_fn(
        environment=make_env(cfg),
        eval_env=make_env(cfg),
        progress_fn=progress,
    )

    policy_lib.save_params(run_dir / "policy_params", params)
    print(f"[pea-train] done in {(time.monotonic() - t0) / 60:.1f} min")
    print(f"[pea-train] policy saved to {run_dir / 'policy_params'}")


if __name__ == "__main__":
    main()
