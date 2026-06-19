"""Policy networks: construction, save/load, inference.

Reconstruction must mirror brax's ppo.train internals exactly (same network
factory kwargs from Playground's recommended config, same observation
preprocessor), so params saved on the GPU trainer slot into a CPU inference
function locally. The CPU smoke test exercises this round-trip.

Consequence: PPO overrides in run configs must NOT touch network architecture,
otherwise reload here would mismatch the trained params.
"""

from __future__ import annotations

import functools
import pathlib

from brax.io import model
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

from pea.config import RunConfig


# Custom env subclasses (g1_run_env, g1_hop_env) share their base robot's obs/action
# sizes and network, so they reuse the stock base's tuned PPO config. brax_ppo_config
# keys on a FIXED env-name list and raises "Unsupported env" otherwise, so map the
# custom names to the stock base for that lookup. (env_name itself stays the custom
# one everywhere else — only the hyperparameter lookup is aliased.)
_PPO_CONFIG_ALIAS = {
    "G1JoystickHop": "G1JoystickFlatTerrain",
    "G1JoystickRun": "G1JoystickFlatTerrain",
}


def ppo_params_for(cfg: RunConfig):
    """Playground's recommended brax PPO config (ml_collections ConfigDict)."""
    name = _PPO_CONFIG_ALIAS.get(cfg.env_name, cfg.env_name)
    return locomotion_params.brax_ppo_config(name, cfg.impl)


def network_factory_for(cfg: RunConfig):
    ppo_params = ppo_params_for(cfg)
    kwargs = (
        dict(ppo_params.network_factory)
        if "network_factory" in ppo_params
        else {}
    )
    return functools.partial(ppo_networks.make_ppo_networks, **kwargs)


def save_params(path: str | pathlib.Path, params) -> None:
    model.save_params(str(path), params)


def latest_checkpoint(path: pathlib.Path) -> pathlib.Path:
    """Resolve a run dir / checkpoints dir to its newest step directory."""
    if (path / "checkpoints").is_dir():
        path = path / "checkpoints"
    steps = [p for p in path.glob("*") if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {path}")
    return max(steps, key=lambda p: int(p.name))


def load_policy_from_checkpoint(env, cfg: RunConfig, ckpt_dir, deterministic: bool = True):
    """Inference fn from a mid-training orbax checkpoint (no policy_params).

    brax's own checkpoint.load_policy chokes on configs whose kernel-init
    names were saved as null (KERNEL_INITIALIZER[None] KeyError), so we load
    only the params from orbax and rebuild the network exactly like
    load_policy does — from Playground's recommended factory kwargs.
    """
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    ppo_params = ppo_params_for(cfg)
    preprocess = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", True)
        else types.identity_observation_preprocessor
    )
    nets = network_factory_for(cfg)(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess,
    )
    make_policy = ppo_networks.make_inference_fn(nets)
    params = ppo_checkpoint.load(str(ckpt_dir))
    return make_policy(params, deterministic=deterministic)


def load_policy(
    env,
    cfg: RunConfig,
    params_path: str | pathlib.Path,
    deterministic: bool = True,
):
    """Rebuild the inference function for a trained policy from saved params."""
    ppo_params = ppo_params_for(cfg)
    preprocess = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", True)
        else types.identity_observation_preprocessor
    )
    nets = network_factory_for(cfg)(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess,
    )
    make_policy = ppo_networks.make_inference_fn(nets)
    params = model.load_params(str(params_path))
    return make_policy(params, deterministic=deterministic)
