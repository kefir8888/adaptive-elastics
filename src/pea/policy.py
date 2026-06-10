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


def ppo_params_for(cfg: RunConfig):
    """Playground's recommended brax PPO config (ml_collections ConfigDict)."""
    return locomotion_params.brax_ppo_config(cfg.env_name, cfg.impl)


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
