"""Payload domain randomization for the Go1 — a box of random mass on the trunk.

Extends MuJoCo Playground's Go1 `domain_randomize` by widening the torso-mass term
from the stock +U(-1, 1) kg to **+U(0, payload_max) kg**, so a single, payload-BLIND
controller learns to walk under any box up to `payload_max`. Everything else (floor
friction, DoF friction/armature, CoM jitter, mass scale, qpos0 jitter) is byte-identical
to the stock Go1 randomizer, so the baseline and spring conditions stay matched and the
ONLY difference between them is the spring.

The controller does not observe the payload (real-world realism: a dog can't read the
box's mass). The payload is fixed within an episode and resampled per reset (no
within-episode ramp). Pushing payload_max high enough that the baseline FAILS to walk is
the point: the constant knee preload offloads exactly the extra support torque, so the
spring should extend the carry-capacity past where the motor alone tops out.
"""
from __future__ import annotations

import jax
from mujoco import mjx

# Go1 scene indices (from mujoco_playground/_src/locomotion/go1/randomize.py).
FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def payload_randomizer(payload_max_kg: float):
    """Return a brax-PPO randomization_fn(model, rng) -> (model, in_axes) that adds a
    U(0, payload_max_kg) payload to the Go1 trunk, plus the stock Go1 randomization."""

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand_dynamics(rng):
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
                jax.random.uniform(key, minval=0.4, maxval=1.0)
            )
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
                key, shape=(12,), minval=0.9, maxval=1.1
            )
            dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)
            rng, key = jax.random.split(rng)
            armature = model.dof_armature[6:] * jax.random.uniform(
                key, shape=(12,), minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[6:].set(armature)
            rng, key = jax.random.split(rng)
            dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
            body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
                model.body_ipos[TORSO_BODY_ID] + dpos
            )
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
            # PAYLOAD: a box of +U(0, payload_max) kg on the trunk (stock is +U(-1, 1)).
            rng, key = jax.random.split(rng)
            payload = jax.random.uniform(key, minval=0.0, maxval=payload_max_kg)
            body_mass = body_mass.at[TORSO_BODY_ID].set(
                body_mass[TORSO_BODY_ID] + payload
            )
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0.at[7:].set(
                model.qpos0[7:]
                + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
            )
            return (geom_friction, body_ipos, body_mass, qpos0,
                    dof_frictionloss, dof_armature)

        (friction, body_ipos, body_mass, qpos0,
         dof_frictionloss, dof_armature) = rand_dynamics(rng)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "geom_friction": 0, "body_ipos": 0, "body_mass": 0,
            "qpos0": 0, "dof_frictionloss": 0, "dof_armature": 0,
        })
        model = model.tree_replace({
            "geom_friction": friction, "body_ipos": body_ipos, "body_mass": body_mass,
            "qpos0": qpos0, "dof_frictionloss": dof_frictionloss, "dof_armature": dof_armature,
        })
        return model, in_axes

    return domain_randomize
