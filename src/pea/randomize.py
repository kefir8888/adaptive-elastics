"""Domain randomizer CENTERED on the nominal model.

The stock MuJoCo-Playground G1 randomizer (`g1/randomize.py`) puts the nominal
(un-randomized) model at the BOUNDARY of the DR set on the damping axes:
  * `dof_armature *= U(1.0, 1.05)` — one-sided, so nominal armature is the MINIMUM;
  * `dof_frictionloss *= U(0.5, 2.0)` — linear-mean 1.25, so nominal sits on the LOW side.
So the nominal model is the LEAST-damped extreme. A policy trained under that DR is
optimized for the *average* of the cloud and is never trained on the nominal point
(probability zero); for an energetic, thin-margin gait (a strong parallel spring) the
nominal point becomes an untrained "hole" where it over-hops and falls — even though it
scores well under DR. That breaks the project's standard nominal-deterministic energy eval.

This randomizer keeps every field of the stock one but CENTERS the damping axes on x1.0
(armature x U(0.95,1.05), frictionloss x U(0.6,1.4)), so the nominal model is an INTERIOR
point the policy is effectively trained for. Friction (nominal 0.6 in U(0.4,1.0)) and mass
(x U(0.9,1.1)) are already centred/interior and unchanged. Opt in with `centered_dr: true`
in a run config (train.py selects this instead of registry.get_domain_randomizer).
"""
import jax
from mujoco import mjx

TORSO_BODY_ID = 16  # matches mujoco_playground/_src/locomotion/g1/randomize.py


def centered_domain_randomize(model: mjx.Model, rng: jax.Array):
    n = model.nv - 6  # actuated DoFs (past the 6 free-base DoFs)

    @jax.vmap
    def rand(rng):
        # Floor/foot friction: U(0.4, 1.0) — nominal 0.6 is interior, unchanged.
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
        pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)
        # Joint friction-loss: CENTERED x U(0.6, 1.4) (was x U(0.5, 2.0), skewed high).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(n,), minval=0.6, maxval=1.4)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)
        # Armature: CENTERED x U(0.95, 1.05) (was one-sided x U(1.0, 1.05)) — the key fix.
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(n,), minval=0.95, maxval=1.05)
        dof_armature = model.dof_armature.at[6:].set(armature)
        # Link masses x U(0.9, 1.1) (centred, unchanged).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
        # Torso mass +U(-1, 1) (centred, unchanged).
        rng, key = jax.random.split(rng)
        dm = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dm)
        # Keyframe pose +U(-0.05, 0.05) (centred, unchanged).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0.at[7:].set(
            model.qpos0[7:] + jax.random.uniform(key, shape=(n,), minval=-0.05, maxval=0.05))
        return pair_friction, dof_frictionloss, dof_armature, body_mass, qpos0

    pair_friction, frictionloss, armature, body_mass, qpos0 = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0, "dof_frictionloss": 0, "dof_armature": 0,
        "body_mass": 0, "qpos0": 0,
    })
    model = model.tree_replace({
        "pair_friction": pair_friction, "dof_frictionloss": frictionloss,
        "dof_armature": armature, "body_mass": body_mass, "qpos0": qpos0,
    })
    return model, in_axes
