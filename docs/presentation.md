---
marp: true
theme: default
paginate: true
---

# Adaptive Parallel Elastics for Legged Locomotion
### Does a tunable passive spring cut the *electrical* energy of walking?

**Headline:** gearing decides the sign. A parallel knee spring **fails** on a high-gear humanoid (Unitree G1)
but **cuts cost of transport −14 to −27%** on a low-gear quadruped (Unitree Go1), and an **adaptive,
self-tuning per-leg preload** carries that win into load-carrying.

*MuJoCo Playground + MJX + brax PPO. Metric: electrical energy (ohmic + mechanical, no regen), not mechanical work.*

---

## The mechanism — why a passive spring *could* help

- Motor **ohmic (resistive) loss scales with torque squared:** `P_loss ≈ (τ/Kt)²·R`.
- A **parallel** elastic element sits beside the motor and carries part of the joint torque.
- Offloading torque `Δτ` cuts the ohmic term **quadratically**, and recovers braking energy the motor would
  otherwise dissipate (no regeneration on these drivers).
- **Catch:** the spring is *always on* — it also fights the motor in phases where its torque is unwanted, and
  the benefit only matters if **ohmic loss is a large share of the budget** → that share is set by the **gear ratio**.

---

## Headline finding: **gearing is the crux**

| Platform | Gear ratio | Ohmic share of budget | Parallel-spring verdict (walking) |
|--|--:|--:|--|
| Unitree **G1** humanoid | 22.5:1 (high) | ~4 % | **Negative** — in-loop hip spring **+7% worse** |
| Unitree **Go1** quadruped | 6.33:1 (low) | ~54 % | **Positive** — see next slide |

- High gear → motor torque is already small (the gearbox multiplies it) → ohmic is negligible → the spring is
  dead weight that fights the gait. **Nine catalogued negative results** on the G1.
- Low gear → motor carries large torque → ohmic dominates → offloading pays.

---

## The positive result — Go1 constant knee preload

**Cost-of-transport reduction (electrical W per m/s), adaptive vs matched no-spring baseline, flat ground:**

| Condition | ΔCoT @0 kg | @2.5 kg | @5 kg | Notes |
|--|--:|--:|--:|--|
| Seed 1 | −16.6% | −19.5% | −22.8% | clean |
| Seed 2 | −3.4% | −8.3% | −6.5% | **weak outlier** |
| Seed 3 | −13.9% | −20.1% | −26.7% | clean |
| Curriculum | −16.8% | −20.4% | −22.0% | clean |

**−14 to −27% in 3 of 4 conditions, growing with load.** Seed 2 is a weak outlier (offload spent on speed, not energy).

![w:680](../outputs/figures/cot_vs_load.png)

---

## The element insight — a *constant preload*, not a linear spring

- The knee (calf) **work-loop is offset-dominated**: it holds a near-constant support torque through stance.
- A **linear** spring fit to that loop **degenerates to k ≈ 0** (the Belov/Osokin τ²-fit is even *anti*-restoring,
  k < 0 — not a passive spring).
- ⟹ the buildable optimum is a **CONSTANT preload** (a heavily pre-wound low-rate coil ≈ constant torque over
  the knee's small range). "Just add a spring" naively fails.

---

## The novel contribution — adaptive, self-tuning, per-leg preload

- A slow **outer loop** around the 50 Hz RL policy: senses each knee's own motor torque (15 s EMA), ramps that
  leg's passive preload to offload it (clipped-proportional, ≤2 N·m/s).
- **No load sensor, no payload observation** — the controller infers load from the motor's own torque.
- **Per-leg** (4 independent reflex loops) → handles front/rear + left/right asymmetry automatically.
- **Train robust, adapt at deploy:** train under preload domain-randomization; run the controller in long
  rollouts so the preload converges. Converged τ₀ scales with load, front > rear — exactly as expected.

---

## Load-carrying — the win grows with load (and a stability caveat)

- The CoT reduction **increases with payload** (−17% → −27% over 0 → 5 kg) — heavier box, more support torque to
  offload. The 15 kg curriculum walks 0–15 kg.
- **Honest caveat:** at high load (≥7.5 kg) the adaptive trades some **stability** — survival drops
  (e.g. 1070/1500 @10 kg) where the matched baseline holds 1500/1500. The energy win is **low-to-mid-load**.
- **Realism bound:** the real Go1 carries **~5–10 kg max** (12 kg robot). Sim "walking" at 30 kg is unphysical —
  the plain sim enforces peak but not continuous/thermal torque or balance limits. The defensible range is 0–6 kg.

---

## Evidence — videos & figure

- `outputs/videos/walk_noload_adaptive.mp4` — flat, no load, 5 clips
- `outputs/videos/walk_5kg_adaptive.mp4` — flat, 5 kg load, 5 clips
- `outputs/videos/walk_rough_noload_adaptive.mp4` — 2.5 cm rough terrain
- `outputs/videos/walk_rough_5kg_adaptive.mp4` — 2.5 cm rough, 5 kg
- `outputs/figures/cot_vs_load.png` — CoT vs load, baseline vs adaptive, all seeds

*(Videos show the per-leg adaptive preload running; the camera tracks the dog.)*

---

## Honest failures & dead-ends

1. **G1 walking + hip spring → negative** (+7% in-loop; reversed the optimistic post-hoc −3.8%). Gearing.
2. **G1 running → two failed attempts.** [0,3] m/s from scratch collapsed to a 0.85 m/s never-fall walk; a
   curriculum + reward-redesign **destabilized** it. The Playground G1 env structurally resists a flight phase.
3. **Rough terrain → unstable.** Energy win survives on 2.5 cm bumps (−10 to −19% CoT) but **~40% survival** for
   *both* arms — rough locomotion is hard for the blind loaded dog regardless of the spring. Full 5 cm inconclusive.
4. **Seed 2** — a weak (−3 to −8%) energy outlier with a stability cost: the win is real but seed-variable.
5. **Capacity-to-failure** — never reached failure (sim too permissive); the high-load regime is unphysical for a Go1.

---

## What's next — knee spring on a *running* dog

- The dog runs with a **real flight phase** (braking-energy recovery at impact) and is the **low-gear** platform
  where springs pay — the natural home for the running-energy claim the G1 couldn't host.
- Design ready (`docs/dog_running_design.md`): warm-start the walker → raise the command amplitude → gate on a
  **measured flight fraction** → test a **one-sided stiffness** spring (which a constant preload can't replace
  for braking recovery) vs no-spring, matched.

---

## Status & paper-readiness

- **Solid:** the gearing-crux thesis, the Go1 −14 to −27% CoT (3 seeds), the constant-preload insight, the
  adaptive per-leg controller, the load-carrying trend, the negative-results catalog.
- **Gaps for publication:** no measured motor Kt/R (absolute watts are a band, not a point); energy-naive
  baseline; no sim-to-real; stability/realism caveats to foreground.
- **Readiness ≈ 40%** for a focused Part-1 paper (negative G1 + positive low-gear Go1 + adaptive load preload).
- **Target:** IEEE Access — *Adaptive Parallel Elastics for Energy-Efficient Quadruped Load Carriage.*
