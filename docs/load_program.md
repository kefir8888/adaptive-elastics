# Quadruped load-carrying program — adaptive parallel knee preload

The quadruped extension of the study, after the Go1 walking result (a constant parallel
knee preload cuts **−17 to −20 %** in-loop on the low-gear Go1; see `RESULTS.md` /
`negative_results.md`). Direction: a Go1 (later other low-gear dogs) carrying **variable
payloads**, with a **self-tuning constant knee preload** that adapts to the measured load.

## Why this is more than the "trivial" result
The bare mechanism — offload the most-loaded leg joint, the motor works less — is simple
physics. The contributions that make it non-trivial:
1. **When it HOLDS vs FAILS (the boundary).** The same idea *reverses* on a high-geared
   humanoid (G1: post-hoc −3.84 % → in-loop **+7.4 % WORSE**); gear ratio + element kind
   decide it. Mapping that boundary is the rigorous result (`negative_results.md`).
2. **Element kind is non-obvious.** A linear spring is NULL on the offset-dominated knee
   (k=0); only a CONSTANT preload works (the Belov/Osokin τ²-fit is anti-restoring,
   k=−12.8 → not a passive spring). "Add a spring" naively fails.
3. **In-loop co-adaptation.** Post-hoc can have the wrong sign (G1) — you must retrain.
4. **The adaptive self-tuning preload — the novel control contribution.** A closed loop
   that senses the knee load and ramps the passive preload to offload it, with **no load
   sensor and no payload observation by the controller** (below).
5. **Capability > energy.** At heavy load the baseline tops out (knee torque hits the
   45 N·m motor ceiling); the preload offloads exactly that support torque, keeping the
   motor under its limit → the spring **extends carry-capacity**, a stronger claim than a %.

The simplicity is a *feature* (deployable, robust); the novelty is the self-tuning loop +
the boundary map + the capability framing.

## Design (locked with the user)
- **Blind controller** — the RL policy does NOT observe the payload (a dog can't read the
  box mass). One policy, load-robust.
- **Payload domain randomization from the start** — a box of **+U(0, 10) kg** on the trunk
  each episode (≤ body mass ~12 kg; realistic Go1 box max), fixed within an episode, resampled
  per reset. Implemented by widening the Go1 randomizer's torso-mass term: `src/pea/payload.py`,
  `cfg.payload_max_kg` (train.py uses it when > 0).
  - **RANGE LESSON (2026-06-15):** an initial **0–25 kg** range (~2× body mass) COLLAPSED the
    blind policy to **standing** — `tracking_lin_vel` reward fell **925 → 367**; the policy walks
    at ~0 m/s at every commanded speed. The >12 kg tail is physically unwalkable, so the policy
    learned standing as the dominant behavior and it carried over to light loads. Energy penalty
    ruled out (identical −1e-4 in the walking original). **Keep the range walkable (≤ body mass)
    and gate the retrain on `tracking_lin_vel >~800`**; if it still collapses, drop to 0–6 kg or add
    a payload curriculum (ramp the max over training so it masters walking before load).
- **Terrain:** FLAT first (clean energy-vs-load curve); rough terrain (`Go1JoystickRoughTerrain`)
  next as robustness, with a *mild terrain curriculum only if it struggles*. Skip stairs + gravel.
- **The "capacity ceiling" is *stops walking*, not *falls over*** — at heavy load the Go1 stands
  stably (does not fall). So the headline is the **energy-vs-load curve** (the preload cuts ohmic
  power at each load); a walk-forward-capacity ceiling is a secondary OOD probe (eval past 10 kg).
- **Matched:** baseline (no spring) and spring (adaptive preload) train identically under the
  same payload DR; the ONLY difference is the preload.

## The adaptation mechanism (self-tuning preload)
A slow OUTER loop around the fast (50 Hz) RL policy, time-scale separated so the blind
policy treats the preload as a quasi-static disturbance ("not shocked"):
- **Sense:** the average motor knee (calf) torque over the last **~15 s** (~30 strides) — the
  quasi-static SUPPORT (offset) component. Read from the motor's OWN torque; **no external
  load sensor, no payload observation.**
- **Act — a rate-saturated (clipped-proportional) integral controller.** Let
  `ē = (15 s-avg motor knee torque) − ē_target`. Then **`τ̇₀ = clip(k_p·ē, −2, +2) N·m/s`,
  k_p ≈ 0.2 s⁻¹**: rate saturates at 2 N·m/s for `ē ≥ 10` N·m, scales down linearly below,
  and → 0 as `ē → 0` (the proportional zone gives a smooth stop → **no chatter**, unlike a
  bang-bang ramp). It drives the motor's *mean* knee torque to `ē_target` and holds.
- **ē_target ≈ 0 (full compensation) is the sensible default.** Ohmic alone is minimized at
  full comp (τ₀ = mean knee torque → motor mean torque 0). The energy *optimum* is a hair below
  full comp — the no-regen mechanical asymmetry: near the mean the constant spring opposes the
  knee's FLEXION phases, forcing un-recovered positive motor work there — but the curve is FLAT
  (Go1: τ₀=3.5 → −14.89% vs τ₀≈mean 4.6 → −14.76%, a 0.1-pt wash), and with regeneration full
  comp is exactly optimal. Full comp also MAXIMIZES motor headroom → best for the capability
  claim. So drive ē → 0; a small positive ē_target only buys a negligible energy edge + margin.
- **Implementation: EMA, not a 750-sample buffer** (one scalar/knee, jit-able, τ=15 s).
- **PER-LEG preloads (4 independent local loops), not one pooled.** τ₀ is a 4-vector; each
  knee senses its OWN torque and ramps its OWN τ₀ — fully decentralized (no coordination,
  like a reflex). Handles front/rear (~20%) + left/right asymmetry from CoM shift, turning,
  and uneven terrain automatically. The adaptive law + preload-DR just become vectorized.
- **Result:** heavier box → higher measured knee load → τ₀ ramps up → offloads more,
  automatically; adaptation ~5–10 s, negligible vs an experiment measured over minutes.

### Train robust, adapt at deploy (NOT the controller inside training)
Training episodes (~12 s) are shorter than the adaptation + 15 s window, so the preload would
never reach steady state in-episode. Clean separation (the time-scale argument):
- **Train:** payload DR **+ preload DR** — randomize τ₀ per episode alongside the box mass.
  The energy-penalized, BLIND policy learns to *exploit* any preload at any load.
- **Deploy/eval:** run the clipped-proportional controller in long rollouts (60 s+) so τ₀
  converges; the preload-robust policy rides the slow change. This yields the capacity +
  energy-vs-load curves under the real adaptive loop, while keeping training cheap.

## Expected knee torques (Go1 ≈ 12 kg robot; measured mean calf ≈ 4.6 N·m no-load)
Support torque scales ~with total weight: mean calf τ ≈ 4.6 × (12 + P)/12.

| payload P | mean calf τ | adaptive preload τ₀ (~0.75×) |
|---|---|---|
| 0 | 4.6 N·m | ~3.5 N·m |
| 5 kg | ~6.5 | ~5 |
| 10 kg | ~8.4 | ~6.5 |
| 15 kg | ~10.4 | ~8 |
| 25 kg | ~14.2 | ~11 |

All preloads are far under the **45 N·m calf peak limit**, and passive (they OFFLOAD the
motor, not consume it). The DYNAMIC peaks (no-load ~20–35 N·m) scale with load and approach
45 N·m at heavy payload — that's where the baseline FAILS and the preload (taking ~8–11 N·m
off the mean) keeps the motor under its ceiling → capacity extension. Ramp 1–2 N·m/s covers
the full 3.5→11 span in ~5–10 s.

## Physical realization (later / hardware)
The "constant" element need not be a true constant-torque device. A **low-stiffness torsion
spring, heavily PRE-WOUND** — like a watch mainspring (many turns of low-rate coil) — gives
near-constant torque over the knee's small operating range (Δθ ≈ 0.83 rad). Torque ripple
over the range = `k·Δθ`; mean `τ₀ = k·(pre-wind)`, so low k + large pre-wind → large τ₀ with
small ripple:
- `k ≈ 0.42 N·m/rad` pre-wound 8.3 rad (~1.3 turns) → **τ₀ ≈ 3.5 N·m, ≤10% ripple** (and only
  ~3% at τ₀=11, since ripple/τ₀ shrinks as τ₀ grows). `k ≈ 1.0` → ~0.55 turns but ~24% ripple.
- **Tunable = adjust the pre-wind.** A small servo slowly rotates the spring's anchor to set
  τ₀ — this *is* the adaptive controller's actuator (≤2 N·m/s ≈ **0.76 turns/s** at k≈0.42).
  Fully compatible with the project's adjustable-spring mechanism (operate it in the low-k /
  large-offset corner; the servo-positioned onset sets the pre-wind).
- **Ripple is benign** — ~3–10% linear variation is negligible on the offset-dominated knee
  (the τ²-optimal linear part is near-zero / anti-restoring anyway).
- **Constraint:** stored energy `E = τ₀²/2k` grows fast — ~15 J at τ₀=3.5 but **~140 J at
  τ₀=11** (k=0.42); heavy preloads bank real energy (packaging/safety). Higher k trades
  pre-wind/energy for ripple.
- **Cheap SIM faithfulness check:** model `kind=linear`, `k≈0.5`, large `θ₀` (≈ constant over
  the range) and confirm it reproduces the −16.7% → validates the *buildable* element, not
  just the idealized constant.

## Robots (Go1 now; rest later)
Go1 ✅ (low gear 6.33:1, datasheet Kt, ready env). Later: Go2 / Barkour (low-gear
confirmations), **Spot** (high-gear COUNTER-test — should NOT pay, proving gear is the crux;
electricals blocked → qualitative only), **big Unitree B1/B2** (~50–60 kg — best for the
heavy-load story; needs env wrap), ANYmal (SEA / already series — defer). Est. **~1.5 GPU-hr
+ 0.5–2 hr setup per robot**; absolute energy needs per-robot Kt/R (only Go1 has a datasheet
Kt) so lead with the **relative %** and Bjelonic's gear-invariant ∫τ²dt.

## Status
- **Payload baseline** (flat, payload 0–25 kg, blind controller, 300 M) — training now (~20 min).
- **Next:** build the **adaptive-preload spring run** (the mechanism above) → compare to the
  baseline as **payload-capacity + energy-vs-load curves**. The headline is the capacity
  curve: where the baseline tops out vs how far the adaptive preload extends it.
