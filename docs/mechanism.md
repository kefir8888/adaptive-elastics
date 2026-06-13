# Spring mechanism: dual offset half-parabola → tunable linear element

How we realize a **parallel knee/hip spring with on-the-fly tunable stiffness
and equilibrium**. The synthesis below is correct and is what we will *use* —
but it is **established prior art, not a contribution** (see §3). Treat it as
adopted, cite it, and put novelty weight on the application (parallel packaging
+ RL gait co-adaptation + electrical CoT), per `related_work.md`.

## 1 · The synthesis

Two **one-sided quadratic** (half-parabolic) elements in opposition, each with a
servo-set onset position `p`:

- A (zero left of `p₁`, parabola right): `τ_A = −k(θ−p₁)²` for `θ>p₁`, else 0.
- B (mirror): `τ_B = +k(p₂−θ)²` for `θ<p₂`, else 0.

With `p₁ < p₂` they overlap on `(p₁, p₂)`, where the `θ²` terms cancel:

```
τ(θ) = −k(θ−p₁)² + k(p₂−θ)² = −2k(p₂−p₁)·[θ − (p₁+p₂)/2]  ≡  −K_eff·(θ−θ₀)
```

- **Stiffness** `K_eff = 2k(p₂−p₁)` (∝ onset separation)
- **Equilibrium** `θ₀ = (p₁+p₂)/2` (onset midpoint)

Exact, not approximate (the potential `U = (k/3)[(θ−p₁)³+(p₂−θ)³]` is exactly
quadratic on the overlap). Outside the overlap a lone quadratic remains, giving
hardening end-stops; the linear band half-width is `δ = K_eff/(4k)`.

**Tuning law** (two servos ↔ two parameters, bijective):
`p₁ = θ₀ − K_eff/(4k)`, `p₂ = θ₀ + K_eff/(4k)`. The servos set the curve between
gaits and hold; the element is passive within a stride (≈no tuning power).

## 2 · Why it fits the project

Direction 1 needs a different `(K_eff, θ₀)` per (speed, incline, load); this
gives both knobs passively, no within-stride active stiffness loop. On a
DecARt-style decoupled leg (`related_work.md`) it would act on the leg-length
axis where load is monotonic.

## 3 · Prior art — the mechanism is NOT novel (verified)

Adversarial check (workflow `wf_7d31115d`, Fable verifiers reading primaries)
found **exact scoops**:

- **Hurst, Chestnutt & Rizzi, "An Actuator with Mechanically Adjustable Series
  Compliance" (AMASC), CMU-RI-TR-04-24, 2004**, Eq. 1, pp. 6–9 — the exact
  identity: two opposed one-sided quadratic springs `F=Kz²` give
  `F_eff = 4K·x₃·(x₂−x₁)`, linear with motor-adjustable stiffness via pretension
  `x₃`. **`x₃ ≡ (p₂−p₁)/2`** maps our onset-separation to their pretension
  one-to-one; same binomial cancellation, midpoint equilibrium, even the
  one-sidedness (pull-only fiberglass). Generalizes (Taylor argument) that
  quadratic is the unique characteristic giving linear output. MABEL/ATRIAS
  lineage. Differs only in *role* (series compliance for running) and *tuning
  hardware* (spiral-pulley pretension vs onset blocks).
- **Migliore, Brown & DeWeerth, "Biologically inspired joint stiffness control,"
  ICRA 2005** — canonical antagonistic-quadratic VSA: stiffness from
  co-contraction, equilibrium from common-mode servo motion, tuned on the fly.
- **"Conceptual… antagonistic variable stiffness joint based on equivalent
  quadratic torsion spring," 2024 (PMC10451064)** — restates the identity in
  torsional form `M(β)=a(φ+β)²−a(φ−β)²=4aφβ`, `K=4aφ`, midpoint equilibrium.
  Shows it's current textbook VSA.
- **Vanderborght et al., "Variable impedance actuators: A review," RAS 2013** —
  establishes the antagonistic-quadratic class (incl. MACCEPA) as named, populated.
- **Belov/Osokin (Skoltech) 2024** — our own group already packages a *tunable
  parallel* spring with the τ² metric (analytic, leg-stand), so even the
  parallel-application framing is partly pre-occupied in-house.

**Verdict: novel-application-only.** Do not claim the synthesis as new. One-line
framing for the paper:

> *We realize the tunable linear parallel element with the classic
> antagonistic-quadratic synthesis (Hurst et al. 2004, Eq. 1; Migliore et al.
> 2005): two one-sided quadratic springs with servo-positioned onsets give
> `K_eff = 2k(p₂−p₁)` and equilibrium `(p₁+p₂)/2`.*

## 4 · The distinguishing capability: full passive disengagement (a tunable clutch)

This is where the construction goes **beyond** Hurst/Migliore, and it is a
mechanism-level claim worth making. Because each element is *semi*-parabolic —
genuinely zero on one side of its onset (true slack, not preload) — separating
the onsets so that `p₂ < p₁` opens a **dead zone `[p₂, p₁]` of exactly zero
torque**. If that dead zone covers the joint's angular excursion, the spring is
**completely disengaged and the joint behaves as a free actuator with no spring
at all.** Verified in code on the hip range [−0.79, +0.02] rad:

| onsets | regime | max |τ| over range |
|---|---|---|
| `p₁=−0.69, p₂=+0.11` (overlap, `p₁<p₂`) | engaged linear spring, `K_eff=68` | 34.4 N·m |
| `p₁=+0.05, p₂=−0.80` (separated, `p₂<p₁`) | **free joint** | **0.0 N·m** |

**Why pretensioned antagonistic VSAs (Hurst 2004, Migliore 2005, MACCEPA) cannot
do this:** their springs are *preloaded* and always in tension (a pretension
winch can pull but cannot create slack), so they have a nonzero minimum stiffness
and always apply a restoring torque. They can soften, not vanish. The one-sided
slack here gives a continuum: overlapping onsets → linear spring of stiffness
`2k(p₂−p₁)`; touching → zero stiffness; separated → free joint with a
tunable-width dead zone. This is a **passive, purely mechanical clutch realized
by the same two servos**, with no added clutch mechanism — and it is the
mechanism-level contribution the novelty survey found is *not* covered by prior
art.

**Precise scope (design-correctness):**
- **Full disengagement between operating conditions — yes, statically.** Set the
  dead zone to cover the excursion and the joint is free for that whole
  condition. Consequence for the adaptive system: "spring off" is always an
  achievable setting, so an adaptively-tuned spring is **weakly dominant over the
  no-spring baseline** — it can never make a condition worse than free, because
  free is one of its settings.
- **Phase-selective gating within one stride — no, not statically.** Engaging in
  stance and freeing in swing would need the dead zone to track gait *phase*, but
  our data (`JOURNAL.md` 2026-06-12) shows the stance and swing angle ranges
  **overlap** at the hip, so a static angle-keyed dead zone cannot separate them;
  that would still require moving the onsets within the stride (fast servos).

A secondary, implementation-level claim also remains (not a principle): measuring
any advantage of onset-block repositioning over pretension tuning (holding power,
tuning under load). Algebraically `x₃ ↔ (p₂−p₁)/2`, so that one is purely an
engineering comparison.
