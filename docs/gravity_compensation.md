# Gravity-compensation program — parallel elastics on constant-sign gravity loads

A Part-1 spinoff (opened & closed 2026-06-19). It tests parallel springs on joints whose
gravity load **never reverses sign** (a body goes up/down, or a leg holds a stance), on two
mobile-manipulator robots loaded from **external ROS/URDF descriptions**:
- **Galaxea R1** — wheeled humanoid; coordinated torso lift (high-gear harmonic torso, motor
  `galaxea_torso`, joint-side `R/Kt²≈2.0e-4`).
- **LimX W1 (WL_P311D)** — wheeled quadruped; knee preload during wheeled roll (low-gear QDD,
  `limx_knee`, `R/Kt²≈0.027`, ~135× higher).

**Status: DONE and positive.** A permanent clutchless parallel spring is unambiguously good
here, because the load never reverses — there is no phase where the spring fights the motion
(unlike the always-on walking spring, the project's central negative result). The win is
**large on BOTH platforms despite opposite gearing**.

Artifacts: reporting bundle `outputs/gravity_compensation/` (videos, plots, combined table,
REPORT.md — gitignored, syncs via Drive). Code: `src/pea/{urdf_loader,gravcomp,render_util}.py`;
scripts `scripts/{galaxea_lift,galaxea_lean,galaxea_reach,galaxea_free,galaxea_knee_plot,limx_roll,limx_phases,gravcomp_table}.py`;
configs `configs/{galaxea_lift,galaxea_lift_kneeonly,galaxea_lean,limx_roll}.yaml`.

## Why this is more than trivial
The bare mechanism (offload a gravity load, motor works less) is simple. What makes it a result:
1. **Gear-INDEPENDENCE — the inversion of the walking finding.** In the walking study, gear
   ratio set the *sign* of the win (low-gear Go1 positive, high-gear G1 negative, because ohmic
   is a tiny share of a high-geared budget). Here the win is large on the high-gear Galaxea AND
   the low-gear W1, because the spring offloads the **constant-sign load** (lift work; stance-holding
   ohmic) — and that part of the budget is gear-independent. Same idea, opposite conclusion, for a
   well-understood reason.
2. **Element kind must match the load SHAPE** (see Findings #3) — a linear spring is the wrong
   device on a constant-load joint, exactly as on the Go1 knee.
3. **Self-tuning adaptive preload** (W1 4-phase) — a closed loop that ramps the passive preload
   to track a *changing* stance torque, with no load sensor. This is the "adaptive" in adaptive
   elastics, not just a static post-hoc fit.

## Results (electrical, no-regen; 150 W onboard computer; prescribed motion)

| robot / motion | spring | targeted-motor avg power | targeted saving | whole-robot |
|---|---|---|---|---|
| Galaxea R1 — coordinated upright lift | linear ×2 + constant preload ×1 (auto-selected) | 41.5 → 1.5 W | **−96%** | −20.9% |
| Galaxea R1 — lift, spring on middle joint only | one linear spring (torso_joint2) | 41.5 → 6.1 W | **−85%** | −19% |
| Galaxea R1 — knee alone (torso_joint2) | one linear spring | 36.3 → 0.9 W | **−98%** | — |
| Galaxea R1 — forward lean about bottom joint | one linear spring (torso_joint1) | — | **−98%** (that joint) | — |
| Galaxea R1 — 5 "free" reaches (fwd/down/up/twist/scan) | bottom-joint linear spring | — | **−97…−99%** | — |
| LimX W1 — wheeled roll | constant knee preload (×4) | 53.4 → 1.1 W | **−98%** | −26% |
| LimX W1 — 4-phase height roll | ON-THE-FLY adaptive knee preload | tracks 4→26 N·m | **−93%** (knee power) | — |

- Whole-robot savings are diluted mainly by the fixed onboard computer (150 W); at 0 W
  (servos only) they are −94% (Galaxea) and −72% (W1).
- Load scales with reach: of the 5 free reaches, reach-DOWN is heaviest, reach-UP lightest.

## Interesting findings
1. **One well-placed spring captures almost the whole win.** The knee (torso_joint2) alone is
   **87%** of the lift's torso-motor power (36.3 of 41.5 W; peak 119 N·m). A single spring there
   gives **−98%**, and a one-spring lift is **−85%** vs **−95%** for all three. Diminishing returns
   from extra springs — the design lever is *placement*, not count.
2. **Gear-independence confirmed** (see above): the saving survives a ~135× swing in joint-side
   `R/Kt²` between the two robots, because it targets the gear-independent constant-sign load.
3. **Element kind is set by the load SHAPE vs the joint's own angle** — and a linear spring is
   the WRONG device for a constant load (re-confirms the Go1-knee rule in `load_program.md` /
   `negative_results.md`):
   - Where gravity *varies with the joint's own angle* — torso_joint1/2 (gravity span 4.9 / 29 N·m,
     corr −0.93 / −0.91) — a **linear** torsion spring `τ=−k(θ−θ₀)` fits cleanly (−98%).
   - Where gravity is *constant in the joint's angle* — torso_joint3 (gravity span **0.0 N·m**,
     corr +0.63, all dynamic) — a linear spring is **mis-specified**: the energy-optimal *linear* fit
     pushed θ₀ to the grid boundary (−2.15 = `θ.min−0.6`) trying to mimic a flat constant, leaving a
     spurious tilt (k=10) and under-fitting at −65%. The correct element is a **constant preload**
     (`springs.py` kind `"constant"`, `tau0`). `fit_spring_per_joint` now selects it automatically
     (τ₀=−11 N·m) → **−84%** on joint3, lifting the 3-spring headline from −95% to **−96.5%**.
   - **General rule:** choose spring kind per joint by load shape — linear where gravity slopes with
     the joint's own angle, constant preload where it doesn't. `fit_spring_per_joint` does this: it
     tries both kinds per joint and keeps the lower-energy one.
4. **Adaptive preload works on-the-fly.** The W1 4-phase run drives an EMA + rate-limited integral
   law (`α=exp(−dt/window)`; `ema←α·ema+(1−α)·τ_motor`; `τ₀←clip(τ₀+clip(kp·(ema−target),±rate)·dt,0,τ₀_max)`)
   that ramps the passive preload to track the stance torque as the body squats (4→26 N·m), with **no
   load sensor and no payload observation** — the controller senses only its own joint torque.

## Honesty caveats
- **Prescribed-motion (post-hoc) energy.** For a fixed task motion there is no policy to re-adapt,
  so post-hoc ≈ in-loop here — unlike the walking study, where the gait changes and only the in-loop
  number counts. (The W1 adaptive run is genuinely closed-loop on the preload, not the base motion.)
- **Motor Kt/R are sourced estimates** (neither vendor publishes them); the win is gear-invariant so
  the **% reductions are robust**, but treat absolute watts as a band.
- **No regeneration** (project convention). On the lift the win is mostly the offloaded lift work +
  ohmic, not braking recovery, so it is less regen-sensitive than the running case.
- **Masses:** Galaxea upper body ~35.9 kg is an estimate; W1 ~43.5 kg is the URDF's own value
  (base 18.19 + legs 25.3). Wheel transport is billed analytically (Crr·m·g·d, Crr=0.015), not from
  the slipping velocity-servo wheels.
- **Render-only model fixes** (no effect on physics): mesh-name collision across packages
  (`base_link.obj`) and `.obj` materials — see `CLAUDE.md` gotchas.

## Open
- Motor constants + Galaxea mass are estimates; an in-the-loop (retrained-motion) check is not
  applicable to a prescribed task but a hardware Kt/R would tighten the watts.
- The dog-running and G1-running locomotion tracks remain SUSPENDED; Part 2 (explosive moves) untouched.

_Done (2026-06-19): per-joint spring-kind selection (`fit_spring_per_joint`, linear vs constant preload)._
