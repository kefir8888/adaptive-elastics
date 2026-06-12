# Project plan — milestones & submilestones

Living roadmap. Reflects the decisions through 2026-06-12: spring target pivoted
to **hip-pitch**, **external stiffness control** (params chosen outside the RL
policy, fed in as observations), mechanism = the **dual-quadratic tunable linear
spring** (adopted prior art — Hurst 2004 / Migliore 2005, see `mechanism.md`),
**Direction 1 first** (G1 hip spring + robustness sweep), **Direction 2 later**
(DecARt decoupled leg). Design-correctness guards in the memory
`experimental-design-watchdog`.

## Phase 0 — Infrastructure ✅ DONE
- Repo + GitHub + Google Drive sync; immers.cloud H100 pipeline driven over SSH;
  CPU rollout/analysis on the Mac. Cross-hardware reproducibility confirmed
  (T4 vs H100 reward-vs-steps curves overlap).

## Milestone 1 — Flat baseline (no spring) ✅ DONE
- 200 M-step G1 flat walker (reward 12.46); replays locally, walks 0.9 m/s.

## Milestone 2 — Knee/joint logging & work-loop analysis ✅ DONE
- θ/τ logged; `E[τ|θ]` decomposition; knee is offset-dominated (DC); hip-pitch is
  the better target (AC, buildable linear spring ~51–60% of mean-square torque).

## Milestone 3 — Post-hoc spring analysis (optimistic bound) ✅ DONE (knee)
- `analyze --spring`: knee copper −36–42%, total knee electrical −16% on fixed gait.
- ☐ 3a: run post-hoc on **hip-pitch** with the linear-spring optimum (cheap CPU).

## Milestone 4 — In-loop GO/NO-GO gate (NEXT)
Cheapest rigorous test that the effect is real, before any sweep.
- ☐ 4a: extend `analyze.py` to **whole-leg CoT**, both regen and no-regen.
- ☐ 4b: settle **G1 Kt/R + regen** (justify from driver specs; no-regen was
  refuted by the survey). If unresolved, lead with copper-% (Kt/R-independent).
- ☐ 4c: `configs/spring_hip_linear.yaml` at the post-hoc optimum `(K_eff, θ₀)`.
- ☐ 4d: train **one fixed-spring specialist** (hip-pitch) + **matched baseline**,
  energy in the reward, **identical reward both arms**; ≥1–2 seeds for the gate.
- ☐ 4e: compare CoT/copper. **Decision:** substantial (> seed variance) → continue;
  negligible → stop and write the negative result.

## Milestone 5 — Conditioned universal policy + spring optimization (if gate passes)
External stiffness control: one policy conditioned on `(K_eff, θ₀)`, DR over the
operating envelope; outer optimizer picks the spring.
- ☐ 5a: conditioned `SpringWrapper` + obs augmentation (randomize `(K_eff,θ₀)`/episode).
- ☐ 5b: **identifiability check** — verify gait kinematics vary with `K_eff`.
- ☐ 5c: outer **BO/grid over `(K_eff,θ₀)`** per condition (policy as oracle);
  landscape plots. (Iterate/alternate — a single closed-form pass isn't jointly optimal.)
- ☐ 5d: **specialist retrains at chosen optima** for the unbiased best-vs-best
  headline (≥3 seeds, mean±std) — sidesteps the generalization tax.

## Milestone 6 — Robustness sweep & adaptive schedule
- ☐ 6a: matrix over **speed × slope × load** (later: direction, terrain roughness).
- ☐ 6b: per-condition optimal `(K_eff, θ₀)`; the outer **adaptive-stiffness selector**.
- ☐ 6c: CoT-reduction **map** across the envelope; where the spring helps vs hurts.

## Milestone 7 — Write-up (target RA-L / ICRA / IROS / Humanoids)
- Position vs Bjelonic 2023, Belov/Osokin 2024, STEPPR; cite Hurst/Migliore for the
  mechanism; honest **integration-novelty** framing; fresh prior-art search at submission.

## Direction 2 (future) — DecARt-style decoupled leg
- Custom MJX leg model; parallel spring on the **leg-length axis** (monotonic load,
  no clutch problem). Higher novelty ceiling, bigger lift (morphology change).

## Beyond — hardware validation
- Physical G1 hip/knee spring retrofit with measured battery energy → the path to a
  top-tier journal. Out of current scope.

## Warm-starting policy (decision)
Warm-start harder policies (rough/incline/load) from the flat baseline checkpoint —
it's the standard Playground recipe (flat→rough finetune) and a real speed win. It
does **not** break reproducibility in the meaningful sense (same code+config+seed+
restore → reproducible; just archive the restore checkpoint, which we have). The
real risk is **comparison validity**: apply the *identical* warm-start protocol to
both spring and baseline arms, and beware that warm-starting a spring policy from a
no-spring checkpoint can anchor it to the baseline gait and **under-report
co-adaptation** — so give it enough extra steps to escape, and validate the
**headline** numbers with from-scratch (independent-seed) runs.
