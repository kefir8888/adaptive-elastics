# Project plan — milestones & submilestones

> **⚠ SUPERSEDED (2026-06-16) — historical roadmap, kept for the record.** This plan is
> anchored to the **2026-06-12 G1 hip-pitch roadmap** (Milestones 4–7 below). That line of
> work is now **closed by a negative result**: the in-loop G1 hip-pitch spring made walking
> **+7.4 % worse** (Milestone 4, see `RESULTS.md` / `negative_results.md`). The project has
> since **pivoted to the Go1 quadruped** — the load-carrying walking program is **done and
> positive**, and the **active direction is the Go1 "dog-running" knee-spring experiment**.
> For the current go-forward plan use **`docs/NEXT_SESSION.md`** (authoritative handoff) and
> **`docs/directions.md`** (direction map). The milestone content below is **historical**;
> do not treat its "NEXT" markers as live.

Living roadmap. Reflects the decisions through 2026-06-12: spring target pivoted
to **hip-pitch**, **external stiffness control** (params chosen outside the RL
policy, fed in as observations), mechanism = the **dual-quadratic tunable linear
spring** (adopted prior art — Hurst 2004 / Migliore 2005, see `mechanism.md`),
**Direction 1 first** (G1 hip spring + robustness sweep), **Direction 2 later**
(DecARt decoupled leg). Design-correctness guards in the memory
`experimental-design-watchdog`.

> **2026-06-14 — strategic directions chosen.** After measuring that the G1-walk
> ohmic lever is small (~4 % of the motor budget, ~3 % whole-body saving,
> ~0 % under regen), a six-direction assessment (`docs/directions.md`) set the
> go-forward order: **(1) in-loop G1 gate → (2) running G1/H1 [needs a running
> policy; appeal = braking energy + bouncing gaits, NOT the clutch] → (5) quadrupeds across
> slopes/loads via a single ZERO-SHOT spring-conditioned policy (no per-condition
> retrain, unlike Bjelonic 2023) → (6) DecART/parallel-kinematics leg-length
> spring (experimental) → (3) low-gear research-platform comparison (lower
> priority)**. Manipulation/static holding (4) is **dropped**. Milestones 5–6
> below still apply as the G1 sub-roadmap; the directions doc is the wider map.

> **2026-06-14 (eve) — running-efficiency program** is now the active execution
> plan: `docs/running_program.md` (G1 running for efficiency + a Go2 quadruped
> track). Infra landed: `env_overrides` (speed-range override), `configs/run_baseline.yaml`
> (the sawtooth is a default-zero `action_rate` penalty — enabling it fixes it),
> multi-joint `pea-sweep` + per-joint braking energy + `metrics.fit_linear_spring`
> (per-joint post-hoc optimum, validated). Reward weights pending the running-RL
> SOTA workflow.

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

## Milestone 4 — In-loop GO/NO-GO gate ✅ DONE — NEGATIVE (historical)
*Ran on the G1 hip-pitch spring; result was **+7.4 % worse** (`RESULTS.md`). The sub-items
below are the historical to-do list as it stood before the run.*
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
