# Project Journal — Parallel-Elastic Knee Efficiency Study

Append a short dated entry at the **end of each work session** (newest at the top).
Keep entries terse — this is a memory aid for future sessions, not documentation.

For each entry note: what you did, what you decided (and why), key numbers if a run
happened, what's open/broken, and the single next step.

---

## 2026-06-13 — Gate fully prepared; reward = total electrical; ready for GPU
- **Did:** Generalised `SpringWrapper` to any joint (`spring.joint`, now hip-pitch).
  Replaced cubic "nonlinear" spring with **semiparabolic** (two one-sided
  quadratic elements; overlap → exactly linear, verified to 1e-15; separated
  onsets → zero-torque dead zone = full passive disengagement, a clutch the
  preloaded VSAs of Hurst/Migliore cannot reach — see `mechanism.md`). Added
  `ElectricalRewardWrapper` (total-electrical penalty in BOTH conditions),
  `--energy-weight` CLI, `scripts/calib_sweep.sh`, whole-body CoT in `analyze.py`.
  All CPU-smoke-tested.
- **Decided:** (1) Spring target = **hip-pitch** (AC joint; buildable linear
  spring captures ~55% of mean-square torque vs the knee's collapse). Gate spring
  = linear, k=68 N·m/rad, θ0=-0.29 rad (offline hip fit). (2) Use the **linear**
  spring first (it is exactly the realizable mechanism in-band). (3) Reward AND
  headline metric = **TOTAL ELECTRICAL** (mechanical + ohmic, no regeneration) —
  battery life depends on total power, not ohmic alone. Report all three:
  ohmic loss, cost of transport, total power. Also report the Kt/R-independent
  ohmic-% for comparability with Osokin/Belov 2024 and Bjelonic 2023 (both used
  τ² only). (4) **No regeneration** (geared G1, walking). (5) Energy weight
  placeholder -2.5e-4 (≈7-12% of the +1.0 tracking reward; raw electrical ~291 W).
- **Open / broken:** Kt, R unknown (no hardware) → being ESTIMATED via deep web
  search (workflow), reported as a band; the relative ohmic-% is Kt/R-independent
  so this does not block the gate. Energy weight to be set by the calibration runs.
- **Next:** provision GPU box(es); run the calibration (5 short no-spring runs,
  `scripts/calib_sweep.sh`), pick the weight, then the gate: hip-linear spring vs
  matched no-spring at that weight, compare ohmic/CoT/total power. See README
  "Current state" and `docs/PLAN.md` Milestone 4.

## 2026-06-12 — Milestone 4 unblocked; literature review; detailed results doc
- **Did:** Implemented `SpringWrapper` in `env.py` (injects `τ_spring(θ)` at knee
  DoFs via `qfrc_applied`, beside the motors so `qfrc_actuator` stays "motor
  torque"); verified under jit and via full CPU smoke-train. Ran the Milestone 3
  post-hoc analysis end to end. Wrote `docs/RESULTS.md` (detailed numbers/methods
  across Milestone 1–Milestone 4). Launched the deep-research prior-art/novelty survey.
- **Decided:** Spring enters through `qfrc_applied`, not the actuator path —
  keeps the energy model honest. Lightened the literature workflow's adversarial
  verification from 3-vote to 1-vote after it twice stalled on the API usage
  limit (75 verify agents → 25); accepting weaker robustness, will spot-check the
  near-scoop papers by hand.
- **Result / numbers:** Milestone 3 post-hoc (constant −12 N·m, fixed gait, placeholder
  Kt/R, no-regen): knee copper loss −41.5 %/−35.8 % (L/R), total knee electrical
  −16.1 %. See `docs/RESULTS.md`. Lit-review preliminary (UNVERIFIED, from the
  failed run's logs): closest prior art is a Skoltech PEA-knee paper (same
  energy accounting but no RL), STEPPR (parallel springs, no gait
  co-adaptation), Duke Humanoid (efficiency RL, no parallel knee element) —
  our RL-co-adaptation-on-a-commercial-humanoid combination appears unclaimed.
- **Open / broken:** Milestone 4 retrain not yet run. Real G1 Kt/R still needed —
  and the survey REFUTED the no-regen assumption (not supported by literature),
  so it must be justified from G1 driver specs, not asserted.
- **Next:** run the Milestone 4 spring arm on a fresh H100 box and compare best-vs-best.

### Update (2026-06-12, eve): spring-mechanism derived but NOT novel; DecARt; Direction 1 chosen
- **Did:** Derived the "two offset half-parabolic springs → tunable linear spring"
  mechanism (`docs/mechanism.md`): K_eff=2k(p₂−p₁), θ₀=(p₁+p₂)/2, exact. Ran a
  cost-disciplined novelty workflow (21 agents, Haiku search/fetch + Fable verify).
- **Found:** **Mechanism is NOT novel** — exact scoop by Hurst et al. AMASC 2004
  (F_eff=4K·x₃·(x₂−x₁), their pretension x₃≡(p₂−p₁)/2), Migliore 2005, PMC10451064
  2024. Verdict: adopt & cite, do not claim. Novelty stays on the application
  stack (parallel packaging + RL co-adaptation + electrical CoT). Mechanism-level
  contribution would have to be *earned* empirically (onset-block vs pretension
  advantage, or non-overlapping dead-zone regime — but our overlap data argues
  against the latter). **DecARt Leg (MIPT, arXiv:2511.10021)** confirmed as the
  Direction-2 decoupled-leg platform (6 servos, proximal; flags springs as future
  work).
- **Decided:** Pursue **Direction 1 first** (hip-pitch tunable spring, sweep over
  speed × incline × load, adapt (K_eff,θ₀), measure electrical CoT reduction);
  Direction 2 (DecARt-style decoupled leg) is the morphological follow-up.
- **Next:** build the Direction-1 experiment setup — config grid + analysis to
  compare energy across conditions — before spending GPU time.

### Update (2026-06-12): literature survey landed + hip-pivot evidence
- **Did:** Completed the deep-research survey (47 agents, 24/25 claims confirmed)
  → full report in `docs/related_work.md`. Ran the E[τ|θ] decomposition on the
  baseline trajectory for knee AND hip-pitch.
- **Decided / found:** (a) **Novelty = integration, not new principle** —
  unclaimed cell is parallel element + in-loop RL co-adaptation + commercial
  humanoid + electrical accounting. Closest prior art: Bjelonic ETH RA-L 2023
  (same τ² metric + RL co-design, but QUADRUPED knee) and the group's OWN
  Belov/Osokin Skoltech 2024 (analytic, fixed PD, leg-stand — must cite/differ).
  Venue: RA-L/ICRA/IROS/Humanoids; top journals need hardware + real Kt/R +
  param co-optimization. (b) **Hip-pitch is the better PEA target than the
  knee**, confirmed in our own data: buildable linear spring captures ~51–60%
  of mean-square hip torque (near the 0.53–0.64 ideal ceiling), vs the knee's
  collapse to a constant at 36–41%. const-preload explains ~0% at the hip (AC
  joint) vs ~40% at the knee (DC joint). STEPPR's biped wins were also at the
  hip — consistent.
- **Open:** stance/swing knee-θ AND hip-θ ranges OVERLAP (both joints), so an
  angle-keyed dead zone can't gate either; hip's AC/symmetric nature may let an
  always-on spring help both phases without a clutch (needs in-loop test).
- **Next (proposed, awaiting go):** pivot Milestone 4 to hip-pitch — add E[τ|θ] decomp to
  `analyze.py`, write `configs/spring_hip_linear.yaml`, retrain on H100.

---

## 2026-06-11 — Milestones 1–3 in two days: baseline walks, spring hypothesis inverted
- **Did:** Full scaffold (2026-06-10) + every connection: GitHub (kefir8888/adaptive-elastics),
  Colab notebook, Google Drive sync, immers.cloud H100 box driven end-to-end over SSH.
  Trained the 200M-step baseline on H100 PCIe (57 min, ~71k steps/s steady, final eval
  reward 12.46, ~325 ₽). Local CPU rollout: walks 10.76 m / 12 s (0.90 of 1.0 m/s command).
  Built work-loop fitting, post-hoc spring subtraction (`pea-analyze --spring`), reward
  plots. Colab T4 ran the same config in parallel: reward-vs-steps curves overlap the
  H100's almost exactly (strong cross-hardware reproducibility evidence).
- **Decided:** (1) Rented H100 over Colab — T4 measured 7–10k steps/s (≈7 h/run) vs 71k
  (≈1 h); per-run cost ≈ identical, ~115 ₽. (2) `impl: jax` everywhere (Playground 0.2.0
  defaults to Warp — broken on Mac). (3) jax pinned <0.10 (brax 0.14.2 incompat).
  (4) Added a `constant` spring kind — see below; it, not the linear spring, is the lead
  Milestone 4 candidate. (5) Train full 200M for comparison runs (reward still climbing at 170M).
- **Result / numbers:** run `pea_runs/2026-06-11_baseline_h100` (+ partial T4 twin
  `_baseline_2`, stopped at ~75M). Knee work loop is OFFSET-dominated: flexed-knee gait
  carries ~−12 N·m gravity-support torque; constrained (k≥0) spring fit degenerates to
  k=0 ⇒ constant-torque (preloaded) element. Post-hoc on fixed gait: knee copper loss
  −41.5%/−35.8% (L/R), total knee electrical −16.1% (placeholder Kt/R; copper %s are
  Kt/R-independent). Optimistic bound by construction.
- **Open / broken:** real G1 knee actuator constants (Kt, R) needed — they set the
  copper:mechanical blend in headline numbers. Swing-phase cost of always-engaged preload
  visible in data but gait-feasibility (foot clearance) unknown until in-loop. Deep
  literature review (novelty/prior art, three-level report) running in background.
  Spring injection into the MJX env step not yet implemented (Milestone 4 blocker).
- **Next:** implement spring torque injection in `env.py` (wrapper adding τ_spring at
  knee DoFs inside step), smoke-test, then Milestone 4: retrain with `spring_constant.yaml` on a
  fresh H100 box and compare best-vs-best.

- **Did:** what got built / run / changed.
- **Decided:** any choice made, and the reason (so it can be revisited later).
- **Result / numbers:** key metrics if a run happened (baseline CoT, spring CoT,
  % copper-loss change) and the run-folder name.
- **Open / broken:** anything unfinished, failing, or uncertain.
- **Next:** the single most important next step.

---

<!-- Example entry — delete once you have real ones:

## 2026-06-11 — Repo scaffold + baseline trains
- Did: set up repo; env.py wrapping Playground G1; stubbed springs.py / energy.py.
  Baseline walk policy trains on Colab T4 in ~40 min.
- Decided: hardcode one baseline + one spring config for now; add the YAML config
  system only when sweeping. Reason: keep the first demo simple.
- Result: policy walks straight; run folder `outputs/2026-06-11_baseline`. No energy
  numbers yet.
- Open: G1 knee motor constants (Kt, R) still approximate — need to confirm.
- Next: implement energy.py copper-loss model, then run the post-hoc spring subtraction.

-->
