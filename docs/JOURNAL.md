# Project Journal — Parallel-Elastic Knee Efficiency Study

Append a short dated entry at the **end of each work session** (newest at the top).
Keep entries terse — this is a memory aid for future sessions, not documentation.

For each entry note: what you did, what you decided (and why), key numbers if a run
happened, what's open/broken, and the single next step.

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
  M4 candidate. (5) Train full 200M for comparison runs (reward still climbing at 170M).
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
  Spring injection into the MJX env step not yet implemented (M4 blocker).
- **Next:** implement spring torque injection in `env.py` (wrapper adding τ_spring at
  knee DoFs inside step), smoke-test, then M4: retrain with `spring_constant.yaml` on a
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
