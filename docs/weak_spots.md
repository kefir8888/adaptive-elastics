# Weak spots — a harsh-reviewer red-team of the Parallel-Elastic Efficiency Study

A deliberately adversarial reading of `RESULTS.md`, `load_program.md`,
`negative_results.md`, `running_program.md`, `CLAUDE.md`, `JOURNAL.md`, and the
source (`energy.py`, `metrics.py`, `env.py`, `payload.py`, `go1_capacity.py`,
`power_compare.py`). The goal is to find every place a Reviewer 2 could plant a
flag. Many of these are already half-acknowledged in the docs; this file collects
them in one place, sharpens them, and ranks the fixes.

---

## Problem-statement weaknesses

1. **Low novelty; the study's own related-work doc concedes it.** Every ingredient
   is prior art: parallel elastic offloading (Plooij 2012, STEPPR 2017), τ²-electrical
   PEA + RL co-design on a *quadruped* (Bjelonic 2023), spring+RL distillation (PIL
   2026), and the tunable-spring mechanism itself is an *exact scoop* (Hurst 2004
   AMASC, Migliore 2005, the group's own Belov/Osokin 2024). `related_work.md:39`
   states the verdict outright: "integration/application novelty, not a new mechanism
   or principle." The "unclaimed cell" is a conjunction (commercial humanoid +
   always-engaged parallel + in-loop RL + battery-electrical accounting). A reviewer
   can reasonably call this **incremental**: change the platform (humanoid→quadruped),
   change the metric (τ²→battery-electrical), keep everyone else's method.

2. **The headline finding is largely a negative result, and the positive result is
   the most-anticipated cell.** "Gearing is the crux" → low-gear quadrupeds benefit,
   high-gear humanoids don't. But Bjelonic 2023 already showed PEA pays on a
   quadruped, and `taxonomy.md`/`directions.md` derive the gear-dependence *a priori*
   from R/Kt². So the one positive result (Go1 −17…−27%) confirms a hypothesis the
   project itself calls predictable, and the negatives confirm that a 22.5:1 humanoid
   has ~4% ohmic — which is arithmetic, not discovery. The genuinely novel-feeling
   claim (post-hoc can flip *sign* in-loop, NR-3) is a methodological caution, not a
   capability.

3. **The goal keeps moving.** Knee → hip-pitch (G1) → "G1 doesn't pay" → Go1
   constant preload → adaptive per-leg preload → load-carrying → capacity ceiling →
   "30 kg is unphysical, real study is 0–6 kg." Each pivot is individually defensible,
   but the cumulative effect is a problem statement defined *post hoc* by what
   happened to work. A reviewer will ask: what was the pre-registered hypothesis, and
   what is the falsification criterion? The current headline ("gearing is the crux")
   was not the starting hypothesis (which was "a hip/knee spring cuts G1 walking
   energy"); it is the explanation for why the original hypothesis failed.

4. **Significance is capped by simulation-only scope.** No hardware, estimated motor
   constants (no datasheet for Kt/R on *either* robot — `energy.py:43-69` flags every
   constant as an estimate), and the central energetic assumption (no-regen) is an
   *engineering judgment*, not a verified spec (`RESULTS.md` open-item 3). The whole
   electrical CoT story rests on numbers the project admits it cannot ground. The
   related-work doc itself notes a top venue "needs hardware + real Kt/R + param
   co-optimization" (`related_work.md:142`).

5. **"Capability > energy" (load_program.md:21-23) is the strongest *claim* but the
   weakest *evidence*.** The capacity-extension story depends entirely on the motor
   torque ceiling binding — and the docs now admit (load_program.md:137-152) that in
   plain sim the Go1 walks at 30 kg (3–6× its real rated max) without failing, so the
   peak-torque ceiling does *not* bind at realistic loads. The capability claim
   therefore has **no validated regime** in the current sim: below ~10 kg nothing
   fails, and above ~10 kg the sim is unphysical. The claim is asserted, not shown.

---

## Methodology weaknesses

1. **The speed confound is not actually fixed — the headline is reported in WATTS, not
   CoT.** This is the single biggest methodological hole. The Go1 headline
   (`RESULTS.md:22`, 153.4 → 127.8 W) and the G1 headline (`negative_results.md:41`,
   151.6 → 162.8 W) are **mean electrical power (W)**, compared at a fixed *command*
   (1.0 m/s). But a retrained policy is free to walk at a different *achieved* speed,
   and `power_compare.py` / the headline comparison do **not** normalize by distance.
   - `go1_capacity.py:81` computes achieved speed but does not gate or speed-match the
     energy comparison on it; `metrics.evaluate` computes CoT (`metrics.py:205`) but
     the headline % is quoted as W.
   - Why it bites: a spring policy that walks even slightly *slower* draws less power
     while looking more efficient; one that walks *faster* draws more power while being
     more efficient per metre. CoT (E / m·g·d) is the only fair fix, and the headline
     does not lead with it. CoT for the Go1 positive result is **not reported at all**
     in `RESULTS.md`; only the G1 negative quotes CoT (+8.4%). **A reviewer will demand
     the achieved-speed table for both arms and the CoT, and will not accept a W-vs-W
     comparison as an efficiency claim.**
   - The watchdog memory explicitly lists "speed confound" as a standing concern; the
     code does not yet enforce speed-matched evaluation.

2. **Seed count is too low and asymmetric across the study.** G1 negatives are
   "single-seed (multi-seed deferred)" (`negative_results.md:17`) — the central
   negative (NR-2, +7.4%) is a 4-*reset-seed* mean of essentially one *training* seed,
   not 4 independent training seeds (the "4-seed mean" in `negative_results.md:40` is
   reset-seed rollouts of one policy; only the cross-condition 2×2 adds a from-scratch
   policy). The Go1 positive is 2–3 *training* seeds. `running_program.md:84` itself
   sets the bar at "≥3 seeds (5 credible)." So the published headline numbers do not
   meet the project's own stated standard, and the comparison is best-of-available, not
   best-of-N.

3. **Baseline parity is compromised by the energy-naive baseline (the project's own
   "key confound").** `RESULTS.md:243-251` and `running_program.md:119-138` admit it:
   the G1 spring numbers are measured against a walker trained with the energy weight
   at **zero**. The correct control is an energy-*aware*, no-spring baseline; an
   energy-aware policy will self-trim torque and de-chatter, shrinking the spring's
   marginal value. So the post-hoc %s are upper bounds against the wrong reference. The
   Go1 runs *do* carry an energy penalty (−1e-4, `go1_baseline.yaml:19`), but it is
   tiny and was never calibrated to be a real lever (NR-6 shows a 10× sweep on the G1
   moved nothing). Net: the baseline is not demonstrably "as efficient as RL can make
   it without a spring," so the spring's measured edge may be partly the baseline's
   slack.

4. **The no-regen assumption is doing nearly all the work, and it is unverified.**
   NR-5 + `RESULTS.md:208-212` are explicit: the entire G1 post-hoc win is intercepted
   braking energy; under true regeneration the saving → ~0%. The Go1 result is also
   "no-regen-dependent" to a degree the docs do not quantify as cleanly as the G1's
   ~24%. Since the headline is "a parallel spring saves electrical energy," and that
   saving exists *only* in the no-regen world, the result is one hardware braking test
   away from collapsing to ~0. The assumption is justified by back-EMF reasoning
   (`running_program.md:18-26`) but explicitly "NOT a verified G1 spec," with named
   counterexamples (MIT Cheetah, Tesla Optimus). A reviewer treats an unverified,
   outcome-determining assumption as a fatal dependency until measured.

5. **Sim-to-real: peak vs continuous/thermal torque is conflated, and the capacity
   ladder is unphysical.** `load_program.md:142-152` now concedes the model enforces
   only *peak* (`jnt_actfrcrange`) torque, not *continuous/thermal* limits, so the Go1
   "walks" at 30 kg = 2.5× body mass, 3–6× real rated max. Two consequences:
   (a) every 15–30 kg energy/capacity number is sim-only and must be labeled as such
   (the docs say so but the numbers still circulate); (b) the *capability* claim
   ("preload extends carry capacity") depends on a thermal limit the sim does not
   model — the planned thermal-cap experiment (load_program.md:148) is unrun, so the
   capability claim is currently **untestable in the existing sim**. The whole-robot
   power accounting (house load, regen tax) is also literature-estimated, not measured.

6. **Eval hygiene: fixed-command rollout masks distance/standing degeneracies.** The
   0–25 kg collapse-to-standing (`JOURNAL.md:11-28`) was only caught because a *constant
   forward* command exposed it while the in-training `tracking_lin_vel` reward (evaluated
   on *random* commands) stayed high (367 vs 925). This is a near-miss: the in-loop
   reward metric did not reveal that the policy had stopped walking forward. It is not
   guaranteed the same masking is absent in the "good" 0–6/0–10 kg runs — the eval must
   confirm genuine forward locomotion (speed ≳ 0.8 m/s) at every load before any energy
   number is trusted, and `go1_capacity.py` only recently added speed (and its sweep was
   capped at 15 kg, load_program.md:152). Survival is counted as "did not terminate"
   (`go1_capacity.py:71`), which a standing-still policy passes trivially.

7. **Identifiability of the mechanism is weak.** The claim is "the spring offloads the
   *constant support* component of knee torque, cutting ohmic." But the spring is
   injected as a raw `qfrc_applied` constant (`env.py:163`) and the policy retrains
   around it. There is no ablation isolating *why* the gait got cheaper: is it the
   offload (the advertised mechanism), or did the retrained gait simply find a lower-
   torque posture it could have found anyway, or did it shift work to *unsprung* joints
   the whole-body W happens to favor? The cross-condition 2×2 (NR-2) is done only for
   the G1 negative; the Go1 positive lacks the symmetric control (spring-trained policy
   in a no-spring world, and vice versa) that would prove the gain is the spring and not
   a luckier seed. The energy-vs-load *curve* + per-leg τ₀-vs-load curve are asserted as
   the headline (JOURNAL "Next") but, per the journal, the clean version was not yet run
   at a walkable load.

8. **Reward-hacking risk in the adaptive loop and the energy penalty.** Two surfaces:
   (a) The adaptive controller drives mean motor knee torque to ~0 (`go1_capacity.py:40`,
   ē_target=0). Reading the offload off the motor's *own* post-offload torque
   (`go1_capacity.py:68`) is a closed loop that can chase its own tail: the preload
   reduces measured torque, which is the signal that sets the preload. The docs argue
   time-scale separation makes this benign, but there is no stability/convergence proof
   beyond one EMA constant, and no demonstration that it doesn't over-preload and inject
   energy on flat ground or push the leg into a different (cheaper-looking but
   higher-CoT) gait. (b) The total-electrical penalty (`ElectricalRewardWrapper`,
   `env.py:80-88`) rewards lowering `τ·ω + (τ/Kt)²R`; a policy can lower this by walking
   slower or shortening stride — exactly the standing degeneracy already observed — which
   is reward hacking against the *efficiency* objective unless tracking holds speed
   firmly. The 0–25 kg standing collapse is a documented instance of precisely this.

9. **The energy model omits iron loss and the chatter inflates ohmic.** `energy.py:18-25`
   and `negative_results.md:115-120` admit iron (core) loss is omitted; it scales with
   *speed* (×gear on the G1), so the denominator is undercounted — diluting (already
   small) G1 savings and, on the Go1, changing the 54% ohmic share. Separately,
   `running_program.md:96-105` shows the baseline gait *chatters* (control reverses
   ~55%/step, peak/RMS torque up to 4.4), and the no-regen clamp *rectifies* that
   high-frequency torque into spurious dissipation — inflating the baseline's measured
   energy by ~26–40% and roughly *doubling* the spring's apparent % saving. If the
   baseline number is partly an artifact of un-penalized chatter, the spring's headline
   reduction is measured against an inflated reference. The fix (action_rate on) exists
   for the runner config but the headline walking/Go1 numbers' chatter status is not
   uniformly stated.

10. **Generalization is thin.** Rough terrain is acknowledged "weak" — only a
    `terrain_height_scale` curriculum knob (`env.py:34-44`) and half-amplitude configs
    (`go1_baseline_rough_half.yaml`), with no reported rough-terrain energy result that
    holds up. One robot (Go1) carries the entire positive result; Go2/Spot/B2 are
    "later" and Spot's electricals are "blocked → qualitative only"
    (`load_program.md:124`). A single-platform positive against a single-platform-class
    prediction is not a generalization claim.

---

## Threats to validity

- **Construct validity:** "electrical CoT" is operationalized with estimated Kt/R,
  no iron loss, and a no-regen clamp that converts the no-regen *assumption* into the
  *source* of the win. The metric measures the assumption as much as the physics.
- **Internal validity:** the energy-naive baseline + low seed count + W-not-CoT
  comparison + un-isolated mechanism mean the measured spring effect cannot be cleanly
  attributed to the spring versus baseline slack, seed luck, speed drift, or chatter.
- **External validity:** sim-only, one robot for the positive, unphysical high-load
  regime, estimated constants, no hardware — generalization to a real Go1 (let alone
  other platforms) is unestablished.
- **Statistical validity:** no variance/CI on the headline %s; "−16.7% / −19.7%"
  (2 seeds) and "−17 to −27%" (3 seeds) are reported as a range, not a mean±std with n,
  and the G1 "4-seed mean" conflates reset-seeds with training-seeds.
- **Conclusion validity:** "gearing is the crux" is supported by exactly two gear
  points (G1 22.5:1, Go1 6.33:1) plus a priori R/Kt² arithmetic; a monotone two-point
  trend is consistent with the claim but does not establish the *boundary* the project
  advertises (`load_program.md:11-13`). Spot (high-gear counter-test) is unrun.

---

## What a reviewer would reject

A top-venue (RA-L/ICRA/IROS) reviewer would most likely **reject or major-revise** on:

1. **No hardware + outcome-determining unverified assumptions** (no-regen, estimated
   Kt/R on both robots). The entire electrical saving lives in an unmeasured braking
   channel. This is the single most likely rejection ground for an *energy-efficiency*
   paper.
2. **Headline efficiency claim reported in watts, not CoT, with no speed-matched eval
   and no achieved-speed table.** An efficiency claim that does not control for speed is
   not publishable as stated.
3. **Insufficient seeds / no confidence intervals**, below the project's own ≥3
   standard, with the G1 negative resting on effectively one training seed.
4. **Wrong baseline** (energy-naive) for the G1 numbers, conceded internally as "the
   key confound" but still underlying the published deltas.
5. **Incremental novelty** — Bjelonic 2023 (quadruped PEA + RL) + Hurst 2004
   (mechanism) leave only an integration claim, and the strongest positive (Go1) is the
   most-expected cell.
6. **Unphysical capacity regime** undercutting the "capability extension" claim, which
   has no validated (thermal-limited) test in the current sim.

It would more plausibly **accept as a careful negative-result / methodology paper**
(the post-hoc-can-flip-sign finding, NR-3, plus the honest gear-dependence map) than as
a positive efficiency contribution — *if* the seeds, CoT, baseline, and one hardware
braking measurement were added.

---

## Concrete fixes (ranked)

1. **Report CoT, not watts, and add a speed-matched / achieved-speed table for every
   comparison.** Lead the Go1 positive with CoT(spring) vs CoT(no-spring) at matched
   achieved speed, with the per-arm m/s in the table. Either command-match *and* verify
   achieved speed within a tolerance, or regress energy on speed and compare at a common
   speed. Cheapest high-impact fix; closes the biggest hole. (`metrics.evaluate` already
   computes CoT — just promote it to the headline and gate on speed.)
2. **Train the energy-aware, no-spring baseline and recompute every delta against it.**
   Calibrate the energy weight to be a real lever first; report
   CoT(energy-aware+spring) − CoT(energy-aware, no-spring). This is the project's own
   stated correct control (`running_program.md` Milestone 1b) — do it before any
   headline.
3. **Raise to ≥3 independent *training* seeds per arm on both the G1 and Go1, and report
   mean ± std (or CI) with n.** Re-label the G1 "4-seed" number as reset-seeds and add
   true training-seed replication. No new physics, just GPU time.
4. **Add the symmetric cross-condition control for the Go1 positive** (spring-trained
   policy in no-spring world and vice versa, as already done for the G1 negative) to
   prove the saving is the spring, not the seed/posture. Add an ablation that holds the
   gait fixed and only toggles the preload to separate offload from gait change.
5. **Quantify the no-regen sensitivity for the Go1 explicitly** (report Go1 CoT under
   regen and no-regen, the way the G1's ~24% is reported), and frame the headline as a
   *band* over that axis. State plainly that the win is a passive substitute for
   regeneration. Gold standard, if any hardware is ever available: one braking test
   (command a braking torque, watch the DC-bus voltage).
6. **Make the capability claim testable: impose a continuous/thermal torque cap** (the
   planned `I²R`/thermal-budget experiment, `load_program.md:148`) so the Go1 fails at a
   realistic ~5–10 kg, and show the preload extends *that* limit. Until then, drop or
   heavily caveat all 15–30 kg numbers as sim-only and remove the capacity-ladder from
   any headline.
7. **Run the gear-boundary counter-tests** (Spot high-gear quadruped should *not* pay;
   one more low-gear point) to turn "two gear points + arithmetic" into an actual
   boundary map, which is the project's claimed rigorous contribution.
8. **Verify forward locomotion at every evaluated load** (speed ≳ 0.8 m/s gate, not just
   "episode did not terminate") and extend `go1_capacity.py` past 15 kg only with the
   thermal cap in place. Add iron-loss bounds (or state the omission's direction) to the
   energy denominator so the % savings are not silently inflated by chatter rectification
   — confirm every headline is on a de-chattered (action_rate-on) policy.
9. **Reframe novelty honestly up front** as integration + a methodological caution
   (post-hoc can invert in-loop) + a gear-dependence map, and cite Bjelonic 2023 /
   Hurst 2004 / Belov-Osokin 2024 as the differentiators they are. Pitch as a
   negative-result/methodology paper unless the hardware + CoT + seed fixes land.
