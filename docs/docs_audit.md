# Docs audit & paper-readiness — 2026-06-16

Audit of all `docs/*.md` + `README.md` + `CLAUDE.md` against each other **and against
the actual experimental record** (run folders + capacity logs in `outputs/`). Question:
do the documents form a coherent, non-contradictory, paper-ready picture? Verdict up
front: **the science is coherent and well-argued, but the documentation has fallen ~1.5
sessions behind the experiments.** The largest, most important block of results — the
entire load-carrying program and a capacity-to-failure ladder run autonomously on
2026-06-15/16 — exists only as `outputs/*.log` and was never written into `RESULTS.md` or
`JOURNAL.md`. Several headline numbers are quoted at three different values across files.

---

## Coherence

The **conceptual spine is strong and consistent across every file:** "gearing is the
crux" — a parallel elastic does **not** pay on the high-geared G1 (22.5:1, ohmic ~4%,
in-loop +7.4% worse) but **does** on the low-gear Go1 (6.33:1, ohmic ~54%, constant knee
preload). This thesis, the post-hoc-vs-in-loop methodology, the no-regen dependence, the
"constant preload not linear spring" element-kind argument, and the series-vs-parallel /
wall-binding framing for Part 2 are stated compatibly in `CLAUDE.md`, `RESULTS.md`,
`negative_results.md`, `directions.md`, `taxonomy.md`, and `mechanism.md`. The
mechanism-novelty verdict (not novel; Hurst 2004 / Migliore 2005) and the
integration-novelty positioning agree across `mechanism.md`, `related_work.md`, and
`negative_results.md` (NR-9). The constants story (`constants_registry.md`,
`g1_motor_constants.md`, `robot_inventory.md`) is internally consistent and unusually
disciplined about provenance.

The doc **set is well-structured by role** (JOURNAL = memory, RESULTS = numbers/methods,
negative_results = headline output, directions/taxonomy = strategy, the three
constants/inventory docs = verification). That separation is sound.

**The break in coherence is temporal, not logical.** `JOURNAL.md`'s newest entry is
2026-06-15 "session 2" (02:06), which ends with the load program *broken* ("payload-DR
broke locomotion") and a "FIX PREPARED" at 0–10 kg that had **not yet been run**.
`RESULTS.md`'s last substantive section is dated 2026-06-14. But the git history and
`outputs/` show that between 2026-06-15 11:00 and 2026-06-16 01:00 a large autonomous
campaign ran to completion: the 0–6 kg fix, a 3rd seed, a 15 kg curriculum, rough
terrain, a full baseline+spring capacity-to-failure ladder to 30 kg, and a first G1
running baseline + jog. **None of this is in RESULTS or JOURNAL.** A reader following the
prescribed ritual ("skim JOURNAL → Current state") would conclude the load program is
broken, when in fact it succeeded. This is the dominant coherence problem.

`README.md` "Current state" is dated **2026-06-14** and still presents the project as if
the *G1 hip gate / running program* is the active focus ("Next focus: G1 running for
EFFICIENCY"), predating the Go1 positive result, the load program, and the capacity work.
`RESULTS.md` still carries **"Milestone 4 — In-loop GO/NO-GO gate (hip-pitch) ◻ ready, not
yet run"** as an open milestone, even though `negative_results.md` NR-2 reports that exact
gate as **run and negative**. So within the doc set, the same experiment is simultaneously
"not yet run" (RESULTS, PLAN) and "the central negative result" (negative_results,
CLAUDE, README).

---

## Contradictions & stale claims

**C1 — The Go1 walking headline % is quoted at three different values.**
- `RESULTS.md:24`, `JOURNAL.md:82`, `negative_results.md:125`: **−16.7% / −19.7%** (2 seeds).
- `CLAUDE.md:25`, `load_program.md:4`, `negative_results.md:6`: **−17 to −20%** (2 seeds).
- `load_program.md:143`: **−14 to −27%** ("our −14 to −27% CoT result").
These cannot all be the headline for the same result. The actual run logs (see "Gaps")
give whole-body *electrical* savings at no-load of roughly **−9.5% (s1), +0.8% (s2),
−5.6% (s3)** and *CoT* savings of **−16.6% (s1), −3.4% (s2), −13.9% (s3)**, plus
**−16.8%** for the curriculum seed. So: (a) the original "−16.7/−19.7% whole-body
electrical" pair does not match the newer p6-family electrical numbers (different
trained policies / payload-DR conditions); (b) seed 2 is nearly a wash at no-load (−3.4%
CoT, +0.8% electrical), which **no doc reports** — every doc presents 2–3 uniformly
strong seeds; (c) the "−27%" upper bound is a per-payload CoT figure, not the headline.
Pick one definition (electrical vs CoT, which seed set, which payload) and quote one band
everywhere, with the per-seed spread shown.

**C2 — Seed count is stale.** `CLAUDE.md`, `RESULTS.md`, `negative_results.md`,
`load_program.md` all say the Go1 positive is confirmed across **2 seeds**. A **3rd seed**
(p6s3) was trained and evaluated (`outputs/box_logs/overnight.log`, `cap_*_p6s3.log`).
The task framing already says "3 seeds". Docs say 2.

**C3 — Load program status: "broken / fix prepared" vs "validated".** `JOURNAL.md:11–39`
(newest entry) and `load_program.md`'s "Status" section (lines 131–135) describe the
0–25 kg collapse and a *prepared* fix, present tense, as the live state. In reality the
fix ran and worked (0–6 kg gates at tracking_lin_vel ~895–908; `pipeline.log`). The
"RESUME (next session)" checklist in JOURNAL and the "Next: build the adaptive-preload
spring run" in load_program are both **already done**. Stale.

**C4 — Capacity ceiling: "stops walking, not falls" vs the actual ladder.**
`load_program.md:44–46` and JOURNAL state the heavy-load failure mode is *standing*
(stops walking), and `load_program.md:137–152` (added 2026-06-16) flags that the sim Go1
"walks at 30 kg = unphysical" and that a capacity-to-failure ladder is **"NOT YET RUN"**.
But `overnight.log` shows the ladder **was run that same night**: baseline *and* spring
both reach "MAX WALKABLE = 30 kg" (tracking gate passed at 20/25/30 kg). So the
"NOT YET RUN — 2026-06-16" label on the beyond-30 kg experiments is half-stale (the
plain-sim ladder to 30 kg ran; the *thermal-limited* capacity experiment, which is the
scientifically meaningful one, genuinely has not). The doc and the log disagree on what
exists.

**C5 — Hip-pitch gear ratio: 22.5:1 vs 14.3:1, unresolved and propagated inconsistently.**
`CLAUDE.md`, `RESULTS.md`, `directions.md` (gear table line 228), and the headline of
every doc treat the **hip-pitch spring target as 22.5:1**. But `g1_motor_constants.md`
(§3), `constants_registry.md` (row "Gear ratio", flagged KNOWN-SUSPECT), and
`robot_inventory.md:55,94` conclude the **hip-pitch is 14.3:1, not 22.5:1**, and that the
team's CLAUDE.md/RESULTS value is *wrong*. This is a live internal contradiction: the
strategy/results docs use 22.5:1 for the spring joint while the verification docs say
14.3:1 and that it changes the per-joint ohmic share and the hip-pitch R/Kt² by ~2.5×.
Not yet reconciled in the headline docs.

**C6 — R/Kt² history reads as self-contradicting unless read in full.** `RESULTS.md`
(Motor constants section) still says the code constants are "estimated… either way
absolute watts are not trustworthy" and elsewhere the project once believed the code was
"8–19× too optimistic." `g1_motor_constants.md` and `constants_registry.md` later
**retract** that alarm (a 100× arithmetic error in the web chase) and conclude the code
value 0.0025 is the *optimistic edge of a defensible band*, not wrong by 10×. The
retraction is correct and well-documented, but `RESULTS.md` was not updated to point at
it, so a reader of RESULTS alone gets the superseded "untrustworthy / possibly 10× off"
framing. Stale cross-reference, not a logic error.

**C7 — Plooij & Wisse citation is internally flagged as disputed but still load-bearing.**
`related_work.md` §1/§2 carry a DISPUTED tag on the ~80%/66% Plooij figure (adversarial
re-check found ~20% on a 2-DOF arm) yet still list Plooij 2012 as the foundational
"parallel torque offloading cuts energy" anchor in the gap table (§3) without the caveat.
Minor, but a reviewer will catch a disputed number used as a foundation. Resolve the
citation before write-up.

**C8 — README workflow is stale.** `README.md` still documents the run path as the **G1
hip-spring Milestone-4 gate** ("To run Milestone 4 on a GPU machine", `spring_hip_linear`
vs `baseline_gate`). The actual active pipeline is the Go1 load program
(`go1_baseline_payload` / `spring_go1_adaptive`, `scripts/phase_b_run.sh`,
`go1_capacity.py`). README points a new user at the wrong, already-completed experiment.

**C9 — Minor numeric/label drifts.** (a) `constants_registry.md:100` itself notes
`analyze.py:116` uses a different distance index than `metrics.py` (a real, unfixed
minor inconsistency in the code that feeds CoT). (b) `taxonomy.md:131` and
`directions.md:137,224` still carry Berkeley Humanoid **9.1:1**, which `robot_inventory.md:96`
explicitly corrects to **9:1** (9.1 is HECTOR) — stale in two docs. (c) `directions.md`
gear table lists Spot at "~80–160:1" while `robot_inventory.md:62` corrects it to hip
**51:1**.

---

## Gaps

**G1 (the big one) — the load-carrying results are not written up anywhere in `docs/`.**
The entire payoff of the current direction lives only in `outputs/*.log` and
`outputs/box_logs/*.log`. Specifically missing from RESULTS/JOURNAL:
- **Energy-vs-load curves**, baseline vs adaptive preload, 0→15 kg, per seed
  (`capacity_*_p6*.log`, `cap_*_p6s3.log`). E.g. seed 1 no-load 164.8→149.1 W; the
  spring's electrical advantage *grows* with load (at 15 kg, 368.8→312.5 W ≈ −15%).
- **The per-leg adaptive τ₀-vs-load behavior actually converging** (`conv_tau0` vectors
  in the logs ramp front>rear, e.g. `[6.2 5.7 4.1 3.9]`@0kg → `[8 8 7.8 7.6]`@15kg),
  which is the *headline mechanism validation* the JOURNAL only described as a plan.
- **The capacity-to-failure ladder** (baseline & spring both walkable to 30 kg, with the
  explicit realism caveat that 30 kg is 3–6× the real Go1 rating).
- **A stability cost that the docs deny exists.** Docs repeatedly claim the Go1 spring has
  "no stability cost / 4/4 survival." But the adaptive policy in the capacity logs
  **loses survival at high load** (seed 1: 1261/1500 @12.5 kg, 1155/1500 @15 kg; seed 2
  drops as early as 871/1500 @5 kg) while the matched **baseline survives 1500/1500** at
  every load. At the loads where they diverge, the spring is *less* stable, not equal.
  This is a real, unreported nuance that contradicts the blanket "no stability cost"
  claim and must be stated honestly (the win is at low-to-mid load; at the heavy tail the
  adaptive preload trades some stability).

**G2 — Multi-seed rigor for the Go1 positive is thinner than the prose implies.** 3 seeds
exist, but at no-load one of them (s2) is essentially a wash (−3.4% CoT, +0.8% W). A
credible paper needs the mean±std across seeds reported, the seed-2 near-null explained
(or more seeds), and the comparison defined at a fixed, physically-meaningful payload
(0–6 kg) rather than cherry-picking the strong seed/payload. No doc currently reports
variance for the Go1 result.

**G3 — Energy-aware baseline confound is acknowledged but never closed.** `RESULTS.md`
(open item 5) and `running_program.md` (Milestone 1b) correctly flag that **every spring
number is measured against an energy-NAIVE walker** and that the proper control is an
energy-aware baseline. This is the project's own watchdog flag — and it is still open for
*both* the G1 and the Go1 tracks. The Go1 load-program baselines (`go1_baseline_payload`)
should be checked for whether they carry the electrical penalty; if not, the headline
Go1 saving is still a naive-baseline upper bound, same caveat the G1 numbers carry.

**G4 — No real Kt/R for any robot; absolute energy is a band.** Well-documented
(`g1_motor_constants.md`), but it remains the #1 publication blocker for any absolute
ohmic/CoT claim. The relative gate is R/Kt²-invariant, so the *direction* of every result
is safe; the *magnitude* is not citable until a bench 7520 (and a Go1/Go2) measurement
exists. The Go2 Kt sign/scale error (motor-side 0.26 used as joint-side; should be ~1.62)
is flagged in three docs but, per `constants_registry.md:67`, **not yet fixed in
`energy.py`** — any Go2 absolute number is ~39× off until corrected.

**G5 — Iron loss / omitted loss channels.** `negative_results.md` notes the energy model
omits iron (core) loss, which would *dilute* the (negative) G1 result and also enlarges
the Go1 denominator. Stated once; not carried into RESULTS or the Go1 headline. A
reviewer will ask whether the Go1 win survives a fuller loss model.

**G6 — Sim-to-real entirely absent (expected at this stage, but it is the gap to a top
venue).** No hardware, no measured battery energy, no real torque-speed curve, no
validation of the no-regen assumption by a bus-voltage test. `related_work.md` and `PLAN.md`
both correctly say this is what separates RA-L/ICRA from T-RO/IJRR. Until then the whole
study is sim-only.

**G7 — Part 2 (explosive) is fully designed but unstarted, and Part 1 is declared the
gate before Part 2.** `CLAUDE.md` says finish Part 1 first. Part 2 has rich reasoning
(B1/B2 split, wall-binding) but zero runs. A max-jump policy — the "decisive test"
named in JOURNAL — has not been trained. This is a planned gap, not a defect, but it
means the paper is a Part-1 paper.

**G8 — G1 running track is started but its status is undocumented.** A G1 running
baseline (202 M steps, reward ~5.9) and a Stage-1 jog (reward ~1.3–1.6) were trained
(`box_logs/g1_run_*.log`); `g1_running_design.md` exists. But Attempt 1 is logged as
FAILED (collapses to 0.85 m/s walk), the jog result is unevaluated in any doc, and
`directions.md`/`running_program.md` still present running as the "next focus" without
recording that two training attempts have now happened. The running track's actual state
(two attempts, one failed, S2 flight gate deferred for human check per overnight.log) is
not written down.

---

## Paper-readiness scorecard (per-section: have / need)

Overall readiness: **~40%** for a focused **negative-result + low-gear-positive
methods paper** (RA-L/ICRA tier); **~25%** for the broader cross-morphology or
load-capacity story; **~10%** for a top-tier (T-RO/Science Robotics) paper.

| Paper section | HAVE | NEED |
|---|---|---|
| **Motivation / framing** | Strong, written. "Gearing is the crux," τ²-copper mechanism, two value axes, bifurcation. | Trim to the result that survived; drop the speculative direction inventory from the paper body. |
| **Related work** | `related_work.md` + `mechanism.md`: thorough, honest integration-novelty verdict, danger-papers identified (Bjelonic 2023, Belov/Osokin 2024). | Fix the disputed Plooij figure (C7); re-run an adversarial prior-art search at submission (the doc says so); resolve the 2026-preprint scoop risk. |
| **Methods — sim/env/energy model** | Clear: MJX Playground, `P=τω+(τ/Kt)²R`, no-regen, post-hoc vs in-loop, work-loop fitting. | One real Kt/R (G1 + Go1) to make watts citable; fix the Go2 Kt scale bug; state iron-loss omission; pin the hip-pitch gear (14.3 vs 22.5). |
| **G1 walking — negative result** | The strongest, most defensible output. NR-1…NR-9 catalogued; in-loop +7.4%, 2×2 cross-condition control, robustness checks. | Multi-seed (currently single-seed per NR conventions); energy-aware baseline (G3); update RESULTS to mark Milestone 4 DONE (C). |
| **Go1 low-gear — positive result** | Real in-loop runs, 3 seeds, work-loop, videos, plots. The counterpoint that gives the negatives meaning. | **Write it into RESULTS** (G1); report mean±std, explain the seed-2 near-null (G2); state the high-load stability cost honestly (G1); energy-aware baseline (G3); converge on one headline number (C1). |
| **Load-carrying / adaptive preload** | Mechanism validated; per-leg τ₀ converges; energy-vs-load + capacity-to-failure ladder all RUN. | **Entirely unwritten** (G1). Then: thermal-limited capacity (the physical, mechanistic version — designed in load_program.md but not run); honest sim-realism caveat (already drafted). |
| **Robustness envelope** | Rough-terrain redo (easier terrain) run; payload DR; speed sweep in capacity eval. | Slope, direction, terrain-roughness sweeps (PLAN Milestone 6) not done; rough-terrain survival is poor in the logs (≤~700/1500) — needs work or honest reporting. |
| **Cross-morphology study** | Blueprint + inventory + cost estimate (`taxonomy.md`, `robot_inventory.md`). | Only G1+Go1 actually run. The £160–210 study is unstarted; this is a future-work section, not a results section. |
| **Part 2 (explosive)** | Design + wall-binding analysis. | No runs at all (G7). Out of scope for the Part-1 paper. |
| **Reproducibility / artifacts** | Run folders, configs per arm, seeds recorded, cross-hardware (T4 vs H100) check. | Some early runs lost (not synced in time — JOURNAL 2026-06-15); ensure all headline runs are archived; reconcile the hardcoded baseline-trajectory paths (constants_registry §7). |

---

## Single biggest gap

**The documentation does not contain the project's current best results.** The load-carrying
program — the *active direction* per `CLAUDE.md` — was designed, debugged (0–25 kg collapse
→ 0–6 kg fix), validated (per-leg adaptive τ₀ converging, energy-vs-load curves, 3 seeds,
a capacity-to-failure ladder to 30 kg), and these results exist as real run folders and
logs in `outputs/`. **None of it is in `RESULTS.md` or `JOURNAL.md`.** The newest JOURNAL
entry still says the load program is broken with a fix merely "prepared." A reader
following the project's own session ritual would mis-conclude the current state by a full
working day of successful experiments.

This single gap subsumes most of the contradictions above (C1 disagreeing headline
numbers, C2 stale seed count, C3 broken-vs-validated status, C4 capacity ladder
not-run-vs-run): they are all symptoms of RESULTS.md and JOURNAL.md not having been
updated after the 2026-06-15/16 autonomous campaign. **The single highest-value action is
to run `/wrap`: read `outputs/*.log` + `box_logs/*.log`, write the load-program and
capacity results into `RESULTS.md` with per-seed numbers (including the honest seed-2
near-null and the high-load stability cost), append a JOURNAL entry covering 2026-06-15
session-2-onward through 2026-06-16, refresh README "Current state," and reconcile the
Go1 headline to one band.** Until that is done the doc set is internally contradictory and
understates the work; after it, the project is a coherent ~40%-of-the-way Part-1 paper
(negative G1 + positive low-gear Go1 + adaptive load preload) whose remaining blockers are
the energy-aware baseline (G3), one real Kt/R measurement (G4), and the thermal-limited
capacity experiment that makes the capability claim physical.
