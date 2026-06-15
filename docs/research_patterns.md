# Research-process meta-analysis — Parallel-Elastic Efficiency Study

A retrospective on *how this project works*, not on what it found. Distilled from the full
`JOURNAL.md` history (2026-06-11 → 06-16), `negative_results.md`, and the program docs.
The science is healthy; the recurring losses are in *experiment setup, range/eval choices,
and GPU-box discipline*. Read this before launching the next GPU box.

---

## Recurring patterns

### Good (keep doing)
- **Post-hoc → in-loop discipline pays, repeatedly.** The project's single most valuable
  habit is refusing to trust a post-hoc (fixed-gait) spring number until an in-loop retrain
  confirms it. It caught a **sign flip** on the G1 hip (post-hoc −3.84 % → in-loop +7.4 %,
  NR-2/NR-3) and *confirmed* the Go1 win held in-loop (−14.9 % → −16.7 %). This is the
  methodological spine; every spring claim is gated on it.
- **Negative results are catalogued, not buried.** `negative_results.md` (NR-1…NR-9, each
  tagged EXPERIMENTAL vs REASONED) turned "the G1 doesn't work" into the load-bearing output.
  This reframing — "gear ratio is the crux" — only exists because failures were written down
  with their mechanism.
- **Diagnose before re-running.** The 0-m/s collapse (2026-06-15) was root-caused *cleanly*
  (ran the original walker through the same eval script → 0.97 m/s → ruled out pin/script/
  mechanism; isolated the single config diff `payload_max_kg`; ruled out the energy penalty
  via reward decomposition) **before** spending another GPU box. This is the right reflex.
- **Cheap probes before expensive RL.** `probe_speed_hold.py`, `motor_budget.py`,
  `power_compare.py` answered "is the lever even armed?" (ohmic ~4 % on G1) with *zero*
  training. The six-direction map was built largely from no-training measurements + research
  workflows.
- **Cross-hardware reproducibility check, once.** The Colab-T4-vs-H100 reward-curve overlap
  (Milestone 1) bought confidence in the whole pipeline cheaply and was not needlessly repeated.
- **Detached + self-correcting automation matured over the project.** `autonomous_load_run.sh`
  trains, *gates on `tracking_lin_vel`*, and auto-falls-back 10 kg → 6 kg without a human —
  the right response to the collapse pattern below.

### Bad (recurring failure modes)
- **The train → collapse → diagnose → re-train loop.** The dominant time sink. A run is
  launched, the policy collapses to a degenerate behaviour (stand-in-place, never-fall slow
  walk), and a *whole GPU session* is spent diagnosing and re-launching. Seen at least 3×:
  payload 0-25 kg → standing; G1 running `[0,3]` from scratch → 0.85 m/s never-fall walk
  (g1_running_design "Attempt 1 FAILED"); the earlier 2.0 m/s command destabilising the
  1 m/s walker. **Common cause: a too-wide command/DR range from scratch with a strong
  `termination` penalty → the policy plays it safe.**
- **Range chosen before realism checked.** The 0-25 kg payload range (~2× body mass) was
  picked, trained, and only *afterward* recognised as physically unwalkable (real Go1 max
  ~10 kg). Same shape as the 30-kg "walking" capacity result later flagged unphysical. The
  realism grounding (`load_program.md` "Capacity realism", 2026-06-16) arrived **after** the
  runs that needed it.
- **Eval scripts lag the experiment.** `go1_capacity.py` swept only to 15 kg while policies
  were trained/evaluated at 20-30 kg — the eval couldn't see the regime under test (flagged
  in `load_program.md` "Eval fix needed first"). Earlier: the env evals on *random* commands
  so `tracking_lin_vel` stayed high (367) while the policy actually *stood* under a constant
  forward command — a metric that didn't measure the thing of interest.
- **Tracking-reward vs forward-speed confusion.** `tracking_lin_vel` was twice mistaken for
  "is it walking forward": it's high for a policy that tracks *random* (including near-zero)
  commands while standing on a constant forward command. Forward speed at a fixed command is
  the honest metric; the eval scripts had to be retrofitted to report it.
- **Sync-after-the-fact loses work.** The payload-baseline *policy* was lost when a box was
  halted+auto-deleted before rsync (2026-06-15, "not synced in time"). Reproducible in ~14 min,
  but the lesson — **sync after EVERY run** — had to be learned by losing a run.
- **Plan churn / direction sprawl.** The strategy doc lists *six* directions; the active focus
  moved walking → hip-pitch → running → Go1 load → G1 running again within days. Much design
  work (running_program, g1_running_design, taxonomy, robot_inventory) is written ahead of any
  run that uses it. Useful, but it is speculative inventory that can age before it is exercised.

---

## Inefficiencies (with cost)

| # | Inefficiency | Concrete instance | Cost (rough) |
|---|---|---|---|
| 1 | **Wasted GPU run from a too-wide range** | Payload 0-25 kg DR → policy learned to STAND; the trained baseline + adaptive policies were unusable for the capacity claim | ~2 full training runs (~0.5 GPU-hr) + a session of diagnosis; ~350-700 ₽ |
| 2 | **Wasted GPU run, G1 running Attempt 1** | `[0,3]` m/s from scratch + walker reward → 0.85 m/s never-fall walk, no flight phase | ~1 training run (~1.5 GPU-hr) + redesign into a 3-stage curriculum |
| 3 | **Box-budget overrun / long idle box** | A box ran ~13 h; boxes are billed per-second and were repeatedly "left UP" by the autonomous scripts pending manual rsync+halt | at ~342 ₽/hr a 13-h box ≈ **4,400 ₽** vs the ~325 ₽ a single run costs — the single largest avoidable spend |
| 4 | **Lost run (no sync)** | Payload-baseline policy gone when box auto-deleted before rsync | ~14 min retrain + the risk it could have been a *non*-reproducible result |
| 5 | **Capacity-realism oversight discovered late** | Sim Go1 "walks" at 30 kg (3-6× real rated max); the 20-30 kg energy/capacity numbers are sim-only artefacts | the capacity-to-failure ladder (overnight_run.sh, runs C/D at 20/25/30 kg) is **scientifically void** as a Go1 result — GPU time spent measuring an unphysical ceiling |
| 6 | **Eval lagging the experiment** | `go1_capacity.py` capped at 15 kg while training at 20-30 kg; needed retrofit to report forward speed | re-runs of eval; risk of trusting a "high-tracking stand" as walking |
| 7 | **Speculative design inventory ahead of runs** | running_program / g1_running_design / taxonomy / robot_inventory written before the runs that consume them; several directions never executed | research-workflow tokens + doc time; some content ages before use (the 1.3M-token lit-review waste in MEMORY is the canonical prior instance) |

---

## Process corrections (checklist)

### Before launching ANY GPU box
- [ ] **Sanity-check the range against physical reality FIRST.** Payload ≤ rated max
      (Go1 ≤ ~10 kg); command range only as wide as the policy can plausibly reach. Write
      the real limit in the config comment. (Kills inefficiencies 1, 5.)
- [ ] **Never widen a command/DR range from scratch with a strong `termination`.** Warm-start
      from a competent policy and raise the top of the range by ≤0.5-1.0 m/s, OR use an
      explicit curriculum. From-scratch wide-range = the collapse trap. (Kills 2.)
- [ ] **Decide the honest success metric before training, and make sure the eval measures it.**
      For locomotion that is **forward speed at a fixed forward command**, not `tracking_lin_vel`
      (high for a random-command stand). Confirm `go1_capacity.py`/eval covers the *trained*
      regime, not a narrower default. (Kills 6, the tracking-vs-speed confusion.)
- [ ] **Confirm the lever is armed with a no-training probe** (ohmic share, braking energy) so
      you don't train to chase a ~4 %-of-budget effect.

### During the run (box discipline)
- [ ] **Gate every run on the success metric** (`tracking_lin_vel` floor, ~650-800) and
      auto-fall-back on collapse — the `autonomous_load_run.sh` pattern. Don't wait for a human
      to notice a standing policy.
- [ ] **rsync after EVERY run completes**, not at the end of the session. Treat the box as
      ephemeral and the local `outputs/` as the only durable store. (Kills 4.)
- [ ] **Cap box wall-time explicitly.** A box "left UP" pending manual halt is the biggest
      spend. Either halt+auto-delete on pipeline completion, or set a hard kill timer; a 13-h
      idle box costs ~10× a run. Budget a session before launching and stop at it. (Kills 3.)

### When a run collapses or a number looks too good
- [ ] **Diagnose before re-running.** Run the known-good policy through the *same* eval to
      isolate pipeline vs policy; decompose the reward; change ONE variable. (Already the norm —
      keep it.)
- [ ] **Ask "is this physically possible?" of any capability number.** A sim that walks at 30 kg
      is telling you the sim is missing a constraint (continuous/thermal torque, structure),
      not that the robot can. Label sim-only numbers as such *in the same breath*.

### Before writing more design docs
- [ ] **Don't build design inventory more than one gated step ahead.** Fail-fast staging
      (g1_running_design's "Stage and gate" is the model): design Stage 1, gate on it, design
      Stage 2 only if it clears. Avoid writing full multi-direction programs that may never run.
</content>
</invoke>
