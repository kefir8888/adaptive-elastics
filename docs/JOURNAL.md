# Project Journal — Parallel-Elastic Efficiency Study

Append a short dated entry at the **end of each work session** (newest at the top).
Keep entries terse — this is a memory aid for future sessions, not documentation.

For each entry note: what you did, what you decided (and why), key numbers if a run
happened, what's open/broken, and the single next step.

---

## 2026-06-19 (cont. 4) — Part 2 RESTARTED on the G1: continuous-hopping env subclass built + validated
- **Did:** User reopened the G1 dynamic-movement track (toward running), starting with JUMPING — specifically
  **CONTINUOUS, two-footed, IN-PLACE HOPPING for ENERGY** (jump → land → jump …). Built `src/pea/g1_hop_env.py`
  (`G1JoystickHop`), a minimal Playground G1-walk subclass: synchronous gait clock (phase `[0,0]`), air-time
  un-capped, walk-only anti-hop terms neutralized (feet_phase/stand_still/joint_deviation_knee→0, pose→−0.05),
  `max_contact_force` 500→2000, in-place zero command, + **five hop reward terms** (signed `hop_rhythm` backbone,
  dense `hop_push` bootstrap, `hop_height` [maximize|track], `hop_sync`, `hop_flight`). Configs: `g1_hop_s1`
  (elicit), `g1_hop_baseline` + `g1_hop_spring` (matched pair, params TBD). Design doc **`docs/hop_design.md`**.
- **Decided / why:** (a) Hopping is **REPETITIVE** → energy recovery + a controlled-task energy metric apply
  (unlike a one-shot jump), and **landing-braking** (the spring's recovery channel) dominates → the defensible
  **B2** case. (b) Metric = **energy per hop (J/hop)** at MATCHED apex+cadence, **NOT CoT** (no forward speed).
  (c) HEIGHT stays off-limits for a parallel spring (knee speed-limited, NR-7) — we target energy/peak-load.
  (d) Spring = knee/ankle **one_sided_linear** pogo, fit from a baseline work-loop (plot all leg joints).
  Reverses the old "hopping dropped" note (that was a Raibert 1-leg hopper; this is a full humanoid both-feet hop).
- **Validated (3 adversarial fresh-agent reviews + local probes).** Reviews: reward-hacking, MJX/API, validity.
  Found + fixed: **(1) standing-still trap / reward barrier** — naive reward let "just stand" win and the path to
  hopping ran downhill first → fixed with the SIGNED `hop_rhythm` (no free +0.5 floor) + the dense `hop_push`
  (rewards upward pelvis velocity while grounded). **(2) height-reference BUG** — used base_height_target=0.5,
  but the real standing pelvis z is **≈0.755** (`self._init_q[2]`); without the fix the robot scored height just
  for standing. **(3) two TRAINING-PIPELINE bugs that would have crashed on the GPU** (both also latent for
  `G1JoystickRun`): custom envs were registered lazily INSIDE make_env but `train.py` calls `ppo_params_for`
  first → moved registration to `pea.env` import; and `brax_ppo_config` has no entry for the custom names →
  aliased `G1JoystickHop`/`Run`→`G1JoystickFlatTerrain` in `policy.ppo_params_for`. Probes (CPU): reset/step
  finite + correct shaping; **full PPO `--smoke` pipeline trains/evals/checkpoints** (reward −4.0→−1.8 in 30k);
  DR randomizer resolves; spring+energy wrappers compose with the hop env (fixed cadence + track mode).
- **Open / broken:** S1 reward WEIGHTS are analysis-seeded, need GPU tuning (failure-mode knobs in the design
  doc); eval **flight-fraction metric** + **work-loop** tooling to build AFTER S1 (adapt `g1_run_probe.py`);
  spring params (k, theta_engage) TBD from the work-loop; **calibrate `hop_height_target` from the S1 rollout**;
  ≥3 seeds + regen sensitivity for the comparison. Nothing committed yet — GPU bootstrap clones `main`.
- **Next:** train `g1_hop_s1` on the GPU (immers box), **gate** on a real both-feet-airborne window + steady
  cadence + survival; then work-loop → matched baseline/spring energy comparison.

---

## 2026-06-19 (cont. 3) — per-joint spring-kind selection implemented (the deferred joint-fit fix)
- **Did:** Added `gravcomp.fit_spring_per_joint` (tries LINEAR + CONSTANT preload per joint, keeps the
  lower-energy one); `fit_linear_spring_per_joint` is now a thin linear-only wrapper. Wired `galaxea_lift.py`
  to it with a kind-aware table/plot legend (`_spring_str`). Regenerated the 02 plot (joint3 panel now a
  sensible flat preload line, no longer the "sus" tilted fit) and the 05 energy table; updated the program
  doc + CLAUDE.md numbers.
- **Result:** torso_joint3 now picks a CONSTANT preload (τ₀=−11 N·m) → **−84%** (was −65% linear); the
  3-spring lift headline rises **−95% → −96.5%** (whole-robot −20.9% @150 W). joint1/joint2 stay linear
  (−98% each). Middle-joint-only (06) and LimX (linear/constant already) unchanged.
- **Open:** the lift VIDEO (01) overlay still shows the old −95% (meter slightly stale; joint3 spring not
  shown in-frame) — re-render if a refreshed clip is wanted.
- **Next:** user's call — new chapter (humanoid training / Part 2 explosive moves) likely in a fresh session.

---

## 2026-06-19 (cont. 2) — gravity-comp direction WRAPPED (positive); bundle finalized; joint-fit finding
- **Did:** Closed the gravity-compensation direction. Wrote the versioned program doc
  **`docs/gravity_compensation.md`** (the durable home — the bundle REPORT.md is gitignored).
  Finalized the reporting bundle: moved all non-experiment / pre-mesh-fix renders into
  `_superseded/` (top level is experiment results only), re-rendered every live GIF to higher
  resolution (720 px / 600 px for the long phases clip; 256-color bayer + `diff_mode=rectangle`)
  and — after the first pass was choppy — back up to a smooth ~15 fps (20 fps turntable), trading
  size for watchability (~6–18 MB); `collect_report.sh` updated to reproduce both. Added
  `scripts/galaxea_knee_plot.py` → a single-panel **single-knee-spring** plot
  (`06b_galaxea_knee_single_spring.png`).
- **Decided / found:** (1) **The approach is validated** — adaptive parallel elastics cut
  electrical energy a lot on BOTH a high-gear humanoid (Galaxea) and a low-gear wheeled quadruped
  (W1), i.e. **gear-independently**, because they offload the constant-sign load (the inversion of
  the walking result where gearing set the sign). (2) **One well-placed spring ≈ the whole win**:
  knee alone −98%, mid-joint-only lift −85% vs −95% for 3 springs (the knee is **87%** of the load).
  (3) **Element kind is set by load SHAPE** (re-confirms the Go1-knee rule): torso_joint3's gravity
  is *constant in its own angle* (span 0.0, corr +0.63), so the linear fit pushed θ₀ to the grid
  boundary (−2.15) and under-fit (−65%; −75% with a wider grid) — a constant preload is the right
  element there. Left the per-joint kind-selection fix UNAPPLIED per user.
- **Result numbers:** Galaxea lift −95% (3 springs) / −85% (1 mid) / −98% (knee alone), whole-robot
  −20.5% @150 W; lean −98%; 5 free reaches −97…−99%. W1 roll −98% targeted / −26% whole; 4-phase
  adaptive preload tracks 4→26 N·m, knee power −93%.
- **Open / broken:** motor Kt/R + Galaxea mass are estimates (% robust, watts a band); `fit_linear_spring_per_joint`
  searches only linear springs (joint3-type under-claimed); galaxea_knee_plot not wired into `collect_report.sh`;
  the `_superseded/` 07/13/free-reach clips still show the pre-fix render bug (need source re-render, not just re-gif).
- **Next:** user's call — either start Part 2 (explosive moves) or resume the suspended dog-/G1-running tracks;
  if continuing gravity-comp, add per-joint spring-kind selection and a hardware Kt/R.

---

## 2026-06-19 (cont.) — gravity-comp experiment suite + presentation renders + two model-loading fixes
- **Did:** Galaxea — coordinated upright lift (3 springs), middle-joint-only lift (1 spring), forward LEAN
  about the BOTTOM joint (1 linear spring), forward-reach knee torque-angle, and **5 "free" reaching motions**
  (forward/down/up/side-twist/scan), each with a bottom-joint linear spring + energy compensation + video + plot
  (`scripts/galaxea_{lift,lean,reach,free}.py`). Dog — redid the 4-phase height roll with the CORRECTED knee
  direction + on-the-fly adaptive preload (`scripts/limx_phases.py`). Presentation — colored turntables + a
  per-part palette (`scripts/pretty_render.py`, `render_util.apply_palette`). Bundle `outputs/gravity_compensation/`
  (01–14 + `free_reaches/` + REPORT.md).
- **Corrected (3 things the user caught):** (1) **W1 knee direction was mislabeled** — EXTENDED legs = HIGH body
  = LOW knee torque; BENT = LOW = HIGH torque (straight legs easy, bent hard); over the full KFE range the knee
  load varies ~4→28 N*m, so the adaptive preload genuinely tracks it. (2) **W1 mass:** the earlier "0.818 kg base,
  added 17 kg torso" was a buggy read (`mj_name2id`→-1, then `body_mass[-1]` wraparound); the URDF `base` is
  already **18.19 kg** (legs 25.3 → **43.5 kg total**), so the 17 kg override is ~a no-op and ~42 kg is the
  vendor's own mass (legs are a heavy 58%). (3) **Galaxea "two bases at the shoulders"** = mesh-name collision
  (A1-arm and R1-base packages both ship `base_link.obj`; MuJoCo keys meshes by stem). Fixed in
  `urdf_loader.resolve_urdf` (package-qualified mesh symlinks); render-only, no result changes.
- **Result / numbers:** Galaxea lift 3-spring **−95%**, 1-spring (middle) **−85%**, lean **−98%**; 5 reaches
  **−97…−99%** (load scales with reach: down heaviest, up lightest); whole-robot −18…−21% @150 W. Dog 4-phase:
  CoM 0.53→0.30 m, knee power rises as it squats, adaptive preload tracks **4→26 N*m**, knee power **−93%**.
- **Open / broken:** motor Kt/R + masses still estimates; the ~25 kg leg mass (58% of the W1) is heavy (URDF
  inertials); Galaxea `.obj` meshes are lower-detail than the repainted GLB (even fixed/colored, the sim look ≠
  the GitHub render). Lots was uncommitted → committing now.
- **Next (user's call):** finalize the dog mass; optionally re-render dog/reach clips in color too; Part-2
  (explosive moves) still untouched.

---

## 2026-06-19 — NEW direction: gravity-compensation springs on mobile manipulators (Galaxea lift + LimX dog) — BOTH positive
- **Did:** Opened a Part-1 spinoff (user-directed): parallel springs on joints with CONSTANT-SIGN gravity load
  (body up/down, loading/unloading) — cleaner/faster than locomotion. Got two external URDF robots running
  locally in MuJoCo: fiveages-sim/robot_descriptions (Galaxea R1, Airbot MMK2, Agibot G1 wheeled; Unitree G1 /
  Booster T1 / RobotEra xbot legged) + LimX W1 wheeled quadruped (WL_P311D). URDF→MuJoCo: resolve `package://`,
  GLB→OBJ (trimesh), `discardvisual=false` + `fusestatic=false`. New modules
  `src/pea/{urdf_loader,gravcomp,render_util}.py`; scripts `galaxea_lift.py`/`limx_roll.py`/`gravcomp_table.py`/
  `setup_robots.sh`/`collect_report.sh`; configs `galaxea_lift.yaml`/`limx_roll.yaml`;
  `energy.MOTORS += limx_knee, galaxea_torso, galaxea_arm`. Robots → gitignored `external/`. Reorganized
  `outputs/`: NEW `outputs/gravity_compensation/` (videos+plots+table+README+raw/); everything locomotion → `outputs/_locomotion_archive/`.
- **Decided:** gravity comp is the new active direction; **dog-running AND G1-running locomotion tracks are
  SUSPENDED**. No regen; onboard computer 150 W (swept 0/50/150/300); full up-down per 2.5 s. Galaxea =
  coordinated UPRIGHT lift (3 torso joints, IK, body stays level). Dog = constant knee preload during wheeled
  roll; wheel transport billed analytically (Crr=0.015·m·g·d), NOT the slip-inflated velocity servo. W1 torso
  mass 17 kg (no official spec; ~41 kg total).
- **Result / numbers:** Galaxea R1 coordinated upright lift (0.33 m, 2.5 s×8, 20 s; upper body ~36 kg; peak
  119 N·m): torso-motor electrical energy **956→48 J (−95%)**, whole-robot **−21% @150 W** (−94% servos-only).
  LimX W1 roll (42.3 kg, 0.72 m/s, 18 s): knee-motor **961→19 J (−98%)**, all-servo **−72%**, whole-robot
  **−26% @150 W**; CoT (all-servo) 0.280→0.077. Win large on BOTH despite opposite gearing (galaxea_torso
  R/Kt²≈2e-4 high-gear; limx_knee ≈0.027 low-gear) — offloads the constant-sign load (lift work / stance-holding
  ohmic), gear-independent. Table `outputs/gravity_compensation/05_energy_savings_table.md`.
- **Open / broken:** motor Kt/R + masses are estimates (no vendor specs); dog τ-θ has a startup settling
  transient; Galaxea lift single-amplitude, no carried payload yet; prescribed-motion (post-hoc≈in-loop here).
  Bugs found+fixed: `mj_inverse` returns spurious base-joint torque (640 vs 15 N·m) → explicit M·q̈+qfrc_bias;
  URDF IK singular at full extension → bent seed; `fusestatic` fuses the IK target body → false;
  `discardvisual=true` drops skybox/floor textures → false; velocity-servo wheel slip → bill Crr·m·g·d analytically.
- **Next:** add payload-scaling (carried mass → saving grows with load) + a regen on/off row to the combined
  table, for both robots.

---

## 2026-06-18 — Forward-command curriculum: the dog RUNS (to ~2.2 m/s); flight stage over-corrected
- **Did:** Diagnosed the standing collapse — the Go1 (and G1) joystick samples a SYMMETRIC, often-zeroed
  velocity command (mean-zero vx), so the policy was never required to move. Built a forward-command override
  (`command_forward` in config.py + `make_forward_sampler`/`bind_forward_command` in env.py: draw vx from a
  one-sided forward band, never zero, force the initial command forward too). Added curriculum configs
  C1/C2/C3 + matched spring configs; switched `scripts/run_dog_pipeline.sh` to the curriculum. Ran it detached
  on a fresh H100 NVL (195.209.208.208), gated stage-by-stage, halted the box. Also this session: built the
  G1 flight-enabling env subclass `G1JoystickRun` (`src/pea/g1_run_env.py`); assembled presentation materials
  in `outputs/presentation/` (work-loops, the G1 "impossible spring" figure, energy/CoT tables, 8 videos);
  reconciled docs; code cleanup (audit v2 + fixes C1–C6, dedup); checkpoint backup to Drive; iron-loss bound.
- **Decided:** the forward-command curriculum (force vx forward + ramp the speed) is the fix for the standing
  collapse; the C3 fast+flight stage was too aggressive and needs softening before flight can emerge.
- **Result / numbers:** S0 walker reward 28.3 → **C1 ran 1.47 m/s** (cmd 1.5) → **C2 ran 2.16 m/s** (cmd 2.2) —
  GENUINE forward running, solving the prior total standing collapse. **C3** ([2.5,3.2] m/s + flight tweaks)
  **REGRESSED to standing at every speed** (0.00 m/s, true_flight False even at 1.5) → no flight → spring stages
  not run. Run folders `2026-06-18_go1_run_{s0_walker,c1,c2,c3}_*` (pulled to `outputs/box_runs_curriculum/`).
  G1 "impossible spring": G1 knee energy-optimal element is **anti-restoring** (k ≈ −13 to −19 N·m/rad, R²≈0.05,
  offset-dominated → impossible as a passive spring); G1 hip k ≈ +66 (buildable, but backfires in-loop).
- **Open / broken:** no flight phase yet (C3 over-corrected — the speed jump + flight reward made standing a
  better optimum); spring-on-running test still pending; G1 running subclass built but untrained; Go1 rough
  still ~40 % survival.
- **Next:** gentler C3 — smaller speed step (e.g. [2.2,2.8]) + milder/STAGED flight tweaks (don't relax
  feet_clearance so hard; add the flight reward only at a speed the policy already runs); retrain C3 from C2;
  re-gate on flight.

---

## 2026-06-17 — Dog-running pipeline ran on an H100; S2 STANDS (no running) → command-distribution fix needed
- **Current state:** ran the full staged Go1 dog-running pipeline on a rented immers H100, fully detached
  (S0 walker → S1 trot → S2 run → flight gate). Rewards: S0 28.2, S1 26.4, S2 22.6; S0–S2 trained in ~40 min
  total (the 12-DoF Go1 on an H100 is ~3–4× faster than the G1-based estimate). Checkpoints pulled to
  `outputs/runs/` (s0, s1) and `outputs/box_runs/` (s0–s2 + probe logs). Box powered off / deleted.
- **Found + fixed a probe bug** (`711d578`): `go1_run_probe` set `state.info["command"]` but not
  `steps_until_next_cmd`, so the Go1 env resampled the command (sometimes to zero) and the policy stood —
  a false "no flight". Pinned the resample counter.
- **The finding (real, not a probe artifact):** after the fix, a direct diagnostic confirmed the command
  reaches the policy (`info.command=[3,0,0]` held every step) but the base velocity decays to ~0 — the S2
  policy genuinely **stands** under a sustained high forward command. Its high training reward
  (`tracking_lin_vel` 541) reflects tracking the mostly-**low** command distribution (the Go1 sampler has a
  high zero-command probability), not running. **Raising `command_config.a` amplitude alone does NOT elicit
  running**; no flight emerged; the spring stages were correctly not run (pre-registered gate honoured).
- **Decided:** stop at the flight gate (no running gait → nothing to test the spring on). Cost ~1.5
  H100-hours (~$2–3; well under the ~$18–20 budget, because we stopped at the gate).
- **Open:** the runner needs a command-distribution **curriculum** to force forward running, not just a higher amplitude.
- **Next:** add an env override / small subclass that reduces the zero-command probability and/or narrows the
  command to a high-forward band (e.g. vx∈[2.5,3.5], trimmed lateral/yaw); retrain S2 from S1; re-gate on
  flight; then S3 work-loop + S4 spring as planned.

---

## 2026-06-16 (session 2) — Code/checkpoint audit, iron-loss bound, doc reconciliation, dog-run prep
- **Current state:** Go1 load-carrying WALKING program **DONE + positive** (CoT −14 to −27 % in 3 of 4
  conditions, growing with load; seed 2 a weak −3 to −8 % outlier; survival degrades >~7.5 kg). G1 walking
  spring **DONE + negative** (in-loop +7.4 %). **ACTIVE = the Go1 dog-running knee-spring experiment**
  (`docs/NEXT_SESSION.md`, `docs/dog_running_design.md`). GPU box OFF.
- **Checkpoint backup:** all training runs mirrored to the Drive `pea_runs` folder (134 MB) and catalogued
  in the new **`docs/checkpoints.md`** (every run + purpose + Drive/local location + has-checkpoint flag).
- **Full code audit** confirmed the energy model is correct; the adaptive controller is now in `src/pea/control.py`.
- **Iron (core) loss quantified** (was only qualitative). Upper-bound estimate: G1 whole-body ~25–35 W likely
  (14–20 % of the ~178 W budget), worst case ~58–65 W (33–37 %); Go1 negligible (~3–5 W, <5 %). Iron loss grows
  with motor *speed*, so a torque-offloading spring can't cut it; including it nudges G1 walking offline −2.9 %
  → −2.1 to −2.4 % and Go1 −14 to −27 % → ~−13.3 to −26.8 % (essentially unchanged). **All headline conclusions
  survive a full iron-loss accounting.** Estimate only — one no-load spin-down test would pin it. (`RESULTS.md`.)
- **Doc reconciliation:** the docs had drifted ~1.5 sessions and contradicted each other. Reconciled CLAUDE.md,
  README, RESULTS.md (Milestone 4 marked DONE/negative; Go1 load results subsection + per-seed CoT table; iron-loss
  caveat; the R/Kt² "10× too optimistic" alarm **retracted** as a 100× arithmetic error — 0.0025 is the optimistic
  edge of a defensible 0.001–0.020 band), PLAN.md (superseded banner), load_program.md, directions.md, gpu_cost_crypto.md
  (superseded banner), JOURNAL. Fixed gear statements (G1 knee 22.5:1, **hip-pitch 14.3:1**, dispute unresolved but
  ohmic stays ~4 %; Berkeley **9:1** not 9.1; Spot hip **~51:1** not ~80–160:1).
- **New project rule + glossary:** added "write plainly — define every term on first use, no unexplained jargon"
  (docs, code comments, manuscript) and created **`docs/glossary.md`** (plain-language definitions), linked from
  CLAUDE.md and README.
- **Infrastructure decided:** stay on **immers.cloud, rubles**; crypto/Vast/RunPod/Spheron dropped (KYC-blocked from
  Russia; revisit only for a large final seed batch). User provisions + pastes SSH; agent drives and destroys the
  box; sync after every run.
- **NEXT (single step):** the user provisions an immers box; the agent runs the staged dog-running experiment
  (S0 walker → S1 trot → S2 run+flight → spring conditions), **gating on the measured flight fraction** before
  spending on the spring arms.

## 2026-06-16 (wrap) — Audit/research/paper session + refactor + dog-run plan + merge to main
- **Autonomous audit/research/writing session** (3 workflows, ~20 subagents, ~1.2 M subagent tokens, ~40 min
  compute): wrote `docs/{code_audit,docs_audit,weak_spots,research_patterns,literature_review,
  g1_running_research,gpu_cost_crypto}.md`, `docs/presentation.md` (+`outputs/slides/presentation.pptx`),
  `paper/paper.tex` (IEEE Access draft), `outputs/figures/cot_vs_load.png`; reconciled the headline to CoT.
- **Refactor:** extracted the adaptive controller to `src/pea/control.py` (was verbatim in `go1_capacity.py`
  + `render_walk.py` — the audit's video-vs-curve divergence risk). Verified behavior-preserving.
- **README:** added a *Methodology & process discipline* section (the 7 pattern fixes); refreshed the docs map.
- **Dog-running plan agreed:** flat, **no-load run first** (S1–S5, gate on a measured flight fraction), then a
  **load-carrying running** extension at **0 / 2.5 / 5 kg** (`configs/go1_run_s1.yaml`, `go1_run_s2.yaml`
  prepared + smoke-tested). NEEDS A BOX to run; warm-start checkpoint was lost with the deleted box.
- **G1 running: deprioritized** (long-shot; needs an env subclass + ideally reference-motion/AMP — see below).
- **Merged `experiments-2026-06-14` → `main`.** Infra decision (2026-06-16): stay on **immers.cloud, paid in
  rubles**; crypto-funded Western providers (Vast/RunPod/Spheron) **dropped** (KYC/sanctions-blocked from Russia,
  not worth it for a ~$18–20 experiment).
- **NEXT:** provision a box on the new provider; run the dog-running experiment — **recommended in a FRESH agent**
  (the docs now support a lossless handoff: README → JOURNAL → dog_running_design.md).

## 2026-06-16 — Load study RESULTS (reconciled) + autonomous audit/research session
**The Go1 load-carrying program is DONE and POSITIVE (this is the headline that was missing from the docs).**
Cost-of-transport reduction (electrical W per m/s), adaptive per-leg knee preload vs matched no-spring, flat:
| | @0 kg | @2.5 kg | @5 kg |
|--|--|--|--|
| seed 1 | −16.6% | −19.5% | −22.8% |
| seed 2 | −3.4% | −8.3% | −6.5% (weak outlier) |
| seed 3 | −13.9% | −20.1% | −26.7% |
| curriculum | −16.8% | −20.4% | −22.0% |
- **Reconciled headline: −14 to −27% CoT in 3 of 4 conditions, GROWING with load; seed 2 a weak −3 to −8%
  outlier.** The earlier docs quoted 3 different bands (−16.7/−19.7, −17/−20, −14/−27) — now reconciled from
  the local capacity logs. Figure: `outputs/figures/cot_vs_load.png`.
- **STABILITY CAVEAT (was unreported):** the adaptive loses survival at high load (e.g. curr15 1070/1500 @10 kg;
  seed 2 871/1500 @5 kg) where the matched baseline holds 1500/1500. The energy win is low-to-mid load.
- **CAPACITY REALISM (validity catch):** the warm-started sim Go1 "walks" at 30 kg, but the real Go1 carries
  ~5–10 kg max — plain sim enforces peak, not continuous/thermal torque or balance. 15–30 kg is sim-only; the
  defensible range is 0–6 kg. Beyond-30 kg experiments planned but NOT run (thermal-limit / B2-class).
- **Rough terrain:** energy win survives on 2.5 cm (CoT −10 to −19%) but ~40% survival for BOTH arms (hard task);
  full 5 cm inconclusive, dropped.
- **G1 RUNNING — two failed attempts** (a [0,3] from-scratch collapse to 0.85 m/s; a curriculum+reward-redesign
  that destabilized it). Playground G1 env structurally resists flight. G1 running = long-shot. Pivoted the
  knee-spring-for-running idea to the **DOG** (committed; design in `docs/dog_running_design.md`).
- **Autonomous audit/research session:** workflows wrote `docs/{code_audit,docs_audit,weak_spots,research_patterns}.md`
  + (running) `{g1_running_research,literature_review,gpu_cost_crypto}.md`; built `docs/presentation.md`; rewrote
  README to current state; drafting the IEEE Access paper in `paper/`. Code audit: energy model CORRECT, but
  rollout loop duplicated 5–6× + the adaptive controller 2× (refactor needed); 1 latent aliasing bug.
- **State:** GPU box OFF, all results synced (26 run dirs, 10 eval logs, 4 videos local). **Next:** provision a
  box, run the dog-running knee-spring experiment (gate on flight fraction), and refactor the duplicated rollout.

## 2026-06-15 (session 2) — Adaptive per-leg mechanism VALIDATED, but payload-DR broke locomotion
- **The adaptive per-leg preload mechanism WORKS (validated):** `PreloadDRWrapper` (env.py)
  trains fine under vmap/jit (reward parity 18.2 vs baseline 18); the clipped-proportional
  controller ramps per-leg τ₀ sensibly with load — `conv_tau0 [4.8,6.5,4.1,2.6]`@0kg →
  `[8.6,8.5,5.4,6.1]`@25kg (front>rear, the asymmetry), and where applied it offloads energy
  (323→208 W at 25 kg). So the SR-integral controller + per-leg preload DR are sound.
- **BUT the capacity result is INVALID:** both payload-DR policies walk at **~0 m/s** (they
  STAND, don't go forward) at *every* commanded speed (0.3→1.0). The **0–25 kg range (up to 2×
  body mass) taught the policy to stand** (safe under heavy load) instead of committing to
  forward walking. DIAGNOSED cleanly: the ORIGINAL Go1 walker runs **0.97 m/s through the SAME
  script** → pin/script/mechanism all fine; only the payload-DR policy is broken. (tracking_lin_vel
  reward is high, 367.5, because the env evals on *random* commands; a *constant forward* command → stand.)
- **ROOT CAUSE CONFIRMED + energy penalty RULED OUT:** the reward decomposition shows
  `energy_reward_weight` is identical (−1e-4) in the walking and the standing policy → not the
  energy penalty. The only config diff is `payload_max_kg` (0 vs 25), and `tracking_lin_vel`
  collapses **925 → 367** (std 296, bimodal: half-tracks, half-stands). The unwalkable >12 kg tail
  (above body mass) is what taught it to stand. Note the "capacity ceiling" is *stops walking*, not *falls*.
- **FIX PREPARED (one-command retrain ready):** configs now set to **payload 0–10 kg** (< body mass,
  realistic Go1 box max; the prior 0–25 kg tail removed): `go1_baseline_payload.yaml`,
  `spring_go1_adaptive.yaml` (tau0 15→12, matched to 0–10 kg loads), + the `_s2` variants.
  `go1_capacity.py` sweep → [0, 2.5, 5, 7.5, 10, 12.5, 15] (in-dist + slight OOD).
- **Safe (local + committed):** `PreloadDRWrapper`, the `*_payload*`/`*adaptive*` configs,
  `scripts/go1_capacity.py` (unified eval w/ speed). Baseline + adaptive policies in `outputs/runs/`.
  Box halted → auto-deleted.
- **RESUME (next session, 1 GPU box):** (1) retrain `go1_baseline_payload` + `spring_go1_adaptive` at
  0–10 kg, **gate on `eval/episode_reward/tracking_lin_vel` >~800** (collapse toward ~370 ⇒ drop to 6 kg
  / add a payload curriculum); (2) re-run `go1_capacity.py` — must show speed ~0.8–1.0 m/s now; (3) headline
  = energy-vs-load curve (baseline vs adaptive preload) + the per-leg τ₀-vs-load curve (already validated);
  (4) Phase B: 2nd seed + rough terrain. Sync every run; halt → auto-delete.

## 2026-06-15 — Box deleted (budget stopped); RESUME PLAN for the load study
- **State:** H100 halted+auto-deleted. Payload-baseline POLICY lost (not synced in time) but
  reproducible from `configs/go1_baseline_payload.yaml` (~14 min). All code/configs/docs/results
  (Go1 −16.7%/−19.7% positive, videos, plots) are LOCAL and safe. **Lesson: sync after EVERY run.**
- **Design settled (see `docs/load_program.md`):** per-leg (4×) preloads; adaptive controller =
  per-knee clipped-proportional integral, `τ̇₀=clip(0.2·ē, ±2 N·m/s)`, ē=15s-EMA motor knee
  torque, ē_target≈0 (full comp); train robust to preload via DR, run the controller at eval;
  almost-constant coil (low-k pre-wound) is the buildable element.
- **RESUME CHECKLIST:**
  - [ ] Fresh box + bootstrap (~10m); push code (~2m); re-train payload baseline (~14m)
  - [ ] Build adaptive controller — per-leg, OFFLINE/no-box (~30–45m)
  - [ ] Wire per-leg preload DR (τ₀ 4-vector) (~15m)
  - [ ] Baseline capacity sweep 0→25kg (~10m); train adaptive spring (~20–25m); spring capacity
        + energy-vs-load curves (~15–20m); headline curve + box-carry video + doc (~25m)
  - [ ] Phase B: rough terrain + payload (~45m); 2nd seed (~40m)
  - [ ] Sync continuously; halt → auto-delete

## 2026-06-14 (night, cont.) — Quadruped LOAD program designed + payload baseline launched
- **Direction:** Go1 carrying variable payloads + a **self-tuning knee preload**. Full design
  + numbers in **`docs/load_program.md`**.
- **Adaptation mechanism (user's):** measure mean knee torque over ~15 s (no load sensor;
  controller BLIND to mass) → integral-ramp the passive preload τ₀ at **≤1–2 N·m/s** to
  offload the support component (slow outer loop, time-scale separated from the 50 Hz policy).
- **Expected knee torques:** mean calf 4.6 N·m (no load) → ~14 (25 kg); preload ~3.5 → ~11,
  all under the 45 N·m calf limit. Dynamic peaks approach 45 at heavy load → baseline fails,
  preload rescues → **CAPABILITY** (extend carry-capacity), not just energy.
- **Triviality (user's worry):** the bare "offload the loaded joint" is simple; the
  contribution is (a) the boundary (it REVERSES on the high-gear G1), (b) element kind
  (constant not linear), (c) the adaptive self-tuning preload, (d) the capability claim.
- **Did:** wired payload DR (`src/pea/payload.py`, `cfg.payload_max_kg`, `train.py`); launched
  flat+payload baseline (0–25 kg, blind, 300 M). Box up. Robots: Go1 now, others later.
- **Next:** build the adaptive-preload spring run → payload-capacity + energy-vs-load curves.

## 2026-06-14 (night) — Go1 quadruped: parallel elasticity PAYS (the one positive result)
- **Result:** Go1 (low gear 6.33:1), constant preload at all 4 knees (calf, τ₀=3.5 N·m),
  in-loop **153.4 → 127.8 W = −16.7% whole-body electrical, 4/4 stochastic survival** (no
  stability cost). Post-hoc was −14.9%; in-loop HELD/IMPROVED it (vs the G1's reversal).
- **Why it works:** ohmic is **54%** of the Go1 budget (vs G1's 4%) → τ² lever armed; the
  calf is offset-dominated (linear null; Belov/Osokin τ²-fit gives k=−12.8 anti-restoring →
  not passive) so a CONSTANT preload is the buildable optimum; a constant offload is
  gait-compatible (no swing-phase fighting) so in-loop *beats* post-hoc.
- **Robustness:** CONFIRMED across 2 seeds — seed-1 −16.7% (153.4→127.8 W), seed-2 −19.7%
  (154.4→123.9 W), both 4/4 stochastic survival. Not a fluke.
- **Did:** wired Go1 (`Go1JoystickFlatTerrain`, `go1_knee` Kt=0.64/R=0.12, calf preload,
  parameterized the energy wrapper via `cfg.energy_motor`), gate + work-loop plot
  (`outputs/plots/go1_calf_work_loops.png`) + 4 videos (`outputs/videos/`). Box up.
- **Next direction (user):** quadruped + **LOAD** program — tunable knee preload scaling with
  payload, a single load-robust controller; dogs = Go1/Go2/Barkour/Spot(high-gear control)/
  ANYmal/big-Unitree (B1/B2). Est. ~1.5 GPU-hr + setup per robot.

## 2026-06-14 (late) — G1 in-loop spring: NEGATIVE, and catalogued
- **Result:** the in-loop hip-pitch spring (matched retrain: same 80M init, +120M, same
  −5e-4 weight, differ only by the spring) makes G1 walking **+7.4% WORSE** (151.6→162.8 W,
  CoT +8.4%, survival 3/4 vs 4/4) — **reversing** the post-hoc **−3.84%**. Robust: a
  fresh-from-scratch spring is no better; a 2×2 cross-condition confirms retraining
  adapts (spring policy beats spring-blind one), so it's a REAL effect, not a bug. The
  spring absorbs hip-pitch braking (9.8→4.0 W) but the motor fights it in the drive phase.
- **Catalogued all negatives → `docs/negative_results.md`** (NR-1 knee linear degenerate;
  NR-2 hip in-loop reversal; NR-3 post-hoc WRONG-SIGN; NR-4 ohmic ~4% gear-killed; NR-5
  win is no-regen-only; NR-6 energy weight not a lever; NR-7 jump height off-G1; NR-8
  clutch can't gate running; NR-9 mechanism not novel).
- **Decided:** parallel elastics don't pay for G1 walking. A per-stride clutch could help
  WALKING (~1 Hz feasible) but NOT running (stance ~100 ms, no clutch that fast) → running
  wants SERIES, not parallel+clutch.
- **Did:** wired the Go1 quadruped track (low gear 6.33:1, go1_knee R/Kt²~0.29 ≈120×G1):
  `go1_knee` constants, parameterized the energy wrapper (`cfg.energy_motor`),
  `configs/go1_baseline.yaml`. Go1 baseline training now.
- **Next step:** when Go1 baseline lands → roll out → fit calf (knee) work-loop spring →
  in-loop springs in all 4 knees → compare. Box still up.

## 2026-06-14 (eve) — Running-efficiency program launched
- **Decided (running program, `docs/running_program.md`):** next task = G1 **running
  for EFFICIENCY** (not speed — knee is speed-limited; the spring's braking-recovery
  lever is bigger in running than walking). **No-regeneration = JUSTIFIED** (no modern
  humanoid driver regenerates; verification workflow running). **Hopping DROPPED**
  (full humanoid, not a Raibert hopper). **Go2 quadruped ADDED** as a parallel track.
  Step-2 per-joint post-hoc = hard **GO/NO-GO gate** (must beat walking's ~3 %).
  Adaptive co-design = **later**; speed/load ranges = **end**. Spring targets = the
  **pitch trio** (hip/knee/ankle), plot all leg joints; knee back in play for
  efficiency (its speed limit only excludes height).
- **Did:** Inspected the Playground G1 reward — **the sawtooth is a disabled penalty**:
  `action_rate`/`dof_acc`/`torques` = 0.0 by default, so enabling `action_rate`
  (negative) fixes the ~55 %-reversal chatter, no code change. Command `lin_vel_x`
  defaults to **[-1,1] m/s** (why 2.0 fell). Added `env_overrides` to RunConfig +
  make_env (override top-level env keys, e.g. the speed range); scaffolded
  `configs/run_baseline.yaml` (wider speed range + smoothness on + energy weight;
  reward weights TODO from SOTA) — loads and builds. Wrote `docs/running_program.md`
  (full pipeline, milestones, validity guards, GPU checklist, Go2 track).
- **Workflows landed, all folded in:** (a) running-RL SOTA `wf_c9783c03` → full
  reward in `configs/run_baseline.yaml` (smoothness trio action_rate -0.01 / dof_acc
  / torques; feet_air_time 2→4 + lin_vel_z; ElectricalRewardWrapper; ≥3 seeds; warm
  from walker; optional rough finetune; velocity-gated flight bonus only if hopping).
  (b) regen+Go2 `wf_924da954` → **no-regen JUSTIFIED for the G1 by back-EMF physics**
  (back-EMF < 48 V bus at locomotion speeds; regen-resistor pattern), NOT a verified
  spec; exceptions MIT Cheetah 2013 + reportedly Tesla Optimus; ~24 % sensitivity.
  Go2 plan filled (6.22:1, Kt 0.26, R unmeasured=blocker, ohmic 39–76 %; differentiate
  vs Bjelonic/PIL by battery-electrical + zero-shot conditioned policy).
- **Infra:** `env_overrides` now also reaches nested reward_config keys (dotted);
  multi-joint `pea-sweep` (pitch trio) + per-joint braking + `metrics.fit_linear_spring`
  (recovers the hip k=68/θ0=-0.29 optimum). All CPU-validated; updated CLAUDE.md,
  RESULTS.md, PLAN.md, running_program.md.
- **Next (GPU):** calibrate the energy weight; train the baseline runner (run_baseline)
  ≥3 seeds; rollout + `pea-sweep` per-joint gate; if it clears walking's ~3 %, the
  fixed-spring in-loop runner.
- **Also: cross-morphology taxonomy** (`docs/taxonomy.md`, wf_dee7acba). Key insight —
  the spring's dominant benefit **shifts with gear ratio**: LOW-gear QDD (Berkeley 9:1,
  Go1/Go2/Barkour 6:1) → ENERGY (8–17 % CoT); MID (G1 22.5:1) → marginal; HIGH-gear
  harmonic/SEA (Apollo, Spot, ANYmal) → WEAR/peak-load (ohmic≈0). Quadrupeds are NOT
  all the same (QDD class homogeneous; ANYmal/Spot high-gear differ); humanoids split
  3 ways. Playground ships ready envs for G1/H1/Berkeley/Apollo/OP3/**Booster T1**
  (∥ ankle confirmed) + Go1/Spot/Barkour → a comprehensive study is **~£160–210**
  (~55–65 runs; Tier-0 post-hoc screening ~£35–50 gives the picture). Placement:
  biped→hip-pitch, quadruped→knee, biped-knee=constant element. Added a wear metric
  (peak+RMS torque) to `metrics`. Cassie already sprung; DecART has no env (model work).

## 2026-06-14 — Motor budget, actuation share, six-direction strategy map
- **Did:** Measured the baseline motor budget (`scripts/motor_budget.py`,
  `power_compare.py`): whole-body motor electrical ≈178 W, ohmic ~4 %, **hip-pitch
  = 27.3 %** of motor electrical (knees 36.7 %). Confirmed the post-hoc hip-spring
  saving exactly: hip-pitch 48.7→43.5 W, whole-body 178.5→173.3 W (−5.2 W, −2.9 %),
  and **~0 W under regeneration** (the win is braking-energy recovery the no-regen
  clamp would otherwise dump; +43 W = +32 % no-regen tax). Researched the
  actuation share of total robot power (G1 421 Wh, ~210 W mixed-use; house load
  ~45 W → **actuation ~80–90 % walking**). Ran 3 research workflows (gear ratios,
  prior art, energetics) and two no-training probes (`probe_speed_hold.py`).
  Wrote **`docs/directions.md`** (six directions + decisions + gear table) and
  fixed RESULTS.md's stale knee-constant Milestone-4 section.
- **Probes:** stand/hold → walking-tuned spring saves ~0 W; faster walking
  (1.23 m/s) → ohmic share stays ~3.8 %, braking 46→63 W, spring gain 5→6 W;
  command 2.0 m/s destabilises the 1 m/s walker (running needs its own policy).
- **Decided (directions):** TRY **(1)** in-loop G1 gate first, **(2)** running
  G1/H1 [appeal = larger braking energy + bouncing gaits; needs a purpose-trained
  running policy. NB the passive dead-zone clutch does NOT solve the within-stride
  swing-fight — it is a between-conditions on/off, so its value lives in the
  adaptive sweep / Direction 5, not running], **(5)** quadrupeds across slopes/loads via a **single zero-shot
  spring-conditioned policy, NO per-condition RL retrain** (differentiator vs
  Bjelonic 2023), **(6)** DecART/parallel-kinematics leg-length spring
  (experimental). **(3)** low-gear humanoid only as a research-platform comparison
  (Berkeley Humanoid is not a product, not thermally efficient) — lower priority.
  **SKIP (4)** manipulation/static (nothing surprising; probe confirms ~0 on G1).
- **Key facts:** G1 22.5:1 is near the LOW end of *commercial* humanoids (most are
  harmonic 100:1+); low-gear = research/QDD (Berkeley 9:1 [9.1:1 is HECTOR], MIT Cheetah/Go ~6:1).
  Corrections: ANYmal is 100:1 (not low-gear); "−31 % joint electrical" is STEPPR,
  not Bjelonic (Bjelonic = +33 % torque-square, −30 % peak, +11 % runtime).
- **Open / broken:** running upside is theoretical until a running policy exists;
  DecART gear ratio unpublished (decides LOW vs MED). No-regen still unjustified
  from G1 driver specs.
- **Next:** run the Milestone 4 in-loop gate on a GPU box (unchanged); then train a
  G1/H1 running policy for Direction 2.
- **Also (later):** generalized the one-off scripts into a reusable harness —
  `pea-sweep` (`src/pea/experiment.py` + `metrics.py`, entry point in pyproject):
  sweeps **robot × task × spring** and tabulates energy AND performance metrics.
  Validated on the baseline (walk no-spring 181 W; k=68 −5.7 %/−3.1 %; k=68 > k=100;
  stand ~0). Motor-constant registry `energy.MOTORS` (g1; go2 placeholder R).
  Reframe recorded (`directions.md` "Two value axes"): adaptive elastics also
  **amplify peak power → jump higher / run faster**, a performance axis that is
  NOT obviously gear-limited the way the ~4 % ohmic lever is — possibly a more
  compelling demo on the G1 than CoT. Corrected the dead-zone clutch framing
  (between-conditions on/off, NOT a within-stride running clutch; its value is
  Direction 5 + per-task engagement). CLAUDE.md goal still says "efficiency only" —
  worth broadening to include the performance axis.
- **Also (later 2): Part 2 reframed + bifurcation + motor envelope.** Recorded the
  **bifurcation** the owner named: Track A = efficiency of REPETITIVE motion (Part 1)
  vs Track B = EXPLOSIVE moves (Part 2), which may be one-shot (a max jump — CoT
  doesn't apply) or cyclic (hopping). Part 2 split into **(B1)** jump HEIGHT/speed —
  series-favoring (SEA 1.4–4×, parallel ~1.3×), gear-discounted, and **unresolved**
  (torque- vs speed-limited at takeoff; stock G1 long jump hits ~139 N·m knee →
  tilts torque-limited) — and **(B2)** efficiency/peak-load of jumps+landings, the
  **defensible parallel case on the G1** (landing load is impact-velocity-driven, so
  the gear speed-cap doesn't apply; jump torques dwarf walking). CLAUDE.md Part 2 +
  `directions.md` updated. **Motor torque–speed envelope added**, then CORRECTED:
  the MuJoCo G1 DOES enforce joint TORQUE limits (`jnt_actfrcrange`, jnt_actfrclimited
  True: knee ±139, hip ±88, ankle ±50 — I'd wrongly checked `actuator_forcerange`
  [0,0]); only the velocity rolloff is missing. `energy.MotorLimits`/`G1_LIMITS`
  (tau_peak from the model; **omega_noload 25 ESTIMATE — still the decisive unknown,
  flips the verdict**), `metrics.saturation()` (reads real per-joint torque cap,
  robust percentiles), `env.joint_torque_limits()`. Walking diagnostic with real
  caps: hip torque-bound (50 % of 88), knee speed-bound (at the estimate).
  A motor-param web chase (wf_7aae1cea) **FAILED — 0 external sources, circled back
  to our own repo estimates**; torque caps then recovered from the local menagerie
  model, and **the velocity limits from the official Unitree G1 URDF**
  (unitree_ros g1_23dof, direct WebFetch): **knee 139 N·m / 20 rad/s, hip 88 / 32,
  ankle 35–50 / 30**. **Torque-vs-speed RESOLVED:** the **knee is SPEED-limited**
  (walker already at ~52–67 % of 20 rad/s) and the **hip is TORQUE-limited** (50 %
  of 88 N·m, 11 % of speed). Consequence: **a parallel spring cannot raise G1 jump
  HEIGHT** (knee speed-capped → series/low-gear for height); it still helps the hip
  (torque) and the efficiency/landing case. `energy.G1_JOINT_VEL` added;
  `metrics.saturation()` now uses both real walls per joint.
  Sawtooth confirmed: control reverses ~55 %/step, peak/RMS up to 4.4 — inflates
  ohmic; the saturation diagnostic now uses robust percentiles to dodge it.
  Fixed two refuted citations in `related_work.md` (Plooij&Wisse ~80% → disputed,
  ~20% on a 2-DOF arm; BirdBot −90% unverified). **Decisive next input:** real 7520
  omega_noload; **decisive test:** a max-jump policy (GPU) + the envelope, then
  `metrics.saturation()` reads off torque- vs speed-limited.

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
