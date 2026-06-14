# Robot Inventory — Experiment-Preparation Reference

## 1. Purpose & provenance note

This document is the consolidated readiness inventory for porting the parallel-elastic
energy study off the Unitree G1 onto a roster of candidate legged platforms (bipeds:
G1, H1, Berkeley Humanoid, Booster T1, Cassie; quadrupeds: Go1, Go2, Barkour, Spot,
ANYmal C; plus the DecART leg). It records, per robot, what model/env is actually
installed and loadable, the actuator electrical constants needed for the ohmic
(`P = (τ/Kt)² · R`) **electrical**-energy model under the project's **no-regeneration**
assumption, the parallel-spring target joint, and every gap that blocks an absolute
electrical number — folding in the adversarial verification corrections.

**Confidence note (read first).** Following team convention, all torque constants are
**joint-side Kt** and the load-bearing quantity is the **gear-invariant `R/Kt²`** — lead
with it, because absolute watts inherit two compounding unknowns (gear ratio and winding
resistance) for almost every robot. **Mechanical/geometric facts** (gear class, peak
torque from `jnt_actfrcrange`, velocity limits, mass, DoF, joint names, env availability)
are HIGH confidence — read directly from the installed MJCF/URDF. **The electrical
constants Kt and R — the quantities the entire ohmic thesis rests on — are LOW confidence
for nearly every robot.** Only **Go1** has a HIGH-confidence external joint-side Kt
(GO-M8010-6 datasheet, 0.639 N·m/A) and only **Go2** has a measured R (Simplexity
teardown). Every value tagged *estimate-grade* / *proxy* / *blocked* below is NOT
externally sourced for that robot; treat all absolute electrical-energy and CoT headline
numbers as provisional. The project's documented failure mode is letting a self-estimate
pass as an external number — every such value is flagged explicitly here.

---

## 2. Table 1 — Model & environment availability

| Robot | Morphology | Gear class | Menagerie model | Playground env | Status | Effort to make experiment-ready |
|---|---|---|---|---|---|---|
| **Unitree G1** | biped | MID (14.3–22.5:1) | `unitree_g1` | `G1JoystickFlatTerrain`, `G1JoystickRoughTerrain` | **ready_env** | none — baseline platform, runs now |
| **Unitree H1** | biped | LOW (~10:1, est.) | `unitree_h1` | `H1JoystickGaitTracking`, `H1InplaceGaitTracking` (gait-tracking only) | **ready_env** | low — env is gait-tracking, not flat-walk joystick; confirm task fit |
| **Berkeley Humanoid** | biped | LOW (9:1) | `berkeley_humanoid` | `BerkeleyHumanoidJoystickFlatTerrain`, `…RoughTerrain` | **ready_env** | none |
| **Booster T1** | biped | MID/LOW (undisclosed) | `booster_t1` | `T1JoystickFlatTerrain`, `T1JoystickRoughTerrain` | **ready_env** | none (sim); electricals blocked |
| **Unitree Go1** | quadruped | LOW (6.33:1) | `unitree_go1` | `Go1JoystickFlatTerrain`, `…RoughTerrain`, `Go1Getup/Handstand/Footstand` | **ready_env** | none |
| **Unitree Go2** | quadruped | LOW (6.22:1) | `unitree_go2` | **none** (Go1 env is near-identical proxy) | **menagerie_needs_env** | medium — wrap env or reuse Go1 env with Go2 constants |
| **Google Barkour vB** | quadruped | LOW (6:1 or 9:1, see gaps) | `google_barkour_vb` (+ `google_barkour_v0`) | `BarkourJoystick` (velocity only) | **ready_env** | none for walking; Part 2 jump needs custom env |
| **Boston Dynamics Spot** | quadruped | HIGH (~51:1 harmonic hip + ball-screw knee) | `boston_dynamics_spot` | `SpotFlatTerrainJoystick`, `SpotGetup`, `SpotJoystickGaitTracking` | **ready_env** | none (sim); electricals fully blocked |
| **ANYmal C** | quadruped | HIGH (harmonic SEA, ratio unpublished) | `anybotics_anymal_c` (+ `anymal_b`) | **none** (only MJX APG demo notebook) | **menagerie_needs_env** | medium — author MJX joystick env + randomizer |
| **Agility Cassie** | biped | MID (16:1 / 25:1 / 50:1) | `agility_cassie` | **none** | **menagerie_needs_env** | medium — author env wrapper |
| **DecART Leg** | leg | UNPUBLISHED (decisive gap) | **none** | **none** | **no_model** | high — model from paper or request non-public URDF |

---

## 3. Table 2 — Actuator specifications

All Kt are **joint-side**. Lead column for the study is **`R/Kt²`** (gear-invariant ohmic-loss
coefficient). Bold = corrected by verification vs the team's prior belief.

| Robot | Actuator | Gear ratio | Kt (joint-side, N·m/A) | Winding R (Ω) | **R/Kt²** (Ω/(N·m/A)²) | Peak torque (N·m) | Max joint vel (rad/s) | Actuation class | Kt/R confidence |
|---|---|---|---|---|---|---|---|---|---|
| **G1** | Unitree 7520-22.5 (knee/hip_roll), **7520-14.3 (hip_pitch/hip_yaw)**, 5020 (ankle) | **knee 22.5:1; hip_pitch 14.3:1** (NOT 22.5) | ~2.3 *(estimate)* | ~0.013 *(estimate)* | team ~0.0025 *(8–19× optimistic vs Go2 proxy → real ~0.02 knee / ~0.05 hip_pitch)* | hip_pitch ±88, hip_roll/knee ±139, ankle ±50 | hip 32, knee 20, ankle 30 | QDD planetary, backdrivable | **LOW** — no primary G1 Kt/R |
| **H1** | M107 (legs); smaller motors ankle/arms | ~10:1 *(estimate, 8–15:1)* | ~2.6 *(estimate, Go2-proxy ×~10)* | ~0.66 *(estimate, Go2 proxy)* | ~0.098 *(estimate; likely lower — M107 ≫ Go2)* | hip_pitch 200, knee 300, ankle 40 *(sim; marketing 220/360)* | hip 23, knee 14, ankle 9 | QDD planetary, backdrivable | **LOW** — no primary H1/M107 Kt/R/gear |
| **Berkeley Humanoid** | 5013/8513/8518/10413 (HFE=8518, KFE=10413) | **9:1** (NOT 9.1:1 — that is HECTOR) | ~0.83 *(estimate, Lite M6C12 proxy)* | ~0.19 *(estimate, Lite proxy)* | ~0.28 *(estimate; ~100× G1)* | HFE 62.6, KFE 81.1 (hw peak); sim caps 30/30 | HFE 29, KFE 27.9 (hw); sim 20/14 | QDD single-stage planetary, backdrivable | **LOW** — cross-robot Lite proxy only |
| **Booster T1** | custom QDD, hollow-shaft dual-encoder | **UNDISCLOSED** | **UNDISCLOSED** | **UNDISCLOSED** | **NOT COMPUTABLE** (no external Kt or R) | model: Hip_Pitch ±45, Knee ±60; marketing **130 peak** | Hip_Pitch 12.5, Knee 11.7, Ankle 18.8 | QDD, full force control, backdrivable | **LOW** — zero external sources |
| **Go1** | Unitree GO-M8010-6 | **6.33:1** (hip/thigh); knee ~9.5:1 effective *(inferred)* | **0.639 (datasheet, joint-side)**; knee ~0.96 *(inferred)* | **UNPUBLISHED** (estimate ~0.05–0.3 motor-side) | ~0.12–0.49 *(estimate; ~4× band because R unmeasured)* | hip/thigh 23.7, calf 35.55 | hip/thigh 30.1, calf 20.06 | QDD single-stage planetary, backdrivable | **Kt HIGH** (datasheet); **R unpublished** |
| **Go2** | GO-M8018-6 family | **6.22:1** (measured) | **~1.62** (= 0.26 motor-side q-axis × 6.22) — **team's 0.26 was MOTOR-side, not joint-side** | **0.44 line / 0.66 phase (MEASURED, teardown)** | ~6.5–14.6 motor-side; **joint-side 0.44/1.62² = 0.168** *(side-consistent, not literally gear-invariant)* | hip/thigh ±23.7, calf ±45.43 | hip/thigh 30.1, calf 15.70 | QDD single-stage planetary, backdrivable | **Kt/R MEDIUM-HIGH** (one teardown) |
| **Barkour vB** | **T-Motor AK80-6 (deployed)** — NOT AK80-9 (build-repo BOM only) | **6:1** (deployed; team's ~6:1 was CORRECT) | ~0.63 *(catalog; = 0.105 × 6)* | ~0.085/phase, 0.17 p-p *(catalog)* | ~0.21–0.43 *(catalog estimate; ~86–170× G1)* | 12 (deployed, = catalog peak); sim cap 18 | ~14 (48 V) / ~7 (24 V on-robot) | QDD single-stage planetary, backdrivable | **LOW** — catalog, not measured; 24 V bus |
| **Spot** | custom (3 identical/leg); hips harmonic, knee ball-screw | **hip 51:1** (NOT 80–160); knee variable | **UNDISCLOSED** (proxy ~3.6–5.1) | **UNDISCLOSED** (proxy ~0.1–0.5) | **NOT COMPUTABLE** (proxy ~31 motor-side only) | hip ~45 (= 0.88×51, team arithmetic); sim-tuned ~97 | ~25 (sim-ID paper) | **harmonic + ball-screw, NOT QDD** (limited backdrive) | **LOW** — BD discloses neither; permanent blocker |
| **ANYmal C** | ANYdrive SEA (BLDC + harmonic + series spring) | **UNPUBLISHED** (team 100:1 unconfirmed; harmonic class 30–160:1) | **UNVERIFIED** (~2.7 back-out, estimate) | **UNPUBLISHED** | **NOT COMPUTABLE** (both unpublished) | 40 peak / 15 nominal (joint-side) | 12 | **SEA harmonic-drive, NOT QDD** | **LOW** — sidestep via Bjelonic τ² metric |
| **Cassie** | Agility custom BLDC QDD; 8 cycloidal legs + 2 harmonic toes | **16:1 hip_pitch/knee; 25:1 hip_roll/yaw; 50:1 toe** (NOT ~10:1) | ~1.9 (16:1) / ~2.95 (25:1) *(MF0127 proxy, NOT confirmed)* | ~0.23 *(MF0127 proxy)* | ~0.064 (16:1) / ~0.026 (25:1) *(estimate; proxy spans 0.026–0.19)* | hip_roll/yaw 112.5, hip_pitch/knee 195.2, toe 45 | hip_roll/yaw 12.15, hip_pitch/knee 8.51, toe 11.52 | QDD cycloidal (legs) + harmonic (toes); separate passive leaf-spring SLIP | **LOW** — MF0127 proxy, no Cassie part named |
| **DecART** | "proprioceptive rotational actuators" (unnamed) | **UNPUBLISHED — DECISIVE** | **UNKNOWN — not derivable** | **UNKNOWN** | **CANNOT COMPUTE** (no external value, no team estimate yet) | UNPUBLISHED (in non-public URDF) | UNPUBLISHED (only derived ~4.18 m/s leg-tip) | QDD intended (unconfirmed) | **LOWEST** — no external data at all |

---

## 4. Table 3 — Platform

Spring-target convention: bipeds → sagittal **hip-pitch**; quadrupeds → sagittal **thigh (hip-pitch)
AND calf/knee**; DecART → telescopic **leg-length** axis.

| Robot | Mass (kg) | DoF | Sagittal leg joints | Parallel-spring target |
|---|---|---|---|---|
| **G1** | ~33.3 (sim) / ~35 (hw) | 29 actuated (sim) | hip_pitch, knee, ankle_pitch | **hip_pitch** (14.3:1, not the 22.5:1 knee) |
| **H1** | ~47 | 19 actuated | hip_pitch, knee, ankle (1-DoF serial) | **hip_pitch** (200 N·m, 23 rad/s) |
| **Berkeley Humanoid** | 16 (no arms) / ~22 (w/ arms) | 12 leg | HFE (hip_pitch), KFE (knee), FFE (ankle) | **HFE (hip_pitch) + KFE (knee)** |
| **Booster T1** | ~30 | 23 (12 leg) | Hip_Pitch, Knee_Pitch, Ankle_Pitch | **Hip_Pitch + Knee_Pitch** |
| **Go1** | ~12 | 12 leg | `_thigh_joint` (HFE), `_calf_joint` (KFE) | **thigh + knee** (knee highest torque) |
| **Go2** | ~15 | 12 leg | `_thigh_joint`, `_calf_joint` | **thigh + knee** (knee 45.43 N·m, lowest speed) |
| **Barkour vB** | ~11.5 (hw) / ~13.5 (vB MJCF) | 12 leg | `hip_*` (thigh), `knee_*` | **thigh + knee** |
| **Spot** | 32.5 | 12 (3/leg) | HY (hip_pitch), KN (knee) | **HY + KN** (knee variable transmission complicates τ_spring(θ)) |
| **ANYmal C** | ~52 (study config) / ~30–33 (commercial) | 12 leg | HFE (hip_pitch), KFE (knee) | **KFE (knee)** — published target (Bjelonic 2023, ks=4154 N/m) |
| **Cassie** | ~31 (hw) / 33.3 (MJCF) | 20 (10 actuated, 2 passive spring/leg) | hip_pitch, knee, foot (actuated); shin/heel passive | **hip_pitch** (leg-length already sprung in series — do NOT re-spring distal leaf) |
| **DecART** | ~35 | 6/leg (12 pair) | j1 hip_pitch, j3 leg-length, j4 ankle_pitch | **leg-length (j3)** — telescopic, prismatic-equivalent; no clutch needed |

---

## 5. Table 4 — Gaps & blockers (with verification corrections folded in)

| Robot | Key gaps / unverified | Blocks absolute electrical? | Verification correction (draft → corrected) |
|---|---|---|---|
| **G1** | Kt, R have NO primary source; team values estimate-only & optimistic | **YES** — R/Kt² 8–19× optimistic; recompute with ~0.02–0.05 | **hip_pitch is 14.3:1, not 22.5:1** (team CLAUDE.md/RESULTS wrong). hip_roll is 139 not 88. OmniXtreme does NOT corroborate 14.3 (it says 22.5) — local armature pins it. |
| **H1** | Gear ratio, Kt, R all estimate-grade; no primary M107 data | **YES** | Go2 proxy Kt is Kt_phase 0.22 (measured), 0.26 is q-axis Kt_q — frame mix understates FOM ~40%. Sim hip-pitch peak is 200 N·m, not the 220 marketing figure. Env is gait-tracking only. DoF confirmed 19 (1-DoF ankle). |
| **Berkeley Humanoid** | Kt/R unpublished for original robot; Lite M6C12 proxy is a different motor (MAD BLDC, 15:1 cycloidal) | **YES** | **Gear 9:1, not 9.1:1** (9.1:1 is HECTOR — stale value still in docs/taxonomy.md:54, directions.md:137-138,224). Web search itself returns Lite numbers mislabeled as "Berkeley" — do not conflate. |
| **Booster T1** | Kt, R, gear ratio, bus voltage ALL undisclosed; only torque/velocity limits exist | **YES** — any Kt/R is 100% team proxy | 130 N·m is verbatim "Max Peak Torque" (confirmed peak, resolves 130-vs-60 sim gap). "Payload 5 kg" traces to aggregators, not booster.tech. Hardware ankle is parallel linkage (2-motor), NOT Stewart; sim models it serial. |
| **Go1** | **R unpublished** (the one hard blocker); knee effective gear ~9.5:1 and Kt ~0.96 are inferred not published | **R blocks absolute**; Kt is solid | Knee Kt ~0.96 and ~9.5:1 are INFERRED from URDF effort ratios (config-dependent), not datasheet. Do NOT reuse energy.py's go2 placeholder R for Go1. |
| **Go2** | R from single teardown only; phase-vs-line convention (~2× band); knee linkage ratio unconfirmed | Adequate for RELATIVE; ~2× R band on absolute | **Team's "Kt_joint ~0.26" is MOTOR-side q-axis; joint-side is ~1.62.** energy.py line 56 plugs 0.26 as joint-side → overstates ohmic loss ~39×. R=0.30 placeholder superseded by measured 0.44–0.66. |
| **Barkour vB** | Catalog (not measured) Kt/R; 24 V bus halves speed & lowers back-EMF threshold; platform fork (paper robot vs build-repo) | **YES** — catalog estimate; G1 comparison denominator is itself team estimate | **Draft's AK80-9/9:1 is WRONG.** Deployed robot uses **AK80-6 (6:1)** per the paper (arXiv:2305.14654, "T-Motor AK80-6 … Elmo … 24 V … 12 N·m"). Team's prior ~6:1 belief was CORRECT. Build-repo BOM's AK80-9 is a different/later rev. |
| **Spot** | Kt, R, phase current, bus voltage NEVER published by BD; no teardown gives windings — **permanent blocker** | **YES — permanently** unless BD discloses | **Hip is ~51:1, not the team's 80–160:1.** Knee is variable ball-screw (no single ratio). ~45 N·m hip is team arithmetic (0.88×51), not a BD spec. arXiv knee torque limits were the HIP's flat range (misattributed). Class is harmonic+ball-screw, NOT QDD. |
| **ANYmal C** | Gear ratio, Kt, R all unpublished; datasheet defers to unprinted motor data sheets | **YES** — but de-risked by Bjelonic | Team's 100:1 & ~2.7 Kt are self-estimates, NOT external. Class is **SEA harmonic-drive, not QDD**. **Published parallel-elastic result (33% CoTr, gear/Kt-invariant ∫τ²dt) needs NO absolute Kt/R** — report that metric. |
| **Cassie** | Kt/R are MF0127 proxy with NO confirmed link to Cassie; only published Cassie loss model is non-dimensional (unusable as SI) | **YES** | **Gear is 16:1/25:1/50:1, not ~10:1** (team belief wrong; weakens ohmic case). "ATRIAS lineage" link is team conjecture, not sourced — drop it. Elasticity is a passive leaf-spring SLIP linkage, not a "cycloidal SEA". No-load RPM IS in the MJCF (user=2900/1300/5500). |
| **DecART** | **Everything electrical + the model itself.** No URDF/MJCF/CAD; no gear ratio, Kt, R, peak torque, velocity — not even a team estimate exists | **YES — fully** | Confirmed (also an IEEE Xplore paper, doc 11247361). Springs are explicitly FUTURE work in the paper. Highest-cost candidate: model-from-scratch or request non-public `DecARt_v01.urdf`. Fixed-base/leg morphology breaks the 6-DoF free-base assumption; spring target is a prismatic axis. |

---

## 6. Loadable now vs. needs work

**Installed versions:** mujoco 3.9.0, mujoco-mjx 3.9.0, mujoco_playground (PyPI `playground`)
0.2.0, jax 0.9.2, jaxlib 0.9.2, brax 0.14.2. Python 3.12, uv venv.
Playground pkg: `…/.venv/lib/python3.12/site-packages/mujoco_playground`. Menagerie is
vendored inside Playground (`…/external_deps/mujoco_menagerie`, fully populated); a
standalone `mujoco_menagerie` package is NOT installed.

**CRITICAL gotcha (confirmed, matches CLAUDE.md):** `registry.load()` fails with
`type object int has no attribute WARP` under the default config — Playground 0.2.0
defaults `impl` to MuJoCo Warp. **Set `config.impl='jax'` before every env load on this Mac**;
all 19 legged envs then instantiate cleanly.

### Ready env (Playground env + Menagerie model, loads now)
- **G1** (nq=36/nu=29), **H1** (nq=26/nu=19, gait-tracking env), **Berkeley Humanoid**
  (nq=19/nu=12), **Booster T1** (nq=30/nu=23), **Go1** (nq=19/nu=12), **Barkour**
  (nq=19/nu=12, uses `google_barkour_vb`), **Spot** (nq=19/nu=12).
- Each instantiated via `registry.load`, not just name-listed.
- Bonus legged envs beyond roster: `ApolloJoystickFlatTerrain`, `Op3Joystick`.

### Menagerie — needs env wrapper (MJCF loads via `MjModel.from_xml_path`; no Playground env)
- **Go2** (nq=19/nu=12) — Go1 env is the near-zero-effort proxy (6.22:1, same family).
- **ANYmal C** (nq=19/nu=12) — only an MJX APG demo notebook exists.
- **Cassie** (nq=35/nu=10) — no env.
- Also present without env (potential extras via custom wrapper): `anybotics_anymal_b`,
  `unitree_a1`, `pal_talos`, `fourier_n1`, `pndbotics_adam_lite`.

### No model
- **DecART** — exists only as a string in `src/pea/metrics.py:6` comment. No model, no env
  anywhere. Model-from-paper or request the non-public URDF is required.

---

## 7. Proposed architectural changes (for the owner to DECIDE — not applied)

The codebase is already half-generalized: `metrics.py` and `experiment.py` take explicit
DoF indices and Kt/R and are robot/task/spring agnostic; `joints_by_substring` and
`joint_torque_limits` read the real model. The remaining G1-coupling is concentrated in
(1) the **live in-loop training reward** (`ElectricalRewardWrapper` hardcodes `G1_KNEE`
Kt/R and a literal `[..., 6:]` base offset — the most dangerous silent failure), (2) the
thin `energy.MOTORS` dict (only g1/go2; the go2 row uses a MOTOR-side Kt that mis-scales
joint torque ~39×), and (3) the knee-bound `rollout.py`/`analyze.py` path.

### Proposal 1 — Per-robot REGISTRY in a new `src/pea/robots.py` (single source of truth) — **do-now (M)**
- **Problem:** robot facts are scattered across 4–5 files (env_name in config.py:35, motors
  in energy.py, spring joint in config.py:23, vel limits in energy.py, base offset literal
  `6` in env.py:66-67); the per-joint gear corrections have nowhere to be recorded against
  the robot they belong to.
- **Change:** add a frozen `RobotSpec` dataclass + `ROBOTS: dict[str, RobotSpec]` with
  env_name, menagerie_model, morphology, per-substring motor constants (so per-joint gear
  differences are first-class, e.g. G1 hip_pitch 14.3:1 vs knee 22.5:1), default_motor,
  spring_joints per morphology, vel_limits, mass_kg, free_base_dofs, and a confidence/blocker
  string per robot. Move `G1_KNEE`/`MOTORS`/`G1_JOINT_VEL` into the registry; keep thin
  back-compat aliases in energy.py.
- **Files:** NEW `src/pea/robots.py`; `src/pea/energy.py` (re-export for back-compat).

### Proposal 2 — Add a `robot` field to `RunConfig`; thread Kt/R/joints/base-offset from the registry — **do-now (M)**
- **Problem:** `ElectricalRewardWrapper` hardwires Kt=2.3, R=0.013 into the live training
  reward. Pointed at a low-gear robot (Go2 joint-side R/Kt² is ~8–200× the G1's) it does NOT
  error — it silently trains against a G1-scaled ohmic penalty off by orders of magnitude,
  optimizing the wrong objective and producing a confident, wrong cross-robot conclusion.
- **Change:** add `RunConfig.robot: str = 'g1'`; default env_name/spring.joint from the
  registry; give `ElectricalRewardWrapper` `kt`/`r` constructor args passed from
  `robot_spec(cfg.robot)`; replace the literal `[..., 6:]` slice with registry
  `free_base_dofs` (prefer `jnt_dofadr`-derived). Add a build-time assertion that the env's
  model joints contain the spring-target substrings (so a quadruped run with `knee` fails
  loudly, not silently).
- **Files:** `src/pea/config.py`, `src/pea/env.py`.

### Proposal 3 — Populate the motor registry for every roster robot with explicit status — **do-now (M)**
- **Problem:** `energy.MOTORS` has only g1/go2; `motor_constants()` raises KeyError for 8 of
  10 robots. The go2 row's Kt=0.26 is MOTOR-side (joint-side ~1.62 → ~39× error) and R=0.30
  is a self-declared placeholder superseded by the teardown.
- **Change:** add rows for go1 (Kt=0.639 datasheet, R flagged unpublished), go2 (FIX to
  joint-side Kt~1.62, R 0.44–0.66 with convention noted), and berkeley/h1/booster_t1/spot/
  anymal/cassie (proxy estimates, each tagged with source + correction). Add a per-row
  `status` enum (measured | datasheet | proxy-estimate | blocked). For Spot/ANYmal/Cassie/
  DecART (no external Kt/R), populate but mark `blocked` and have `motor_constants()` refuse
  absolute-watt-grade constants without explicit `--kt/--r`, so an unsourced number cannot
  silently become a headline. Keep R/Kt² as the documented comparison field per row.
- **Files:** `src/pea/energy.py` (or `src/pea/robots.py`).

### Proposal 4 — De-knee-couple `rollout.py` and `analyze.py` (or retire for cross-robot) — **do-now (M)**
- **Problem:** rollout.py writes knee-specific arrays via `knee_joints()`, which raises on any
  robot with no `knee` joint. analyze.py reads `tr['knee_*']` and hardcodes `kt,r=G1_KNEE`
  with no escape hatch — errors or silently reports G1-scaled energy on a non-G1 trajectory.
- **Change:** (A, preferred) make rollout.py log the registry's `spring_joints` generically
  and give analyze.py `--robot/--kt/--r` defaulting from `config.robot`; OR (B) demote
  analyze.py to an explicitly G1-only diagnostic and standardize on experiment.py+metrics.py.
- **Files:** `src/pea/rollout.py`, `src/pea/env.py` (`knee_joints`), `src/pea/analyze.py`.

### Proposal 5 — Env-wrapper plan for Menagerie-only robots (Go2, ANYmal C, Cassie) — **later (M)**
- **Problem:** these have an MJCF but no Playground env; `make_env`/`get_domain_randomizer`/
  `brax_ppo_config` all raise for an unregistered name. "Just set env_name" fails.
- **Change:** do NOT author full envs now. Write `docs/cross_robot_envs.md`; for Go2 record
  reuse-Go1-env-with-Go2-constants as a deliberate env/motor mismatch flag; for ANYmal/Cassie
  mark `model_availability='needs_env'` and scope each wrapper as a separate L task, gated on
  whether an in-loop result is actually needed (post-hoc metrics.py can score a hand-rolled
  trajectory). Note ANYmal's Bjelonic result needs no absolute Kt/R.
- **Files:** NEW `docs/cross_robot_envs.md`; `src/pea/robots.py`.

### Proposal 6 — DecART modeling effort — scope and gate it explicitly — **later (L)**
- **Problem:** no public model, no env, no external gear/Kt/R/torque/velocity — not even a team
  estimate. Fixed-base/leg morphology breaks the `free_base_dofs=6` assumption; spring target
  is a telescopic (prismatic) axis the substring/springs machinery does not model.
- **Change:** gate behind (1) request `DecARt_v01.urdf` from authors OR decide to build from
  paper; (2) only then add a robots.py entry. Until the model arrives, only a placeholder
  registry stub + docs note. Mark BLOCKED and highest-cost so it is not scheduled before the
  G1/Go-family results land.
- **Files:** `docs/cross_robot_envs.md`; `src/pea/robots.py` (stub only).

### Proposal 7 — Preserve matched-reward / identical-constants discipline across robots — **do-now (M)**
- **Problem:** validity rests on baseline-vs-spring parity. Across robots: (a) an
  energy_reward_weight calibrated for G1 is mis-scaled for a low-gear robot (its ohmic term
  is far larger); (b) biped `['hip_pitch']` vs quadruped `['thigh','calf']` means the "same"
  sweep is not the same joints (and 2 legs vs 4); (c) reward-side and metric-side Kt/R must be
  the SAME registry value or the alignment that justifies the wrapper breaks.
- **Change:** build both arms from ONE RunConfig differing only in `spring.kind`; add a
  config-diff assert that the two run folders share identical robot/env/reward_scales/
  env_overrides/energy_reward_weight/motor constants. Make energy_reward_weight robot-relative
  (fraction of mean tracking reward). Document the biped-vs-quadruped joint-count asymmetry
  (compare per-joint % offload, not whole-body). Pull reward-side and metric-side Kt/R from
  the same `robot_spec`.
- **Files:** `src/pea/config.py`, `src/pea/env.py`, `src/pea/experiment.py`, `docs/cross_robot_envs.md`.

### Proposal 8 — Per-robot smoke test scaffolding (catch silent mis-targeting before GPU spend) — **do-now (M)**
- **Problem:** nothing verifies per robot that the env loads (impl='jax' gotcha), spring
  substrings match joints, base offset is right, and motor_constants resolves. Bad configs
  surface only after a run, i.e. after GPU spend.
- **Change:** add `tests/test_robots.py` (or `scripts/check_robot.py`) that, for each
  ready_env/menagerie entry: loads with impl='jax', asserts spring_joints resolve, asserts
  free_base_dofs matches the model's free-joint DoF count, asserts robot_spec returns and
  surfaces its status, and runs ONE metrics.evaluate step on a random policy for finite
  numbers. Mark no-model/no-env robots as expected-skip with reasons. CPU-only, fast.
- **Files:** NEW `tests/test_robots.py` (or `scripts/check_robot.py`).

### Proposal 9 — Surface estimate/blocker status in outputs and headline numbers — **do-now (S)**
- **Problem:** for most robots R/Kt² is estimate-grade or unsourced; today analyze.py and
  experiment.py print Kt/R but not provenance, so a CoT from a blocked robot looks identical
  to one from go1 (datasheet Kt). A reader cannot tell which absolute numbers are trustworthy.
- **Change:** carry the per-robot/per-constant status through to every output: a `status`
  column in experiment.py's table/CSV; status-aware analyze.py that for blocked robots
  REFUSES to print an absolute CoT and prints only the Kt/R-invariant ohmic-% reduction and
  the ∫τ²dt-style relative metric (the externally-defensible quantity, per the ANYmal
  precedent). Mostly string-plumbing on top of the registry status field.
- **Files:** `src/pea/experiment.py`, `src/pea/analyze.py` (reads status from `src/pea/robots.py`).

---

## 8. Open decisions for the owner

These gate the work and need an explicit call:

1. **Tier-0 cheap screen — which robots?** The cleanest, lowest-cost screen uses only
   **ready_env robots that load now and need no env authoring**: G1, H1, Berkeley Humanoid,
   Booster T1, Go1, Barkour, Spot. Of these, the **low-gear bipeds/quadrupeds where the ohmic
   lever is largest** (Berkeley 9:1, Go1 6.33:1, H1 ~10:1) are the strongest energy-thesis
   candidates; G1 (MID) and Spot (HIGH, harmonic) are the "gear-limited, modest-win" controls.
   **Recommendation to decide:** run Tier-0 on {G1 (baseline), Go1 (LOW, datasheet Kt — the
   one trustworthy absolute), Berkeley (LOW), Spot (HIGH control)} to bracket the gear axis
   with the most defensible constants.

2. **The Go1/Go2 winding-R blocker.** Go1's Kt is the single HIGH-confidence external value
   in the roster but its **R is unpublished**; Go2 has a **measured** R (teardown). Decide:
   (a) adopt the Go2 teardown R (0.44 line / 0.66 phase) as the Go-family R and run Go1 with
   datasheet Kt + Go2 R, or (b) bench-measure a real GO-M8010-6 phase resistance. Until one
   is chosen, all Go-family absolute ohmic watts carry the R gap. Also pick the R
   **convention** (phase vs line — ~2× band) before quoting any absolute number.

3. **The DecART modeling cost.** Highest-cost candidate: no model, no env, no external
   constants. Decide whether to (a) **request `DecARt_v01.urdf` from the authors** (the
   FAST-metric README invites this), (b) build from the paper, or (c) **drop DecART** from the
   roster. Do not start blind — the gear ratio alone decides whether the ohmic win exists, and
   it is unpublished.

4. **Robots whose Kt/R are wholly unverifiable.** **Spot** (BD discloses nothing — permanent
   blocker), **ANYmal C**, **Cassie**, **Booster T1**, and **DecART** have no external
   Kt/R. Decide the reporting posture: per the ANYmal precedent (Bjelonic 2023, 33% on the
   gear/Kt-invariant ∫τ²dt), **report only the Kt/R-invariant ohmic-% reduction and the
   relative ∫τ²dt metric for these robots, and SUPPRESS absolute CoT** (Proposal 9). This
   keeps the headline externally defensible. Confirm this is the agreed convention before any
   cross-robot run produces a quotable watt figure.

5. **Go2 / ANYmal / Cassie env authoring — build or proxy?** Decide whether an **in-loop**
   (retrained) result is required for these three, or whether the **post-hoc** metrics path
   (score a hand-rolled trajectory, no trained policy) suffices. For Go2 the cheap path is
   reusing the Go1 env with corrected Go2 constants. ANYmal/Cassie each need a real MJX
   joystick env + randomizer + brax_ppo_config entry (an L task each) — only commit if an
   in-loop number there is actually needed.
