# CLAUDE.md — Parallel-Elastic Efficiency Study

## Project goal
Two parts, **in sequence — finish Part 1 before starting Part 2.**

### Part 1 — Energy efficiency
Test whether adding a tunable **parallel elastic** spring (a spring mounted *beside* the
motor, so it shares the joint's load rather than sitting in the force path) reduces the
*electrical* energy of walking, by offloading motor torque. Headline metric: cost of
transport (CoT) — electrical power divided by walking speed. Plain-language definitions of
every term used here are in `docs/glossary.md`; the full run/checkpoint inventory is in
`docs/checkpoints.md`.

Mechanism / why this should work:
- Motor copper loss scales with torque squared: `P_loss ≈ (τ/Kt)² · R`. Offloading
  torque to a passive spring cuts heat **quadratically**, not linearly.
- Two distinct wins: (1) lower RMS torque → lower copper loss; (2) energy recovery —
  the spring stores energy the motor would otherwise dissipate (braking) and returns it.
- **Parallel**, not series: the goal is torque offloading, not force control. A parallel
  spring sits beside the motor and avoids the large-force-bandwidth penalty of series
  compliance.
- *Reality check — the study's MAIN FINDING (2026-06-14/16): **gearing is the crux**.*
  "Gearing" = the gear ratio between motor and joint; a high ratio multiplies torque so the
  motor runs at low current, which is why ohmic heating (the heat in the motor windings,
  `(τ/Kt)²·R`) is a tiny share of the energy bill on a high-geared robot. On the high-geared
  G1 (knee 22.5:1, hip-pitch 14.3:1) ohmic is only ~4 % of the budget, and the **in-loop**
  spring (a spring present in the simulation while the policy retrains) is actually **WORSE**
  for walking (**+7.4 %**, reversing the optimistic offline −3.84 %; the always-on spring
  fights the gait, no clutch) — this is the project's **central negative result**, catalogued
  with the others in `docs/negative_results.md`. But on the **LOW-gear Go1 quadruped**
  (6.33:1, ohmic **54 %**) a **CONSTANT knee preload** (a near-constant offload torque) cuts
  **CoT −14 to −27 % in 3 of 4 conditions, growing with load** (3 training seeds; seed 2 is a
  weak −3 to −8 % outlier) — the one positive result. It carries **no stability cost at
  low-to-mid load, but survival degrades above ~7.5 kg payload**. The **Go1 load-carrying
  WALKING program is DONE and positive** (`docs/load_program.md`). **ACTIVE direction:** the
  **Go1 "dog-running" knee-spring experiment** — testing whether the same idea pays for a
  *running* quadruped, which adds a braking-energy-recovery channel walking lacks. Authoritative
  plan: **`docs/NEXT_SESSION.md`** + design in **`docs/dog_running_design.md`**. Direction map:
  `docs/directions.md`. **UPDATE 2026-06-18:** the forward-command curriculum SOLVED the standing collapse —
  the dog now genuinely RUNS (C1 1.47 m/s, C2 2.16 m/s) — but the fast+flight C3 stage over-corrected back to
  standing, so there is **no flight phase yet** and the running-spring test is still pending (next: a gentler
  C3). New infrastructure: a `command_forward` forward-command override (the collapse was a symmetric/zeroed
  command distribution, not a too-high target) and the G1 flight-enabling env subclass `G1JoystickRun`
  (`src/pea/g1_run_env.py`). Presentation materials archived in `outputs/_locomotion_archive/presentation/`.
  **UPDATE 2026-06-19 — gravity-compensation direction DONE and POSITIVE (program doc:
  `docs/gravity_compensation.md`); the dog-running AND G1-running LOCOMOTION tracks remain SUSPENDED.**
  A Part-1 spinoff testing parallel springs on joints with CONSTANT-SIGN gravity load
  (body up/down, loading/unloading) on two mobile-manipulator robots loaded from EXTERNAL URDFs (Galaxea R1
  wheeled humanoid; LimX W1 wheeled quadruped). The load never reverses, so a PERMANENT clutchless parallel
  spring is unambiguously good — and the win is LARGE on BOTH high-gear (Galaxea torso lift, −96% torso-motor /
  −21% whole-robot @150 W computer) and low-gear (LimX knee during wheeled roll, −98% / −26%) platforms,
  because it offloads the constant-sign load (lift work / stance-holding ohmic), which is gear-INDEPENDENT — the
  opposite of the walking result where gearing was the crux. Two findings worth carrying forward: (a) **one
  well-placed spring ≈ the whole win** — the Galaxea knee (torso_joint2) is 87 % of the lift load, a single spring
  there gives −98 %; (b) **element kind follows load SHAPE** (re-confirms the Go1-knee rule): a linear spring fits
  joints whose gravity slopes with their own angle, but is mis-specified on a constant-load joint (torso_joint3,
  gravity span 0 → linear fit pushes θ₀ to the grid edge and under-fits) where a CONSTANT preload is correct
  — now auto-selected per joint by `gravcomp.fit_spring_per_joint` (joint3 −65 % → −84 %).
  Reporting bundle `outputs/gravity_compensation/` (videos, plots, combined table, README — gitignored);
  see `docs/gravity_compensation.md` + the 2026-06-19 JOURNAL entries.

### Part 2 — Explosive moves (after Part 1)
Test whether the **same adaptive elastic** helps EXPLOSIVE moves: vertical **jump
height**, broad-**jump distance**, **drop-landing** (jumping down), explosive
**sprint** start. Unlike Part 1 these may be **one-shot** (a single max jump — no
cross-cycle energy recovery, cost-of-transport does not apply) or **repetitive**
(continuous hopping — recovery applies). This is a genuine **bifurcation** from
Part 1 (repetitive-motion efficiency); see `docs/directions.md`. Part 2 itself
splits into two sub-cases with opposite verdicts on the high-geared G1:
- **(B1) Performance — jump HEIGHT / top speed.** Which architecture helps is set
  ENTIRELY by which wall takeoff hits, because a **series** spring can never add
  torque (force through it equals the motor force; it only amplifies *speed/power*
  by storing energy then releasing it fast) and a **parallel** spring can never
  beat the speed wall (it only adds *force*). So: **torque-limited takeoff →
  parallel helps; speed-limited takeoff → series helps and parallel does nothing.**
  **RESOLVED from the real specs** (model jnt_actfrcrange + Unitree G1 URDF velocity
  limits): the **knee is SPEED-limited** (139 N·m but only **20 rad/s**, and the
  walker already uses ~52–67 % of that speed; a jump needs the knee faster still),
  while the **hip is TORQUE-limited** (88 N·m, only ~11 % of its 32 rad/s). Since
  the knee is the jump extensor and it is speed-capped, **a parallel spring cannot
  raise G1 jump HEIGHT** — height points off the G1 (series compliance or a low-gear
  platform). The hip-torque case and the efficiency/landing case (B2) remain valid
  for a parallel spring. (Confirm with a max-jump policy; the walking data already
  pins the knee near its speed ceiling.)
- **(B2) Efficiency / peak-load of explosive moves — the defensible parallel case,
  LEAD with this.** Even with height capped, the spring cuts the **energy and peak
  load** of each push-off and landing. Parallel is the correct architecture here,
  and the gear speed-cap does NOT apply to **landing** (load is set by impact
  velocity, not motor speed). Jump/landing torques (≥139 N·m) dwarf walking, so the
  ohmic (∝ τ²) and braking-recovery channels are far larger than the ~3 % walking
  lever — this is where the spring most plausibly pays on the stock G1.

Metric: performance (height/distance, takeoff velocity, top speed, peak power) for
B1; energy + peak load for B2. The tunable spring + dead-zone clutch is the
between-conditions knob (engage for explosive, disengage for precise/efficient).
Gearing is the crux for BOTH parts; maximum jump HEIGHT may point off the G1 (a
low-gear quadruped or a DecART-style leg-length / series element).

**ACTIVE (2026-06-19) — Part 2 restarted on the G1: continuous-hopping for ENERGY.**
User-directed: a parallel knee/ankle spring on **continuous, two-footed, IN-PLACE
HOPPING** (jump→land→jump…) — the **repetitive** corner of B2, where landing-braking
energy (the spring's recovery channel) dominates and cross-cycle recovery applies
(unlike a one-shot jump). HEIGHT stays off-limits (knee speed-limited, NR-7); we
target **energy per hop (J/hop) at a matched apex+cadence — NOT CoT** (zero forward
speed). Built `G1JoystickHop` (`src/pea/g1_hop_env.py`): synchronous gait clock, five
hop reward terms (signed `hop_rhythm` + dense `hop_push` defeat the stand-still trap),
in-place command, config-fixable cadence/apex for the matched comparison. Spring =
knee/ankle **one_sided_linear** pogo, fit from a baseline work-loop. Authoritative
plan + validity checklist: **`docs/hop_design.md`**; configs `g1_hop_s1` (elicit → run
first), `g1_hop_baseline`+`g1_hop_spring` (matched pair). Validated locally (reset/step
+ full PPO `--smoke`).
**UPDATE 2026-06-20:** S1 hopper TRAINED (reward 0.76→66, a real two-footed hop). The pogo-knee-spring
energy comparison ran but is **INCONCLUSIVE, not negative** — confounded (the spring arm hopped 18 %
higher; ~13 % cheaper per-unit-apex, hinting it HELPS). For HOPPING the spring's win is braking-energy
recovery (a *mechanical-work* reduction, gear-INDEPENDENT), so the "gearing is the crux" argument does
NOT straightforwardly apply here. **Next: a FAIR apex-pinned re-run** (`scripts/run_hop_compare.sh`; apex
capped via the new `hop_overshoot` term + tight `hop_height_var`) → `scripts/hop_energy_compare.py`
(FAIRNESS GATE). A **gait-conditioned single controller** (hop / single-leg / run in one policy, commanded
by a per-phase foot-contact schedule) is proposed for the chapter after — `docs/gait_controller_design.md`.
The bounding env `G1JoystickBound` (`src/pea/g1_bound_env.py`) is built + validated, but its autonomous
run FAILED on infra (a busy-loop driver wedged the box — nothing trained; see the box-safety gotcha).
(The dog-/G1-running and gravity-comp tracks stay as they were.)
**UPDATE 2026-06-21 — hop-spring FAIR re-run DONE (+VE but spin-caveated); pervasive SPIN found; CLEAN curriculum built.**
The apex-matched comparison (staged stiffness ramp `scripts/run_spring_ramp.sh` → stable k=106 spring, 3/3 survival)
gives the spring **−4.4 % hop electrical energy at matched 0.13 m apex** (fairness gate FAIR) — the first clean POSITIVE
Part-2 G1 result, gear-INDEPENDENT (ohmic ~1.5 %; energy ~98.5 % mechanical, braking ~14.5 % = regen ceiling, ~0
recovered no-regen). Write-up: **`docs/hop_jump_report.md`**. BUT every hopper this campaign **SPINS** ~+1.8–3.8 rad/s
at zero command (a diagonal stance inherited from S1); **only bounding doesn't** (its forward command breaks the
symmetry) — so the −4.4 % is caveated (the two arms spin at different rates) and the deferred leg-symmetry was the
ROOT cause, not cosmetic. Directional jumps also spin (NEGATIVE — trains by reward but ignores the command).
**UPDATE 2026-06-21 (cont.) — the SPIN is SOLVED.** Clean s1 (new `leg_symmetry -2.0` + strong heading hold,
from scratch, 200M on an immers H100) trained to reward ~80 (vigorous two-footed hop), 3/3 survival, and yaw @
zero command dropped to **0.016 rad/s** (from +1.8–3.8 across the whole prior lineage) — the diagonal-stance spin
that contaminated the hop lineage is FIXED. The pre-flight-added **automated s1 yaw gate** (`scripts/hop_yaw_gate.py`,
wired into the driver) FAILED only on a SECONDARY **0.16 m/s drift** (not spin), so the driver correctly aborted
before s2–s4 (saving ~2 h box time). s1 saved at `outputs/clean_curriculum/2026-06-21_g1_clean_s1_clean_s1/`
(+ `clean_s1_hop.mp4`). **Drift is benign for the in-place energy comparison** (both arms drift equally; s3 trains
it out) — so next session either RELAX the gate to yaw+survival and continue s2→s4 warm-started from this saved s1,
or add a light anti-drift to s1 and re-elicit (`docs/NEXT_SESSION.md`). Then the −4.4 % spring-energy re-run on the
clean base. **ACTIVE-direction infra:** `leg_symmetry` reward (`g1_hop_env.py`, body-frame foot symmetry) +
`configs/g1_clean_s1..s4` + `scripts/run_clean_curriculum.sh` (auto-gates s1 on |yaw|<0.15).
**UPDATE 2026-06-21 (cont. 2) — curriculum DONE (s1-s4); fair spring run BRITTLE → DR-boundary fix.** Relaxed the
gate (drift benign), warm-start-continued s2 (r113) → s3 (r117) → s4 bounding (`run_clean_curriculum_cont.sh`); videos
in `outputs/clean_curriculum/streak_videos/`. s3 yaw tracks both ways but forward translation is s4's job (s4 cmd
0.5/0.7/1.0 → 0.30/0.48/0.74 m/s straight). Work-loop (`hop_workloop_6joint.py`): **knee is the spring joint** (braking
−51 W ≈ 85% of leg; pogo k=93.3, θ_engage 0.701), ankle secondary, others ≈0. Built the FAIR comparison
(`g1_hop_fair_{baseline,spring}.yaml` byte-identical except spring; `run_hop_fair.sh`: baseline + staged ramp
k40→75→93.3) — but **it FAILED: the spring policy trains to r122 under DR yet falls in ~2 s at every local eval
(nominal AND DR, det AND stoch)**, so no valid J/hop. **Root cause:** the stock G1 DR puts the NOMINAL model at the
LEAST-DAMPED boundary (`dof_armature ×U(1.0,1.05)` one-sided → nominal=min; frictionloss skews high), so the policy
is never trained at nominal and the energetic spring over-hops + topples there (baseline's wide margin survives).
**Fix (`d2aab0b`): `pea/randomize.py centered_domain_randomize` + `centered_dr: true`** centers the damping axes →
nominal interior; both fair configs set it. **NEXT: re-run `run_hop_fair.sh` with centered DR (both arms; configs
ready), ideally parallelized across 2 boxes, GATE the spring's nominal survival after k=40, then
`hop_energy_compare.py` → the J/hop number, then render the pair.** Box `…52` UP (s2 present); box `…74` to delete.
**UPDATE 2026-06-22 — in-place jump = NEGATIVE; PIVOTED to RUNNING; full-strength bound spring shows a PROMISING
CoT win (confounded). NEXT = energy-objective training.** The centered-DR fair re-run still FAILED (spring over-hops
0.14 vs 0.10, 0/3); root cause was that the **base hop itself was never stable** (in-place two-footed hop is
inherently marginal, 1/3-2/3 — not the spring/DR). Stabilization v1→v4 fixed the topple (v2 attitude penalties:
tilt 70-126°→15-17°) and added a `hop_stay` in-place anchor (displacement + 0.3 m dead-zone) but plateaued 1/3-2/3.
**Spring on the best base (v2): 0/3, collapses** — so an always-on (no-clutch) spring is incompatible with stable
in-place hopping (the old −4.4 % was a spinning-base artifact). **User-directed PIVOT to running (bounding):** re-fit
knee pogo on s4 bound (k=127.7, braking −59.6 W). Fixed-cadence compare NEUTRAL (−1.4 %, ohmic ~3 %) — but **the
spring's real channel is mechanical braking-recovery (gear-independent), and the energy-optimal cadence may differ
(resonance), so fixed-cadence + energy-OFF handicaps it**. Cadence sweep: no resonance shift, but at **FULL strength
the bound RUNS (vx 0.31-0.82 m/s vs baseline ~0.1) at comparable E/hop → CoT several-fold lower** (the spring's
energy → propulsion). CONFOUNDED: spring 0/3 (runs-then-falls), baseline barely translates, spring trained +100 M.
**DECISIVE NEXT TEST: energy-objective training** — `energy_reward_weight` ON (gait optimizes electrical energy →
exploits the spring), FREE cadence (bound 1.6-2.6, don't fix `hop_freq`), PARITY baseline (s4 + equal steps, no
spring), compare CoT at a matched ACHIEVED speed, both ≥2/3. Results + plan: `outputs/clean_curriculum/fair_centered/
running_spring_results.md`. New infra: `hop_stay` anchor (`g1_hop_env.py`), command-aware `hop_failure_diag`/
`hop_energy_compare`/`hop_spring_prep`, `cadence_sweep.py`, `g1_bound_spring.yaml`+`run_bound_spring.sh`. Box DESTROYED
(s4 warm-start + configs/results saved local); provision fresh tomorrow, upload s4, run the energy-objective bound spring.

## Session ritual
Project memory lives on disk, not in any single conversation. Work in short, task-scoped
sessions and start fresh ones freely (a new session reloads this file automatically).
- **Start** a session: skim `docs/JOURNAL.md` → "Current state".
- **End** a session: run `/wrap` (`.claude/commands/wrap.md`) to update `docs/JOURNAL.md`
  and, if anything changed, this file.

## Experiment design (staged — do in this order)
1. **Baseline**: train a G1 flat-ground "walk straight" policy with **no spring**.
2. **Log**: roll the policy out, record knee angle θ and knee torque τ over the gait.
3. **Post-hoc analysis** (fast, optimistic): subtract `τ_spring(θ)` from the recorded
   motor torque on the *same* trajectory and recompute electrical losses. Upper bound —
   it assumes the gait doesn't change. Use it only to sanity-check that the effect exists.
4. **In-loop** (the credible result): inject `τ_spring(θ)` into the simulation, **retrain**
   the policy so the gait adapts, then compare best-without-spring vs best-with-spring.
   The post-hoc number alone is NOT a sufficient result.

## Metrics — measure ELECTRICAL energy, not mechanical work
- Copper loss `P = (τ/Kt)² · R`. Need approximate motor constants (Kt, R) for the G1
  actuators; approximate is fine for a *relative* comparison.
- Negative work is **dissipated, not regenerated** (no-regen) — JUSTIFIED for the G1
  by back-EMF physics (at locomotion joint speeds the back-EMF is below the ~48 V bus,
  so returning current needs a boost converter commercial drivers lack) + the
  documented regenerative-resistor pattern. NOT a verified spec; exceptions exist (MIT
  Cheetah 2013, reportedly Tesla Optimus). The spring's win lives precisely in this
  dissipated braking, so it is **no-regen-dependent** — report the ~24 % regen-vs-no-regen
  sensitivity. Full treatment: `docs/running_program.md`.
- Headline metric: **cost of transport (CoT)**. Report % reduction in CoT and copper loss.

## Tech stack
- **Simulator: MuJoCo, end to end.** MuJoCo Playground (ships a Unitree G1 flat-walk env)
  + MJX for GPU training; the same MJX env on CPU JAX for local rollout/analysis.
- **Training: rented immers.cloud H100 PCIe** (~342 ₽/hr, per-second billing, SSH —
  Claude drives it end to end: bootstrap via `scripts/gpu_box_setup.sh`, train, rsync
  results into the Drive folder, then DELETE the server, not stop). Measured: ~47k
  env-steps/s with DR → ~70 min per 200M-step run. Colab T4 notebook remains as free
  fallback (~9k steps/s, ~7 h, checkpoint-resume across disconnects).
- **Analysis/inference: local MacBook Pro**, CPU MuJoCo + **CPU JAX** (only GPU/Metal JAX
  is dead; CPU JAX runs fine on Mac).
- **Language: Python.**

## Repo structure (three layers)
- `src/pea/` — shared core, imported by everything:
  - `env.py` (G1 walking env; optionally injects spring/energy wrappers; **registers the custom
    env subclasses at import** so `train.py`'s `ppo_params_for` finds them before `make_env`)
  - dynamic-movement env subclasses: `g1_run_env.py` (`G1JoystickRun`, flight-enabling),
    `g1_hop_env.py` (`G1JoystickHop`, continuous two-footed hopping — ACTIVE Part-2 dir),
    `g1_bound_env.py` (`G1JoystickBound`, alternating-foot bounding/running — built, untrained).
    Custom G1 env names alias to `G1JoystickFlatTerrain`'s PPO config in `policy.ppo_params_for`.
  - `springs.py` (`τ_spring(θ)`: linear + nonlinear/tunable; `one_sided_linear` = hop pogo element)
  - `energy.py` (copper-loss model + cost of transport; `MOTORS` incl. `limx_knee`, `galaxea_torso`, `galaxea_arm`)
  - `policy.py` (network def + load/save), `config.py`
  - gravity-comp direction: `urdf_loader.py` (load EXTERNAL ROS/URDF robots into MuJoCo: package:// + glb→obj),
    `gravcomp.py` (trajectory, torque, electrical energy, spring fit, reduced-model IK), `render_util.py` (offscreen video + scene)
- `scripts/` — `train.py` (Colab), `rollout.py` (local), `analyze.py` (local); G1 hopping/running:
  `run_hop_compare.sh` + `run_curriculum.sh` (SAFE box drivers — nice-19/guards), `hop_spring_prep.py`
  (work-loop + pogo fit), `hop_energy_compare.py` (matched-task energy, FAIRNESS GATE), `render_hop.py`; gravity-comp:
  `galaxea_lift.py`, `limx_roll.py`, `gravcomp_table.py`, `setup_robots.sh` (fetch/convert robots), `collect_report.sh`
- `configs/` — no-spring (`baseline.yaml`, `walk_baseline.yaml`) + spring arms (`spring_hip_linear.yaml`, `spring_linear.yaml`, `spring_constant.yaml`, `spring_semiparabolic.yaml`); G1 hopping (`g1_hop_s1.yaml` elicit, `g1_hop_baseline.yaml` + `g1_hop_spring.yaml` matched pair); gravity-comp: `galaxea_lift.yaml`, `limx_roll.yaml`
- `notebooks/colab_train.ipynb` — thin runner: pip-install repo, mount Drive, call train.py
- `external/` — gitignored; cloned robot descriptions (fiveages + LimX), regenerate with `scripts/setup_robots.sh`
- `outputs/` — gitignored; `gravity_compensation/` (active reporting bundle), `_locomotion_archive/` (suspended-track artifacts), one folder per training run (config, checkpoint, metrics, trajectory)

## Critical design rules
- The spring lives in `src/pea/springs.py` and MUST be callable two ways: as an in-sim
  torque (in-loop training/rollout) AND as a standalone `τ_spring(θ)` function (post-hoc).
  This single decision lets the same code serve both halves of the experiment.
- The spring is **config-selected, not baked into `train.py`** — the same script produces
  the baseline and spring-active runs by swapping a config (keeps them on identical paths).
- All real logic lives in versioned `.py` files; the Colab notebook is only a thin runner
  that `pip install`s the repo. Never let the notebook hold logic.

## Workflow / data flow (no manual downloads)
Edit in VS Code → `git push` to GitHub → Colab `pip install`s the repo + runs `train.py`
→ writes a run folder (config, checkpoint, metrics, trajectory) to **Google Drive** →
"Google Drive for desktop" mirrors it to the Mac → `analyze.py` reads the synced folder.

## Constraints & gotchas — what to AVOID
- **GPU-box autonomous runs — REAL MONEY AT STAKE (see README "GPU-box safety").** A busy-loop driver
  (a shell `stage` fn that ended with `log`, so it returned 0 and `|| break` was dead code) spun a failing
  `pea-train` fast enough to **starve sshd and wedge a still-billing box** — unkillable over SSH; the instance
  had to be destroyed from the immers console. MANDATORY for any box-side loop / unattended run: run launchers
  at **`nice -19 ionice -c3`** (sshd then never starves → box stays reachable); the loop must capture the REAL
  exit code, enforce a min stage duration + run-dir check + iteration cap + **`sleep 60` on failure**;
  `git reset --hard origin/main` + `uv run python -c "from pea.env import make_env"` BEFORE launching (never
  trust a flaky-link transfer); watch the first eval; run an independent watchdog. The safe drivers are
  `scripts/run_hop_compare.sh` / `run_curriculum.sh`. (macOS has no `timeout` — use ssh `-o ConnectTimeout`.)
- **Gravity-comp direction — MuJoCo `mj_inverse` is unreliable** on the reduced torso model: it returned a
  large SPURIOUS base-joint torque (640 vs 15 N·m, verified against the explicit equation of motion). Use
  `M·q̈ + qfrc_bias` (`gravcomp.multi_joint_torque`), NOT `mj_inverse`.
- **Loading external URDFs into MuJoCo** (`urdf_loader`): resolve `package://`, convert `.glb`→`.obj` (MuJoCo
  decodes only STL/OBJ/MSH; `setup_robots.sh` pre-converts), and compile with `discardvisual="false"` (default
  true drops visual meshes AND any added skybox/floor textures → dark renders) + `fusestatic="false"` (default
  fuses welded link bodies, removing IK target bodies like `torso_link4`). **Same-stem meshes across packages
  COLLIDE** — MuJoCo keys meshes by filename stem, so the Galaxea A1-arm and R1-base packages both shipping
  `base_link.obj` made the arms render a second base ("two bases at the shoulders"); `resolve_urdf` now symlinks
  each mesh to a package-qualified name. Render-only: body MASSES come from URDF `<inertial>`, not meshes.
  **Read masses by the RIGHT body name** — the W1 root is `base` (18.19 kg), not `base_link`; `mj_name2id` returns
  −1 for a missing name and `body_mass[-1]` then silently returns the LAST body's mass (a 0.8 kg wheel). The W1
  is ~43.5 kg in the URDF (base 18 + legs 25); don't "add a torso" to a misread near-zero base. **`.obj` meshes
  carry no material** → flat grey in MuJoCo; `render_util.apply_palette` colors parts by body name for renders.
  URDF full-extension pose is an IK singularity → seed the IK bent. Velocity-servo wheels micro-slip on soft
  contact → bill rolling-resistance transport analytically (Crr·m·g·distance), not the servo actuator force.
- **Always `impl: jax`** (a RunConfig field): Playground 0.2.0 defaults the env to
  MuJoCo Warp, which is broken on Mac and was never validated for this project.
- **jax pinned `<0.10`**: brax 0.14.2 (latest) calls `jax.device_put_replicated`,
  removed in jax 0.10. Revisit when brax releases.
- **On the Mac, prefix every venv command with `env -u PYTHONPATH`** — the shell
  profile sources a ROS 2 workspace whose PYTHONPATH shadows venv numpy/jax.
- **Drive mount is locale-named** («Мой диск», not "My Drive") — `config.py`
  handles it; don't hardcode the English name anywhere.
- **GPU box over VPN**: SSH banner-exchange timeouts = flaky VPN exit; bypass the
  box IP (`route add -host <ip> <gateway>`) or switch VPN location. Always launch
  remote training detached (`nohup … &`) so it survives drops; monitor
  `metrics.jsonl` (per-eval append), not stdout (block-buffered under nohup).
- **NO Isaac Sim / Isaac Lab** — they need an NVIDIA RTX GPU + CUDA; dev machine is a
  MacBook Pro (no CUDA). This is the reason we're MuJoCo-only.
- **Don't rely on jax-metal / GPU MJX on the Mac** — jax-metal is abandoned. CPU only locally.
- **Colab free tier**: T4 GPU (allocation is a lottery), **ephemeral disk** — always mount
  Drive and save checkpoints there or you lose the run. MJX JIT compile ≈ 1–3 min/session.
- **Don't** put code on Drive (loses version control) or model binaries in Git (bloats repo).
- **Don't** train on Mac CPU except to smoke-test the pipeline (too slow for a 29-DoF humanoid).
- The spring is **always engaged**, so it resists the motor in phases where its torque is
  unwanted. Tuning (rate, equilibrium angle, nonlinearity) is a real optimization — ideally
  co-optimize spring params with the policy. (This is why some hardware PEAs add a clutch.)
- First demo: one sensibly hand-tuned spring curve is enough; parameter sweeps come later.

## Background / framing
The G1 ships with stiff quasi-direct-drive actuators and **no springs**, making it a clean
testbed for adding parallel elasticity. (Even Agility's commercial Digit V4 appears to have
dropped the Cassie-lineage leg leaf springs for a rigid leg — distal passive compliance pays
off most in running/hopping, less for precise manipulation.) The project is essentially
testing whether *targeted parallel elasticity is worth the added mechanism and control
complexity* for walking efficiency.

## Session ritual (end of every work session)
Keep the project's memory on disk, not in chat. Before closing a session:
1. Append a dated entry to `docs/JOURNAL.md` — what you did, what you decided and why,
   what's open/broken, and the single next step.
2. If any decision, convention, constraint, or file-structure choice changed, update this
   `CLAUDE.md` to match.

Claude: when a session is wrapping up, or the user signals they're done, proactively draft
the `docs/JOURNAL.md` entry and point out any updates this file needs.
