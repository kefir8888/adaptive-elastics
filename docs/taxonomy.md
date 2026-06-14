# Cross-morphology taxonomy — where parallel elasticity pays

Blueprint for a comparative parallel-elastic study across **dynamic** legged robots,
split by leg count. Built 2026-06-14 from the MuJoCo Playground / Menagerie inventory
(local) + a verified classification workflow (`wf_dee7acba`). Restricted to robots
**capable of running / highly dynamic movement** — the slow harmonic walkers
(Apollo, OP3, TALOS, Fourier, legacy Valkyrie/HRP) are excluded.

## The organizing insight: the spring's dominant benefit SHIFTS with gear ratio

| actuation class | gear | ohmic share | spring's dominant benefit |
|---|---|---|---|
| **LOW-gear QDD** | ~6–10:1 | 25–76 % | **ENERGY** (ohmic + braking), ~8–17 % CoT |
| **MID-gear QDD** | ~22:1 | ~4 % | marginal energy (braking-recovery, ~3 %) |
| **HIGH-gear harmonic/SEA** | 80–160:1 | negligible | **WEAR / peak-load + runtime** (not CoT) |

A single study maps the **benefit-vs-gear landscape**: energy at low gear, wear at high
gear, the G1 at the honest middle. The **max-loads (peak + RMS torque)** metric is the
headline for the high-gear class. No prior work maps this across morphologies with
battery-electrical *and* wear accounting.

**An asymmetry the dynamic filter exposes:** **dynamic bipeds are all low/mid-gear**
(high-gear harmonic humanoids can't run — joint speed too low — so they're excluded),
whereas **dynamic quadrupeds span the whole gear range** (SEA/clever transmissions let
ANYmal and Spot stay dynamic at high gear). So the **wear story lives mainly in
quadrupeds**; the biped story is energy.

---

## Table 1 — Quadrupeds ("dogs"), dynamic-capable

avail: ✓ ready Playground env · ◑ Menagerie model (needs an env) · ○ external model
(robot_descriptions / URDF, needs an env)

| robot | gear | class | avail | spring joint | benefit |
|---|---|---|---|---|---|
| Unitree Go1 | 6.33:1 QDD | LOW | ✓ flat+rough | thigh + knee | **energy** |
| Google Barkour | 6:1 QDD | LOW | ✓ | thigh + knee | **energy** (small — scale check) |
| Boston Dynamics Spot | ~80–160:1 harmonic | HIGH | ✓ | thigh + knee | **wear** / peak-load |
| ANYmal C | 100:1 SEA | HIGH | ◑ Menagerie | knee (Bjelonic) | wear + torque² |

Quadruped placement = **thigh (hip-pitch / HFE) AND knee — plot both.** The knee is the
prior-art default (Bjelonic/PIL), but the thigh also carries sagittal gravity-support, so
let the data choose (as for the biped pitch trio). **Dropped:** MIT Mini-Cheetah (the
ancestor of this QDD class — Go/Barkour inherit its ~6:1 serial architecture, so it adds
no coverage) and Unitree Go2 (near-identical to Go1: 6.22 vs 6.33:1, same motor — use
Go1's ready env + Go2's teardown Kt≈0.26). The QDD class is homogeneous → one rep suffices;
Spot/ANYmal are the high-gear outliers (the wear story).

## Table 2 — Bipeds (humanoids), dynamic-capable

| robot | gear | class | avail | kinematics | spring joint | benefit |
|---|---|---|---|---|---|---|
| Berkeley Humanoid | 9:1 QDD | LOW | ✓ flat+rough | **fully serial** (motor/joint) | hip-pitch, knee | **energy** (8–17 %) |
| Unitree G1 | 22.5:1 QDD | MID | ✓ flat+rough | serial hip/knee + **∥ ankle** | hip-pitch (knee=const) | marginal energy |
| Unitree H1 | custom servo (?) | MID? | ✓ | **serial** (1-DoF pitch ankle) | hip-pitch | energy (larger abs) |
| Booster T1 | custom (?) | ? | ✓ flat+rough | serial hip/knee + **∥ ankle** | hip-pitch, knee | energy/wear |
| Agility Cassie | ~10:1 SEA | LOW | ◑ Menagerie | **∥ 5-bar + leaf** | leg-length (already sprung) | running |
| DecART (MIPT) | ? QDD | ? | ✗ no model | **∥ decoupled** | **leg-length** | running (SLIP), no clutch |

Biped placement = **hip-pitch** (AC joint, linear spring; STEPPR + G1). The biped
**knee is offset-dominated** → its optimal passive element is a **constant-torque
preload**, not a torsion spring. DecART → **leg-length axis** (monotonic stance, no
clutch). Cassie already has distal leaf springs — don't re-spring; study at the hip.

**Kinematics note (important correction).** "serial" here describes the MuJoCo *model*
(open chain of revolute joints). In *hardware*, **G1 and Booster T1 both drive the ANKLE
via a 2-DOF parallel mechanism** (Stewart-platform-style: 2 proximal motors + 2 pushrod
linkages, to cut distal inertia) — the common modern-humanoid ankle. The Menagerie models
abstract this into 2 serial ankle joints (no linkage, `neq=0`), which is fine for
HIP/KNEE spring studies (our targets) but would NOT faithfully capture an ANKLE spring on
the parallel mechanism — that needs the linkage added (equality constraints). The real kinematic
split (verified): **truly serial** = Berkeley Humanoid, Unitree H1 (1-DoF pitch ankle) ·
**parallel ankle** (2-DoF Stewart-style, 2 motors + 2 pushrods) = G1, Booster T1, H1-2 ·
**full-leg parallel** = Cassie (5-bar + leaf), DecART (decoupled). The parallel-ankle
class is the common modern dynamic-humanoid design.

**Excluded (not dynamic / can't run):** Apptronik Apollo (warehouse walker, 100–160:1
harmonic+SEA), Robotis OP3 (hobby Dynamixel ~200–350:1), PAL TALOS, Fourier N1, legacy
Valkyrie/HRP. All high-gear harmonic → joint speed too low to run.

---

## Robot models beyond Menagerie — what's available

Almost everything dynamic we care about HAS a model; only DecART is missing.

- **MuJoCo Menagerie** (curated MJCF, best quality): G1, H1, Go1, Go2, Spot, Berkeley
  Humanoid, Booster T1, Cassie, ANYmal B/C, A1, + the excluded Apollo/OP3/TALOS.
- **`robot_descriptions.py`** ([185+ models](https://github.com/robot-descriptions/robot_descriptions.py),
  URDF + MJCF, `pip install`, loads into MuJoCo; also wraps Menagerie) — adds
  **MIT Mini-Cheetah** and many more.
- **Manufacturer ROS URDFs** (GitHub): `unitree_ros` (G1/H1/Go — we used the G1 one for
  the velocity limits), anybotics, agility.
- **IsaacLab / IsaacGym assets**: ANYmal, Go2, H1, G1, Spot.
- **URDF → MJCF**: any URDF compiles to MuJoCo (the URDF importer / `mujoco-urdf-loader`).
- **No public model (build from the paper):** **DecART** (MIPT, Nov 2025 — pantograph
  decoupled leg; no code/URDF release found).

**Effort tiers for a non-ready robot:** ◑ Menagerie/○ external model → write a Playground
locomotion env wrapper around it (moderate, days). ✗ DecART → model from the paper CAD
+ env (the real cost; the parallel-kinematics frontier).

## Experiment matrix, count, cost

Reuse: per robot, train one energy-aware **walk** baseline → warm-start **run** finetune.
Jumping is mostly NOT a parallel story (knee speed-limited) — performance sidebar only.
Metrics (all from one rollout): thermal/ohmic, total electrical (no-regen), CoT, **peak +
RMS torque (gearbox wear)**.

Spanning set (ready envs, no model-building): bipeds **Berkeley, G1, H1, Booster T1**;
quadrupeds **Go1, Barkour, Spot** → covers LOW/MID/HIGH gear × serial/∥-ankle × 2-leg/4-leg.

| tier | what | runs | cost |
|---|---|---|---|
| **0 — post-hoc screening** | ~7 robots × (walk + run), 1 seed; per-joint post-hoc (free CPU) | ~14 | **~£35–50** |
| **1 — in-loop gates** | top ~8–10 cells × 3 seeds + baseline seeds | ~40–50 | **~£120–160** |
| **total** | comprehensive cross-morphology study | **~55–65 runs** | **~£160–210** (~16–21 k ₽) |

(~£3 / ~325 ₽ per 200 M-step H100 run; warm-start finetunes ~half.) Tier 0 alone gives
the whole picture. Adding Cassie/Go2/ANYmal/Mini-Cheetah = env-wrapper effort (model
exists); DecART = real modeling effort.

## Paper framing
*The benefit-vs-gear-ratio landscape of parallel elasticity across dynamic legged
morphologies — energy at low gear, wear at high gear, G1 the instructive middle; biped
hip-pitch vs quadruped knee — with honest electrical (no-regen) + gearbox-wear
accounting, walk controllers reused for running.*

## Data-quality flags
H1 gear unconfirmed (custom servo, 360 N·m peak); Berkeley 9:1 medium-confidence (9.1:1 is HECTOR);
Spot/Booster/DecART gear ratios undisclosed; Bjelonic robust numbers = +33 % torque-square,
−30 % peak, +11 % runtime.
