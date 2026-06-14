# Directions — where parallel elasticity pays, and what we will try

Strategic map produced 2026-06-14 from (a) measurements on our own G1 baseline
trajectory and (b) two multi-agent research workflows (gear ratios, prior art,
energetics across platforms). It assesses **six** candidate regimes for the
parallel-elastic energy thesis and records the **go/skip decision** for each.

## The reframe (why this map exists)

On the G1 *walking*, the headline lever is weak: the actuator is **high-geared
(22.5:1)**, so ohmic loss is only **~4 %** of the motor budget and the post-hoc
hip-pitch spring saves **~3 %** whole-body electrical — and **~0 %** if the robot
regenerates (the win is recovering negative/braking work that no-regen dumps as
heat, not the quadratic copper offload the original framing implied). The thesis
is strongest where ohmic dominates (**low gear**) or where mechanical power → 0
(**static torque**) or where braking energy is large (**running**). The six
directions below are the ways to escape the high-gear-walking corner.

## The bifurcation — two almost-separate directions

The project has split into two directions that share the adaptive parallel spring
but little else (different motions, metrics, spring roles, even favourable
platforms). The crosscutting question is whether the motion is **repetitive
(cyclic)** — because energy recovery and cost-of-transport only mean something
across cycles.

**Track A — efficiency of REPETITIVE motion (Part 1, where we started).** Walking,
running, hopping: the spring offloads torque (lower ohmic) and recovers energy each
cycle. Metric: cost of transport / electrical energy. Gear-limited on the G1 (~3 %
walking; larger for running/hopping but still modest). Parallel is the right
architecture.

**Track B — EXPLOSIVE moves (Part 2).** Jumps (vertical height, broad-jump
distance), drop-landing, sprint starts — which may or may NOT be repetitive: a
single max jump is one-shot (no cross-cycle recovery; CoT does not apply),
continuous hopping/pronking is cyclic (recovery AND amplification both apply). Two
sub-cases with OPPOSITE verdicts on the high-geared G1:

- **Jump HEIGHT / top speed** — set ENTIRELY by which wall takeoff hits (series
  adds speed → helps iff speed-limited; parallel adds force → helps iff
  torque-limited). **RESOLVED from the real Unitree specs** (model jnt_actfrcrange +
  G1 URDF velocity limits): the **knee is SPEED-limited** (139 N·m / only **20
  rad/s**; the walker already hits ~52–67 % of that speed) and the **hip is
  TORQUE-limited** (88 N·m / ~11 % of its 32 rad/s). The knee drives jump extension
  and it is speed-capped → **a parallel spring cannot raise G1 jump height**; height
  points off the G1 (series / low-gear). Parallel still helps the hip (torque) and
  the efficiency/landing case (B2).
- **Efficiency / peak-load of explosive moves (the defensible G1 case).** Even with
  height capped, the spring cuts the ENERGY and PEAK LOAD of each jump and landing.
  Parallel is architecturally correct here, and the gear speed-cap does NOT apply
  to DROP-LANDING (load is set by impact velocity, not motor speed → −10–20 % peak
  actuator load on quadrupeds). Jump/landing torques (≥139 N·m) dwarf walking, so
  both the ohmic (∝ τ²) and braking-recovery channels are far larger than the ~3 %
  walking lever. **This is the strongest explosive story on the stock G1.**

Platform implication: Track A and the explosive-EFFICIENCY half of Track B stay on
the G1 (parallel, hardware-minimal). Maximum jump HEIGHT points OFF the G1 — to a
low-gear quadruped (Go2, +8.8 % measured) or the DecART leg-length axis (closest to
a series catapult; highest upside if its gear <12:1).

- **The metric set is two-dimensional.** Efficiency: electrical energy / CoT /
  ohmic %. Performance: peak mechanical power, peak GRF/torque, takeoff velocity,
  jump height, top speed. The harness (`pea-sweep`) reports both columns.
- **Series vs parallel maps to the WALL, not to the task:** series amplifies
  *speed* (helps iff speed-limited; it never adds torque), parallel adds *force*
  (helps iff torque-limited; it never beats the speed wall). Height, sprint, and
  landing each fall on one side only once you know which wall binds — so the whole
  Part-2 question reduces to measuring the wall. Landing is the exception that is
  unambiguously parallel: its load is set by impact velocity, not motor speed.
- **The tunable spring + dead-zone clutch is the knob between the two axes**, and
  this is a correctly-scoped (between-conditions) use of the clutch: engage a stiff
  spring for an explosive jump/sprint, disengage it (dead zone) for precise or
  efficient walking — "spring for the sprint, off for the walk." It adds explosive
  tasks (jump, sprint, accelerate) to the task axis and strengthens Direction 2
  (running) and Direction 6 (leg-length axis — the natural home for a jump spring).

## Supporting measurements (this session, baseline trajectory, est. Kt=2.3 R=0.013)

**Motor budget, 10 s steady walk (`scripts/motor_budget.py`):** whole-body motor
electrical ≈ **178 W** (no-regen); mechanical ~96 %, ohmic ~4 %. Per-joint share:
right knee 20.6 %, left knee 16.1 %, **right+left hip-pitch 27.3 %**, shoulders
~13 %, rest small. The two hip-pitch motors are the spring's target; the four
leg-pitch joints (knees + hip-pitch) are ~64 % of the budget.

**Actuation share of total robot power (research workflow):** G1 battery 421 Wh,
~2 h mixed-use → ~210 W average; steady walking ~250–350 W (cf. Cassie ~300 W,
ANYmal ~280 W). House load (Jetson Orin NX 10–25 W + Livox 6.5 W + RealSense ~2 W
+ standby) ≈ **40–50 W**. So **actuation ≈ 80–90 % of total while walking**, but
the fixed ~45 W house load is the majority when standing. The spring's ~5 W motor
saving is **~2 % of whole-robot power**, and ~0 % under regeneration.

**No-training probes (`scripts/probe_speed_hold.py`):** see table in JOURNAL
2026-06-14. Standing: walking-tuned spring saves ~0 W (Direction 4 confirmed
inert on G1). Faster walking (1.23 m/s): ohmic share stays ~3.8 %, braking grows
46→63 W, spring gain 5→6 W — speed alone does not arm the copper lever, and the
1 m/s walker destabilises when commanded to 2.0 m/s (Direction 6 needs a
purpose-trained running policy).

## The six directions

| # | direction | payoff | whole-body electrical saving | cell | decision |
|---|---|---|---|---|---|
| 1 | In-loop G1, walking | MEDIUM | 3–6 % | **open** (humanoid + in-loop RL + electrical) | **TRY — the gate, do first** |
| 2 | Running, humanoid (G1/H1) | MEDIUM | 5–15 % (more only with a *triggered* clutch — not our passive dead-zone) | partly occupied | **TRY — needs a running policy** |
| 3 | Low-gear humanoid | MEDIUM | 8–17 % at 8–10:1 | open | **TRY, qualified** (research platforms only) |
| 4 | Manipulation / static hold | HIGH physics, but trivial | ~75 %/joint static; ~0 whole-task on G1 | crowded (exos) | **SKIP** (nothing surprising) |
| 5 | Quadrupeds, slopes × loads | HIGH physics | 8–15 % | occupied (Bjelonic 2023, PIL 2026) | **TRY — zero-shot, no per-condition retrain** |
| 6 | DecART / parallel kinematics | LOW→MED (gear-dependent) | 2–5 % (8–15 % if gear<12:1) | open but thin | **TRY — experimental only** |

### 1 · In-loop G1 (the gate) — TRY, first
The one unclaimed cell (commercial humanoid + true in-loop RL co-adaptation +
electrical accounting). Cheap (~700 ₽). Converts the post-hoc estimate into the
**post-hoc-vs-in-loop delta** — the methodological contribution. Expect modest
(3–6 %); a conservative spring can't lower the 96 %-mechanical net work on a fixed
gait, only the gait change + braking recovery can. **Next:** Milestone 4 as
designed (calibration sweep → hip-linear vs matched no-spring, ≥2 seeds).

### 2 · Running on a humanoid (G1 / H1) — TRY (needs a running policy)
Attractive on **energy magnitude, not on our clutch**: running's landing/braking
work is several times walking's, and RL can adopt spring-mass *bouncing* gaits a
walker cannot (BirdBot −38.8 % CoT; spring-mass running returns ~59 % of stride
energy). Our own data tempers the copper story — ohmic share stays ~7–11 % even at
2–3× torque on the G1 gearing — so the win is the larger braking energy and the
gait change, not the quadratic term. **The running swing-fight is NOT solved by
our passive dead-zone clutch** (see "On the dead-zone clutch" below): freeing in
swing and engaging in stance is within-stride phase gating, which a static
angle-keyed dead zone cannot do while the hip's stance/swing ranges overlap.
BirdBot's −90 % came from a *triggered* bistable clutch — a different mechanism.
So running needs either a tolerable always-engaged spring or a genuinely triggered
within-stride clutch. The probe confirms a sped-up walker can't fake it — a
**purpose-trained running/bounding policy** is required. Best target: **H1**
(3.3 m/s, 47 kg, 360 N·m peak) or G1. Differentiator vs ATRIAS/MABEL/Cassie/
BirdBot: parallel + tunable + RL + electrical on a commercial humanoid.

### 3 · Low-gear humanoid — TRY, qualified
Physics is sound (8–10:1 → ohmic ~25–35 % → ~8–17 % achievable), but the truly
low-gear humanoids are **research platforms, not products**: Berkeley Humanoid (9:1),
HECTOR (9.1:1), MIT Humanoid, NING (10:1). Per the project owner: Berkeley
Humanoid is fast and hard to break, **not** built for thermal efficiency — so it
is a research-platform comparison, not a product story. The G1 (22.5:1) is
already near the *low* end of **commercial** humanoids (most are harmonic 100:1+).
Model is in MuJoCo Menagerie if we want a sim comparison. **Lower priority** than
1/2/5.

### 4 · Manipulation / static holding — SKIP
At ω≈0 all electrical is ohmic, so a spring offloads ~75 % of a joint's *hold*
cost — but absolute watts are tiny on the high-geared G1 (~6.6 W/joint at 60 N·m)
and the probe shows the walking-tuned spring saves ~0 W standing. "With static
load there is nothing surprising." Cell is crowded by exoskeletons/STEPPR/Geeroms.
**Dropped from the go-forward plan.**

> **Paper line (banked):** *"A force that must be held constant and indefinitely is
> the job of a shelf, not a robot."* The static-holding application falls outside an
> efficiency study because the task itself does not warrant a robot — and so cannot
> warrant a spring on one. The elastic element pays only where the load is *dynamic
> and time-varying* (the braking-heavy, cyclic regime of locomotion). Formal form:
> "Sustained, quasi-static load support is better borne by a passive structure than
> by an actuated joint; we restrict attention to dynamic loading, where elastic
> storage and return can pay."

### 5 · Quadrupeds, slopes × loads — TRY (zero-shot, no per-condition retrain)
Highest raw physics payoff (QDD 6:1 → ohmic 39–76 %, copper lever fully armed),
but Bjelonic 2023 (ANYmal: +33 % torque-square, −30 % peak torque, +11 % runtime)
and PIL 2026 (Go2 sim) occupy the basic cell. **Our differentiator:** a **single
robust policy conditioned on the spring parameters (K_eff, θ₀) and on
slope/load, generalising ZERO-SHOT** — no per-condition iterative RL retraining
(Bjelonic used design-conditioned RL + Bayesian Optimization *with* retraining;
PIL distils a spring). If one policy handles a range of spring settings and
terrains zero-shot, the outer spring-parameter optimisation becomes cheap and the
method is materially distinct. **This is where the dead-zone clutch earns its
keep:** "spring off → free joint" is one of the conditioned settings, so the
weak-dominance guarantee holds per condition — the adaptive spring can never make
a slope/load worse than the no-spring baseline (`mechanism.md` §4). Best platform:
**Unitree Go2** (6.22:1, teardown'd, Kt≈0.26) — accessible and the exact low-gear
regime. Vary slope and payload.

### 6 · DecART-like / parallel kinematics — TRY, experimental only
DecARt Leg (MIPT, arXiv:2511.10021): 6 proximal servos, decoupled pitch +
leg-length, springs flagged as the authors' own future work. The appeal — a
parallel spring on the **leg-length axis**, where stance load is one-signed, so
no clutch and no stance/swing overlap (unlike the G1 hip) — and the one place a
*static* dead zone could gate within a stride (positions separate; see clutch
note below). Two caveats: the
leg-length load is **biphasic in magnitude** during walking (double-bump GRF), so
a fixed spring mismatches the mid-stance dip; and the actuator **gear ratio is
unpublished** — if high-gear, the ~4 % ohmic ceiling applies and the saving is
~2–5 %; only if <12:1 does it reach ~8–15 %. The SLIP clean-return story holds
for *running*, linking this to Direction 2. "Only experiments can show" — so a
sim model is the way to settle it. **Next:** obtain/build a DecART-style MJX leg,
confirm gear ratio, put a leg-length spring on it.

## On the dead-zone clutch — precise scope

The semiparabolic spring's distinguishing capability (full passive disengagement
to a free joint, `mechanism.md` §4) is a **between-conditions** setting, not a
**within-stride** action. The two servos set the onsets between strides and hold;
within a stride the element is passive.

- **What it gives (the real prize):** "spring fully off → free joint" — and any
  stiffness/equilibrium between — is always a reachable static setting. So an
  adaptively-tuned spring is **weakly dominant over the no-spring baseline**: for
  any speed/slope/load it can fall back to free, so it can never do worse than
  no-spring. This powers **Direction 5** (one policy conditioned on spring params
  across conditions — "off" is always an option) and the Direction-1 robustness
  sweep.
- **What it does NOT give:** within-stride phase gating (engage in stance, free in
  swing). A static angle-keyed dead zone can do that only if stance and swing
  occupy **non-overlapping** ranges; at the hip they overlap (our data), so it
  cannot — in walking or running. Running's swing-fight therefore needs a
  *triggered* clutch (BirdBot's bistable) or fast within-stride servos — **not**
  our passive dead zone.
- **The one within-stride exception:** the **leg-length axis** (Direction 6),
  where stance (extension/compression under load) and swing (shortening for
  clearance) separate in *position* — there a static dead zone (or a plain
  one-signed spring) can disengage in swing without a clutch. So the within-stride
  story belongs to DecART, not running.

## Gear-ratio reference (verified)

| class | platform | gear ratio | type |
|---|---|---|---|
| LOW (copper-loss-dominated, lever armed) | MIT Cheetah 2 / 3 | 5.8:1 / 7.67:1 | QDD planetary |
| | MIT Mini-Cheetah, Unitree Go1/Go2 | 6:1 / 6.33 / 6.22:1 | QDD |
| | Berkeley Humanoid | 9:1 | QDD planetary |
| | HECTOR | 9.1:1 | QDD planetary |
| | NING humanoid | 10:1 | planetary |
| | 1X Neo / Eve | ~direct-drive (tendon) | near-gearless |
| MID | **Unitree G1 (knee, hip-pitch)** | **22.5:1** (hip-yaw 14.3, ankle 5020) | QDD |
| | Cassie / Digit | ~10:1 cycloidal | SEA, backdrivable |
| HIGH (ohmic negligible) | Figure 01/02/03 | 50–160:1 | harmonic/strain-wave |
| | Apptronik Apollo | 100–160:1 | harmonic + SEA |
| | Tesla Optimus (rotary) | 50–160:1 (partial) | harmonic |
| | ANYmal (quadruped) | 100:1 | harmonic + SEA |
| | HRP-2/4, TALOS, Valkyrie, WALK-MAN | 100–160:1 | harmonic |

**Two corrections to earlier notes:** (a) ANYmal is **100:1** (higher-geared than
the G1), so Bjelonic's win there came via the torque-square + peak-torque +
runtime channel, not a low-gear copper bonanza; the low-gear quadrupeds are the
MIT-Cheetah / Unitree-Go lineage (~6:1). (b) The "−31 % joint electrical" figure
is **STEPPR's hip** (biped, clutched), not Bjelonic — Bjelonic 2023 = +33 %
torque-square, −30 % peak torque, +11 % runtime.

## Go-forward plan (ordered)

1. **Direction 1** — run the G1 in-loop gate now (Milestone 4). Settles the headline
   and the post-hoc-vs-in-loop delta.
2. **Direction 2** — train a G1/H1 running (or bounding) policy, then add the
   tunable spring; appeal is the larger braking energy + bouncing gaits, NOT the
   clutch (our passive dead zone can't gate the within-stride swing-fight).
3. **Direction 5** — Go2 with a single zero-shot spring-/terrain-conditioned
   policy across slopes and loads (no per-condition retrain).
4. **Direction 6** — DecART-style MJX leg with a leg-length spring (experimental;
   confirm gear ratio first).
5. **Direction 3** — low-gear research-platform comparison (Berkeley Humanoid),
   lower priority, sim-only.
6. **Direction 4** — dropped.

## Provenance

- `scripts/motor_budget.py`, `scripts/power_compare.py`, `scripts/probe_speed_hold.py`
  — local analysis on `2026-06-11_baseline_h100/trajectory.npz`.
- Research workflows: `wf_dfc27275-25a` (actuation share, 13 agents),
  `wf_4c572c68-47d` (4-direction assessment, 31 agents, 184 facts),
  `wf_78c14fd0-7f2` (DecART + running, 16 agents). Haiku search/fetch + Sonnet
  synthesis + adversarial verification. Re-run an adversarial search at write-up
  time (fast-moving area; several 2025–2026 preprints).
