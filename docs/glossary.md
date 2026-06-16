# Glossary — plain-language definitions

Project rule: **write plainly — define every term on first use, no unexplained jargon**, in docs,
code comments, and the manuscript. This file is the shared reference. One short definition each;
if a sentence anywhere in the project uses a term below without explaining it, this is where to look.

Symbols used: **τ** (tau) = torque (turning force at a joint), **ω** (omega) = joint speed
(how fast a joint rotates), **θ** (theta) = joint angle, **Kt** = a motor's torque-per-amp
constant, **R** = motor winding resistance.

## Analysis methods
- **Post-hoc / offline analysis (on the recorded trajectory).** A fast, optimistic estimate:
  record one walk, then on paper subtract the spring's torque from the motor torque at each
  recorded instant and recompute the energy — *without* re-running the robot. It assumes the
  gait does not change, so it is an upper bound, not a result. It can even have the wrong sign
  (the G1 went from −3.84 % offline to +7.4 % once retrained).
- **In-loop training (retraining with the spring in the simulation).** The credible test: put
  the spring into the physics simulation and **retrain the control policy from scratch** so the
  gait adapts to the spring, then compare best-with-spring against best-without. This is the
  number the project trusts.

## Energy quantities
- **Mechanical power (τ·ω).** The useful turning power a motor delivers = torque × joint speed.
- **Ohmic power ((τ/Kt)²·R).** Heat lost in the motor's copper windings as current flows
  through them; "ohmic" because it follows Ohm's law (I²R heating). It grows with torque
  *squared*, so offloading torque to a spring cuts it quadratically. Also called copper loss.
- **Iron (core) loss.** Heat lost in the motor's magnetic core (hysteresis + eddy currents).
  It grows with motor *speed*, not torque, so a torque-offloading spring cannot reduce it. The
  energy model omits it; including it slightly dilutes the percentage savings (see `RESULTS.md`).
- **Electrical power (battery draw).** What the battery actually supplies = mechanical power +
  ohmic power, then clamped to be ≥ 0 (under "no regeneration", below). This is what determines
  battery life and the headline metric.
- **No regeneration ("no-regen").** The engineering assumption that when a motor brakes (does
  negative work), that energy is **dumped as heat, not returned to the battery** — so braking
  power counts as zero saved, never as a recharge. It is a judgment, not a verified spec
  (back-EMF below the bus voltage at locomotion speeds). The spring's energy win lives entirely
  in this dumped braking energy; under true regeneration the G1 win goes to ~0 % (report the
  ~24 % regeneration sensitivity).
- **Cost of transport (CoT).** Electrical power divided by forward speed (watts ÷ m/s) — energy
  per distance travelled. The project's **headline metric**, because the spring lets the robot
  walk faster and raw watts would confound speed; watts must always be speed-matched.

## Springs and clutches
- **Parallel vs series elastic element.** A **parallel** spring sits *beside* the motor and
  shares the joint's load, so it can add force/torque to offload the motor (the project's
  choice). A **series** spring sits *in line* between motor and joint; it cannot add torque (the
  same force passes through it) but it can store energy and release it fast, amplifying speed/power.
- **Constant preload (an "almost-constant" spring).** A spring set up so its torque barely
  changes over the joint's small range of motion — built as a **low-stiffness coil wound up
  (pre-wound)** so it delivers a roughly fixed offload torque, like a watch mainspring. The
  right element for the Go1 knee, whose support torque is nearly constant.
- **One-sided stiffness spring.** A spring that engages only past a set angle in one direction:
  it stores energy as the joint flexes and returns it as the joint extends, and exerts **zero
  force on the other side** — so it does not fight the swing phase. Candidate element for the
  dog-running braking-recovery arm.
- **Clutch / dead-zone clutch.** A mechanism that can disengage the spring (a "dead zone" of
  zero force). Used here as a *between-conditions* switch (spring on for a sprint, off for a
  precise walk), **not** a fast within-stride switch.

## Gait and work-loop terms
- **Braking lobe.** The part of a stride where a joint *absorbs/removes* energy, acting like a
  brake. On a plot of joint torque vs joint angle (a closed loop), it is one side of the loop;
  it is exactly the energy a spring can store and give back.
- **Work-loop.** The closed curve traced by plotting a joint's torque against its angle over one
  stride. Its shape says what kind of spring fits: an offset (whole loop shifted off zero) wants
  a **constant preload**; a tilted loop wants **stiffness** (a linear spring).
- **Flight fraction.** The share of a stride during which **all feet are off the ground** at
  once — the signature of true running (vs walking, where a foot is always down). The dog-running
  experiment gates on a measured flight fraction before spending on the spring arms.

## Training terms
- **Energy zeroing.** Setting the energy-use penalty weight to **0** during a training stage, so
  the policy temporarily ignores energy cost (used to stabilise a stage; it killed the G1 runner,
  so the dog-running plan forbids it during the flight stage).
- **Domain randomization (DR).** Randomly varying simulation parameters across training episodes
  so the learned policy is robust to the real spread. Here, **only the constant preload magnitude
  τ₀ is randomized** (uniform 0 to a payload-dependent cap: 8 N·m at 6 kg, 12 at 10 kg, 14 at
  15 kg); the spring stiffness k and rest-angle θ₀ are **never** randomized.

## Gearing
- **Gear ratio.** How much the gearbox multiplies motor torque (and divides motor speed) between
  motor and joint. A high ratio (e.g. G1 knee 22.5:1) lets a small motor make large joint torque
  at low current; a low ratio (e.g. Go1 6.33:1) means the motor itself supplies more torque at
  higher current.
- **"Gearing is the crux."** The project's main finding: the gear ratio decides whether a spring
  helps. High gear → low current → ohmic heating is a tiny share of the energy bill (~4 % on the
  G1), so offloading torque saves little and the always-on spring just fights the gait
  (**negative**). Low gear → high current → ohmic dominates (~54 % on the Go1), so the spring's
  torque offload pays (**positive**).
