# Literature Review — Elastic Actuators for Legged Locomotion

*Emphasis: adaptive and parallel elasticity, electrical-energy offloading.*

> **Scope and provenance.** This review surveys the prior art on elastic
> actuation for legged robots, organized to seed a paper's Related Work. It
> leads with **parallel** elastic actuators (PEAs) and **adaptive /
> variable-stiffness** mechanisms, because those are the architecture and the
> knob this project uses. It draws on six structured subtopic surveys (parallel
> PEAs; SEA-vs-PEA; adaptive/VSA; clutched/quasi-passive; biological springs;
> RL with compliant actuators). A complementary, narrower **novelty
> assessment** for this specific study lives in [`related_work.md`](related_work.md);
> the two are meant to be read together — that file scores the gap, this file
> situates the field. Numerical claims attributed here to individual papers
> should be re-checked against the primary before being copied into a
> manuscript (one figure — the Plooij & Wisse 2012 ~80 % number — is flagged as
> disputed in `related_work.md`).

---

## Introduction

Adding a passive elastic element to a robot joint is one of the oldest ideas in
legged robotics, and for a clear physical reason: animals run cheaply by
cycling elastic strain energy through tendons rather than by doing all the
work with muscle. The robotics question is *which* compliance, placed *where*,
controlled *how* — and the answer turns out to depend sharply on the actuator's
gearing and on the task. Two topologies dominate. A **series elastic actuator
(SEA)** puts the spring in line between motor and load, so motor and spring see
the same force; the spring decouples motor inertia from the load, enables
clean force control, and can amplify power by storing energy slowly and
releasing it fast ([Pratt & Williamson, IROS 1995](https://ieeexplore.ieee.org/document/525827/);
[Paluska & Herr, RAS 2006](https://biomech.media.mit.edu/portfolio_page/cvsea/)).
A **parallel elastic actuator (PEA)** puts the spring beside the motor, so the
spring torque *adds* to the motor torque; the motor only supplies the
difference, which cuts root-mean-square and peak torque and therefore — because
ohmic (Joule / copper) loss scales as torque squared — cuts motor heat
*quadratically* ([Bjelonic et al., RA-L 2023](https://arxiv.org/abs/2301.03509)).

This project tests the parallel case for *electrical* efficiency: does a
tunable parallel spring reduce the cost of transport (CoT) of legged
locomotion, measured in electrical energy (ohmic loss, no regeneration)? The
core finding — that the answer is set by **gearing** (the win fails on the
high-geared Unitree G1 humanoid but pays on the low-geared Unitree Go1
quadruped) — is foreshadowed throughout the literature: every result below that
reports a large parallel-spring win is on a low-geared, backdrivable, or
ohmic-dominated system, and the systems that build *around* parallel
elasticity (STEPPR, MIT Cheetah's design philosophy) do so by deliberately
keeping the transmission transparent. The recurring practical lesson is
equally consistent: an always-engaged parallel spring resists the motor in the
phases where its torque is unwanted, so the highest-yield designs add a
**clutch** to gate the spring to the load phase, and the most flexible designs
make the spring **adaptive** (tunable preload / stiffness) so one element
serves many operating points.

The sections below cover parallel PEAs and their measured savings; the
series-vs-parallel trade and its classic references; adaptive / variable-stiffness
actuators; clutched / quasi-passive ("dead-zone") mechanisms; the biological
springs that motivate all of the above; and reinforcement-learning approaches
that co-design or exploit compliant actuators. A final section states the gap
this project fills.

---

## Parallel elastic actuators

A PEA places a passive spring mechanically in parallel with the motor so that
the joint torque is the sum of motor and spring contributions. The motor only
supplies the residual, which reduces RMS and peak torque and (since ohmic loss
∝ τ²) heat quadratically. PEAs are at their best for gravity compensation and
torque offloading under near-constant or cyclically-repeating loads. Their
central nuisance — and the recurring theme of this review — is that an
always-engaged parallel spring also *resists* the motor in phases where its
torque is unwanted, which is exactly why many designs add a clutch.

The closest in-domain demonstration is the ANYmal quadruped knee PEA of
[Bjelonic, Lee, Arm, Tateo, Peters & Hutter, RA-L 2023](https://arxiv.org/abs/2301.03509)
(ETH Zürich RSL + TU Darmstadt). A parallel-elastic spring on the knee,
co-optimized with the RL controller via a design-conditioned policy plus
Bayesian optimization, improved "torque-square efficiency" by 33 %, reduced
maximum joint torque by 30 % with no loss of tracking, and gave ~11 % longer
operation time on flat terrain. The authors stress that during *dynamic*
locomotion the benefit is non-trivial precisely because the actuators must
repeatedly work *against* the spring (unlike static gravity compensation) — the
key caveat for the present study, and the one that the G1 negative result later
confirmed.

On bipeds, the strongest electrical precedent is the Sandia **STEPPR** walker
([Mazumdar et al., IEEE T-RO 2017](https://www.osti.gov/pages/biblio/1333717);
[ICRA 2015](https://ieeexplore.ieee.org/document/7139275/)). Parallel springs
at the hip (roll/adduction) and ankle (pitch) were predicted to cut CoT by
~30–50 % and, in tests, reduced dissipated/joint electrical power substantially
per joint (reported ~37 % at the ankle and up to ~94 % at the hip for selected
gaits), with on the order of ~13 % overall walking power reduction. The benefit
was strongly gait-dependent across the three gaits tested (human walking,
human-like robot walking, crouched robot walking) — and, importantly for the
present project, STEPPR's springs were *selectively engaged*, and the platform
was built around highly backdrivable rope transmissions specifically so the
springs could pay off.

The cleanest proof that pure parallel torque offloading saves energy needs no
actuator at all: the unpowered ankle exoskeleton of
[Collins, Wiggin & Sawicki, Nature 2015](https://www.nature.com/articles/nature14288)
(CMU / NC State) placed a light spring in parallel with the calf, gated by a
passive mechanical clutch only during stance, and cut the metabolic cost of
human walking by 7.2 ± 2.6 % while consuming *no* chemical or electrical energy
and delivering *no* net positive mechanical work. For powered legs, the
clutched parallel-elastic actuator (CPEA) of
[Plooij & Wisse, IROS 2012](https://ieeexplore.ieee.org/document/6290722/)
(TU Delft) used a compact electric clutch to engage/disengage a parallel spring;
in a knee-extensor rebounding emulation it reported large cuts in energy and
peak torque, and periodic-hopping simulation predicted further reductions at
optimal stiffness — establishing the *upper bound* for cyclic tasks. (The exact
2012 figures are disputed; see the note in `related_work.md`. The robustly
attested version of the clutched-PEA result is the bidirectional follow-on,
discussed under *Clutched / quasi-passive* below.)

Two results bear directly on this project's Part 2 (explosive / load-carrying).
**SpaceBok** ([ETH Zürich / ZHAW / ESA](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Jumping_space_robot_flies_like_a_spacecraft))
uses an optimized parallel-motion leg with parallel elastic elements that store
landing energy and release it at takeoff; in simulated lunar gravity the springs
roughly halved the energy per jump (jumps to ~1.05 m, >2× hip height). And the
open-loop monopod of
[Yesilevskiy / Remy, IEEE T-RO 2016](https://ieeexplore.ieee.org/document/7782370/)
shows parallel elastic actuation letting the spring carry body-mass/payload
support while the motor supplies only energy input, with total CoT as low as
~0.10 under heavy, constant payload — directly analogous to this project's Go1
load-carrying program. A more modest, realistic in-loop walker number comes from
the **SLIDER** straight-legged biped
([Wang et al., IROS 2021](https://www.researchgate.net/publication/354362624_Improved_Energy_Efficiency_via_Parallel_Elastic_Elements_for_the_Straight-Legged_Vertically-Compliant_Robot_SLIDER)):
optimally tuned parallel elastic elements cut energy consumption by ~15 % (and
peak motor torque substantially) — a useful corrective against the optimistic
75–94 % figures from idealized cyclic emulations.

Finally, the model-based frontier confirms the dynamic case:
[Zhuang, Wang & Ding (arXiv:2503.05666, ICRA 2025)](https://arxiv.org/pdf/2503.05666)
report an energy-aware kinodynamic MPC for a legged robot with parallel springs
achieving a 38.8 % CoT reduction in simulation during high-speed hopping, with
preliminary hardware confirming a 14.8 % energy reduction — a clean
sim-to-hardware reality check (the realized gain is ~1/2.6 of the simulated
one, a discount factor worth carrying into our own projections).

Across all of these, **gearing / backdrivability is the recurring enabling
condition**: STEPPR's backdrivable ropes, ANYmal's relatively low-geared
actuators, the monopod and SLIDER's compliant legs. High-gear quasi-direct-drive
joints have a small ohmic fraction, so the parallel-spring win shrinks —
independently corroborating this project's headline that "gearing is the crux."

---

## Series vs parallel

The load-bearing distinction for this project is the topology. In an **SEA**,
motor and spring see the same torque, so the spring can reduce the motor's
required *power* but not its *torque*; in a **PEA**, the spring torque adds to
the motor's, so it directly offloads *torque* (and hence ohmic loss ∝ τ²) but
cannot move the motor's speed/power operating point. Stated bluntly: a series
spring amplifies speed/power and protects against impact but cannot add net
torque; a parallel spring adds force/torque but cannot beat a speed/power wall.

The classic SEA reference is
[Pratt & Williamson, IROS 1995](https://ieeexplore.ieee.org/document/525827/)
([Williamson MIT thesis](https://apps.dtic.mil/sti/pdfs/ADA299658.pdf)).
Against the "stiffer is better" intuition, deliberately *lowering* interface
stiffness buys shock tolerance, lower reflected/output inertia, more accurate
and stable low-noise force control, and energy storage — at the cost of lower
zero-motion force bandwidth. Their argument that locomotion and other "natural
tasks" do not need high zero-motion force bandwidth is what makes the trade
worthwhile. The power-amplification side is quantified by
[Paluska & Herr, RAS 2006](https://biomech.media.mit.edu/portfolio_page/cvsea/)
(MIT Media Lab): a correctly tuned series spring increases the energy an
actuator can deliver to a mass by up to a factor of 4 and lets it deliver peak
power exceeding the source's limit by ~1.4× — a "catapult" / temporal power
amplification — but it adds *no* static torque; it only moves the operating
point. This is the physics behind "series helps a speed-limited takeoff,
parallel does nothing there."

The torque-offloading (parallel) side is the
[ANYmal PEA result](https://arxiv.org/abs/2301.03509) above: +33 % torque-square
efficiency, −30 % max joint torque, +11 % runtime, with the τ² metric being
essentially this project's ohmic channel on a quadruped knee. The PEA win is
largest exactly when Joule heating dominates the motor budget: in hip-walking
assistance, parallel elements cut peak torque by up to ~31 % and RMS power by up
to ~36 %, and when ohmic loss dominates an optimally designed PEA can cut motor
energy up to ~63 % versus an SEA
([STEPPR / OSTI](https://www.osti.gov/servlets/purl/1333717)). This is the same
ohmic-dominated-regime logic this project uses to explain why the win appears on
the low-geared Go1 (ohmic ~54 %) but not the high-geared G1 (ohmic ~4 %).

Crucially, there is **no universal winner** — the choice is task-dependent.
[Verstraten, Beckerle et al., Mech. & Machine Theory 2016](https://www.sciencedirect.com/science/article/abs/pii/S0094114X16300301)
show that SEA consumes less energy up to certain offset angles and PEA wins
elsewhere; the crossover depends on the task's natural dynamics, the
equilibrium/offset angle, and whether negative work is regenerated. They state
plainly that there is still no definitive answer to which topology is more
energy-efficient for an arbitrary task — it must be evaluated per motion. This
validates the project's "gearing / task is the crux" stance over any blanket
claim.

The two production lineages embody the two answers. The **Hutter / ETH** line —
[StarlETH](https://rsl.ethz.ch/robots-media/starleth.html), the first SEA-driven
quadruped, then ANYmal's ANYdrive (series-elastic on a high-ratio harmonic
drive) — uses series elasticity for force control and impact robustness; the
parallel spring in [Bjelonic 2023](https://arxiv.org/abs/2301.03509) was added
*on top*, for efficiency. The two roles are complementary, not the same spring.
The counter-camp is **MIT Cheetah**, which rejects series elasticity in favor of
quasi-direct drive (QDD):
[Seok, Wang, Kim et al., T-Mech 2015](https://dspace.mit.edu/bitstream/handle/1721.1/126619/IROS.pdf)
and
[Wensing et al., T-RO 2017](https://dspace.mit.edu/server/api/core/bitstreams/53fde66c-bd98-4dd7-a95b-3c9a5d11cf69/content)
use the largest motor with the smallest gear reduction (Cheetah 2, ~5.8:1,
~58 N·m/kg). The low ratio makes the actuator backdrivable and transparent, so
high-bandwidth "proprioceptive" force control is done through motor current, and
impact mitigation matches SEA-equipped quadrupeds *without* any series spring —
the reflected inertia an SEA protects against is instead eliminated at the
source by lowering the gear ratio.

This is the direct intellectual ancestor of the present study's platform
selection. **Why gearing is the crux**, synthesized: a high gear ratio makes the
motor electrically efficient and torque-cheap (ohmic is a small slice), so the
SEA route — protect the geartrain from shock and reflected inertia, control force
through the spring — is the natural fit (ANYmal). A low gear ratio leaves ohmic
loss dominant and the geartrain already transparent (Cheetah / Go1), so there is
little reflected inertia to hide *and* a large τ² ohmic bill to attack — making a
parallel torque-offloading spring the natural efficiency win. SEA and PEA are
answers to *different* bottlenecks: SEA targets force-control fidelity, shock,
and reflected inertia; PEA targets steady-state torque and ohmic energy.

The clean decision rule for explosive moves (Part 2) follows directly. Because a
series spring transmits exactly the motor's force (amplifying only speed/power)
and a parallel spring only adds force (never raising the speed ceiling): a
**torque-limited takeoff → parallel helps; a speed/power-limited takeoff →
series helps and parallel does nothing.** This routes G1 jump *height* (whose
knee extensor is speed-capped) off the parallel spring while keeping
landing/efficiency — where load is set by impact velocity and the channel is
torque-dominated — on it.

Energy storage differs by topology, and the PEA win is no-regen-dependent. An
SEA stores energy in cyclic force-velocity modulation and returns it as power
(catapult), recovered automatically through the motor path. A PEA stores energy
during one half of the joint's angle sweep and returns it during the other,
*directly substituting for motor braking torque* — so its win lives precisely in
the negative-work phases the motor would otherwise dissipate (no-regen). If the
drive can regenerate, the PEA advantage shrinks, and several comparison papers
note that preloading the spring can *cost* net energy unless tuned — which is why
spring rate, equilibrium angle, and clutch timing are a genuine optimization,
not a free lunch.

---

## Adaptive & variable-stiffness

One fixed spring is optimal for exactly one operating point; a fixed parallel
preload that is right for one speed/load/posture fights the motor everywhere
else (precisely this project's G1 "always-on spring, no clutch" negative
result). An adaptive actuator re-optimizes equilibrium and rate as speed,
payload, gait frequency, or task change, and can go compliant for impact then
stiff for precise tracking — matching the "engage for explosive, disengage for
precise" logic. The variable-stiffness-actuator (VSA) literature is the menu of
ways to build that knob.

The canonical adjustable-compliance actuator is **MACCEPA**
([Van Ham et al., VUB, 2006–07](https://mech.vub.ac.be/multibody/topics/maccepa/Actuator-VanHam1.pdf)):
a lever/cam pulls a tension spring, with one servo setting the equilibrium angle
and a second setting pretension, so compliance and equilibrium position are
*independently* controllable; it was implemented in the biped "Lucy." **MACCEPA
2.0** reshaped the lever to give a stiffening (progressive) characteristic tuned
for energy-efficient hopping
([MACCEPA 2.0](https://www.researchgate.net/publication/224557024_MACCEPA_20_Adjustable_compliant_actuator_with_stiffening_characteristic_for_energy_efficient_hopping))
— the closest hardware ancestor of an adaptive parallel preload knob.

A different design philosophy changes stiffness by varying a *lever arm* rather
than spring pretension, which matters enormously for the energy cost of
retuning. **AwAS / AwAS-II**
([Jafari & Tsagarakis, IIT, IROS 2010](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_IROS_2010/data/papers/1088.pdf);
[AwAS-II](https://www.researchgate.net/publication/224252483_AwAS-II_A_new_Actuator_with_Adjustable_Stiffness_based_on_the_novel_principle_of_adaptable_pivot_point_and_variable_lever_ratio))
move the spring's attachment point (AwAS) or the lever's pivot (AwAS-II, giving a
force ratio tunable from ~0 to ∞) — so stiffness can be re-tuned with near-zero
energy cost, without changing the energy already stored. The University of Twente
**vsaUT / vsaUT-II**
([Visser, Carloni & Stramigioli](https://research.utwente.nl/en/publications/the-variable-stiffness-actuator-vsaut-ii-mechanical-design-modeli),
[PubMed](https://pubmed.ncbi.nlm.nih.gov/22256239/)) uses a variable transmission
ratio whose kinematics guarantee a *zero-work* stiffness change: output stiffness
varies independently of both output position and stored potential energy. This
decoupling of stiffness-change work from stiffness value is the central design
principle for cheap online retuning, and it dictates a practical rule for an
electrical-CoT study: **an adaptive parallel preload should be retuned on a slow
schedule (per-leg, per-payload), not per-step**, so the retuning servo's own
ohmic losses stay negligible — matching this project's "adaptive per-leg preload
that scales with payload."

The most direct in-domain quantitative precedent is from this project's own
lab:
[Belov, Erkhov, Khabibullin, Pestova, Satsevich, Osokin, Osinenko & Tsetserukou (Skoltech/MIPT, arXiv:2411.18295, 2024)](https://arxiv.org/pdf/2411.18295)
mount a torsion spring in parallel with a quadruped knee, driven by a worm-gear
servo that adapts *both* the equilibrium angle α₀ and the stiffness μ to the
real-time load. They derive a closed-form optimum for μ* and α₀* that minimizes
ohmic energy E = K·Στ² over a cyclic trajectory (energy quadratic in torque — the
same model this project uses), and in simulation across varied
mass/frequency/amplitude/start-height the spring drops servo energy to roughly
0.15–4 % of the no-spring value, with closed-form parameters tracking the load a
single fixed spring cannot. (Method contrast: that work is analytic optimization
with a fixed PD controller on a leg test-stand; this project's contribution is
in-loop RL co-adaptation on a walking quadruped — see `related_work.md`.)

The discrete cousin of variable stiffness is variable **recruitment**: instead
of continuously retuning one spring, engage/disengage a *bank* of parallel
springs via clutches, analogous to recruiting motor units in muscle. The
**SPEA** (Series-Parallel Elastic Actuator,
[Mathijssen, Vanderborght, VUB 2014](https://www.researchgate.net/publication/266380421_Variable_Recruitment_of_Parallel_Elastic_Elements_Series-Parallel_Elastic_Actuators_SPEA_With_Dephased_Mutilated_Gears))
uses dephased mutilated gears to recruit parallel elastic elements; a clutched
parallel-elastic prototype in the Plooij/Wisse lineage cut actuator energy ~80 %
and peak motor torque ~66 % on knee-extensor rebound tasks. This is the most
direct evidence that engageable parallel elasticity pays for the explosive / load
tasks of Part 2.

For *explosive* moves the *when* of stiffness change matters, not just the value.
[Braun, Howard & Vijayakumar (RSS 2011 / T-RO 2013, "Exploiting Variable Stiffness in Explosive Movement Tasks")](https://www.roboticsproceedings.org/rss07/p04.pdf)
synthesize time-varying torque *and* stiffness profiles by optimal control; on a
throwing/hammer task a tuned time-varying stiffness reaches roughly double the
output velocity of the best *fixed* stiffness, by storing energy then releasing
it at resonance just before release. **Bi-Stiffness Actuation (BSA)**
([Pfanne / Beckerle et al., arXiv:2309.07873, ICRA 2024 Best Paper](https://arxiv.org/abs/2309.07873))
adds a switch-and-hold clutch that gives full control over *when* stored energy
is dumped, matching power-equivalent VSA peak velocity without the long, risky
oscillatory swing-up an SEA needs — reinforcing that a dead-zone clutch is what
makes an always-engaged spring usable for one-shot explosive moves.

The "expensive" end of the design space is the antagonistic / agonist-antagonist
VSA (qbmove / VSA-Cube, Pisa/IIT; the DLR FSJ
[Wolf & Eiberger](https://ieeexplore.ieee.org/document/5980303/)): two motors pull
the output through nonlinear springs, with co-contraction setting stiffness and
differential motion setting position. Powerful but it costs two actuators and
suffers a torque-vs-stiffness-range trade — the design this project's single-servo
parallel preload deliberately avoids. The community has consolidated these into a
taxonomy and a standardized "VSA datasheet" (stiffness range, max torque,
stiffening characteristic, stiffness-change time/energy)
([Wolf et al., T-Mech 2016](https://www.semanticscholar.org/paper/Variable-Stiffness-Actuators:-Review-on-Design-and-Wolf-Grioli/0358a1178cf9c434ed3252f721956efcfe351ecf);
[Vanderborght et al., RAS 2013](https://journals.sagepub.com/doi/full/10.1177/0278364914566515)
— the right checklist for specifying an adaptive parallel preload — splitting
designs into (a) antagonistic, (b) lever-arm / variable-transmission, and (c)
tunable physical spring.

Two regimes therefore exist for closing the tuning loop: **offline co-design** of
spring + policy (fix the optimal spring, simplest hardware — the
[Bjelonic 2023](https://arxiv.org/abs/2301.03509) design-conditioned-policy
recipe) versus **online adaptation** (a servo retunes preload/stiffness as load
changes). This project's Go1 load-carrying program is the online branch — one
blind controller whose per-leg preload scales with payload.

---

## Clutched / quasi-passive

The clutch is what makes a passive parallel spring genuinely useful. An
always-engaged spring fights the gait in the phases where its torque is unwanted
(exactly the project's G1 problem); a clutch (or freewheel / dead-zone) engages
the spring only in the phase where it offloads torque, then disengages so the
limb swings free. The canonical taxonomy is
[Plooij, Mathijssen, Cherelle, Lefeber & Vanderborght, "Lock Your Robot," IEEE RAM 2015](https://cris.vub.be/ws/files/26793095/201504_lockingdevices.pdf),
which splits locking devices into mechanical, friction-based, and singularity
locking, each passive or active.

The headline demonstration is again the
[Collins, Wiggin & Sawicki unpowered ankle exo (Nature 2015)](https://www.nature.com/articles/nature14288):
a mechanical clutch engages the parallel calf spring only while the foot is on
the ground (storing energy in dorsiflexion, returning it at push-off) and
disengages it in swing, cutting metabolic cost 7.2 ± 2.6 % at *zero* electrical
cost and zero net positive work — the strongest "dead-zone clutch"
demonstration in the literature. On a prosthetic knee, the **clutchable SEA
(CSEA)** of
[Rouse, Mooney & Herr, IJRR 2014](https://journals.sagepub.com/doi/abs/10.1177/0278364914545673)
([ICORR 2013 PDF](https://aspirin.media.mit.edu/biomechatronics/wp-content/uploads/sites/8/2013/07/Rouse_et_al_ICORR_final.pdf))
places a low-power clutch in parallel with the motor inside an SEA: engaged, the
device is a pure passive spring tuned to the elastically-conservative region of
the knee torque-angle curve, holding stored energy with almost no electrical
cost — ~70 % less electrical energy than a traditional SEA knee in simulation,
isolating the clutch as the energy-saving element.

The closest published analogue to this project's load-carrying program is the
[Walsh, Endo & Herr quasi-passive load-carrying exoskeleton (IJHR 2007)](https://aspirin.media.mit.edu/biomechatronics/wp-content/uploads/sites/8/2013/07/Walsh-2007_A-QUASI-PASSIVE-LEG-EXOSKELETON-FOR-LOAD-CARRYING-AUGMENTATION.pdf):
no actuators at all — a hip spring, an ankle spring, and a knee variable damper,
each engaged per gait phase — total 11.7 kg drawing only ~2 W (just to modulate
the damper/clutch), transferring ~80 % of a 36 kg payload to the ground during
single-support stance. Phase-scheduled passive elements *sized to the payload*
are exactly what the Go1 program targets. A unidirectional clutch ("ribbon
stop") in the MIT quasi-passive ankle-foot prosthesis (Au & Herr,
[US20130110256A1](https://patents.google.com/patent/US20130110256A1/en))
demonstrates the same "engage only against the load direction" freewheel idea.

For running legs specifically,
[**SPEAR** (Liu et al., IROS 2016)](https://www.researchgate.net/publication/308852888_SPEAR_A_monopedal_robot_with_Switchable_Parallel_Elastic_actuation)
engages a parallel spring only in stance and disengages it in flight, reducing
the required knee-flexing torque to ~1/10 of an equivalent *non-clutching*
parallel-elastic leg of the same kinematics — i.e. the clutch removes the
swing-phase torque penalty an always-on spring imposes, the exact G1 failure
mode. The same ~10× reduction is achieved *purely passively* by **BirdBot**
([Badri-Sproewitz, Sarvestani, Sitti & Daley, Science Robotics 2022](https://www.science.org/doi/10.1126/scirobotics.abg4055)):
a contact-triggered, morphologically embodied clutch self-engages the
parallel-elastic legs at touchdown and a bistable joint disengages them at
toe-off, with no sensor, no electric clutch, and feedforward-only control —
demonstrating bipedal locomotion with only four actuators. BirdBot shows the
clutch can replace feedback control entirely.

The state of the art for *programmable* clutched parallel elasticity is the
**bidirectional clutched PEA (BIC-PEA)**
([Plooij, Wisse & Vallery, T-RO 2016](https://repository.tudelft.nl/file/File_ee63534c-0cf6-4caa-8c26-d59f0135b736))
— a motor plus a parallel spring that can be loaded in *either* direction; in
simulated periodic hopping, stance energy fell from 27.29 J (motor-only) to
6.90 J (spring engaged), a ~75 % reduction, with the key trick being to *preload
the spring in flight* — directly relevant to the Part-2 hopping/running
extension. And the
[Stanford elastic energy-recycling actuator (EERA), Krimsky & Collins, Science Robotics 2024](https://www.science.org/doi/10.1126/scirobotics.adj7246)
([Stanford report](https://news.stanford.edu/stories/2024/03/new-efficient-motor-alternative-next-gen-robotics))
puts a motor in parallel with an *array* of six clutched springs, each gated by
low-power electroadhesive clutches (<0.4 W, 12 Hz, >6 N·m programmable passive
torque): the augmented motor used at least 50 % less power than a bare motor, up
to 97 % in the best cyclic case. Multiple springs let it program an arbitrary
phase-scheduled torque-angle curve — generalizing this project's single tunable
spring.

The clutch-mechanism design menu, synthesized across BIC-PEA, EERA, and "Lock
Your Robot," is a trade of controllability against parasitic power/mass: (a)
electric/friction clutches — fast, programmable, but draw holding power and add
mass; (b) electroadhesive — very low power, high bandwidth, light, but limited
force density; (c) passive mechanical / freewheel / ratchet / bistable
self-clutching — zero electrical power and no controller, but a fixed engagement
schedule. The synthesis for this project is blunt: *every* successful
parallel-elastic locomotion result that does not degrade other gait phases uses a
clutch / dead-zone to time the spring to the load phase. The project's own
finding — an always-on G1 hip spring is +7 % worse in-loop because it fights
swing — is precisely the failure these mechanisms were invented to fix. The
constant-preload Go1 knee win works only because that preload happens not to
fight the gait; a contact-triggered or learned-clutch gate (BirdBot-style
self-clutching, or a policy-commanded clutch state) is the natural next step to
both rescue the G1 case and preload in flight for the hopping extension. The
energy-recovery-in-stance principle is quantified consistently across systems —
~7 % metabolic (Collins ankle), ~70 % electrical (CSEA knee sim), ~75 % (BIC-PEA
hopping), up to ~97 % (EERA cyclic), ~10× torque (SPEAR/BirdBot), ~80 % load
transfer at ~2 W (Walsh) — all recovering the *same* braking (negative-work)
energy a no-regen motor would otherwise dissipate as heat.

---

## Biological springs & running energetics

The mechanistic justification for everything above is biological. Walking and
running save energy by *opposite* mechanisms
([Cavagna, Heglund & Taylor, Am. J. Physiol. 1977](https://journals.physiology.org/doi/10.1152/ajpregu.1977.233.5.R243);
[Cavagna, Thys & Zamboni, J. Physiol. 1976](https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/jphysiol.1976.sp011613)).
Walking uses an *inverted pendulum*: gravitational potential and forward kinetic
energy of the center of mass swap out of phase as the body vaults over a stiff
stance leg, recovering up to ~65 % of mechanical energy at intermediate speeds
with little tendon strain. Running uses a *spring-mass / bouncing* mechanism:
potential and kinetic energy fall *in phase* (both minimal at mid-stance) and are
stored as elastic strain energy in tendons, then returned. This is the core
reason running cycles far more elastic energy than walking — and a first-order
caution for any walking-efficiency spring study.

The canonical template is the **spring-loaded inverted pendulum (SLIP)**,
introduced by [Blickhan (1989)](https://academic.oup.com/imamat/article/88/3/429/7116029)
and analyzed by McMahon & Cheng (1990): a point mass on a massless linear leg
spring, from which the stance dynamics, CoM trajectory, and ground-reaction-force
profile of running animals emerge from a single leg-stiffness parameter. The same
template underlies hopping robots (Raibert) and SLIP-based control — and is the
direct conceptual ancestor of parallel-elastic legged robots.

Tendons are the hardware. The Achilles is the flagship human spring:
[Ker, Bennett, Bibby, Kester & Alexander (Nature 1987)](https://onlinelibrary.wiley.com/doi/10.1111/brv.13002)
estimated it stores ~35 J of strain energy per step at 4.5 m/s, plus ~17 J in the
foot arch, together cutting the metabolic cost of running by roughly half (the
Achilles alone ~35 % mechanical saving). Elastic savings scale with
specialization across species — ~35 % human, ~40 % horse, ~45 % wallaby, up to
~50 % kangaroo — as cursorial animals concentrate muscle proximally and run long,
thin, high-stress distal tendons. The strongest natural case is the **kangaroo**:
oxygen consumption of hopping red kangaroos is roughly *independent of speed*
(Dawson & Taylor 1973), unlike essentially all other terrestrial mammals,
attributed to elastic storage in the gastrocnemius/plantaris tendons with up to
~70 % saving at ~6 m/s
([review, Kram](https://spot.colorado.edu/~kram/kangaroo.pdf);
[Alexander, J. Zool. 1975](https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/j.1469-7998.1975.tb05983.x)).
This decoupling of cost from speed is the signature of a tuned biological spring.

Two principles transfer directly to a parallel-spring robot. First, the
**muscle-tendon decoupling / "active strut"** principle:
[Roberts, Marsh, Weyand & Taylor (Science 1997)](https://www.science.org/doi/10.1126/science.275.5303.1113)
measured force and fascicle length in running turkeys and found the active muscle
fibers stay near-isometric (little length change, little net work) while the
tendon does the cyclic stretch-and-recoil. Economy comes from minimizing muscle
*work* while the spring cycles energy — exactly what a parallel passive spring
buys: the motor holds force at low velocity (low ohmic/work cost) while the
spring exchanges the cyclic energy. Modern in-vivo studies confirm the walk-vs-run
asymmetry quantitatively, with Achilles strain energy rising sharply from walking
to running and nearly ~50 % of total body mechanical energy passing through the
Achilles+arch springs during running stance
([Lai et al., J. Exp. Biol. 2014](https://journals.biologists.com/jeb/article/217/17/3159/12443/Tendon-elastic-strain-energy-in-the-human-ankle);
[Sci. Rep. 2021](https://www.nature.com/articles/s41598-021-84847-w)).

Second, the **catapult (power-amplification)** mechanism is the explosive-move
analog: for one-shot maximal efforts a slow muscle pre-loads a tendon against a
latch, then releases it to deliver power exceeding the muscle's own
force-velocity limit. [Astley & Roberts (Biol. Lett. 2012)](https://royalsocietypublishing.org/doi/10.1098/rsbl.2011.0982)
showed frog plantaris muscle shortening *before* joint movement (loading the
tendon), which then recoils rapidly; the locust hind-leg jump (resilin + cuticle
springs, Bennet-Clark 1975) and the flea are the classics. This is the biological
basis for series/clutched springs in jumping robots, and it cleanly distinguishes
Part-2 explosive moves (catapult, power-amplifying) from Part-1 efficiency
(cyclic recovery).

Two further points close the loop with this project's accounting. The
biomechanical mechanism is that during the braking (energy-absorption) first half
of stance, the leg spring stores energy the muscle would otherwise have to absorb
as negative work, then returns it in push-off — *exactly* the channel this
project relies on, since no-regen motors dissipate braking energy as heat. And
the parameters biology tunes — moment arm, tendon stiffness, posture — are the
same knobs a tunable parallel-elastic robot spring exposes: shorter moment arms
store more energy (Sci. Rep. 2021), and leg stiffness must be matched to
speed/CoM dynamics for the bounce to work (McMahon & Cheng). The practical
lesson — equilibrium angle (preload), rate (stiffness), and effective lever arm,
tuned gait/speed-specifically — matches the project's finding that a constant
preload helps but must be co-tuned
([Blazevich, Biol. Rev. 2023](https://onlinelibrary.wiley.com/doi/10.1111/brv.13002)).

---

## RL with compliant actuators

The methodological frontier is co-designing or learning to exploit compliant
actuators with reinforcement learning. The most relevant analog is again
[Bjelonic, Lee, Arm, Tateo, Peters & Hutter (RA-L 2023)](https://arxiv.org/abs/2301.03509):
a parallel-elastic knee on ANYmal, co-designed with the controller by training a
single design-*conditioned* model-free RL policy over a range of spring
parameters and then Bayesian-optimizing the design using that policy (avoiding a
retrain per design), yielding +33 % torque-square efficiency, −30 % max joint
torque, and +11 % runtime. Their torque-square metric *is* this project's ohmic
channel on a quadruped knee, validating both the mechanism and the
design-conditioned-policy co-design recipe.

Compliance can also be exploited without co-design.
[Raffin, Kober, Albu-Schäffer, Silvério & Stulp (DLR, ICRA 2023, "Learning to Exploit Elastic Actuators")](https://arxiv.org/abs/2209.07171)
take a *series*-elastic cat-sized quadruped ("bert"), synthesize an open-loop CPG
gait, and let RL close the loop with corrective actions on top; it learned
trotting/pronking *directly on the real robot* in <1.5 h with no massively
parallel sim, and spring-exploitation emerged from simply optimizing for dynamic
motion — showing compliant-actuator exploitation is learnable, though on series
(not parallel) elasticity.

On the energy-reward side — directly relevant to building an energy-aware reward
on a Go1 in MuJoCo —
[Liang, Sun, Zhu, Zhang & Xiong (UC Berkeley, "Adaptive Energy Regularization", arXiv:2403.20001)](https://arxiv.org/html/2403.20001)
train PPO on a Unitree Go1 (validated on ANYmal-C) with a *velocity-normalized*
energy reward Rₑₙ = exp(−Σ|τᵢ·q̇ᵢ| / (σₓ|vₓ| + σᵤ|ω_z|)) so the penalty is
comparable across speeds; the policy autonomously discovers speed-dependent gaits
(4-beat walk, 2-beat, trot, fly-trot) and lowers CoT without explicit gait
scheduling. The companion methodological caution is
[Mahankali, Lee, Margolis, Hong & Agrawal (MIT, ICRA 2024, EIPO)](https://srinathm1359.github.io/eipo-locomotion/):
hand-picking the CoT/energy reward weight is brittle, and Extrinsic-Intrinsic
Policy Optimization reaches higher task performance than weight-tuned PPO at
comparable energy on a real quadruped — the principled alternative to single-scalar
reward shaping.

The parallel-spring dynamic case is confirmed model-based by
[Zhuang, Wang & Ding (arXiv:2503.05666, ICRA 2025)](https://arxiv.org/pdf/2503.05666):
a monoped with unidirectional parallel springs under hierarchical kinodynamic MPC
hits −38.8 % CoT in simulation and −14.8 % energy on hardware in high-speed
hopping — confirming parallel springs pay for dynamic vertical motion (Part 2) and
fixing a sim-to-real discount of ~2.5×. Competitive baselines for the metric come
from
[PSTO (Machines 2022)](https://www.mdpi.com/2075-1702/10/3/185)
(lowest CoT ~0.45 trotting across learned gaits, with the energy-optimal gait
being speed/gait-dependent) and the upper-bound torque-offload illustration from
[SLIDER](https://www.researchgate.net/publication/354362624_Improved_Energy_Efficiency_via_Parallel_Elastic_Elements_for_the_Straight-Legged_Vertically-Compliant_Robot_SLIDER)
(~80 % energy, ~66 % peak-torque in analysis/sim — why peak torque, not just
average, is a headline metric).

Morphology/control **co-design** is an established RL subfield, generally framed
as bi-level optimization (outer loop = morphology/spring params, inner loop =
control) — e.g.
[Luck, Amor et al., "Data-efficient Co-Adaptation" (arXiv:1911.06832)](https://arxiv.org/pdf/1911.06832)
reuses experience across designs, and
[DRL+CPG co-optimization (IEEE 10324384)](https://ieeexplore.ieee.org/document/10324384/)
does it at small scale. The recurring lesson is that the controller must *not* be
relearned from scratch per design — share or condition the policy across designs
(the Bjelonic trick) or co-adaptation is intractable. On the simulation-practice
front, MuJoCo is the standard tool for modeling passive parallel/series
elasticity (tendon elasticity as linear springs; surrogate-compliance modeling
that injects soft-deformation variables into a rigid-body sim,
[arXiv:2512.07114](https://arxiv.org/html/2512.07114)), confirming this project's
choice — inject τ_spring(θ) in-loop in MJX — is the field-standard way to simulate
parallel elasticity for RL. The active frontier on the Part-2 / landing side is
tunable-stiffness passive elements that store landing energy for the next
push-off, e.g. the
[ETH/Max-Planck PELE electrohydraulic leg (Nature Comms 2024)](https://www.nature.com/articles/s41467-024-51568-3)
and 3D-printed auxetic springs reporting +35 % energy storage vs linear.

The cross-cutting consensus is three points. (1) Parallel elasticity reliably
offloads torque and cuts the τ² ohmic channel, with the largest gains on
low-geared / dynamic systems (monoped hopping −38.8 % CoT, ANYmal knee +33 %
torque-square) — consistent with "gearing is the crux." (2) The spring should be
co-designed with the controller, not bolted on, and the policy retrained or
conditioned so the gait *adapts* (the in-loop > post-hoc rule). (3) Energy/CoT
reward shaping is brittle — prefer velocity-normalized rewards (Liang) or
constraint/intrinsic formulations (EIPO) over a single hand-tuned scalar.

---

## Gap our work fills

Each ingredient of this project is established prior art; the *conjunction* is
not. The literature establishes that:

- Parallel torque offloading cuts motor energy (Collins ankle; STEPPR; CPEA),
  but the cleanest demonstrations use a **clutch**, and the always-engaged case
  on a real walker is under-tested.
- Torque-squared *electrical* accounting for a PEA, validated on hardware, is
  done — but on a **quadruped knee** ([Bjelonic 2023](https://arxiv.org/abs/2301.03509)).
- RL with elastic actuators is done — but on **quadrupeds**, and either by
  Bayesian optimization over a design-conditioned policy (Bjelonic 2023) or by
  exploiting *series* springs ([Raffin 2023](https://arxiv.org/abs/2209.07171)).
- A *tunable parallel* spring optimized with the τ² metric is done — by **this
  project's own group**
  ([Belov/Osokin, Skoltech 2024](https://arxiv.org/pdf/2411.18295)) — but with
  closed-form analytic optimization and a fixed PD controller on a leg test-stand:
  no RL, no in-loop gait adaptation, no running or load-carrying program.

This project occupies the unclaimed cell at the intersection: a **low-gear
quadruped** (Unitree Go1, 6.33:1) where ohmic loss dominates the budget (~54 %),
an **adaptive per-leg parallel preload** that scales with payload (the *online*
adaptation branch, one blind load-robust controller), trained **in-loop** so the
gait co-adapts to the offload rather than fighting it, and scored in **electrical
energy** (ohmic-loss CoT, no-regen) rather than mechanical work. The defining
empirical contributions are:

1. **Gearing as the crux, shown by a controlled contrast.** The same parallel-
   elastic mechanism *fails* in-loop on the high-geared G1 (ohmic ~4 %; an
   always-on hip-pitch spring is +7 % worse because it fights swing — the
   no-clutch failure mode the clutched literature predicts) but *pays* on the
   low-geared Go1 (ohmic ~54 %; an adaptive constant knee preload cuts cost of
   transport −14 to −27 % in 3 of 4 conditions, growing with load, across 3 seeds —
   seed 2 a weak −3 to −8 % outlier — with no stability cost at low-to-mid load,
   survival degrading above ~7.5 kg). This is the
   platform-level confirmation of the SEA-vs-PEA and Cheetah-vs-ANYmal gearing
   argument, on commercial hardware, with electrical accounting.
2. **Post-hoc vs in-loop methodology.** Reporting the optimistic post-hoc upper
   bound *and* the credible in-loop retrain, and showing they can disagree in
   sign (G1) or magnitude (Go1) — the contribution no surveyed work has made.
3. **Adaptive per-leg preload for load-carrying.** Extending the constant-preload
   Go1 win to a single controller whose per-leg preload scales with payload — the
   online-adaptation analog of the quasi-passive load-carrying exoskeletons
   ([Walsh 2007](https://aspirin.media.mit.edu/biomechatronics/wp-content/uploads/sites/8/2013/07/Walsh-2007_A-QUASI-PASSIVE-LEG-EXOSKELETON-FOR-LOAD-CARRYING-AUGMENTATION.pdf))
   and the monopod-with-payload PEA result
   ([Remy 2016](https://ieeexplore.ieee.org/document/7782370/)), but on a learned,
   blind, multi-legged controller.

The honest verdict matches `related_work.md`: this is **integration /
application novelty** (commercial low-gear platform + in-loop RL co-adaptation +
electrical accounting + the gearing contrast), not a new mechanism or principle.
The clearest path to a stronger contribution — flagged throughout the
literature — is to add the missing ingredient the strongest prior results all
share: a **clutch / dead-zone** to gate the spring to the load phase (rescuing
the G1 case and enabling flight-preloading for the Part-2 hopping/running
extension), and to **co-optimize** spring parameters with the policy rather than
hand-tuning a single curve.
