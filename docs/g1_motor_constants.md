# Unitree G1 7520 Actuator Electrical Constants — Determination

Date: 2026-06-14. Scope: the ohmic-loss model `P = (tau/Kt)^2 * R = tau^2 * (R/Kt^2)`.
The load-bearing quantity is the **joint-side R/Kt^2** (Ohm/(N*m/A)^2).

## TL;DR — the band (motor-level R/Kt^2)

| | Ohm/(N*m/A)^2 |
|------|---------------|
| LOW (aggressive size-scaling) | 0.5 |
| **BEST** | **1.2** |
| HIGH (G1 barely larger than Go2) | 4.0 |

Joint-side = motor-level / gear^2. Across the full band x gear uncertainty, the
**lowest plausible joint-side R/Kt^2 is ~0.001 (knee, 22.5 gear, optimistic) and
the highest is ~0.020 (hip-pitch, 14.3 gear, pessimistic).** The current code value
(0.0025 joint-side) sits at the **optimistic edge** but is now defensible as a band
endpoint rather than wrong by orders of magnitude.

---

## 1. Is there a real external G1 source now, or still a proxy?

**Still a proxy. There is NO external primary source for the Unitree G1 7520
electrical constants (Kt, R, or R/Kt^2).** This is confirmed by all six search
angles and is a permanent blocker: Unitree does not publish motor constants for the
7520 series (the part number is an internal designation, not a standalone product),
no datasheet, teardown, academic measurement, or URDF/MJCF electrical metadata exists
for it.

The team's own estimate (Kt 2.3, R 0.013, R/Kt^2 0.0025 joint-side) is **REJECTED as
a source**: it has no upstream trace and is the documented failure mode — the
2026-06-14 web chase "circled back to the repo's own estimates." A self-estimate (or
anything derived from it) cannot count as external.

**Genuine external data that DOES survive** (all for SMALLER, lower-gear motors than
the G1 7520, so usable only as scaling anchors, not as G1 values):

- **Go2 (GO-M8018-6) teardown — Simplexity Product Development** (simplexitypd.com,
  independent third-party teardown with FOC back-EMF characterization + direct
  multimeter resistance measurement). The single genuinely MEASURED anchor for a
  Unitree-family motor: motor-side Kt_q ~0.26 N*m/A (q-axis), Kt_phase ~0.22 N*m/A;
  R = 0.44 Ohm line / 0.66 Ohm phase (36N42P Delta winding); gear 6.22:1.
- **Go1 (GO-M8010-6) manual** (ManualsLib excerpt / generation-robots.com datasheet
  PDF): joint-side Kt = 0.639 N*m/A, gear 6.33:1 (motor-side Kt = 0.101). A Kt
  convention anchor only; R is unpublished.
- **G1 joint torque/speed limits** (MuJoCo Menagerie jnt_actfrcrange + Unitree g1_23dof
  URDF velocity limits): knee 139 N*m @ 20 rad/s; hip-pitch 88 N*m @ 32 rad/s; ankle
  50 N*m @ 30 rad/s. Output/joint side. These bound implied stall current once Kt is
  assumed but do NOT pin Kt or R.

---

## 2. Motor-level R/Kt^2 band (with basis)

**LOW 0.5 / BEST 1.2 / HIGH 4.0 Ohm/(N*m/A)^2.**

**Anchor (corrected):** the Go2 MOTOR-LEVEL R/Kt^2 from the measured teardown is
`R_phase / Kt_q^2 = 0.66 / 0.26^2 = 9.76`, with a spread of **6.5 to 13.6**
(`R_line/Kt_q^2 = 0.44/0.26^2 = 6.5` to `R_phase/Kt_phase^2 = 0.66/0.22^2 = 13.6`).

> **CRITICAL ERROR FOUND AND CORRECTED.** Search angles 4 and 5 reported the Go2
> motor-level anchor as **0.065-0.098** — a **100x arithmetic error** (they divided R
> by Kt instead of Kt-squared, or mislabeled the joint-side 0.168 figure). Every G1
> motor-level band derived from 0.065-0.098 (angle 5's "0.008-0.040") is therefore
> ~100x too low and is REJECTED. Only the scaling-law FORM survives from those angles.

**Size-scaling to the G1 7520.** The G1 leg motor is LARGER than the Go2 M8018 (G1 leg
joint torque ~88-139 N*m vs Go2 ~24-45 N*m; ~2-3x stator volume). Larger BLDC motors
have LOWER motor-level R/Kt^2 via `R/Kt^2 ~ D^-3 to D^-5` (R ~ D, Kt ~ D^2). Scaling the
6.5-13.6 anchor down:

- **BEST ~1.2** = ~2-2.5x size with the conservative D^-3 exponent. This **coincides
  with the team's implied motor-level value** (0.00246 joint x 22.5^2 = 1.24), so the
  team number is defensible at the optimistic edge, not wrong.
- **HIGH ~4.0** = minimal scaling (G1 barely larger, D^-3, size factor ~1.3): the
  pessimistic "G1 motor is essentially Go2-like" case. (The team's competing "real
  ~0.02 knee joint-side" claim implies motor-level ~10, i.e. an UN-scaled Go2 — this
  over-corrects by ignoring the larger-motor physics, so 4.0 is the defensible upper
  edge, not 10.)
- **LOW ~0.5** = aggressive scaling (x3 size, steeper D^-4 exponent).

This is a deliberately WIDE (~8x) honest band because the only measured input is one
Go2 teardown plus a size-ratio guess; no G1 datasheet exists. **Motor-level R/Kt^2 is
gear-INVARIANT**, so the hip-pitch gear dispute (below) does not touch this band.

---

## 3. Joint-side R/Kt^2 = motor-level / gear^2

Knee gear 22.5:1 (22.5^2 = 506). Hip-pitch gear DISPUTED — both shown (14.3^2 = 204.5;
22.5^2 = 506).

| Joint (gear) | LOW (mot 0.5) | BEST (mot 1.2) | HIGH (mot 4.0) |
|---|---|---|---|
| **Knee** (22.5) | 0.00099 | 0.0024 | 0.0079 |
| **Hip-pitch (14.3)** *(default)* | 0.0024 | 0.0059 | 0.0196 |
| **Hip-pitch (22.5)** *(alt)* | 0.00099 | 0.0024 | 0.0079 |

**Overall lowest plausible joint-side R/Kt^2: ~0.0010** (knee, or hip-pitch at 22.5,
motor-level LOW). **Overall highest plausible: ~0.020** (hip-pitch at 14.3, motor-level
HIGH). That ~20x window is the full envelope the absolute headline must be reported over.

The hip-pitch gear (14.3 vs 22.5) is itself a 2.48x swing on its joint-side R/Kt^2 and
is UNRESOLVED — 14.3:1 from arXiv RL-paper convention + local model/armature inspection
vs 22.5:1 from the team's CLAUDE.md/RESULTS and (per robot_inventory.md line 94) the
OmniXtreme paper. Default to **14.3:1** (the Part 1 spring target, per local inspection)
but carry the 2.48x sensitivity. No primary Unitree source resolves it.

---

## 4. Best-estimate per-joint Kt and R (R/Kt^2 is what actually matters)

These split the BEST motor-level R/Kt^2 (1.2) into a Kt and an R, anchored to the Go2
convention (motor-side Kt ~0.1-0.15 N*m/A for this family, scaled up for the larger G1
motor). The Kt/R split is much softer than the R/Kt^2 product; treat these as
illustrative, not independently sourced.

| Joint (gear) | Joint-side Kt (N*m/A) | R (Ohm, joint-side equiv) | Joint-side R/Kt^2 |
|---|---|---|---|
| Knee (22.5) | ~2.3 (motor ~0.10) | ~0.013 | 0.0024 |
| Hip-pitch (14.3) | ~1.5 (motor ~0.10) | ~0.013 | 0.0059 |
| Hip-pitch (22.5 alt) | ~2.3 | ~0.013 | 0.0024 |

Caveat: only **R/Kt^2** enters the ohmic-loss model. The individual Kt and R above are
a defensible-but-unsourced factoring of the BEST product; do not cite them as measured.

---

## 5. Comparison: has the number moved?

| Value | Joint-side R/Kt^2 (knee) | Status |
|---|---|---|
| Current code | 0.0025 | optimistic edge of the new band |
| Prior Go2-proxy claim | 0.02-0.05 | **too pessimistic** — built on the 100x-wrong anchor and an un-scaled Go2 |
| **This determination (BEST)** | **0.0024** (band 0.0010-0.0079) | corrected |

**Direction: the number moved DOWN (back toward the code value), not up.** The earlier
"the code is 8-19x too optimistic, real is ~0.02-0.05" conclusion was itself an
artifact of the 100x arithmetic error in angles 4/5 plus ignoring the larger-motor
scaling. With the corrected Go2 anchor (6.5-13.6, not 0.065-0.098) and proper size
down-scaling, the team's 0.0025 lands at the OPTIMISTIC edge of a defensible band whose
BEST point is essentially the same value. So the code value is no longer "wrong by an
order of magnitude" — it is the optimistic endpoint of a band that must be reported as
a band.

---

## 6. Recommendation

**Use the BEST motor-level R/Kt^2 = 1.2 Ohm/(N*m/A)^2 as the central value**, i.e.
joint-side **0.0024 for the knee (22.5)** and **0.0059 for hip-pitch (14.3)**. This is
within rounding of what the code already uses for the knee; the only substantive change
is the hip-pitch number, which should be ~2.4x larger than the knee because of its lower
14.3 gear (the code currently applies one 22.5-based value to all DoFs).

**Report all absolute CoT/ohmic-% results as a BAND over motor-level R/Kt^2 0.5-4.0**
(joint-side knee 0.0010-0.0079, hip-pitch-14.3 0.0025-0.020), and **LEAD with the
Kt/R-invariant relative reduction** (spring vs no-spring %), which does not depend on
the absolute constant at all.

**Crucially, the gate's VALIDITY is R/Kt^2-invariant:** the same constant is used in
both the spring and no-spring conditions, so the comparison is fair regardless of which
value in the band is true. Only the absolute "is the win big (ohmic-dominated) or small
(braking-recovery-dominated)" headline depends on the constant — and that is exactly
what the band communicates honestly.

**Bench measurement still needed for publication?** YES, if the paper makes any
ABSOLUTE ohmic/CoT claim. The only true fix is a real 7520 teardown bench test: measure
phase winding resistance (multimeter, line-to-line then convert) and torque constant
(spin-down / back-EMF curve, or a static torque-vs-current point). One such measurement
collapses the ~20x band to a point and removes the project's #1 publication risk. For
the RELATIVE gate result, no measurement is needed — it is already invariant.

Also flag for the code (do NOT modify here): energy.py reportedly plugs the Go2
motor-side Kt 0.26 in as a joint-side value (~39x error on the Go2 cross-check row); the
correct Go2 joint-side Kt is 1.62 (= 0.26 x 6.22), with R 0.44/0.66. This does not
affect the G1 band but corrupts the Go2 comparison if used.

---

## 7. Provenance / confidence note + source list

- **Confidence: MEDIUM on the band, LOW on any single point.** The band rests on ONE
  measured external input (the Go2 teardown) plus a first-principles size-scaling law
  and a size-ratio guess. No G1 primary source exists.
- **What is solid:** the Go2 measured constants; the corrected Go2 motor-level anchor
  6.5-13.6 (arithmetic on measured inputs, HIGH confidence); the gear-invariance of
  motor-level R/Kt^2; knee gear 22.5:1; the relative-result invariance.
- **What is soft:** the G1/Go2 size ratio (2-3x, no published G1 motor dimensions), the
  scaling exponent (D^-3 to D^-5), the hip-pitch gear (14.3 vs 22.5, unresolved), and the
  individual Kt/R split.

Source list (date-stamped 2026-06-14):
- Simplexity Product Development — Unitree Go2 motor teardown (simplexitypd.com):
  measured Kt_q 0.26, Kt_phase 0.22 (motor-side); R 0.44 line / 0.66 phase; 36N42P
  Delta; gear 6.22:1. MEASURED, MEDIUM-HIGH; URL not pinned in repo.
- Unitree GO-M8010-6 User Manual (ManualsLib excerpt) + generation-robots.com Go1
  datasheet PDF: Go1 joint-side Kt 0.639, gear 6.33:1. MEDIUM (re-retrieval flaky).
- MuJoCo Menagerie G1 jnt_actfrcrange + Unitree g1_23dof URDF velocity limits
  (unitreerobotics/unitree_ros): knee 139 N*m/20 rad/s, hip-pitch 88 N*m/32 rad/s,
  ankle 50 N*m/30 rad/s. HIGH for sim; output side.
- Gear ratios: knee 22.5:1 (7520-22.5, consistent across all sources). Hip-pitch
  DISPUTED 14.3 vs 22.5 (arXiv RL convention + local inspection vs OmniXtreme/team
  docs); default 14.3, carry 2.48x sensitivity.
- BLDC scaling law R/Kt^2 ~ D^-3 to D^-5: classical machine-design physics (Katz/MIT
  Mini-Cheetah lineage). Method, not a G1 number.
- REJECTED: team self-estimate (Kt 2.3, R 0.013, R/Kt^2 0.0025) — no external trace;
  the 2026-06-14 web chase circled back to it. H1 M107 proxy — out of scope, no source.
  Angles 4/5 motor-level 0.065-0.098 and the G1 0.008-0.040 band derived from it — 100x
  arithmetic error.
