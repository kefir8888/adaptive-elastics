# Gait-conditioned single controller — design proposal (2026-06-20, user)

A unification idea for Part 2: train **one** G1 policy that can two-footed hop, single-leg hop,
and run (and walk), selected by the COMMAND — then study the parallel spring across the whole
gait family with one artifact.

## The command

`(vx, vy, omega, L1, R1, L2, R2)` where the last four are a binary **foot-contact schedule** over a
2-phase gait cycle: which feet are the propulsive/stance feet in phase 1 (`L1,R1`) and phase 2
(`L2,R2`). Flight emerges in the transitions between phases. `(0,0,0,0)` is forbidden (every phase
must have at least one propelling foot).

Examples (small forward speed, e.g. 0.1 m/s):
- `(0.1, 0.1, 0.1, 1,0,0,1)` — left propels in phase 1, right in phase 2 → **alternating run/walk**.
- `(0.1, 0.1, 0.1, 1,1,1,1)` — both feet propel every phase → **two-footed hop** (flight between cycles).
- `(0.1, 0.1, 0.1, 1,0,1,0)` — only left ever propels → **single-leg (left) hop**.

## Why pursue it

- **Transfer / curriculum.** The two-footed hop is PROVEN trainable (`G1JoystickHop`, reward 0.76→66).
  It shares the push-off + flight skill with bounding/running, so a conditioned policy can let the
  easy pattern scaffold the hard one — a more promising route to running than a cold bounding start
  (the 2026-06-20 bounding attempt failed on infra, never trained; the cold-start is genuinely hard).
- **One artifact for the spring study.** Measure the spring's effect across hop / single-leg / run
  cheaply, and answer "does the spring help RUNNING on the G1" (more braking energy than hopping —
  the original Direction 2) in the same policy that hops. Fits the project's "one conditioned policy,
  zero-shot" theme (Direction 5).
- **Clean energy comparison.** Small commanded velocity keeps it near-in-place, comparable to the hop.

## The work to get right

1. **Reward must ENFORCE the commanded schedule** — the core new piece. Generalize the existing
   anti-phase `feet_phase` / synchronous `hop_rhythm` to read the foot-pattern command: reward the
   commanded feet being in stance during their phase, penalize wrong-foot contact, reward a flight
   phase between phases. The 2-phase clock already exists (generalize `info["phase"]`); cadence is
   `hop_freq`.
2. **Observe the command.** Add the 4 schedule bits (and v) to the observation so the policy conditions.
3. **Curriculum, don't cold-start the full command set.** Warm-start from the hopper; widen the
   command distribution from the two-footed pattern to single-leg, then alternating. Jointly training
   very different gaits can be harder than each alone — conditioning + curriculum mitigates it.
4. **Some command combinations are physically marginal** (e.g. single-leg hop turning forward) — fine,
   the policy does its best; sample feasible patterns more.

## Sequencing

NOT the immediate step. Today: settle the spring question with the fair, apex-controlled hop
comparison (`configs/g1_hop_{baseline,spring}.yaml`, now with the `hop_overshoot` apex cap). The
unified controller is the next chapter — a deliberate env/reward redesign (likely `G1JoystickGait`
subclassing the hop env), best started from the proven hopper.
