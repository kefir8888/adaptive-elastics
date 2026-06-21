# NEXT SESSION — actionable handoff (updated 2026-06-21 cont. 2)

**Goal:** finally get the FAIR hop-spring energy number. The clean non-spinning curriculum is DONE (s1–s4);
the spring-vs-no-spring comparison was BLOCKED by a brittle spring policy, now diagnosed and fixed (centered DR).
Read first: `CLAUDE.md`, top of `docs/JOURNAL.md` (2026-06-21 cont. 2), and the README "GPU-box safety".

## Where things stand
- **Clean curriculum DONE.** s1 (non-spinning, yaw 0.016) → s2 (height, r113) → s3 (velocity, r117) → s4
  (bounding, r24). All in `outputs/clean_curriculum/`. Videos in `streak_videos/`. s3 yaw tracks both ways;
  forward translation is s4's job (s4: cmd 0.5/0.7/1.0 → 0.30/0.48/0.74 m/s straight).
- **Spring fit done.** Knee pogo (k=93.3, θ_engage 0.701, engage_sign +1) from the s2 work-loop; −51 W braking.
- **Fair comparison attempt #1 FAILED.** The spring arm trained (reward ~120) but is BRITTLE — falls in ~2 s at
  every local eval condition (the stock DR puts the nominal model at the least-damped boundary; the energetic
  spring over-hops there). Diagnosis + fix in the journal.
- **FIX in place (`d2aab0b`):** `centered_dr: true` (pea/randomize.py) makes the nominal an interior point. Both
  `configs/g1_hop_fair_{baseline,spring}.yaml` already carry it.
- **Box state:** `…52` is UP (no auto-poweroff; s2 at `~/runs/2026-06-21_g1_clean_s2_height_clean_s2`). Box `…74`
  powered off — **DELETE it from the immers console** if not already.

## THE NEXT RUN — re-do the fair comparison with centered DR (PARALLELIZED)
User will provide a **second box IP** to parallelize (the ramp is sequential; the baseline is independent).

1. On **box A (`…52`)**: `bash scripts/run_hop_fair.sh /home/ubuntu/runs/2026-06-21_g1_clean_s2_height_clean_s2`
   — but to parallelize, run ONLY the spring ramp there (the ~73-min long pole). On **box B**: run ONLY the
   baseline (~30 min). (Either split `run_hop_fair.sh` into two, or just launch the two `pea-train` chains by hand:
   baseline = `g1_hop_fair_baseline.yaml --restore <s2>`; ramp = `g1_hop_fair_spring.yaml --spring_k 40/75/93.3`
   chained.) Both arms use centered DR (in the configs). git reset + import-verify + nice-19 + no-poweroff babysitter.
2. **GATE the spring early:** after the k=40 stage, check survival on NOMINAL deterministic (`scripts/diag` pattern /
   the `/tmp/diag_spring.py` approach) — if the centered-DR spring is STILL brittle (falls <3 s nominal), STOP and
   rethink (maybe shorten the ramp, lower apex, or train without warm-start). Don't spend the whole ramp on faith.
3. When both arms finish: `uv run python scripts/hop_energy_compare.py <fair_baseline_dir> <fair_spring_k93_dir>`
   — deterministic nominal, now VALID (apex/cadence match, J/hop no-regen+regen, ohmic share, % delta).
   Free cross-check: also compare vs the OLD stock-DR baseline (`outputs/clean_curriculum/fair/...fair_baseline`).
4. **Then render** the fair pair (user asked): baseline + spring in-place + side-by-side, into
   `outputs/clean_curriculum/fair/videos/` with a README.

## Box runbook + safety (unchanged, validated this campaign)
Bootstrap (`gpu_box_setup.sh` or the `~/bootstrap.sh` pattern) → `git reset --hard origin/main` → verify
`from pea.env import make_env` + `git rev-parse HEAD`. Launch detached at `nice -19 ionice -c3`. Retry-ssh
(`-o IPQoS=none`; banner stalls = flaky VPN). Mac helpers persist as `/tmp/bx`, `/tmp/bxpull`, `/tmp/bxpush`.
No auto-poweroff while the user is iterating; rsync each stage; **destroy boxes when truly done**.

Deferred: energy-objective-ON retrain + ≥3 seeds (after a clean positive). Dog-/G1-running + gravity-comp SUSPENDED.
