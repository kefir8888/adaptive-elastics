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

## THE NEXT RUN — fair comparison with centered DR, PARALLELIZED across 2 boxes (TURNKEY)
DECIDED 2026-06-21: re-run NEXT session (not now) — supervised, parallelized, with an early gate
(the centered-DR fix is UNVERIFIED for the spring; don't spend the whole ramp on faith). Scripts are
committed: `run_hop_fair_ramp.sh`, `run_hop_fair_baseline.sh`, `hop_spring_survival.py`. The ramp is the
~73-min long pole; the baseline (~30 min) is independent → run them on different boxes.

0. **Box A = `…52`** (already up, bootstrapped, has s2 at `~/runs/2026-06-21_g1_clean_s2_height_clean_s2`).
   **Box B = the second IP the user gives** — bootstrap it, then UPLOAD s2 to it (local copy exists at
   `outputs/clean_curriculum/2026-06-21_g1_clean_s2_height_clean_s2/`, ~37 MB; `/tmp/bxpush` it to
   `~/runs/2026-06-21_g1_clean_s2_height_clean_s2/`).
1. **Box A — ramp** (long pole): `nohup nice -19 ionice -c3 bash scripts/run_hop_fair_ramp.sh \
   /home/ubuntu/runs/2026-06-21_g1_clean_s2_height_clean_s2 &`. **Box B — baseline**:
   `nohup nice -19 ionice -c3 bash scripts/run_hop_fair_baseline.sh \
   /home/ubuntu/runs/2026-06-21_g1_clean_s2_height_clean_s2 &`. Both: git reset + import-verify first;
   no-poweroff babysitter (boxes stay up).
2. **GATE (the safety valve):** when box A logs `END fair_spring_k40`, rsync that run dir to the Mac and
   `env -u PYTHONPATH uv run python scripts/hop_spring_survival.py <k40_dir>`. PASS = survives full episode +
   hops on NOMINAL deterministic. **If it FAILS (still ~2 s), `pkill -f run_hop_fair_ramp` on box A and STOP** —
   the centered DR didn't fix it; rethink (shorter ramp / lower apex 0.10 / no warm-start / k cap lower). If PASS,
   let the ramp finish k75→k93.
3. **Compare:** `env -u PYTHONPATH uv run python scripts/hop_energy_compare.py <fair_baseline_dir> <fair_spring_k93_dir>`
   — deterministic nominal, now VALID (apex/cadence match, J/hop no-regen+regen, ohmic share, % delta). Free
   cross-check: also vs the OLD stock-DR baseline at `outputs/clean_curriculum/fair/2026-06-21_g1_hop_fair_baseline_fair_baseline`.
   (DR-rollout fallback if ever needed: `hop_energy_compare_dr.py`.)
4. **Render** the fair pair (user asked): baseline + spring in-place + side-by-side (`render_hop.py`, loads the
   spring active), into `outputs/clean_curriculum/fair/videos/` + a README.
5. **Destroy box B** when done; leave/destroy box A per the user.

## Box runbook + safety (unchanged, validated this campaign)
Bootstrap (`gpu_box_setup.sh` or the `~/bootstrap.sh` pattern) → `git reset --hard origin/main` → verify
`from pea.env import make_env` + `git rev-parse HEAD`. Launch detached at `nice -19 ionice -c3`. Retry-ssh
(`-o IPQoS=none`; banner stalls = flaky VPN). Mac helpers persist as `/tmp/bx`, `/tmp/bxpull`, `/tmp/bxpush`.
No auto-poweroff while the user is iterating; rsync each stage; **destroy boxes when truly done**.

Deferred: energy-objective-ON retrain + ≥3 seeds (after a clean positive). Dog-/G1-running + gravity-comp SUSPENDED.
