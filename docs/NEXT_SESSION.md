# NEXT SESSION — actionable handoff (updated 2026-06-21 cont.)

**The spin is SOLVED.** Clean s1 (leg_symmetry -2.0 + strong heading hold) trained to reward ~80,
3/3 survival, **yaw 0.016 rad/s** at zero command (was +1.8–3.8 across the whole prior lineage). The
automated s1 gate failed only on a **secondary 0.16 m/s drift** (not spin), so the driver correctly
aborted before s2–s4. The s1 policy is saved locally:
`outputs/clean_curriculum/2026-06-21_g1_clean_s1_clean_s1/` (policy_params + checkpoints + metrics).
Context: `CLAUDE.md`, `docs/JOURNAL.md` (2026-06-21 cont.), `docs/hop_jump_report.md`.

---

## The decision to make first (drift handling)

The yaw contaminant is gone; the only blemish is ~0.16 m/s lateral drift (the hopper goes straight but
wanders ~1.9 m / 12 s). Two paths:

- **(a) RECOMMENDED — drift is benign, proceed.** For the in-place spring-energy comparison both arms
  drift equally and s3's velocity tracking trains it out, so drift does NOT contaminate the result the
  way spin did. Relax the gate to **yaw + survival** (drift a soft ~0.25 m/s diagnostic, not a hard
  fail), and continue **s2→s4 warm-started from THIS saved s1** (re-upload it to a fresh box and
  `--restore` from it — skips re-training s1, ~52 min saved). Edit `hop_yaw_gate.py`'s PASS logic to
  separate the yaw verdict from the drift diagnostic.
- **(b) cleaner base — re-elicit s1 with anti-drift.** The drift, like the old yaw, comes from tracking
  being a *positive bonus* flat near zero. Add a direct penalty: tighten `tracking_sigma` 0.3→0.2 in
  `g1_clean_s1.yaml`, and/or add an explicit horizontal-base-velocity cost term to `g1_hop_env.py`
  (mirror `_reward_leg_symmetry`: a small negative cost on `|get_global_linvel(pelvis)[:2]|` at zero
  command). Smoke-test, then re-run s1 (~52 min) and re-gate.

## Then (either path) — the actual science

1. Continue the curriculum: s2 (height) → s3 (velocity) → s4 (bounding). The driver
   (`scripts/run_clean_curriculum.sh`) chains these with the automated gate.
2. **Re-run the −4.4 % hop-spring energy comparison on the clean (non-spinning) base** for the rigorous
   number (the whole point — the 2026-06-20 −4.4 % was caveated because both arms spun at different rates).
3. Deferred: energy-on retrain, ≥3 seeds, regen-sensitivity band.

## Box runbook (unchanged, validated this session)

1. Provision H100, paste IP. Bootstrap: `scripts/gpu_box_setup.sh` (or the `~/bootstrap.sh` pattern) →
   `git reset --hard origin/main` → assert `from pea.env import make_env` + `git rev-parse HEAD`.
2. Arm the `&&` fail-safe watchdog FIRST (`nohup bash -c 'sleep <budget> && { pkill -9 -f "[p]ea-train";
   sudo poweroff; }' &`). Confirm **passwordless sudo** (`sudo -n true`).
3. Launch `run_clean_curriculum.sh` detached at `nice -19 ionice -c3`. The driver **auto-gates s1** on
   `|yaw|<0.15` (hop_yaw_gate.py) and aborts the chain on fail.
4. Babysit from the Mac (the `/tmp/babysit.sh` pattern): retry-ssh (`-o IPQoS=none`, banner stalls are
   the flaky VPN — retry), rsync each stage as it ends, final rsync **then** poweroff (only with a
   verified local copy). **DELETE the instance** (powered off ≠ deleted; storage may still bill).

Working infra this session: `scripts/hop_yaw_gate.py` (yaw probe, exits non-zero on fail), the automated
gate in `run_clean_curriculum.sh`, the Mac-side `/tmp/bx` (retry-ssh) + `/tmp/bxpull` (retry-rsync) +
`/tmp/babysit.sh` helpers. The dog-running / G1-running / gravity-comp tracks remain SUSPENDED.
