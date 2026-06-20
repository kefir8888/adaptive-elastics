# NEXT SESSION — actionable handoff (updated 2026-06-21)

**Paste the prompt below to start the next session.** Active campaign: re-elicit a
**NON-SPINNING** G1 hopper via the clean curriculum (the fix for the pervasive yaw spin).
Context: `CLAUDE.md` (UPDATE 2026-06-21), `docs/JOURNAL.md` (2026-06-21), `docs/hop_jump_report.md`.

---

```
We're continuing the parallel-elastic study (G1 humanoid, MuJoCo Playground + MJX + brax PPO on a
rented immers.cloud GPU). Before anything, read in this order: CLAUDE.md, the top of docs/JOURNAL.md
(2026-06-21), docs/hop_jump_report.md, and the README "⚠️ GPU-box safety" section.

GOAL this session: re-elicit a NON-SPINNING G1 hopper via the clean curriculum. Last campaign EVERY
hopper spun ~+1.8–3.8 rad/s at zero command (a diagonal stance inherited from S1) — only bounding
escaped (its forward command breaks the symmetry). This contaminated the whole hop lineage and caveats
the −4.4% spring-energy result (the two arms spin at different rates). The fix is committed: a new
leg_symmetry reward + a 4-stage curriculum (configs/g1_clean_s1..s4, scripts/run_clean_curriculum.sh).

PLAN: I provision a fresh box and paste the SSH (ip:port, ubuntu user). Then you:
1. Bootstrap (scripts/gpu_box_setup.sh) → git reset --hard origin/main → verify
   `uv run python -c "from pea.env import make_env"` (and git rev-parse HEAD, not ls).
2. Launch the FAIL-SAFE watchdog FIRST — &&-chained so a killed sleep ABORTS the poweroff:
   nohup bash -c 'sleep <budget> && { pkill -9 -f "[p]ea-train"; sudo poweroff; }' &
   Then run scripts/run_clean_curriculum.sh detached at nice -19 ionice -c3. Watch stage 1's first eval.
3. *** GATE STAGE 1 HARD ***: when clean_s1 finishes, measure mean yaw at zero command by PATCHING
   sample_command to a constant (the raw rollout command-override is overwritten by the env's internal
   re-sample). Require |yaw| < 0.15 rad/s, ~0 net drift, survival. If it still spins, STOP and debug —
   do NOT build s2–s4 on a spinning base.
4. If s1 passes, the driver chains s2 (height) → s3 (velocity) → s4 (support exchange). Validate s3
   yaw-tracking with the sample_command patch; check s4 alternation via actual foot-floor gap vs the
   contact flag. rsync each stage off-box BEFORE launching the next (backup-first).

NON-NEGOTIABLE (real money lost + a self-poweroff happened last time): fail-safe &&-watchdog; NEVER kill
a watchdog's `sleep` child (kill the parent); mkdir-lock launches (pgrep -f self-matches its own cmdline);
rsync before destroy; SSH with -o IPQoS=none as `ubuntu` (flaky VPN).

WHEN DONE: rsync everything, render videos, then re-run the −4.4% energy comparison on the clean
(non-spinning) base for a rigorous number.
```

---

Estimate: ~3.5 h box / ~$12–14 (happy path); +1 session if stage 1's anti-spin needs debugging.
Deferred (after the clean base): energy-on retrain, ≥3 seeds, commanded/varied-height obs change.
The dog-running / G1-running / gravity-comp tracks remain SUSPENDED (see their design docs).
