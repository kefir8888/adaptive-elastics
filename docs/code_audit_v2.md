# Code audit — v2 (2026-06-16, full re-audit)

> Automated multi-agent re-audit of src/pea + scripts (duplication, bugs, inconsistencies),
> each finding adversarially verified, then reconciled against the prior docs/code_audit.md.
> 114 of 127 findings confirmed against current code. Nothing from the prior audit has been
> fixed yet (all prior P0 items still present). See the prioritized fix list at the end.

# Code-audit reconciliation report

Scope: the 114 new findings against the prior audit (`docs/code_audit.md`, 2026-06-16). All cited code was re-checked against the current files. Tags: **[NEW]** = not in prior audit; **[KNOWN — prior]** = same issue already written up and still present; **[KNOWN — fixed]** = prior audit flagged it but code now differs (none found — see note).

Note: every prior-audit P0 item I re-checked is still present in the code (e.g. `metrics.evaluate:179` `tq = td = []` aliasing; `experiment.rollout` does step→break-before-append). So nothing has been fixed since the prior audit; there are no [KNOWN — fixed] entries.

---

## Bugs

| Tag | Sev | Location | Issue / one-line fix |
|---|---|---|---|
| **[NEW]** | medium | `scripts/g1_run_probe.py:53` | Speed denominator `(n-w0)*dt` should be `(n-1-w0)*dt`; ~0.2% off on long runs, up to ~20% on early falls. Fix: subtract one interval to match `metrics.py:110`. |
| **[NEW]** | medium | `scripts/go1_capacity.py:76-77` | Speed is computed/printed even when the run is too short to be valid (energy shows `nan` but speed shows `0.00`); also `np.stack` empty-list risk. Fix: move `QPa`/`spd` inside the `nstep > w0+100` guard, `else nan`. |
| **[NEW]** | medium | `scripts/go1_capacity.py:73` | `np.stack(QF)` re-built a second time though `QF2` already holds the slice. Fix: `ctau = mean(abs(QF2[:, da]))`. |
| **[NEW]** | medium | `scripts/motor_budget.py:71-72` | `tot_mech_pos` re-reads the raw trajectory instead of reusing `tau`/`mech_dof`; duplicate `[skip:]` slice. Fix: compute from `tau[:, act]*qvel[:, act]`. |
| **[NEW]** | medium | `scripts/probe_speed_hold.py:81` | `np.stack` on empty list if robot falls on step 0 (likely for the stand/hold `[0,0,0]` command). Fix: guard `if not qpos: continue`. |
| **[KNOWN — prior]** | high (new) / low (prior) | `scripts/motor_budget.py:59` | Dead `if False else` branch with a wrong 2-arg `mj_id2name` call. Prior audit Bug #10 already flags this (as cosmetic). Fix: delete the ternary, keep the 3-arg call. |
| **[NEW]** | high | `scripts/render_walk.py:34` | Joint name hardcoded `"calf"` instead of `cfg.spring.joint`; crashes on startup for any non-Go1 model, even when adaptation is off. Fix: read `cfg.spring.joint` and only look it up when `ADAPT`. |

Low-severity bugs passed through without independent verification (all **[NEW]**, not in prior audit): `g1_run_probe.py:51` (redundant double-stack of `QF`/`QV`), `g1_run_probe.py:53` (window differs from `metrics.performance`), `recalc_rkt2.py:42-43` (joint- vs dof-iteration, latent for non-hinge models), `render_walk.py:78` (8 aliased black frames), `go1_capacity.py:55,57` (in-place mutation of a frozen flax struct's info dict — note this same pattern is in `env.py` and is an established project convention).

---

## Duplication

| Tag | Sev | Location | Issue / one-line fix |
|---|---|---|---|
| **[KNOWN — prior]** | medium | `recalc_rkt2.py:56-64`, `power_compare.py:50-66`, `motor_budget.py:31-38`, `probe_speed_hold.py:88-91` | Electrical-power math reimplemented per script instead of calling `energy.electrical_power`/`ohmic_power`. Prior audit Dup #3. Fix: call the `energy.py` functions. |
| **[KNOWN — prior]** | medium | `recalc_rkt2.py:42-54` vs `power_compare.py:41-74` | act/hip index-building copy-pasted. Prior audit Dup #4. Fix: shared `load_baseline_arrays()` helper. |
| **[KNOWN — prior]** | medium | `scripts/render_walk.py:54-79` vs `go1_capacity.py:53-69` | Adaptive-preload rollout loop copy-pasted. Prior audit Dup #1+#2 (rollout loop + adaptive controller). Fix: one shared rollout helper with an `on_step` callback. (Caveat: prior audit's claim that the canonical loop "already lives in `rollout.py`" is wrong — `rollout.py` has no preload code; a helper must be created.) |

Low-severity **[NEW]** pass-throughs: `g1_run_probe.py:37-50` (rollout loop copied from `rollout.py`), `go1_capacity.py:71-73` (steady-window slicing duplicated across scripts).

---

## Inconsistency

| Tag | Sev | Location | Issue / one-line fix |
|---|---|---|---|
| **[NEW]** | medium | `scripts/g1_run_probe.py:36` | `KT` (a knee-torque list) reads like the motor constant `kt` (a float) used two lines away. Fix: rename to `KTAU`. |
| **[NEW]** | medium | `scripts/go1_capacity.py:77` | Speed uses the steady half-window while `metrics.performance` uses the full window; both printed as plain "speed" → comparison trap. Fix: label `speed_steady` and comment the difference. |
| **[NEW]** | medium | `recalc_rkt2.py:29` vs `energy.py:55` | Sweep value `0.0025` re-hardcoded; real `G1_KNEE` ratio is `0.00246`, so the "current" tag drifts if `G1_KNEE` changes. Fix: derive `CVALS[0]` from `energy.G1_KNEE`. |

Low-severity **[NEW]** pass-throughs: `g1_run_probe.py:53` (window vs `metrics.performance`), `recalc_rkt2.py:80-81` (missing "knee est" tag for `c==0.02`), `go1_capacity.py:39` (`ADAPT` only matches exact `preload_dr`), `render_walk.py:34` vs `env.py:48` (joint sourcing inconsistent).

---

## Dead code

| Tag | Sev | Location | Issue / one-line fix |
|---|---|---|---|
| **[NEW]** | medium | `scripts/render_walk.py:34-35,41` | `da`/`nleg` computed unconditionally but used only inside `if ADAPT:`; combined with the hardcoded `"calf"` this crashes non-Go1 models. Fix: move the lookup inside the `if ADAPT:` branch. |
| **[KNOWN — prior]** | high (new) | `scripts/motor_budget.py:59` | Same dead `if False` branch listed under Bugs (prior audit Bug #10). |

Low-severity **[NEW]** pass-through: `motor_budget.py:47` (`T` assigned, never used).

---

## Hardcoded

| Tag | Sev | Location | Issue / one-line fix |
|---|---|---|---|
| **[KNOWN — prior]** | high | `recalc_rkt2.py:23-26`, `motor_budget.py:23-26`, `power_compare.py:20-23` | Absolute Google-Drive path with username/email/locale hardcoded. Prior audit Dup #7. Fix: use `config.resolve_runs_dir()` or a `--run` argument. |
| **[NEW]** | medium | `scripts/go1_capacity.py:29,47`, `render_walk.py:29-30` | Trunk body index hardcoded as `1`. (Note: `payload.py:23` already defines `TORSO_BODY_ID = 1`; these scripts bypass it.) Fix: use a `mj_name2id("trunk")` lookup or import the constant. |
| **[NEW]** | medium | `recalc_rkt2.py:40,49` (+ `power_compare.py:37,48`, `motor_budget.py:53,104`) | Config paths are bare relative strings → `FileNotFoundError` unless run from repo root. Fix: anchor with `Path(__file__).resolve().parents[1]`. |

Low-severity **[NEW]** pass-throughs: `g1_run_probe.py:32` (speed sweep hardcoded, no check vs training range), `motor_budget.py:53,104` (config paths cwd-dependent).

---

## Prioritized fix list

**P0 — correctness / affects reported numbers**
1. `g1_run_probe.py:53` — speed off-by-one (biases achieved speed low, badly on short runs).
2. `go1_capacity.py:76-77` — print `0.00` speed on invalid (energy-`nan`) runs → misleading capacity-curve rows; also empty-stack crash risk.
3. `probe_speed_hold.py:81` — empty-stack crash when the policy falls on the stand command.
4. `render_walk.py:34` — hardcoded `"calf"` crashes any non-Go1 run on startup.
5. `recalc_rkt2.py:29` — `CVALS[0]=0.0025` should track `energy.G1_KNEE` so the "current" sweep row is the real project constant.
6. Carry over still-open prior-audit P0s: `metrics.evaluate:179` aliasing + redundant `joint_addrs`; `rollout.py`/`experiment.rollout` done/append-order mismatch + 0-length guard; post-hoc spring-sign guard.

**P1 — deduplication**
7. One shared rollout helper (with `on_step` callback) to replace the 5-6 copied loops — covers `render_walk.py`/`go1_capacity.py` adaptive-preload duplication. (Prior audit Dup #1/#2.)
8. Route all script electrical-power math through `energy.electrical_power`/`ohmic_power`; delete the per-script `block`/`per_dof_elec`. (Prior audit Dup #3.)
9. Shared `load_baseline_arrays()` for the copied act/hip index-building in `recalc_rkt2.py`/`power_compare.py`. (Prior audit Dup #4.)

**P2 — hygiene**
10. Delete dead `if False` ternary at `motor_budget.py:59`; remove unused `T` (`:47`); reuse `QF2` at `go1_capacity.py:73`; drop redundant slice at `motor_budget.py:71-72`.
11. Replace hardcoded Drive paths with `config.resolve_runs_dir()`; anchor config paths with `Path(__file__)`. (Prior audit Dup #7.)
12. Rename `KT`→`KTAU` in `g1_run_probe.py`; label `speed_steady` in `go1_capacity.py`; replace trunk body index `1` with the named lookup / `payload.TORSO_BODY_ID`.

Relevant files: `/Users/elijah/programming/adaptive_elastics/scripts/g1_run_probe.py`, `/Users/elijah/programming/adaptive_elastics/scripts/go1_capacity.py`, `/Users/elijah/programming/adaptive_elastics/scripts/render_walk.py`, `/Users/elijah/programming/adaptive_elastics/scripts/motor_budget.py`, `/Users/elijah/programming/adaptive_elastics/scripts/recalc_rkt2.py`, `/Users/elijah/programming/adaptive_elastics/scripts/power_compare.py`, `/Users/elijah/programming/adaptive_elastics/scripts/probe_speed_hold.py`, `/Users/elijah/programming/adaptive_elastics/src/pea/metrics.py`, `/Users/elijah/programming/adaptive_elastics/src/pea/experiment.py`, `/Users/elijah/programming/adaptive_elastics/src/pea/energy.py`, `/Users/elijah/programming/adaptive_elastics/src/pea/payload.py`.