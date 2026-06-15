# Code audit — src/pea + scripts (2026-06-16)

Scope: all of `src/pea/*.py`, `scripts/*.py`, skim of `configs/`. Goal: find bugs,
duplication, needlessly-reinvented library code, and assess the energy-model and
spring-injection correctness. File:line references throughout.

Overall the core is in good shape: the spring lives in one place, is config-selected,
and is callable both in-loop and post-hoc as required by CLAUDE.md. The energy/no-regen
convention is implemented consistently (per-DoF clamp before sum) in every site that
matters. The main issues are (1) a lot of copy-pasted rollout/energy/joint-lookup logic
across the one-off scripts that has already drifted, (2) a real latent bug in
`metrics.evaluate`, and (3) a few correctness foot-guns in the spring wrappers.

---

## Duplication

The biggest theme. `src/pea/metrics.py` was clearly written to consolidate the one-off
scripts, but the scripts were never deleted and now duplicate it.

1. **The rollout loop is reimplemented 5 times**, all near-identical (jit reset/step/policy,
   pin `state.info["command"]`, append qpos/qvel/qfrc, break on `done`):
   - `src/pea/rollout.py:70-84`
   - `src/pea/experiment.py:92-102` (`rollout()`)
   - `scripts/go1_capacity.py:55-72` (`at_payload`)
   - `scripts/g1_run_probe.py:37-50`
   - `scripts/render_walk.py:60-80`
   - `scripts/probe_speed_hold.py:70-80`
   None of these share a helper. They have already drifted (see Bugs #2: the `done`/append
   order differs between `rollout.py` and `experiment.py`). Recommend one
   `rollout(env, policy, command, steps, on_step=None)` helper in `pea/` that all six call;
   the adaptive-preload variants pass an `on_step` callback.

2. **The adaptive per-leg preload controller is implemented twice, verbatim**:
   - `scripts/go1_capacity.py:38-70` (`ALPHA`, `KP=0.2`, `RATE=2.0`, EMA, clip ramp)
   - `scripts/render_walk.py:38-71` (same constants, same update line)
   This is the project's *one positive-result mechanism* (the load program) and it lives in
   two copy-pasted blocks in throwaway scripts, not in `src/pea/`. If the gains change in one,
   the rendered video silently diverges from the measured capacity curve. Strong candidate to
   promote into `pea/` (e.g. `pea/adaptive_preload.py` with an `AdaptivePreload` stepper).

3. **Energy / power breakdown reimplemented per script** instead of calling
   `metrics.power_breakdown` (`src/pea/metrics.py:81-95`):
   - `scripts/power_compare.py:50-66` (`block`)
   - `scripts/motor_budget.py:31-38` (`per_dof_elec`)
   - `scripts/recalc_rkt2.py:56-64` (`block`)
   - `scripts/probe_speed_hold.py:88-92`
   - `src/pea/analyze.py:77-87` recomputes the whole-body no-regen sum inline rather than
     reusing `metrics.power_breakdown` / `energy.electrical_power`.
   `metrics.power_breakdown` already returns exactly these fields (mech/ohm/regen/noregen);
   four of these blocks could be deleted.

4. **Actuated-DoF lookup duplicated.** `metrics.actuated_dof_adrs` (`metrics.py:29-32`) vs
   inline list comprehensions `[int(model.jnt_dofadr[j]) ... >= 6]` in
   `scripts/power_compare.py:41`, `scripts/motor_budget.py:62`, `scripts/probe_speed_hold.py:50`,
   `scripts/recalc_rkt2.py:42`. Also note the `ElectricalRewardWrapper` uses a flat `[..., 6:]`
   slice (`env.py:84`) — equivalent for the G1/Go1 (free joint = first 6 DoFs, all others
   actuated and contiguous) but a *different* code path than `actuated_dof_adrs`; see Bugs #5.

5. **Joint-by-substring lookup duplicated three ways** with three slightly different return
   shapes: `env.joints_by_substring` (dict, `env.py:204`), `metrics.joint_addrs` (tuple of
   lists, `metrics.py:35`), and the inline loop in `motor_budget.py:57-63`. `env.knee_joints`
   (`env.py:223`) is a thin alias kept only for `rollout.py`. Consolidate to one.

6. **Motor-constant selection duplicated.** `analyze.py`, `motor_budget.py`, `power_compare.py`,
   `probe_speed_hold.py` all hardcode `Kt=2.3, R=0.013` (`G1_KNEE`) directly instead of
   `energy.motor_constants(cfg.energy_motor)` (the path `experiment.py`, `go1_capacity.py`,
   `g1_run_probe.py` correctly use). `probe_speed_hold.py:25` even re-types the literals
   `KT, R = 2.3, 0.013` rather than importing them. These scripts are G1-only by construction,
   so it is not *wrong*, but it is why those scripts can't be pointed at the Go1.

7. **Hardcoded absolute Drive paths** repeated in `motor_budget.py:23`, `power_compare.py:20`,
   `probe_speed_hold.py:21`, `recalc_rkt2.py:23` — the exact thing `config.resolve_runs_dir()`
   exists to avoid. These will break for anyone but the original user / if the run is renamed.

---

## Reinvented wheels (should have come from a library)

1. **Video rendering loop is hand-rolled twice.**
   - `rollout.py:124-139` uses `mujoco.Renderer` + `mediapy.write_video` (reasonable).
   - `render_walk.py:44-95` hand-rolls a raw-RGB pipe into an `ffmpeg` subprocess
     (`subprocess.Popen([... "libx264" ...])`). MuJoCo Playground ships a tested
     `mujoco_playground` rollout/render path and `mediapy.write_video` already wraps ffmpeg;
     the manual pipe is fragile (no error handling on `proc.stdin.write`, no returncode check).
     At minimum render_walk should reuse `rollout.render_video`.

2. **Manual MJX rollout instead of `brax`/`mujoco_playground` evaluators.** All six rollout
   loops drive `jit_step` in a Python `for` loop one env at a time. brax provides
   `Evaluator` / `generate_unroll`, and Playground has batched eval wrappers. For a single
   10–20 s CPU rollout the Python loop is fine and arguably clearer, but the *batched*
   capacity sweep in `go1_capacity.py` (7 payloads × 1500 steps, re-JITs `reset/step/policy`
   inside `at_payload` every payload, `go1_capacity.py:49`) would be much faster with a vmapped
   rollout. The re-JIT per payload is wasted compile time.

3. **EMA via `exp(-dt/tau)`** (`go1_capacity.py:39`, `render_walk.py:39`) — fine, but the
   whole adaptive controller is a textbook clipped-integral controller that could be a tiny
   reusable class rather than inlined twice.

4. **`fit_linear_spring`** (`metrics.py:64-78`) hand-rolls `np.polyfit(…,1)` then derives
   `k=-a, theta0=-b/a`. That is the correct minimal thing and `np.polyfit` *is* the library
   call — fine. Noting only that there is no weighting/robustness; acceptable for screening.

5. **Domain randomization** (`payload.py`) — correctly *extends* Playground's randomizer rather
   than reinventing it, and documents that every other term is byte-identical to stock. Good.
   One nit: the stock Go1 randomizer is re-typed in full (friction/armature/ipos/mass/qpos0)
   rather than imported and wrapped, so it must be kept in sync by hand with Playground
   (`payload.py:33-67`). If Playground changes its randomizer, this silently diverges.

---

## Bugs / inconsistencies

1. **`metrics.evaluate`: `tq = td = []` aliases two names to the *same* list object**
   (`metrics.py:179`). Harmless today because both are immediately reassigned from
   `joint_addrs(...)` when `spec is not None` (`metrics.py:183`) and otherwise unused, but it
   is a latent aliasing bug: any future `tq.append(...)` would mutate `td` too. Should be
   `tq, td = [], []` or just drop the line. Also `tq`/`td` are computed at line 183 and then
   `tq2`/`td2` are recomputed identically at line 194 for the same `joint_substr` — redundant
   second `joint_addrs` call.

2. **`experiment.rollout` drops the first post-`done` state but also silently shortens the
   trajectory differently from `rollout.py`.** In `experiment.py:97-102` the order is
   `step → if done: break (before append) → append`, so the terminating step is *not* recorded
   and a 0-length trajectory is possible if the env reports `done` on step 0 (then
   `np.stack([])` at `experiment.py:104` raises). In `rollout.py:80-84` the order is
   `append → if done: break`, so the terminating step *is* recorded. The two rollout copies
   therefore log different windows for the same policy — a reproducibility inconsistency. (The
   `_trim` later drops the transient, masking it in the common case.)

3. **`render_walk.py:32` mutates a read-only-ish numpy view on the CPU model that has no effect
   on the JAX physics.** `mj.hfield_size[:, 2] *= cfg.terrain_height_scale` scales the
   *rendering* model only; the dynamics run on `go1._mjx_model`, whose hfield is scaled
   separately at env-build time in `env.py:42-44`. So the comment "match render terrain to
   scaled dynamics" is correct *by luck* (both get scaled), but if `make_env` ever stops
   scaling `_mjx_model`, the render and physics terrain would silently disagree. The two
   scalings should share one code path.

4. **`SpringWrapper`/`PreloadDRWrapper` never clear `qfrc_applied`, and assume the base env
   does.** Each step does `qfrc_applied.at[dof].set(tau)` (`env.py:122-123`, `163-164`). If the
   underlying Playground env does **not** reset `qfrc_applied` to zero each step, the *other*
   DoFs' applied forces would persist (here they stay 0, so OK), and more importantly the set
   value persists into the next control step's substeps — which is the intended "hold across
   substeps" behavior, but it relies on Playground zeroing `qfrc_applied` at reset and on no
   other code writing it. This is undocumented coupling; a defensive `.at[dof].set` from a
   zeroed base each step would be safer. Worth a one-line assertion/test that
   `qfrc_applied` is zero for non-target DoFs after a step.

5. **Two different "actuated DoF" conventions coexist.** `ElectricalRewardWrapper` slices
   `qfrc_actuator[..., 6:]` and `qvel[..., 6:]` (`env.py:84-85`), assuming the first 6 DoFs are
   the free base and *all* remaining DoFs are actuated and contiguous. `metrics.actuated_dof_adrs`
   instead enumerates `jnt_dofadr >= 6` (`metrics.py:29-32`). For the G1/Go1 these agree, but if
   any model has an unactuated non-base joint (e.g. a passive wheel/ball joint), the training
   reward would penalize a DoF the metric excludes — the reward and the eval metric would no
   longer be on the same quantity (the wrapper's whole justification, `env.py:64`). Use the same
   helper in both.

6. **`ElectricalRewardWrapper` penalizes `qfrc_actuator` including the spring DoFs' motor torque
   — correct — but its `omega` is `qvel[...,6:]`, raw joint velocity, while the eval metric uses
   the same. Consistent. No bug; flagging that the reward uses the *no-regen* clamp
   (`jnp.maximum(..., 0)`, `env.py:86`) per-DoF-per-step, matching eval. Good.**

7. **`analyze.py` and `motor_budget.py`/`power_compare.py` still print "PLACEHOLDER" Kt/R**
   (`analyze.py:61,96`, etc.) while `energy.py:55` now documents these as *estimates*, not
   placeholders, and the headline result is stated to be R/Kt²-invariant. The "PLACEHOLDER"
   labels are stale and undersell the (now-justified) constants — cosmetic but misleading in
   saved logs.

8. **`go1_capacity.py:31` and `render_walk.py:33` reach into `env._env._env...` via
   `while hasattr(go1, "_env")`** to find the base env holding `_mjx_model`. Fragile: it depends
   on the wrapper nesting order (`SpringWrapper`/`PreloadDR` → `ElectricalReward` → base). If the
   wrapper order in `make_env` (`env.py:45-54`) changes, the mass-injection target could change.
   A `base_env` property on the wrappers would make this robust.

9. **`recalc_rkt2.py` applies a *single* whole-body R/Kt² to all 29 DoFs** (`recalc_rkt2.py:74`,
   `block(..., c)`) and itself notes this is "an approximation". Combined with the fact that
   `energy.MOTORS` only has one G1 row used for every joint (`energy.py:54` admits hip-yaw/ankle
   differ), the absolute whole-body numbers mix motor types. The doc strings acknowledge this,
   so it's a known limitation rather than a hidden bug — but the *only* defensible output is the
   ratio, and `power_compare.py`/`motor_budget.py` still print absolute watts prominently.

10. **`motor_budget.py:59` contains dead `if False else` scaffolding**
    (`mujoco.mj_id2name(...) if False else mujoco.mj_id2name(model, ...)`) — leftover debugging,
    should be removed.

11. **`SpringConfig`/`SpringSpec` carry `theta0`/`p1`/`p2` but `springs.tau_spring` ignores
    `theta0` for `constant` and ignores `tau0` for `linear`/`semiparabolic`** — silently. A
    config with `kind: linear, tau0: -12` (easy copy-paste error from `spring_constant.yaml`)
    would apply *no* preload with no warning. Consider validating that only the active kind's
    params are nonzero in `from_config` (`springs.py:34-43`).

---

## Energy / spring correctness

**Energy model — correct and internally consistent.**
- Copper loss `(τ/Kt)²·R` (`energy.py:128-134`) and electrical power
  `τ·ω + ohmic`, no-regen via per-element `max(·,0)` (`energy.py:141-146`) match the project
  spec in CLAUDE.md and `docs/running_program.md`.
- The **no-regen clamp granularity is consistent everywhere**: it is applied to the
  `(T, k)` array *before* summing in `metrics.power_breakdown` (`metrics.py:86`), the reward
  wrapper (`env.py:86`, per-DoF-per-step), `power_compare.py:64`, `motor_budget.py:37`,
  `analyze.py:87`. So "no-regen" consistently means "each actuator, each instant, cannot return
  power" — the strict, physically-motivated reading. Good; this is the kind of thing that is
  easy to get subtly different between training and eval, and here it matches.
- CoT `E/(m g d)` (`energy.py:154-156`, `metrics.py:205`) is standard. `metrics.evaluate`
  computes `dist` as planar `‖xy_T − xy_0‖` (`metrics.py:203`), robust to randomized initial
  heading — correct and matches the rollout/analyze note.
- **Mean-power normalization is correct but obscure.** `power_breakdown` uses
  `s = dt/(len*dt) = 1/T` then `sum(power)*s` (`metrics.py:85`), i.e. `mean(power)` — correct,
  just a convoluted way to write `power.mean(0).sum()`. Same idiom copied in every script.

**Spring injection — correct, with sign convention consistent across in-loop and post-hoc.**
- In-loop: `SpringWrapper` adds `+τ_spring(θ)` through `qfrc_applied` (external generalized
  force), leaving `qfrc_actuator` as the pure motor torque (`env.py:117-124`). This is the right
  channel — the energy model reads `qfrc_actuator`, so the passive spring is correctly excluded
  from the motor's electrical cost. This is the single most important correctness property of
  the whole study and it is implemented correctly.
- Post-hoc: `subtract_spring` / `analyze.posthoc` compute the motor's *residual* torque as
  `τ_motor − τ_spring(θ)` (`metrics.py:60`, `analyze.py:45`). Sign is consistent: in-loop the
  spring *adds* assistive torque, so on a fixed gait the motor would have to supply that much
  *less* — subtracting is the matching post-hoc operation. ✔
- **Sign-convention caveat (not a bug, but verify per config):** `springs.tau_spring` returns
  `−k(θ−θ0)` for linear and `+tau0` for constant (`springs.py:58-67`). Whether this *assists* or
  *fights* the motor depends entirely on the sign of `tau0`/the side of `θ0` relative to the gait,
  in the joint frame. `spring_constant.yaml` uses `tau0: -12` and `spring_go1.yaml` `tau0: +3.5`
  — opposite signs, because "knee" vs "calf" joint frames differ. There is no in-code check that
  the configured preload actually offloads (rather than adds to) the measured support torque;
  the configs rely on hand-derived signs from the work-loop fits. A one-line guard (post-hoc:
  assert the spring reduces RMS motor torque) would catch a sign flip.

**Saturation / limits model (`metrics.saturation`, `energy.MotorLimits`) — sound.** Reads the
real enforced `jnt_actfrcrange` torque cap and pairs it with URDF velocity limits, judges the
binding wall from the median of the top-5%-power window rather than the noisy max
(`metrics.py:153-164`). This robust-percentile choice is well-reasoned and matches the B1
analysis in CLAUDE.md.

**One physics caveat worth a line in the model docstring (already partly noted):** the energy
model omits iron loss (`energy.py:18-25` documents this) and uses joint-side constants for *all*
DoFs. Both bias the % savings conservatively (larger denominator), so the headline reductions are
lower bounds — acceptable and honestly documented.

---

## Recommendations (prioritized)

**P0 — correctness, do before trusting more numbers**
1. Fix `metrics.evaluate:179` `tq = td = []` aliasing and remove the redundant second
   `joint_addrs` call (lines 183 vs 194).
2. Unify the `done`/append ordering between `rollout.py` and `experiment.rollout`
   (`experiment.py:97-102`) so the two log identical windows; guard against 0-length
   `np.stack` when the env terminates on step 0.
3. Add a post-hoc sign guard: assert a configured spring *reduces* RMS motor torque at the
   target joint (catches a `tau0`/`theta0` sign flip between knee/calf frames).

**P1 — kill the duplication that has already drifted**
4. Promote the adaptive per-leg preload controller into `src/pea/` (one `AdaptivePreload`
   stepper) and have `go1_capacity.py` + `render_walk.py` call it — it is the project's
   headline mechanism living in two copy-pasted script blocks.
5. Extract one shared `rollout()` helper (with an optional per-step callback for the preload)
   and delete the 5–6 near-duplicate loops.
6. Make the script-side energy math call `metrics.power_breakdown`; delete the `block`/
   `per_dof_elec` reimplementations in `power_compare.py`, `motor_budget.py`, `recalc_rkt2.py`,
   `probe_speed_hold.py`.

**P2 — robustness / hygiene**
7. Replace the `while hasattr(go1, "_env")` unwrapping and the `mj.hfield_size` double-scaling
   with a `base_env` property / shared terrain-scale helper on the wrappers.
8. Make `ElectricalRewardWrapper` use `actuated_dof_adrs` instead of the `[...,6:]` slice so the
   reward and eval metric provably cover the same DoFs.
9. Replace hardcoded Drive paths and hardcoded `Kt=2.3,R=0.013` in the four G1 scripts with
   `config.resolve_runs_dir()` and `energy.motor_constants(cfg.energy_motor)`.
10. Have `render_walk.py` reuse `rollout.render_video` instead of the hand-piped ffmpeg.
11. Cosmetic: drop the dead `if False else` in `motor_budget.py:59`; refresh the stale
    "PLACEHOLDER Kt/R" labels (now estimates, not placeholders); validate spring configs so
    inactive-kind params (`tau0` on a linear spring, etc.) can't be silently ignored.
