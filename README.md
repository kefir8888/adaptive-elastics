# adaptive-elastics — Parallel-Elastic Efficiency Study

Does a tunable **parallel elastic spring** reduce the *electrical* energy (cost of transport) of legged
locomotion by offloading motor torque? Motor ohmic loss scales with torque squared (`P ≈ (τ/Kt)²·R`), so
offloading torque to a passive spring cuts heat quadratically — **if the gait can exploit it and the gearing
makes ohmic loss matter.** MuJoCo end-to-end (Playground + MJX + brax PPO); the metric is electrical energy /
cost of transport, not mechanical work. Full design in `CLAUDE.md`.

> **▶ Resuming / new session?** Start with **`docs/NEXT_SESSION.md`** — the actionable handoff plan for the
> next campaign (the dog-running knee-spring experiment on immers.cloud).

## Headline finding: gearing is the crux

| Platform | Gear ratio | Ohmic share of budget | Parallel-spring verdict |
|--|--|--|--|
| Unitree **G1** humanoid | 22.5:1 (high) | ~4 % | **Negative** — in-loop hip spring **+7 % worse** for walking (9 catalogued negatives) |
| Unitree **Go1** quadruped | 6.33:1 (low) | ~54 % | **Positive** — constant knee (calf) preload cuts CoT **−17 to −27 %** (3 seeds) |

On the low-gear Go1, ohmic loss dominates, so offloading the knee's support torque pays. On the high-gear G1
ohmic is negligible and the always-on spring just fights the gait. **The gear ratio decides the sign.**

## Current state (2026-06-16)

**Done + validated — Go1 quadruped (the positive track):**
- **Constant parallel knee preload cuts cost of transport −17 to −27 %** on flat ground, **3 seeds**
  (seeds 1 & 3 strong, seed 2 weaker at low load). The element must be a CONSTANT preload, not a linear
  spring — the knee work-loop is offset-dominated, so a linear fit degenerates to k ≈ 0.
- **Adaptive per-leg self-tuning preload** — a slow clipped-proportional loop ramps each leg's preload to
  offload its own measured knee torque, with **no load sensor and no payload observation**. Validated; extends
  the win to **load-carrying (0–6 kg payloads)**, where the CoT benefit **grows with load**.
- **Walking videos** rendered (flat + rough, no-load + 5 kg) in `outputs/videos/`.

**Caveats / negative / inconclusive:**
- **Rough terrain:** the energy win survives on mild (2.5 cm) terrain (CoT −10 to −19 %) but **stability is
  poor** (~40 % survival) — rough locomotion is hard for the blind loaded dog, spring or not. Full 5 cm was
  inconclusive (under-powered, dropped).
- **Capacity realism (validity catch):** the warm-started sim Go1 "walks" at 30 kg, but the **real Go1 carries
  ~5–10 kg max** — the plain sim enforces peak but not continuous/thermal torque or structural/balance limits.
  High-load (15–30 kg) numbers are **sim-only**; 30 kg is realistic only for a B2-class robot. The physically
  meaningful Go1 study is **0–6 (10) kg**.
- **G1 running — two failed attempts:** a [0,3] m/s from-scratch collapse to a 0.85 m/s never-fall walk, then
  a curriculum + reward-redesign that destabilized it. The Playground G1 walk env structurally resists a flight
  phase (gait clock + hard-capped air-time). G1 running is a **long-shot** (`docs/g1_running_design.md`).

**Active direction:** **knee spring on a RUNNING dog** (`docs/dog_running_design.md`). The Go1 env is far more
flight-permissive than the G1 (no gait clock, soft termination), and a quadruped runs with a real flight phase
so the calf can recover braking energy at impact — likely favouring a one-sided *stiffness* spring over the
walking constant-preload. Designed; gated on whether a true all-feet-off phase actually emerges.

**Infrastructure:** GPU box is **off** (all results synced locally); no new training until a box is provisioned.

## Methodology & process discipline
Hard-won rules from the project's *own* failures (full analysis in `docs/research_patterns.md`). Follow them;
most wasted time and every retracted claim traces to breaking one of these.
1. **Cheap feasibility probe before every full RL run.** The dominant time-sink was the
   train→collapse→diagnose→retrain loop. Probe first — a short rollout, a torque-budget check, a narrow-range
   smoke — before committing a multi-hour run.
2. **Check physical realism *before* choosing ranges.** The 0–25 kg payload (collapsed training) and the 30 kg
   "capacity" (unphysical for a 12 kg robot) both came from ranges set before checking the real robot's limits.
   Ground payloads/torques in the hardware spec first.
3. **One change per training stage; warm-start and gate.** Both G1-running failures changed the command range
   *and* several rewards at once, from scratch. Change one thing, warm-start from the last good policy, and gate
   on a measured metric before the next stage.
4. **Keep eval scripts in lockstep with the experiment.** The capacity sweep was capped at 15 kg while policies
   trained past it; `tracking_lin_vel` was twice mistaken for forward speed. Evaluate *at* the trained condition,
   and measure the quantity you claim (forward speed, not tracking reward).
5. **Sync after every run; never halt before syncing.** A trained baseline was lost when a box was deleted
   mid-sync. rsync results off the box after each run.
6. **Report cost of transport (W ÷ m/s), with per-seed spread and the stability cost — never raw watts.** The
   spring makes the policy walk faster, so watts confound speed; CoT is the only fair metric. Report the band
   across seeds and the survival/stability cost, not a single optimistic number.
7. **One active direction at a time.** Six parallel directions diluted focus; commit (currently: the running dog).

## Docs map
- `CLAUDE.md` — full experiment design + project rules (**start here for design**).
- `docs/JOURNAL.md` — dated session history (**start here for "what happened"**).
- `docs/RESULTS.md`, `docs/negative_results.md` — results + the catalogued negatives (a headline output).
- `docs/load_program.md` — load-carrying adaptive-preload program + the capacity-realism analysis.
- `docs/g1_running_design.md`, `docs/dog_running_design.md` — the running-spring designs (G1 hard, dog active).
- `docs/directions.md`, `docs/taxonomy.md`, `docs/running_program.md` — direction / cross-morphology maps.
- **Audit / research / writing outputs:** `docs/code_audit.md`, `docs/docs_audit.md`, `docs/weak_spots.md`,
  `docs/research_patterns.md`, `docs/literature_review.md` (74 cites), `docs/g1_running_research.md`,
  `docs/gpu_cost_crypto.md`, `docs/presentation.md` (+ `outputs/slides/presentation.pptx`), `paper/paper.tex`
  (IEEE Access draft), `outputs/figures/cot_vs_load.png`.

## Stack & layout
MuJoCo Playground (Go1/G1 joystick envs) + MJX (GPU) for training on a rented H100; CPU MuJoCo/JAX locally for
rollout, analysis, and video rendering. brax PPO, 8192 parallel envs. Layers: `src/pea/` (core: env, springs,
energy, policy, config, payload, train, metrics), `scripts/` (train/eval/render/probe), `configs/` (one YAML
per arm), `docs/`, `outputs/` (gitignored — one folder per run).

### Run training on a GPU box
```sh
curl -fsSL https://raw.githubusercontent.com/kefir8888/adaptive-elastics/main/scripts/gpu_box_setup.sh | bash
cd ~/adaptive-elastics && git checkout experiments-2026-06-14
uv run pea-train --config configs/<arm>.yaml --output_dir ~/runs --suffix <tag>
# then eval: uv run python scripts/go1_capacity.py <run_dir> 1500
```
