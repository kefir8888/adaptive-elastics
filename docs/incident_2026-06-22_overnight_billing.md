# Incident post-mortem — overnight idle GPU billing (2026-06-22 → 06-23)

## Summary
A rented immers.cloud H100 billed for an entire night and **trained nothing**. Two faults stacked,
plus one enabling condition:

1. **PRIMARY — permission-prompt freeze.** A VS Code permission prompt opened and **suspended the
   whole agent loop** overnight. The loop only advances on a user message or a tracked-task
   completion; a blocking prompt halts even that. Proof: the two background-task failure
   notifications were delivered only the next morning, the instant the user dismissed the prompt.
2. **SECONDARY — non-durable connectivity.** The box became unreachable (ping 100 % loss, TCP 22
   timeout) within ~1–2 minutes of the *one* successful SSH probe, and stayed unreachable for >24 h.
   The immers VPN exit is documented-flaky; a single probe is not a durable link.
3. **ENABLING — no remote kill-switch.** Control was SSH-only. With SSH down and **no immers
   API/CLI destroy method**, the bill could not be stopped from any channel the agent had. Only a
   human deleting the instance from the immers web console stops it.

Net: the box never bootstrapped, never received the s4 warm-start, never started a `pea-train`
process, and could not be stopped by the agent — full bill, zero training.

## Timeline
- User provisioned the box, pasted the IP. Agent fixed an SSH host-key issue; **one** probe
  succeeded (NVIDIA H100 NVL 95.8 GB, host `pea-h100`).
- Agent launched bootstrap (`gpu_box_setup.sh`) and the s4 upload as **fire-and-forget background
  tasks** and **ended its turn**, intending to be re-invoked on their completion.
- Connectivity to the box died. Every SSH/rsync retry timed out.
- A permission prompt opened in VS Code → the agent loop **froze**.
- Background tasks failed within ~1–2 min; their failure notifications **queued** (loop frozen).
- Box billed all night doing nothing.
- Next morning: user dismissed the prompt → loop resumed → queued failure notifications arrived.

## Root causes (ranked)
1. **Permission-prompt freeze (primary).** A `.claude/settings.local.json` edit had added a bypass
   mode + allowlist, but **that file is read only at session start** — a mid-session edit does NOT
   change the running session's permission mode, and a separate auto-mode classifier gates tool
   calls regardless. So a prompt still appeared and still froze the loop. The settings fix gave a
   false sense of safety.
2. **One probe treated as a durable link (secondary).** Over a known-flaky VPN, the agent assumed
   connectivity would persist and launched unattended work against that assumption.
3. **SSH-only control (enabling).** No control channel independent of the VPN/SSH. An unreachable
   billing resource with no out-of-band kill is, by construction, an uncancellable bill.
4. **Fire-and-forget then end-turn while a box bills (process).** No watchdog, no self-scheduled
   re-check, no box-side dead-man's switch, no hard cap — nothing actively noticed or acted.

## Conclusions / durable lessons
1. **Never end a turn while a box bills unless THREE safeguards are simultaneously armed:**
   (a) verified-**live** bypass mode, (b) an active **watchdog** (`ScheduleWakeup`) re-checking
   reachability + training progress, and (c) a **box-side self-destruct/idle-poweroff** that fires
   with no action from the agent. The box-side timer is the only safeguard that does not depend on
   the agent being alive, reachable, or unblocked — it is **mandatory, not optional**.
2. **A settings-file edit is not a live permission change.** Bypass mode must be toggled in the
   running session (Shift+Tab) or the session restarted, then **verified empirically** (run a
   command that previously prompted; confirm no prompt) before any unattended billing run.
3. **One SSH probe is not connectivity.** Confirm the link with multiple probes over time; design so
   the run survives a link drop (box-side `nohup`, `metrics.jsonl` as the source of truth) **and**
   so a dropped link cannot leave the box billing (box-side timer + out-of-band kill).
4. **SSH-only control is insufficient for a billing resource.** The single highest-value fix is a
   control channel independent of the VPN/SSH — an **immers API/CLI destroy token** — so an
   unreachable box can always be killed. **Until that exists, an unattended billing run requires a
   human watching the immers console**, because (see caveat) a box cannot reliably stop its own bill.

### Hard caveat: a box cannot reliably stop its own bill
`sudo poweroff` from inside the instance stops the OS/compute, but on most cloud providers the VM
remains *allocated* and **keeps billing** (the README's rule is "DELETE the server, not stop" —
stop still bills). So the box-side self-destruct is **harm-reduction** (stops wasted compute, and
the resulting unreachability is a signal the watchdog detects), **not a guaranteed billing stop**.
The only reliable billing stop is a console/API **DELETE**. Therefore the runbook requires *either*
an API destroy token *or* a human at the console for any unattended run. (Verify on the next box
whether immers bills a powered-off instance; if it does NOT, the box-side timer becomes a full fix.)

## Hardened unattended-run runbook

**Pre-flight — all must pass before provisioning/spending:**
- [ ] **Bypass mode verified LIVE** in this session (test a command that previously prompted; no prompt).
- [ ] **VPN up; box reachable on 3 consecutive probes over ~30 s** (`ping` + `nc -z … 22` + `ssh echo`).
- [ ] **Remote kill path exists:** an immers API/CLI destroy token configured locally, **OR** the user
      is watching the console for the duration (box-side timer alone does not guarantee billing stop).

**At launch, ON THE BOX, arm BEFORE any training** (`scripts/box_safety_arm.sh`):
- [ ] **Hard self-destruct timer:** detached process that powers off after a wall-clock ceiling
      (default 5 h) regardless of state.
- [ ] **Idle dead-man's switch:** after a grace period, powers off if no `pea-train` for >N min
      (default 20).
- [ ] **Launch training detached:** `nohup nice -19 ionice -c3 uv run pea-train … > log 2>&1 &` so it
      survives SSH drops and never starves `sshd`.

**After launch, the agent MUST:**
- [ ] `ScheduleWakeup` every **~240 s** during active phases (keeps prompt cache warm, <300 s); each
      tick probe: reachable? `pea-train` alive? `metrics.jsonl` line-count grown since last tick?
- [ ] On a stall (unreachable, or process gone, or metrics flat for 2 ticks): attempt reconnect once;
      if still bad, **escalate to the user loudly** and rely on the box-side self-destruct + console.
      Do NOT silently keep sleeping.
- [ ] **Hard iteration cap** on the wake-loop (abort after expected run duration + margin).
- [ ] Never sleep at exactly 300 s (cache-miss worst case): 240 s warm, or ≥1200 s for genuinely idle.
- [ ] On completion: rsync results → **DELETE the box from the console/API** → confirm it's gone.

**Residual risks to track:**
- `poweroff` ≠ billing stop on immers (storage/allocation may bill) — only console DELETE fully stops.
- The watchdog itself can be frozen by a prompt → hence *bypass-mode-live* is a hard precondition AND
  the box-side timer is agent-independent.
- A partial bootstrap (interrupted `uv sync`) leaves the box up but unable to train → the watchdog's
  "`pea-train` alive" check catches this within one tick; the idle dead-man's switch powers it off.

## Action items
- [ ] **Obtain an immers API/CLI destroy token** (top priority — converts an uncancellable bill into a
      one-command kill). Until then, unattended runs need a human at the console.
- [x] `scripts/box_safety_arm.sh` — box-side hard timer + idle dead-man's switch (this commit).
- [x] README "GPU-box safety" updated with the three-safeguard rule and the live-bypass precondition.
- [x] Memory `no-idle-billing-while-blocked` updated with the prompt-freeze mechanism.
