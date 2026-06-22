#!/usr/bin/env python3
"""Drive the rented immers GPU box over SSH from the Mac.

A thin, dependency-free CLI wrapper around ssh/scp/rsync for the parallel-elastic
training runs, so the box can be driven without hand-typing long SSH commands —
and as a fallback for when Claude's Bash tool is blocked (e.g. the auto-mode
safety classifier is transiently unavailable). Run it yourself and paste the
output back, or use it standalone.

Examples:
  python3 scripts/box.py status                 # run dir, uptime, trainer, metrics, GPU, log tail
  python3 scripts/box.py metrics energy_spring  # parse the latest matching run's last eval row
  python3 scripts/box.py launch-spring          # launch the energy spring arm detached
  python3 scripts/box.py arm-safety 6 30 20     # (re)arm box-side self-destruct: 6h hard, 30m idle, 20m grace
  python3 scripts/box.py pull-runs              # rsync both energy run dirs to the local outputs folder
  python3 scripts/box.py watch energy_spring    # poll until the trainer exits (local loop, flaky-VPN tolerant)
  python3 scripts/box.py exec "df -h /"         # run an arbitrary command on the box
  python3 scripts/box.py kill                   # pkill the trainer
  python3 scripts/box.py poweroff               # stop compute (DELETE from the immers console to stop billing)

Repoint without editing: BOX_IP=1.2.3.4 BOX_USER=ubuntu python3 scripts/box.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HOST = os.environ.get("BOX_IP", "195.209.215.115")
USER = os.environ.get("BOX_USER", "ubuntu")
SSH_OPTS = [
    "-o", "IPQoS=none",            # flaky immers VPN: don't let QoS bits stall the handshake
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",         # never hang on an interactive prompt
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=15",
]
REPO = "~/adaptive-elastics"
S4 = "~/runs/2026-06-21_g1_clean_s4_bound_clean_s4"      # bounding warm-start (both arms)
PATHX = "export PATH=$HOME/.local/bin:$PATH"             # uv isn't on a non-login shell's PATH
LOCAL_RUNS = "outputs/clean_curriculum/fair_centered"    # where pulled runs land locally

# Fields worth surfacing from a metrics.jsonl eval row (energy is folded into the
# TOTAL reward by ElectricalRewardWrapper, so there is no separate energy term).
METRIC_KEYS = [
    "num_steps", "training/sps", "eval/episode_reward", "eval/avg_episode_length",
    "eval/episode_reward/tracking_lin_vel", "eval/episode_reward/hop_flight",
    "eval/episode_reward/hop_push", "eval/episode_reward/termination",
    "eval/episode_reward/orientation", "eval/episode_reward/feet_slip",
]


def _target() -> str:
    return f"{USER}@{HOST}"


def ssh(cmd: str, retries: int = 6, check: bool = False) -> subprocess.CompletedProcess:
    """Run one remote command, retrying through transient VPN drops."""
    last = None
    for i in range(retries):
        try:
            last = subprocess.run(
                ["ssh", *SSH_OPTS, _target(), cmd],
                capture_output=True, text=True, timeout=150,
            )
            if last.returncode == 0:
                return last
        except subprocess.TimeoutExpired as e:
            last = subprocess.CompletedProcess(e.cmd, 124, "", "ssh timeout")
        if i < retries - 1:
            time.sleep(6)
    if check and (last is None or last.returncode != 0):
        sys.stderr.write(f"[box] ssh failed after {retries} tries: {cmd}\n")
        if last:
            sys.stderr.write(last.stderr)
    return last


def rsync(src: str, dst: str, retries: int = 5) -> int:
    """rsync over the same ssh opts, retrying through VPN drops."""
    e = " ".join(["ssh", *SSH_OPTS])
    for i in range(retries):
        r = subprocess.run(["rsync", "-az", "--partial", "-e", e, src, dst])
        if r.returncode == 0:
            return 0
        if i < retries - 1:
            time.sleep(8)
    return 1


# Detached training launch: nice/ionice low priority so sshd never starves; nohup
# so it survives an SSH/VPN drop; metrics.jsonl is the source of truth.
def _launch_cmd(suffix: str) -> str:
    return (
        f"cd {REPO} && {PATHX} && nohup nice -n 19 ionice -c3 uv run pea-train "
        f"--config configs/g1_bound_{suffix}.yaml --restore {S4} "
        f"--output_dir ~/runs --suffix {suffix} > ~/{suffix}.log 2>&1 < /dev/null & echo PID=$!"
    )


def _status_cmd(glob: str) -> str:
    return (
        f'R=$(ls -dt ~/runs/*{glob}* 2>/dev/null | head -1); echo "RUN=$(basename "$R")"; '
        f'uptime; echo "trainer:"; pgrep -af "[p]ea-train --config" || echo "  (none)"; '
        f'echo "metrics_lines=$(wc -l < "$R/metrics.jsonl" 2>/dev/null)"; '
        f'echo "GPU:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; '
        f'echo "last_metric:"; tail -1 "$R/metrics.jsonl" 2>/dev/null'
    )


def cmd_status(args):
    r = ssh(_status_cmd(args.glob))
    print(r.stdout if r else "(unreachable)")
    if r and r.stderr.strip():
        print("STDERR:", r.stderr.strip())


def cmd_metrics(args):
    r = ssh(f'R=$(ls -dt ~/runs/*{args.glob}* 2>/dev/null | head -1); tail -1 "$R/metrics.jsonl"')
    if not r or not r.stdout.strip():
        print("(no metrics yet / unreachable)")
        return
    try:
        row = json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        print("could not parse metrics row:\n", r.stdout[:400])
        return
    print(f"=== latest eval row ({args.glob}) ===")
    for k in METRIC_KEYS:
        if k in row:
            print(f"  {k:42s} {row[k]:.3f}" if isinstance(row[k], float) else f"  {k:42s} {row[k]}")
    print("  (energy penalty is folded into eval/episode_reward, not logged separately)")


def cmd_exec(args):
    r = ssh(args.command)
    if r:
        sys.stdout.write(r.stdout)
        if r.stderr.strip():
            sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
    print("(unreachable)")
    sys.exit(1)


def cmd_launch_baseline(args):
    print(ssh(_launch_cmd("energy_baseline")).stdout)


def cmd_launch_spring(args):
    print(ssh(_launch_cmd("energy_spring")).stdout)


def cmd_arm_safety(args):
    # scp the script then arm it (hard self-destruct + idle dead-man).
    subprocess.run(["scp", *SSH_OPTS, "scripts/box_safety_arm.sh", f"{_target()}:~/box_safety_arm.sh"])
    r = ssh(f"chmod +x ~/box_safety_arm.sh && bash ~/box_safety_arm.sh {args.max_h} {args.idle_min} {args.grace_min}")
    print(r.stdout if r else "(unreachable)")


def cmd_pull(args):
    sys.exit(rsync(f"{_target()}:{args.remote}", args.local))


def cmd_push(args):
    sys.exit(rsync(args.local, f"{_target()}:{args.remote}"))


def cmd_pull_runs(args):
    os.makedirs(LOCAL_RUNS, exist_ok=True)
    rc = 0
    for suffix in ("energy_baseline", "energy_spring"):
        # resolve the newest matching remote run dir, then pull it
        r = ssh(f'ls -dt ~/runs/*{suffix}* 2>/dev/null | head -1')
        path = (r.stdout.strip() if r else "")
        if not path:
            print(f"  {suffix}: no remote run found")
            continue
        print(f"  pulling {path} -> {LOCAL_RUNS}/")
        rc |= rsync(f"{_target()}:{path}", f"{LOCAL_RUNS}/")
    sys.exit(rc)


def cmd_tail(args):
    r = ssh(f"tail -n {args.n} ~/{args.log}")
    print(r.stdout if r else "(unreachable)")


def cmd_watch(args):
    """Poll status until the trainer exits or it stalls (VPN-tolerant: 8 misses = stall)."""
    prev, flat, miss = -1, 0, 0
    for i in range(args.max_ticks):
        r = ssh(
            f'A=$(pgrep -f "[p]ea-train --config" | wc -l); '
            f'R=$(ls -dt ~/runs/*{args.glob}* 2>/dev/null | head -1); '
            f'N=$(wc -l < "$R/metrics.jsonl" 2>/dev/null || echo 0); echo "ALIVE=$A LINES=$N"',
            retries=1,
        )
        if not r or r.returncode != 0:
            miss += 1
            print(f"[{i}] unreachable x{miss}")
            if miss >= 8:
                print("STALL: unreachable 8x"); sys.exit(3)
            time.sleep(args.interval); continue
        miss = 0
        out = r.stdout.strip()
        alive = int(out.split("ALIVE=")[1].split()[0]) if "ALIVE=" in out else 0
        lines = int(out.split("LINES=")[1].split()[0]) if "LINES=" in out else 0
        print(f"[{i}] {out}")
        flat = 0 if lines > prev else flat + 1
        prev = lines
        if alive == 0:
            print(f"DONE: trainer gone (metrics={lines})"); sys.exit(0)
        if lines > 0 and flat >= 10:
            print(f"STALL: metrics flat (lines={lines})"); sys.exit(4)
        time.sleep(args.interval)
    print("CAP reached"); sys.exit(1)


def cmd_kill(args):
    print(ssh('pkill -9 -f "[p]ea-train --config"; echo killed').stdout)


def cmd_poweroff(args):
    print(ssh("sudo poweroff").stdout or "(poweroff sent — DELETE from console to stop billing)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status"); s.add_argument("glob", nargs="?", default="energy"); s.set_defaults(func=cmd_status)
    s = sub.add_parser("metrics"); s.add_argument("glob", nargs="?", default="energy_baseline"); s.set_defaults(func=cmd_metrics)
    s = sub.add_parser("exec"); s.add_argument("command"); s.set_defaults(func=cmd_exec)
    sub.add_parser("launch-baseline").set_defaults(func=cmd_launch_baseline)
    sub.add_parser("launch-spring").set_defaults(func=cmd_launch_spring)
    s = sub.add_parser("arm-safety")
    s.add_argument("max_h", nargs="?", type=int, default=6)
    s.add_argument("idle_min", nargs="?", type=int, default=30)
    s.add_argument("grace_min", nargs="?", type=int, default=20)
    s.set_defaults(func=cmd_arm_safety)
    s = sub.add_parser("pull"); s.add_argument("remote"); s.add_argument("local"); s.set_defaults(func=cmd_pull)
    s = sub.add_parser("push"); s.add_argument("local"); s.add_argument("remote"); s.set_defaults(func=cmd_push)
    sub.add_parser("pull-runs").set_defaults(func=cmd_pull_runs)
    s = sub.add_parser("tail"); s.add_argument("log", nargs="?", default="baseline.log"); s.add_argument("-n", type=int, default=8); s.set_defaults(func=cmd_tail)
    s = sub.add_parser("watch")
    s.add_argument("glob", nargs="?", default="energy_spring")
    s.add_argument("--interval", type=int, default=60)
    s.add_argument("--max-ticks", type=int, default=140)
    s.set_defaults(func=cmd_watch)
    sub.add_parser("kill").set_defaults(func=cmd_kill)
    sub.add_parser("poweroff").set_defaults(func=cmd_poweroff)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
