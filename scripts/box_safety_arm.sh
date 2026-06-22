#!/usr/bin/env bash
# Box-side safety: arm a HARD self-destruct timer + an idle dead-man's switch BEFORE launching any
# training on a billing immers.cloud box. Defense-in-depth against the 2026-06-22 overnight idle-bill
# incident (agent froze on a permission prompt + box went unreachable -> bill all night, zero training).
# See docs/incident_2026-06-22_overnight_billing.md.
#
# RUN ON THE BOX, right after bootstrap, BEFORE `pea-train`.
# Usage: box_safety_arm.sh [MAX_HOURS] [IDLE_MIN] [GRACE_MIN]
#   MAX_HOURS  hard wall-clock ceiling, power off no matter what (default 5)
#   IDLE_MIN   power off if no pea-train process for this long (default 20)
#   GRACE_MIN  delay before the idle check starts, to allow bootstrap+first launch (default 15)
#
# CAVEAT (read this): `poweroff` stops compute but on most clouds the VM stays ALLOCATED and KEEPS
# BILLING (immers rule: "DELETE, not stop"). So this is harm-reduction + a signal (the box goes
# unreachable -> the agent watchdog detects the stall), NOT a guaranteed billing stop. The only
# reliable billing stop is a console/API DELETE. Verify whether immers bills a powered-off instance;
# if it does NOT, this becomes a full fix.
set -uo pipefail

MAX_HOURS=${1:-5}
IDLE_MIN=${2:-20}
GRACE_MIN=${3:-15}
MAX_SEC=$(( MAX_HOURS * 3600 ))
IDLE_SEC=$(( IDLE_MIN * 60 ))
GRACE_SEC=$(( GRACE_MIN * 60 ))
STAMP=$HOME/.box_safety
mkdir -p "$STAMP"

# Decide how we power off. Prefer passwordless sudo; warn loudly if unavailable (then the timers
# cannot fire and the box-side safety is INERT -- the agent watchdog + console become the only nets).
if sudo -n true 2>/dev/null; then
  POWEROFF="sudo poweroff"
  echo "[box_safety] passwordless sudo OK -> poweroff is '$POWEROFF'"
else
  POWEROFF="poweroff"
  echo "[box_safety] WARNING: no passwordless sudo. '$POWEROFF' may fail -> box-side self-destruct is"
  echo "[box_safety] WARNING: likely INERT. Do NOT rely on it; keep the console/API kill as primary."
fi

# 1) HARD self-destruct: power off after MAX_HOURS regardless of training state.
nohup bash -c "sleep $MAX_SEC; logger 'box_safety: HARD timeout ${MAX_HOURS}h -> poweroff'; \
  echo \$(date) hard-timeout >> $STAMP/fired; $POWEROFF" >/dev/null 2>&1 < /dev/null &
echo "[box_safety] armed HARD self-destruct: poweroff in ${MAX_HOURS}h (pid $!)"

# 2) IDLE dead-man's switch: after GRACE, poweroff if no pea-train for IDLE_SEC.
nohup bash -c "
  sleep $GRACE_SEC
  last=\$(date +%s)
  while true; do
    if pgrep -f '[p]ea-train' >/dev/null; then last=\$(date +%s); fi
    now=\$(date +%s)
    if [ \$(( now - last )) -ge $IDLE_SEC ]; then
      logger 'box_safety: no pea-train for ${IDLE_MIN}min -> poweroff'
      echo \$(date) idle-timeout >> $STAMP/fired
      $POWEROFF
      exit 0
    fi
    sleep 60
  done" >/dev/null 2>&1 < /dev/null &
echo "[box_safety] armed IDLE dead-man: poweroff if no pea-train for ${IDLE_MIN}min (grace ${GRACE_MIN}min, pid $!)"

echo "[box_safety] ARMED. Remember: poweroff may NOT stop immers billing -- DELETE from console/API when done."
