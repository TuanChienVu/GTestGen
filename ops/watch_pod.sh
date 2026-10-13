#!/usr/bin/env bash
# Laptop-side fuse of one pod:
#   KNOWN_HOSTS=FILE bash ops/watch_pod.sh IP PORT KEY HEARTBEAT STALE_MIN DEADLINE_UTC  (DRY_RUN=1 prints 3 commands)
# Every POLL seconds (default 120) it reads over the pod's direct SSH port the queue's HEARTBEAT (a pod path, e.g.
# /workspace/queue/heartbeat.json, which jobqueue rewrites every minute while a job runs), started.json and outcome.json
# beside it (jobqueue writes started.json once it holds the pod lock), and whether the on-pod watchdog runs (its marker
# V2GPU_WATCHDOG, matched as "V2GPU_WATCHDO[G]" so the read's own shell never matches itself).
# - Every ssh runs with ConnectTimeout, ServerAliveInterval and ServerAliveCountMax AND under a command timeout
#   (SSH_TIMEOUT, default 60 s; STOP_TIMEOUT 120 s for a stop): a hung read counts as a failed round.
# - An outcome ends the run that wrote started.json when it is not older than started.json nor than the heartbeat
#   (that run has ENDED). It counts as this watch's end only if it is not older than the watch's first read either
#   (an outcome.json of an earlier run, or an old started/outcome pair read before a rerun takes the
#   lock, is never this run's end). Then REFUSED_AT_START is reported (QUEUE_REFUSED, once) and the watch goes on (a
#   refused queue nobody restarts is stopped by the grace rule below); any other outcome prints QUEUE_END. With
#   "self_stop": false (a queue without self_stop_cmd, e.g. the training canary: copy, then stop the pod) the watch
#   ends there (exit 0). Otherwise it does NOT exit before the pod is gone: it probes SSH at once and for
#   STOP_GRACE seconds (default 300) while the queue's own self-stop acts, then retries `runpodctl stop` over SSH up to
#   STOP_TRIES times (default 3, backoff END_POLL x try); SSH refused or closed SSH_FAILS probes in a row prints
#   SSH_CLOSED and CHECK_EXITED_VIA_MCP (exit 0: EXITED is then confirmed with get-pod); a pod that still answers
#   after every try is an ALARM with STOP_VIA_MCP (exit 3).
# - A heartbeat of a queue that runs now (not older than started.json, or than the watch's first read while
#   started.json is unknown, and its run has not ENDED) older than STALE_MIN minutes (the pod's clock), or DEADLINE_UTC
#   passed (this machine's clock), stops the pod over SSH with the pod's own
#   `source /etc/rp_environment; runpodctl stop pod "$RUNPOD_POD_ID"` (STOPPED_BY_WATCH, exit 5). With no such heartbeat
#   (none, unreadable, older, or of an ended run) the watch counts max(STALE_MIN, GRACE_MIN = 15) minutes from the
#   queue start (from the watch's first read when started.json is unknown or its run ended before the watch): a
#   heartbeat.json left by an earlier queue never stops the pod at once. A non-numeric heartbeat is said
#   (HEARTBEAT_UNREADABLE), not fatal.
# - The on-pod watchdog missing (a container or pod restart loses it) is an alarm: WATCHDOG_MISSING, once per loss; the
#   operator re-arms it. The watch goes on.
# - SSH failing SSH_FAILS rounds in a row (default 3), or a stop that fails STOP_TRIES times while SSH still answers,
#   prints STOP_VIA_MCP and exits 3: this laptop has no RunPod API key, so the pod is stopped through MCP
#   (pod-action stop) and confirms EXITED.
# Host keys and ended runs:
# - KNOWN_HOSTS (required) names the pod's own known_hosts file on this laptop (deleted at every
#   (re)start of the pod): every ssh runs with UserKnownHostsFile=$KNOWN_HOSTS and StrictHostKeyChecking=accept-new,
#   so a new pod's first key is taken and a changed key is refused. A read refused for a changed host key (a container
#   restart makes new keys) prints HOST_KEY_CHANGED once: re-arm the watchdog with a fresh known_hosts.
# - A queue whose run ended before this watch began (its outcome.json is not older than started.json and older than
#   the watch's first read; a refused start aside) arms no stop at all: the watch says QUEUE_ALREADY_ENDED once and
#   waits for a new run of the queue (a newer started.json, then the rules above); past the deadline it exits 3 and
#   leaves the pod to a stop through MCP. A watcher restarted after the queue's end never stops a pod mid-copy or during
#   a fuse test.
# A stop is not a delete: the volume stays. SSH_CMD replaces ssh (the tests use a fake). Exit 2: a malformed call.
set -euo pipefail
usage() {
  echo "usage: KNOWN_HOSTS=<the pod's known_hosts file> bash ops/watch_pod.sh IP PORT KEY HEARTBEAT STALE_MIN" \
       "DEADLINE_UTC ($1)" >&2
  exit 2
}
[ $# -eq 6 ] || usage "six arguments"
KNOWN_HOSTS=${KNOWN_HOSTS:-}
[[ "$KNOWN_HOSTS" =~ ^/[^[:space:]]+$ ]] ||
  usage "KNOWN_HOSTS must name the pod's own known_hosts file, an absolute path without spaces"
IP=$1; PORT=$2; KEY=$3; HB=$4; STALE_MIN=$5; DEADLINE=$6
[[ "$HB" =~ ^/[A-Za-z0-9_./-]+\.json$ ]] || usage "HEARTBEAT must be a plain absolute pod path"
[[ "$STALE_MIN" =~ ^[0-9]+$ ]] || usage "STALE_MIN must be whole minutes"
END=$(python3 -c 'import datetime, sys
d = datetime.datetime.fromisoformat(sys.argv[1])
if d.tzinfo is None:
    sys.exit(1)
print(int(d.timestamp()))' "$DEADLINE" 2>/dev/null) || usage "DEADLINE_UTC must be an ISO time with its timezone"
QD=${HB%/*}
POLL=${POLL:-120}; SSH_FAILS=${SSH_FAILS:-3}; DRY_RUN=${DRY_RUN:-0}; GRACE_MIN=${GRACE_MIN:-15}
SSH_TIMEOUT=${SSH_TIMEOUT:-60}; STOP_TIMEOUT=${STOP_TIMEOUT:-120}; STOP_TRIES=${STOP_TRIES:-3}
STOP_GRACE=${STOP_GRACE:-300}; END_POLL=${END_POLL:-30}
SSH=("${SSH_CMD:-ssh}" -i "$KEY" -p "$PORT" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new
     -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o BatchMode=yes "root@$IP")
READ="printf 'now=%s\n' \"\$(date +%s)\"; if [ -f $HB ]; then printf 'beat=%s\n' \"\$(stat -c %Y $HB)\"; \
else echo beat=none; fi; if [ -f $QD/started.json ]; then printf 'started=%s\n' \"\$(stat -c %Y $QD/started.json)\"; \
else echo started=none; fi; if [ -f $QD/outcome.json ]; then \
printf 'outmtime=%s\n' \"\$(stat -c %Y $QD/outcome.json)\"; \
printf 'outcome=%s\n' \"\$(tr -d '\n' < $QD/outcome.json)\"; \
else echo outmtime=none; echo outcome=none; fi; \
if pgrep -f \"V2GPU_WATCHDO[G]\" >/dev/null; then echo wd=yes; else echo wd=no; fi"
PROBE="echo ALIVE"
# shellcheck disable=SC2016  # $RUNPOD_POD_ID expands on the pod, after /etc/rp_environment
STOP='source /etc/rp_environment; runpodctl stop pod "$RUNPOD_POD_ID"'

if [ "$DRY_RUN" = 1 ]; then
  echo "+ ${SSH[*]} $READ"
  echo "+ ${SSH[*]} $PROBE"
  echo "+ ${SSH[*]} $STOP"
  echo "WATCH_DRY_RUN"
  exit 0
fi

with_timeout() {                 # with_timeout SECONDS CMD...: CMD, killed after SECONDS (macOS has no timeout(1))
  # Poll instead of a background guard. The guard subshell raced its own pkill/kill and left an
  # orphaned `sleep 5` in most quick calls (a live test-user process on a pod's step 5); now no process exists but CMD.
  local t=$1; shift
  "$@" &
  local pid=$! n=0 rc=0
  while kill -0 "$pid" 2>/dev/null; do
    [ "$n" -eq $((t * 10)) ] && kill -TERM "$pid" 2>/dev/null
    [ "$n" -eq $(((t + 5) * 10)) ] && kill -KILL "$pid" 2>/dev/null
    sleep 0.1; n=$((n + 1))
  done
  wait "$pid" || rc=$?
  return "$rc"
}

alive() { with_timeout "$SSH_TIMEOUT" "${SSH[@]}" "$PROBE" >/dev/null 2>&1; }

gone() {                         # SSH refused or closed SSH_FAILS probes in a row
  local n=0
  while [ "$n" -lt "$SSH_FAILS" ]; do
    if alive; then return 1; fi
    n=$((n + 1))
    if [ "$n" -lt "$SSH_FAILS" ]; then sleep "$END_POLL"; fi
  done
  return 0
}

stop_pod() {                     # the stale or deadline stop: STOP_TRIES tries while SSH answers
  local i=1
  while [ "$i" -le "$STOP_TRIES" ]; do
    if with_timeout "$STOP_TIMEOUT" "${SSH[@]}" "$STOP"; then
      echo "STOPPED_BY_WATCH $IP:$PORT: $1"
      echo "CHECK_EXITED_VIA_MCP: confirm the pod is EXITED with MCP get-pod"
      exit 5
    fi
    if gone; then
      echo "SSH_CLOSED $IP:$PORT after stop try $i: $1"
      echo "CHECK_EXITED_VIA_MCP: confirm the pod is EXITED with MCP get-pod, else pod-action stop"
      exit 5
    fi
    sleep $((END_POLL * i)); i=$((i + 1))
  done
  echo "STOP_VIA_MCP $IP:$PORT: $1, and $STOP_TRIES SSH stops failed: stop the pod with MCP (pod-action stop)," \
       "confirm EXITED"
  exit 3
}

end_watch() {                    # QUEUE_END: wait until the pod is gone
  echo "QUEUE_END $IP:$PORT: $1"
  case "$1" in
    *'"self_stop": false'*)
      echo "CHECK: no self_stop_cmd in this queue: copy its results, then stop the pod (MCP pod-action stop) and" \
           "confirm EXITED"
      exit 0 ;;
  esac
  local t0; t0=$(date +%s)
  while :; do                    # SSH is probed at least once, however short STOP_GRACE is
    if gone; then
      echo "SSH_CLOSED $IP:$PORT: the queue's self-stop took"
      echo "CHECK_EXITED_VIA_MCP: confirm the pod is EXITED with MCP get-pod (else pod-action stop)"
      exit 0
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$STOP_GRACE" ]; then break; fi
    sleep "$END_POLL"
  done
  echo "SELF_STOP_LATE $IP:$PORT: SSH still answers ${STOP_GRACE} s after the queue ended: stopping it over SSH"
  local i=1
  while [ "$i" -le "$STOP_TRIES" ]; do
    echo "stop try $i"
    with_timeout "$STOP_TIMEOUT" "${SSH[@]}" "$STOP" || true
    sleep $((END_POLL * i))
    if gone; then
      echo "SSH_CLOSED $IP:$PORT after stop try $i"
      echo "CHECK_EXITED_VIA_MCP: confirm the pod is EXITED with MCP get-pod (else pod-action stop)"
      exit 0
    fi
    i=$((i + 1))
  done
  echo "ALARM STOP_VIA_MCP $IP:$PORT: the queue ended, its self-stop and $STOP_TRIES SSH stops failed and the pod" \
       "still answers: stop it with MCP (pod-action stop), confirm EXITED"
  exit 3
}

ERR=$(mktemp); trap 'rm -f "$ERR"' EXIT
FIRST=""; MISS=0; REPORTED=0; WD_ALARM=0; HOSTKEY=0; EARLIER=0
while :; do
  if ! STATE=$(with_timeout "$SSH_TIMEOUT" "${SSH[@]}" "$READ" 2>"$ERR"); then
    if [ "$HOSTKEY" = 0 ] && grep -q "HOST IDENTIFICATION HAS CHANGED" "$ERR"; then
      echo "HOST_KEY_CHANGED $IP:$PORT: the pod's host key is not the one in $KNOWN_HOSTS (a container restart makes" \
           "new keys and loses the watchdog and the queue): read the pod with MCP get-pod; inside its cap, run the" \
           "fuse block of step 4 (it deletes that file and re-arms the watchdog), then restart this watcher"
      HOSTKEY=1
    fi
    MISS=$((MISS + 1))
    if [ "$MISS" -ge "$SSH_FAILS" ]; then
      SUFFIX=""
      if [ "$HOSTKEY" = 1 ]; then SUFFIX=" (a restart: see HOST_KEY_CHANGED above)"; fi   # order is unambiguous
      echo "STOP_VIA_MCP $IP:$PORT: SSH failed $MISS times in a row: the pod is stopped or unreachable; read it with" \
           "MCP get-pod and stop it with pod-action stop unless it is EXITED$SUFFIX"
      exit 3
    fi
    sleep "$POLL"; continue
  fi
  MISS=0; NOW=""; BEAT=none; STARTED=none; OUTM=none; OUTCOME=none; WD=unknown
  while IFS='=' read -r k v; do
    case "$k" in
      now) NOW=$v ;; beat) BEAT=$v ;; started) STARTED=$v ;; outmtime) OUTM=$v ;; outcome) OUTCOME=$v ;;
      wd) WD=$v ;;
    esac
  done <<< "$STATE"
  [[ "$NOW" =~ ^[0-9]+$ ]] || NOW=$(( $(date +%s) ))
  [ -n "$FIRST" ] || FIRST=$NOW                                   # the watch's start, on the pod's clock
  [[ "$STARTED" =~ ^[0-9]+$ ]] || STARTED=none
  [[ "$OUTM" =~ ^[0-9]+$ ]] || OUTM=none
  if [ "$BEAT" != none ] && ! [[ "$BEAT" =~ ^[0-9]+$ ]]; then
    echo "HEARTBEAT_UNREADABLE $IP:$PORT: beat '$BEAT': counted as no current heartbeat"
    BEAT=none
  fi
  if [ "$WD" = no ] && [ "$WD_ALARM" = 0 ]; then
    echo "WATCHDOG_MISSING $IP:$PORT: the on-pod watchdog is not running (a restart?): re-arm it with the time left," \
         "or stop the pod if it is past its cap"
    WD_ALARM=1
  elif [ "$WD" = yes ]; then
    WD_ALARM=0
  fi
  ENDED=0                          # the run that wrote started.json (and the heartbeat) has written its outcome
  if [ "$OUTM" != none ] && { [ "$STARTED" = none ] || [ "$OUTM" -ge "$STARTED" ]; } &&
     { [ "$BEAT" = none ] || [ "$OUTM" -ge "$BEAT" ]; }; then
    ENDED=1
  fi
  if [ "$ENDED" = 1 ] && [ "$OUTM" -lt "$FIRST" ] && [[ "$OUTCOME" != *REFUSED_AT_START* ]]; then   # ended before this watch
    if [ "$EARLIER" = 0 ]; then
      echo "QUEUE_ALREADY_ENDED $IP:$PORT: queue ended before first read → stop the pod via MCP now (read" \
           "outcome.json, copy its results, then pod-action stop); no stop is armed here (a watcher restarted after" \
           "the queue's end must not stop the pod mid-copy or during a fuse test), so this watch instead waits for a" \
           "new run of the queue until $DEADLINE, then it does exit 3"
      EARLIER=1
    fi
    if [ "$(date +%s)" -ge "$END" ]; then
      echo "QUEUE_ALREADY_ENDED $IP:$PORT: deadline $DEADLINE passed and no new run started: this watch stops" \
           "nothing; read the pod with MCP get-pod and stop it (pod-action stop) once its results are home"
      exit 3
    fi
    sleep "$POLL"; continue
  fi
  if [ "$(date +%s)" -ge "$END" ]; then
    stop_pod "deadline $DEADLINE passed"
  fi
  if [ "$ENDED" = 1 ] && [ "$OUTM" -ge "$FIRST" ]; then          # an older one is not this run's
    case "$OUTCOME" in
      *REFUSED_AT_START*)
        if [ "$REPORTED" = 0 ]; then echo "QUEUE_REFUSED $IP:$PORT: $OUTCOME"; REPORTED=1; fi ;;
      *)
        end_watch "$OUTCOME" ;;
    esac
  fi
  CURRENT=0                        # a heartbeat written since this watch began (proof the queue is alive NOW)?
  LEFTOVER=0     # a heartbeat older than this watch's first read is a possibly-dead run's
                 # leftover (e.g. a container killed along with its queue, no outcome.json ever written): a restarted
                 # watcher must not judge it by the tight STALE_MIN clock at once, only by the grace from FIRST. A
                 # heartbeat that appears (or is renewed) during this watch's life is >= FIRST and follows the old rule.
  if [ "$BEAT" != none ] && [ "$ENDED" = 0 ]; then
    if [ "$BEAT" -lt "$FIRST" ]; then
      LEFTOVER=1
    elif [ "$STARTED" != none ]; then
      [ "$BEAT" -ge "$STARTED" ] && CURRENT=1
    else
      CURRENT=1
    fi
  fi
  if [ "$CURRENT" = 1 ]; then
    AGE=$((NOW - BEAT)); LIMIT=$((STALE_MIN * 60)); WHAT="heartbeat stale"
  elif [ "$LEFTOVER" = 1 ]; then
    AGE=$((NOW - FIRST)); LIMIT=$((STALE_MIN > GRACE_MIN ? STALE_MIN * 60 : GRACE_MIN * 60))
    WHAT="heartbeat from before this watch began (a dead run's leftover), none current since"
  else                             # grace from the queue start; from the watch's start if that run already ended
    BASE=$FIRST
    if [ "$STARTED" != none ] && { [ "$ENDED" = 0 ] || [ "$STARTED" -gt "$FIRST" ]; }; then BASE=$STARTED; fi
    AGE=$((NOW - BASE)); LIMIT=$((STALE_MIN > GRACE_MIN ? STALE_MIN * 60 : GRACE_MIN * 60))
    WHAT="no heartbeat of the current queue"
  fi
  if [ "$AGE" -ge "$LIMIT" ]; then
    stop_pod "$WHAT: ${AGE} s >= $((LIMIT / 60)) min"
  fi
  sleep "$POLL"
done
