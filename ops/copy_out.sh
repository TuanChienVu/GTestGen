#!/usr/bin/env bash
# Copy one finished run home and verify it, BEFORE anything on the pod is stopped or deleted:
#   bash ops/copy_out.sh IP PORT KEY REMOTE_RUNS_DIR JOB LOCAL_DIR [REMOTE_QUEUE_DIR]
# Uses the pod's direct SSH port (the platform's SSH proxy refuses non-PTY commands). No state/, state.tmp/ or state.old/
# at any depth is copied (after a job only the epoch adapters, receipts, metrics and logs matter; the canary alone holds
# about 8-9 GB of optimizer state). With REMOTE_QUEUE_DIR, <job>.log, <job>.status, outcome.json, self_stop.log and
# heartbeat.json also land in LOCAL_DIR/<job>.queue/ (outside the manifest); none of them existing is fine (-T /dev/null
# keeps GNU tar from refusing an empty archive). Exit 0 only if MANIFEST.sha256 verifies.
# COPY_OUT_LOCAL=1 reads the remote paths on this machine instead of over SSH (used by the tests).
# KNOWN_HOSTS (required, also with COPY_OUT_LOCAL) names the pod's own known_hosts
# file: ssh runs with UserKnownHostsFile=$KNOWN_HOSTS, StrictHostKeyChecking=accept-new, a connect
# timeout and keepalives.
set -euo pipefail
usage() {
  echo "usage: KNOWN_HOSTS=<the pod's known_hosts file> bash ops/copy_out.sh IP PORT KEY REMOTE_RUNS_DIR JOB" \
       "LOCAL_DIR [REMOTE_QUEUE_DIR] ($1)" >&2
  exit 2
}
[ $# -eq 6 ] || [ $# -eq 7 ] || usage "six or seven arguments"
KNOWN_HOSTS=${KNOWN_HOSTS:-}
[[ "$KNOWN_HOSTS" =~ ^/[^[:space:]]+$ ]] ||
  usage "KNOWN_HOSTS must name the pod's own known_hosts file, an absolute path without spaces"
IP=$1 PORT=$2 KEY=$3 REMOTE=$4 JOB=$5 DEST=$6 QUEUE=${7:-}
mkdir -p "$DEST"
if [ -e "$DEST/$JOB" ]; then echo "REFUSED: $DEST/$JOB already exists" >&2; exit 2; fi
remote() {  # run a shell snippet where the files are
  if [ "${COPY_OUT_LOCAL:-0}" = 1 ]; then bash -c "$1"; else
    ssh -i "$KEY" -p "$PORT" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "root@$IP" "$1"; fi
}
EXCL=""
for d in state state.tmp state.old; do EXCL="$EXCL --exclude $(printf %q "$JOB/$d") --exclude $(printf %q "*/$d")"; done
remote "tar -C $(printf %q "$REMOTE") -cf - $EXCL $(printf %q "$JOB")" | tar -C "$DEST" -xf -
[ -f "$DEST/$JOB/MANIFEST.sha256" ] || { echo "FAILED: no MANIFEST.sha256 in $JOB" >&2; exit 1; }
(cd "$DEST/$JOB" && shasum -a 256 -c --quiet MANIFEST.sha256)
if [ -n "$QUEUE" ]; then
  mkdir -p "$DEST/$JOB.queue"
  remote "cd $(printf %q "$QUEUE") && set -- && for f in $(printf %q "$JOB.log") $(printf %q "$JOB.status") outcome.json self_stop.log heartbeat.json; do [ -e \"\$f\" ] && set -- \"\$@\" \"\$f\"; done; tar -cf - -T /dev/null \"\$@\"" \
    | tar -C "$DEST/$JOB.queue" -xf -
fi
echo "COPY_OK $JOB $(shasum -a 256 "$DEST/$JOB/MANIFEST.sha256" | cut -d' ' -f1)"
