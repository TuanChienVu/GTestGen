#!/usr/bin/env bash
# Ship a QUEUE's inputs to its pod, from the laptop:  bash ops/ship.sh QUEUE.json IP PORT KEY   (DRY_RUN=1 only prints)
# Every target of the QUEUE's "ship" map (genqueue: prompt files, sums, META, excluded, pools, leak-audit receipt,
# adapters, the originals of cross-pod rechecks) is copied file by file (a directory source file by file under its
# target), QUEUE.json itself goes to <pod root>/<its name> (the pod root is the parent of queue_dir), and then the pod
# checks every copied file against the sha256 computed here (sha256sum -c of SHIP_SHA256SUMS.<queue name>). A
# misplaced or torn file stops here, not as REFUSED_PREFLIGHT on the pod. Prints SHIP_OK.
# KNOWN_HOSTS (required) names the pod's own known_hosts file:
# every ssh and scp runs with UserKnownHostsFile=$KNOWN_HOSTS, StrictHostKeyChecking=accept-new, a connect timeout
# and keepalives.
set -euo pipefail
usage() {
  echo "usage: KNOWN_HOSTS=<the pod's known_hosts file> bash ops/ship.sh QUEUE.json IP PORT KEY ($1)" >&2
  exit 2
}
[ $# -eq 4 ] || usage "four arguments"
KNOWN_HOSTS=${KNOWN_HOSTS:-}
[[ "$KNOWN_HOSTS" =~ ^/[^[:space:]]+$ ]] ||
  usage "KNOWN_HOSTS must name the pod's own known_hosts file, an absolute path without spaces"
Q=$1; IP=$2; PORT=$3; KEY=$4
DRY_RUN=${DRY_RUN:-0}
run() { echo "+ $*"; if [ "$DRY_RUN" != 1 ]; then "$@"; fi; }
ALIVE=(-o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
SSH=(ssh -i "$KEY" -p "$PORT" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new "${ALIVE[@]}"
     "root@$IP")
SCP=(scp -q -i "$KEY" -P "$PORT" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new
     "${ALIVE[@]}")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
ROOT=$(python3 - "$Q" "$TMP" <<'PY'
import hashlib, json, sys
from pathlib import Path
queue, tmp = Path(sys.argv[1]), Path(sys.argv[2])
q = json.loads(queue.read_text())
root = q["queue_dir"].rsplit("/", 1)[0]
files = []
for section, items in sorted(q["ship"].items()):
    for target, source in sorted(items.items()):
        src = Path(source)
        if src.is_dir():
            files += [(p, f"{target}/{p.relative_to(src).as_posix()}") for p in sorted(src.rglob("*")) if p.is_file()]
        elif src.is_file():
            files.append((src, target))
        else:
            sys.exit(f"ship {section}: the source {source} of {target} is missing")
files.append((queue, f"{root}/{queue.name}"))
bad = [t for _, t in files if not t.startswith(root + "/") or any(c.isspace() for c in t) or "/../" in t]
if bad:
    sys.exit(f"ship targets outside {root} or with spaces: {bad[:5]}")
(tmp / "copy.tsv").write_text("".join(f"{s}\t{t}\n" for s, t in files))
(tmp / "dirs.txt").write_text("".join(sorted({t.rsplit("/", 1)[0] + "\n" for _, t in files})))
(tmp / "sums").write_text("".join(f"{hashlib.sha256(s.read_bytes()).hexdigest()}  {t}\n" for s, t in files))
print(root)
PY
)
SUMS="$ROOT/SHIP_SHA256SUMS.$(basename "$Q")"
DIRS=()                                   # no mapfile: the laptop's /bin/bash may be 3.2
while IFS= read -r d; do DIRS+=("$d"); done < "$TMP/dirs.txt"
run "${SSH[@]}" mkdir -p "${DIRS[@]}"
while IFS=$'\t' read -r src target; do
  run "${SCP[@]}" "$src" "root@$IP:$target"
done < "$TMP/copy.tsv"
run "${SCP[@]}" "$TMP/sums" "root@$IP:$SUMS"
if [ "$DRY_RUN" = 1 ]; then sed 's/^/  expect /' "$TMP/sums"; fi
run "${SSH[@]}" sha256sum -c --quiet "$SUMS"
echo "SHIP_OK"
