#!/usr/bin/env bash
# Set up a rented pod from the repo checkout:  bash ops/pod_bootstrap.sh [MODEL ...]   (DRY_RUN=1 only prints)
# KIND=gpu (default): the training venv $V2GPU_LOCAL/venv from requirements-gpu.{lock,in}, tests_gpu and the nine tests/
# modules the training pods ran (see docs/KNOWN_ISSUES.md).
# KIND=gen: the generation venv $V2GPU_LOCAL/venv-gen from requirements-gen.{lock,in} (vLLM 0.27.1, torch 2.13.0), the
# generation test files only (training code is never run in it), then an import check of vLLM and bitsandbytes.
# GEN_SET (KIND=gen only; default main): a pre-declared pin set of v2gpu.engines.PIN_SETS, chosen ONLY at the canary
# (decision 24/09): requirements-gen-<set>.{lock,in} instead of requirements-gen.{lock,in}.
# A MooseFS /workspace imported torch in 58.9 s (smoke pod 28/09, EU-RO-1), so the repo and the
# venvs live on the container's own disk, V2GPU_LOCAL (default /opt/v2); $WS (default /workspace) keeps only what must
# survive a stop or be copied home -- the bundle and inputs (in/), the HF cache, the uv cache, runs, queues, logs and
# receipts. A stop/start or container restart wipes the container disk: the operator re-runs the clone and this
# bootstrap before any queue (the venv is rebuilt from the committed lock and the uv cache on $WS).
# Without the committed requirements-<kind>[-<set>].lock the bootstrap refuses, unless FIRST_RUN=1 says
# this is the first pod of that KIND (it freezes the lock, writes $VENV/FIRST_RUN_LOCK.json, and the lock is committed
# before any other pod); FIRST_RUN=1 with a committed lock is refused too.
# The test step runs without KIND and GEN_SET (env -i): the suite on the pod sees the laptop's
# environment, not this script's selectors. KIND=gpu leaves out the nested gen-list simulation
# (--deselect, in the count and in the run; the gate is told exactly one test is deselected): the gen
# bootstrap runs that list for real, in the real generation venv.
# The venv uses the image's own /usr/bin/python3 (an earlier lesson: a uv-managed Python on the container disk
# vanished after a stop and broke the venv). uv is installed if the image lacks it. An HF_TOKEN, when needed, comes
# from a RunPod secret in the pod environment and is never echoed.
set -euo pipefail
WS=${WS:-/workspace}
DRY_RUN=${DRY_RUN:-0}
KIND=${KIND:-gpu}
# The pod's env file and the test user's HOME come from two variables that default to the real
# paths. Only the bootstrap's own tests move them (into their tmp_path: a real, not dry, run of this script by a test
# never chmods the machine's env file nor deletes its /home/v2test), and only with V2GPU_BOOTSTRAP_TEST=1: without it a
# non-default value is refused before anything runs, so nothing can point step 5's chmod 600 at another file.
# Nothing in ops/ (scripts and RUNSHEET files) sets any of the four (tests_gpu/test_pod_safety.py scans for it). The same rule
# holds for V2GPU_LOCAL, the container-disk root of the repo and the venvs (default /opt/v2).
RP_ENV_FILE=${RP_ENV_FILE:-/etc/rp_environment}
V2TEST_HOME=${V2TEST_HOME:-/home/v2test}
V2GPU_LOCAL=${V2GPU_LOCAL:-/opt/v2}
if [ "$RP_ENV_FILE" != /etc/rp_environment ] || [ "$V2TEST_HOME" != /home/v2test ] || [ "$V2GPU_LOCAL" != /opt/v2 ]; then
  [ "${V2GPU_BOOTSTRAP_TEST:-0}" = 1 ] || { echo "RP_ENV_FILE=$RP_ENV_FILE V2TEST_HOME=$V2TEST_HOME V2GPU_LOCAL=$V2GPU_LOCAL:" \
    "moving the pod's env file, the v2test HOME or the local venv root is for the bootstrap's own tests only" \
    "(V2GPU_BOOTSTRAP_TEST=1): refusing" >&2; exit 2; }
fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
# Since 05/10 the HF cache and uv's cache live on the container disk with the venvs: on 04/10, uv extracting
# wheels into $WS/uv-cache on EUR-IS-1's MooseFS took 2.4 h of a 2.7 h bootstrap (1.1 MB/s; the pod's own downloads
# ran at 40-45 MB/s). A restart wipes them with the venvs: the re-bootstrap downloads again (the VENV_GONE step).
HF_HOME="$V2GPU_LOCAL/hf"; UV_CACHE_DIR="$V2GPU_LOCAL/uv-cache"   # reads the root only (test_pod_safety's knob scan)
export HF_HOME UV_CACHE_DIR
# On 04/10 uv's default 30 s network timeout failed a FIRST_RUN gen bootstrap on EUR-IS-1 (z3-solver
# "network timeout" while extracting); a longer wait only, the resolution and the lock are unchanged
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
# The laptop gate: ops/pod_parity_gate.sh runs this script in a container with no GPU and no
# network, to prove step 5 (and the 1.5 fail-fast before it) never touches a real system path. GATE_ONLY=1 skips
# steps 1 (nvidia-smi), 2-3 (uv/venv/packages), 4 (model fetch) and 5b (vLLM/bitsandbytes import), none possible
# offline, and step 6 (the receipt, not needed to prove what the gate checks); $VENV/bin/python must already exist when
# this is set. Never set on a pod.
# Until 27/09/2026 GATE_ONLY had no guard at all, so a pasted or edited launch
# line could reach a real pod's bootstrap with GATE_ONLY=1 and skip the lock sync, the model check, the vLLM import
# and env_receipt while still printing BOOTSTRAP_OK (cheap on a fresh pod with no venv, silent on a re-bootstrapped
# one). It may run only inside the pod-parity gate's own container: nvidia-smi absent (a GPU pod always has it, the
# gate image never does) AND the gate's own marker present (GATE_MARKER, created only by pod_parity_gate.sh's inner
# script, never on a real pod). GATE_MARKER is overridable for the bootstrap's own tests (it names no real system
# path and is never chmod'd, chowned or deleted, unlike RP_ENV_FILE/V2TEST_HOME above, so moving it needs no
# V2GPU_BOOTSTRAP_TEST gate); tests_gpu/test_pod_safety.py's gate_problems() scans ops/ so nothing but this script
# and pod_parity_gate.sh may name GATE_ONLY, and nothing but this script may name GATE_MARKER.
GATE_ONLY=${GATE_ONLY:-0}
GATE_MARKER=${GATE_MARKER:-/etc/v2gpu-parity-gate}
if [ "$GATE_ONLY" = 1 ]; then
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GATE_ONLY=1 but nvidia-smi is on PATH: this looks like a real pod, never the pod-parity gate's own" \
         "container: refusing" >&2
    exit 2
  fi
  if [ ! -e "$GATE_MARKER" ]; then
    echo "GATE_ONLY=1 needs $GATE_MARKER (created only by ops/pod_parity_gate.sh's inner container script):" \
         "refusing" >&2
    exit 2
  fi
fi
SIM=tests_gpu/test_genqueue.py::test_the_generation_test_list_passes_in_a_simulated_generation_venv
case "$KIND" in
  gpu) VENV="$V2GPU_LOCAL/venv"; LOG="$WS/pytest.log"; RECEIPT="$WS/env_receipt.json"; DESELECT="--deselect $SIM"; NDESELECT=1
       TESTS="$(printf 'tests/test_%s.py ' bind gherkin_steps history patterns scopes split stepdefs wrong_suite_control yield_report)tests_gpu" ;;
  gen) VENV="$V2GPU_LOCAL/venv-gen"; LOG="$WS/pytest.gen.log"; RECEIPT="$WS/env_receipt.gen.json"; DESELECT=""
       NDESELECT=0
       TESTS=$(printf 'tests_gpu/test_%s.py ' runs engines outlog genprep generate gencoverage gencanary genqueue \
               serialize examples audit special_tokens); TESTS=${TESTS% } ;;
  *) echo "KIND must be gpu or gen, not $KIND" >&2; exit 2 ;;
esac
GEN_SET=${GEN_SET:-main}
REQS="$REPO/requirements-$KIND"
if [ "$GEN_SET" != main ]; then
  [ "$KIND" = gen ] || { echo "GEN_SET names a generation pin set: KIND=gen only" >&2; exit 2; }
  REQS="$REPO/requirements-gen-$GEN_SET"
fi
[ -f "$REQS.in" ] || { echo "no $REQS.in: GEN_SET must be a pin set of v2gpu.engines.PIN_SETS" >&2; exit 2; }
LOCK=${REQS##*/}.lock                      # no basename: the bootstrap needs only bash, dirname and pwd before uv
if [ -f "$REQS.lock" ] && [ "${FIRST_RUN:-0}" = 1 ]; then
  echo "FIRST_RUN=1 but $LOCK is committed: every pod after the first installs the lock; unset FIRST_RUN" >&2; exit 2
fi
if [ ! -f "$REQS.lock" ] && [ "${FIRST_RUN:-0}" != 1 ]; then
  echo "no $LOCK in this checkout: only the FIRST pod of KIND=$KIND resolves packages, with FIRST_RUN=1 said" \
       "explicitly; commit the lock it froze and rebuild the bundle before any other pod" >&2; exit 2
fi
PY="$VENV/bin/python"
run() { echo "+ $*"; if [ "$DRY_RUN" != 1 ]; then "$@"; fi; }
TEST_USER=v2test; TEST_HOME=$V2TEST_HOME
# decision 28/09: how long each step takes goes into $WS/bootstrap_timings.txt ("<step> <epoch s>" per line, then
# "end"), so the watchdog cap's bootstrap allowance (TRAIN_RUNSHEET.md §2, BOOT_S) comes from measurements.
BOOT_T0=$(date +%s 2> /dev/null || echo 0)       # a dry run on a PATH without date still prints every step
TIMINGS="$WS/bootstrap_timings.txt"
mark() { if [ "$DRY_RUN" != 1 ]; then echo "$1 $(date +%s)" >> "$TIMINGS"; fi; }
# A bootstrap run again on the same pod volume (the resume rule) keeps the
# earlier run's evidence: moved, never deleted, into $WS/bootstrap-prev/<start epoch>-<pid>/ before this run writes its
# own. The frozen <lock>.new stays in place: a first run started again installs it (step 3).
EVIDENCE="bootstrap_timings.txt volume_receipt.txt pytest.log pytest.gen.log env_receipt.json env_receipt.gen.json fetch_report.json"
keep_previous_evidence() {
  local prev="$WS/bootstrap-prev/$BOOT_T0-$$" moved="" f
  for f in $EVIDENCE; do
    if [ -e "$WS/$f" ]; then mkdir -p "$prev" && mv "$WS/$f" "$prev/" && moved="$moved $f"; fi
  done
  if [ -n "$moved" ]; then echo "   the earlier bootstrap's evidence moved to $prev:$moved"; fi
}
if [ "$DRY_RUN" != 1 ]; then mkdir -p "$WS"; keep_previous_evidence; echo "start $BOOT_T0" > "$TIMINGS"; fi
# A positive layout check. The repo and the venv root must not be under $WS, and when $WS is a mount of its own
# (a pod's /workspace) the venv root must sit on a different filesystem: otherwise refuse, before step 1.
layout_check() {
  mkdir -p "$V2GPU_LOCAL"
  ws_real=$(cd "$WS" && pwd -P); local_real=$(cd "$V2GPU_LOCAL" && pwd -P); repo_real=$(cd "$REPO" && pwd -P)
  for p in "$local_real" "$repo_real"; do
    case "$p/" in "$ws_real"/*)
      echo "LAYOUT_ON_WORKSPACE: $p is under $WS; the repo and the venvs belong on the container's own disk" \
           "(V2GPU_LOCAL, default /opt/v2): refusing" >&2
      exit 2 ;;
    esac
  done
  if command -v mountpoint > /dev/null && mountpoint -q "$ws_real" && \
     [ "$(stat -c %d "$local_real")" = "$(stat -c %d "$ws_real")" ]; then
    echo "LAYOUT_ON_WORKSPACE: $local_real is on the same filesystem as the pod volume $WS: refusing" >&2
    exit 2
  fi
  echo "   layout: repo $repo_real, venvs under $local_real, pod volume $ws_real"
}
# And no process of the test user may be alive from here to step 5's relock: one left by
# an earlier bootstrap in this container (a re-run of §5, GEN's fallback pin set) stops the bootstrap first. A missing
# pgrep is a refusal too (the check would be blind). The bootstrap's OWN
# ancestors and descendants are not such a process -- on a pod, step 5 runs this repo's suite as the test user, and a
# fixture's real bootstrap there has pytest as an ancestor and its own subshells as children, all of that user; on a
# real pod's bootstrap (root, started over ssh with setsid) neither can be the test user. A process that is gone by the
# time it is looked at is not alive either. The same check runs again when step 5's test run
# ends, before no_skips.sh: a detached process the test user started there must not outlive the bootstrap.
parent_of() { ps -o ppid= -p "$1" 2> /dev/null | tr -d ' ' || true; }   # empty once the process is gone
test_user_idle() {       # test_user_idle WHEN
  command -v pgrep > /dev/null || { echo "pgrep not found: cannot check that $TEST_USER runs nothing: refusing" >&2; exit 2; }
  local mine=" " p q alive=""
  p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do mine="$mine$p "; p=$(parent_of "$p"); done
  for p in $(pgrep -u "$TEST_USER" 2> /dev/null); do
    case "$mine" in *" $p "*) continue ;; esac         # this bootstrap, or one of its ancestors
    q=$(parent_of "$p")
    [ -n "$q" ] || continue                            # gone already
    while [ -n "$q" ] && [ "$q" -gt 1 ] && [ "$q" != $$ ]; do q=$(parent_of "$q"); done
    [ "$q" = $$ ] && continue                          # one of this bootstrap's own children
    alive="$alive $p"
  done
  if [ -n "$alive" ]; then
    echo "V2TEST_PROCESS_ALIVE ($1): $TEST_USER still runs process(es)$alive (left by an earlier bootstrap in this" \
         "container, or by the test run); stop them (pkill -u $TEST_USER) and run the bootstrap again: refusing" >&2
    exit 2
  fi
}
# layout_check saw V2GPU_LOCAL and the repo; the venv itself could still be a symlink onto
# the pod volume (/opt/v2/venv -> /workspace/venv). Once it exists, its own real path must be off $WS too.
venv_check() {
  ws_real=$(cd "$WS" && pwd -P); venv_real=$(cd "$VENV" && pwd -P)
  case "$venv_real/" in "$ws_real"/*)
    echo "LAYOUT_ON_WORKSPACE: the venv $VENV is $venv_real, under $WS; the venvs belong on the container's own disk" \
         "(V2GPU_LOCAL, default /opt/v2): refusing" >&2
    exit 2 ;;
  esac
  if command -v mountpoint > /dev/null && mountpoint -q "$ws_real" && \
     [ "$(stat -c %d "$venv_real")" = "$(stat -c %d "$ws_real")" ]; then
    echo "LAYOUT_ON_WORKSPACE: the venv $venv_real is on the same filesystem as the pod volume $WS: refusing" >&2
    exit 2
  fi
}
# decision 28/09: venv + venv-gen are about 16 GB; with the repo and a margin, 20 GB free on the container disk
# before a venv is built, or stop with a clear message. Since 05/10 the container disk also holds uv's cache
# (about 15 GB for the generation set, measured on the first generation pod) and the HF weights (about 14 GB per model): 30 GB plus 15 GB
# per model of the pod (deepseek alone 45 GB, all three 75 GB; the packs create pods with "disk": 100).
local_disk_check() {
  local models=${1:-1}
  if [ "$models" -lt 1 ]; then models=1; fi   # no MODEL argument: fetch_models still fetches one model's weights
  local need_gb=$(( 30 + 15 * models ))
  mkdir -p "$V2GPU_LOCAL"
  avail_kb=$(df -Pk "$V2GPU_LOCAL" | awk 'NR == 2 {print $4}')
  if [ "${avail_kb:-0}" -lt $((need_gb * 1024 * 1024)) ]; then
    echo "LOCAL_DISK_LOW: $V2GPU_LOCAL has $(( ${avail_kb:-0} / 1024 / 1024 )) GB free; this pod needs about" \
         "$need_gb GB (the venvs, uv's cache and the HF weights of $models model(s)): refusing -- create the pod" \
         "with a larger container disk (\"disk\" of create-pod)" >&2
    exit 2
  fi
  echo "   container disk: $(( avail_kb / 1024 / 1024 )) GB free under $V2GPU_LOCAL"
}
# What $WS is differs by data center (smoke pod 28/09, EU-RO-1) -- there it was a
# MooseFS FUSE mount (a RunPod network volume) that refuses every ownership change, even for root, where the A6000 pod
# of US-TX-1 had a volume that allowed them. Every bootstrap records it FIRST, before anything that could fail on such
# a volume, in $WS/volume_receipt.txt and in its own log: df -T, the mount line, and two probes on a scratch file
# (an ownership change, and the non-blocking flock v2gpu.generate/outlog take on their workdirs).
volume_receipt() {     # a record only: nothing in it may stop the bootstrap, except $WS itself being unusable
  local p="$WS/.v2gpu-volume-probe"
  mkdir -p "$WS"
  {
    echo "== df -T $WS"; df -T "$WS" 2>&1 || df "$WS" 2>&1 || true
    echo "== mount"; mount | grep -F " $WS " || echo "(no mount of its own: part of its parent's filesystem)"
    mkdir -p "$V2GPU_LOCAL"
    echo "== df -T $V2GPU_LOCAL (the repo and the venvs)"; df -T "$V2GPU_LOCAL" 2>&1 || df "$V2GPU_LOCAL" 2>&1 || true
    rm -rf "$p" "$p.d"
    if : > "$p"; then
      chmod 600 "$p" 2> /dev/null || true      # the EU-RO-1 MooseFS showed every mode as 777
      m=$(stat -c %a "$p" 2> /dev/null || stat -f %Lp "$p" 2> /dev/null || echo "?")
      if [ "$m" = 600 ]; then echo "chmod_probe: 600 kept"; else echo "chmod_probe: asked 600, got $m (not kept)"; fi
      if chown 65534:65534 "$p" 2> /dev/null; then echo "chown_probe: ok"; else echo "chown_probe: refused"; fi
      if python3 -c 'import fcntl, sys; fcntl.flock(open(sys.argv[1], "a"), fcntl.LOCK_EX | fcntl.LOCK_NB)' "$p" \
           2> /dev/null; then echo "flock_probe: ok"; else echo "flock_probe: refused"; fi
      # Does a read-only mode stop this (root) user too? Step 5 makes in/, hf/hub and uv-cache a-w; step 2 of a
      # later bootstrap gives the user write bit back first, so nothing here rests on the answer, which is recorded
      chmod 444 "$p" 2> /dev/null || true
      if (echo x >> "$p") 2> /dev/null; then echo "ro_write_probe: wrote a 444 file (mode bits do not stop this user)"
      else echo "ro_write_probe: refused (mode bits stop this user too)"; fi
      rm -f "$p"
    else echo "probe file: $p not writable"; fi
    if mkdir "$p.d"; then          # v2gpu.generate locks its workdir DIRECTORY's own fd, so probe a directory too
      if python3 -c 'import fcntl, os, sys; fcntl.flock(os.open(sys.argv[1], os.O_RDONLY), fcntl.LOCK_EX | fcntl.LOCK_NB)' \
           "$p.d" 2> /dev/null; then echo "flock_dir_probe: ok"; else echo "flock_dir_probe: refused"; fi
      rmdir "$p.d" || true
    else echo "probe directory: $p.d not creatable"; fi
  } > "$WS/volume_receipt.txt" 2>&1
  cat "$WS/volume_receipt.txt"
}
mark 0
echo "== 0. the volume under $WS (df -T, mount, ownership-change and flock probes) -> $WS/volume_receipt.txt"
run volume_receipt
run layout_check

if [ "$GATE_ONLY" != 1 ]; then
mark 1
echo "== 1. card and driver (torch 2.14.0 train, 2.13.0 generation: CUDA 13.0 builds, so minCudaVersion 13.0)"
run nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

mark 1.5
echo "== 1.5. fail-fast: the $TEST_USER user switch, before the long setup of steps 2-4 (about 15-25 min):"
echo "   a switch broken on this image is caught here at once, not after that wait"
run bash -c "id -u $TEST_USER >/dev/null 2>&1 || useradd --no-create-home --home-dir $TEST_HOME --shell /usr/sbin/nologin --user-group $TEST_USER"
run bash -c "runuser -u $TEST_USER -- env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin true || { echo \"the $TEST_USER user switch failed (fail-fast check): fix it before the long setup runs\" >&2; exit 2; }"

if [ "$GATE_ONLY" != 1 ]; then
mark 2
echo "== 2. uv, then the venv on the container disk ($VENV)"
# A bootstrap run again after a (re)start (the resume rule: the container disk was wiped, the pod
# volume was not) finds in/, hf/hub and uv-cache left read-only by the last run's step 5 (a-w). This user's own writes
# in steps 2-4 (uv's cache, the model fetch) must not rest on it overriding mode bits on a FUSE volume (step 0's
# ro_write_probe records whether it does), so the user write bit comes back here; step 5 takes it away again before
# the test user runs anything. Since 05/10 hf/hub and uv-cache sit on the container disk, so after a restart
# they are gone with the venvs and only in/ is found read-only; a re-run of the bootstrap without a restart finds
# all three read-only.
run test_user_idle "before the pod volume is writable again"
run bash -c "for d in '$WS/in' '$HF_HOME/hub' '$UV_CACHE_DIR'; do if [ -e \"\$d\" ]; then chmod -R u+w \"\$d\"; fi; done"
if [ ! -x "$PY" ]; then run local_disk_check "$#"; fi
if command -v uv > /dev/null; then UV=(uv); else
  run /usr/bin/python3 -m pip install --break-system-packages --quiet uv
  UV=(/usr/bin/python3 -m uv)
fi
if [ ! -x "$PY" ]; then run "${UV[@]}" venv "$VENV" --python /usr/bin/python3; fi
run venv_check

mark 3
echo "== 3. packages ($KIND)"
echo "   uv HTTP timeout: $UV_HTTP_TIMEOUT s"
if [ -f "$REQS.lock" ]; then
  run "${UV[@]}" pip sync --python "$PY" "$REQS.lock"
elif [ -f "$WS/$LOCK.new" ]; then
  # FIRST_RUN again on the same pod volume (a restart wiped the container disk): install
  # the lock this pod's first run froze, never resolve the packages again (they could drift between the two runs)
  run "${UV[@]}" pip sync --python "$PY" "$WS/$LOCK.new"
  echo "   FIRST RUN again on this pod volume: installed $WS/$LOCK.new, the lock its first run froze"
else
  run "${UV[@]}" pip install --python "$PY" -r "$REQS.in"
  if [ "$DRY_RUN" != 1 ]; then "${UV[@]}" pip freeze --python "$PY" > "$WS/$LOCK.new"; fi
fi
if [ ! -f "$REQS.lock" ]; then
  # the venv records the lock it froze: every job receipt names it (receipt.environment first_run_lock), and gencoverage
  # accepts it as the lock the later pods install; only a 64-hex sha256 of a sha256sum that exited 0
  # is written, anything else (it fails, prints nothing or a malformed value) stops the bootstrap with exit 2, and a
  # FIRST_RUN_LOCK.json of an earlier run is deleted first
  run bash -c "rm -f '$VENV/FIRST_RUN_LOCK.json'; if s=\$(set -o pipefail; sha256sum '$WS/$LOCK.new' | cut -d' ' -f1) && [[ \"\$s\" =~ ^[0-9a-f]{64}\$ ]]; then printf '{\"%s\": \"%s\"}\n' '$LOCK' \"\$s\" > '$VENV/FIRST_RUN_LOCK.json'; else echo \"no sha256 of $WS/$LOCK.new ('\$s'): FIRST_RUN_LOCK.json not written\" >&2; exit 2; fi"
  echo "   FIRST RUN: copy $WS/$LOCK.new home and commit it as $LOCK"
fi

mark 4
echo "== 4. models at the pinned revisions, every LFS file checked against its sha256"
run bash -c "cd '$REPO' && '$PY' -m v2gpu.fetch_models '$WS/fetch_report.json' $*"
fi

if [ "$GATE_ONLY" = 1 ]; then run venv_check; fi   # step 2, which runs it otherwise, is skipped here

mark 5
echo "== 5. the CPU test suites on the pod's own packages, as the unprivileged user $TEST_USER; anything but 'N passed'"
echo "   (N = collected) fails the bootstrap"
# A pod-side guard that does not depend on the tests. The pod's env file (it holds the pod's
# RUNPOD_API_KEY) becomes root-only first: the watchdog, the queues' self-stop and every read of it over ssh stay root,
# and the operator's fuse block re-applies it after every (re)start (/start.sh rewrites the file). The suite then runs
# as the unprivileged TEST_USER (made if absent: no home skeleton, no login shell), from a fresh empty HOME (no profile
# that could source the file), with a clean environment (env -i: PATH, HOME, LANG, HF_HOME, HF_HUB_OFFLINE,
# V2GPU_REQUIRE_TORCH only; no KIND, GEN_SET, FIRST_RUN or RUNPOD_*). The repo, the venv and the model cache are made
# readable, never writable, to it (chmod a+rX,go-w: a MooseFS /workspace showed everything 777, so without go-w the
# test user could change the repo or the venv root runs later), and no pytest of this step writes
# bytecode (PYTHONDONTWRITEBYTECODE=1); the log is written by root; the gate is the same "N passed" as under root.
# Since the smoke pod of 28/09 (EU-RO-1 MooseFS), nothing under $WS changes ownership any more -- the old
# `install -d -o $TEST_USER ... $REPO/.pytest_cache` was refused there ("Operation not permitted") and stopped the
# bootstrap -- and every pytest of this step runs with -p no:cacheprovider: the suite never uses pytest's cache
# (no --lf/--ff), so no cache directory is written anywhere, by $TEST_USER or by root.
# The v2test gate: find_spec("torch") can lie under a permission gap for
# TEST_USER and used to make tests_gpu/conftest.py silently skip the whole directory while the gate stayed green;
# V2GPU_REQUIRE_TORCH=1 makes conftest fail loudly instead when torch is missing, and the v2test-collected count N is
# also compared against a plain root recollection (belt and braces): any difference refuses the bootstrap.
run bash -c "if [ -e '$RP_ENV_FILE' ]; then chmod 600 '$RP_ENV_FILE' && chown root:root '$RP_ENV_FILE'; fi"
run bash -c "id -u $TEST_USER >/dev/null 2>&1 || useradd --no-create-home --home-dir $TEST_HOME --shell /usr/sbin/nologin --user-group $TEST_USER"
run bash -c "rm -rf '$TEST_HOME' && install -d -m 700 -o $TEST_USER -g $TEST_USER '$TEST_HOME'"
run chmod a+rx "$WS" "$HF_HOME"
run chmod -R a+rX,go-w "$REPO" "$VENV" "$HF_HOME"   # on the container disk, go-w holds (since
                                                    # 05/10 HF_HOME too, now there; hf/hub and uv-cache lose a-w below)
# decision 28/09: everything under $WS the test user never writes loses EVERY write bit (a-w): on the EU-RO-1
# MooseFS the user bits are copied to group and other (go-w is lost), but a 444 file refused v2test's write; root is
# not held back by mode bits, so later root writes (a rerun's fetch, uv) still work.
# Since 05/10 hf/hub and uv-cache are on the container disk, where go-w would hold; they keep a-w all the same
# (the same policy as before, read-only for everyone while the test user runs).
run bash -c "for d in '$WS/in' '$HF_HOME/hub' '$UV_CACHE_DIR'; do if [ -e \"\$d\" ]; then chmod -R a+rX,a-w \"\$d\"; fi; done"
# The 600:root stat re-check runs before the first runuser of this step that runs repo
# code (the v2test count below), no longer after it: no conftest or module-level code runs as $TEST_USER before the file
# is verified. (The step 1.5 probe only runs `true`, before steps 2-4.)
run bash -c "if [ -e '$RP_ENV_FILE' ]; then chmod 600 '$RP_ENV_FILE' && chown root:root '$RP_ENV_FILE'; st=\$(stat -c '%a:%U' '$RP_ENV_FILE'); [ \"\$st\" = 600:root ] || { echo \"$RP_ENV_FILE is \$st right before the $TEST_USER switch, not 600:root: refusing (closes the window where /start.sh had not yet (re)written the file)\" >&2; exit 2; }; fi"
# Measured on the real EU-RO-1 MooseFS (smoke 28/09): a file made 666 stays 666 after chmod 600, chmod 755 gives
# 777 -- mode bits are not kept, so the go-w above cannot stop $TEST_USER there. So the bootstrap detects instead of
# preventing. Before $TEST_USER runs anything, root records a stat manifest (ctime included: a
# non-root user cannot set it back) of the repo, the venv and the HF hub cache, with its own copy of
# ops/tree_manifest.py in a fresh root-only mktemp -d directory; after EACH of $TEST_USER's runs -- before root's own
# recollection runs the repo's code, and before no_skips.sh and step 6 -- it compares: any added, removed or changed
# entry stops the bootstrap with BOOTSTRAP_TAMPERED.
if [ "$DRY_RUN" = 1 ]; then TAMPER_DIR=/tmp/v2gpu-tamper.DRY; else TAMPER_DIR=$(mktemp -d); fi
# decision 28/09: the manifest covers the WHOLE pod volume $WS -- in/ and uv-cache (what a
# re-bootstrap after a restart trusts), runs/<job>/state, the queue files, GEN's inputs -- and leaves out, by exact
# name, only what root itself writes there while the test user runs: this KIND's pytest log, the launch logs, the
# watchdog's log, and $WS's own entry (which their creation touches).
# Since 05/10 HF_HOME and uv's cache are on the container disk, not $WS, so they are roots of their own (when they
# exist; one that appears between two snapshots is a difference too).
tamper_snapshot() {
  local roots=("$REPO" "$VENV") d
  for d in "$HF_HOME" "$UV_CACHE_DIR"; do if [ -e "$d" ]; then roots+=("$d"); fi; done
  python3 "$TAMPER_DIR/tree_manifest.py" --skip "$WS" --skip "$LOG" --skip "$WS/bootstrap.log" \
    --skip "$WS/bootstrap.gen.log" --skip "$WS/watchdog.log" "${roots[@]}" "$WS" > "$TAMPER_DIR/$1"
}
tamper_check() {
  tamper_snapshot after
  if cmp -s "$TAMPER_DIR/before" "$TAMPER_DIR/after"; then echo "   tamper check: unchanged after $1"; return 0; fi
  echo "BOOTSTRAP_TAMPERED: $TEST_USER changed the repo, the venv or the pod volume $WS during $1:" >&2
  diff "$TAMPER_DIR/before" "$TAMPER_DIR/after" | head -n 40 >&2 || true
  exit 2
}
run cp "$REPO/ops/tree_manifest.py" "$TAMPER_DIR/tree_manifest.py"
run tamper_snapshot before
TEST_ENV=(runuser -u "$TEST_USER" -- env -i "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
          "HOME=$TEST_HOME" LANG=C.UTF-8 "HF_HOME=$HF_HOME" HF_HUB_OFFLINE=1 V2GPU_REQUIRE_TORCH=1
          PYTHONDONTWRITEBYTECODE=1)
# The root recollection below is the
# only root process of this step that used to run with no env -i at all, so on a real pod it was the sole ROOT
# process still carrying RUNPOD_API_KEY (and every other pod env var) in its own environment. It gets the same
# minimal, clean environment as $TEST_USER's runs, minus the user switch (it already runs as root).
ROOT_ENV=(env -i "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" "HOME=$WS" LANG=C.UTF-8
          "HF_HOME=$HF_HOME" HF_HUB_OFFLINE=1 V2GPU_REQUIRE_TORCH=1 PYTHONDONTWRITEBYTECODE=1)
if [ "$DRY_RUN" = 1 ]; then
  echo "+ ${TEST_ENV[*]} pytest --collect-only -q $TESTS $DESELECT -p no:cacheprovider"
  echo "+ tamper_check the $TEST_USER collection"
  echo "+ ${ROOT_ENV[*]} pytest --collect-only -q $TESTS $DESELECT -p no:cacheprovider"  # root's own count
  echo "+ tamper_snapshot before"
  N=N
else
  # shellcheck disable=SC2086  # TESTS and DESELECT are word lists
  N=$(cd "$REPO" && "${TEST_ENV[@]}" "$PY" -m pytest --collect-only -q $TESTS $DESELECT -p no:cacheprovider |
      tail -n 1 |
      awk '{split($1, n, "/"); print n[1]}')          # "723/724 tests collected (1 deselected)" -> 723
  tamper_check "the $TEST_USER collection"            # before root runs the repo's code itself
  # shellcheck disable=SC2086
  ROOT_N=$(cd "$REPO" && "${ROOT_ENV[@]}" "$PY" -m pytest --collect-only -q $TESTS $DESELECT -p no:cacheprovider |
           tail -n 1 |
           awk '{split($1, n, "/"); print n[1]}')
  [ "$N" = "$ROOT_N" ] || { echo "$TEST_USER collected $N tests but root collected $ROOT_N (a permission gap may be" \
       "hiding tests, e.g. torch, from $TEST_USER; the v2test gate): refusing" >&2; exit 2; }
  tamper_snapshot before   # taken again after root's own recollection (HOME=$WS), which is not the test user
fi
# tee writes two copies of the log -- root's own, in the root-only $TAMPER_DIR, which no_skips.sh reads,
# and the one on the pod volume for people, the gate and the copy home (the manifest skips it, and on MooseFS the test
# user may write it, so nothing is decided from it)
LOG_ROOT="$TAMPER_DIR/${LOG##*/}"                 # (no basename: a GATE_ONLY dry run's PATH holds only dirname)
run bash -c "set -o pipefail; cd '$REPO' && ${TEST_ENV[*]} '$PY' -m pytest -q -rs $TESTS $DESELECT -p no:cacheprovider 2>&1 | tee '$LOG_ROOT' '$LOG'"
run test_user_idle "after the test run"            # nothing of the test user outlives its run
# Idle FIRST, so no test-user process is alive when the last manifest is taken and none can
# start afterwards (only v2test starts v2test): the comparison below is final
run tamper_check "the $TEST_USER test run"          # before root runs no_skips.sh and step 6
run bash "$REPO/ops/no_skips.sh" "$LOG_ROOT" "$N" "$NDESELECT"
if [ "$KIND" = gen ] && [ "$GATE_ONLY" != 1 ]; then
  mark 5b
  echo "== 5b. vLLM and bitsandbytes import in the generation venv"
  run "$PY" -c "import vllm, bitsandbytes; print('vllm', vllm.__version__, 'bitsandbytes', bitsandbytes.__version__)"
fi

if [ "$GATE_ONLY" != 1 ]; then
mark 6
echo "== 6. environment receipt"
run bash -c "cd '$REPO' && '$PY' -m v2gpu.receipt > '$RECEIPT'"
fi
mark end
if [ "$DRY_RUN" != 1 ]; then echo "   bootstrap total: $(( $(date +%s) - BOOT_T0 )) s (per step: $TIMINGS)"; fi
echo "BOOTSTRAP_OK"
