#!/usr/bin/env bash
# Pod-parity gate (the laptop gate): proves ops/pod_bootstrap.sh's step 1.5 and
# step 5 (the only parts that switch to the unprivileged v2test user and touch the pod's own env file) are
# safe to run on a real pod, by actually running them in a Linux container that mimics one: the real (not DRY_RUN)
# bootstrap, root creates and switches to v2test as it does on the pod, /etc/rp_environment planted at 644:root with
# a fake key (as /start.sh leaves it on the first boot; step 5 itself chmods it to 600), --network none (no network,
# no GPU: GATE_ONLY=1 skips steps 1-4 and 6, which need one or the other; see ops/pod_bootstrap.sh's own comment on
# GATE_ONLY). Run before building the pod bundle after any change; CANARY_RUNSHEET.md §0 and GEN_RUNSHEET.md §0
# name it as a required pre-pod check. Must print POD_PARITY_GATE_OK; anything else is a refusal.
#
# Usage: bash ops/pod_parity_gate.sh [gpu|gen|both]   (default: both)
# The image (default gtestgen-cpu-gate, built from ops/gate_images/cpu-test; override with PODSAFETY_IMAGE) must already have on its PATH: bash, useradd,
# runuser, id, chmod, chown, stat, install, sha256sum, and a python3 with pytest, torch (CPU) and transformers
# installed matching requirements-gpu.in/requirements-gen.in closely enough for tests tests_gpu to collect and pass
# (KIND=gen's own vllm/bitsandbytes import, step 5b, is skipped by GATE_ONLY: this gate is about v2test's
# permissions, not about proving every generation package importable in a CPU container). PODSAFETY_HF_CACHE
# (default: the repository's own hf-cache/hub, read-only, never fetched over the network here) is copied
# into the container's HF_HOME/hub, so tests that load a real small tokenizer/config already cached locally do not
# have to skip; unset it, or leave nothing cached, and those tests just skip as they do without a GPU pod's models.
# The default is the project's own hf-cache, not the user's default
# Hugging Face cache, which the gate must never read.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${PODSAFETY_IMAGE:-gtestgen-cpu-gate}
HF_CACHE=${PODSAFETY_HF_CACHE:-$REPO/hf-cache/hub}
# Only a HF hub cache is ever mounted (read-only); a parent directory could hold credentials (token files),
# so a cache whose basename is not "hub" is refused outright rather than silently mounted.
if [ -d "$HF_CACHE" ] && [ "$(basename "$HF_CACHE")" != hub ]; then
  echo "PODSAFETY_HF_CACHE=$HF_CACHE: its basename must be 'hub' (only a HF hub cache is mounted, read-only):" \
       "refusing" >&2
  exit 2
fi
ARG=${1:-both}
case "$ARG" in
  both) KINDS="gpu gen" ;;
  gpu|gen) KINDS="$ARG" ;;
  *) echo "usage: $0 [gpu|gen|both]" >&2; exit 2 ;;
esac

command -v docker > /dev/null || { echo "docker not found: cannot run the pod-parity gate" >&2; exit 2; }
docker image inspect "$IMAGE" > /dev/null 2>&1 || {
  echo "image $IMAGE not found locally (set PODSAFETY_IMAGE to a Linux image with the packages this script's own" \
       "comment lists): refusing" >&2; exit 2; }
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")   # provenance, since the image has no Dockerfile here

WORK=$(mktemp -d)
# Logs of a FAILED run are kept ($OUT/run.out is needed to diagnose), only a clean run cleans up.
trap '[ "${FAIL:-1}" = 0 ] && rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"
git -C "$REPO" archive HEAD | tar -x -C "$WORK/src"   # only committed content: this gate checks what a pod checkout gets

FAIL=0
for KIND in $KINDS; do
  echo "=== pod-parity gate: KIND=$KIND (image $IMAGE $IMAGE_ID, --network none) ==="
  OUT="$WORK/out-$KIND"
  mkdir -p "$OUT"
  # FIRST_RUN=1 only when this checkout has no committed lock yet (its own first pod of this KIND has not run): the
  # lock check runs before GATE_ONLY's guards, and FIRST_RUN otherwise only affects step 2-3, which GATE_ONLY skips.
  # Decided from the ARCHIVED tree ($WORK/src, what a pod checkout actually gets), not the working tree,
  # which may hold a lock not yet committed (or lack one that is).
  if [ -f "$WORK/src/requirements-$KIND.lock" ]; then FR=0; else FR=1; fi
  # The second pass below re-runs step 5's exact v2test suite (same env -i, same N, same
  # no_skips.sh) with a real-pod-like PATH, without going back through GATE_ONLY's own nvidia-smi guard (which
  # would immediately refuse a real re-run of the whole bootstrap once nvidia-smi is on PATH -- that guard is
  # exactly the thing this pass exists to get PAST, not to re-trigger). Mirrors ops/pod_bootstrap.sh's own KIND
  # case (TESTS, DESELECT, the venv) so the replay is the same suite, not a redefinition of it.
  if [ "$KIND" = gpu ]; then
    TESTS2="$(printf 'tests/test_%s.py ' bind gherkin_steps history patterns scopes split stepdefs wrong_suite_control yield_report)tests_gpu"
    DESELECT2="--deselect tests_gpu/test_genqueue.py::test_the_generation_test_list_passes_in_a_simulated_generation_venv"
    NLOG2=pytest.log; NDS2=1; VENV2=venv
  else
    TESTS2=$(printf 'tests_gpu/test_%s.py ' runs engines outlog genprep generate gencoverage gencanary genqueue \
             serialize examples audit special_tokens); TESTS2=${TESTS2% }
    DESELECT2=""
    NLOG2=pytest.gen.log; NDS2=0; VENV2=venv-gen
  fi
  # The container script: plant the env file as /start.sh would (644 root, a fake key: real key material never
  # touches this laptop-only run), make the venvs from the image's own already-installed packages (no uv, no
  # network: --system-site-packages), copy in whatever HF cache is mounted (read-only; empty is fine), then run the
  # real (unshimmed) bootstrap with GATE_ONLY=1, so only step 1.5 (the fail-fast switch) and step 5 (the v2test
  # permission gate, the point of this gate) do anything.
  INNER=$(cat <<INNER
set -euo pipefail
cp -R /repo /work
cd /work
WS=/work-ws
# HF_HOME and uv's cache are on the container disk (/opt/v2/hf, /opt/v2/uv-cache), as on a pod
mkdir -p /opt/v2/hf   # HF_HOME/hub is normally made by step 4 (fetch_models), which GATE_ONLY skips; step 5 chmods it
if [ -d /hfcache ]; then cp -R /hfcache /opt/v2/hf/hub; else mkdir -p /opt/v2/hf/hub; fi
# The venvs live on the container's own disk (/opt/v2, the bootstrap's default root), never under \$WS
python3 -m venv --system-site-packages /opt/v2/venv
if [ "$KIND" = gen ]; then python3 -m venv --system-site-packages /opt/v2/venv-gen; fi
mkdir -p "\$WS/in" /opt/v2/uv-cache   # the bundle on the pod volume, uv's cache beside the venvs: step 5 makes both a-w
echo bundle > "\$WS/in/v2gpu-gate.bundle"; echo cache > /opt/v2/uv-cache/entry
printf 'export RUNPOD_API_KEY=FAKE-KEY\nexport RUNPOD_POD_ID=fake-pod-$KIND\n' > /etc/rp_environment
chmod 644 /etc/rp_environment
chown root:root /etc/rp_environment
touch /etc/v2gpu-parity-gate   # the ONLY place this marker is created; GATE_ONLY refuses without it
# Smoke pod 28/09 (EU-RO-1): a RunPod /workspace can be a MooseFS FUSE mount that refuses every
# ownership change (chmod works). ops/moosefs_shim.sh stands in front of chown and install for the whole bootstrap,
# refusing any ownership change under the workspace or the repo (on a pod the repo is under /workspace), so every
# gate run covers that case; the real binaries stay in /usr/bin (MOOSEFS_REAL_DIR).
[ -x /usr/bin/chown ] && [ -x /usr/bin/install ] || { echo "MOOSEFS_SHIM_SETUP_FAILED: no /usr/bin/chown or install"; exit 1; }
for b in chown install; do cp ops/moosefs_shim.sh /usr/local/bin/\$b; chmod 755 /usr/local/bin/\$b; done
export MOOSEFS_ROOTS="\$WS" MOOSEFS_REAL_DIR=/usr/bin   # only the pod volume; repo and venvs are local
FIRST_RUN=$FR GATE_ONLY=1 DRY_RUN=0 WS="\$WS" KIND="$KIND" bash ops/pod_bootstrap.sh
# A positive control -- the emulation proves nothing unless it was really active during the bootstrap
grep -qx "chown_probe: refused" "\$WS/volume_receipt.txt" || {
  echo "MOOSEFS_EMULATION_INACTIVE: the bootstrap's own ownership probe was not refused"; cat "\$WS/volume_receipt.txt"; exit 1; }
# decision 28/09: positive layout proof -- the venv the bootstrap used is on the container disk, none under \$WS
[ -x /opt/v2/venv/bin/python ] && [ ! -e "\$WS/venv" ] && [ ! -e "\$WS/venv-gen" ] && [ ! -e "\$WS/hf" ] && [ ! -e "\$WS/uv-cache" ] || {
  echo "LAYOUT_NOT_PROVEN: expected /opt/v2/venv and no venv, hf or uv-cache under \$WS"; ls -la /opt/v2 "\$WS"; exit 1; }
# The test user can write nothing root keeps for itself (in/ on the volume; hf/hub and uv-cache on the container
# disk since 05/10): a-w in step 5
for d in "\$WS/in/" "/opt/v2/hf/hub/" "/opt/v2/uv-cache/"; do
  if runuser -u v2test -- sh -c "echo x > '\$d.v2test-probe'" 2> /dev/null; then
    echo "WORKSPACE_WRITABLE_BY_TEST_USER: v2test wrote into \$d"; exit 1; fi
done
echo "-- second pass: a real pod's PATH (nvidia-smi, runpodctl, ssh, scp, curl, wget) --"
FAKELOG=/work-ws/fake-calls.log
: > "\$FAKELOG"
chmod 666 "\$FAKELOG"   # root creates the log; without this, v2test's own calls below could never
                        # write it (Permission denied, exit 0) and would silently vanish from the check further down
cat > /usr/local/bin/nvidia-smi <<'FAKEEOF'
#!/bin/sh
echo "\$0 \$*" >> /work-ws/fake-calls.log
echo "fake-gpu-0, driver 999.99, 16384 MiB"
FAKEEOF
for b in fake_runpodctl fake_ssh fake_scp fake_curl fake_wget; do
cat > "/usr/local/bin/\${b#fake_}" <<'FAKEEOF'
#!/bin/sh
echo "\$0 \$*" >> /work-ws/fake-calls.log
exit 0
FAKEEOF
done
chmod +x /usr/local/bin/nvidia-smi /usr/local/bin/runpodctl /usr/local/bin/ssh /usr/local/bin/scp /usr/local/bin/curl /usr/local/bin/wget
# A positive control -- an EMPTY log proves nothing only if v2test can actually write it. Prove that
# BEFORE trusting the suite's own silence: a real v2test call through the fake ssh must land in the log, or the
# gate refuses outright instead of giving a false SECOND_PASS_OK. The control's own line is then excluded (the log
# truncated) so the suite's own calls below are the only thing the closing check sees.
runuser -u v2test -- /usr/local/bin/ssh gate-selftest
grep -qx "/usr/local/bin/ssh gate-selftest" "\$FAKELOG" || {
  echo "GATE_SELFTEST_FAILED: v2test could not write \$FAKELOG"; exit 1; }
: > "\$FAKELOG"
N2=\$(tail -n 1 "\$WS/$NLOG2" | grep -oE '^[0-9]+')
LOG2=\$WS/pytest2.$KIND.log
runuser -u v2test -- env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/v2test LANG=C.UTF-8 HF_HOME=/opt/v2/hf HF_HUB_OFFLINE=1 V2GPU_REQUIRE_TORCH=1 PYTHONDONTWRITEBYTECODE=1 "/opt/v2/$VENV2/bin/python" -m pytest -q -rs $TESTS2 $DESELECT2 -p no:cacheprovider > "\$LOG2" 2>&1 || true
bash ops/no_skips.sh "\$LOG2" "\$N2" "$NDS2"
if [ -s "\$FAKELOG" ] && grep -qv "/usr/local/bin/nvidia-smi" "\$FAKELOG"; then
  echo "SECOND_PASS_REAL_POD_FAKE_CALLED"; cat "\$FAKELOG"; exit 1
fi
echo "SECOND_PASS_OK"
INNER
)
  HF_MOUNT=()
  [ -d "$HF_CACHE" ] && HF_MOUNT=(-v "$HF_CACHE:/hfcache:ro")
  set +e
  # bash 3.2 (macOS's /bin/bash) aborts expanding an EMPTY array under `set -u` ("unbound variable"); the
  # +"${HF_MOUNT[@]}" form is the portable no-op-when-empty idiom.
  docker run --rm --network none -v "$WORK/src:/repo:ro" ${HF_MOUNT[@]+"${HF_MOUNT[@]}"} "$IMAGE" bash -c "$INNER" \
    > "$OUT/run.out" 2> "$OUT/run.err"
  RC=$?
  set -e
  echo "  exit=$RC"
  tail -3 "$OUT/run.out" | sed 's/^/  | /'
  ok=1
  [ "$RC" = 0 ] || { echo "  REFUSED: bootstrap exited $RC (see $OUT/run.out, $OUT/run.err)" >&2; ok=0; }
  grep -qx "BOOTSTRAP_OK" "$OUT/run.out" || { echo "  REFUSED: no BOOTSTRAP_OK line" >&2; ok=0; }
  if grep -q "TRAP:" "$OUT/run.out" "$OUT/run.err"; then
    echo "  REFUSED: a real pod command was reached (see TRAP: lines above)" >&2; ok=0
  fi
  if grep -qi "FAKE-KEY" "$OUT/run.out" "$OUT/run.err"; then
    echo "  REFUSED: the fake key leaked into the gate's own output" >&2; ok=0
  fi
  # The second pass's own fake-call log must be provably writable by v2test before an empty log is
  # trusted; the container's positive control (a real v2test ssh call before the suite) failed to land in it.
  if grep -q "GATE_SELFTEST_FAILED" "$OUT/run.out" "$OUT/run.err"; then
    echo "  REFUSED: v2test could not prove it can write the fake-call log (see GATE_SELFTEST_FAILED above)" >&2
    ok=0
  fi
  # Both passes must be green -- the first (GATE_ONLY's own no-GPU-tooling path, BOOTSTRAP_OK
  # above) and the second (a real pod's PATH, step 5 replayed and no fake but nvidia-smi reached).
  if grep -q "SECOND_PASS_REAL_POD_FAKE_CALLED" "$OUT/run.out" "$OUT/run.err"; then
    echo "  REFUSED: the second pass's real-pod-like PATH reached a fake other than nvidia-smi (see the log above)" >&2
    ok=0
  fi
  grep -qx "SECOND_PASS_OK" "$OUT/run.out" || { echo "  REFUSED: no SECOND_PASS_OK line" >&2; ok=0; }
  if [ "$ok" = 1 ]; then echo "  KIND=$KIND: OK"; else FAIL=1; fi
done

if [ "$FAIL" = 0 ]; then
  echo "POD_PARITY_GATE_OK"
else
  echo "POD_PARITY_GATE_FAILED (logs kept: $WORK)" >&2
  exit 1
fi
