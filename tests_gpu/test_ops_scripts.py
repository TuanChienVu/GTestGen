import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests_gpu.lockfree import clean_env, dry_run, dry_run_no_gpu_path, hermetic_local, lockfree_checkout
from v2gpu.jobqueue import write_manifest

OPS = Path(__file__).resolve().parents[1] / "ops"


def user_write_back(root) -> None:
    """Give the user write bit back to root and to every directory and file under it (symlinks untouched)."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, os.lstat(p).st_mode | 0o200)
    os.chmod(root, os.lstat(root).st_mode | 0o200)


@pytest.fixture(autouse=True)
def _owner_write_back_at_teardown(request):
    """Step 5 of a real bootstrap run makes ws/in, hf/hub and uv-cache (since 05/10: local/) read-only (a-w), which pytest cannot
    remove when it rotates old base temp directories; every test of this file gets its tmp_path writable again."""
    # fetched at setup (a teardown is too late: tmp_path would already be torn down), only for a test that uses it
    root = request.getfixturevalue("tmp_path") if "tmp_path" in request.fixturenames else None
    yield
    if root is not None and root.exists():
        user_write_back(root)


def test_scripts_parse():
    for s in ("pod_bootstrap.sh", "copy_out.sh", "no_skips.sh", "ship.sh", "watch_pod.sh", "pod_parity_gate.sh",
             "populate_hf_cache.sh"):
        assert subprocess.run(["bash", "-n", str(OPS / s)]).returncode == 0


def test_populate_hf_cache_never_deletes_and_uses_tokenizer_only(tmp_path):
    """The populate script only ever downloads and verifies (no rm/unlink/shutil.rmtree of
    any kind), and forces TOKENIZER_ONLY (never the weight patterns) for every model, checked against the Hub's own
    LFS sha256 like v2gpu.fetch_models.check() already does for a real pod fetch."""
    src = (OPS / "populate_hf_cache.sh").read_text()
    assert not re.search(r"\brm\s+-", src) and "rmtree" not in src and "unlink" not in src
    assert "TOKENIZER_ONLY" in src and "allow_patterns=TOKENIZER_ONLY" in src
    assert "weights=False" in src   # v2gpu.fetch_models.check(..., weights=False): no weight index required
    assert 'DEFAULT_HOME=$(cd "$(dirname "$0")/.." && pwd)/hf-cache' in src


# ops/pod_parity_gate.sh hardening (27/09/2026) that does not need Docker to check


def test_a_hf_cache_whose_basename_is_not_hub_is_refused(tmp_path):
    """(f): only a HF hub cache is ever mounted (read-only); a parent directory could hold credentials, so a cache
    whose basename is not "hub" is refused before docker is even looked for (no Docker needed for this test)."""
    bad = tmp_path / "not-hub"
    bad.mkdir()
    out = subprocess.run(["bash", str(OPS / "pod_parity_gate.sh")], capture_output=True, text=True,
                         env={**os.environ, "PODSAFETY_HF_CACHE": str(bad)})
    assert out.returncode == 2 and "hub" in out.stderr, (out.stdout, out.stderr)


def test_a_hf_cache_named_hub_is_not_refused_by_the_basename_check(tmp_path):
    """The other side: a real 'hub' basename passes this specific check. PODSAFETY_IMAGE names a tag that cannot
    exist, so the script still refuses quickly (the very next checks, no network) rather than actually launching a
    container -- this test is only about the basename check, never about running the gate for real. Two refusals are
    acceptable past the basename check: "not found locally" (docker present, image absent) or "docker not found"
    (no docker on PATH at all, e.g. nested inside the gate's own step 5, which runs with no docker-in-docker)."""
    good = tmp_path / "hub"
    good.mkdir()
    out = subprocess.run(["bash", str(OPS / "pod_parity_gate.sh")], capture_output=True, text=True,
                         env={**os.environ, "PODSAFETY_HF_CACHE": str(good),
                             "PODSAFETY_IMAGE": "no-such-image-f4j-m3f:1"})
    assert "basename must be 'hub'" not in out.stderr, (out.stdout, out.stderr)
    assert out.returncode == 2 and ("not found locally" in out.stderr or "docker not found" in out.stderr), \
        (out.stdout, out.stderr)


def test_the_gate_uses_the_bash_3_2_safe_empty_array_idiom():
    """(c): /bin/bash on macOS is 3.2, where `set -u` aborts expanding an EMPTY array ("unbound variable"); the
    plain "${HF_MOUNT[@]}" form used to crash the gate on an unset PODSAFETY_HF_CACHE. Pinned two ways: the source
    uses the safe idiom, and that exact idiom is exercised against a real bash 3.2 when this machine has one.
    Since a pod-parity gate run (27/09), the dynamic half no-ops instead of pytest.skip()-ing on a modern bash
    (every Linux pod and the gate's own container ship bash 5.x) -- a skip here made no_skips.sh refuse the whole
    gate, since it treats any skip as a sign of a missing package or tokenizer, never an intentionally
    machine-conditional check."""
    src = (OPS / "pod_parity_gate.sh").read_text()
    assert '${HF_MOUNT[@]+"${HF_MOUNT[@]}"}' in src


def test_podsafety_hf_cache_defaults_to_the_v2_only_artifacts_cache():
    """PODSAFETY_HF_CACHE's default is the project's own hf-cache "hub", not
    the user's default Hugging Face cache, which the gate must never read."""
    src = (OPS / "pod_parity_gate.sh").read_text()
    assert 'HF_CACHE=${PODSAFETY_HF_CACHE:-' \
           '$REPO/hf-cache/hub}' in src
    assert "$HOME/.cache/huggingface/hub" not in src
    assert 'docker run --rm --network none -v "$WORK/src:/repo:ro" "${HF_MOUNT[@]}"' not in src   # the bare, unsafe form
    version = subprocess.run(["/bin/bash", "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout
    if not version.startswith("3."):
        return   # the static assertions above already ran; the dynamic exercise below only applies to bash 3.x
    script = ('set -euo pipefail\nHF_MOUNT=()\n'
             'echo -n "" ${HF_MOUNT[@]+"${HF_MOUNT[@]}"}\necho OK\n')
    out = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, (out.returncode, out.stdout, out.stderr)
    broken = 'set -euo pipefail\nHF_MOUNT=()\necho -n "" "${HF_MOUNT[@]}"\necho OK\n'
    bad = subprocess.run(["/bin/bash", "-c", broken], capture_output=True, text=True)
    assert bad.returncode != 0 and "unbound variable" in bad.stderr, bad.stderr   # confirms the idiom is load-bearing


# Root created the second pass's fake-call log at 0644,
# so a real v2test ssh/scp/curl/wget/runpodctl call failed "Permission denied" (exit 0) and was never recorded --
# the closing check then saw an empty log and printed a false SECOND_PASS_OK.


def test_fakelog_is_world_writable_and_proves_itself_with_a_selftest_before_the_suite():
    """Fixed two ways, both pinned here: (1) the log is made world-writable right after it is created; (2) a
    positive control -- a real v2test ssh call ("gate-selftest") -- must land in the log BEFORE the suite runs, or
    the gate refuses outright (GATE_SELFTEST_FAILED), proving the log is actually writable rather than just
    happening to stay empty; the control's own line is then excluded (the log truncated again) so the suite's own
    calls are the only thing the closing SECOND_PASS_REAL_POD_FAKE_CALLED check sees."""
    src = (OPS / "pod_parity_gate.sh").read_text()
    # inside the docker heredoc, "$" is escaped as "\$" so the container (not this outer script) expands it
    assert ': > "\\$FAKELOG"\nchmod 666 "\\$FAKELOG"' in src
    selftest = "runuser -u v2test -- /usr/local/bin/ssh gate-selftest"
    assert selftest in src
    assert 'grep -qx "/usr/local/bin/ssh gate-selftest" "\\$FAKELOG"' in src
    assert 'GATE_SELFTEST_FAILED: v2test could not write \\$FAKELOG' in src
    # the control runs, and its own line is excluded, strictly BEFORE the real suite's pytest invocation
    assert src.index(selftest) < src.index("-m pytest -q -rs")
    assert src.count(': > "\\$FAKELOG"') == 2   # once at creation, once again right after the self-test passes
    # the outer loop (run OUTSIDE the container, after `docker run`) also refuses by name on this signal
    outer = src[src.index("RC=$?"):]
    assert 'grep -q "GATE_SELFTEST_FAILED" "$OUT/run.out" "$OUT/run.err"' in outer


def test_bootstrap_dry_run_walks_every_step_in_order(tmp_path):
    out = dry_run(tmp_path).stdout
    steps = ["nvidia-smi", "venv", "-r", "v2gpu.fetch_models", "pytest -q -rs tests/test_bind.py tests/test_gherkin_steps.py tests/test_history.py tests/test_patterns.py tests/test_scopes.py tests/test_split.py tests/test_stepdefs.py tests/test_wrong_suite_control.py tests/test_yield_report.py tests_gpu", "no_skips.sh",
             "v2gpu.receipt", "BOOTSTRAP_OK"]
    pos = [out.index(s) for s in steps]
    assert pos == sorted(pos)
    assert not (tmp_path / "venv").exists()   # dry run touches nothing


def gate_marker_env(tmp_path):
    """GATE_ONLY's own guard needs the gate's marker present (GATE_MARKER, a
    test-only knob like RP_ENV_FILE/V2TEST_HOME) -- moving it needs V2GPU_BOOTSTRAP_TEST=1 too."""
    marker = tmp_path / "v2gpu-parity-gate"
    marker.write_text("")
    return {"GATE_MARKER": str(marker), "V2GPU_BOOTSTRAP_TEST": "1"}


def test_gate_only_skips_the_gpu_and_network_steps_but_keeps_1_5_and_5(tmp_path):
    """The laptop gate: ops/pod_parity_gate.sh runs the bootstrap in a network-less, GPU-less
    container, so it sets GATE_ONLY=1 to skip nvidia-smi (1), uv/venv/packages (2-3), fetch_models (4) and the
    receipt (6) — none of them possible offline without a GPU — while still running the fail-fast switch (1.5)
    and the real test step (5) it is about. Uses dry_run_no_gpu_path, not
    dry_run, because this test means to exercise the "no GPU tooling" path, and the process running the SUITE ITSELF
    may have a real nvidia-smi on its PATH (any GPU pod's own step 5 does) -- that must never make this test see a
    real pod and refuse."""
    out = dry_run_no_gpu_path(tmp_path, GATE_ONLY="1", **gate_marker_env(tmp_path)).stdout
    for absent in ("nvidia-smi", "-m uv", "v2gpu.fetch_models", "v2gpu.receipt"):
        assert absent not in out, (absent, out)
    for present in ("fail-fast check", "pytest -q -rs tests/test_bind.py tests/test_gherkin_steps.py tests/test_history.py tests/test_patterns.py tests/test_scopes.py tests/test_split.py tests/test_stepdefs.py tests/test_wrong_suite_control.py tests/test_yield_report.py tests_gpu", "no_skips.sh", "BOOTSTRAP_OK"):
        assert present in out, (present, out)
    assert out.index("fail-fast check") < out.index("pytest -q -rs tests/test_bind.py tests/test_gherkin_steps.py tests/test_history.py tests/test_patterns.py tests/test_scopes.py tests/test_split.py tests/test_stepdefs.py tests/test_wrong_suite_control.py tests/test_yield_report.py tests_gpu")
    # KIND=gen's step 5b (vllm/bitsandbytes import) needs packages this container image never has; also skipped.
    gen_out = dry_run_no_gpu_path(tmp_path, GATE_ONLY="1", KIND="gen", **gate_marker_env(tmp_path)).stdout
    assert "vllm" not in gen_out and "bitsandbytes" not in gen_out and "BOOTSTRAP_OK" in gen_out
    assert "vllm" in dry_run(tmp_path, KIND="gen").stdout                     # unchanged without GATE_ONLY


def test_gate_only_is_never_set_together_with_a_real_pod_run(tmp_path):
    """A static pin: GATE_ONLY must default to 0, so a RUNSHEET file or a forgotten env var never silently turns a real
    pod's bootstrap into the laptop gate's short-circuited version."""
    src = (OPS / "pod_bootstrap.sh").read_text()
    assert "GATE_ONLY=${GATE_ONLY:-0}" in src


# Until 27/09/2026 GATE_ONLY had no guard at all -- a pasted or edited launch
# line could reach a real pod's bootstrap with GATE_ONLY=1 and every test stayed green (a mutant launch line with GATE_ONLY=1: 213/213 passed)


def test_gate_only_without_the_markers_own_file_is_refused(tmp_path):
    """The gate's own marker is what lets GATE_ONLY run at all; without it, the bootstrap refuses before touching
    anything. GATE_MARKER is pointed at a tmp_path file that is provably absent, rather than relying on the
    default /etc/v2gpu-parity-gate being absent on the machine running the test: nested inside the gate's own step
    5, the OUTER pod_parity_gate.sh invocation (KIND=gpu, GATE_ONLY=1) has already planted that exact default file
    on the container, which let this test pass there with the marker present (pod-parity gate run, 27/09).
    Uses dry_run_no_gpu_path, not dry_run, because this test means to exercise the
    marker-missing refusal specifically, never the "looks like a real pod" one -- a real nvidia-smi on the suite
    process's own PATH (any GPU pod's step 5 has one) must not make it refuse for the WRONG reason first."""
    out = dry_run_no_gpu_path(tmp_path, GATE_ONLY="1", GATE_MARKER=str(tmp_path / "v2gpu-parity-gate"))
    assert out.returncode == 2 and "v2gpu-parity-gate" in out.stderr
    assert not (tmp_path / "venv").exists()


def test_gate_only_with_nvidia_smi_on_path_and_no_marker_exits_2_with_empty_stdout(tmp_path):
    """A real-run test: a fake nvidia-smi on PATH (this looks like a real pod) refuses GATE_ONLY even before
    it would have reached the marker check, with nothing printed to stdout (no step, no BOOTSTRAP_OK). RP_ENV_FILE
    and V2TEST_HOME are still moved into tmp_path, as every real (non-DRY_RUN) test run of this script must
    (test_pod_safety.py's static scan), even though this refusal is expected to land before either is ever read."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "nvidia-smi").write_text("#!/bin/sh\necho fake\n")
    (shim / "nvidia-smi").chmod(0o755)
    env = {**clean_env(), "WS": str(tmp_path), "FIRST_RUN": "1", "DRY_RUN": "0", "GATE_ONLY": "1",
          "RP_ENV_FILE": str(tmp_path / "rp_environment"), "V2TEST_HOME": str(tmp_path / "v2test-home"),
          "V2GPU_LOCAL": str(tmp_path / "local"), "V2GPU_BOOTSTRAP_TEST": "1",
          "PATH": f"{shim}:{os.environ.get('PATH', '')}"}
    out = subprocess.run(["bash", str(lockfree_checkout(tmp_path))], capture_output=True, text=True, env=env)
    assert out.returncode == 2 and out.stdout == "" and "nvidia-smi" in out.stderr


def test_gate_only_static_scan_only_the_bootstrap_and_the_gate_may_name_it(tmp_path):
    """Only pod_bootstrap.sh (default, guard) and pod_parity_gate.sh (its docker INNER line) may name
    GATE_ONLY; GATE_MARKER (test-only) is the bootstrap's alone. A planted RUNSHEET file or script flags the scan."""
    from tests_gpu.test_pod_safety import gate_problems

    assert gate_problems(OPS) == []
    planted = tmp_path / "ops"
    shutil.copytree(OPS, planted)
    (planted / "GEN_RUNSHEET.md").write_text(
        (planted / "GEN_RUNSHEET.md").read_text() + "\nGATE_ONLY=1 bash ops/pod_bootstrap.sh\n")
    (planted / "CANARY_RUNSHEET.md").write_text(
        (planted / "CANARY_RUNSHEET.md").read_text() + "\nGATE_MARKER=/tmp/x bash ops/pod_bootstrap.sh\n")
    problems = gate_problems(planted)
    assert len(problems) == 2, problems
    assert any(p.startswith("GEN_RUNSHEET.md: ") and "GATE_ONLY" in p for p in problems)
    assert any(p.startswith("CANARY_RUNSHEET.md: ") and "GATE_MARKER" in p for p in problems)


def test_clean_env_strips_gate_only(monkeypatch):
    """A mutant (dropping GATE_ONLY from lockfree.SELECTORS) was never pinned
    directly -- test_ops_scripts.py, test_pod_safety.py and test_bootstrap_locks.py alone gave 89/89 passed under
    it. clean_env() must strip GATE_ONLY exactly like KIND/GEN_SET/FIRST_RUN: a pod's own bootstrap invocation
    (GATE_ONLY=1, the laptop gate) must never leak into a nested dry_run's environment."""
    monkeypatch.setenv("GATE_ONLY", "1")
    assert "GATE_ONLY" not in clean_env()


def test_bootstrap_rerun_keeps_the_existing_venv(tmp_path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (tmp_path / "venv" / "bin" / "python").chmod(0o755)
    out = dry_run(tmp_path, V2GPU_LOCAL=str(tmp_path), V2GPU_BOOTSTRAP_TEST="1").stdout   # venv root moved
    commands = [line for line in out.splitlines() if line.startswith("+ ")]
    assert not any(" venv " in c for c in commands) and "BOOTSTRAP_OK" in out


def test_the_dry_run_helpers_keep_the_venv_root_in_tmp_path_unless_told(tmp_path):
    """/opt/v2 exists where this suite also runs (the gate container, a pod's step 5), so a dry run keeps its venv
    root in tmp_path; only default_local=True keeps the default (tests_gpu/test_pod_safety.py pins who does)."""
    moved = dry_run(tmp_path / "a", KIND="gpu").stdout
    assert f"{tmp_path / 'a' / 'local' / 'venv'}/bin/python" in moved and "/opt/v2" not in moved
    kept = dry_run(tmp_path / "b", KIND="gpu", default_local=True).stdout
    assert "/opt/v2/venv/bin/python" in kept
    src = (Path(__file__).parent / "lockfree.py").read_text()
    body = src[src.index("def dry_run_no_gpu_path"):]
    assert "local = {} if default_local else hermetic_local(tmp_path)" in body and "**local, **env" in body


def test_bootstrap_installs_uv_when_the_image_lacks_it(tmp_path):
    bin_dir = tmp_path / "bin"   # a PATH with bash and coreutils but no uv
    bin_dir.mkdir()
    for tool in ("bash", "dirname", "pwd"):
        for d in ("/bin", "/usr/bin"):
            if Path(d, tool).exists():
                (bin_dir / tool).symlink_to(Path(d, tool))
                break
    out = subprocess.run(["bash", str(lockfree_checkout(tmp_path)), "deepseek"], capture_output=True, text=True,
                         env={**hermetic_local(tmp_path), "PATH": str(bin_dir), "DRY_RUN": "1",   # as in the gate container
                              "WS": str(tmp_path / "ws"), "FIRST_RUN": "1"}).stdout
    assert "pip install --break-system-packages --quiet uv" in out and "-m uv venv" in out
    assert "fetch_report.json' deepseek" in out


def test_a_failing_test_run_is_not_hidden_by_tee(tmp_path):
    line = next(l for l in (OPS / "pod_bootstrap.sh").read_text().splitlines() if "pytest" in l and "tee" in l)
    assert "set -o pipefail;" in line
    r = subprocess.run(["bash", "-c", "set -o pipefail; false | tee /dev/null"])
    assert r.returncode != 0


def gate(log, text, expected, *deselected):
    log.write_text(text)
    return subprocess.run(["bash", str(OPS / "no_skips.sh"), str(log), str(expected), *map(str, deselected)],
                          capture_output=True).returncode


def test_no_skips_gate_requires_exactly_the_collected_count(tmp_path):
    log = tmp_path / "pytest.log"
    assert gate(log, "148 passed in 30.0s\n", 148) == 0
    assert gate(log, "SKIPPED [3] tokenizer not cached\n145 passed, 3 skipped in 30.0s\n", 148) == 1
    assert gate(log, "140 passed in 30.0s\n", 148) == 1              # tests vanished from collection
    assert gate(log, "147 passed, 1 xfailed in 30.0s\n", 148) == 1


def test_bootstrap_counts_the_collected_tests_first(tmp_path):
    out = dry_run(tmp_path).stdout
    assert out.index("--collect-only") < out.index("pytest -q -rs tests/test_bind.py tests/test_gherkin_steps.py tests/test_history.py tests/test_patterns.py tests/test_scopes.py tests/test_split.py tests/test_stepdefs.py tests/test_wrong_suite_control.py tests/test_yield_report.py tests_gpu") < out.index("no_skips.sh")


def make_run(root: Path, job: str) -> Path:
    wd = root / job
    (wd / "epoch-1").mkdir(parents=True)
    (wd / "epoch-1" / "adapter_model.safetensors").write_bytes(b"weights")
    (wd / "receipt-train.json").write_text("{}")
    (wd / "state").mkdir()
    (wd / "state" / "trainer.pt").write_bytes(b"big optimizer state")
    write_manifest(wd)
    return wd


def copy(tmp_path, job="deepseek-s42-all", queue=None, env=None):
    args = [str(queue)] if queue else []
    return subprocess.run(["bash", str(OPS / "copy_out.sh"), "ip", "22", "key", str(tmp_path / "pod"), job,
                           str(tmp_path / "home"), *args], capture_output=True, text=True,
                          env={**os.environ, "COPY_OUT_LOCAL": "1", "KNOWN_HOSTS": str(tmp_path / "known_hosts.pod-1"),
                               **(env or {})})


def test_copy_out_verifies_and_skips_state(tmp_path):
    make_run(tmp_path / "pod", "deepseek-s42-all")
    r = copy(tmp_path)
    assert r.returncode == 0 and "COPY_OK deepseek-s42-all" in r.stdout
    home = tmp_path / "home" / "deepseek-s42-all"
    assert (home / "epoch-1" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert not (home / "state").exists()


def test_copy_out_fails_on_a_changed_file(tmp_path):
    wd = make_run(tmp_path / "pod", "deepseek-s42-all")
    (wd / "epoch-1" / "adapter_model.safetensors").write_bytes(b"weightz")   # changed after the manifest
    assert copy(tmp_path).returncode != 0


def test_copy_out_refuses_to_overwrite(tmp_path):
    make_run(tmp_path / "pod", "deepseek-s42-all")
    (tmp_path / "home" / "deepseek-s42-all").mkdir(parents=True)
    assert copy(tmp_path).returncode == 2


def test_copy_out_skips_nested_state_and_fetches_the_queue_logs(tmp_path):
    wd = tmp_path / "pod" / "canary"
    (wd / "deepseek-4096" / "run" / "state").mkdir(parents=True)
    (wd / "deepseek-4096" / "run" / "state" / "trainer.pt").write_bytes(b"optimizer")
    (wd / "deepseek-4096" / "run" / "state.old").mkdir()
    (wd / "deepseek-4096" / "run" / "receipt-train.json").write_text("{}")
    write_manifest(wd)
    q = tmp_path / "pod-queue"
    q.mkdir()
    (q / "canary.log").write_text("Traceback ...")
    (q / "canary.status").write_text("FAILED\n")
    (q / "outcome.json").write_text("{}")
    r = copy(tmp_path, "canary", queue=q)
    assert r.returncode == 0, r.stdout + r.stderr
    home = tmp_path / "home"
    assert (home / "canary" / "deepseek-4096" / "run" / "receipt-train.json").exists()
    assert not (home / "canary" / "deepseek-4096" / "run" / "state").exists()
    assert not (home / "canary" / "deepseek-4096" / "run" / "state.old").exists()
    assert sorted(p.name for p in (home / "canary.queue").iterdir()) == ["canary.log", "canary.status", "outcome.json"]


GNU_TAR_SHIM = """#!/usr/bin/env bash
# GNU tar refuses to create an archive with no file operand and no -T list ("Cowardly refusing ..."); bsdtar does not.
create=0; list=0; ops=0
for a in "$@"; do
  case "$a" in -c*|-*c*f) create=1 ;; esac
  case "$a" in -T|--files-from*) list=1 ;; -*|-) ;; *) ops=$((ops + 1)) ;; esac
done
if [ $create = 1 ] && [ $list = 0 ] && [ $ops = 0 ]; then echo "tar: Cowardly refusing to create an empty archive" >&2; exit 2; fi
exec /usr/bin/tar "$@"
"""


def test_copy_out_survives_an_empty_queue_dir_under_gnu_tar(tmp_path):
    wd = tmp_path / "pod" / "job"
    wd.mkdir(parents=True)
    (wd / "receipt-train.json").write_text("{}")
    write_manifest(wd)
    (tmp_path / "pod-queue").mkdir()                                  # the queue died before writing anything
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "tar").write_text(GNU_TAR_SHIM)
    (shim / "tar").chmod(0o755)
    env = {"PATH": f"{shim}:{os.environ['PATH']}"}
    r = copy(tmp_path, "job", queue=tmp_path / "pod-queue", env=env)
    assert r.returncode == 0 and "COPY_OK job" in r.stdout, r.stdout + r.stderr



@pytest.mark.parametrize("summary, deselected, ok", [
    ("148 passed, 1 warning in 30.0s", 0, True),               # a pod package's warning is not a failure
    ("148 passed, 12 warnings in 30.0s", 0, True),
    ("148 passed, 1 deselected in 30.0s", 1, True),            # the gpu bootstrap's own deselect, counted
    ("148 passed, 1 deselected, 3 warnings in 30.0s", 1, True),
    ("148 passed, 1 deselected in 30.0s", 0, False),           # a deselect nobody declared
    ("148 passed in 30.0s", 1, False),                         # the declared deselect did not happen
    ("148 passed, 2 deselected in 30.0s", 1, False),
    ("147 passed, 1 skipped, 2 warnings in 30.0s", 0, False),
    ("147 passed, 1 xfailed, 1 warning in 30.0s", 0, False),
    ("147 passed, 1 xpassed in 30.0s", 0, False),
    ("147 passed, 1 failed, 1 warning in 30.0s", 0, False),
    ("148 passed, 1 warning, 1 error in 30.0s", 0, False),
    ("148 passed, 1 warnings in 30.0s", 0, True),
    ("148 passed, one warning in 30.0s", 0, False),
])
def test_no_skips_accepts_warnings_and_the_declared_deselect_only(tmp_path, summary, deselected, ok):
    assert gate(tmp_path / "pytest.log", summary + "\n", 148, deselected) == (0 if ok else 1)


def test_results_and_ship_folders_stay_out_of_git():
    """The laptop's ship/, results/ and results-lost/ live in the worktree; the tree stays clean (§0 of the RUNSHEET files)."""
    ignored = set((OPS.parent / ".gitignore").read_text().split())
    assert {"ship/", "results/", "results-lost/"} <= ignored


# The requirements lock


def repo_copy(tmp_path, lock=None, kind="gen"):
    """The bootstrap in a checkout of its own (ops/, requirements-<kind>.in and, if given, the lock)."""
    repo = tmp_path / "repo"
    (repo / "ops").mkdir(parents=True)
    for s in ("pod_bootstrap.sh", "no_skips.sh", "tree_manifest.py"):     # step 5's tamper check
        (repo / "ops" / s).write_bytes((OPS / s).read_bytes())
    (repo / f"requirements-{kind}.in").write_text("vllm==0.27.1\n")
    if lock is not None:
        (repo / f"requirements-{kind}.lock").write_text(lock)
    return repo


@pytest.mark.parametrize("kind", ["gen", "gpu"])
def test_a_bootstrap_without_its_lock_refuses_unless_it_is_the_first_run(tmp_path, kind):
    """Only the first pod of a KIND resolves packages (FIRST_RUN=1, said explicitly); every later pod installs the
    committed lock, so pods never resolve the unpinned transitive set on their own."""
    repo = repo_copy(tmp_path, kind=kind)
    env = {**clean_env(), **hermetic_local(tmp_path), "DRY_RUN": "1", "WS": str(tmp_path / "ws"), "KIND": kind}
    out = subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 2 and f"requirements-{kind}.lock" in out.stderr and "FIRST_RUN=1" in out.stderr
    first = subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                           env={**env, "FIRST_RUN": "1"})
    assert first.returncode == 0 and "FIRST RUN" in first.stdout and "FIRST_RUN_LOCK.json" in first.stdout
    locked = repo_copy(tmp_path / "l", "vllm==0.27.1\n", kind=kind)
    ok = subprocess.run(["bash", str(locked / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True, env=env)
    assert ok.returncode == 0 and "pip sync" in ok.stdout and "FIRST RUN" not in ok.stdout
    both = subprocess.run(["bash", str(locked / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                          env={**env, "FIRST_RUN": "1"})
    assert both.returncode == 2 and "FIRST_RUN" in both.stderr         # a lock exists: never re-resolve


@pytest.mark.parametrize("kind", ["gen", "gpu"])
def test_uv_waits_300_s_on_the_network_by_default(tmp_path, kind):
    """First generation pod, 04/10: uv's default 30 s HTTP timeout failed a FIRST_RUN gen bootstrap on EUR-IS-1
    ("Failed to extract z3_solver ... network timeout"). The bootstrap exports UV_HTTP_TIMEOUT=300 unless set: a
    longer wait only, the resolution and the lock are unchanged. Step 3 prints the value uv will use."""
    locked = repo_copy(tmp_path, "vllm==0.27.1\n", kind=kind)
    env = {**clean_env(), **hermetic_local(tmp_path), "DRY_RUN": "1", "WS": str(tmp_path / "ws"), "KIND": kind,
           "UV_HTTP_TIMEOUT": ""}                      # unset or empty: the default (bash ${VAR:-300})
    out = subprocess.run(["bash", str(locked / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "   uv HTTP timeout: 300 s" in out.stdout, (out.stdout[-1500:], out.stderr[-800:])
    assert out.stdout.index("uv HTTP timeout") < out.stdout.index("pip sync")
    mine = subprocess.run(["bash", str(locked / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                          env={**env, "UV_HTTP_TIMEOUT": "600"})
    assert mine.returncode == 0 and "   uv HTTP timeout: 600 s" in mine.stdout


# A mutant that once survived


def test_the_bootstrap_test_step_never_passes_its_selectors_or_first_run_to_the_suite(tmp_path):
    """The pod's suite must see the laptop's environment: every pytest invocation of step 5 (both v2test runs and the
    root recollection) uses a clean env -i prefix, run exactly as the bootstrap prints it, dropping KIND, GEN_SET and
    FIRST_RUN that the operator exported (FIRST_RUN=1 on the first pod), and never re-exporting RUNPOD_* either. The
    root recollection used to inherit the bootstrap's own (a real pod's) environment outright; it
    now gets the same minimal env -i as v2test, just without the runuser wrapper and with a different HOME."""
    out = dry_run(tmp_path, KIND="gen", GEN_SET="fallback-tf").stdout
    lines = [l for l in out.splitlines() if "pytest" in l and ("--collect-only" in l or " -q -rs " in l)]
    assert len(lines) == 3
    switched = [l for l in lines if "env -i" in l]
    assert len(switched) == 3                          # the root recollection now also gets a clean env -i
    prefixes = {re.search(r"env -i(?: \S+=\S+)+", l).group(0) for l in switched}   # runuser's and root's own
    assert len(prefixes) == 2                    # v2test's (HOME=.../home/v2test) and root's (HOME=$WS) differ only
    for prefix in prefixes:
        shown = subprocess.run([*prefix.split(), "env"], capture_output=True, text=True,
                               env={**os.environ, "KIND": "gen", "GEN_SET": "fallback-tf", "FIRST_RUN": "1",
                                    "RUNPOD_POD_ID": "pod-x"}).stdout
        names = {line.split("=", 1)[0] for line in shown.splitlines()}
        assert names == {"PATH", "HOME", "LANG", "HF_HOME", "HF_HUB_OFFLINE", "V2GPU_REQUIRE_TORCH",
                         "PYTHONDONTWRITEBYTECODE"}, sorted(names)          # no bytecode written


# The first run's lock record


def first_run_lock_command(tmp_path, **env) -> str:
    """The bootstrap's FIRST_RUN_LOCK.json command, as its dry run prints it (`+ bash -c <script>`)."""
    out = dry_run(tmp_path, V2GPU_LOCAL=str(tmp_path), V2GPU_BOOTSTRAP_TEST="1", **env).stdout
    line = next(l for l in out.splitlines() if l.startswith("+ bash -c ") and "FIRST_RUN_LOCK.json" in l)
    return line[len("+ bash -c "):]


@pytest.mark.parametrize("env, lock, venv", [({}, "requirements-gpu.lock", "venv"),
                                             ({"KIND": "gen"}, "requirements-gen.lock", "venv-gen"),
                                             ({"KIND": "gen", "GEN_SET": "fallback-bnb"},
                                              "requirements-gen-fallback-bnb.lock", "venv-gen")])
def test_the_first_run_records_the_sha256_of_the_lock_it_froze_under_its_own_name(tmp_path, env, lock, venv):
    """FIRST_RUN_LOCK.json is exactly {<the kind's (and pin set's) lock name>: sha256 of the frozen
    <lock>.new}, never of the .in; the command runs as the dry run prints it."""
    script = first_run_lock_command(tmp_path, **env)
    (tmp_path / venv).mkdir()
    (tmp_path / f"{lock}.new").write_text("torch==2.14.0\npeft==0.21.0\n")
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    record = json.loads((tmp_path / venv / "FIRST_RUN_LOCK.json").read_text())
    frozen = hashlib.sha256((tmp_path / f"{lock}.new").read_bytes()).hexdigest()
    assert record == {lock: frozen}


SHA_FAILS = "#!/bin/sh\necho 'sha256sum: broken' >&2\nexit 1\n"
SHA_EMPTY = "#!/bin/sh\nexit 0\n"
SHA_SHORT = "#!/bin/sh\necho 'abc123  x'\n"
SHA_LIES = "#!/bin/sh\necho '" + "a" * 64 + "  x'\nexit 1\n"          # 64 hex printed, but it failed


@pytest.mark.parametrize("fake", [SHA_FAILS, SHA_EMPTY, SHA_SHORT, SHA_LIES, None],
                         ids=["fails", "prints-nothing", "prints-a-short-value", "prints-64-hex-and-fails",
                              "no-new-file"])
def test_the_first_run_never_records_an_empty_or_malformed_sha256(tmp_path, fake):
    """A failing sha256sum (or a missing .new) used to write {"requirements-gpu.lock": ""} and exit 0; the
    command now checks for a 64-hex value from a sha256sum that exited 0, writes nothing and exits 2; a
    FIRST_RUN_LOCK.json left by an earlier run is deleted first, never kept."""
    script = first_run_lock_command(tmp_path)
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "FIRST_RUN_LOCK.json").write_text('{"requirements-gpu.lock": "' + "b" * 64 + '"}\n')
    env = dict(os.environ)
    if fake is not None:
        (tmp_path / "requirements-gpu.lock.new").write_text("torch==2.14.0\n")
        shim = tmp_path / "shim"
        shim.mkdir()
        (shim / "sha256sum").write_text(fake)
        (shim / "sha256sum").chmod(0o755)
        env["PATH"] = f"{shim}:{env['PATH']}"
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert run.returncode == 2 and "FIRST_RUN_LOCK.json not written" in run.stderr, (run.returncode, run.stderr)
    assert not (tmp_path / "venv" / "FIRST_RUN_LOCK.json").exists()


FAKE_UV = """#!/bin/bash
# uv for a bootstrap run without a card: `uv venv DIR` makes DIR/bin/python, `uv pip freeze` prints a lock, the rest
# does nothing
case "$1" in
  venv) mkdir -p "$2/bin" && printf '#!/bin/sh\\nexit 0\\n' > "$2/bin/python" && chmod +x "$2/bin/python" ;;
  pip) [ "$2" = freeze ] && echo "torch==2.14.0" ;;
esac
exit 0
"""

# A df with 100 GB free: the container-disk check before step 2 reads df, and one model needs 45 GB.
ROOMY_DF = """#!/bin/sh
echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'
echo 'overlay 209715200 104857600 104857600 50% /'
"""


def test_the_bootstrap_stops_with_exit_2_when_the_first_run_lock_has_no_sha256(tmp_path):
    """End to end: the real (not dry) first-run bootstrap with a sha256sum that fails stops at step 3 with exit
    2, before the model download, and writes no FIRST_RUN_LOCK.json."""
    shim = tmp_path / "shim"
    shim.mkdir()
    # ROOMY_DF keeps this test independent of the free space of the machine running it.
    (shim / "df").write_text(ROOMY_DF)
    (shim / "df").chmod(0o755)
    # The fail-fast pre-check (step 1.5) runs before step 3: a fake id (already there) and runuser (succeeds) let it
    # pass through silently, so this test still pins step 3's own sha256sum failure, not the (unrelated) fail-fast gate.
    for name, body in (("uv", FAKE_UV), ("nvidia-smi", "#!/bin/sh\necho fake-card\n"), ("sha256sum", SHA_FAILS),
                       ("id", "#!/bin/sh\nexit 0\n"), ("runuser", "#!/bin/sh\nexit 0\n")):
        (shim / name).write_text(body)
        (shim / name).chmod(0o755)
    ws = tmp_path / "ws"
    # Defense in depth, even though this run fails at step 3, before either real path is
    # touched (see run_v2test_gate_fixture and test_no_test_runs_the_real_bootstrap_on_the_machines_env_file_or_v2test_home)
    run = subprocess.run(["bash", str(lockfree_checkout(tmp_path))], capture_output=True, text=True,
                         env={**clean_env(), "PATH": f"{shim}:{os.environ['PATH']}", "WS": str(ws), "FIRST_RUN": "1",
                              **bootstrap_test_knobs(tmp_path)})
    assert run.returncode == 2 and "FIRST_RUN_LOCK.json not written" in run.stderr, (run.stdout, run.stderr)
    assert (ws / "requirements-gpu.lock.new").exists() and not (ws / "venv" / "FIRST_RUN_LOCK.json").exists()
    assert "fetch_models" not in run.stdout and "BOOTSTRAP_OK" not in run.stdout


def test_only_first_run_1_counts_and_the_lock_is_the_pin_sets_own(tmp_path):
    """FIRST_RUN=yes (or 0) is not FIRST_RUN=1, and a fallback pin set needs its own lock, the main gen lock
    does not do."""
    repo = repo_copy(tmp_path, kind="gen")
    env = {**clean_env(), **hermetic_local(tmp_path), "DRY_RUN": "1", "WS": str(tmp_path / "ws"), "KIND": "gen"}
    for value in ("yes", "0", "true"):
        out = subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                             env={**env, "FIRST_RUN": value})
        assert out.returncode == 2 and "FIRST_RUN=1" in out.stderr, value
    locked = repo_copy(tmp_path / "l", "vllm==0.27.1\n", kind="gen")
    (locked / "requirements-gen-fallback-bnb.in").write_text("vllm==0.27.1\nbitsandbytes==0.49.2\n")
    out = subprocess.run(["bash", str(locked / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                         env={**env, "GEN_SET": "fallback-bnb"})
    assert out.returncode == 2 and "requirements-gen-fallback-bnb.lock" in out.stderr


# A pod-side guard that does not depend on the tests: the suite runs as an unprivileged user
# that cannot read the pod's env file (RUNPOD_API_KEY)


def step_5(out: str) -> list[str]:
    """The commands of the bootstrap's step 5, as its dry run prints them."""
    return [l for l in out[out.index("== 5."):out.index("== 6.")].splitlines() if l.startswith("+ ")]


@pytest.mark.parametrize("kind", ["gpu", "gen"])
def test_the_test_step_runs_as_an_unprivileged_user_after_the_env_file_is_made_root_only(tmp_path, kind):
    """Before the suite, the pod's env file becomes root-only (the root watchdog still reads it, the test user
    cannot); the user v2test is made if absent (no home skeleton, no login shell) with a fresh empty HOME; the repo,
    the venv and the model cache are readable by it; both pytest runs (count and gate) switch to it with runuser and a
    clean environment (env -i: PATH, HOME, LANG, HF_HOME, HF_HUB_OFFLINE, V2GPU_REQUIRE_TORCH only); the log is
    written by root. The bootstrap also collects the same list as root and refuses on a mismatch with v2test's count
    (the
    v2test gate), and re-verifies 600:root before the switch (a stat check): before the v2test count, the first
    pytest of the step."""
    ws = tmp_path / "ws"
    # Checks the DEFAULT venv root /opt/v2 itself (default_local=True); a dry run, so nothing is written there
    lines = step_5(dry_run(tmp_path, ws=ws, default_local=True, KIND=kind).stdout)
    venv = Path("/opt/v2") / ("venv" if kind == "gpu" else "venv-gen")    # the default container-disk root
    env_file = next(i for i, l in enumerate(lines) if "chmod 600" in l and "rp_environment" in l)
    assert "chmod 600 '/etc/rp_environment' && chown root:root '/etc/rp_environment'" in lines[env_file]
    user = next(i for i, l in enumerate(lines) if "useradd" in l)
    assert "id -u v2test" in lines[user] and "--no-create-home" in lines[user] and "nologin" in lines[user]
    home = next(i for i, l in enumerate(lines) if "install -d" in l)
    assert "rm -rf '/home/v2test' && install -d -m 700 -o v2test -g v2test '/home/v2test'" in lines[home]
    access = next(i for i, l in enumerate(lines) if "chmod -R a+rX" in l)
    for path in (lockfree_checkout(tmp_path).parents[1], venv):
        assert f" {path}" in lines[access], path
    local = Path("/opt/v2")                         # HF_HOME and uv's cache beside the venvs
    assert f" {local / 'hf'}" in lines[access]
    assert f"chmod a+rx {ws} {local / 'hf'}" in lines[access - 1]
    read_only = next(l for l in lines[access + 1:] if "a+rX,a-w" in l)   # what the test user never writes, a-w
    assert all(f"'{p}'" in read_only for p in (ws / "in", local / "hf" / "hub", local / "uv-cache"))
    # Since the smoke run of 28/09 on an EU-RO-1 MooseFS: no ownership change under the workspace any more -- the old
    # `install -d -o v2test ... $REPO/.pytest_cache` is gone, and every pytest of the step runs without a cache
    assert not any(".pytest_cache" in l for l in lines)
    tests = [i for i, l in enumerate(lines) if "pytest" in l and ("--collect-only" in l or " -q -rs " in l)]
    assert len(tests) == 3 and max(env_file, user, home, access) < min(tests)
    assert all("-p no:cacheprovider" in lines[i] for i in tests), [lines[i] for i in tests]
    v2test_collect, root_collect, main_run = tests
    switch = (f"runuser -u v2test -- env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
              f"HOME=/home/v2test LANG=C.UTF-8 HF_HOME={local / 'hf'} HF_HUB_OFFLINE=1 V2GPU_REQUIRE_TORCH=1 ")
    assert switch in lines[v2test_collect] and switch in lines[main_run], [lines[i] for i in tests]
    assert "runuser" not in lines[root_collect] and "pytest --collect-only" in lines[root_collect]
    # Root's own recollection also
    # gets a clean env -i (never the pod's own environment, so never a RUNPOD_API_KEY it may carry).
    root_env = (f"env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME={ws} LANG=C.UTF-8 "
                f"HF_HOME={local / 'hf'} HF_HUB_OFFLINE=1 V2GPU_REQUIRE_TORCH=1 ")
    assert root_env in lines[root_collect], lines[root_collect]
    assert lines[main_run].index(switch) < lines[main_run].index("pytest -q -rs") < lines[main_run].index("| tee")
    assert "runuser" not in lines[main_run].split("| tee")[1] and "no_skips.sh" in lines[main_run + 3]
    assert "test_user_idle" in lines[main_run + 1]                    # idle first, then the tamper check
    assert "tamper_check" in lines[main_run + 2]                      # before root runs no_skips.sh
    # no_skips.sh reads root's own copy of the log (a root-only mktemp -d), never the one on the pod volume
    assert "/tmp/v2gpu-tamper.DRY/" in lines[main_run + 3] and "/tmp/v2gpu-tamper.DRY/" in lines[main_run]
    stat_check = next(i for i, l in enumerate(lines) if "stat -c" in l and "rp_environment" in l)
    assert max(env_file, user, home, access) < stat_check < v2test_collect, (stat_check, tests)
    assert "chmod 600 '/etc/rp_environment'" in lines[stat_check] and "600:root" in lines[stat_check]


def test_only_the_test_step_switches_user_and_it_never_reads_the_env_file():
    """Nothing else of the bootstrap runs as the test user except the fail-fast probe (before the
    long setup, not a bypass of it); the bootstrap never sources or reads the pod's env file, it only makes it
    root-only (twice: the early chmod and the stat re-check before the switch). Since 27/09 it names
    the file through RP_ENV_FILE (default /etc/rp_environment; only the tests move it, with V2GPU_BOOTSTRAP_TEST=1)."""
    code = [l for l in (OPS / "pod_bootstrap.sh").read_text().splitlines() if not l.lstrip().startswith("#")]
    switch = [l for l in code if re.search(r"\b(runuser|setpriv|sudo|su)\b", l)]
    assert len(switch) == 2, switch
    assert any(l.lstrip().startswith("TEST_ENV=(runuser -u \"$TEST_USER\" -- env -i ") for l in switch)
    assert any("runuser -u $TEST_USER -- env -i" in l and "true" in l for l in switch)   # the fail-fast probe
    uses = [l for l in code if "TEST_ENV" in l and not l.lstrip().startswith("TEST_ENV=(")]
    assert len(uses) == 3 and all("pytest" in l for l in uses), uses
    env_file = [l for l in code if "rp_environment" in l or "RP_ENV_FILE" in l]
    uses = [l for l in env_file if "chmod 600 '$RP_ENV_FILE'" in l]
    assert len(uses) == 2 and all(l.lstrip().startswith("run bash -c \"if [ -e '$RP_ENV_FILE' ]; then chmod 600 ")
                                  for l in uses), uses
    assert not any(re.search(r"\b(source|cat|head|tail|grep|awk|sed)\b|(^|[;&|]\s*)\. ", l) for l in env_file), env_file
    assert not [l for l in code if re.search(r"V2GPU_WATCHDOG|runpodctl|jobqueue", l)]


# A quick fail-fast check of the v2test switch, before the long setup


def test_the_user_switch_is_checked_before_the_long_setup(tmp_path):
    """A broken user switch on this image must be caught right after step 1 (card/driver), before step 2 (uv,
    packages, models: about 15-25 min), not only when the real test step reaches it."""
    out = dry_run(tmp_path).stdout
    steps = ["== 1.", "id -u v2test", "runuser -u v2test -- env -i", "== 2."]
    pos = [out.index(s) for s in steps]
    assert pos == sorted(pos), pos
    probe = next(l for l in out.splitlines() if l.startswith("+ bash -c ") and "runuser -u v2test -- env -i" in l
                and "true" in l)
    assert "|| {" in probe and "exit 2" in probe and "fail-fast check" in probe


def test_a_broken_user_switch_stops_the_bootstrap_at_once_with_exit_2(tmp_path):
    """End to end: a runuser that cannot switch to v2test on this image is caught right after step 1, before uv,
    packages or models ever run (about 15-25 min saved on a broken pod image). A fake uv (never the real, networked
    one) guards against ever reaching a real package resolution even if the fail-fast check is missing or misplaced."""
    repo = repo_copy(tmp_path, "torch==2.14.0\n", kind="gpu")
    shim = tmp_path / "shim"
    shim.mkdir()
    for name, body in (("id", "#!/bin/sh\nexit 1\n"), ("useradd", "#!/bin/sh\nexit 0\n"),
                       ("runuser", "#!/bin/sh\necho 'runuser: cannot set groups: Operation not permitted' >&2\n"
                                   "exit 1\n"),
                       ("nvidia-smi", "#!/bin/sh\necho fake-card\n"), ("uv", "#!/bin/sh\nexit 0\n")):
        (shim / name).write_text(body)
        (shim / name).chmod(0o755)
    ws = tmp_path / "ws"
    env = {**clean_env(), "PATH": f"{shim}:{os.environ['PATH']}", "WS": str(ws), "KIND": "gpu",
           **bootstrap_test_knobs(tmp_path)}
    run = subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True, env=env)
    assert run.returncode == 2 and "fail-fast check" in run.stderr, (run.stdout, run.stderr)
    assert "== 2." not in run.stdout and "fetch_models" not in run.stdout


# A stat re-check of /etc/rp_environment right before the v2test switch


def test_stat_600_root_is_re_checked_right_before_the_v2test_switch(tmp_path):
    """Closes the window where /start.sh had not yet (re)written the pod's env file when the early chmod ran: right
    before the switch to v2test, the bootstrap re-applies chmod 600 root:root and refuses (exit 2) if a stat of the
    file right after does not read 600:root. "before the switch" is before the FIRST
    runuser of step 5 that runs repo code, the v2test count; it used to run after it, so conftest and every test
    module's top level ran as v2test before the file was verified."""
    out = dry_run(tmp_path).stdout
    lines = step_5(out)
    stat_check = next(i for i, l in enumerate(lines) if "stat -c" in l and "rp_environment" in l)
    switches = [i for i, l in enumerate(lines) if "runuser" in l]
    pytests = [i for i, l in enumerate(lines) if "pytest --collect-only" in l or " -m pytest -q -rs " in l]
    assert len(switches) == 2 and len(pytests) == 3 and stat_check < min(switches + pytests), (stat_check, lines)
    assert "chmod 600 '/etc/rp_environment'" in lines[stat_check] and "600:root" in lines[stat_check]
    assert "exit 2" in lines[stat_check]


# The v2test gate: a permission gap must not silently shrink the collected count


# Root's own recollection now ALSO gets a minimal env -i (same PATH as v2test's, so PATH can no
# longer tell them apart); HOME still differs (v2test's is V2TEST_HOME, root's is $WS), so the fake python keys off
# HOME instead. run_v2test_gate_fixture always names the v2test HOME "v2test-home" under its own tmp_path.
def fake_py_mismatch(v2test_home: str) -> str:
    return f"""#!/bin/sh
V2TEST_HOME="{v2test_home}"
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then
  if [ "$HOME" = "$V2TEST_HOME" ]; then echo "3/3 tests collected"; else echo "5/5 tests collected"; fi
  exit 0
fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then
  if [ "$HOME" = "$V2TEST_HOME" ]; then echo "3 passed, 1 deselected in 0.01s"
  else echo "5 passed, 1 deselected in 0.01s"; fi
  exit 0
fi
exit 0
"""

FAKE_PY_AGREE = """#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then echo "5/5 tests collected"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then echo "5 passed, 1 deselected in 0.01s"; exit 0; fi
exit 0
"""

FAKE_RUNUSER_DROP_PRIVILEGE = """#!/bin/sh
# a test shim for `runuser -u USER -- CMD...`: drop the privilege switch, keep the env -i wrapper it protects
shift 3
exec "$@"
"""

FAKE_INSTALL_NOOP = "#!/bin/sh\nexit 0\n"


def bootstrap_test_knobs(tmp_path) -> dict:
    """A real (not dry) bootstrap run by a test takes the env file and the v2test HOME from
    tmp_path, never the machine's own (on a pod: the pod's env file, chmod/chown'd as the test user; the pod's
    /home/v2test, deleted). The env file is not created here: a test that needs one writes it."""
    return {"RP_ENV_FILE": str(tmp_path / "rp_environment"), "V2TEST_HOME": str(tmp_path / "v2test-home"),
            "V2GPU_LOCAL": str(tmp_path / "local"), "V2GPU_BOOTSTRAP_TEST": "1"}


def run_v2test_gate_fixture(tmp_path, fake_py: str, shims: tuple = (), runpod_api_key: str = "",
                            extra_env: dict | None = None, with_venv: bool = True, left_read_only: bool = False,
                            prepare=None, lock: str | None = "torch==2.14.0\n"):
    repo = repo_copy(tmp_path, lock, kind="gpu")
    ws = tmp_path / "ws"
    venv = tmp_path / "local" / "venv"            # the venv lives on the container's own disk, never under $WS
    (venv / "bin").mkdir(parents=True)
    local = tmp_path / "local"          # the HF cache and uv's cache live there too, not under $WS
    (local / "hf" / "hub").mkdir(parents=True)
    (local / "hf" / "hub" / "blob").write_text("w")
    (ws / "in").mkdir(parents=True)
    (ws / "in" / "v2gpu-x.bundle").write_text("b")
    (local / "uv-cache").mkdir()
    if left_read_only:                  # as an earlier bootstrap's step 5 (a-w) leaves them, before a restart
        for d in (ws / "in", local / "hf" / "hub", local / "uv-cache"):
            for q in sorted([d, *d.rglob("*")], reverse=True):
                q.chmod(q.stat().st_mode & ~0o222)
    if with_venv:
        (venv / "bin" / "python").write_text(fake_py)
        (venv / "bin" / "python").chmod(0o755)
    shim = tmp_path / "shim"
    shim.mkdir()
    for name, body in (("uv", "#!/bin/sh\nexit 0\n"), ("nvidia-smi", "#!/bin/sh\necho fake-card\n"),
                       ("id", "#!/bin/sh\nexit 1\n"), ("useradd", "#!/bin/sh\nexit 0\n"),
                       ("install", FAKE_INSTALL_NOOP), ("runuser", FAKE_RUNUSER_DROP_PRIVILEGE), *shims):
        (shim / name).write_text(body)
        (shim / name).chmod(0o755)
    if prepare is not None:                               # a test's own change to the layout, before the run
        prepare()
    home = tmp_path / "v2test-home"                        # what an earlier run left in the (test's own) v2test HOME
    home.mkdir()
    (home / "left-by-an-earlier-run").write_text("x")
    # Since 27/09/2026 runpod_api_key defaults to "" (harmless) so every existing caller is unaffected;
    # it stays a literal, non-spread key so test_pod_safety.py's static scan can still read this dict as a whole.
    # extra_env (the MooseFS shim's own two variables) comes before the key and the knobs, so neither can be
    # overridden by it, and the dict stays one literal test_pod_safety.py's static scan can read
    env = {**clean_env(), "PATH": f"{shim}:{os.environ['PATH']}", "WS": str(ws), "KIND": "gpu",
           **(extra_env or {}), "RUNPOD_API_KEY": runpod_api_key, **bootstrap_test_knobs(tmp_path)}
    return subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True, env=env)


def test_the_hf_and_uv_caches_live_on_the_container_disk(tmp_path):
    """First generation pod, 04/10: uv extracting wheels into $WS/uv-cache on EUR-IS-1's MooseFS took 2.4 h of a
    2.7 h bootstrap (1.1 MB/s, the pod's own downloads ran at 40-45 MB/s). HF_HOME and UV_CACHE_DIR are now
    $V2GPU_LOCAL/hf and $V2GPU_LOCAL/uv-cache, on the container disk like the venvs; step 5 still makes hf/hub and
    uv-cache read-only for everyone (a-w) and the test user's environment names the new HF_HOME."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    local, ws = tmp_path / "local", tmp_path / "ws"
    assert f"HF_HOME={local}/hf " in run.stdout and f"HF_HOME={ws}/hf" not in run.stdout
    assert not (ws / "hf").exists() and not (ws / "uv-cache").exists()
    for d in (local / "hf" / "hub", local / "uv-cache"):
        assert all(not q.stat().st_mode & 0o222 for q in [d, *d.rglob("*")]), d


def test_v2test_gate_refuses_when_v2tests_collected_count_differs_from_roots(tmp_path):
    """The v2test gate: find_spec('torch') can lie under a v2test permission gap (e.g. 3 of 5 tests hidden), and
    without a cross-check the bootstrap would trust v2test's own smaller count and gate on it alone: a false green.
    A plain root recollection runs too, and the bootstrap refuses (exit 2) on any difference from v2test's count."""
    run = run_v2test_gate_fixture(tmp_path, fake_py_mismatch(str(tmp_path / "v2test-home")))
    assert run.returncode == 2 and "collected" in run.stderr and "v2test" in run.stderr, (run.stdout, run.stderr)
    assert "== 6." not in run.stdout and "BOOTSTRAP_OK" not in run.stdout


def test_v2test_gate_passes_when_the_counts_agree(tmp_path):
    """The other side of the same gate: a v2test whose count matches root's own is never refused by it, and the
    bootstrap completes normally."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout, run.stderr)


# In the smoke run of 28/09 (EU-RO-1), the pod's /workspace was a MooseFS FUSE mount
# (a RunPod MooseFS volume), which allows chmod but refuses every ownership change, so step 5's
# `install -d -o v2test -g v2test /workspace/repo/.pytest_cache` failed and the bootstrap stopped before its suite.
# ops/moosefs_shim.sh emulates that volume: installed as chown and install ahead of the real ones, it refuses any
# ownership change of a path under MOOSEFS_ROOTS and hands everything else to MOOSEFS_REAL_DIR/<name>.

MOOSEFS_SHIM = (OPS / "moosefs_shim.sh")


def moosefs_bin(tmp_path, name="shim"):
    """A directory holding the shim as chown and install, plus a 'real' directory of recording no-op binaries."""
    real = tmp_path / f"{name}-real"
    real.mkdir()
    for b in ("chown", "install"):
        (real / b).write_text(f"#!/bin/sh\necho \"REAL {b} $*\" >> '{tmp_path}/real-calls.log'\nexit 0\n")
        (real / b).chmod(0o755)
    return real


def run_shim(tmp_path, name, *args, roots=""):
    real = moosefs_bin(tmp_path)
    d = tmp_path / "shimbin"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(MOOSEFS_SHIM.read_bytes())
    (d / name).chmod(0o755)
    env = {**clean_env(), "MOOSEFS_ROOTS": roots, "MOOSEFS_REAL_DIR": str(real)}
    return subprocess.run([str(d / name), *args], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("name, args, refused", [
    ("chown", ["65534:65534", "/ws/x"], True),
    ("chown", ["root:root", "/etc/hostname"], False),
    ("chown", ["v2test", "/wsx/y"], False),                        # a sibling prefix is not under /ws
    ("install", ["-d", "-o", "v2test", "-g", "v2test", "/repo/.pytest_cache"], True),
    ("install", ["-d", "--owner=v2test", "/ws/a"], True),
    ("install", ["-d", "-m", "700", "/ws/a"], False),               # mkdir and chmod only: MooseFS allows these
    ("install", ["-d", "-m", "700", "-o", "v2test", "-g", "v2test", "/home/v2test"], False),
])
def test_the_moosefs_shim_refuses_only_ownership_changes_under_its_roots(tmp_path, name, args, refused):
    run = run_shim(tmp_path, name, *args, roots="/ws /repo")
    log = tmp_path / "real-calls.log"
    if refused:
        assert run.returncode == 1 and "Operation not permitted" in run.stderr and "MooseFS" in run.stderr
        assert not log.exists()
    else:
        assert run.returncode == 0, run.stderr
        assert log.read_text() == f"REAL {name} {' '.join(args)}\n"


def test_the_bootstrap_passes_on_a_workspace_that_refuses_ownership_changes(tmp_path):
    """The real (not dry) bootstrap, with the MooseFS shim in front of chown and install for the workspace AND the
    repo (on a pod the repo is under /workspace): it must reach BOOTSTRAP_OK. Earlier, step 5's
    `install -d -o v2test ... $REPO/.pytest_cache` was refused here, exactly as on the EU-RO-1 pod."""
    real = moosefs_bin(tmp_path, "mfs")
    shim = MOOSEFS_SHIM.read_text()
    run = run_v2test_gate_fixture(
        tmp_path, FAKE_PY_AGREE, shims=(("chown", shim), ("install", shim)),
        extra_env={"MOOSEFS_ROOTS": f"{tmp_path / 'ws'} {tmp_path / 'repo'}", "MOOSEFS_REAL_DIR": str(real)})
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-2000:], run.stderr[-2000:])


def test_every_bootstrap_records_its_workspace_volume_first(tmp_path):
    """decision 28/09: every pod's bootstrap records what $WS is (df -T, its mount line) and whether ownership changes
    and flock work there, in $WS/volume_receipt.txt and in the bootstrap log, as step 0 -- before anything that could
    fail on such a volume, so even a failed bootstrap leaves the record."""
    out = dry_run(tmp_path).stdout
    zero = out.index("== 0.")
    assert zero < out.index("== 1.") and "volume" in out[zero:out.index("\n", zero)]
    real = moosefs_bin(tmp_path, "mfs")
    shim = MOOSEFS_SHIM.read_text()
    run = run_v2test_gate_fixture(
        tmp_path, FAKE_PY_AGREE, shims=(("chown", shim), ("install", shim)),
        extra_env={"MOOSEFS_ROOTS": f"{tmp_path / 'ws'} {tmp_path / 'repo'}", "MOOSEFS_REAL_DIR": str(real)})
    receipt = (tmp_path / "ws" / "volume_receipt.txt").read_text()
    for needle in ("== df -T", "== mount", "chown_probe: refused", "flock_probe: ", "flock_dir_probe: ",
                   "chmod_probe: 600 kept"):     # does the volume keep mode bits (the laptop's does)
        assert needle in receipt, (needle, receipt)
    assert "chown_probe: refused" in run.stdout                         # also in the bootstrap's own log
    assert not (tmp_path / "ws" / ".v2gpu-volume-probe").exists()        # the probe file is removed
    assert not (tmp_path / "ws" / ".v2gpu-volume-probe.d").exists()      # and the probe directory


def test_the_container_gate_always_runs_the_bootstrap_on_an_emulated_moosefs_workspace():
    """decision 28/09: the pod-parity gate must cover the MooseFS case on every run. Its container installs
    ops/moosefs_shim.sh as chown and install (the real ones kept as MOOSEFS_REAL_DIR) BEFORE the bootstrap, with the
    workspace and the repo as MOOSEFS_ROOTS; after the bootstrap it checks the volume receipt says the ownership probe
    was refused (a positive control: the emulation was really active), and its second pass runs without pytest's cache
    like step 5 does."""
    gate = (OPS / "pod_parity_gate.sh").read_text()
    inner = gate[gate.index("INNER=$(cat <<INNER"):gate.index("\nINNER\n")]
    shim = inner.index("ops/moosefs_shim.sh")
    boot = inner.index("bash ops/pod_bootstrap.sh")
    assert shim < boot
    for needle in ("for b in chown install; do cp ops/moosefs_shim.sh /usr/local/bin/\\$b;",
                   'MOOSEFS_ROOTS="\\$WS"', "MOOSEFS_REAL_DIR=/usr/bin"):   # repo and venv are local now
        assert needle in inner[:boot], needle
    after = inner[boot:]
    assert "chown_probe: refused" in after and "MOOSEFS_EMULATION_INACTIVE" in after
    second = next(l for l in after.splitlines() if "runuser -u v2test -- env -i" in l and "pytest" in l)
    assert "-m pytest -q -rs " in second and " -p no:cacheprovider " in second


# The pod volume on MooseFS (decision 28/09)


@pytest.mark.parametrize("args, refused", [
    (["-d", "-ov2test", "/ws/a"], True),                    # attached values, pinned
    (["-d", "-gv2test", "/ws/a"], True),
    (["-dov2test", "/ws/a"], True),                         # a short-option cluster that includes -o
    (["-d", "--group=v2test", "/ws/a"], True),
    (["-Dm644", "src", "/ws/b"], False),                    # no ownership flag in the cluster: MooseFS allows it
])
def test_the_moosefs_shim_catches_every_owner_form_install_accepts(tmp_path, args, refused):
    real = moosefs_bin(tmp_path)
    d = tmp_path / "shimbin"
    d.mkdir()
    (d / "install").write_bytes(MOOSEFS_SHIM.read_bytes())
    (d / "install").chmod(0o755)
    ws = tmp_path / "ws"
    ws.mkdir()
    args = [a.replace("/ws", str(ws)) for a in args]
    run = subprocess.run([str(d / "install"), *args], capture_output=True, text=True, cwd=ws,
                         env={**clean_env(), "MOOSEFS_ROOTS": str(ws), "MOOSEFS_REAL_DIR": str(real)})
    assert (run.returncode == 1 and "Operation not permitted" in run.stderr) if refused else run.returncode == 0, \
        (args, run.returncode, run.stderr)


def test_the_moosefs_shim_falls_back_to_usr_bin_without_its_real_dir(tmp_path):
    """Inside the gate, the bootstrap's own env -i test runs have the shim first on PATH but none of the
    MOOSEFS_* variables; a test there calling install must reach the real one (/usr/bin), never fail on the shim."""
    d = tmp_path / "shimbin"
    d.mkdir()
    (d / "install").write_bytes(MOOSEFS_SHIM.read_bytes())
    (d / "install").chmod(0o755)
    target = tmp_path / "made-by-real-install"
    run = subprocess.run([str(d / "install"), "-d", str(target)], capture_output=True, text=True,
                         env={k: v for k, v in clean_env().items() if not k.startswith("MOOSEFS_")})
    assert run.returncode == 0 and target.is_dir(), run.stderr


def test_step_5_writes_no_bytecode_and_leaves_nothing_group_or_world_writable(tmp_path):
    """On the EU-RO-1 smoke pod everything under /workspace was world-writable (777), so v2test's step-5
    pytest could write __pycache__ into, or change, the repo and the venv that root runs later. Step 5 now removes
    group/other write from the repo, the venv and the HF hub cache (chmod -R a+rX,go-w) before the switch, and every
    pytest of the step runs with PYTHONDONTWRITEBYTECODE=1 (v2test's and root's; the gate's second pass too)."""
    lines = step_5(dry_run(tmp_path).stdout)
    access = next(l for l in lines if "chmod -R" in l)
    assert "a+rX,go-w" in access
    tests = [l for l in lines if "pytest" in l and ("--collect-only" in l or " -q -rs " in l)]
    assert len(tests) == 3 and all("PYTHONDONTWRITEBYTECODE=1" in l for l in tests), tests
    gate = (OPS / "pod_parity_gate.sh").read_text()
    second = next(l for l in gate.splitlines() if "runuser -u v2test -- env -i" in l and "pytest" in l)
    assert "PYTHONDONTWRITEBYTECODE=1" in second


# Measured on the real EU-RO-1 MooseFS (smoke 28/09): a file made 666 stays 666 after chmod 600, and chmod 755
# gives 777 -- mode bits are not kept, so chmod go-w cannot stop v2test there. So the bootstrap detects instead of
# preventing. Root records a manifest of the repo, the venv and the HF hub cache before the test user
# runs anything, and compares it after EACH of that user's runs (before root's own recollection, and before step 6):
# any added, removed or changed entry stops the bootstrap with BOOTSTRAP_TAMPERED.


def fake_py_tamper(v2test_home: str, target: str, when: str) -> str:
    """A fake python that behaves (counts agree) but, as the test user (HOME = its v2test home), changes `target`
    during its collect run (when="collect") or its main run (when="run")."""
    return f"""#!/bin/sh
V2TEST_HOME="{v2test_home}"
T="{target}"
# On the laptop the fake test user OWNS the fixture's files, so it can undo the bootstrap's a-w first (a real
# v2test on a pod cannot: it owns nothing there); what the manifest check must then still catch is the change itself
# (only the target's own directory and the target, never -R: a recursive chmod of $WS would itself change the
# manifest elsewhere, and the test would pass for the wrong reason)
tamper() {{ chmod u+w "$(dirname "$T")" 2> /dev/null; [ -e "$T" ] && chmod u+w "$T"; echo "# injected" >> "$T"; }}
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then
  if [ "$HOME" = "$V2TEST_HOME" ] && [ "{when}" = collect ]; then tamper; fi
  echo "5/5 tests collected"; exit 0
fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then
  if [ "$HOME" = "$V2TEST_HOME" ] && [ "{when}" = run ]; then tamper; fi
  echo "5 passed, 1 deselected in 0.01s"; exit 0
fi
exit 0
"""


@pytest.mark.parametrize("where, when", [("repo", "collect"), ("repo", "run"), ("venv", "collect"),
                                         ("venv", "run"), ("hub", "run"), ("hub", "collect"), ("hf", "run"),
                                         ("uv-cache", "run"), ("uv-cache", "collect")])
def test_a_test_user_change_to_the_repo_venv_or_cache_stops_the_bootstrap(tmp_path, where, when):
    """Since 05/10 the HF cache and uv's cache moved to the container disk ($V2GPU_LOCAL/hf, /uv-cache), off the
    manifest's $WS root; the manifest covers both there, every entry of HF_HOME (not only hub/) included."""
    targets = {"repo": tmp_path / "repo" / "ops" / "no_skips.sh",
               "venv": tmp_path / "local" / "venv" / "lib-planted.py",   # a NEW file counts too
               "hub": tmp_path / "local" / "hf" / "hub" / "planted",
               "hf": tmp_path / "local" / "hf" / "planted-beside-hub",
               "uv-cache": tmp_path / "local" / "uv-cache" / "planted"}
    run = run_v2test_gate_fixture(tmp_path, fake_py_tamper(str(tmp_path / "v2test-home"), str(targets[where]), when))
    assert run.returncode == 2 and "BOOTSTRAP_TAMPERED" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert "BOOTSTRAP_OK" not in run.stdout and "== 6." not in run.stdout
    if when == "collect":        # caught before root runs the repo's code for its own recollection
        assert run.stdout.count("pytest --collect-only") <= 1 or "tamper check" in run.stderr


def fake_py_swaps_in(v2test_home: str, ws: str) -> str:
    """A fake python whose test user, during its main run, renames the pod volume's in/ aside and puts its own in place
    (MooseFS gives /workspace 777, so a rename there works whatever in/'s own mode)."""
    return f"""#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then echo "5/5 tests collected"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then
  if [ "$HOME" = "{v2test_home}" ]; then mv "{ws}/in" "{ws}/in.old" && mkdir "{ws}/in" && echo x > "{ws}/in/v2gpu-x.bundle"; fi
  echo "5 passed, 1 deselected in 0.01s"; exit 0
fi
exit 0
"""


@pytest.mark.parametrize("where", ["in", "state", "queue", "top"])          # uv-cache left $WS
@pytest.mark.parametrize("when", ["collect", "run"])
def test_a_test_user_change_anywhere_on_the_pod_volume_stops_the_bootstrap(tmp_path, where, when):
    """decision 28/09: the manifest covers the WHOLE $WS: in/ and uv-cache are what a
    re-bootstrap after a restart trusts (the bundle, the venv rebuild), and runs/<job>/state, the queue files and the
    GEN inputs are on the volume too; a change to any of them, or a new file anywhere on it, stops the bootstrap."""
    ws = tmp_path / "ws"
    (ws / "runs" / "deepseek-s42-all" / "state").mkdir(parents=True)
    (ws / "runs" / "deepseek-s42-all" / "state" / "trainer_state.json").write_text("{}")
    (ws / "queue-train.json").write_text("{}")
    targets = {"in": ws / "in" / "v2gpu-x.bundle",
               "state": ws / "runs" / "deepseek-s42-all" / "state" / "trainer_state.json",
               "queue": ws / "queue-train.json", "top": ws / "planted-by-the-test-user"}
    run = run_v2test_gate_fixture(tmp_path, fake_py_tamper(str(tmp_path / "v2test-home"), str(targets[where]), when))
    assert run.returncode == 2 and "BOOTSTRAP_TAMPERED" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert "BOOTSTRAP_OK" not in run.stdout and "== 6." not in run.stdout


def test_a_swapped_in_directory_stops_the_bootstrap(tmp_path):
    run = run_v2test_gate_fixture(tmp_path, fake_py_swaps_in(str(tmp_path / "v2test-home"), str(tmp_path / "ws")))
    assert run.returncode == 2 and "BOOTSTRAP_TAMPERED" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert "in.old" in run.stderr


def fake_py_roots_own_writes(ws: str) -> str:
    """Counts agree; root's own writes on the pod volume during step 5 -- its recollection (HOME = $WS) writing a cache,
    the launch logs and the watchdog log growing while the test user's main run goes on -- are not the test user's.
    The paths are written in (the bootstrap runs this python under env -i)."""
    return f"""#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then
  if [ "$HOME" = "{ws}" ]; then mkdir -p "{ws}/.cache" && echo x > "{ws}/.cache/root-recollection"; fi
  echo "5/5 tests collected"; exit 0
fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then
  echo launch >> "{ws}/bootstrap.log"; echo launch >> "{ws}/bootstrap.gen.log"; echo armed >> "{ws}/watchdog.log"
  echo "5 passed, 1 deselected in 0.01s"; exit 0
fi
exit 0
"""


def test_roots_own_writes_on_the_pod_volume_do_not_stop_the_bootstrap(tmp_path):
    """The manifest skips by exact name only what root itself writes under $WS while the test user runs (the pytest
    log, the launch logs, the watchdog log, and $WS's own entry that their creation touches), and is taken again after
    root's own recollection (HOME=$WS), so neither is ever blamed on the test user."""
    run = run_v2test_gate_fixture(tmp_path, fake_py_roots_own_writes(str(tmp_path / "ws")))
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    assert (tmp_path / "ws" / ".cache" / "root-recollection").exists() and "tamper check: unchanged" in run.stdout


def test_an_untouched_tree_passes_the_tamper_check(tmp_path):
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    assert run.stdout.count("tamper check: unchanged") == 2


def test_the_manifest_tool_is_copied_out_of_the_test_users_reach_first():
    """The checker must not be something the test user can change: root copies ops/tree_manifest.py into a fresh
    mktemp -d directory before the switch and runs that copy for every check."""
    text = (OPS / "pod_bootstrap.sh").read_text()
    step5 = text[text.index('echo "== 5.'):text.index('echo "== 6.')]
    assert "mktemp -d" in step5 and "tree_manifest.py" in step5
    first_switch = step5.index('"${TEST_ENV[@]}"')
    assert step5.index("mktemp -d") < first_switch and step5.index("tamper_check") < step5.index("ROOT_N=")


def test_tree_manifest_sees_added_removed_and_changed_entries(tmp_path):
    tool = OPS / "tree_manifest.py"
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "a.py").write_text("x")
    (d / "sub" / "b").write_text("y")
    before = subprocess.run([sys.executable, str(tool), str(d)], capture_output=True, text=True, check=True).stdout
    again = subprocess.run([sys.executable, str(tool), str(d)], capture_output=True, text=True, check=True).stdout
    assert before == again and "a.py" in before and "sub/b" in before
    for change in (lambda: (d / "a.py").write_text("x2"), lambda: (d / "new").write_text(""),
                   lambda: (d / "sub" / "b").unlink(), lambda: (d / "a.py").chmod(0o600)):
        change()
        after = subprocess.run([sys.executable, str(tool), str(d)], capture_output=True, text=True, check=True).stdout
        assert after != before
        before = after


# decision 28/09, after the EU-RO-1 smoke pod: the repo and the venvs live on the container's
# own disk (V2GPU_LOCAL, default /opt/v2), /workspace keeps only what must survive a stop or be copied home


def test_a_dry_run_puts_every_venv_under_opt_v2(tmp_path):
    for kind, venv in (("gpu", "/opt/v2/venv"), ("gen", "/opt/v2/venv-gen")):
        out = dry_run(tmp_path, default_local=True, KIND=kind).stdout     # the default itself, dry
        assert f"{venv}/bin/python" in out, (kind, out[-1500:])
        assert f"{tmp_path}/venv" not in out and f"{tmp_path}/venv-gen" not in out


def test_layout_on_workspace_is_refused(tmp_path):
    knobs = bootstrap_test_knobs(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {**clean_env(), "WS": str(ws), "KIND": "gpu", "DRY_RUN": "0", "FIRST_RUN": "1",
           "RP_ENV_FILE": knobs["RP_ENV_FILE"], "V2TEST_HOME": knobs["V2TEST_HOME"],
           "V2GPU_LOCAL": str(ws / "local"), "V2GPU_BOOTSTRAP_TEST": "1"}
    run = subprocess.run(["bash", str(lockfree_checkout(tmp_path))], capture_output=True, text=True, env=env)
    assert run.returncode == 2 and "LAYOUT_ON_WORKSPACE" in run.stderr, (run.stdout[-800:], run.stderr[-800:])
    assert "== 1." not in run.stdout


def test_a_fixture_run_builds_nothing_under_the_workspace_and_says_where_things_are(tmp_path):
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    layout = next(l for l in run.stdout.splitlines() if "layout: repo " in l)
    assert str((tmp_path / "repo").resolve()) in layout and str((tmp_path / "local").resolve()) in layout
    assert not (tmp_path / "ws" / "venv").exists() and not (tmp_path / "ws" / "venv-gen").exists()


def test_nothing_under_the_workspace_that_the_test_user_does_not_write_stays_writable(tmp_path):
    """decision 28/09: a-w on everything under /workspace the test user never writes (in/, hf/hub, uv-cache/) -- on the
    EU-RO-1 MooseFS a 444 file refused v2test's write while a "644" one (stored as 666) took it; root is not held back
    by mode bits."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0, run.stderr[-1500:]
    ws = tmp_path / "ws"
    local = tmp_path / "local"          # hf/hub and uv-cache on the container disk, still a-w
    for p in (ws / "in", ws / "in" / "v2gpu-x.bundle", local / "hf" / "hub", local / "hf" / "hub" / "blob",
              local / "uv-cache"):
        assert p.stat().st_mode & 0o222 == 0, (p, oct(p.stat().st_mode))
    for d in (ws / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):   # pytest cleans tmp_path
        for q in [d, *d.rglob("*")]:
            q.chmod(q.stat().st_mode | 0o200)


def test_a_low_local_disk_stops_the_bootstrap_before_building_a_venv(tmp_path):
    """decision 28/09: venv + venv-gen are about 16 GB on a 50 GB container disk; below 20 GB free the bootstrap stops
    with a clear message before step 2 builds anything (a pod whose venv already exists is not re-checked). Since
    05/10, uv's cache and the weights are there too: 30 GB plus 15 GB per model (one model when none is named)."""
    low_df = ("#!/bin/sh\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
              "echo 'overlay 52428800 49000000 3428800 94% /'\n")
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, shims=(("df", low_df),), with_venv=False)
    assert run.returncode == 2 and "LOCAL_DISK_LOW" in run.stderr and "45 GB" in run.stderr, run.stderr[-800:]
    assert "== 3." not in run.stdout


def test_every_bootstrap_records_how_long_each_step_took(tmp_path):
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0, run.stderr[-1500:]
    timings = (tmp_path / "ws" / "bootstrap_timings.txt").read_text().split("\n")
    steps = [l.split()[0] for l in timings if l.strip()]
    assert steps[0] == "start" and steps[-1] == "end" and {"0", "1", "1.5", "2", "3", "4", "5", "6"} <= set(steps)
    assert all(l.split()[1].isdigit() for l in timings if l.strip())
    assert "bootstrap total:" in run.stdout and run.stdout.index("bootstrap total:") < run.stdout.index("BOOTSTRAP_OK")


def test_the_container_gate_runs_the_new_layout_and_proves_it():
    """decision 28/09: the gate runs the new layout -- the venvs on the container's own disk (/opt/v2), the repo at
    /work, the workspace MooseFS-emulated -- and proves it: the venv really is off the workspace, and the test user
    cannot write in/, hf/hub or uv-cache/ there; the second pass runs the /opt/v2 venv."""
    gate = (OPS / "pod_parity_gate.sh").read_text()
    inner = gate[gate.index("INNER=$(cat <<INNER"):gate.index("\nINNER\n")]
    boot = inner.index("bash ops/pod_bootstrap.sh")
    before, after = inner[:boot], inner[boot:]
    assert 'python3 -m venv --system-site-packages /opt/v2/venv' in before
    assert '/opt/v2/venv-gen' in before and 'venv --system-site-packages "\\$WS' not in inner
    assert 'mkdir -p "\\$WS/in" /opt/v2/uv-cache' in before and "cp -R /hfcache /opt/v2/hf/hub" in before   # (d)
    assert "LAYOUT_NOT_PROVEN" in after and "WORKSPACE_WRITABLE_BY_TEST_USER" in after
    for d in ("\\$WS/in/", "/opt/v2/hf/hub/", "/opt/v2/uv-cache/"):
        assert d in after, d
    second = next(l for l in after.splitlines() if "runuser -u v2test -- env -i" in l and "pytest" in l)
    assert "/opt/v2/$VENV2/bin/python" in second


FAKE_UV_BUILDS_THE_VENV = """#!/bin/sh
# a test uv: records every call; like the real one, every call first writes into uv's cache (so a cache left read-only
# stops it), and `uv venv DIR ...` makes DIR/bin/python (a copy of the fake python named by FAKE_PY_SRC)
echo "uv $* UV_CACHE_DIR=$UV_CACHE_DIR" >> "$UV_LOG"
: > "$UV_CACHE_DIR/.written-by-uv" || exit 2
if [ "$1" = venv ]; then mkdir -p "$2/bin" && cp "$FAKE_PY_SRC" "$2/bin/python" && chmod 755 "$2/bin/python"; fi
exit 0
"""

PLENTY_DF = ("#!/bin/sh\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
             "echo 'overlay 104857600 10485760 94371840 10% /'\n")


def test_resume_after_the_container_disk_was_wiped_rebuilds_the_venv_from_the_lock(tmp_path):
    """decision 28/09: a pod (re)start wipes the container disk (/opt/v2: the repo and the venvs), never the pod volume.
    The operator then re-runs §5's clone and bootstrap (VENV_GONE), and the bootstrap must rebuild the venv from the
    COMMITTED lock (uv pip sync; no FIRST_RUN, no .lock.new) with the pod volume's own uv cache -- which the previous
    run's step 5 left read-only (a-w), so the user write bit must come back before uv writes to it -- and leave the
    training state on the volume untouched; step 5 makes the cache read-only again."""
    ws = tmp_path / "ws"
    state = ws / "runs" / "deepseek-s42-all" / "state"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text('{"global_step": 1200}')
    (tmp_path / "fake-python").write_text(FAKE_PY_AGREE)
    log = tmp_path / "uv.log"
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, with_venv=False, left_read_only=True,
                                  shims=(("uv", FAKE_UV_BUILDS_THE_VENV), ("df", PLENTY_DF)),
                                  extra_env={"UV_LOG": str(log), "FAKE_PY_SRC": str(tmp_path / "fake-python")})
    try:
        assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
        calls = log.read_text().splitlines()
        venv = tmp_path / "local" / "venv"
        assert calls[0] == f"uv venv {venv} --python /usr/bin/python3 UV_CACHE_DIR={tmp_path / 'local' / 'uv-cache'}", calls
        sync = next(c for c in calls if c.startswith("uv pip "))
        assert sync.startswith(f"uv pip sync --python {venv}/bin/python ") and "requirements-gpu.lock " in sync
        assert not any("pip install" in c or "pip freeze" in c for c in calls), calls
        assert "FIRST RUN" not in run.stdout and not (ws / "requirements-gpu.lock.new").exists()
        assert (venv / "bin" / "python").exists() and not (ws / "venv").exists()
        assert (state / "trainer_state.json").read_text() == '{"global_step": 1200}'
        assert (tmp_path / "local" / "uv-cache").stat().st_mode & 0o222 == 0   # step 5 took the write bit away again
    finally:
        for d in (ws / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):   # pytest cleans tmp_path
            for q in [d, *d.rglob("*")]:
                q.chmod(q.stat().st_mode | 0o200)


FAKE_RUNUSER_RECORDS_THE_VOLUME = """#!/bin/sh
# a test shim for `runuser -u USER -- CMD...`: records, at every switch to the test user, which of in/, hf/hub and
# uv-cache it could write, then drops the privilege switch like FAKE_RUNUSER_DROP_PRIVILEGE
w=""
for d in "$WS/in" "$V2GPU_LOCAL/hf/hub" "$V2GPU_LOCAL/uv-cache"; do if [ -w "$d" ]; then w="$w $d"; fi; done
echo "runuser $4${w:+ WRITABLE:$w}" >> "$RUNUSER_LOG"
shift 3
exec "$@"
"""


def test_the_test_user_runs_nothing_while_the_volume_is_writable(tmp_path):
    """decision 28/09, the condition on step 2's user-write restore: the test user must run nothing between step 2
    giving the user write bit back (in/, hf/hub, uv-cache) and step 5 taking it away again. Run for real on the
    resume path (the directories left read-only by an earlier run): at every switch to the test user -- step 1.5's
    fail-fast, the collection, the main run -- none of the three can be written."""
    (tmp_path / "fake-python").write_text(FAKE_PY_AGREE)
    ulog, rlog = tmp_path / "uv.log", tmp_path / "runuser.log"
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, with_venv=False, left_read_only=True,
                                  shims=(("uv", FAKE_UV_BUILDS_THE_VENV), ("df", PLENTY_DF),
                                         ("runuser", FAKE_RUNUSER_RECORDS_THE_VOLUME)),
                                  extra_env={"UV_LOG": str(ulog), "FAKE_PY_SRC": str(tmp_path / "fake-python"),
                                             "RUNUSER_LOG": str(rlog)})
    try:
        assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
        switches = rlog.read_text().splitlines()
        assert len(switches) >= 3, switches                     # 1.5's fail-fast, the collection, the main run
        assert not [l for l in switches if "WRITABLE" in l], switches
        assert ulog.read_text().strip()                          # uv did write its cache in between (step 2 ran)
    finally:
        for d in (tmp_path / "ws" / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):
            for q in [d, *d.rglob("*")]:
                q.chmod(q.stat().st_mode | 0o200)


def test_no_user_switch_is_written_between_the_owner_write_restore_and_the_relock():
    """The same invariant, pinned in the script's text: from step 2's `chmod -R u+w` to step 5's `a+rX,a-w`, no
    command switches user (runuser, su, sudo, setpriv); making the test user and its home (as root) is allowed."""
    src = (OPS / "pod_bootstrap.sh").read_text().splitlines()
    start = next(i for i, l in enumerate(src) if "chmod -R u+w" in l and not l.lstrip().startswith("#"))
    end = next(i for i, l in enumerate(src) if "a+rX,a-w" in l and not l.lstrip().startswith("#"))
    assert start < end
    code = [l for l in src[start:end] if not l.lstrip().startswith("#")]
    assert not [l for l in code if re.search(r"\b(runuser|su|sudo|setpriv)\b", l)], code
    step2 = max(i for i, l in enumerate(src[:start]) if '"== 2.' in l)       # the restore is inside step 2
    assert not [l for l in src[step2:start] if '"== 3.' in l] and "== 5." in "\n".join(src[start:end])


def test_a_read_only_tree_gets_its_owner_write_bit_back(tmp_path):
    t = tmp_path / "t"
    (t / "a" / "b").mkdir(parents=True)
    (t / "a" / "b" / "f").write_text("x")
    (t / "l").symlink_to("a")
    for q in sorted([t, *t.rglob("*")], reverse=True):
        if not q.is_symlink():
            q.chmod(q.stat().st_mode & ~0o222)
    user_write_back(t)
    assert all(q.stat().st_mode & 0o200 for q in [t, *t.rglob("*")] if not q.is_symlink())
    shutil.rmtree(t)


def test_a_bootstrap_run_by_a_test_leaves_a_removable_tmp_path(tmp_path):
    """Step 5 of every real bootstrap a test runs makes ws/in, ws/hf/hub and ws/uv-cache read-only (a-w); pytest then
    could not remove those trees when it rotated old base temp directories (PytestWarning rm_rf, garbage-* left in the
    system temp by every suite run, seen on repeated suite runs). This file's autouse teardown gives the
    user write bit back, so a real bootstrap test leaves a tmp_path that removes cleanly."""
    bt = tmp_path / "bt"
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(bt),
                          f"{__file__}::test_v2test_gate_passes_when_the_counts_agree"],
                         capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                         cwd=OPS.parent)
    assert run.returncode == 0 and "1 passed" in run.stdout, run.stdout[-1500:]
    ro = [q for q in [bt, *bt.rglob("*")] if not q.is_symlink() and not q.stat().st_mode & 0o200]
    assert not ro, ro[:5]
    shutil.rmtree(bt)


FAKE_PGREP_FROM_FILE = "#!/bin/sh\ncat \"$PGREP_PIDS\"\n"     # `pgrep -u USER` answers the pids the test wrote
FAKE_PGREP_NONE = "#!/bin/sh\nexit 1\n"                        # none is alive
# On a pod, step 5 runs this very suite AS the test user, so a fixture's real bootstrap sees pytest -- its own
# ancestor -- and its own pipelines as that user's processes: `pgrep -u USER`
# answers the caller's whole chain of ancestors, from its parent (a subshell of the bootstrap, or the bootstrap) up
FAKE_PGREP_OWN_TREE = """#!/bin/sh
p=$PPID
while [ -n "$p" ] && [ "$p" -gt 1 ]; do echo "$p"; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
exit 0
"""


def test_a_live_test_user_process_stops_the_bootstrap_before_the_volume_is_writable(tmp_path):
    """A process of the test user still alive from an earlier bootstrap in this container (a
    re-run of §5, GEN's fallback pin set) could write in/, hf/hub or uv-cache between step 2's user-write restore and
    step 5's relock; step 2 refuses first, and the three stay read-only. The process here is a real one that is
    neither an ancestor nor a descendant of the bootstrap (a background sleep of this test)."""
    sleeper = subprocess.Popen(["sleep", "60"])
    try:
        (tmp_path / "pids").write_text(f"{sleeper.pid}\n")
        run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, left_read_only=True,
                                      shims=(("pgrep", FAKE_PGREP_FROM_FILE),),
                                      extra_env={"PGREP_PIDS": str(tmp_path / "pids")})
    finally:
        sleeper.kill()
        sleeper.wait()
    assert run.returncode == 2 and "V2TEST_PROCESS_ALIVE" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert str(sleeper.pid) in run.stderr and "== 3." not in run.stdout
    for d in (tmp_path / "ws" / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):
        assert d.stat().st_mode & 0o222 == 0, d
    ok = run_v2test_gate_fixture(tmp_path / "none", FAKE_PY_AGREE, shims=(("pgrep", FAKE_PGREP_NONE),))
    assert ok.returncode == 0 and "BOOTSTRAP_OK" in ok.stdout, (ok.stdout[-1500:], ok.stderr[-1500:])


def test_the_bootstraps_own_ancestors_and_children_are_not_a_live_test_user_process(tmp_path):
    """When this suite runs as the test user -- a pod's step 5 -- a
    fixture's real bootstrap sees pytest (its ancestor) and its own subshells (its descendants) as that user's
    processes. Those are the bootstrap itself, not a process left behind; only the others count."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, shims=(("pgrep", FAKE_PGREP_OWN_TREE),))
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])


def fake_py_leaves_a_detached_process(v2test_home: str, marker: str) -> str:
    """Counts agree, but the test user's main run leaves a detached process behind (nohup ... &, its output off the
    pipe to tee), alive after the run -- what `setsid runuser -u v2test -- sleep 600` from a test would do."""
    return f"""#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then echo "5/5 tests collected"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then
  if [ "$HOME" = "{v2test_home}" ]; then nohup sleep {marker} > /dev/null 2>&1 < /dev/null & fi
  echo "5 passed, 1 deselected in 0.01s"; exit 0
fi
exit 0
"""


def test_a_test_user_process_left_alive_by_the_test_run_stops_the_bootstrap(tmp_path):
    """Step 2's check alone let a detached test-user process born in step 5 live
    past the last tamper check with BOOTSTRAP_OK printed. The same check (the bootstrap's own ancestors excepted) runs
    again when step 5's test run ends, BEFORE no_skips.sh reads the log."""
    marker = f"97.{os.getpid()}"
    shim = f'#!/bin/sh\nexec /usr/bin/pgrep -f "^sleep {marker}$"\n'   # `pgrep -u v2test`: the one the run left
    try:
        run = run_v2test_gate_fixture(tmp_path, fake_py_leaves_a_detached_process(str(tmp_path / "v2test-home"),
                                                                                  marker), shims=(("pgrep", shim),))
    finally:
        subprocess.run(["/usr/bin/pkill", "-f", f"^sleep {marker}$"])
    assert run.returncode == 2 and "V2TEST_PROCESS_ALIVE" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert "after the test run" in run.stderr
    assert "BOOTSTRAP_OK" not in run.stdout and "no_skips.sh" not in run.stdout


def test_no_skips_reads_roots_own_copy_of_the_log(tmp_path):
    """The manifest skips the pytest log on the pod volume (root's tee writes it while the test user
    runs), and on MooseFS the test user may write it; no_skips.sh therefore reads root's own copy in the root-only
    mktemp -d. The copy on the volume is still written, for people, the gate and the copy home."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    gate = next(l for l in run.stdout.splitlines() if "no_skips.sh" in l).split()
    log = Path(gate[gate.index(next(a for a in gate if a.endswith("no_skips.sh"))) + 1])
    assert log.name == "pytest.log" and not str(log).startswith(str(tmp_path / "ws")), gate
    assert "5 passed" in (tmp_path / "ws" / "pytest.log").read_text()


def test_the_gate_only_path_checks_the_venv_too(tmp_path):
    """venv_check ran only in step 2, which GATE_ONLY skips, so the container gate
    never exercised it; GATE_ONLY now runs it before step 5."""
    out = dry_run_no_gpu_path(tmp_path, GATE_ONLY="1", **gate_marker_env(tmp_path)).stdout
    assert "+ venv_check" in out and out.index("+ venv_check") < out.index("== 5."), out[-2000:]


def test_the_end_of_step_5_idle_check_runs_before_the_last_tamper_check(tmp_path):
    """With test_user_idle first, no test-user process is alive when the last manifest is
    taken, and none can start afterwards (only v2test starts v2test), so the comparison is final. The reverse order let
    a detached process write after the snapshot and exit before the idle check. Both still come before no_skips.sh."""
    out = dry_run_no_gpu_path(tmp_path, GATE_ONLY="1", **gate_marker_env(tmp_path)).stdout
    idle = out.index("+ test_user_idle after the test run")
    tamper = out.index("+ tamper_check the v2test test run")
    assert idle < tamper < out.index("no_skips.sh"), out[-2500:]


def test_a_process_gone_before_it_is_looked_at_is_not_alive(tmp_path):
    gone = subprocess.Popen(["true"])
    gone.wait()
    (tmp_path / "pids").write_text(f"{gone.pid}\n")
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, shims=(("pgrep", FAKE_PGREP_FROM_FILE),),
                                  extra_env={"PGREP_PIDS": str(tmp_path / "pids")})
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])


def test_a_venv_that_is_a_symlink_into_the_workspace_is_refused(tmp_path):
    """layout_check sees V2GPU_LOCAL and the repo; /opt/v2/venv -> /workspace/venv passed it.
    Once step 2 has the venv, its own real path must be off $WS too."""
    def into_ws():
        venv, moved = tmp_path / "local" / "venv", tmp_path / "ws" / "venv"
        shutil.move(str(venv), str(moved))
        venv.symlink_to(moved)
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, prepare=into_ws)
    assert run.returncode == 2 and "LAYOUT_ON_WORKSPACE" in run.stderr, (run.stdout[-1500:], run.stderr[-1500:])
    assert "== 3." not in run.stdout


# Four safety mutants survived because the tests pinned words, not behaviour
# -- layout_check without its device check, the manifest without ctime, and the gate's LAYOUT_NOT_PROVEN and
# WORKSPACE_WRITABLE_BY_TEST_USER checks disabled. Each is now run and must fire on a bad layout.


def device_shims(same: bool, ws: Path) -> tuple:
    """`mountpoint` says every path is a mount; `stat -c %d PATH` gives one device for everything (same=True) or one
    for the pod volume and another for the rest; every other stat call goes to the real stat."""
    stat = f"""#!/bin/sh
if [ "$1" = -c ] && [ "$2" = %d ]; then
  case "$3" in "{ws}"|"{ws}"/*) echo 41 ;; *) {"echo 41" if same else "echo 42"} ;; esac; exit 0
fi
exec /usr/bin/stat "$@"
"""
    return (("mountpoint", "#!/bin/sh\nexit 0\n"), ("stat", stat))


def test_a_venv_root_on_the_pod_volumes_filesystem_is_refused(tmp_path):
    """layout_check's device check: when $WS is a mount of its own, the venv root must be on another filesystem."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE,
                                  shims=device_shims(True, (tmp_path / "ws").resolve()))
    assert run.returncode == 2 and "same filesystem" in run.stderr and "LAYOUT_ON_WORKSPACE" in run.stderr, \
        (run.stdout[-1500:], run.stderr[-1500:])
    assert "== 1." not in run.stdout
    ok = run_v2test_gate_fixture(tmp_path / "other", FAKE_PY_AGREE,
                                 shims=device_shims(False, (tmp_path / "other" / "ws").resolve()))
    assert ok.returncode == 0 and "BOOTSTRAP_OK" in ok.stdout, (ok.stdout[-1500:], ok.stderr[-1500:])


def test_tree_manifest_sees_a_change_that_restores_mtime_and_size(tmp_path):
    """ctime is what an unprivileged user cannot set back: same size, mtime put back with os.utime, and the manifest
    still differs."""
    root = tmp_path / "t"
    root.mkdir()
    f = root / "f"
    f.write_text("aaaa")
    manifest = lambda: subprocess.run([sys.executable, str(OPS / "tree_manifest.py"), str(root)],
                                      capture_output=True, text=True, check=True).stdout
    before, st = manifest(), f.stat()
    time.sleep(0.05)
    f.write_text("bbbb")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert f.stat().st_size == st.st_size and f.stat().st_mtime_ns == st.st_mtime_ns
    assert manifest() != before


GATE = (OPS / "pod_parity_gate.sh").read_text()


def gate_lines(first: str, last: str) -> str:
    """The gate's own inner-script lines from the one starting with `first` to the next one containing `last`, as the
    container sees them (the heredoc's \\$ unescaped)."""
    lines = GATE.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith(first))
    j = next(n for n in range(i, len(lines)) if last in lines[n])
    return "\n".join(lines[i:j + 1]).replace("\\$", "$")


def test_the_gates_layout_proof_fires_on_a_bad_layout(tmp_path):
    body = gate_lines("[ -x /opt/v2/venv/bin/python ]", "LAYOUT_NOT_PROVEN").replace("/opt/v2", str(tmp_path / "local"))
    ws, py = tmp_path / "ws", tmp_path / "local" / "venv" / "bin" / "python"
    ws.mkdir()
    run = lambda: subprocess.run(["bash", "-c", body], capture_output=True, text=True, env={**clean_env(), "WS": str(ws)})
    assert "LAYOUT_NOT_PROVEN" in run().stdout                       # no venv on the container disk
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    good = run()
    assert good.returncode == 0 and "LAYOUT_NOT_PROVEN" not in good.stdout
    (ws / "venv").mkdir()                                            # a venv under the pod volume
    bad = run()
    assert bad.returncode == 1 and "LAYOUT_NOT_PROVEN" in bad.stdout
    (ws / "venv").rmdir()
    for cache in ("hf", "uv-cache"):        # the HF cache and uv's cache belong on the container disk
        (ws / cache).mkdir()
        bad = run()
        assert bad.returncode == 1 and "LAYOUT_NOT_PROVEN" in bad.stdout, cache
        (ws / cache).rmdir()
    assert run().returncode == 0


def test_the_gates_write_probe_fires_on_a_writable_volume_directory(tmp_path):
    body = gate_lines('for d in "$WS/in/"'.replace("$", "\\$"), "done").replace("/opt/v2", str(tmp_path / "local"))
    ws, shim, local = tmp_path / "ws", tmp_path / "shim", tmp_path / "local"
    dirs = (ws / "in", local / "hf" / "hub", local / "uv-cache")   # the caches on the container disk
    for d in dirs:
        d.mkdir(parents=True)
    shim.mkdir()
    (shim / "runuser").write_text(FAKE_RUNUSER_DROP_PRIVILEGE)
    (shim / "runuser").chmod(0o755)
    env = {**clean_env(), "WS": str(ws), "PATH": f"{shim}:{os.environ['PATH']}"}
    run = lambda: subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=env)
    bad = run()
    assert bad.returncode == 1 and "WORKSPACE_WRITABLE_BY_TEST_USER" in bad.stdout and f"{ws}/in/" in bad.stdout
    for probe in tmp_path.rglob(".v2test-probe"):    # the bad run's own probe file (a fresh gate has none)
        probe.unlink()
    for d in dirs[:2]:
        d.chmod(0o555)
    half = run()                                     # uv's cache on the container disk is probed too
    assert half.returncode == 1 and f"{local}/uv-cache/" in half.stdout
    for probe in tmp_path.rglob(".v2test-probe"):
        probe.unlink()
    dirs[2].chmod(0o555)
    good = run()
    assert good.returncode == 0 and "WORKSPACE_WRITABLE_BY_TEST_USER" not in good.stdout


EVIDENCE = ("bootstrap_timings.txt", "volume_receipt.txt", "pytest.log", "env_receipt.json", "fetch_report.json")


def test_a_second_bootstrap_on_the_same_volume_keeps_the_first_ones_evidence(tmp_path):
    """The resume rule runs the bootstrap again on the same pod volume, which used to overwrite
    the first run's evidence (§8 asks for the pod's real bootstrap total). The earlier files are moved, never deleted,
    into $WS/bootstrap-prev/<start epoch>-<pid>/ before this run writes its own."""
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in EVIDENCE:
        (ws / name).write_text(f"first run's {name}\n")
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    [prev] = list((ws / "bootstrap-prev").iterdir())
    for name in EVIDENCE:
        assert (prev / name).read_text() == f"first run's {name}\n", name
    for name in ("bootstrap_timings.txt", "volume_receipt.txt", "pytest.log"):   # this run's own, written afresh
        assert (ws / name).read_text() != f"first run's {name}\n", name
    assert f"moved to {prev}" in run.stdout


def test_a_first_bootstrap_moves_nothing(tmp_path):
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and not (tmp_path / "ws" / "bootstrap-prev").exists() and "moved to" not in run.stdout


def test_first_run_again_on_the_same_volume_installs_the_lock_it_froze(tmp_path):
    """On the first pod of a KIND (no committed lock, FIRST_RUN=1) a restart wipes the venv but
    not $WS/<lock>.new; running the bootstrap again must install exactly that frozen lock, not resolve the packages
    again (they could drift between the two), and keep recording its sha256 in the venv's FIRST_RUN_LOCK.json."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "requirements-gpu.lock.new").write_text("torch==2.14.0\nfrozen-dep==1.2.3\n")
    frozen = (ws / "requirements-gpu.lock.new").read_bytes()
    (tmp_path / "fake-python").write_text(FAKE_PY_AGREE)
    log = tmp_path / "uv.log"
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, with_venv=False, lock=None,
                                  shims=(("uv", FAKE_UV_BUILDS_THE_VENV), ("df", PLENTY_DF)),
                                  extra_env={"FIRST_RUN": "1", "UV_LOG": str(log),
                                             "FAKE_PY_SRC": str(tmp_path / "fake-python")})
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
    calls = log.read_text().splitlines()
    venv = tmp_path / "local" / "venv"
    assert any(c.startswith(f"uv pip sync --python {venv}/bin/python {ws}/requirements-gpu.lock.new ") for c in calls)
    assert not any(" pip install " in f" {c} " or " pip freeze " in f" {c} " for c in calls), calls
    assert (ws / "requirements-gpu.lock.new").read_bytes() == frozen
    record = json.loads((venv / "FIRST_RUN_LOCK.json").read_text())
    assert record == {"requirements-gpu.lock": hashlib.sha256(frozen).hexdigest()}
    assert "FIRST RUN again on this pod volume" in run.stdout


def test_tree_manifest_skips_exactly_the_named_entries(tmp_path):
    """--skip PATH drops the one line of that exact entry (a root's own line, or a file), never its children or a
    neighbour that shares a prefix."""
    root = tmp_path / "ws"
    (root / "d").mkdir(parents=True)
    for name in ("pytest.log", "pytest.log.2", "d/pytest.log"):
        (root / name).write_text("x")
    out = subprocess.run([sys.executable, str(OPS / "tree_manifest.py"), "--skip", str(root), "--skip",
                          str(root / "pytest.log"), str(root)], capture_output=True, text=True, check=True).stdout
    paths = [l.split("\t")[0] for l in out.splitlines()]
    assert paths == sorted(f"{root}/{r}" for r in ("d", "d/pytest.log", "pytest.log.2")), paths


def test_a_container_disk_that_survived_keeps_its_venv(tmp_path):
    """The other side of the resume rule: the operator re-runs §5 on a pod whose container disk survived (VENV_OK
    misread, or a re-run by hand); the bootstrap reuses the venv (no `uv venv`, no disk check) and only re-syncs it to
    the committed lock."""
    (tmp_path / "fake-python").write_text(FAKE_PY_AGREE)
    log = tmp_path / "uv.log"
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE, shims=(("uv", FAKE_UV_BUILDS_THE_VENV),),
                                  extra_env={"UV_LOG": str(log), "FAKE_PY_SRC": str(tmp_path / "fake-python")})
    try:
        assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout[-1500:], run.stderr[-1500:])
        calls = log.read_text().splitlines()
        assert not any(c.startswith("uv venv ") for c in calls), calls
        assert any(c.startswith("uv pip sync ") for c in calls) and "container disk:" not in run.stdout
    finally:
        for d in (tmp_path / "ws" / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):
            for q in [d, *d.rglob("*")]:
                q.chmod(q.stat().st_mode | 0o200)


def test_step_0_records_whether_a_read_only_mode_stops_root(tmp_path):
    """decision 28/09 asked for a-w on the pod volume; whether a 444 file stops root on the EU-RO-1 MooseFS was never
    measured, so step 0 records it (ro_write_probe) and nothing rests on the answer (step 2 restores the user write
    bit first)."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    try:
        assert run.returncode == 0, run.stderr[-1500:]
        receipt = (tmp_path / "ws" / "volume_receipt.txt").read_text()
        probe = next(l for l in receipt.splitlines() if l.startswith("ro_write_probe: "))
        assert probe.startswith(("ro_write_probe: wrote a 444 file", "ro_write_probe: refused")), probe
        if os.geteuid() != 0:                     # the laptop user is not root: a 444 file stops it
            assert probe.startswith("ro_write_probe: refused"), probe
        assert receipt.index("flock_probe:") < receipt.index("ro_write_probe:")   # after the append-mode flock probe
    finally:
        for d in (tmp_path / "ws" / "in", tmp_path / "local" / "hf" / "hub", tmp_path / "local" / "uv-cache"):
            for q in [d, *d.rglob("*")]:
                q.chmod(q.stat().st_mode | 0o200)


# Pinned 27/09/2026: a mutant (env -i dropped from the REAL root recount, only the dry
# echo kept) survived 78/78 -- nothing pinned the real, non-dry root recollection's own environment


FAKE_PY_REFUSE_RUNPOD_ENV = """#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pytest ] && env | grep -q '^RUNPOD_'; then
  echo "RUNPOD_* leaked into this pytest call's environment" >&2
  exit 1
fi
if [ "$1" = -m ] && [ "$2" = pytest ] && [ "$3" = --collect-only ]; then echo "5/5 tests collected"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = pytest ]; then echo "5 passed, 1 deselected in 0.01s"; exit 0; fi
exit 0
"""


def test_the_real_root_recount_never_carries_runpod_api_key(tmp_path):
    """RUNPOD_API_KEY=FAKE is put directly into the bootstrap's OWN environment (as a real pod's /etc/rp_environment
    would leak it if env -i were ever dropped from the root recollection), and the fake python refuses to run
    either of step 5's two pytest calls (TEST_ENV's v2test one, ROOT_ENV's root one) if it sees any RUNPOD_*
    variable -- step 4 (fetch_models), which runs unwrapped by design, is left alone."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_REFUSE_RUNPOD_ENV, runpod_api_key="FAKE")
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout, run.stderr)
    assert "RUNPOD_" not in run.stderr


# decision 27/09: the bootstrap's test-only knobs (RP_ENV_FILE, V2TEST_HOME, only with
# V2GPU_BOOTSTRAP_TEST=1), and the stat check before the first v2test pytest


def test_the_v2test_gate_fixture_touches_only_its_own_tmp_path(tmp_path):
    """The fixture runs the REAL bootstrap (not DRY_RUN). It used to chmod/chown the machine's
    /etc/rp_environment (on a pod as v2test: EPERM, so both gate tests failed the bootstrap gate; as root: the pod's
    own file) and rm -rf the machine's /home/v2test (as root on a pod: the real one). Both now come from tmp_path: the
    HOME it wipes is the test's own, and no command it runs names either real path."""
    run = run_v2test_gate_fixture(tmp_path, FAKE_PY_AGREE)
    assert run.returncode == 0 and "BOOTSTRAP_OK" in run.stdout, (run.stdout, run.stderr)
    assert not (tmp_path / "v2test-home").exists()           # its rm -rf hit the test's HOME (install is a no-op here)
    commands = "\n".join(l for l in run.stdout.splitlines() if l.startswith("+ "))
    assert "/etc/rp_environment" not in commands and "/home/v2test" not in commands, commands
    env_file, home = tmp_path / "rp_environment", tmp_path / "v2test-home"
    assert f"chmod 600 '{env_file}' && chown root:root '{env_file}'" in commands and f"HOME={home} " in commands


def test_an_env_file_not_600_root_stops_the_bootstrap_before_any_v2test_pytest(tmp_path):
    """On a real (not dry) run: a chmod that silently does nothing on the env file (an odd filesystem) leaves it
    644; the stat check refuses (exit 2) before the v2test count, so no pytest runs at all, as v2test or as root (no
    conftest, no test module's top level). chown is a no-op as well, so a run by a non-root user (a pod's v2test)
    reaches the check instead of failing the early chmod on EPERM."""
    env_file = tmp_path / "rp_environment"
    env_file.write_text("export X=1\n")
    env_file.chmod(0o644)
    log = tmp_path / "python-calls.log"
    fake_py = '#!/bin/sh\necho "$*" >> "' + str(log) + '"\n' + FAKE_PY_AGREE.split("\n", 1)[1]
    chmod_noop = ('#!/bin/sh\nfor a in "$@"; do [ "$a" = "' + str(env_file) + '" ] && exit 0; done\n'
                  'exec "' + shutil.which("chmod") + '" "$@"\n')
    run = run_v2test_gate_fixture(tmp_path, fake_py, shims=(("chmod", chmod_noop), ("chown", "#!/bin/sh\nexit 0\n")))
    assert run.returncode == 2 and "not 600:root: refusing" in run.stderr, (run.stdout, run.stderr)
    calls = log.read_text()
    assert "-m v2gpu.fetch_models" in calls and "-m pytest" not in calls, calls   # step 4 ran, then no pytest at all
    assert env_file.stat().st_mode & 0o777 == 0o644 and "== 6." not in run.stdout


@pytest.mark.parametrize("moved", ["env-file", "v2test-home", "both"])
def test_moving_the_env_file_or_the_v2test_home_needs_the_test_flag(tmp_path, moved):
    """(c): a variable that moves the env file could point step 5's chmod 600 at another file. The bootstrap
    takes a non-default RP_ENV_FILE or V2TEST_HOME only with V2GPU_BOOTSTRAP_TEST=1 (exactly 1); otherwise it refuses
    with exit 2 before step 1, printing nothing. With the flag, every use moves."""
    knobs = bootstrap_test_knobs(tmp_path)
    names = {"env-file": ["RP_ENV_FILE"], "v2test-home": ["V2TEST_HOME"], "both": ["RP_ENV_FILE", "V2TEST_HOME"]}[moved]
    chosen = {k: knobs[k] for k in names}
    for flag in (None, "0", "yes", "true", "1 "):
        # default_local=True, so the only test flag in play is this loop's own
        out = dry_run(tmp_path, **chosen, default_local=True, **({} if flag is None else {"V2GPU_BOOTSTRAP_TEST": flag}))
        assert out.returncode == 2 and out.stdout == "", (flag, out.stdout, out.stderr)
        assert "V2GPU_BOOTSTRAP_TEST=1" in out.stderr and "refusing" in out.stderr, (flag, out.stderr)
    ok = dry_run(tmp_path, **chosen, default_local=True, V2GPU_BOOTSTRAP_TEST="1")
    assert ok.returncode == 0 and "BOOTSTRAP_OK" in ok.stdout, ok.stderr
    if "RP_ENV_FILE" in chosen:
        assert "/etc/rp_environment" not in ok.stdout and ok.stdout.count(f"chmod 600 '{knobs['RP_ENV_FILE']}'") == 2
    if "V2TEST_HOME" in chosen:
        assert "/home/v2test" not in ok.stdout and f"rm -rf '{knobs['V2TEST_HOME']}'" in ok.stdout


def test_a_real_run_with_moved_paths_but_no_test_flag_never_starts(tmp_path):
    """(c), on a real (not dry) run: the refusal comes before step 1, so not even the card check runs, and nothing is
    chmod'ed, chown'ed or deleted."""
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "calls.log"
    for name in ("nvidia-smi", "id", "useradd", "runuser", "uv", "chmod", "chown", "rm", "install", "sha256sum"):
        (shim / name).write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 0\n')
        (shim / name).chmod(0o755)
    run = subprocess.run(["bash", str(lockfree_checkout(tmp_path))], capture_output=True, text=True,
                         env={**clean_env(), "PATH": f"{shim}:{os.environ['PATH']}", "WS": str(tmp_path / "ws"),
                              "FIRST_RUN": "1", "RP_ENV_FILE": str(tmp_path / "rp_environment"),
                              "V2TEST_HOME": str(tmp_path / "v2test-home"), "V2GPU_LOCAL": str(tmp_path / "local")})
    assert run.returncode == 2 and "refusing" in run.stderr and run.stdout == "", (run.stdout, run.stderr)
    assert not log.exists()
