import importlib.util
import os
import re
from pathlib import Path

import pytest

# CPU tests: on a GPU pod they must still run on CPU (the tiny model is never placed on the card). Set before torch loads.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# The repo's own .venv has only pytest: there, skip this directory instead of failing on "import torch".
# The v2test gate: find_spec("torch") can lie under a permission gap (e.g. a v2test that
# cannot read the venv's site-packages), and the plain skip above used to make this whole directory silently vanish
# from collection while the bootstrap's gate stayed green (the collected count shrinks to match). The pod_bootstrap
# test step exports V2GPU_REQUIRE_TORCH=1 for KIND=gpu and KIND=gen: with it set, a missing torch fails collection
# loudly instead of skipping.
if importlib.util.find_spec("torch"):
    collect_ignore_glob = []
elif os.environ.get("V2GPU_REQUIRE_TORCH") == "1":
    raise RuntimeError(
        "V2GPU_REQUIRE_TORCH=1 but torch is not importable here: refusing instead of silently skipping the whole "
        "tests_gpu directory (the v2test gate)")
else:
    collect_ignore_glob = ["test_*.py"]

# The laptop must not read the user's default Hugging Face cache -- a project-only HF_HOME
# (ops/populate_hf_cache.sh fills its "hub" with the pinned models' tokenizer-only files, checked against
# the Hub's LFS sha256), used ONLY when HF_HOME is unset and the path exists. On a pod (and inside the pod-parity
# gate container) the bootstrap always sets HF_HOME itself (/opt/v2/hf); this default never overrides that. The
# path is relative: hf-cache/ at the repository root, used when pytest runs from there.
_V2_HF_CACHE = Path("hf-cache")
if "HF_HOME" not in os.environ and _V2_HF_CACHE.is_dir():
    os.environ["HF_HOME"] = str(_V2_HF_CACHE)

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # tests read the pinned tokenizers from the local HF cache only
# generate --mode test writes job variants (main, dev_*) only when the test suite says so
os.environ.setdefault("V2GPU_TEST_JOB_VARIANTS", "1")


@pytest.fixture(scope="session")
def tokenizers():
    from v2gpu.models import MODELS, load_tokenizer

    out = {}
    for key, spec in MODELS.items():
        try:
            out[key] = load_tokenizer(spec)
        except OSError as e:  # not in the local cache
            pytest.skip(f"tokenizer {spec.hf_id}@{spec.revision} not cached: {e}")
    return out


# On a pod this suite runs inside the bootstrap's own gate, beside the pod's real
# runpodctl, and /etc/rp_environment exports that pod's PATH, RUNPOD_POD_ID and RUNPOD_API_KEY. No test may reach any of
# them. Traps for runpodctl, ssh and scp go first on PATH for the whole session. RP_ENV (the env file the sheets' watchdog
# sources, default /etc/rp_environment) names a file that keeps the traps first. A test that reaches a trap fails at its
# teardown; tests that need these commands bring fakes of their own, first on their own PATH.
TRAP = """#!/bin/sh
echo "${{0##*/}} $*" >> "{log}"
echo "TRAP: a test reached a real pod command: ${{0##*/}} $* (tests_gpu/conftest.py)" >&2
exit 97
"""
# The file-transfer and HTTP commands too; the trap needs no command of PATH
TRAPPED = ("runpodctl", "ssh", "scp", "sftp", "rsync", "curl", "wget")


# No test reads or passes on RUNPOD_API_KEY or any RUNPOD_* variable of the process that runs the suite (on
# a pod: the pod's own). They are removed here, when this conftest loads, before collection and before any test, ops
# script or subprocess runs; a test that needs RUNPOD_POD_ID sets a fake of its own.
for _key in [k for k in os.environ if k.startswith("RUNPOD_")]:
    del os.environ[_key]


@pytest.fixture(scope="session", autouse=True)
def _pod_traps(tmp_path_factory):
    d = tmp_path_factory.mktemp("pod-traps")
    log = d / "reached.log"
    log.write_text("")
    for tool in TRAPPED:
        (d / tool).write_text(TRAP.format(log=log))
        (d / tool).chmod(0o755)
    (d / "rp_environment").write_text(f'export PATH="{d}:$PATH"\n')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", f"{d}{os.pathsep}{os.environ.get('PATH', '')}")
        mp.setenv("RP_ENV", str(d / "rp_environment"))
        yield log


@pytest.fixture(autouse=True)
def _no_real_pod_command(_pod_traps):
    before = _pod_traps.read_text()
    yield
    reached = _pod_traps.read_text()[len(before):]
    assert not reached, f"a test reached a real pod command: {reached.strip()} (use a fake of the test's own)"


# Seen in a pod-parity run: pod_bootstrap.sh's step 5 can chmod/chown
# RP_ENV_FILE and rm -rf TEST_HOME for real, when a test runs it without DRY_RUN=1 and without overriding both. A
# test must always point both at its own tmp path (see test_pod_safety.py's static scan); this is the runtime
# backstop for when one does not. The two defaults are read out of ops/pod_bootstrap.sh itself (never spelled out
# here) so this file names neither real path, matching test_pod_safety's "no test names the pod's env file" rule.
# test_pod_safety.py's sim_dir() copies only this file into an isolated tmp dir with no sibling ops/: there the
# default can't be read, and there is also nothing for this fixture to protect (the isolated copy structurally
# cannot reach the real bootstrap), so _bootstrap_default() returns None and the check below becomes a no-op.
def _bootstrap_default(var: str):
    path = Path(__file__).resolve().parents[1] / "ops" / "pod_bootstrap.sh"
    if not path.is_file():
        return None
    m = re.search(var + r"=\$\{" + var + r":-([^}]+)\}", path.read_text())
    return Path(m.group(1)) if m else None


_REAL_RP_ENV = _bootstrap_default("RP_ENV_FILE")
_REAL_TEST_HOME = _bootstrap_default("V2TEST_HOME")
_REAL_LOCAL = _bootstrap_default("V2GPU_LOCAL")      # the container-disk venv root (/opt/v2)


def _snapshot(path):
    # (st_ino, st_mode, st_uid), never st_mtime_ns (since 27/09/2026). On a pod,
    # V2TEST_HOME IS the v2test suite's own $HOME (step 5): any test or library that merely creates a
    # top-level entry there (.cache, .nv, .config) updates the directory's mtime, which used to fail this
    # backstop for a reason that has nothing to do with tampering. Inode, mode and ownership instead: unaffected by an
    # entry
    # added inside the directory, but changed by exactly what this check guards against (chmod, chown, or a
    # delete+recreate).
    if path is None:
        return None
    try:
        st = path.stat()
        return st.st_ino, st.st_mode, st.st_uid
    except OSError:
        return None


@pytest.fixture(autouse=True)
def _real_system_paths_untouched():
    before = _snapshot(_REAL_RP_ENV), _snapshot(_REAL_TEST_HOME), _snapshot(_REAL_LOCAL)
    yield
    if _REAL_RP_ENV is None and _REAL_TEST_HOME is None and _REAL_LOCAL is None:
        return
    after = _snapshot(_REAL_RP_ENV), _snapshot(_REAL_TEST_HOME), _snapshot(_REAL_LOCAL)
    assert before == after, (
        f"a test touched a real system path ({_REAL_RP_ENV}, {_REAL_TEST_HOME} or {_REAL_LOCAL}): a real, non-DRY_RUN "
        f"bootstrap invocation must always override RP_ENV_FILE, V2TEST_HOME and V2GPU_LOCAL to a tmp path")
