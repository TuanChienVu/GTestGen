"""Dry runs of ops/pod_bootstrap.sh for the tests, on a lock-free copy of this checkout's bootstrap.

After the first pod of a KIND, requirements-<kind>[-<set>].lock is committed, and the real bootstrap then
refuses FIRST_RUN=1. Tests that ran the checkout's own script with FIRST_RUN=1 therefore turned the
suite red on the laptop and failed every later pod's bootstrap gate. Here every dry run uses a copy that holds ops/ and
every requirements-*.in of this checkout but never a lock, so the first-run path is taken explicitly, whatever locks the
checkout holds. The locked path is tested on copies that add a lock (test_ops_scripts). Nothing under v2gpu/ changes,
so gen_code_sha256 stays the same."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTORS = ("KIND", "GEN_SET", "FIRST_RUN")
# The bootstrap's test-only knobs (its env file, its v2test HOME, the flag that allows moving
# them); a test that needs them sets its own, into its tmp_path
TEST_KNOBS = ("RP_ENV_FILE", "V2TEST_HOME", "V2GPU_LOCAL", "V2GPU_BOOTSTRAP_TEST")   # + V2GPU_LOCAL


def clean_env() -> dict:
    """This process's environment without the selectors of a bootstrap that may be running this suite on a pod (once,
    KIND=gen leaked into the tests and failed the gen bootstrap's own gate), nor its test-only
    knobs."""
    return {k: v for k, v in os.environ.items() if k not in SELECTORS + TEST_KNOBS}


def hermetic_local(tmp_path: Path) -> dict:
    """The bootstrap's venv root under tmp_path, with the test flag that allows
    moving it. The default, /opt/v2, EXISTS where this suite also runs -- the gate container and a pod's own step 5
    (the venv of the suite itself) -- and a bootstrap that finds a venv there skips building one, so a test that leaves
    the default in place depends on the machine it runs on."""
    return {"V2GPU_LOCAL": str(tmp_path / "local"), "V2GPU_BOOTSTRAP_TEST": "1"}


def lockfree_checkout(root: Path) -> Path:
    """ops/pod_bootstrap.sh of a lock-free copy under root, made once per root: ops/*.sh and requirements-*.in."""
    script = root / "lockfree-checkout" / "ops" / "pod_bootstrap.sh"
    if not script.exists():
        (script.parent).mkdir(parents=True)
        for s in (ROOT / "ops").glob("*.sh"):
            shutil.copy2(s, script.parent / s.name)
        for r in ROOT.glob("requirements-*.in"):
            shutil.copy2(r, script.parents[1] / r.name)
    assert not list(script.parents[1].glob("requirements-*.lock"))
    return script


def dry_run(tmp_path: Path, *args: str, ws: Path | None = None, default_local: bool = False,
            **env: str) -> subprocess.CompletedProcess:
    """`FIRST_RUN=1 DRY_RUN=1 bash ops/pod_bootstrap.sh ARGS` on the lock-free copy, WS = ws (default tmp_path); env
    adds KIND, GEN_SET or anything else, and may override FIRST_RUN, never DRY_RUN (it comes last: a dry run stays
    dry). The venv root is hermetic_local(tmp_path) unless default_local=True, which only a test that
    checks the default /opt/v2 itself passes."""
    local = {} if default_local else hermetic_local(tmp_path)
    run_env = {**clean_env(), "WS": str(ws or tmp_path), "FIRST_RUN": "1", **local, **env, "DRY_RUN": "1"}
    return subprocess.run(["bash", str(lockfree_checkout(tmp_path)), *args], capture_output=True, text=True,
                          env=run_env)


# Since 27/09/2026 GATE_ONLY is a selector too (like KIND/GEN_SET/FIRST_RUN, it
# changes which steps the bootstrap runs), so a suite run that happened to inherit it from an enclosing pod bootstrap
# must never leak it into a dry run. GATE_MARKER is the same idea for GATE_ONLY's own marker file.
SELECTORS = SELECTORS + ("GATE_ONLY",)
TEST_KNOBS = TEST_KNOBS + ("GATE_MARKER",)


# A DRY_RUN=1 GATE_ONLY=1 invocation only ever echoes the commands wrapped in the
# script's own run() (nothing runs for real); the handful of unwrapped lines (the RP_ENV_FILE/GATE_ONLY guards, the
# REPO= assignment, the KIND case) need only `dirname` as an external binary -- confirmed empirically by running the
# real script with PATH pointing only at a symlink to it. A GATE_ONLY dry run that is not deliberately testing the
# "nvidia-smi on PATH" refusal (test_gate_only_with_nvidia_smi_on_path_...) must use this, not dry_run: the test
# process's own real PATH may hold a real nvidia-smi (a GPU pod's step 5 runs with the NVIDIA container runtime's
# /usr/bin on PATH), which would make the guard refuse regardless of what the test means to exercise.
SANITIZED_BINARIES = ("dirname",)


def sanitized_bin(tmp_path: Path) -> Path:
    """A PATH directory holding symlinks to only SANITIZED_BINARIES, resolved once from this process's own real
    PATH before it is discarded."""
    bindir = tmp_path / "sanitized-bin"
    if not bindir.exists():
        bindir.mkdir()
        for name in SANITIZED_BINARIES:
            real = shutil.which(name)
            assert real, f"{name} not found on this machine's PATH: needed for the sanitized dry-run PATH"
            (bindir / name).symlink_to(real)
    return bindir


def dry_run_no_gpu_path(tmp_path: Path, *args: str, ws: Path | None = None, default_local: bool = False,
                        **env: str) -> subprocess.CompletedProcess:
    """Like dry_run, except PATH is sanitized_bin(tmp_path) (never this process's own PATH) and bash is invoked by
    its absolute path, so a real nvidia-smi (or anything else) on the HOST's PATH cannot affect a GATE_ONLY dry run
    that is not deliberately testing the "looks like a real pod" refusal."""
    bash_abs = shutil.which("bash")
    assert bash_abs, "bash not found on this machine's PATH"
    local = {} if default_local else hermetic_local(tmp_path)          # as in dry_run
    run_env = {**clean_env(), "WS": str(ws or tmp_path), "FIRST_RUN": "1", **local, **env, "DRY_RUN": "1",
              "PATH": str(sanitized_bin(tmp_path))}
    return subprocess.run([bash_abs, str(lockfree_checkout(tmp_path)), *args], capture_output=True, text=True,
                          env=run_env)
