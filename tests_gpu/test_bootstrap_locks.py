"""The suite stays green whether or not the requirements locks are committed.

requirements-gpu.lock is committed after the training canary and requirements-gen[-<set>].lock after the first
generation pod of each pin set; from then on the real bootstrap refuses FIRST_RUN=1. The bootstrap tests run on a
lock-free copy of the bootstrap (tests_gpu.lockfree), so their outcome must not depend on the checkout's locks. Proof:
a copy of this checkout, without any lock and with every lock, runs the bootstrap test group green. The copy's own
bootstrap takes the path its state calls for."""
import shutil
import subprocess
import sys

import pytest

from tests_gpu.lockfree import ROOT, clean_env, hermetic_local

LOCKS = ("gpu", "gen", "gen-fallback-tf", "gen-fallback-bnb", "gen-fallback-tf-bnb")
# every test that runs ops/pod_bootstrap.sh, but for the nested gen-list simulation (it runs the whole generation list
# and takes its test list from the same lockfree.dry_run) and for the pod-parity check, which runs test_ops_scripts.py
# itself
GROUP = ["tests_gpu/test_ops_scripts.py", "tests_gpu/test_genqueue.py", "-k",
         "bootstrap and not selectors_exported"]


def checkout_copy(dest, locks):
    """The parts of this checkout the bootstrap tests read (code, tests, ops, config, requirements), no lock but those
    given (a one-line fake lock each)."""
    for d in ("v2gpu", "tests_gpu", "ops", "config"):
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for r in ROOT.glob("requirements-*"):
        if r.suffix != ".lock":
            shutil.copy2(r, dest / r.name)
    for kind in locks:
        (dest / f"requirements-{kind}.lock").write_text("pkg==1.0\n")
    return dest


@pytest.mark.parametrize("locks", [(), LOCKS], ids=["no-lock", "every-lock"])
def test_the_bootstrap_tests_pass_with_and_without_committed_locks(tmp_path, locks):
    repo = checkout_copy(tmp_path / "repo", locks)
    assert sorted(p.name for p in repo.glob("requirements-*.lock")) == sorted(f"requirements-{k}.lock" for k in locks)
    for kind in ("gpu", "gen"):                         # the copy's own bootstrap follows its state
        own = subprocess.run(["bash", str(repo / "ops" / "pod_bootstrap.sh")], capture_output=True, text=True,
                             env={**clean_env(), **hermetic_local(tmp_path), "DRY_RUN": "1",   # not /opt/v2
                                  "WS": str(tmp_path / f"ws-{kind}"), "KIND": kind})
        if locks:
            assert own.returncode == 0 and "pip sync" in own.stdout and "BOOTSTRAP_OK" in own.stdout, own.stderr
        else:
            assert own.returncode == 2 and "FIRST_RUN=1" in own.stderr
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                          str(tmp_path / "bt"), *GROUP], cwd=repo, capture_output=True, text=True, env=clean_env())
    assert run.returncode == 0, run.stdout[-3000:]
    assert " passed" in run.stdout and "failed" not in run.stdout and "skipped" not in run.stdout
