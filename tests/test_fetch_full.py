import importlib.util
import subprocess
from pathlib import Path

import pytest

from tests.phase3_fixture import make_repo

ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("repo"))


def test_fetch_full_flags_missing_commits_and_partial_clones(repo):
    ff = script("fetch_full")
    bare, head = repo
    assert ff.problems(bare, head) == []
    assert ff.problems(bare, "0" * 40) == [f"commit {'0' * 40} absent"]
    subprocess.run(["git", "-C", str(bare), "config", "remote.origin.promisor", "true"], check=True)
    try:
        assert ff.problems(bare, head) == ["partial clone (remote.origin.promisor)"]
    finally:
        subprocess.run(["git", "-C", str(bare), "config", "--unset", "remote.origin.promisor"], check=True)
    assert ff.problems(bare.parent / "missing.git", head) == ["missing"]
