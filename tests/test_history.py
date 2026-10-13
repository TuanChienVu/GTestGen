import os
import subprocess
from datetime import datetime
from pathlib import Path

from v2tools.history import earliest, first_seen, line_times, min_time

def commit(repo, date, message):
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True, env=env)

def test_line_times_and_oldest_line_rule(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.feature").write_text("Feature: x\n  Scenario: old\n    Given a\n")
    commit(tmp_path, "2023-06-01T00:00:00Z", "old")
    (tmp_path / "a.feature").write_text("Feature: x\n  Scenario: old\n    Given a changed\n")
    commit(tmp_path, "2024-06-01T00:00:00Z", "edit")
    times = line_times(tmp_path, "a.feature")
    assert times[2] < times[3]
    assert min_time(times, [2, 3]) == times[2]
    assert min_time(times, []) is None

def ts(day):
    return int(datetime.fromisoformat(day + "T00:00:00+00:00").timestamp())

def test_line_times_use_author_time(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.feature").write_text("Feature: x\n")
    env = dict(os.environ, GIT_AUTHOR_DATE="2023-01-01T00:00:00Z", GIT_COMMITTER_DATE="2025-01-01T00:00:00Z",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "rebased later"], cwd=tmp_path, check=True, env=env)
    assert line_times(tmp_path, "a.feature")[1] == ts("2023-01-01")

def test_first_seen_survives_splitting_a_file(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "big.feature").write_text("Feature: x\n  Scenario: A\n    Given a\n  Scenario: B\n    Given b\n")
    commit(tmp_path, "2022-01-01T00:00:00Z", "old")
    (tmp_path / "big.feature").write_text("Feature: x\n  Scenario: A\n    Given a\n")
    (tmp_path / "b.feature").write_text("Feature: y\n  Scenario: B\n    Given b\n")
    commit(tmp_path, "2025-01-01T00:00:00Z", "split")
    blamed = line_times(tmp_path, "b.feature")[2]
    assert blamed == ts("2025-01-01")                                   # blame dates the line to the split
    seen = first_seen(tmp_path, ["*.feature"])
    assert seen["Scenario: B"] == ts("2022-01-01") and seen["Feature: y"] == ts("2025-01-01")
    assert earliest(blamed, seen["Scenario: B"]) == ts("2022-01-01") and earliest(None, None) is None
