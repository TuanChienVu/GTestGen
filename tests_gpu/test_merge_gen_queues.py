# tests_gpu/test_merge_gen_queues.py
"""ops/merge_gen_queues.py (grid pack 05/10, in the repo since decision 05/10): several genqueue queues of ONE pod become
one queue with every gen/recheck job in order, ONE copy window and ONE self-stop. The queues here are real genqueue
output (the fixtures of test_genqueue), so a change of genqueue's queue shape is caught here too."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests_gpu.test_genqueue import argv, trained  # noqa: F401  (trained is a fixture used by name)
from v2gpu.genqueue import main

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "ops" / "merge_gen_queues.py"


def run(*args):
    return subprocess.run([sys.executable, str(MERGE), *map(str, args)], capture_output=True, text=True)


@pytest.fixture
def two_queues(tmp_path, trained):  # noqa: F811
    """A base queue and a seed-42 queue of the same pod, deadline and prompts (two genqueue runs)."""
    run_dir, _ = trained
    base = argv(tmp_path, trained, weights=("base",))
    assert main(base + ["--out", str(tmp_path / "q-base.json")]) == 0
    s42 = list(base)
    s42[s42.index("--weights") + 1] = "s42"
    s42 += ["--train-receipts", str(run_dir / "receipt-train.json"), "--adapters", str(run_dir / "epoch-3"),
            "--out", str(tmp_path / "q-s42.json")]
    assert main(s42) == 0
    return tmp_path / "q-base.json", tmp_path / "q-s42.json"


def test_two_genqueue_queues_merge_into_every_job_then_one_copy_window(tmp_path, two_queues):
    a, b = (json.loads(p.read_text()) for p in two_queues)
    out = tmp_path / "merged.json"
    r = run(out, *two_queues)
    assert r.returncode == 0, r.stderr
    want = {k: a[k] for k in ("queue_dir", "lock_path", "self_stop_cmd", "deadline_utc")}
    want["jobs"] = a["jobs"][:-1] + b["jobs"][:-1] + [b["jobs"][-1]]
    want["ship"] = {s: {**a["ship"].get(s, {}), **b["ship"].get(s, {})} for s in sorted(set(a["ship"]) | set(b["ship"]))}
    assert out.read_bytes() == json.dumps(want, indent=1, sort_keys=True).encode()
    names = [j["name"] for j in want["jobs"]]
    assert sum(n.startswith("copy-window-") for n in names) == 1 and names[-1].startswith("copy-window-")
    assert len(names) == len(set(names)) and want["self_stop_cmd"] == ["runpodctl", "stop", "pod", "$RUNPOD_POD_ID"]


def test_the_merged_queue_passes_jobqueues_own_checks(tmp_path, two_queues, monkeypatch):
    from v2gpu import jobqueue

    fake = tmp_path / "bin"                    # what a pod has and this laptop lacks (resolved by _refusal, never run)
    fake.mkdir()
    (fake / "runpodctl").write_text("#!/bin/sh\nexit 99\n")
    (fake / "runpodctl").chmod(0o755)
    monkeypatch.setenv("RUNPOD_POD_ID", "fake-pod")
    monkeypatch.setenv("PATH", f"{fake}:/usr/bin:/bin")
    out = tmp_path / "merged.json"
    assert run(out, *two_queues).returncode == 0
    assert jobqueue._refusal(json.loads(out.read_text())) is None


def _changed(src: Path, dst: Path, change, first: dict) -> Path:
    q = json.loads(src.read_text())
    change(q, first)
    dst.write_text(json.dumps(q))
    return dst


def _pin(q, first):
    for j in q["jobs"]:
        if "--pin-set" in j["cmd"]:
            j["cmd"][j["cmd"].index("--pin-set") + 1] = "fallback-bnb"


def _ship(q, first):
    """Another source for a target both queues ship (e.g. the prompts), as a hand edit would give."""
    section, target = next((s, t) for s in q["ship"] for t in q["ship"][s] if t in first["ship"].get(s, {}))
    q["ship"][section][target] = "/elsewhere/" + Path(q["ship"][section][target]).name


@pytest.mark.parametrize("change, words", [
    (lambda q, first: q.update(deadline_utc="2026-10-09T00:00:00+00:00"), "deadline_utc"),
    (_pin, "--pin-set"),
    (_ship, "two sources"),
    (lambda q, first: q["jobs"].insert(0, dict(q["jobs"][-1])), "copy-window"),
    (lambda q, first: q["jobs"].insert(0, {**first["jobs"][0], "workdir": "/workspace/runs/other"}), "job name"),
    (lambda q, first: q["jobs"].insert(0, {**first["jobs"][0], "name": "gen-other"}), "workdir"),
])
def test_a_mismatched_queue_is_refused_and_nothing_is_written(tmp_path, two_queues, change, words):
    other = _changed(two_queues[1], tmp_path / "other.json", change, json.loads(two_queues[0].read_text()))
    out = tmp_path / "merged.json"
    r = run(out, two_queues[0], other)
    assert r.returncode == 2 and "REFUSED" in r.stderr and words in r.stderr and not out.exists(), r.stderr


def test_the_same_queue_twice_or_one_queue_alone_is_refused(tmp_path, two_queues):
    out = tmp_path / "merged.json"
    twice = run(out, two_queues[0], two_queues[0])
    assert twice.returncode == 2 and "twice" in twice.stderr and not out.exists()
    alone = run(out, two_queues[0])
    assert alone.returncode == 2 and not out.exists()
