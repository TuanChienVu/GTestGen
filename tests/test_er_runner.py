import fcntl
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from er import db, runner
from er.manifest import load_manifest, write_run
from er.runner import IMAGE, OutputError, classify, docker_cmd, read_capped, run_once

REC = {"id": "r", "scenario_name": "Adding a zone", "steps": ["a", "b"], "background": None,
       "scenario": "Scenario: Adding a zone\n    Given a\n    When b\n", "step_kinds": ["reuse", "create"],
       "suite": "api_x", "feature_path": "f.feature", "scenario_line": 2}
JUNIT = ('<?xml version="1.0" encoding="UTF-8"?>\n<testsuites name="api_x"><testsuite name="F" tests="1">'
         '<testcase name="{name}" status="{status}"></testcase></testsuite></testsuites>\n')
IMAGE_ID = "sha256:" + "ab" * 32
REAL_RUN = subprocess.run


def write(out, name="Adding a zone", status="passed", steps=("passed", "passed"), junit=None, trace=None):
    (out / "junit").mkdir(parents=True)
    (out / "junit" / "api_x.xml").write_text(junit or JUNIT.format(name=name, status=status), encoding="utf-8")
    t = trace or {"steps": [{"i": i, "type": k, "text": x, "status": s}
                            for i, (x, k, s) in enumerate(zip("ab", ("Given", "When"), steps))], "events": []}
    (out / "trace.json").write_text(json.dumps(t), encoding="utf-8")


def test_pass_undefined_and_oom(tmp_path):
    write(tmp_path / "p")
    assert classify(REC, tmp_path / "p", {"behat_exit": 0})[0] == "PASS"
    write(tmp_path / "u", steps=("passed", "undefined"))
    assert classify(REC, tmp_path / "u", {"behat_exit": 0})[0] == "UNDEFINED"
    assert classify(REC, tmp_path / "p", {"behat_exit": 137, "oom_kill": 1})[0] == "OOM"


def test_oom_only_when_behat_was_killed(tmp_path):
    write(tmp_path / "p")
    # an OOM-killed helper process does not make a run that Behat finished an OOM
    assert classify(REC, tmp_path / "p", {"behat_exit": 0, "oom_kill": 1})[0] == "PASS"
    assert classify(REC, tmp_path / "p", {"behat_exit": 124, "oom_kill": 1})[0] == "TIMEOUT"
    assert classify(REC, tmp_path / "p", {"behat_exit": 137})[0] == "TIMEOUT"


def test_wrong_scenario_zero_scenarios_and_crash_are_errors(tmp_path):
    write(tmp_path / "w", name="Other")
    assert classify(REC, tmp_path / "w", {"behat_exit": 0})[0] == "ERROR"
    (tmp_path / "z" / "junit").mkdir(parents=True)
    assert classify(REC, tmp_path / "z", {"behat_exit": 0})[0] == "ERROR"
    write(tmp_path / "c")
    assert classify(REC, tmp_path / "c", {"behat_exit": 255})[0] == "ERROR"
    assert classify(REC, tmp_path / "c", {"behat_exit": 124})[0] == "TIMEOUT"


def test_hostile_junit_bad_trace_and_symlinks_are_errors(tmp_path):
    bomb = ('<?xml version="1.0"?>\n<!DOCTYPE t [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]>\n'
            '<testsuites><testsuite><testcase name="&b;" status="passed"/></testsuite></testsuites>\n')
    write(tmp_path / "x", junit=bomb)
    assert classify(REC, tmp_path / "x", {"behat_exit": 0})[0] == "ERROR"
    write(tmp_path / "s", trace={"steps": "all passed", "events": []})
    status, detail, _ = classify(REC, tmp_path / "s", {"behat_exit": 0})
    assert status == "ERROR" and "TraceError" in detail["error"]
    write(tmp_path / "l")
    (tmp_path / "l" / "trace.json").unlink()
    (tmp_path / "l" / "trace.json").symlink_to(tmp_path / "p_elsewhere.json")
    (tmp_path / "p_elsewhere.json").write_text("{}", encoding="utf-8")
    assert classify(REC, tmp_path / "l", {"behat_exit": 0})[0] == "ERROR"


def test_inner_json_must_be_an_object(tmp_path):
    write(tmp_path / "p")
    for inner in ([0], "x", None, 3):
        status, detail, trace = classify(REC, tmp_path / "p", inner)
        assert status == "ERROR" and "inner.json" in detail["error"] and trace is None


def test_noop_steps_must_match_the_request(tmp_path):
    write(tmp_path / "m", steps=("passed", "noop"))
    assert classify(REC, tmp_path / "m", {"behat_exit": 0}, noop=(1,))[0] == "FAIL"
    status, detail, _ = classify(REC, tmp_path / "m", {"behat_exit": 0})
    assert status == "ERROR" and detail["error"] == "noop steps differ from the request"
    write(tmp_path / "p")
    status, detail, _ = classify(REC, tmp_path / "p", {"behat_exit": 0}, noop=(1,))
    assert status == "ERROR" and detail["error"] == "noop steps differ from the request"
    assert classify(REC, tmp_path / "m", {"behat_exit": 0}, noop=(0,))[0] == "ERROR"


def test_read_capped_refuses_large_files(tmp_path):
    (tmp_path / "big").write_bytes(b"x" * 100)
    assert read_capped(tmp_path / "big", 100) == b"x" * 100
    with pytest.raises(OutputError):
        read_capped(tmp_path / "big", 99)


def no_hang(fn, fifos, seconds=5.0):
    """fn() in a thread; if it is still running after `seconds` (a reader blocked opening a FIFO), give the FIFOs a
    writer so that it can finish, then fail."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
            box["exc"] = exc
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    hung = t.is_alive()
    for _ in range(100):
        if not t.is_alive():
            break
        for f in fifos:
            try:
                os.close(os.open(f, os.O_WRONLY | os.O_NONBLOCK))
            except OSError:
                pass
        t.join(0.1)
    if hung:
        pytest.fail(f"blocked on a FIFO ({', '.join(Path(f).name for f in fifos)})")
    if "exc" in box:
        raise box["exc"]
    return box["value"]


def test_read_capped_refuses_fifos_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "fifo")

    def read():
        with pytest.raises(OutputError, match="not a regular file"):
            read_capped(tmp_path / "fifo", 100)
    no_hang(read, [tmp_path / "fifo"])


def test_read_head_reads_at_most_n_bytes_of_a_regular_file_only(tmp_path):
    (tmp_path / "big").write_bytes(b"0123456789" * 1000)
    assert runner.read_head(tmp_path / "big", 25) == b"0123456789012345678901234"
    assert runner.read_head(tmp_path / "big", 1 << 20) == b"0123456789" * 1000  # no size limit on the file
    (tmp_path / "link").symlink_to(tmp_path / "big")
    (tmp_path / "dir").mkdir()
    os.mkfifo(tmp_path / "fifo")
    for bad in ("link", "dir", "fifo", "missing"):
        def read(bad=bad):
            with pytest.raises(OutputError):
                runner.read_head(tmp_path / bad, 10)
        no_hang(read, [tmp_path / "fifo"])


# ---------- run_once with a fake docker (no Docker needed) ----------

@pytest.fixture
def lanes(tmp_path, monkeypatch):
    """Lane secrets and locks under tmp_path, a fixed image id: unit tests never touch the real lanes."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    for n in (1, 2, 3):
        (secrets / f"lane{n}.pw").write_text(f"pw-secret-{n}\n", encoding="utf-8")
    monkeypatch.setattr(db, "SECRETS_DIR", secrets)
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(runner, "image_id", lambda image=IMAGE: IMAGE_ID)
    return secrets


def entry_of(tmp_path):
    item = {"record": {**REC, "id": "r", "t_vis": 1}, "condition": "gold", "slug": "f",
            "base": {"src/Sylius/Behat/A.php": "<?php\n"}, "files": {}}
    run_dir = write_run(tmp_path / "m", "r", [item])
    (e,) = load_manifest(run_dir)
    return tmp_path / "m", e


def lock_held(lane: int) -> bool:
    with open(runner.LOCK_DIR / f"gtestgen-er-lane{lane}.lock", "w", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False


class FakeDocker:
    """Stands in for subprocess.run: records every docker call (and whether the lane lock was held); `docker run`
    writes the container's outputs with `outputs(out)` or raises `exc`. Other commands (chmod) really run."""

    def __init__(self, outputs=None, exc=None):
        self.outputs, self.exc, self.calls = outputs, exc, []

    def __call__(self, cmd, **kw):
        if cmd[0] != "docker":
            return REAL_RUN(cmd, **kw)
        assert kw.get("timeout"), f"no timeout on {cmd[:3]}"
        name = cmd[cmd.index("--name") + 1] if "--name" in cmd else cmd[-1]
        lane = int(name.rsplit("lane", 1)[1]) if "gtestgen-er-lane" in name else None
        self.calls.append((list(cmd), kw, lane is not None and lock_held(lane)))
        if cmd[1] == "run":
            if self.exc is not None:
                raise self.exc
            if self.outputs is not None:
                self.outputs(Path(next(v[:-len(":/out")] for v in cmd if v.endswith(":/out"))))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def outputs(inner=None, log="", **kw):
    def w(out: Path) -> None:
        write(out, **kw)
        text = inner if isinstance(inner, str) else json.dumps(inner or {"cold": 0, "behat_exit": 0})
        (out / "inner.json").write_text(text, encoding="utf-8")
        (out / "behat.log").write_text(log, encoding="utf-8")
    return w


def no_secret(obj) -> bool:
    return "pw-secret" not in json.dumps(obj, default=str)


def test_docker_cmd_is_hardened_and_uses_the_lane_user(tmp_path, lanes):
    root, e = entry_of(tmp_path)
    cmd = docker_cmd(REC, root, e, 2, tmp_path / "o", (1,), "gtestgen-er-lane2")
    for flag in ("--cap-drop", "ALL", "no-new-privileges", "--memory-swap", "--pids-limit", "--ulimit"):
        assert flag in cmd
    assert "ER_NOOP=1" in cmd and "--network" in cmd and "gtestgen-er_er" in cmd
    # the immutable image id, not the tag; the URL (with the lane password) only by name, never on a command line
    assert IMAGE_ID in cmd and IMAGE not in cmd
    assert cmd[cmd.index("DATABASE_URL") - 1] == "-e" and no_secret(cmd)
    assert db.db_url(2) == "mysql://er_lane2:pw-secret-2@db:3306/sylius_lane2?serverVersion=8.4.0&charset=utf8mb4"


def test_lane_password_comes_from_the_secrets_dir(tmp_path, lanes):
    assert db.lane_password(3) == "pw-secret-3"
    with pytest.raises(db.InfraError, match="init-db"):
        db.lane_password(4)
    (lanes / "lane1.pw").write_text("\n", encoding="utf-8")
    with pytest.raises(db.InfraError):
        db.lane_password(1)


def test_run_once_fixed_name_orphan_removed_under_the_lock(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    fake = FakeDocker(outputs())
    monkeypatch.setattr(subprocess, "run", fake)
    r = run_once(REC, root, e, 2, tmp_path / "o")
    assert r["status"] == "PASS" and "infra_source" not in r and no_secret(r)
    (rm, _, rm_locked), (run, kw, run_locked) = fake.calls
    assert rm == ["docker", "rm", "-f", "gtestgen-er-lane2"] and rm_locked and run_locked
    assert run[:2] == ["docker", "run"] and run[run.index("--name") + 1] == "gtestgen-er-lane2"
    assert kw["env"]["DATABASE_URL"] == db.db_url(2) and kw["timeout"] == runner.RUN_TIMEOUT + 60
    assert all(no_secret(cmd) for cmd, _, _ in fake.calls)


@pytest.mark.parametrize("exc, source", [
    (subprocess.TimeoutExpired(["docker", "run", "-e", "DATABASE_URL"], 120), "timeout"),
    (subprocess.CalledProcessError(125, ["docker", "run"], b"",
                                   b"docker: Error response: mysql://er_lane2:pw-secret-2@db failed"), "docker"),
    (OSError(2, "No such file or directory: docker"), "os"),
])
def test_run_once_infra_sources(tmp_path, lanes, monkeypatch, exc, source):
    root, e = entry_of(tmp_path)
    fake = FakeDocker(exc=exc)
    monkeypatch.setattr(subprocess, "run", fake)
    r = run_once(REC, root, e, 2, tmp_path / "o")
    assert r["status"] == "INFRA" and r["infra_source"] == source and r["trace"] is None and no_secret(r)
    if source == "timeout":  # the outer timeout is INFRA now; the container is removed before the lock is released
        assert [c for c, _, _ in fake.calls][-1] == ["docker", "rm", "-f", "gtestgen-er-lane2"]
        assert fake.calls[-1][2]
    if source == "docker":
        assert "***" in r["error"]


SECRET = "Zq7XkP9wLm3VtR8sQ2hJ"


def leaked(obj, secret: str = SECRET, n: int = 4) -> list[str]:
    """The fragments (n chars or more) of `secret` found in obj: a password cut by a truncation leaks a part of it."""
    text = json.dumps(obj, default=str)
    return [secret[i:i + n] for i in range(len(secret) - n + 1) if secret[i:i + n] in text]


def step_text_cut_in_the_secret() -> str:
    """A step text such that trace.validate_trace's `str(step)[:80]` ends inside SECRET (after its first 10 chars)."""
    head = str({"i": 0, "type": "Given", "text": ""})[:-2]  # "{'i': 0, 'type': 'Given', 'text': '"
    return "x" * (80 - len(head) - 10) + SECRET


def secret_outputs(case: str):
    if case == "junit name":  # "ran '<name[:80]>'"
        return outputs(name="x" * 70 + SECRET)
    if case == "junit status":  # "behat exit 0, junit <status>": capped (80) too, after masking
        return outputs(status="y" * 70 + SECRET + "z" * 1000)
    trace = {"steps": [{"i": 0, "type": "Given", "text": step_text_cut_in_the_secret(), "status": "bogus"}],
             "events": []}
    return outputs(trace=trace)  # TraceError("bad step <str(step)[:80]>"): cut inside the password upstream


@pytest.mark.parametrize("case", ["docker stderr", "junit name", "junit status", "trace error"])
def test_the_password_is_masked_before_any_truncation(tmp_path, lanes, monkeypatch, case):
    (lanes / "lane2.pw").write_text(SECRET + "\n", encoding="utf-8")
    root, e = entry_of(tmp_path)
    if case == "docker stderr":  # the stderr tail keeps the last 300 chars: 10 of them in the password
        stderr = ("A" * 50 + SECRET + "B" * 290).encode()
        fake = FakeDocker(exc=subprocess.CalledProcessError(125, ["docker", "run"], b"", stderr))
    else:
        fake = FakeDocker(secret_outputs(case))
    monkeypatch.setattr(subprocess, "run", fake)
    r = run_once(REC, root, e, 2, tmp_path / "o")
    assert r["status"] in ("ERROR", "INFRA") and "***" in r["error"], r
    assert leaked(r) == [] and len(r["error"]) <= 400, r["error"]


def test_missing_inner_json_is_infra_and_other_bad_inner_is_error(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeDocker(lambda out: None))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert (r["status"], r["infra_source"]) == ("INFRA", "inner")
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(inner="[0, 1]")))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert r["status"] == "ERROR" and "infra_source" not in r
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(inner="{not json")))
    assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "ERROR"


@pytest.mark.parametrize("line", [
    "Doctrine\\DBAL\\Exception\\ConnectionException: An exception occurred in the driver",
    "Doctrine\\DBAL\\Exception\\ConnectionLost: SQLSTATE[HY000]: General error: 2006 MySQL server has gone away",
    "SQLSTATE[HY000] [1040] Too many connections",
    "SQLSTATE[HY000] [1045] Access denied for user 'er_lane1'@'172.18.0.3' (using password: YES)",
    "SQLSTATE[HY000] [2013] Lost connection to MySQL server during query",
    "SQLSTATE[HY000]: General error: 2013 Lost connection to server during query",
    # a full database disk is the environment too
    "SQLSTATE[HY000]: General error: 1114 The table 'sylius_order' is full",
    "SQLSTATE[HY000]: General error: 1021 Disk full (/var/lib/mysql/x); waiting for someone to free some space...",
    "SQLSTATE[HY000] [1114] The table 'sylius_channel' is full",
    "fwrite(): Write of 8192 bytes failed with errno=28 No space left on device",
])
def test_db_connection_errors_in_the_log_are_infra_log(tmp_path, lanes, monkeypatch, line):
    root, e = entry_of(tmp_path)
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(inner={"cold": 0, "behat_exit": 1}, log=line)))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert (r["status"], r["infra_source"], r["behat_exit"]) == ("INFRA", "log", 1)


def test_the_log_scan_reads_the_first_mib_of_a_log_of_any_size(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    monkeypatch.setattr(runner, "FSIZE", 2 << 20)  # a log above the file cap is still scanned: its head
    line = "SQLSTATE[HY000] [2002] Connection refused\n"
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(inner={"cold": 0, "behat_exit": 1},
                                                              log=line + "x" * (3 << 20))))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert (r["status"], r.get("infra_source")) == ("INFRA", "log")
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(inner={"cold": 0, "behat_exit": 1},
                                                              log="y" * (1 << 20) + line)))
    assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "ERROR"  # beyond the first MiB: not read


@pytest.mark.parametrize("name, code, status", [
    ("trace.json", 0, "ERROR"), ("inner.json", 0, "ERROR"), ("behat.log", 1, "ERROR"), ("behat.log", 0, "PASS")])
def test_a_fifo_in_the_output_is_read_as_bad_output_not_a_hang(tmp_path, lanes, monkeypatch, name, code, status):
    # model code can mkfifo in /out: opening one for reading would block the harness forever
    root, e = entry_of(tmp_path)
    out = tmp_path / "o"

    def fifo(o):
        outputs(inner={"cold": 0, "behat_exit": code})(o)
        (o / name).unlink()
        os.mkfifo(o / name)
    monkeypatch.setattr(subprocess, "run", FakeDocker(fifo))
    r = no_hang(lambda: run_once(REC, root, e, 1, out), [out / name])
    assert r["status"] == status and "infra_source" not in r
    if name == "behat.log":
        assert r["log_sha256"] is None


def test_a_pass_is_never_infra_log(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    log = "Doctrine\\DBAL\\Exception\\ConnectionException: retried, reconnected"
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs(log=log)))
    assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "PASS"


def test_run_once_clears_read_only_leftovers(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    out = tmp_path / "o"
    (out / "sub" / "deep").mkdir(parents=True)
    (out / "sub" / "deep" / "f").write_text("x", encoding="utf-8")
    (out / "sub" / "deep").chmod(0o500)
    (out / "sub").chmod(0o000)
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs()))
    assert run_once(REC, root, e, 1, out)["status"] == "PASS"
    assert not (out / "sub").exists()


def test_run_once_is_infra_os_when_the_output_dir_cannot_be_cleared(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    (tmp_path / "o").mkdir()

    def refuse(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(runner.shutil, "rmtree", refuse)
    fake = FakeDocker(outputs())
    monkeypatch.setattr(subprocess, "run", fake)
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert (r["status"], r["infra_source"]) == ("INFRA", "os") and fake.calls == []


def test_output_directory_too_large_is_error(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    monkeypatch.setattr(runner, "MAX_OUT_BYTES", 10_000)
    (tmp_path / "outside").write_bytes(b"x" * 50_000)

    def big(out):
        outputs()(out)
        (out / "junk").write_bytes(b"x" * 20_000)
    monkeypatch.setattr(subprocess, "run", FakeDocker(big))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert r["status"] == "ERROR" and r["error"] == "output directory too large"

    def linked(out):  # a symlink counts as itself: the target is never followed
        outputs()(out)
        (out / "link").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(subprocess, "run", FakeDocker(linked))
    assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "PASS"


def test_too_many_files_in_the_output_is_error(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)
    monkeypatch.setattr(runner, "MAX_OUT_FILES", 50)

    def many(out):
        outputs()(out)
        (out / "junk").mkdir()
        for i in range(60):
            (out / "junk" / str(i)).touch()
    monkeypatch.setattr(subprocess, "run", FakeDocker(many))
    r = run_once(REC, root, e, 1, tmp_path / "o")
    assert r["status"] == "ERROR" and r["error"] == "too many files in the output directory"
    monkeypatch.setattr(subprocess, "run", FakeDocker(outputs()))  # the next run still clears them (no cap there)
    assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "PASS"


def test_the_output_walk_stops_at_the_file_cap(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    for i in range(200):
        (tmp_path / "d" / f"f{i}").touch()
    seen = []
    real = os.scandir

    class Spy:  # counts the directory entries the walk consumes
        def __init__(self, path):
            self.it = real(path)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.it.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.it)
            seen.append(entry.name)
            return entry

        def close(self):
            self.it.close()
    monkeypatch.setattr(os, "scandir", Spy)
    with pytest.raises(OutputError, match="too many files"):
        runner.tree_bytes(tmp_path, 50)
    monkeypatch.undo()
    assert len(seen) <= 52  # stopped early, not after listing all 201 entries
    everything = [tmp_path / "d", *(tmp_path / "d").iterdir()]
    assert runner.tree_bytes(tmp_path, 201) == sum(os.lstat(p).st_size for p in everything)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 directory")
def test_an_unreadable_output_directory_is_error(tmp_path, lanes, monkeypatch):
    root, e = entry_of(tmp_path)

    def locked(out):
        outputs()(out)
        (out / "junk").mkdir()
        (out / "junk" / "f").write_text("x", encoding="utf-8")
        (out / "junk").chmod(0o000)
    monkeypatch.setattr(subprocess, "run", FakeDocker(locked))
    try:
        r = run_once(REC, root, e, 1, tmp_path / "o")
        assert r["status"] == "ERROR" and r["error"].startswith("output directory unreadable"), r
        monkeypatch.setattr(subprocess, "run", FakeDocker(outputs()))  # cleared by the next run (chmod, then rmtree)
        assert run_once(REC, root, e, 1, tmp_path / "o")["status"] == "PASS"
    finally:
        if (tmp_path / "o" / "junk").exists():
            (tmp_path / "o" / "junk").chmod(0o700)  # pytest must be able to remove tmp_path


def test_spike_runner_uses_the_lane_user_hardening_and_the_lane_lock(tmp_path, lanes, monkeypatch):
    # er/er_run.py also works as a top-level module (er/ on sys.path, not the repo root)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "er"))
    monkeypatch.delitem(sys.modules, "er_run", raising=False)
    er_run = importlib.import_module("er_run")
    assert er_run.LANES == (1, 2, 3)
    monkeypatch.setattr(er_run, "find_record", lambda rid: REC)
    fake = FakeDocker(outputs())
    monkeypatch.setattr(subprocess, "run", fake)
    res = er_run.run("r", "gold", None, 3, tmp_path / "o", 60, False)
    (rm, _, rm_locked), (run, kw, run_locked) = fake.calls
    assert rm == ["docker", "rm", "-f", "gtestgen-er-lane3"] and rm_locked and run_locked
    assert run[run.index("--name") + 1] == "gtestgen-er-lane3" and run[run.index("--network") + 1] == db.NETWORK
    assert " ".join(runner.HARDENING) in " ".join(run) and kw["env"]["DATABASE_URL"] == db.db_url(3)
    assert no_secret(run) and res["status"] == "PASS" and kw["timeout"] == 120


def test_image_id_has_a_timeout(monkeypatch):
    seen = {}

    def fake(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0, "sha256:x\n", "")
    monkeypatch.setattr(subprocess, "run", fake)
    assert runner.image_id.__wrapped__("some/image") == "sha256:x" and seen["timeout"] == 30


# ---------- er.db with a fake mysql / docker ----------

USERS = [("er_lane1", "%"), ("er_lane2", "%"), ("er_lane3", "%"), ("mysql.infoschema", "localhost"),
         ("mysql.session", "localhost"), ("mysql.sys", "localhost"), ("root", "localhost")]


def grants(n: int) -> str:
    return (f"GRANT USAGE ON *.* TO `er_lane{n}`@`%`\n"
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON `sylius_lane{n}`.* TO `er_lane{n}`@`%`")


def fake_sql(users=USERS, extra=None):
    def sql(query: str) -> str:
        if "mysql.user" in query:
            return "\n".join(f"{u}\t{h}" for u, h in users)
        n = int(query.split("er_lane", 1)[1][0])
        return grants(n) + ("\n" + extra[n] if extra and n in extra else "")
    return sql


def test_check_accounts(monkeypatch):
    monkeypatch.setattr(db, "sql", fake_sql())
    db.check_accounts([1, 2, 3])
    for users in (USERS + [("root", "%")], USERS + [("er_migrate", "%")], USERS + [("er_lane1", "localhost")],
                  USERS + [("mallory", "%")], [u for u in USERS if u[0] != "er_lane2"]):
        monkeypatch.setattr(db, "sql", fake_sql(users))
        with pytest.raises(db.InfraError):
            db.check_accounts([1, 2])
    for extra in ("GRANT TRIGGER ON `sylius_lane3`.* TO `er_lane3`@`%`",
                  "GRANT SELECT ON `sylius_lane1`.* TO `er_lane3`@`%`"):
        monkeypatch.setattr(db, "sql", fake_sql(extra={3: extra}))
        with pytest.raises(db.InfraError, match="er_lane3"):
            db.check_accounts([1, 2])  # every lane user present is checked, not only the batch's lanes


def test_sql_is_root_over_the_socket_with_a_timeout(monkeypatch):
    seen = []

    def fake(cmd, **kw):
        seen.append((cmd, kw))
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    with pytest.raises(db.InfraError, match="no answer from docker within 30 s"):
        db.sql("select 1")
    cmd, kw = seen[0]
    assert cmd[:2] == ["docker", "exec"] and kw["timeout"] == 30 and len(seen) == 2  # retried once
    assert not any(a.startswith(("-h", "-p")) or a == "--host" for a in cmd[cmd.index("mysql"):])


def test_lane_connect_passes_the_password_by_name_only(tmp_path, lanes, monkeypatch):
    seen = []

    def fake(cmd, **kw):
        seen.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, "sha256:mysql\n" if cmd[1] == "inspect" else "1\n", "")
    monkeypatch.setattr(subprocess, "run", fake)
    db.lane_connect(2)
    (run, kw), = [(c, k) for c, k in seen if c[1] == "run"]
    assert run[run.index("MYSQL_PWD") - 1] == "-e" and all(no_secret(c) for c, _ in seen)  # name only on argv
    assert kw["env"]["MYSQL_PWD"] == "pw-secret-2" and kw["timeout"]
    assert run[run.index("--network") + 1] == db.NETWORK and "-hdb" in run and "-uer_lane2" in run


def test_free_bytes_reads_df_in_the_db_container(monkeypatch):
    seen = []
    df = ("Filesystem       1-blocks      Used Available Capacity Mounted on\n"
          "tmpfs          1073741824 240680960 833060864      23% /var/lib/mysql\n")

    def fake(cmd, **kw):
        seen.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, df, "")
    monkeypatch.setattr(subprocess, "run", fake)
    assert db.free_bytes() == 833060864
    assert seen[0][0] == ["docker", "exec", db.DB_CONTAINER, "df", "-P", "-B1", "/var/lib/mysql"]
    assert seen[0][1]["timeout"] == 30


def test_lane_healthy_never_raises_and_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr(db, "network_internal", lambda name=db.NETWORK: True)
    monkeypatch.setattr(db, "check_accounts", lambda lanes: calls.append(("accounts", tuple(lanes))))
    monkeypatch.setattr(db, "lane_connect", lambda lane: calls.append(("connect", lane)))
    monkeypatch.setattr(db, "check_lane", lambda lane: calls.append(("lane", lane)))
    monkeypatch.setattr(db, "free_bytes", lambda: db.MIN_FREE_BYTES)
    assert db.lane_healthy(1) is True
    assert db.preflight([1, 2]) == {"network": db.NETWORK, "schema_fingerprint": db.SCHEMA_FINGERPRINT, "lanes": [1, 2]}
    assert ("accounts", (1, 2)) in calls and ("lane", 2) in calls and ("connect", 2) in calls
    monkeypatch.setattr(db, "free_bytes", lambda: db.MIN_FREE_BYTES - 1)
    assert db.lane_healthy(1) is False
    with pytest.raises(db.InfraError, match="free"):
        db.preflight([1])
    monkeypatch.setattr(db, "free_bytes", lambda: db.MIN_FREE_BYTES)

    def broken(lane):
        raise db.InfraError("lane 1: schema fingerprint differs")
    monkeypatch.setattr(db, "check_lane", broken)
    assert db.lane_healthy(1) is False
    monkeypatch.setattr(db, "check_lane", lambda lane: None)
    monkeypatch.setattr(db, "lane_connect", lambda lane: 1 / 0)  # anything at all: still no exception
    assert db.lane_healthy(1) is False
    monkeypatch.setattr(db, "network_internal", lambda name=db.NETWORK: False)
    with pytest.raises(db.InfraError, match="internal"):
        db.preflight([1])


def test_check_lane_fails_on_low_free_space(monkeypatch):
    # the batch calls check_lane after every run that did not PASS: a full db tmpfs must stop it there
    monkeypatch.setattr(db, "table_count", lambda lane: db.N_TABLES)
    monkeypatch.setattr(db, "schema_fingerprint", lambda lane: db.SCHEMA_FINGERPRINT)
    monkeypatch.setattr(db, "free_bytes", lambda: db.MIN_FREE_BYTES)
    db.check_lane(1)
    monkeypatch.setattr(db, "free_bytes", lambda: db.MIN_FREE_BYTES - 1)
    with pytest.raises(db.InfraError, match="free"):
        db.check_lane(1)


def test_check_lane_says_how_to_reset(monkeypatch):
    # the hint resets only the broken lane (ER_RESET=1 alone would re-migrate every lane)
    monkeypatch.setattr(db, "table_count", lambda lane: 83)
    monkeypatch.setattr(db, "schema_fingerprint", lambda lane: "0" * 64)
    with pytest.raises(db.InfraError, match=re.escape("ER_LANES=2 ER_RESET=1 er/docker/init-db.sh")):
        db.check_lane(2)
    monkeypatch.setattr(db, "table_count", lambda lane: 0)
    with pytest.raises(db.InfraError, match=re.escape("ER_LANES=1 ER_RESET=1 er/docker/init-db.sh")):
        db.check_lane(1)


# ---------- er/docker/init-db.sh against a fake docker (the real one is not on PATH) ----------

INIT_DB = Path(__file__).resolve().parents[1] / "er" / "docker" / "init-db.sh"
FAKE_DOCKER = """#!/bin/sh
# logs every call; answers ps, compose up/config and inspect; everything else (SQL through compose exec) fails
printf '%s\\n' "$*" >> "$FAKE_LOG"
case "$*" in
  ps\\ *) [ -z "$FAKE_PS" ] || echo "$FAKE_PS"; exit 0 ;;
  compose\\ *\\ config\\ *) echo "db $FAKE_HASH"; exit 0 ;;
  compose\\ *\\ up\\ *) exit 0 ;;
  inspect\\ *) echo "$FAKE_LABEL"; exit 0 ;;
esac
exit 1
"""


@pytest.fixture
def init_db(tmp_path):
    """run(**env) -> (exit code, stderr, docker calls) of init-db.sh with ER_LANES=3, a fake docker, lane locks in
    tmp_path/locks and secrets in tmp_path/secrets."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER, encoding="utf-8")
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "python3").symlink_to(sys.executable)
    path = f"{bin_dir}:/usr/bin:/bin"
    assert shutil.which("docker", path=path) == str(bin_dir / "docker")
    (tmp_path / "locks").mkdir()
    log = tmp_path / "docker.log"

    def run(**env):
        base = {"PATH": path, "HOME": str(tmp_path), "ER_LANES": "3", "ER_LOCK_DIR": str(tmp_path / "locks"),
                "ER_SECRETS_DIR": str(tmp_path / "secrets"), "FAKE_LOG": str(log), "FAKE_PS": "",
                "FAKE_HASH": "h1", "FAKE_LABEL": "h1"}
        log.unlink(missing_ok=True)
        p = REAL_RUN([str(INIT_DB)], env={**base, **env}, capture_output=True, text=True, timeout=60)
        return p.returncode, p.stderr, log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return run


def touched_db(calls: list[str]) -> bool:
    return any(c.startswith("compose") for c in calls)


def test_init_db_refuses_while_a_lane_lock_is_held(init_db, tmp_path):
    with open(tmp_path / "locks" / "gtestgen-er-lane1.lock", "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # any lane: a batch on lane 1 while lane 3 is set up is refused too
        code, err, calls = init_db()
    assert code == 1 and "gtestgen-er-lane1.lock" in err and "ER_FORCE=1" in err, err
    assert not touched_db(calls), calls
    code, err, calls = init_db()  # the lock file is still there, but free: set-up goes on (the fake SQL then fails)
    assert "lock" not in err and touched_db(calls), (err, calls)


def test_init_db_refuses_while_a_lane_container_exists(init_db):
    code, err, calls = init_db(FAKE_PS="0123456789ab")
    assert code == 1 and "gtestgen-er-lane" in err and "ER_FORCE=1" in err, err
    assert calls == ["ps -aq -f name=^gtestgen-er-lane"]


def test_init_db_force_skips_the_refusals(init_db, tmp_path):
    with open(tmp_path / "locks" / "gtestgen-er-lane3.lock", "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        code, err, calls = init_db(ER_FORCE="1", FAKE_PS="0123456789ab")
    assert "ER_FORCE=1" in err and touched_db(calls), (err, calls)


def test_init_db_warns_when_the_db_container_predates_compose_yaml(init_db):
    code, err, calls = init_db(FAKE_HASH="h2", FAKE_LABEL="h1")
    assert "config-hash" in err and "--force-recreate" in err, err
    assert any(" exec " in c for c in calls), calls  # a warning only: the set-up went on to the SQL
    code, err, calls = init_db()
    assert "config-hash" not in err and any(" exec " in c for c in calls), (err, calls)
    code, err, calls = init_db(FAKE_HASH="")  # compose could not compute it: say that it was not compared
    assert "could not compute the config-hash" in err and any(" exec " in c for c in calls), (err, calls)


# ---------- integration (names contain "real_run": `-k "not real_run"` never touches Docker) ----------
# Docker, the trace image, lane ER_TEST_LANE (default 3) and repos/Sylius at c096ca3

HEAD = Path(__file__).resolve().parents[1] / "repos" / "Sylius"
CTX = "src/Sylius/Behat/Context/Api/Admin/ManagingZonesContext.php"
RID = "Sylius/Sylius|features/admin/addressing/managing_zones/adding_zone.feature|46|api_managing_zones"
LANE = int(os.environ.get("ER_TEST_LANE", "3"))


def docker_ready() -> bool:
    if shutil.which("docker") is None or not (HEAD / CTX).is_file():
        return False
    return REAL_RUN(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30).returncode == 0


def lane_mysql(user_lane: int, database: str, query: str, password: str | None = None) -> subprocess.CompletedProcess:
    """mysql as er_lane<user_lane> over TCP inside the db container; the password travels in the environment."""
    env = {**os.environ, "MYSQL_PWD": db.lane_password(user_lane) if password is None else password}
    return REAL_RUN(["docker", "exec", "-e", "MYSQL_PWD", db.DB_CONTAINER, "mysql", "-h127.0.0.1",
                     f"-uer_lane{user_lane}", database, "-e", query],
                    capture_output=True, text=True, timeout=30, env=env)


def pdo_from_er_network(dsn: str, user: str, password: str) -> subprocess.CompletedProcess:
    return REAL_RUN(["docker", "run", "--rm", "--network", db.NETWORK, IMAGE, "php", "-r",
                     f"new PDO('{dsn}', '{user}', '{password}'); echo 'connected';"],
                    capture_output=True, text=True, timeout=120)


@pytest.mark.skipif(not docker_ready(), reason="needs Docker, the ER trace image and repos/Sylius at c096ca3")
def test_real_run_head_neg_cold_m1_and_lane_isolation(tmp_path, monkeypatch):
    from er.records import load_records
    db.check_lane(LANE)  # 83 tables and the pinned schema fingerprint
    denied = lane_mysql(LANE, f"sylius_lane{LANE}", "CREATE TRIGGER t BEFORE INSERT ON sylius_channel "
                                                     "FOR EACH ROW SET @x = 1")
    assert denied.returncode != 0 and "denied" in denied.stderr
    other = 1 if LANE != 1 else 2
    cross = lane_mysql(LANE, f"sylius_lane{other}", "select count(*) from sylius_channel")
    assert cross.returncode != 0 and "denied" in cross.stderr  # a lane user cannot reach another lane's database
    record = load_records()[RID]
    head = (HEAD / CTX).read_text(encoding="utf-8")
    neg = "\n".join(line for line in head.split("\n")
                    if "I provide a too long zone member code" not in line
                    and "the provided zone member code is too long" not in line)
    run_dir = write_run(tmp_path, "it", [
        {"record": record, "condition": "neg", "slug": "z", "base": {CTX: neg}, "files": {}},
        {"record": record, "condition": "gold", "slug": "z", "base": {CTX: neg}, "files": {CTX: head}},
        {"record": record, "condition": "stub", "slug": "z", "base": {CTX: neg},
         "files": {CTX: head, "src/Sylius/Behat/Context/Api/Admin/ErNewHelper.php": "<?php\n"}}])
    e = {x.condition: x for x in load_manifest(run_dir)}
    seen = []

    def spy(cmd, **kw):
        seen.append(list(cmd))
        return REAL_RUN(cmd, **kw)
    monkeypatch.setattr(subprocess, "run", spy)
    assert run_once(record, tmp_path, e["neg"], LANE, tmp_path / "o_neg")["status"] == "UNDEFINED"
    gold = run_once(record, tmp_path, e["gold"], LANE, tmp_path / "o_gold")
    assert gold["status"] == "PASS" and gold["cold"] == 0 and gold["trace_sha256"]
    assert any(ev.get("route") for ev in gold["trace"]["events"] if ev["kind"] == "http")
    runs = [c for c in seen if c[:2] == ["docker", "run"]]
    assert runs and all(c[c.index("--name") + 1] == f"gtestgen-er-lane{LANE}" for c in runs)
    assert all(runner.image_id(IMAGE) in c and IMAGE not in c for c in runs)
    secret = db.lane_password(LANE)
    assert not any(secret in " ".join(c) for c in seen) and secret not in json.dumps(gold)
    assert secret.encode() not in (tmp_path / "o_gold" / "behat.log").read_bytes()
    cold = run_once(record, tmp_path, e["stub"], LANE, tmp_path / "o_new")  # a new file: container rebuilt
    assert cold["status"] == "PASS" and cold["cold"] == 1
    m1 = run_once(record, tmp_path, e["gold"], LANE, tmp_path / "o_m1", noop=(9,))
    assert [s["status"] for s in m1["trace"]["steps"]][9:] == ["noop", "passed", "failed"]
    assert db.sql("select 1") == "1"


@pytest.mark.skipif(not docker_ready(), reason="needs Docker, the ER trace image and repos/Sylius at c096ca3")
def test_real_run_root_and_old_passwords_are_denied_from_the_er_network():
    for dsn, user, password in (("mysql:host=db", "root", "root"),
                                (f"mysql:host=db;dbname=sylius_lane{LANE}", f"er_lane{LANE}", f"er_lane{LANE}")):
        r = pdo_from_er_network(dsn, user, password)
        assert r.returncode != 0 and "connected" not in r.stdout, (user, r.stdout)
        assert "[1045]" in r.stdout + r.stderr or "[1130]" in r.stdout + r.stderr, (user, r.stdout + r.stderr)
    db.check_accounts([1, 2, 3])
    assert db.lane_healthy(LANE)
    assert db.preflight([LANE])["schema_fingerprint"] == db.SCHEMA_FINGERPRINT


@pytest.mark.skipif(not docker_ready(), reason="needs Docker, the ER trace image and repos/Sylius at c096ca3")
def test_real_run_fingerprint_sees_an_extra_index():
    # lane-local on purpose: a trigger or view would also change the other lanes' fingerprints (global counts)
    db.check_lane(LANE)
    probe = (f"SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema = 'sylius_lane{LANE}' "
             f"AND table_name = 'sylius_channel' AND index_name = 'er_probe'")
    try:
        db.sql(f"CREATE INDEX er_probe ON sylius_lane{LANE}.sylius_channel (code, hostname)")
        with pytest.raises(db.InfraError, match="fingerprint"):
            db.check_lane(LANE)
        assert not db.lane_healthy(LANE)
    finally:  # also when the CREATE ran but was reported as failed (sql() retries a timed-out call)
        if db.sql(probe) != "0":
            db.sql(f"DROP INDEX er_probe ON sylius_lane{LANE}.sylius_channel")
    db.check_lane(LANE)


@pytest.mark.skipif(not docker_ready(), reason="needs Docker, the ER trace image and repos/Sylius at c096ca3")
def test_real_run_a_fatal_step_attribute_is_the_outputs_error_and_the_lane_stays_clean(tmp_path):
    # a malformed PHP-8 step attribute (e.g. #[Given()]) makes Behat 4.0.0-alpha1 hit a PHP Fatal error while it
    # loads contexts, before any scenario runs: behat_exit 255, junit has 0 testcases. classify() (er/runner.py)
    # does not treat 255 as TIMEOUT/OOM, and INFRA_LOG only matches a DB-connection error or a full disk, never a
    # PHP fatal, so this is the output's own ERROR, not INFRA: no retry, no batch stop, and the lane stays clean.
    from er.batch import run_retry
    from er.records import load_records
    record = load_records()[RID]
    head = (HEAD / CTX).read_text(encoding="utf-8")
    assert head.endswith("}\n") and "use Behat\\Step\\Given;" not in head
    bad_method = "    #[Given()]\n    public function probeFatalGivenAttribute(): void\n    {\n    }\n"
    malformed = head.replace("use Behat\\Step\\Then;\n", "use Behat\\Step\\Given;\nuse Behat\\Step\\Then;\n", 1)
    malformed = malformed[:-len("}\n")] + bad_method + "}\n"
    run_dir = write_run(tmp_path, "it", [
        {"record": record, "condition": "gold", "slug": "z", "base": {CTX: head}, "files": {CTX: head}},
        {"record": record, "condition": "m.s1.null_empty", "slug": "z", "base": {CTX: head},
         "files": {CTX: malformed}}])
    e = {x.condition: x for x in load_manifest(run_dir)}
    bad = run_once(record, tmp_path, e["m.s1.null_empty"], LANE, tmp_path / "o_bad")
    assert bad["status"] == "ERROR" and bad["error"] == "not exactly one scenario ran" and bad["behat_exit"] == 255
    assert "infra_source" not in bad
    db.check_lane(LANE)  # a Fatal at context load leaves the lane exactly as clean as any other run
    good = run_once(record, tmp_path, e["gold"], LANE, tmp_path / "o_good")
    assert good["status"] == "PASS"
    retried = run_retry(run_once, record, tmp_path, e["m.s1.null_empty"], LANE, tmp_path / "o_retry",
                        lane_healthy=db.lane_healthy)
    assert retried["status"] == "ERROR" and retried["attempts"] == 1


def test_sql_retries_a_failed_docker_exec_once(monkeypatch):
    calls = []

    def flaky(cmd, **kw):
        calls.append(cmd)
        rc = 1 if len(calls) == 1 else 0
        return subprocess.CompletedProcess(cmd, rc, "83\n" if rc == 0 else "", "Error response from daemon" if rc else "")
    monkeypatch.setattr(db.subprocess, "run", flaky)
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    assert db.sql("select 1") == "83" and len(calls) == 2

    def exits_1(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "Error response from daemon")

    def no_docker(cmd):
        raise OSError("docker: not found")
    for fail in (exits_1, no_docker):
        calls.clear()

        def dead(cmd, **kw):
            calls.append(cmd)
            return fail(cmd)
        monkeypatch.setattr(db.subprocess, "run", dead)
        with pytest.raises(db.InfraError, match="daemon|not found"):
            db.sql("select 1")
        assert len(calls) == 2  # one retry, not more
