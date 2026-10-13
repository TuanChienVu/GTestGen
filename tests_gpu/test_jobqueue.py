import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from v2gpu import jobqueue
from v2gpu.jobqueue import QueueError, run_queue, write_manifest

PY = sys.executable


def job(tmp_path, name, code, resume_args=None, max_hours=0.5):
    return {"name": name, "workdir": str(tmp_path / "runs" / name), "max_hours": max_hours,
            "cmd": [PY, "-c", code.replace("WD", str(tmp_path / "runs" / name))], "resume_args": resume_args or []}


def spec(tmp_path, jobs, deadline_minutes=120, stop=None):
    s = {"queue_dir": str(tmp_path / "queue"), "lock_path": str(tmp_path / "pod.lock"), "jobs": jobs, "self_stop_cmd": stop,
         "deadline_utc": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=deadline_minutes)).isoformat()}
    p = tmp_path / "queue.json"
    p.write_text(json.dumps(s))
    return p


WRITE = "import pathlib,sys; p=pathlib.Path('WD'); p.mkdir(parents=True, exist_ok=True); (p/'out.txt').write_text('ok ' + ' '.join(sys.argv[1:]))"


def test_success_writes_a_manifest_that_shasum_accepts(tmp_path):
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)]), poll=0.05) == 0
    wd = tmp_path / "runs" / "a"
    assert (tmp_path / "queue" / "a.status").read_text() == "DONE\n"
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd).returncode == 0
    (wd / "out.txt").write_text("tampered")
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd,
                          capture_output=True).returncode != 0


def test_manifest_skips_state(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "trainer.pt").write_text("x")
    (tmp_path / "epoch-1").mkdir()
    (tmp_path / "epoch-1" / "a.safetensors").write_text("y")
    assert write_manifest(tmp_path).read_text().split()[1::2] == ["epoch-1/a.safetensors"]


def test_failure_stops_the_queue(tmp_path):
    jobs = [job(tmp_path, "a", "raise SystemExit(1)"), job(tmp_path, "b", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    assert (tmp_path / "queue" / "a.status").read_text() == "FAILED\n"
    assert not (tmp_path / "runs" / "b").exists()
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1   # a rerun does not step past a failed job
    assert not (tmp_path / "runs" / "b").exists()


def test_deadline_terminates_the_running_job(tmp_path, monkeypatch):
    """The start rule makes this a safety net (a job only starts with max_hours left), so the clock is jumped."""
    from v2gpu import jobqueue

    real, calls = jobqueue._now, []

    def clock():
        calls.append(1)
        return real() + dt.timedelta(days=1 if len(calls) > 3 else 0)

    monkeypatch.setattr(jobqueue, "_now", clock)
    jobs = [job(tmp_path, "slow", "import time; time.sleep(30)"), job(tmp_path, "b", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05, grace=1) == 1
    assert (tmp_path / "queue" / "slow.status").read_text() == "DEADLINE\n"
    assert not (tmp_path / "runs" / "b").exists()


def test_done_jobs_are_skipped_and_state_triggers_resume_args(tmp_path):
    jobs = [job(tmp_path, "a", WRITE), job(tmp_path, "b", WRITE, resume_args=["--resume"])]
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "a.status").write_text("DONE\n")
    (tmp_path / "queue" / "a.spec-sha256").write_text(jobqueue._spec_sha(jobs[0]))  # matches this spec
    (tmp_path / "runs" / "b" / "state").mkdir(parents=True)
    (tmp_path / "runs" / "b" / "state" / "pos.json").write_text("{}")
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    assert not (tmp_path / "runs" / "a").exists()
    assert (tmp_path / "runs" / "b" / "out.txt").read_text() == "ok --resume"


def test_self_stop_runs_last_with_the_pod_id(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    stop = [PY, "-c", f"import sys,pathlib; pathlib.Path(r'{tmp_path}/stopped').write_text(sys.argv[1])", "$RUNPOD_POD_ID"]
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", "raise SystemExit(1)")], stop=stop), poll=0.05) == 1
    assert (tmp_path / "stopped").read_text() == "pod123"   # also after a failure


def test_self_stop_with_unresolved_variable_is_refused_before_any_job(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    stop = [PY, "-c", f"import pathlib; pathlib.Path(r'{tmp_path}/stopped').write_text('x')", "$RUNPOD_POD_ID"]
    with pytest.raises(QueueError, match="unresolved variable"):
        run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)], stop=stop), poll=0.05)
    assert not (tmp_path / "runs" / "a").exists()
    assert not (tmp_path / "stopped").exists()


def test_self_stop_with_missing_command_is_refused_before_any_job(tmp_path):
    stop = ["no-such-cmd-xyz"]
    with pytest.raises(QueueError, match="not found"):
        run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)], stop=stop), poll=0.05)
    assert not (tmp_path / "runs" / "a").exists()


def test_a_second_queue_on_the_same_dir_is_refused(tmp_path):
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    held = (tmp_path / "pod.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)      # the first queue, still running
    with pytest.raises(QueueError, match="already running"):
        run_queue(p, poll=0.05)
    assert not (tmp_path / "runs" / "a").exists()
    fcntl.flock(held, fcntl.LOCK_UN)
    assert run_queue(p, poll=0.05) == 0


def test_deadline_without_timezone_is_refused_before_any_job(tmp_path):
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    s = json.loads(p.read_text())
    s["deadline_utc"] = "2030-01-01T00:00:00"
    p.write_text(json.dumps(s))
    with pytest.raises(QueueError, match="timezone"):
        run_queue(p, poll=0.05)
    assert not (tmp_path / "runs" / "a").exists()


def test_every_job_needs_max_hours(tmp_path):
    j = job(tmp_path, "a", WRITE)
    del j["max_hours"]
    with pytest.raises(QueueError, match="max_hours"):
        run_queue(spec(tmp_path, [j]), poll=0.05)


def test_a_job_that_cannot_finish_before_the_deadline_is_not_started(tmp_path):
    jobs = [job(tmp_path, "a", WRITE), job(tmp_path, "big", WRITE, max_hours=2.0)]
    assert run_queue(spec(tmp_path, jobs, deadline_minutes=60), poll=0.05) == 1
    assert (tmp_path / "queue" / "a.status").read_text() == "DONE\n"
    assert (tmp_path / "queue" / "big.status").read_text() == "NO_TIME\n"
    assert not (tmp_path / "runs" / "big").exists()


def test_a_job_running_past_its_max_hours_is_terminated(tmp_path):
    jobs = [job(tmp_path, "slow", "import time; time.sleep(30)", max_hours=0.3 / 3600), job(tmp_path, "b", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05, grace=1) == 1
    assert (tmp_path / "queue" / "slow.status").read_text() == "TIMEOUT\n"
    assert not (tmp_path / "runs" / "b").exists()


# For the run manifest of 3C/3D ("Evidence from failed jobs"): every terminal status gets
# workdir/MANIFEST.sha256, not only DONE, so copy_out.sh can copy a failed workdir with its checksums


def test_a_failed_job_that_wrote_partial_output_still_gets_a_manifest(tmp_path):
    partial = "import pathlib; p=pathlib.Path('WD'); p.mkdir(parents=True, exist_ok=True); " \
             "(p/'partial.txt').write_text('x'); raise SystemExit(1)"
    jobs = [job(tmp_path, "a", partial)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    wd = tmp_path / "runs" / "a"
    assert (tmp_path / "queue" / "a.status").read_text() == "FAILED\n"
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd).returncode == 0
    assert (wd / "MANIFEST.sha256").read_text().split()[1::2] == ["partial.txt"]


def test_a_timed_out_job_that_wrote_partial_output_still_gets_a_manifest(tmp_path):
    partial = "import pathlib,time; p=pathlib.Path('WD'); p.mkdir(parents=True, exist_ok=True); " \
             "(p/'partial.txt').write_text('x'); time.sleep(30)"
    jobs = [job(tmp_path, "slow", partial, max_hours=0.3 / 3600), job(tmp_path, "b", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05, grace=1) == 1
    wd = tmp_path / "runs" / "slow"
    assert (tmp_path / "queue" / "slow.status").read_text() == "TIMEOUT\n"
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd).returncode == 0


def test_a_job_cut_by_the_deadline_that_wrote_partial_output_still_gets_a_manifest(tmp_path, monkeypatch):
    """The clock only jumps past the deadline once the child has actually written its marker file, so this never
    races the child process's own startup time (unlike a jump on a fixed call count)."""
    real = jobqueue._now
    marker = tmp_path / "runs" / "slow" / "partial.txt"

    def clock():
        return real() + (dt.timedelta(days=1) if marker.exists() else dt.timedelta(0))

    monkeypatch.setattr(jobqueue, "_now", clock)
    partial = "import pathlib,time; p=pathlib.Path('WD'); p.mkdir(parents=True, exist_ok=True); " \
             "(p/'partial.txt').write_text('x'); time.sleep(30)"
    jobs = [job(tmp_path, "slow", partial), job(tmp_path, "b", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05, grace=1) == 1
    wd = tmp_path / "runs" / "slow"
    assert (tmp_path / "queue" / "slow.status").read_text() == "DEADLINE\n"
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd).returncode == 0


def test_a_no_time_job_with_a_prior_partial_workdir_still_gets_a_manifest(tmp_path):
    """NO_TIME cuts the job before it (re)starts, but a resumed job's workdir can already hold evidence from an
    earlier attempt (e.g. a receipt-generate.json left CRASHED); that evidence must still be manifested."""
    jobs = [job(tmp_path, "a", WRITE), job(tmp_path, "big", WRITE, max_hours=2.0)]
    wd = tmp_path / "runs" / "big"
    wd.mkdir(parents=True)
    (wd / "receipt-generate.json").write_text('{"status": "CRASHED"}')
    assert run_queue(spec(tmp_path, jobs, deadline_minutes=60), poll=0.05) == 1
    assert (tmp_path / "queue" / "big.status").read_text() == "NO_TIME\n"
    assert subprocess.run(["shasum", "-a", "256", "-c", "MANIFEST.sha256"], cwd=wd).returncode == 0


def test_write_manifest_of_a_workdir_that_was_never_created_is_a_no_op(tmp_path):
    """A job whose command never even created its workdir (NO_TIME, or the command itself could not start) is not a
    crash: there is nothing to manifest."""
    assert write_manifest(tmp_path / "never-created") is None
    assert not (tmp_path / "never-created").exists()


def test_exit_0_with_no_workdir_is_written_failed_not_done(tmp_path):
    """A job that exits 0 without ever creating its own workdir (a bug, or a
    cmd/workdir mismatch) used to be written DONE anyway -- copy_out.sh then finds nothing home for it, on the
    laptop, well after the pod already stopped. Fail closed instead: FAILED, and the queue stops (BLOCKED_BY_ on a
    rerun), same as any other real failure. Non-DONE terminal statuses (NO_TIME, FAILED, DEADLINE, TIMEOUT) already
    keep write_manifest's no-op for a missing workdir; only the success path is fixed here."""
    jobs = [job(tmp_path, "a", "import sys; sys.exit(0)")]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    assert (tmp_path / "queue" / "a.status").read_text() == "FAILED\n"
    assert not (tmp_path / "runs" / "a").exists()
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1        # a rerun does not step past it either
    assert (tmp_path / "queue" / "a.status").read_text() == "FAILED\n"


# The copy window
# must never get NO_TIME, or the pod self-stops before the last workdir is copied home


def test_the_copy_window_is_exempt_from_no_time(tmp_path):
    """The exact repro: deadline +35 min, one fast job, a copy-window job whose OWN max_hours (0.6 h = 36 min) is more
    than what remains after the fast job. Before the fix this always wrote NO_TIME_FOR_copy-window-x and self-stopped
    the pod before the window ran; a job named copy-window-* is now exempt from the NO_TIME pre-check."""
    jobs = [job(tmp_path, "gen-x", WRITE), job(tmp_path, "copy-window-x", WRITE, max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs, deadline_minutes=35), poll=0.05) == 0
    assert (tmp_path / "queue" / "gen-x.status").read_text() == "DONE\n"
    assert (tmp_path / "queue" / "copy-window-x.status").read_text() == "DONE\n"
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "ALL_DONE"


def test_the_deadline_still_cuts_the_copy_window(tmp_path):
    """The exemption is not a blank check: a copy-window job that overruns is still terminated at the deadline, same
    as any other job (it just always gets to START, unlike before)."""
    jobs = [job(tmp_path, "copy-window-x", "import time; time.sleep(5)", max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs, deadline_minutes=0.02), poll=0.05, grace=1) == 1
    assert (tmp_path / "queue" / "copy-window-x.status").read_text() == "DEADLINE\n"


# EVERY way the queue can end -- success, FAILED, a TIMEOUT, or a
# controlled refusal (the same as a FAILED job to jobqueue) -- must still run a trailing copy-window-* job before
# self_stop_cmd fires, not only the ALL_DONE path. Before 28/09/2026, a FAILED/DEADLINE/TIMEOUT/NO_TIME job stopped
# _jobs() immediately: a trailing copy-window-* job never even started, so self_stop_cmd could fire while a
# stopped-mid-run job's own receipt/log was still only on the (about to be unreachable) pod.


@pytest.mark.parametrize("ending, code, outcome_prefix", [
    ("failed", "raise SystemExit(1)", "FAILED_a"),
    ("timeout", "import time; time.sleep(5)", "TIMEOUT_a"),
])
def test_a_stopped_job_still_lets_a_trailing_copy_window_run(tmp_path, ending, code, outcome_prefix):
    max_hours = 0.0005 if ending == "timeout" else 0.5      # ~1.8 s, well under the sleep(5) above
    jobs = [job(tmp_path, "a", code, max_hours=max_hours), job(tmp_path, "copy-window-b", WRITE, max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05, grace=1) == 1        # not ALL_DONE: outcome is not success
    assert (tmp_path / "queue" / "a.status").read_text() == f"{outcome_prefix.split('_', 1)[0]}\n"
    assert (tmp_path / "queue" / "copy-window-b.status").read_text() == "DONE\n"    # ran anyway
    assert (tmp_path / "runs" / "copy-window-b" / "out.txt").exists()
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == outcome_prefix     # the FIRST non-DONE reason is still what is reported


def test_success_still_works_exactly_as_before_all_done(tmp_path):
    """The success ending (ALL_DONE) of the same "every ending" rule -- unchanged, listed
    here so one parametrized-style trio (success, FAILED, TIMEOUT) covers the rule in one place (the fourth ending, a
    probe's own controlled
    REFUSED_WOULD_NOT_FIT, is exactly the "failed" case above from jobqueue's point of view -- see
    tests_gpu/test_train_probe.py: a REAL v2gpu.jobqueue.run_queue with a stand-in job that exits with the probe's
    refusal code 5, not v2gpu.train_probe itself)."""
    jobs = [job(tmp_path, "a", WRITE), job(tmp_path, "copy-window-b", WRITE, max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    assert (tmp_path / "queue" / "a.status").read_text() == "DONE\n"
    assert (tmp_path / "queue" / "copy-window-b.status").read_text() == "DONE\n"
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["outcome"] == "ALL_DONE"


# The fix is scoped to copy-window-* only: test_failure_stops_the_queue (above) already pins that a plain FAILED
# job still never lets a plain LATER job (one not named copy-window-*) run -- unchanged.


def test_two_copy_windows_after_a_failure_both_still_run(tmp_path):
    """Not just one: every later copy-window-* job runs, in order, exactly like the ALL_DONE path always could."""
    jobs = [job(tmp_path, "a", "raise SystemExit(1)"), job(tmp_path, "copy-window-b", WRITE, max_hours=0.6),
            job(tmp_path, "copy-window-c", WRITE, max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    assert (tmp_path / "queue" / "copy-window-b.status").read_text() == "DONE\n"
    assert (tmp_path / "queue" / "copy-window-c.status").read_text() == "DONE\n"
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["outcome"] == "FAILED_a"


# A copy window runs after a failed job and is recorded DONE; the recovery after a job crash (delete the failed
# job's .status, rerun the same queue so the job resumes) then found the window already DONE, skipped it, reached ALL_DONE and self-stopped the pod before COPY_OK.
# A copy-window-* job is an idempotent keep-alive: its earlier terminal status never ends it, it always runs again.

FAIL_ONCE = ("import pathlib,sys; m=pathlib.Path('WD').parent/'MARK'; p=pathlib.Path('WD'); "
             "p.mkdir(parents=True, exist_ok=True)\nif not m.exists(): m.write_text('x'); sys.exit(1)\n"
             "(p/'out.txt').write_text('ok')")
COUNT = ("import pathlib; p=pathlib.Path('WD'); p.mkdir(parents=True, exist_ok=True); "
         "c=p.parent.parent/'COUNT'; c.write_text(c.read_text()+'x' if c.exists() else 'x')")


@pytest.mark.parametrize("window", ["copy-window-train", "copy-window-gen"])    # TRAIN's and GEN's queue shapes
def test_the_copy_window_runs_again_on_the_recovery_rerun(tmp_path, window):
    jobs = [job(tmp_path, "main", FAIL_ONCE), job(tmp_path, window, COUNT, max_hours=0.6)]
    count = tmp_path / "COUNT"
    assert run_queue(spec(tmp_path, jobs, stop=["true"]), poll=0.05) == 1          # run 1: main FAILED
    assert (tmp_path / "queue" / "main.status").read_text() == "FAILED\n"
    assert (tmp_path / "queue" / f"{window}.status").read_text() == "DONE\n" and count.read_text() == "x"
    (tmp_path / "queue" / "main.status").unlink()                                     # the sheets' recovery step
    assert run_queue(spec(tmp_path, jobs, stop=["true"]), poll=0.05) == 0          # run 2: main resumes, DONE
    assert (tmp_path / "queue" / "main.status").read_text() == "DONE\n"
    assert count.read_text() == "xx"                                   # the window ran AGAIN before the self-stop
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["outcome"] == "ALL_DONE"


def test_an_earlier_timeout_of_the_copy_window_does_not_block_the_rerun(tmp_path):
    """A window cut at its own max_hours (TIMEOUT) on run 1 must not turn into BLOCKED_BY_ on the rerun either."""
    jobs = [job(tmp_path, "main", WRITE), job(tmp_path, "copy-window-train", COUNT, max_hours=0.6)]
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "copy-window-train.status").write_text("TIMEOUT\n")
    from v2gpu.jobqueue import _spec_sha
    (tmp_path / "queue" / "copy-window-train.spec-sha256").write_text(_spec_sha(jobs[1]))
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    assert (tmp_path / "COUNT").read_text() == "x"
    assert (tmp_path / "queue" / "copy-window-train.status").read_text() == "DONE\n"


def test_a_copy_window_whose_spec_changed_is_still_refused(tmp_path):
    """Always running again is not a blank check: the spec binding still applies to a copy window."""
    jobs = [job(tmp_path, "copy-window-train", COUNT, max_hours=0.6)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    changed = [job(tmp_path, "copy-window-train", COUNT + "\n", max_hours=0.7)]
    assert run_queue(spec(tmp_path, changed), poll=0.05) == 1
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["outcome"] == "SPEC_CHANGED_copy-window-train"


# A terminal status must be bound to the job
# spec that earned it, or a job whose cmd changed under the same name silently inherits an old DONE/FAILED/etc.


def test_a_done_job_whose_cmd_changed_under_the_same_name_is_refused(tmp_path):
    jobs = [job(tmp_path, "a", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    assert (tmp_path / "queue" / "a.status").read_text() == "DONE\n"
    out_before = (tmp_path / "runs" / "a" / "out.txt").read_text()
    changed = [job(tmp_path, "a", WRITE + "\nraise SystemExit(1)")]   # same name, a different cmd
    assert run_queue(spec(tmp_path, changed), poll=0.05) == 1
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "SPEC_CHANGED_a"
    assert (tmp_path / "runs" / "a" / "out.txt").read_text() == out_before      # neither skipped-as-done nor re-run


@pytest.mark.parametrize("bad_status", ["FAILED", "DEADLINE", "TIMEOUT", "NO_TIME"])
def test_a_changed_spec_under_another_terminal_status_is_also_refused(tmp_path, bad_status):
    jobs = [job(tmp_path, "a", WRITE)]
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "a.status").write_text(bad_status + "\n")
    (tmp_path / "queue" / "a.spec-sha256").write_text("not-the-real-sha")
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "SPEC_CHANGED_a"
    assert not (tmp_path / "runs" / "a").exists()


# Since 27/09/2026 a status file with no .spec-sha256 sidecar at all (a legacy queue
# dir from before this check) is a distinct, more precise outcome than "the spec changed" -- the sha was never recorded


@pytest.mark.parametrize("bad_status", ["DONE", "FAILED", "DEADLINE", "TIMEOUT", "NO_TIME"])
def test_a_legacy_status_with_no_sha_sidecar_gives_its_own_outcome(tmp_path, bad_status):
    jobs = [job(tmp_path, "a", WRITE)]
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "a.status").write_text(bad_status + "\n")     # no a.spec-sha256 beside it
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 1
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "SPEC_SHA_MISSING_a"                   # fails closed, same as SPEC_CHANGED
    assert not (tmp_path / "runs" / "a").exists()
    assert (tmp_path / "queue" / "a.status").read_text() == bad_status + "\n"   # the evidence is left untouched


def test_the_sha_sidecar_is_written_before_the_status(tmp_path, monkeypatch):
    """So a crash between the two writes leaves "no status at all" (a fresh job on the next run), never "a status
    with no sha" (which would need its own SPEC_SHA_MISSING handling on THIS same run instead)."""
    from v2gpu import jobqueue

    order = []
    real_write = Path.write_text

    def traced(self, *a, **kw):
        if self.name.endswith(".status"):
            order.append("status")
        elif self.name.endswith(".spec-sha256"):
            order.append("spec-sha256")
        return real_write(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", traced)
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)]), poll=0.05) == 0
    assert "spec-sha256" in order and "status" in order
    assert order.index("spec-sha256") < order.index("status")


def test_an_unchanged_spec_is_skipped_as_before(tmp_path):
    """The other side of the spec binding: rerunning the identical queue file after DONE still just skips, as always."""
    jobs = [job(tmp_path, "a", WRITE)]
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0
    out_before = (tmp_path / "runs" / "a" / "out.txt").read_text()
    assert run_queue(spec(tmp_path, jobs), poll=0.05) == 0          # a plain rerun of the same spec
    assert (tmp_path / "runs" / "a" / "out.txt").read_text() == out_before


def stop_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    return [PY, "-c", f"import pathlib; pathlib.Path(r'{tmp_path}/stopped').write_text('x')"]


def test_a_command_that_cannot_start_fails_the_job_and_still_self_stops(tmp_path, monkeypatch):
    j = job(tmp_path, "a", WRITE)
    j["cmd"] = [str(tmp_path / "no-such-binary")]
    assert run_queue(spec(tmp_path, [j], stop=stop_marker(tmp_path, monkeypatch)), poll=0.05) == 1
    assert (tmp_path / "queue" / "a.status").read_text() == "FAILED\n"
    assert (tmp_path / "stopped").exists() and (tmp_path / "queue" / "outcome.json").exists()


def test_any_crash_of_the_queue_still_self_stops(tmp_path, monkeypatch):
    """write_manifest no longer raises on a workdir a job never created (it is a no-op,
    None), so this crash is made explicit instead of relying on that former, incidental behaviour."""
    def boom(wd):
        raise RuntimeError("boom")

    monkeypatch.setattr(jobqueue, "write_manifest", boom)
    j = job(tmp_path, "a", "pass")
    assert run_queue(spec(tmp_path, [j], stop=stop_marker(tmp_path, monkeypatch)), poll=0.05) == 1
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "CRASHED" and "error" in outcome and (tmp_path / "stopped").exists()


def test_one_lock_for_the_whole_pod(tmp_path):
    """Two queue files (canary, fuse) on one pod share lock_path: the second is refused while the first holds it."""
    lock = tmp_path / "pod.lock"
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    s = json.loads(p.read_text())
    s["lock_path"] = str(lock)
    p.write_text(json.dumps(s))
    held = lock.open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(QueueError, match="already running"):
        run_queue(p, poll=0.05)
    fcntl.flock(held, fcntl.LOCK_UN)


def test_jobs_inherit_the_lock_and_run_in_their_own_process_group(tmp_path, monkeypatch):
    import subprocess as sp

    seen = {}
    real = sp.Popen

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(sp, "Popen", spy)
    run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)]), poll=0.05)
    assert seen["start_new_session"] is True and len(seen["pass_fds"]) == 1


def test_timeout_kills_the_whole_process_group(tmp_path):
    import os
    import time

    pidfile = tmp_path / "grandchild.pid"
    code = ("import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            f"pathlib.Path(r'{pidfile}').write_text(str(p.pid)); time.sleep(60)")
    assert run_queue(spec(tmp_path, [job(tmp_path, "slow", code, max_hours=1.0 / 3600)]), poll=0.05, grace=1) == 1
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the grandchild survived the TIMEOUT")


def test_any_complete_state_triggers_resume_args(tmp_path):
    j = job(tmp_path, "b", WRITE, resume_args=["--resume"])
    (tmp_path / "runs" / "b" / "state.tmp").mkdir(parents=True)       # pod died inside the state swap
    (tmp_path / "runs" / "b" / "state.tmp" / "pos.json").write_text("{}")
    assert run_queue(spec(tmp_path, [j]), poll=0.05) == 0
    assert (tmp_path / "runs" / "b" / "out.txt").read_text() == "ok --resume"


def test_a_job_whose_receipt_says_done_is_not_run_again(tmp_path):
    wd = tmp_path / "runs" / "a"
    wd.mkdir(parents=True)
    (wd / "receipt-train.json").write_text(json.dumps({"status": "DONE"}))   # finished; the queue died before DONE
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", "raise SystemExit(1)")]), poll=0.05) == 0
    assert (tmp_path / "queue" / "a.status").read_text() == "DONE\n" and (wd / "MANIFEST.sha256").exists()


def test_manifest_skips_nested_state(tmp_path):
    (tmp_path / "deepseek-4096" / "run" / "state").mkdir(parents=True)
    (tmp_path / "deepseek-4096" / "run" / "state" / "trainer.pt").write_text("x")
    (tmp_path / "deepseek-4096" / "run" / "receipt-train.json").write_text("{}")
    assert write_manifest(tmp_path).read_text().split()[1::2] == ["deepseek-4096/run/receipt-train.json"]


def test_lock_path_is_required(tmp_path):
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    s = json.loads(p.read_text())
    del s["lock_path"]
    p.write_text(json.dumps(s))
    with pytest.raises(QueueError, match="lock_path"):
        run_queue(p, poll=0.05)


def _gone(pid, seconds=10.0):
    import os
    import time

    for _ in range(int(seconds * 10)):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_stragglers_of_a_finished_job_are_reaped(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    code = ("import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            f"pathlib.Path(r'{pidfile}').write_text(str(p.pid)); " + WRITE)
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", code)]), poll=0.05) == 0
    assert _gone(int(pidfile.read_text())), "a grandchild outlived its finished job"


def test_a_restarted_queue_is_refused_while_an_orphaned_job_runs(tmp_path):
    """End to end: the queue is killed while its job runs; the job still holds the pod lock."""
    import os
    import signal
    import time

    pidfile = tmp_path / "job.pid"
    code = f"import os,pathlib,time; pathlib.Path(r'{pidfile}').write_text(str(os.getpid())); time.sleep(4)"
    first = spec(tmp_path, [job(tmp_path, "a", code)])
    q = subprocess.Popen([PY, "-c", f"from v2gpu.jobqueue import run_queue; run_queue({str(first)!r}, poll=0.05)"])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.05)
    q.send_signal(signal.SIGKILL)
    q.wait()
    second = tmp_path / "second.json"
    s = json.loads(first.read_text())
    s["queue_dir"], s["jobs"] = str(tmp_path / "queue2"), [job(tmp_path, "b", WRITE)]
    second.write_text(json.dumps(s))
    with pytest.raises(QueueError, match="already running"):
        run_queue(second, poll=0.05)
    assert _gone(int(pidfile.read_text()))
    assert run_queue(second, poll=0.05) == 0 and os.path.exists(tmp_path / "runs" / "b" / "out.txt")


# The laptop loop must see a refused start; the self-stop cannot hang


@pytest.mark.parametrize("case, words", [("no timezone", "timezone"), ("no max_hours", "max_hours"),
                                         ("unresolved", "unresolved variable"), ("missing", "not found")])
def test_a_queue_refused_at_start_says_so_in_its_queue_dir(tmp_path, monkeypatch, case, words):
    """Nothing ran and nothing will: outcome.json says REFUSED_AT_START with the reason, so the laptop's copy loop
    stops at once instead of polling empty statuses. The pod is NOT stopped here (a refused start is fixed on the live
    pod; a stop on Community stock may not come back); the on-pod watchdog caps an unattended pod."""
    j = job(tmp_path, "a", WRITE)
    stop = stop_marker(tmp_path, monkeypatch)
    p = spec(tmp_path, [j], stop=stop)
    s = json.loads(p.read_text())
    if case == "no timezone":
        s["deadline_utc"] = "2030-01-01T00:00:00"
    elif case == "no max_hours":
        del s["jobs"][0]["max_hours"]
    elif case == "unresolved":
        monkeypatch.delenv("RUNPOD_POD_ID")
        s["self_stop_cmd"] = stop + ["$RUNPOD_POD_ID"]
    else:
        s["self_stop_cmd"] = ["no-such-cmd-xyz"]
    p.write_text(json.dumps(s))
    with pytest.raises(QueueError, match=words):
        run_queue(p, poll=0.05)
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert outcome["outcome"] == "REFUSED_AT_START" and words in outcome["error"]
    assert "watchdog" in outcome["self_stop"] and not (tmp_path / "stopped").exists()
    assert not (tmp_path / "runs" / "a").exists()


def test_a_queue_refused_because_another_runs_leaves_that_queues_files_alone(tmp_path):
    """The lock is held: the other queue owns queue_dir and the pod; nothing is written, nothing is stopped."""
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    held = (tmp_path / "pod.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    s = json.loads(p.read_text())
    s["deadline_utc"] = "2030-01-01T00:00:00"                    # even with a refusable spec
    p.write_text(json.dumps(s))
    with pytest.raises(QueueError, match="already running"):
        run_queue(p, poll=0.05)
    assert not (tmp_path / "queue" / "outcome.json").exists()
    fcntl.flock(held, fcntl.LOCK_UN)


def test_a_hanging_self_stop_is_cut_and_tried_once_more(tmp_path, monkeypatch):
    """runpodctl hanging on the network must not hold the queue (and the pod) forever: each try has a timeout, a
    second try follows, and self_stop.log records both."""
    import time

    from v2gpu import jobqueue

    monkeypatch.setattr(jobqueue, "SELF_STOP_TIMEOUT", 0.5)
    stop = [PY, "-c", "import time; time.sleep(30)"]
    t0 = time.monotonic()
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)], stop=stop), poll=0.05) == 0
    assert time.monotonic() - t0 < 10
    log = (tmp_path / "queue" / "self_stop.log").read_text()
    assert log.count("timed out after 0.5 s") == 2 and "try 1" in log and "try 2" in log


def test_a_failed_self_stop_is_tried_once_more(tmp_path, monkeypatch):
    """A transient API error (exit 1) gets a second try; the log keeps both results."""
    marker = tmp_path / "tries"
    code = (f"import pathlib,sys; p=pathlib.Path(r'{marker}'); n=len(p.read_text()) if p.exists() else 0; "
            f"p.write_text('x'*(n+1)); sys.exit(1 if n == 0 else 0)")
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)], stop=[PY, "-c", code]), poll=0.05) == 0
    log = (tmp_path / "queue" / "self_stop.log").read_text()
    assert marker.read_text() == "xx" and "try 1" in log and "-> 1" in log and "try 2" in log and "-> 0" in log


def test_a_self_stop_that_works_is_run_once(tmp_path, monkeypatch):
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)], stop=stop_marker(tmp_path, monkeypatch)),
                     poll=0.05) == 0
    log = (tmp_path / "queue" / "self_stop.log").read_text()
    assert "try 1" in log and "try 2" not in log


# What the laptop watcher reads is always the current queue's


def test_a_stale_outcome_is_moved_aside_when_a_queue_starts(tmp_path, monkeypatch):
    """An outcome.json left by an earlier attempt (e.g. REFUSED_AT_START) must never read as the current queue's: it
    is kept under outcome.<utc>.json, and the queue writes its own."""
    (tmp_path / "queue").mkdir()
    old = {"outcome": "REFUSED_AT_START", "error": "earlier attempt"}
    (tmp_path / "queue" / "outcome.json").write_text(json.dumps(old))
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)]), poll=0.05) == 0
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["outcome"] == "ALL_DONE"
    kept = sorted((tmp_path / "queue").glob("outcome.*.json"))
    assert len(kept) == 1 and json.loads(kept[0].read_text()) == old


def test_a_queue_marks_its_start_and_says_whether_a_self_stop_follows(tmp_path, monkeypatch):
    """started.json (its mtime is the queue start the watcher's grace counts from) is written once the pod lock is
    held; outcome.json says whether self_stop_cmd follows it (the watcher then waits for the pod to stop)."""
    before = dt.datetime.now(dt.timezone.utc)
    assert run_queue(spec(tmp_path, [job(tmp_path, "a", WRITE)]), poll=0.05) == 0
    started = json.loads((tmp_path / "queue" / "started.json").read_text())
    assert dt.datetime.fromisoformat(started["utc"]) >= before - dt.timedelta(seconds=1)
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["self_stop"] is False
    (tmp_path / "queue" / "a.status").unlink()
    assert run_queue(spec(tmp_path, [job(tmp_path, "b", WRITE)], stop=stop_marker(tmp_path, monkeypatch)),
                     poll=0.05) == 0
    assert json.loads((tmp_path / "queue" / "outcome.json").read_text())["self_stop"] is True


def test_a_queue_refused_for_the_lock_writes_no_start_mark(tmp_path):
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    held = (tmp_path / "pod.lock").open("w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(QueueError, match="already running"):
        run_queue(p, poll=0.05)
    assert not (tmp_path / "queue" / "started.json").exists()
    fcntl.flock(held, fcntl.LOCK_UN)


# outcome.json is never read half-written


@pytest.mark.parametrize("refused", [False, True], ids=["end", "refused-at-start"])
def test_outcome_json_is_written_to_a_temp_file_and_moved_into_place(tmp_path, monkeypatch, refused):
    """A watcher or copy loop that reads outcome.json while the queue writes it saw a torn file: the queue
    writes it beside, then os.replace moves it into place; nothing writes outcome.json in place, and no temp file is
    left behind."""
    from pathlib import Path

    from v2gpu import jobqueue

    real_write, real_replace, moved = Path.write_text, os.replace, []

    def write_text(self, *args, **kwargs):
        assert self.name != "outcome.json", "outcome.json written in place"
        return real_write(self, *args, **kwargs)

    def replace(src, dst):
        if Path(dst).name == "outcome.json":
            assert Path(src).parent == Path(dst).parent and Path(src).name != "outcome.json"
            moved.append(json.loads(Path(src).read_text()))
        return real_replace(src, dst)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(jobqueue.os, "replace", replace)
    p = spec(tmp_path, [job(tmp_path, "a", WRITE)])
    if refused:
        s = json.loads(p.read_text())
        s["deadline_utc"] = "2030-01-01T00:00:00"
        real_write(p, json.dumps(s))
        with pytest.raises(QueueError, match="timezone"):
            run_queue(p, poll=0.05)
    else:
        assert run_queue(p, poll=0.05) == 0
    outcome = json.loads((tmp_path / "queue" / "outcome.json").read_text())
    assert moved == [outcome] and outcome["outcome"] == ("REFUSED_AT_START" if refused else "ALL_DONE")
    assert sorted(x.name for x in (tmp_path / "queue").iterdir() if x.name.startswith("outcome")
                  or "tmp" in x.name) == ["outcome.json"]
