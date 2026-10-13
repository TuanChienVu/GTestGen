"""Run the pod's jobs one after another: python -m v2gpu.jobqueue QUEUE.json

QUEUE.json: {"queue_dir": ".../queue", "deadline_utc": "2026-10-01T12:00:00+00:00",
             "lock_path": "/root/v2gpu-queue.lock",               (required; container-local, not the network volume)
             "self_stop_cmd": ["runpodctl", "stop", "pod", "$RUNPOD_POD_ID"],       (optional)
             "jobs": [{"name": "deepseek-s42-all", "workdir": ".../runs/deepseek-s42-all", "max_hours": 7.5,
                       "cmd": [...], "resume_args": ["--resume"]}]}

Rules (lessons from earlier runs):
- status, log and heartbeat live in queue_dir, never inside a job's workdir (train_job refuses a non-empty workdir);
- a DONE job is skipped on a rerun, and so is a job whose receipt-train.json already says DONE (the queue died after the
  job finished); a job whose workdir holds a complete state (train_loop.state_dir: state, state.tmp or state.old) is
  restarted with its resume_args;
- a failed job stops the queue (a human looks before anything else runs);
- past the deadline the running job is terminated and the queue stops;
- per-job spending fuse: every job declares max_hours (measured duration x margin); a job is not started when less
  than max_hours remain before the deadline, and a job still running after max_hours is terminated (TIMEOUT);
- a finished job gets workdir/MANIFEST.sha256 (relative paths, state/ excluded) for the copy-out check. For
  the run manifest of 3C/3D, EVERY terminal status (DONE, FAILED, DEADLINE, TIMEOUT, NO_TIME) gets one too, when the
  job's workdir exists (write_manifest is a no-op on a workdir that was never created), so copy_out.sh can copy a
  failed workdir home with its checksums for gencoverage's --lost-workdirs evidence;
- self_stop_cmd is checked (variables resolved, command found) before any job starts, so a self_stop_cmd that could
  never run does not let every job run first and bill idle at the end; whatever ends the queue afterwards, self_stop_cmd
  runs last, so the pod does not bill while nobody watches;
- only one queue runs per lock_path (an flock; use one lock_path for every queue file of a pod, on a container-local
  disk such as /root: flock on a network volume is not reliable). The job inherits the locked file descriptor, so a
  queue restarted while an orphaned job still runs is refused too. Grandchildren do not inherit it (close_fds), so a
  grandchild orphaned together with its queue is the one case the lock cannot see;
- a job runs in its own process group; DEADLINE and TIMEOUT terminate the whole group (canary -> train_job -> ...), and
  after a normal exit whatever is left in the group is killed, so no straggler outlives its job;
- deadline_utc must carry a timezone.
Also:
- a queue refused at start (no timezone, a job without max_hours, a self_stop_cmd that cannot run) writes
  outcome.json {"outcome": "REFUSED_AT_START", "error": ..., "self_stop": ...} in its queue_dir before raising, so the
  laptop's copy loop sees the refusal at once instead of polling empty statuses. The checks run after the pod lock is
  taken: a queue refused because another one runs writes nothing (that queue owns the pod and its files). A refused
  start does NOT stop the pod: it is fixed in minutes on the live pod, and a stopped Community pod may not get its card
  back; the on-pod watchdog (started right after SSH) caps an unattended pod, and it also covers the refusals
  that cannot run self_stop_cmd at all ($RUNPOD_POD_ID unresolved, runpodctl missing) and a missing lock_path;
- self_stop_cmd runs with a timeout (SELF_STOP_TIMEOUT, 120 s) and is tried once more after a timeout or a non-zero
  exit; self_stop.log records every try.
Right after taking the pod lock the queue moves an
outcome.json left by an earlier run aside (outcome.<utc>.json), so no consumer reads it as this run's end, and writes
started.json (its mtime is this run's start: ops/watch_pod.sh counts its heartbeat grace from it and both consumers
ignore an outcome older than it); outcome.json says whether self_stop_cmd follows it ("self_stop"), so the laptop
watcher waits for the pod to stop instead of leaving at the queue's end.
outcome.json is written atomically (a temp file beside it, then os.replace), so
a watcher or copy loop that reads it while the queue writes it never sees a torn record.
Copy window: a non-DONE job (FAILED, DEADLINE, TIMEOUT, NO_TIME, a controlled refusal such as
v2gpu.train_probe's REFUSED_WOULD_NOT_FIT, ...) used to stop the WHOLE queue at once -- _self_stop ran within about a
second, before a trailing copy-window-* job ever got to run, so nothing
protected the pod from stopping before its own last receipt/log could be copied home.
Now: once any job's own verdict is not DONE, every LATER job that is not itself named copy-window-* is skipped (as
before -- "a failed job stops the queue" still holds for the real jobs), but a LATER copy-window-* job still runs,
to its own max_hours or the overall deadline_utc, whichever comes first (it is already exempt from the NO_TIME
pre-check regardless). So EVERY way the queue can end -- success (ALL_DONE), a genuine FAILED job, a TIMEOUT, or a
controlled refusal -- now passes through the same copy window before self_stop_cmd fires, as long as the queue's own
job list ends with one. The queue's own outcome.json still names the FIRST non-DONE job/reason (unchanged for every
existing caller): a later copy-window-* job's own run is visible in its own <name>.status file, exactly as GEN's
copy loop (and TRAIN_RUNSHEET.md §7) already reads per-job status directly, not only the aggregate outcome.
A DEADLINE is the one ending that does NOT get a copy window: the deadline has already passed, so the window is cut at
once (test_the_deadline_still_cuts_the_copy_window).
A copy-window-* job's earlier terminal status never ends it on a rerun -- it
always runs again (still bound to its spec sha), so the recovery rerun copies the resumed job home too.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .train_loop import state_dir

SKIP_DIRS = {"state", "state.tmp", "state.old"}
SELF_STOP_TIMEOUT = 120      # seconds per try of self_stop_cmd; two tries
SELF_STOP_TRIES = 2
REFUSED_STOP_NOTE = ("not attempted: a refused start is fixed on the live pod; the on-pod watchdog caps an "
                     "unattended pod")


def write_manifest(workdir: Path) -> Path | None:
    """For the run manifest of 3C/3D: None, and nothing written, when workdir was never created (a job
    cut before its own code made it, e.g. NO_TIME, or a command that could not even start) -- there is nothing to
    manifest, and creating an empty workdir here would be its own lie."""
    workdir = Path(workdir)
    if not workdir.is_dir():
        return None
    lines = []
    for p in sorted(workdir.rglob("*")):
        rel = p.relative_to(workdir)
        if not p.is_file() or SKIP_DIRS & set(rel.parts) or rel.name == "MANIFEST.sha256":
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel.as_posix()}\n")
    out = workdir / "MANIFEST.sha256"
    out.write_text("".join(lines))
    return out


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _write_outcome(qdir: Path, record: dict) -> None:
    """outcome.json, atomically: written to a temp file in queue_dir, then moved into place."""
    tmp = qdir / f".outcome.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(record))
    os.replace(tmp, qdir / "outcome.json")


def _stop(proc: subprocess.Popen, grace: float) -> None:
    """Terminate the job's whole process group (it was started in a new session), then kill what is left."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(wait)
            if sig == signal.SIGTERM:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)   # stragglers in the group outlive its first process
                except ProcessLookupError:
                    pass
            break
        except subprocess.TimeoutExpired:
            continue


class QueueError(RuntimeError):
    pass


def _refusal(spec: dict) -> str | None:
    """Why the queue must not start, or None: checked before any job runs."""
    try:
        deadline = dt.datetime.fromisoformat(spec["deadline_utc"])
    except (TypeError, ValueError):
        return f"deadline_utc {spec['deadline_utc']!r} is not an ISO time"
    if deadline.tzinfo is None:
        return f"deadline_utc {spec['deadline_utc']!r} has no timezone"
    for job in spec["jobs"]:
        if not isinstance(job.get("max_hours"), (int, float)) or job["max_hours"] <= 0:
            return f"job {job.get('name')!r} needs a positive max_hours"
    stop_cmd = spec.get("self_stop_cmd")
    if stop_cmd:
        args = [os.path.expandvars(a) for a in stop_cmd]
        if any("$" in a for a in args):
            return f"self_stop_cmd has an unresolved variable: {args}"
        if shutil.which(args[0]) is None:
            return f"self_stop_cmd not found: {args[0]}"
    return None


def run_queue(spec_path: Path, poll: float = 5.0, heartbeat_every: float = 60.0, grace: float = 60.0) -> int:
    spec = json.loads(Path(spec_path).read_text())
    qdir = Path(spec["queue_dir"])
    if not spec.get("lock_path"):
        raise QueueError("lock_path is required (one container-local path for every queue of the pod)")
    qdir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(spec["lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise QueueError(f"another queue is already running in {qdir}") from None
    try:
        _mark_start(qdir)                 # the lock is ours; an earlier run's outcome is set aside
        refused = _refusal(spec)
        if refused:
            _write_outcome(qdir, {"outcome": "REFUSED_AT_START", "error": refused, "self_stop": REFUSED_STOP_NOTE,
                                  "utc": _now().isoformat()})
            # "self_stop" is a note here, never true: no stop follows a refused start (the watchdog caps the pod)
            raise QueueError(refused)
        deadline = dt.datetime.fromisoformat(spec["deadline_utc"])
        return _run(spec, qdir, deadline, poll, heartbeat_every, grace, lock.fileno())
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _mark_start(qdir: Path) -> None:
    """Set an earlier run's outcome.json aside and write started.json."""
    old = qdir / "outcome.json"
    if old.exists():
        stamp = _now().strftime("%Y%m%dT%H%M%S%fZ")
        old.rename(qdir / f"outcome.{stamp}.json")
    (qdir / "started.json").write_text(json.dumps({"utc": _now().isoformat(), "pid": os.getpid()}))


def _receipt_done(workdir: Path) -> bool:
    try:
        return json.loads((workdir / "receipt-train.json").read_text()).get("status") == "DONE"
    except (OSError, ValueError):
        return False


def _spec_sha(job: dict) -> str:
    """The canonical sha256 of a job's identity (name, workdir, cmd,
    max_hours, env), computed the same way whenever a terminal status is written or read."""
    canonical = {"name": job["name"], "workdir": job["workdir"], "cmd": job["cmd"], "max_hours": job["max_hours"],
                "env": job.get("env") or {}}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _write_status(status_file: Path, spec_sha_file: Path, spec_sha: str, text: str) -> None:
    # Since 27/09/2026 the sha is written BEFORE the status, so a crash between the two
    # writes leaves "no status at all" (retried as a fresh job) rather than "a status with no sha" (SPEC_SHA_MISSING).
    spec_sha_file.write_text(spec_sha)
    status_file.write_text(text)


def _execute_job(job: dict, qdir: Path, deadline: dt.datetime, poll: float, heartbeat_every: float, grace: float,
                 lock_fd: int) -> str:
    """Run ONE job to a terminal state and return it: "DONE", or a stopping reason (SPEC_SHA_MISSING_*,
    SPEC_CHANGED_*, BLOCKED_BY_*, NO_TIME_FOR_*, DEADLINE, TIMEOUT_*, FAILED_*). Extracted from _jobs so _jobs can
    still call this for a LATER copy-window-* job after an earlier job's own
    verdict was not DONE, instead of that job never running at all -- see _jobs's own docstring."""
    status_file = qdir / f"{job['name']}.status"
    spec_sha_file = qdir / f"{job['name']}.spec-sha256"
    spec_sha = _spec_sha(job)
    status = status_file.read_text().strip() if status_file.exists() else None
    if status is not None:
        # A terminal status (DONE and
        # every other one below) is bound to the job spec that earned it. A job whose cmd (or workdir, max_hours,
        # env) changed under the same name must never silently inherit that old status: it is refused, neither
        # skipped nor run, until the queue file (or the job's name) is fixed.
        # A status with no .spec-sha256 sidecar at all (a legacy queue dir from before this check,
        # or a status file dropped in by hand) is a DIFFERENT, more precise outcome than "the spec changed" -- the
        # sha was simply never recorded. Both fail exactly the same way (neither skipped nor run; the queue stops
        # and self-stops as usual): only the outcome string, and the advice given for it, differ.
        if not spec_sha_file.exists():
            return f"SPEC_SHA_MISSING_{job['name']}"
        recorded = spec_sha_file.read_text().strip()
        if recorded != spec_sha:
            return f"SPEC_CHANGED_{job['name']}"
    # A copy-window-* job is an idempotent keep-alive, never a result. Its
    # earlier terminal status (DONE, TIMEOUT, ...) must not end it: the recovery (delete the failed job's
    # .status, rerun the same queue) would otherwise skip the window as already DONE, reach ALL_DONE and self-stop the
    # pod before the resumed job's work is copied home. The spec binding above still applies to it.
    is_window = job["name"].startswith("copy-window-")
    if status == "DONE" and not is_window:
        return "DONE"
    if status in ("FAILED", "DEADLINE", "TIMEOUT", "NO_TIME") and not is_window:
        return f"BLOCKED_BY_{job['name']}"
    workdir = Path(job["workdir"])
    if _receipt_done(workdir):
        if not (workdir / "MANIFEST.sha256").exists():
            write_manifest(workdir)
        _write_status(status_file, spec_sha_file, spec_sha, "DONE\n")
        return "DONE"
    # A job named copy-window-* is exempt from the NO_TIME pre-check. Its whole
    # purpose is to keep the pod up while the operator's copy loop empties the last workdirs home; refusing to
    # even start it whenever Sigma(max_hours) of the real jobs left less than the window's own max_hours before
    # the deadline used to self-stop the pod before that copy ran. It still runs for at most min(max_hours, the
    # time left), since the deadline (and its own max_hours, via TIMEOUT) still cut it once started.
    if not job["name"].startswith("copy-window-") and \
            (deadline - _now()).total_seconds() < job["max_hours"] * 3600:
        write_manifest(workdir)      # for the run manifest: a prior attempt's workdir may already hold evidence
        _write_status(status_file, spec_sha_file, spec_sha, "NO_TIME\n")
        return f"NO_TIME_FOR_{job['name']}"
    cmd = list(job["cmd"])
    if state_dir(workdir) is not None:
        cmd += job.get("resume_args", [])
    with (qdir / f"{job['name']}.log").open("a") as log:
        log.write(f"# {_now().isoformat()} start {cmd}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    pass_fds=(lock_fd,))
        except OSError as e:
            log.write(f"# {_now().isoformat()} cannot start: {e!r}\n")
            write_manifest(workdir)
            _write_status(status_file, spec_sha_file, spec_sha, "FAILED\n")
            return f"FAILED_{job['name']}"
        started, last_beat = time.monotonic(), 0.0
        while proc.poll() is None:
            if _now() >= deadline:
                _stop(proc, grace)
                write_manifest(workdir)
                _write_status(status_file, spec_sha_file, spec_sha, "DEADLINE\n")
                return "DEADLINE"
            if time.monotonic() - started > job["max_hours"] * 3600:
                _stop(proc, grace)
                write_manifest(workdir)
                _write_status(status_file, spec_sha_file, spec_sha, "TIMEOUT\n")
                return f"TIMEOUT_{job['name']}"
            if time.monotonic() - last_beat >= heartbeat_every:
                (qdir / "heartbeat.json").write_text(json.dumps(
                    {"job": job["name"], "pid": proc.pid, "utc": _now().isoformat()}))
                last_beat = time.monotonic()
            time.sleep(poll)
        try:
            os.killpg(proc.pid, signal.SIGKILL)      # stragglers the job left in its group
        except (ProcessLookupError, PermissionError):
            pass
        log.write(f"# {_now().isoformat()} exit {proc.returncode}\n")
    if proc.returncode != 0:
        write_manifest(workdir)
        _write_status(status_file, spec_sha_file, spec_sha, "FAILED\n")
        return f"FAILED_{job['name']}"
    # Exit 0 but no workdir at all (write_manifest returns None) is not
    # really DONE -- the job's own code never created it (a bug, or a cmd/workdir mismatch), so copy_out.sh
    # would find nothing home for it, well after the pod already stopped. Fail closed instead of claiming
    # success with no evidence; every OTHER terminal status above already keeps write_manifest's no-op here.
    if write_manifest(workdir) is None:
        _write_status(status_file, spec_sha_file, spec_sha, "FAILED\n")
        return f"FAILED_{job['name']}"
    _write_status(status_file, spec_sha_file, spec_sha, "DONE\n")
    return "DONE"


def _jobs(spec: dict, qdir: Path, deadline: dt.datetime, poll: float, heartbeat_every: float, grace: float,
          lock_fd: int) -> str:
    """Run the jobs in order; return the outcome (the FIRST non-DONE job's reason, or "ALL_DONE"). Exceptions
    propagate to _run, which still self-stops.

    Once any job's own verdict is not "DONE", every LATER job whose
    name does not start with "copy-window-" is skipped outright (never run -- "a failed job stops the queue" still
    holds for the real jobs, unchanged). A LATER copy-window-* job, though, still runs
    through _execute_job exactly as it would on the ALL_DONE path (its NO_TIME exemption already does not
    care whether an earlier job stopped the queue): success, a genuine FAILED job, a TIMEOUT, or a
    controlled refusal (e.g. v2gpu.train_probe's REFUSED_WOULD_NOT_FIT, which reaches here as an ordinary nonzero
    exit -> FAILED_<name>, exactly as any other crash) now all pass through the SAME copy window before
    self_stop_cmd fires, as long as the queue's job list ends with one -- not only the ALL_DONE path, which is all
    the earlier code protected."""
    stopped: str | None = None
    for job in spec["jobs"]:
        if stopped is not None and not job["name"].startswith("copy-window-"):
            continue
        verdict = _execute_job(job, qdir, deadline, poll, heartbeat_every, grace, lock_fd)
        if verdict != "DONE" and stopped is None:
            stopped = verdict
    return stopped or "ALL_DONE"


def _self_stop(spec: dict, qdir: Path) -> None:
    stop_cmd = spec.get("self_stop_cmd")
    if not stop_cmd:
        return
    args = [os.path.expandvars(a) for a in stop_cmd]
    log = []

    def write() -> None:
        try:
            (qdir / "self_stop.log").write_text("".join(log))
        except OSError:
            pass

    if any("$" in a for a in args):
        log.append(f"not run, unresolved variable in {args}\n")
        return write()
    for k in range(1, SELF_STOP_TRIES + 1):     # a hanging runpodctl must not hold the pod forever
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=SELF_STOP_TIMEOUT)
            log.append(f"try {k}: {args} -> {res.returncode}\n{res.stdout}{res.stderr}")
            if res.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            log.append(f"try {k}: {args} timed out after {SELF_STOP_TIMEOUT} s\n")
        except Exception as e:  # noqa: BLE001 - a self-stop problem must not raise over the queue's own outcome
            log.append(f"try {k}: {args} -> failed: {e!r}\n")
        write()
    write()


def _run(spec: dict, qdir: Path, deadline: dt.datetime, poll: float, heartbeat_every: float, grace: float,
         lock_fd: int) -> int:
    """Whatever happens inside, outcome.json is attempted and self_stop_cmd runs."""
    outcome, error = "CRASHED", None
    try:
        outcome = _jobs(spec, qdir, deadline, poll, heartbeat_every, grace, lock_fd)
    except Exception as e:  # noqa: BLE001 - recorded, then the pod still stops
        error = repr(e)
    finally:
        try:
            record = {"outcome": outcome, "utc": _now().isoformat(), "self_stop": bool(spec.get("self_stop_cmd"))}
            if error:
                record["error"] = error
            _write_outcome(qdir, record)
        finally:
            _self_stop(spec, qdir)
    return 0 if outcome == "ALL_DONE" else 1


if __name__ == "__main__":
    raise SystemExit(run_queue(Path(sys.argv[1])))
