"""One ER run: a fresh container of the trace image, the record's base tree and the condition files mounted read-only,
Behat --strict on exactly one scenario, the harness trace (ER_TRACE=1, soft mode) read back.

The runner never composes code and never applies the insertion rule: it only mounts what the manifest lists.
Soft mode (every step executes) gives the same PASS/FAIL as plain strict ER (measured on 148 records: 148/148 stub,
148/148 gold), so the
traced run replaces the plain ER run; the ER status is derived from the step results (trace.status_from_steps).

Threat model: model code is NOT adversarial, but it runs in the same PHP process as the harness. What holds:
  - the container starts from the immutable image id (a clean tree), on the internal network (no Internet), with no
    capabilities, no-new-privileges and memory/pids/file-size limits;
  - it connects as er_laneN (DML only on its own lane's database, random password rotated by init-db.sh; root has no
    network account, see er.db), so it cannot change the schema or touch another lane; the URL reaches docker through
    its environment, never a command line (it is in the container's config, see er.db), and every error string kept
    here has the password masked BEFORE it is cut to length (no piece of it survives a truncation);
  - one container per lane at a time (lane_lock across processes, fixed name gtestgen-er-lane<N>): an orphan left by a
    crashed harness is removed before the next run, and a run killed by the host-side timeout before the lock is freed;
  - everything read back from /out is untrusted: regular files only (no symlinks; FIFOs and devices are refused
    without blocking), size-capped files, at most MAX_OUT_FILES entries and MAX_OUT_BYTES in the directory (counted
    while walking it, stopping at the cap), schema-checked JSON, hostile XML refused, error strings capped, only the
    first MiB of behat.log scanned. Model code can still write /out (e.g. forge trace.json): the tamper scan flags it.
Statuses: PASS | FAIL | UNDEFINED | ERROR | TIMEOUT | OOM, and INFRA (the environment failed, or the output looks like
it did: the batch retries it and decides with db.lane_healthy). INFRA carries infra_source: docker (docker run
failed), timeout (host-side timeout), inner (no inner.json), log (a database connection error or a full database disk
in the first MiB of behat.log), os (the output directory could not be cleared, or docker could not be started).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from defusedxml import ElementTree as SafeET

from .db import NETWORK, db_url, lane_password
from .manifest import Entry
from .steps import step_plan
from .trace import aligned, status_from_steps, statuses, validate_trace

IMAGE = "gtestgen-er/sylius-trace:c096ca3"
COMPOSER_LOCK_SHA256 = "f71ed33c72a4e30df98db06bf8fa4688e40b3328af9ac982ff1a5bfd86a9e72f"
RUN_TIMEOUT = 60  # seconds for Behat inside the container; a warm run takes 1-3 s, a cold one about 6 s
DOCKER_TIMEOUT = 30  # docker inspect / rm; `docker run` gets RUN_TIMEOUT + 60
FSIZE = 64 * 2**20  # largest file a run may write
MAX_READ = 16 * 2**20  # largest junit/trace/inner file read back
MAX_OUT_BYTES = 256 * 2**20  # largest output directory (sum of lstat sizes)
MAX_OUT_FILES = 10_000  # most entries (files, directories, links) scored in an output directory
LOG_SCAN = 1 << 20  # INFRA_LOG looks at the first MiB of behat.log only
MIN_FRAGMENT = 4  # a piece of the password this long, left at an edge by an upstream cut, is masked too
HARDENING = ["--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "1536m", "--memory-swap", "1536m",
             "--cpus", "2", "--pids-limit", "512", "--ulimit", f"fsize={FSIZE}:{FSIZE}"]
LOCK_DIR = Path(os.environ.get("ER_LOCK_DIR", "/tmp"))
# a database connection error, or a full database disk (1114 table full, 1021 disk full, ENOSPC)
INFRA_LOG = re.compile(r"ConnectionException|ConnectionLost|SQLSTATE\[HY000\] \[(1040|1045|1203|2002|2006|2013)\]|"
                       r"General error: 20(06|13)|General error: (1114|1021)|No space left on device|\[(1114|1021)\]")


class OutputError(ValueError):
    pass


def mask(text: str, secret: str) -> str:
    """`text` without the lane password: every occurrence, and a piece of it that a truncation made upstream left at
    an edge (a prefix of the password ending `text`, a suffix starting it, MIN_FRAGMENT chars or more)."""
    if not secret:
        return text
    text = text.replace(secret, "***")
    for k in range(min(len(secret) - 1, len(text)), MIN_FRAGMENT - 1, -1):
        if text.endswith(secret[:k]):
            text = text[:-k] + "***"
            break
    for k in range(min(len(secret) - 1, len(text)), MIN_FRAGMENT - 1, -1):
        if text.startswith(secret[-k:]):
            text = "***" + text[k:]
            break
    return text


def clip(text: str, n: int, secret: str = "", tail: bool = False) -> str:
    """At most n chars of `text` (its head, or its tail), the password masked BEFORE the cut (a cut through the
    password would otherwise keep a piece of it that no later mask can recognise)."""
    text = mask(text, secret)
    return text[-n:] if tail else text[:n]


@lru_cache(maxsize=None)
def image_id(image: str = IMAGE) -> str:
    return subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], check=True,
                          capture_output=True, text=True, timeout=DOCKER_TIMEOUT).stdout.strip()


def container_name(lane: int) -> str:
    return f"gtestgen-er-lane{lane}"


@contextmanager
def lane_lock(lane: int):
    """Exclusive use of a lane (its database) across processes on this host, for the whole docker run."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_DIR / f"gtestgen-er-lane{lane}.lock", "w", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def remove_container(name: str) -> None:
    """docker rm -f, result ignored: if the container is still there, the next `docker run --name` fails (INFRA)."""
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=DOCKER_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        pass


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    """An fd on a regular file written from inside the container. Symlinks are refused (O_NOFOLLOW); O_NONBLOCK makes
    opening a FIFO return at once instead of waiting for a writer, and fstat then refuses anything not regular."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise OutputError(f"{path.name}: {exc.strerror}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OutputError(f"{path.name}: not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd, st


def _read_upto(fd: int, n: int) -> bytes:
    chunks, size = [], 0
    while size < n:
        chunk = os.read(fd, min(1 << 20, n - size))
        if not chunk:
            break
        size += len(chunk)
        chunks.append(chunk)
    return b"".join(chunks)


def read_capped(path: Path, limit: int = MAX_READ) -> bytes:
    """Read a regular file written from inside the container: refuse symlinks, FIFOs and other non-regular files
    (never blocks), and anything larger than `limit`."""
    fd, st = _open_regular(path)
    try:
        if st.st_size > limit:
            raise OutputError(f"{path.name}: larger than {limit} bytes")
        data = _read_upto(fd, limit + 1)  # the file may still be growing
        if len(data) > limit:
            raise OutputError(f"{path.name}: larger than {limit} bytes")
        return data
    finally:
        os.close(fd)


def read_head(path: Path, n: int) -> bytes:
    """The first `n` bytes (at most) of a regular file written from inside the container, whatever its size; same
    refusals as read_capped."""
    fd, _ = _open_regular(path)
    try:
        return _read_upto(fd, n)
    finally:
        os.close(fd)


def sha256_capped(path: Path) -> str | None:
    try:
        return hashlib.sha256(read_capped(path, FSIZE)).hexdigest()
    except OutputError:
        return None


def docker_cmd(record: dict, root: Path, entry: Entry, lane: int, out: Path, noop: tuple[int, ...], name: str,
               image: str = IMAGE, timeout: int = RUN_TIMEOUT) -> list[str]:
    """The docker run command. DATABASE_URL is passed by name: its value (with the lane password) must be in the
    environment of the docker CLI (run_once does that), so it never appears on a command line."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", NETWORK, *HARDENING,
           "-e", "DATABASE_URL", "-e", f"ER_TIMEOUT={timeout}", "-e", "ER_TRACE=1", "-e", "ER_SOFT=1",
           "-e", f"ER_NOOP={','.join(map(str, noop))}", "-v", f"{out.resolve()}:/out",
           "-v", f"{(root / entry.base).resolve()}:/overlay_base:ro"]
    if entry.dir is not None:
        cmd += ["-v", f"{(root / entry.dir).resolve()}:/overlay:ro"]
    return cmd + [image_id(image), "er-inner", record["suite"], f"{record['feature_path']}:{record['scenario_line']}"]


def junit_cases(out: Path) -> list:
    d = out / "junit"
    if d.is_symlink() or not d.is_dir():
        return []
    cases = []
    for item in sorted(os.scandir(d), key=lambda x: x.name):
        if item.is_symlink() or not item.is_file(follow_symlinks=False):
            raise OutputError(f"junit/{item.name}: not a regular file")
        if item.name.endswith(".xml"):
            cases += list(SafeET.fromstring(read_capped(Path(item.path))).iter("testcase"))
    return cases


def classify(record: dict, out: Path, inner: dict, noop: tuple[int, ...] = (),
             secret: str = "") -> tuple[str, dict, dict | None]:
    """(status, detail, trace). Never raises: anything unreadable or malformed is an ERROR. Error strings are capped,
    `secret` (the lane password) masked before any cut."""
    if not isinstance(inner, dict):
        return "ERROR", {"error": "inner.json is not a JSON object"}, None
    code = inner.get("behat_exit")
    if inner.get("oom_kill") and code == 137:  # the kernel killed Behat itself (an OOM of a helper is not fatal)
        return "OOM", {}, None
    if code in (124, 137):
        return "TIMEOUT", {}, None
    try:
        cases = junit_cases(out)
        detail = {"n_scenarios": len(cases)}
        if len(cases) != 1:
            return "ERROR", {**detail, "error": "not exactly one scenario ran"}, None
        if cases[0].get("name") != record["scenario_name"]:
            return "ERROR", {**detail, "error": f"ran {clip(str(cases[0].get('name')), 80, secret)!r}"}, None
        if code != 0 or cases[0].get("status") != "passed":
            # soft mode reports every step as passed to Behat: anything else is a hook, harness or PHP-level failure
            junit = clip(str(cases[0].get("status")), 80, secret)
            return "ERROR", {**detail, "error": f"behat exit {code}, junit {junit}"}, None
        trace = json.loads(read_capped(out / "trace.json").decode("utf-8"))
        validate_trace(trace)
        if not aligned(trace, record, step_plan(record)):
            return "ERROR", {**detail, "error": "trace steps differ from the record"}, None
        if [i for i, s in enumerate(statuses(trace)) if s == "noop"] != sorted(set(noop)):
            return "ERROR", {**detail, "error": "noop steps differ from the request"}, None
        return status_from_steps(statuses(trace)), detail, trace
    except Exception as exc:  # hostile/malformed XML or JSON, symlink, oversized file, bad trace schema
        return "ERROR", {"error": clip(f"{type(exc).__name__}: {exc}", 200, secret)}, None


def clear_dir(out: Path) -> None:
    """Remove a previous run's output (it may hold read-only files or directories) and create `out` empty."""
    if os.path.lexists(out):
        subprocess.run(["chmod", "-R", "u+rwX", str(out)], capture_output=True, timeout=120)  # never follows symlinks
        shutil.rmtree(out)
    out.mkdir(parents=True)


def tree_bytes(path: Path, max_files: int) -> int:
    """Total lstat size of everything under `path`: symlinks count as themselves and are never followed. Entries are
    counted as they are listed: OutputError as soon as there are more than `max_files` (a huge directory is never
    listed to its end); OSError when a directory cannot be read."""
    total, count, todo = 0, 0, [os.fspath(path)]
    while todo:
        with os.scandir(todo.pop()) as it:
            for entry in it:
                count += 1
                if count > max_files:
                    raise OutputError("too many files in the output directory")
                st = os.lstat(entry.path)
                total += st.st_size
                if stat.S_ISDIR(st.st_mode):
                    todo.append(entry.path)
    return total


def _text(b: bytes | str | None) -> str:
    return b if isinstance(b, str) else (b or b"").decode(errors="replace")


def run_once(record: dict, root: Path, entry: Entry, lane: int, out: Path, noop: tuple[int, ...] = (),
             image: str = IMAGE, timeout: int = RUN_TIMEOUT) -> dict:
    """Run one (entry, mutation) and return its result; the trace dict is under key "trace" (not serialised).
    Never raises for anything the run produced (a missing lane password is an InfraError: rerun init-db.sh)."""
    secret = lane_password(lane)
    env = {**os.environ, "DATABASE_URL": db_url(lane)}
    t0 = time.monotonic()

    def result(status: str, error: str | None = None, source: str | None = None, **kw) -> dict:
        r = {"status": status, "seconds": round(time.monotonic() - t0, 3), "behat_exit": None, "trace": None, **kw}
        if source is not None:
            r["infra_source"] = source
        if error is not None:
            r["error"] = mask(error, secret)
        return r
    try:
        clear_dir(out)
    except (OSError, subprocess.SubprocessError) as exc:
        return result("INFRA", f"cannot clear the output directory: {type(exc).__name__}: {exc}", "os")
    name = container_name(lane)
    try:
        with lane_lock(lane):
            remove_container(name)  # an orphan of a crashed harness would still be using the lane database
            try:
                subprocess.run(docker_cmd(record, root, entry, lane, out, noop, name, image, timeout), check=True,
                               capture_output=True, timeout=timeout + 60, env=env)
            except subprocess.TimeoutExpired:  # never str(exc): it holds the command line
                remove_container(name)
                return result("INFRA", f"docker did not finish within {timeout + 60} s", "timeout")
    except subprocess.CalledProcessError as exc:
        return result("INFRA", f"docker: {clip(_text(exc.stderr), 300, secret, tail=True)}", "docker")
    except OSError as exc:
        return result("INFRA", f"{type(exc).__name__}: {exc}", "os")
    seconds = round(time.monotonic() - t0, 3)
    try:
        too_large = tree_bytes(out, MAX_OUT_FILES) > MAX_OUT_BYTES
    except OutputError as exc:
        return result("ERROR", str(exc), seconds=seconds)
    except OSError as exc:
        return result("ERROR", f"output directory unreadable: {type(exc).__name__}", seconds=seconds)
    if too_large:
        return result("ERROR", "output directory too large", seconds=seconds)
    if not os.path.lexists(out / "inner.json"):
        return result("INFRA", "inner.json missing", "inner", seconds=seconds)
    try:
        inner = json.loads(read_capped(out / "inner.json", 4096).decode("utf-8"))
    except (OutputError, ValueError) as exc:
        return result("ERROR", f"inner.json: {exc}", seconds=seconds)
    if not isinstance(inner, dict):
        return result("ERROR", "inner.json is not a JSON object", seconds=seconds)
    status, detail, trace = classify(record, out, inner, noop, secret)
    if status != "PASS":  # a PASS proves the database worked, whatever the log says
        try:
            log_head = read_head(out / "behat.log", LOG_SCAN).decode("utf-8", errors="replace")
        except OutputError:
            log_head = ""
        if INFRA_LOG.search(log_head):
            return result("INFRA", "database connection failed", "log", seconds=seconds,
                          behat_exit=inner.get("behat_exit"))
    if "error" in detail:
        detail = {**detail, "error": mask(detail["error"], secret)}
    return {"status": status, "seconds": seconds, "behat_exit": inner.get("behat_exit"),
            "behat_seconds": inner.get("behat_seconds"), "cold": inner.get("cold"),
            "mem_peak_bytes": inner.get("mem_peak_bytes"), "log_sha256": sha256_capped(out / "behat.log"),
            "trace_sha256": sha256_capped(out / "trace.json") if trace is not None else None, **detail, "trace": trace}
