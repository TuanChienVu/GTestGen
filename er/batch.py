"""Run and score every (record, condition) of a manifest run; write <out>/<run>/er_results.jsonl.

    python -m er.batch --root <manifest root> --run <run> --head-root repos/Sylius [--out DIR] [--ref-run RUN]
                       [--lanes 2] [--conditions a,b] [--limit N] [--supersede]

Controls run (no --ref-run; the manifest has neg/gold/stub, a gold for every record): per record the gold reference
(normal + M1) is run once and FROZEN in <out>/<run>/er_refs.jsonl (traces under er_runs/<slug>/_gold_ref/,
sha256-pinned; a later line for a record wins). Whenever the gold condition is scored, reference just computed or loaded
on resume, it must reproduce the reference (evaluable parts, and its M1 run: action trace and step statuses), else the
record gets ref_error "nondeterministic". A loaded reference whose gold status the gold run contradicts is suspect: its
line and every row of the record are archived at the end and recomputed on resume. A regenerated gold overlay needs
--supersede: the reference line and every row of the record are archived (*.superseded.jsonl) and recomputed. A row
without its reference line, or scored against another version of it, cannot be valid: archived and recomputed, no
--supersede needed. A SKIPped gold nulls its record (ref_error). GOLD_BREAKER broken references in a row on a lane: a
healthy lane and host mean broken golds (bad data: null rows, the run goes on), else the batch stops and those records
are rerun. A stopped batch also reruns every lane's current streak (broken references, rows without a trace): the same
failure may have caused it. Output runs (--ref-run <controls run>): the controls run's lock is held SHARED for the whole
run (no controls batch meanwhile); the controls run must be complete, unchanged since its freeze and match record by
record (gold overlay, base tree); a row scored against another frozen reference (gold overlay, traces, error) needs
--supersede. Canaries picked by rule (first record of this batch with a token, with a numeric id, with an evaluable
Then) re-run gold on every lane at the start, and on a lane after BREAKER_K rows in a row without a trace: gold must
PASS with the frozen traces, else the environment drifted (InfraError; those rows are archived and rerun on resume).
Each canary call keeps its runs under _canary/<trigger>_<seq>/. At a normal finish, a lane whose last rows form a
streak its breaker never reached is checked as the breaker would: the canaries on that lane (output run), the host and
the lane (controls run).

Every (record, condition) gets exactly one line, SKIP included; er_complete.json ("running": true while a batch runs)
lists what is missing from the WHOLE manifest, and from this run's scope. An output without a usable trace (ERROR,
TIMEOUT, OOM, SKIP) scores 0 on every part its record can evaluate; null + ref_error only when the record's gold
reference is broken. A scoring exception keeps ER and sets score_error. INFRA is retried (3 attempts) and, if it
persists, stops the batch without being recorded; INFRA from the log or a missing inner.json of an output, twice with a
healthy lane and host, is that output's ERROR (gold, trusted code, is never gated). --head-root must be a clean clone at
HEAD (the tamper scan diffs against it). prepare and the breakers need MIN_HOST_FREE bytes free under --out and a
writable run directory. An exception in any lane, or Ctrl-C, stops every lane at its next run boundary; once the batch
is stopping (or once every lane finished and the final join began), a Ctrl-C only waits (sigint_while_lanes_run: no
exception, not even of its notice on a broken stderr, may enter the join of a lane still in its run, which CPython
3.12 would take for a stopped lane), so the run lock is held until every lane returned; Ctrl-C is ignored while
finish() runs. One batch per run (.er_batch.lock). Resumable: rows already written for the same overlay_sha256 and
reference are kept; prepare() writes nothing until every check passed (its archives and er_canary.jsonl read too: a
corrupt line stops it), then repairs a cut last line (atomically; a line cut inside a multi-byte character is a cut
line too).

The journal (er_complete.json, "archiving"): while the lanes run it names the suspects and every lane's current streak
(saved whenever they change; a row without a trace is journaled before it is written, and a streak leaves it only once
the row with a trace that ends it is written), and finish() writes the exact lines it will move, with the archives'
sizes, before its first move. A batch that dies (SIGKILL in its run, a kill at any step of finish) leaves it; the next
prepare() leaves those lines out of every check and, once all passed, archives them (a line the interrupted move
already appended after the offset of that archive is not appended again) and drops the journal; they are recomputed
like any missing row. Journaled lines skip the --supersede check: archived as suspect (or stale_ref), then recomputed
on the current manifest and gold.

A resume needs the same RESUME_KEYS, harness code included: harness_code (hash of the files under er/ that git tracks
or that are untracked and not ignored, the *.md docs left out; the real path refuses when it cannot be computed) and
harness_dirty (a local change or an untracked file there, docs left out); harness_commit (HEAD) is kept for audit; all
three are in the provenance of every row. An output run needs its controls run's harness_code and both runs clean: a
code change after the controls run needs a new controls run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from typing import Callable

from . import db
from .manifest import Entry, ManifestError, load_manifest, sha256_file, tamper_flags, verify_entry
from .records import load_records, seal_sha256
from .runner import COMPOSER_LOCK_SHA256, IMAGE, image_id, read_capped, run_once
from .steps import step_plan
from .trace import GoldRef, action_trace, gold_reference, score_missing, score_output, statuses

DEFAULT_OUT = Path("out/phase3/er")  # relative to the working directory: run from the repository root
HEAD = "c096ca3fc01545ee25078ba3df914bd2069c7fa6"  # Sylius commit of the ER trees; --head-root must be a clone at it
HARNESS_ROOT = Path(__file__).resolve().parents[1]  # the git checkout that holds er/ (harness_code, _dirty, _commit)
RunFn = Callable[..., dict]
KEEP = ("seconds", "behat_exit", "behat_seconds", "cold", "mem_peak_bytes", "log_sha256", "trace_sha256",
        "n_scenarios", "error")
REF_KEYS = ("gold_overlay_sha256", "gold_status", "gold_trace_sha256", "gold_m1_trace_sha256", "ref_error")
RESUME_KEYS = ("image_id", "phase2_sha256sums_sha256", "schema_fingerprint", "ref_run", "manifest_root", "manifest_run",
               "harness_code", "harness_dirty")
RETRY_SLEEP = 5.0
MAX_INFRA, MAX_TIMEOUT = 3, 2  # attempts
BREAKER_K = 8  # --ref-run: rows in a row on a lane without a trace before the canaries are re-run there
GOLD_BREAKER = 3  # controls run: broken gold references in a row on a lane before the batch stops
NO_TRACE = ("ERROR", "TIMEOUT", "OOM")
MIN_HOST_FREE = 2 * 2**30  # bytes free under --out (results, traces, logs) for prepare and the breakers


def slug_of(entry: Entry) -> str:
    return PurePosixPath(entry.base).parts[-2]


def _git(head_root: Path, *args: str, stdin: str | None = None) -> str | None:
    """stdout of a git command in head_root (None when it fails)."""
    try:
        out = subprocess.run(["git", "-C", str(head_root), *args], input=stdin, capture_output=True, text=True,
                             timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def head_commit(head_root: Path) -> str:
    """HEAD of a git checkout ('' when it is not one)."""
    return _git(head_root, "rev-parse", "HEAD") or ""


def head_clean(head_root: Path) -> bool:
    """No local change and no untracked file, whatever the git config says (tamper flags diff the outputs against this
    tree). Ignored files are out of scope: the tamper scan only reads the tracked CREATE targets."""
    return _git(head_root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal",
                "--ignore-submodules=none") == ""


def harness_code(root: Path) -> str:
    """Identity of the harness code: sha256 of the sorted lines "<path> <blob sha>\\n" of every file under er/ that
    git tracks, or that is untracked and not ignored, except the *.md docs, each blob sha taken from the file on disk
    (git hash-object), so an uncommitted edit changes it too. On a clean tree these are the lines of
    `git ls-tree -r HEAD -- er` without the *.md files: anyone can
    recompute it from the commit. '' when git fails (not a checkout)."""
    listed = _git(root, "ls-files", "-z", "--", "er")
    others = _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", "er")
    if listed is None or others is None:
        return ""
    paths = sorted({p for p in (listed + "\0" + others).split("\0")
                    if p and not p.endswith(".md") and (root / p).is_file()})
    shas = _git(root, "hash-object", "--stdin-paths", stdin="".join(p + "\n" for p in paths)) if paths else ""
    if shas is None or len(shas.split()) != len(paths):
        return ""
    return hashlib.sha256("".join(f"{p} {s}\n" for p, s in zip(paths, shas.split())).encode()).hexdigest()


def harness_dirty(root: Path) -> bool:
    """A local change (staged or not) or an untracked file under er/, the *.md docs left out: the code that runs is
    not a commit. True when git fails."""
    changed = _git(root, "--no-optional-locks", "diff", "--name-only", "-z", "HEAD", "--", "er")
    untracked = _git(root, "--no-optional-locks", "ls-files", "-z", "--others", "--exclude-standard", "--", "er")
    if changed is None or untracked is None:
        return True
    return any(p and not p.endswith(".md") for p in (changed + "\0" + untracked).split("\0"))


def lane_healthy(lane: int) -> bool:
    return db.lane_healthy(lane)


class Stopped(Exception):
    """Another lane failed, or Ctrl-C: this lane left its current entry unrecorded (rerun on resume)."""


def host_problem(out_dir: Path) -> str | None:
    """Why the host cannot take more results (None when it can): MIN_HOST_FREE free bytes, out_dir writable."""
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError as exc:  # e.g. the run directory was removed during the batch
        return f"{out_dir}: {exc}"
    if free < MIN_HOST_FREE:
        return f"{out_dir}: {free} bytes free on the host, need {MIN_HOST_FREE}"
    probe = out_dir / f".er_probe.{os.getpid()}.{threading.get_ident()}"
    try:
        probe.write_bytes(b"x")
        probe.unlink()
    except OSError as exc:
        return f"{out_dir}: not writable ({exc})"
    return None


def run_retry(run_fn: RunFn, record: dict, root: Path, entry: Entry, lane: int, out: Path, noop: tuple[int, ...] = (),
              lane_healthy: Callable[[int], bool] | None = None, stop: threading.Event | None = None) -> dict:
    """INFRA is retried, 3 attempts in all, then InfraError. INFRA from the log or a missing inner.json (the output may
    print or cause it) on a healthy lane is retried once like any INFRA; the second such result on a healthy lane is the
    output's ERROR. lane_healthy=None (gold: trusted code) never gates: its INFRA is always the environment. TIMEOUT is
    retried once. INFRA and TIMEOUT attempts are counted apart (at most 3 + 2 runs). A set `stop` ends the wait before
    a retry at once: Stopped."""
    infra = timeouts = gated = 0
    while True:
        r = run_fn(record, root, entry, lane, out, noop=noop)
        r["attempts"] = infra + timeouts + 1
        if r["status"] == "INFRA":
            source = r.get("infra_source")
            if source in ("log", "inner") and lane_healthy is not None and lane_healthy(lane):
                gated += 1
                if gated == 2:
                    return {**r, "status": "ERROR", "error": f"infra-like output, environment healthy ({source})"}
            infra += 1
            if infra == MAX_INFRA:
                raise db.InfraError(f"infrastructure failure after {infra} attempts ({source}): {r.get('error')}")
            delay = RETRY_SLEEP * infra
        elif r["status"] == "TIMEOUT" and timeouts < MAX_TIMEOUT - 1:
            timeouts += 1
            delay = 0.0
        else:
            return r
        if stop is None:
            time.sleep(delay)
        elif stop.wait(delay):
            raise Stopped(f"stopped before retrying {entry.record_id} {entry.condition}")


def atomic_write(path: Path, text: str) -> None:
    """Replace `path` by `text` (temporary file, fsync, os.replace): a crash leaves the old or the new file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_jsonl(path: Path, repair: bool = False) -> list[dict]:
    """Objects of a JSONL file ([] if absent). A partial LAST line (a write cut short, even inside a multi-byte
    character) is dropped with a warning; any other bad line (not UTF-8, not a JSON object) stops. repair=True
    (Batch.prepare, and the archives before an append) also fixes the file atomically: the partial line removed, a
    missing final newline added, so the next append cannot be glued to it. Readers never write."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1  # the newline-terminated lines end here; the rest is the last line
    try:
        text = data[:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{path}: a complete line is not UTF-8 ({exc.reason} at byte {exc.start})") from None
    try:
        text += data[end:].decode("utf-8")
    except UnicodeDecodeError:  # the write was cut inside a multi-byte character: the partial last line
        print(f"warning: {path}: dropped a partial last line", file=sys.stderr)
    parts = text.split("\n")
    rows = []
    for i, line in enumerate(parts):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            rows.append(obj)
            continue
        if i < len(parts) - 1:  # a complete line (newline-terminated) that is not a JSON object
            raise SystemExit(f"{path}: line {i + 1} is not a JSON object")
        print(f"warning: {path}: dropped a partial last line", file=sys.stderr)
        text = text[:text.rfind("\n") + 1]
    if text and not text.endswith("\n"):
        text += "\n"
    if repair and text.encode("utf-8") != data:
        atomic_write(path, text)
    return rows


def load_results(path: Path, repair: bool = False) -> list[dict]:
    """Rows of er_results.jsonl; a (record, condition, overlay) twice means two writers: stop."""
    rows = load_jsonl(path, repair)
    seen = Counter((r["record_id"], r["condition"], r.get("overlay_sha256")) for r in rows)
    dup = [k for k, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"{path}: {len(dup)} duplicate rows, e.g. {dup[0][:2]} (two batches on one run?)")
    return rows


def load_refs(path: Path, repair: bool = False) -> dict[str, dict]:
    """er_refs.jsonl by record_id; a later line for a record replaces an earlier one (a gate update)."""
    return {o["record_id"]: o for o in load_jsonl(path, repair)}


def row_key(r: dict) -> tuple[str, str, str]:
    return r["record_id"], r["condition"], r["overlay_sha256"]


def off_reference(r: dict, refs: dict[str, dict]) -> bool:
    """Controls run: the row's record has no reference line, or the row was scored against another version of it."""
    info = refs.get(r["record_id"])
    return info is None or any(r.get(k) != info.get(k) for k in REF_KEYS)


def move_lines(path: Path, gone: Callable[[dict], bool], note: dict | None = None, since: int | None = None) -> int:
    """Move the lines of a JSONL file for which gone() holds to <stem>.superseded.jsonl (appended and synced first),
    then rewrite the file atomically without them. Returns how many moved. since (the replay of a finish journal):
    the archive's size when the journal was written; a line already appended after it by the interrupted move is not
    appended again (a cut line there is dropped first, like any cut last line)."""
    rows = load_jsonl(path)
    moved = [r for r in rows if gone(r)]
    if moved:
        archive = path.with_name(path.stem + ".superseded.jsonl")
        load_jsonl(archive, repair=True)  # never glue onto a cut line
        todo = [json.dumps({**r, **(note or {})}, ensure_ascii=False) + "\n" for r in moved]
        if since is not None and archive.exists():
            there, left = Counter(ln + "\n" for ln in archive.read_bytes()[since:].decode("utf-8").split("\n")[:-1]), []
            for ln in todo:
                if there[ln]:  # appended by the interrupted move: counted once
                    there[ln] -= 1
                else:
                    left.append(ln)
            todo = left
        with archive.open("a", encoding="utf-8") as fh:
            fh.writelines(todo)
            fh.flush()
            os.fsync(fh.fileno())
        atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if not gone(r)))
    return len(moved)


@contextmanager
def sigint_ignored():
    """Ctrl-C is ignored while the block runs (finish(): archives and er_complete.json, well under a second); a
    KeyboardInterrupt there would act like a SIGKILL. Only the main thread can change a signal handler."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


WAITING = b"still waiting for the lanes' current runs (at most one run each)...\n"


@contextmanager
def sigint_while_lanes_run(stop: threading.Event, joining: threading.Event):
    """While the lanes run and are joined (main thread, over Python's default Ctrl-C handler only; an ignored SIGINT,
    e.g. under nohup, and a caller's own handler are left alone): the first Ctrl-C raises KeyboardInterrupt, as
    always, unless the batch is stopping (`stop`: by that Ctrl-C, or by a lane that failed) or every lane is done and
    the final join has begun (`joining`); then a Ctrl-C only writes WAITING to fd 2 and never raises (not
    KeyboardInterrupt; os.write, not print: no BrokenPipeError when stderr went into a `| tee` that the first Ctrl-C
    killed, no RuntimeError of a reentrant write to sys.stderr). So no exception can enter the join of a lane that may
    still run: on CPython 3.12 (bpo-45274) it would take that lane for a stopped one, and finish() would run and the
    run lock be released while the lane still runs."""
    if threading.current_thread() is not threading.main_thread() or \
            signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
        yield
        return

    def on_sigint(signum, frame):
        if not stop.is_set() and not joining.is_set():
            stop.set()
            raise KeyboardInterrupt
        try:
            os.write(2, WAITING)
        except OSError:  # no stderr to say it on (closed, or a pipe without a reader): keep waiting all the same
            pass
    old = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, on_sigint)
        yield
    finally:
        signal.signal(signal.SIGINT, old)


@contextmanager
def run_lock(run_dir: Path, shared: bool = False):
    """One writer per run directory (a batch, or tamper apply), across processes; refused at once when held.
    shared=True: an output run reading a controls run's references; output runs share it, a controls batch (exclusive)
    cannot start or supersede meanwhile, and an output run cannot start while a controls batch runs."""
    if shared and not run_dir.is_dir():
        raise SystemExit(f"{run_dir}: no such controls run")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ".er_batch.lock", "a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{run_dir}: a batch is running on this controls run" if shared else
                             f"{run_dir}: another batch is running on this run (or an output run is reading its "
                             f"references)") from None
        yield


class Batch:
    def __init__(self, root: Path, run: str, records: dict[str, dict], out: Path, head_root: Path, lanes: int = 2,
                 conditions: set[str] | None = None, limit: int | None = None, ref_run: str | None = None,
                 supersede: bool = False, run_fn: RunFn = run_once, lane_check: Callable[[int], None] = db.check_lane,
                 preflight: Callable[[list[int]], dict] = db.preflight, provenance: dict | None = None,
                 lane_healthy: Callable[[int], bool] = lane_healthy):
        self.root, self.run, self.records, self.head_root = root, run, records, head_root
        self.out_dir = out / run
        self.lanes, self.conditions, self.limit, self.ref_run, self.supersede = lanes, conditions, limit, ref_run, supersede
        self.run_fn, self.lane_check, self.preflight_fn, self.lane_healthy = run_fn, lane_check, preflight, lane_healthy
        self.results = self.out_dir / "er_results.jsonl"
        self.refs_path = (out / ref_run if ref_run else self.out_dir) / "er_refs.jsonl"
        self.lock = threading.RLock()  # the appends, and the journal of the streaks (save_journal)
        self.stop = threading.Event()  # set when a lane fails or on Ctrl-C: the other lanes stop between entries
        self.streaks: dict[int, list] = defaultdict(list)  # per lane, for the circuit breakers (journaled)
        self.lanes_running = False  # True from the start of run_lanes until every lane has returned
        self.suspect_rows: set[tuple[str, str, str]] = set()  # archived at the end, rerun on resume
        self.suspect_refs: set[str] = set()
        self.prov = None if provenance is None else {**provenance, "ref_run": ref_run}
        self.out = out

    # ---------- setup ----------

    def prepare(self) -> None:
        manifest_dir = self.root / self.run
        self.entries = load_manifest(manifest_dir)
        self.by_record: dict[str, dict[str, Entry]] = defaultdict(dict)
        for e in self.entries:
            if e.record_id not in self.records:
                raise SystemExit(f"{e.record_id}: not a Sylius record of the sealed Phase 2 data")
            try:
                verify_entry(self.root, e, self.records[e.record_id])
            except ManifestError as exc:
                raise SystemExit(f"manifest: {exc}")
            self.by_record[e.record_id][e.condition] = e
        unknown = sorted(set(self.conditions or ()) - {e.condition for e in self.entries})
        if unknown:
            raise SystemExit(f"--conditions: {unknown} not in the manifest")
        no_gold = sorted(rid for rid, c in self.by_record.items() if "gold" not in c)
        if no_gold and not self.ref_run:
            raise SystemExit(f"{len(no_gold)} records without a gold condition (e.g. {no_gold[0]}): a controls run "
                             f"needs one per record (or pass --ref-run)")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        why = host_problem(self.out_dir)
        if why:
            raise db.InfraError(why)
        if self.prov is None:  # the real path (tests inject provenance)
            if not self.head_root.is_dir() or head_commit(self.head_root) != HEAD:
                raise SystemExit(f"--head-root {self.head_root}: not a git clone at {HEAD}")
            if not head_clean(self.head_root):
                raise SystemExit(f"--head-root {self.head_root}: not a clean tree (git status --porcelain)")
            code = harness_code(HARNESS_ROOT)
            if not code:
                raise SystemExit(f"{HARNESS_ROOT}: harness_code cannot be computed (not a git checkout, or git "
                                 f"failed): a run needs the identity of the harness code that scores it")
            pf = self.preflight_fn(list(range(1, self.lanes + 1)))
            self.prov = {"image_id": image_id(IMAGE), "composer_lock_sha256": COMPOSER_LOCK_SHA256,
                         "phase2_sha256sums_sha256": seal_sha256(), "schema_fingerprint": pf["schema_fingerprint"],
                         "results_path": str(self.results.resolve()), "head_root": str(self.head_root.resolve()),
                         "harness_code": code, "harness_dirty": harness_dirty(HARNESS_ROOT),
                         "harness_commit": head_commit(HARNESS_ROOT), "ref_run": self.ref_run}
        meta = {**self.prov, "manifest_root": str(self.root.resolve()), "manifest_run": self.run,
                "manifest_sha256": sha256_file(manifest_dir / "manifest.jsonl")}
        pre = self.out_dir / "er_preflight.json"
        if pre.exists():
            old = json.loads(pre.read_text(encoding="utf-8"))
            for k in RESUME_KEYS:
                if old.get(k) == meta.get(k):
                    continue
                if k in ("harness_code", "harness_dirty"):
                    raise SystemExit(f"{pre}: {k} changed since this run started ({old.get(k)!r} -> {meta.get(k)!r}, "
                                     f"harness_commit {old.get('harness_commit')} -> {meta.get('harness_commit')}): "
                                     f"the code under er/ changed since this run started: restore it or start a new "
                                     f"run")
                raise SystemExit(f"{pre}: {k} changed since this run started ({old.get(k)!r} -> {meta.get(k)!r}); "
                                 f"start a new run")
        rids = sorted(self.by_record)
        self.todo = rids[:self.limit] if self.limit else rids
        self.scope = {(rid, c): e.overlay_sha256 for rid in self.todo for c, e in self.by_record[rid].items()
                      if self.selected(c)}
        self.expected = {(e.record_id, e.condition): e.overlay_sha256 for e in self.entries}  # SKIP included
        self.manifest_sha256 = meta["manifest_sha256"]
        self.refs_info = load_refs(self.refs_path)  # read-only until every check passed
        if self.ref_run:
            self.check_ref_run(meta)
        rows = load_results(self.results)
        # the journal of a batch that died before the end of its finish() (in its run, or in finish itself): what it
        # would have archived is left out of every check below, and archived once they all passed
        state = self.out_dir / "er_complete.json"
        journal = json.loads(state.read_text(encoding="utf-8")).get("archiving") if state.exists() else None
        if journal:
            j_refs = set() if self.ref_run else set(journal["refs"])
            j_rows = {tuple(k) for k in journal["rows"] + journal["stale_rows"]}
            rows = [r for r in rows if r["record_id"] not in j_refs and row_key(r) not in j_rows]
            self.refs_info = {rid: o for rid, o in self.refs_info.items() if rid not in j_refs}
        current = {(e.record_id, e.condition): e.overlay_sha256 for e in self.entries}
        gold_now = {rid: c["gold"].overlay_sha256 for rid, c in self.by_record.items() if "gold" in c}
        # controls run: a reference frozen for another gold overlay is recomputed, with every row of its record
        moved = set() if self.ref_run else {rid for rid, info in self.refs_info.items()
                                            if rid in gold_now and info.get("gold_overlay_sha256") != gold_now[rid]}

        def old_reference(r: dict) -> bool:
            """The row was scored against another gold reference than the current one."""
            rid = r["record_id"]
            if self.ref_run:  # the frozen reference line of the controls run (gold overlay, traces, error)
                info = self.refs_info.get(rid, {})
                return any(r.get(k) != info.get(k) for k in REF_KEYS)
            return rid in moved or rid in gold_now and r.get("gold_overlay_sha256") != gold_now[rid]
        stale = {row_key(r) for r in rows
                 if current.get((r["record_id"], r["condition"])) != r["overlay_sha256"] or old_reference(r)}
        if (stale or moved) and not self.supersede:
            raise SystemExit(f"{len(stale)} rows scored with another overlay, against another gold reference or no "
                             f"longer in the manifest{f' (e.g. {sorted(stale)[0][:2]})' if stale else ''}, "
                             f"{len(moved)} references frozen for another gold overlay; pass --supersede to archive "
                             f"them and recompute")
        kept_refs = {rid: o for rid, o in self.refs_info.items() if rid not in moved}
        # controls run: a row without its reference line, or scored against another version of it, cannot be valid
        # (a crash between two archives, a gate update): archived and recomputed, no --supersede needed
        orphans = set() if self.ref_run else {row_key(r) for r in rows
                                              if row_key(r) not in stale and off_reference(r, kept_refs)}
        # read only: a corrupt line (not a cut last one) of an archive or of er_canary.jsonl stops here
        for path in self.archive_paths() + [self.out_dir / "er_canary.jsonl"]:
            load_jsonl(path)
        # every check passed: only now write (a refused run leaves the run directory as it was)
        atomic_write(pre, json.dumps(meta, indent=1, ensure_ascii=False))
        if journal and "archive_bytes" not in journal:  # died in its run: no archive append of it yet
            journal["archive_bytes"] = self.archive_sizes()
        # a SIGKILL leaves no stale complete: true (and the journal stays until its moves are done)
        self.write_complete({"complete": False, "running": True, **({"archiving": journal} if journal else {})})
        load_results(self.results, repair=True)
        if not self.ref_run:  # a --ref-run's refs belong to its controls run: read-only
            load_refs(self.refs_path, repair=True)
        load_jsonl(self.out_dir / "er_canary.jsonl", repair=True)
        if journal:
            print("note: the last batch died before it had archived what its journal names (er_complete.json, "
                  "\"archiving\"): archived now, and recomputed", file=sys.stderr)
            self.archive(journal)
            self.write_complete({"complete": False, "running": True})
        if moved:  # the reference lines first: rows left without one are orphans on the next prepare
            move_lines(self.refs_path, lambda o: o["record_id"] in moved)
            self.refs_info = kept_refs
        if stale:
            move_lines(self.results, lambda r: row_key(r) in stale)
        if orphans:
            print(f"note: {len(orphans)} rows without a matching reference line archived and recomputed",
                  file=sys.stderr)
            move_lines(self.results, lambda r: row_key(r) in orphans, {"stale_ref": True})
        self.done = {row_key(r) for r in rows} - stale - orphans

    def check_ref_run(self, meta: dict) -> None:
        """--ref-run: the controls run must be unchanged since its freeze and match this run record by record (same
        gold overlay as the frozen reference, same base tree for every output) and complete (every row of its
        manifest; a smoke run with --limit is not). Both runs ran the same harness code, committed and clean."""
        ref_dir = self.out / self.ref_run
        try:
            ref_pre = json.loads((ref_dir / "er_preflight.json").read_text(encoding="utf-8"))
            ref_done = json.loads((ref_dir / "er_complete.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SystemExit(f"--ref-run {self.ref_run}: not a finished controls run ({exc.filename} "
                             f"missing)") from None
        if ref_done.get("complete") is not True or ref_done.get("manifest_sha256") != ref_pre["manifest_sha256"]:
            raise SystemExit(f"--ref-run {self.ref_run}: the controls run is not complete (er_complete.json); finish "
                             f"it first")
        commits = f"controls run harness_commit {ref_pre.get('harness_commit')}, this run {meta.get('harness_commit')}"
        for k in ("image_id", "phase2_sha256sums_sha256", "schema_fingerprint"):
            if ref_pre.get(k) != meta.get(k):
                raise SystemExit(f"--ref-run {self.ref_run}: {k} differs from this run")
        if ref_pre.get("harness_code") != meta.get("harness_code"):
            raise SystemExit(f"--ref-run {self.ref_run}: harness_code differs from this run ({commits}): the code "
                             f"under er/ changed since the controls run: start a new controls run")
        if ref_pre.get("harness_dirty") is not False or meta.get("harness_dirty") is not False:
            raise SystemExit(f"--ref-run {self.ref_run}: harness_dirty (controls run {ref_pre.get('harness_dirty')}, "
                             f"this run {meta.get('harness_dirty')}; {commits}): both need committed harness code, no "
                             f"local change under er/")
        root, run = Path(ref_pre["manifest_root"]), ref_pre["manifest_run"]
        if sha256_file(root / run / "manifest.jsonl") != ref_pre["manifest_sha256"]:
            raise SystemExit(f"--ref-run {self.ref_run}: its controls manifest changed after its references were "
                             f"frozen")
        self.ref_manifest = (root, run)
        self.ref_gold = {e.record_id: e for e in load_manifest(root / run) if e.condition == "gold"}
        for rid in self.todo:
            info, gold = self.refs_info.get(rid), self.ref_gold.get(rid)
            if info is None or gold is None:
                raise SystemExit(f"--ref-run {self.ref_run} has no reference (or no gold) for {rid}")
            if info.get("gold_overlay_sha256") != gold.overlay_sha256:
                raise SystemExit(f"{rid}: the frozen reference was computed for another gold overlay")
            for e in self.by_record[rid].values():
                if gold.skip is None and e.skip is None and e.base_files != gold.base_files:
                    raise SystemExit(f"{rid} {e.condition}: base tree differs from the frozen reference's")

    def retry(self, record: dict, root: Path, entry: Entry, lane: int, out: Path, noop: tuple[int, ...] = (),
              gold: bool = False) -> dict:
        """gold=True: the gold overlay (reference, gold condition, canary) is trusted code, never health-gated. The
        gate asks the host too: a full or read-only --out is the environment, not the output."""
        def healthy(lane: int) -> bool:
            return host_problem(self.out_dir) is None and self.lane_healthy(lane)
        return run_retry(self.run_fn, record, root, entry, lane, out, noop, None if gold else healthy, self.stop)

    def check_stop(self) -> None:
        """Between the runs of one entry (reference, gold, M1): a stopping batch leaves the entry for the resume."""
        if self.stop.is_set():
            raise Stopped("stopped between two runs")

    # ---------- references ----------

    def ref_base(self) -> Path:
        return self.refs_path.parent

    def compute_ref(self, record: dict, gold: Entry, lane: int) -> tuple[GoldRef | None, dict]:
        plan = step_plan(record)
        rel = f"er_runs/{slug_of(gold)}/_gold_ref"
        g = self.retry(record, self.root, gold, lane, self.out_dir / rel / "normal", gold=True)
        if g["status"] != "PASS":
            self.lane_check(lane)
        info = {"record_id": record["id"], "gold_overlay_sha256": gold.overlay_sha256, "ref_dir": rel,
                "gold_status": g["status"], "gold_trace_sha256": g.get("trace_sha256"), "gold_m1_status": None,
                "gold_m1_trace_sha256": None, "ref_error": None}
        if g["status"] != "PASS":
            info["ref_error"] = f"gold is {g['status']}"
            return None, info
        gm = None
        if plan.act_create:
            self.check_stop()
            gm = self.retry(record, self.root, gold, lane, self.out_dir / rel / "m1", noop=plan.act_create, gold=True)
            if gm["status"] != "PASS":
                self.lane_check(lane)
            info.update(gold_m1_status=gm["status"], gold_m1_trace_sha256=gm.get("trace_sha256"))
            if gm["trace"] is None:
                info["ref_error"] = f"gold M1 is {gm['status']}"
                return None, info
        return gold_reference(plan, g["trace"], gm["trace"] if gm else None), info

    def frozen_traces(self, info: dict) -> tuple[dict, dict | None]:
        """The pinned gold (normal, M1) traces of a reference line, sha256-checked."""
        def trace(kind: str, sha: str):
            p = self.ref_base() / info["ref_dir"] / kind / "trace.json"
            data = read_capped(p)
            if hashlib.sha256(data).hexdigest() != sha:
                raise SystemExit(f"{p}: frozen reference trace changed")
            return json.loads(data.decode("utf-8"))
        g = trace("normal", info["gold_trace_sha256"])
        return g, trace("m1", info["gold_m1_trace_sha256"]) if info.get("gold_m1_trace_sha256") else None

    def load_ref(self, record: dict, info: dict) -> GoldRef | None:
        """Rebuild a frozen reference from its pinned traces."""
        if info.get("ref_error"):
            return None
        return gold_reference(step_plan(record), *self.frozen_traces(info))

    @staticmethod
    def skip_info(gold: Entry) -> dict:
        return {"record_id": gold.record_id, "gold_overlay_sha256": gold.overlay_sha256, "ref_dir": None,
                "gold_status": "SKIP", "gold_trace_sha256": None, "gold_m1_status": None, "gold_m1_trace_sha256": None,
                "ref_error": f"gold is SKIP ({gold.skip})"}

    def reproduces(self, record: dict, ref: GoldRef, info: dict, gold_row: dict, m_trace: dict | None) -> bool:
        """The gold condition, scored as an output, matches its frozen reference, M1 run included."""
        if ref.act_evaluable and gold_row["trace_action"] != 1 or ref.then_evaluable and gold_row["trace_then"] != 1:
            return False
        if m_trace is None:  # M1 did not run (no evaluable Then)
            return True
        plan = step_plan(record)
        _, gm = self.frozen_traces(info)
        want = getattr(ref, "m1_action_trace", None)
        if want is None:
            want = action_trace(gm, plan.actions)
        return action_trace(m_trace, plan.actions) == want and statuses(m_trace) == statuses(gm)

    def gate(self, record: dict, ref: GoldRef | None, info: dict, gold_row: dict, m_trace: dict | None):
        """Determinism gate, whenever the gold condition is scored in a controls run (reference just computed or
        loaded): gold must reproduce the reference, else the record's reference is 'nondeterministic' (null rows)."""
        if ref is not None and not self.reproduces(record, ref, info, gold_row, m_trace):
            info["ref_error"], ref = "nondeterministic", None
            gold_row.update({k: None for k in ("trace_action", "trace_then", "er_trace", "act_evaluable",
                                               "n_then_evaluable")}, ref_error="nondeterministic")
        # True/False when the gold condition was scored against a working reference, else None
        info["deterministic"] = (None if info["ref_error"] not in (None, "nondeterministic")
                                 else info["ref_error"] is None)
        return ref

    def pick_canaries(self) -> dict[str, list[str]]:
        """--ref-run: records of this batch with a usable frozen reference: the first whose gold action trace has a
        token, the first with a numeric id, the first with an evaluable Then (its M1 run is checked too)."""
        rules = {"tok": lambda ref: any(k[0] == "http" and "{tok" in k[2] for k in ref.action_trace),
                 "n": lambda ref: any(k[0] == "http" and "{n}" in k[2] for k in ref.action_trace),
                 "then": lambda ref: bool(ref.then_evaluable)}
        usable = []
        for rid in self.todo:
            info, gold = self.refs_info.get(rid), self.ref_gold.get(rid)
            if info and not info.get("ref_error") and gold is not None and gold.skip is None:
                usable.append((rid, self.load_ref(self.records[rid], info)))
        picked: dict[str, list[str]] = {}
        for kind, rule in rules.items():
            rid = next((rid for rid, ref in usable if rule(ref)), None)
            if rid is not None:
                picked.setdefault(rid, []).append(kind)
        if not picked:
            raise SystemExit(f"--ref-run {self.ref_run}: no canary (no record of this batch has a usable reference "
                             f"with a token, a numeric id or an evaluable Then)")
        return picked

    def canary(self, lanes: list[int] | range, trigger: str) -> None:
        """Re-run the canaries' gold on each lane: it must PASS with the frozen action trace (and, for the Then canary,
        the frozen M1 statuses and M1 action trace), else the environment drifted: InfraError."""
        root, _ = self.ref_manifest
        canaries = self.out_dir / "er_canary.jsonl"
        with self.lock:  # one directory per canary call: _canary/<trigger>_<lines already written>/
            seq = len(load_jsonl(canaries))
        for rid, kinds in self.canaries.items():
            record, gold, info = self.records[rid], self.ref_gold[rid], self.refs_info[rid]
            verify_entry(root, gold, record)
            plan, ref = step_plan(record), self.load_ref(record, info)
            _, gm = self.frozen_traces(info)
            for lane in lanes:
                out = self.out_dir / "er_runs" / slug_of(gold) / "_canary" / f"{trigger}_{seq}"
                r = self.retry(record, root, gold, lane, out / f"lane{lane}_normal", gold=True)
                why = None if r["status"] == "PASS" and action_trace(r["trace"], plan.actions) == ref.action_trace \
                    else f"gold is {r['status']} or its action trace differs from the frozen one"
                m = None
                if why is None and "then" in kinds:
                    m = self.retry(record, root, gold, lane, out / f"lane{lane}_m1", noop=plan.act_create, gold=True)
                    want = getattr(ref, "m1_action_trace", None)
                    if want is None:
                        want = action_trace(gm, plan.actions)
                    if m["trace"] is None or statuses(m["trace"]) != statuses(gm) or \
                            action_trace(m["trace"], plan.actions) != want:
                        why = f"gold M1 is {m['status']} or differs from the frozen M1 run"
                self.write(canaries, {
                    "trigger": trigger, "lane": lane, "record_id": rid, "kinds": kinds, "status": r["status"],
                    "trace_sha256": r.get("trace_sha256"), "m1_status": m and m["status"],
                    "m1_trace_sha256": m and m.get("trace_sha256"), "ok": why is None, "error": why})
                if why is not None:
                    raise db.InfraError(f"canary {rid} on lane {lane} ({trigger}): {why}")

    # ---------- one entry ----------

    def row(self, entry: Entry, record: dict, info: dict) -> dict:
        plan = step_plan(record)
        return {"record_id": entry.record_id, "condition": entry.condition, "overlay_sha256": entry.overlay_sha256,
                "er": 0, "er_status": None, "skip": entry.skip, "n_act_create": len(plan.act_create),
                "n_then_create": len(plan.then_create), "trace_action": None, "trace_then": None, "er_trace": None,
                "act_evaluable": None, "n_then_evaluable": None, "mutation_status": None,
                "mutation_trace_sha256": None, "tamper_flags": [], "attempts": 0, "score_error": None,
                **{k: None for k in KEEP}, **{k: info.get(k) for k in REF_KEYS}, **self.prov}

    def process(self, entry: Entry, record: dict, ref: GoldRef | None, info: dict, lane: int) -> dict:
        return self.score(entry, record, ref, info, lane)[0]

    def score(self, entry: Entry, record: dict, ref: GoldRef | None, info: dict, lane: int) -> tuple[dict, dict | None]:
        """The row of one entry, and the trace of its M1 run (None when M1 did not run or left no trace)."""
        plan = step_plan(record)
        row = self.row(entry, record, info)
        if entry.skip is not None:
            row["er_status"] = "SKIP"
            if ref is not None:
                row.update(score_missing(ref))
            return row, None
        row["tamper_flags"] = tamper_flags(self.root, entry, self.head_root)
        out = self.out_dir / "er_runs" / slug_of(entry) / entry.condition
        trusted = entry.condition == "gold"
        n = self.retry(record, self.root, entry, lane, out / "normal", gold=trusted)
        if n["status"] != "PASS":
            self.lane_check(lane)
        row.update({k: n.get(k) for k in KEEP})
        row.update(attempts=n["attempts"], er_status=n["status"], er=int(n["status"] == "PASS"))
        if ref is None:
            return row, None
        if n["trace"] is None:
            row.update(score_missing(ref))
            return row, None
        m_trace = None
        if ref.then_evaluable:  # outside the try below: an exception of the run itself stops the batch
            self.check_stop()
            m = self.retry(record, self.root, entry, lane, out / "m1", noop=plan.act_create, gold=trusted)
            if m["status"] != "PASS":
                self.lane_check(lane)
            row["mutation_status"], row["mutation_trace_sha256"] = m["status"], m.get("trace_sha256")
            m_trace = m["trace"]
        try:
            row.update(score_output(plan, ref, n["trace"], m_trace))
        except Exception as exc:  # a trace the scorer cannot read: ER is kept, the trace parts score 0
            row.update(score_missing(ref), score_error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return row, m_trace

    def write(self, path: Path, obj: dict) -> None:
        with self.lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---------- one record ----------

    def selected(self, condition: str) -> bool:
        return self.conditions is None or condition in self.conditions or (not self.ref_run and condition == "gold")

    def wanted(self, rid: str) -> list[Entry]:
        return [e for c, e in sorted(self.by_record[rid].items())
                if self.selected(c) and (rid, c, e.overlay_sha256) not in self.done]

    def do_record(self, rid: str, lane: int) -> None:
        todo = self.wanted(rid)
        if not todo:
            return
        record = self.records[rid]
        if self.ref_run:
            info = self.refs_info[rid]
            ref = self.load_ref(record, info)
        else:
            ref, info, todo = self.controls_ref(rid, record, todo, lane)
        for e in todo:
            if self.stop.is_set():
                return
            row = self.process(e, record, ref, info, lane)
            counted = self.ref_run and e.skip is None  # a SKIP row neither extends nor breaks a streak
            if counted and row["er_status"] in NO_TRACE:  # journaled before its row: no unjournaled traceless row
                self.streak_add(lane, row_key(row))
            self.write(self.results, row)
            if counted:
                self.breaker(lane, row)

    def breaker(self, lane: int, row: dict) -> None:
        """--ref-run: BREAKER_K rows in a row on a lane without a trace: re-run the canaries there. If they fail, the
        environment broke: those rows are suspect (archived at the end, so a resume reruns them) and the batch stops.
        A row with a trace ends the streak (once that row is written)."""
        if row["er_status"] not in NO_TRACE:
            self.clear_streak(lane)
        elif len(self.streaks[lane]) >= BREAKER_K:
            self.check_output_streak(lane, "breaker")

    def streak_add(self, lane: int, key) -> None:
        with self.lock:
            self.streaks[lane].append(key)
            self.save_journal()

    def clear_streak(self, lane: int) -> None:
        with self.lock:
            if self.streaks[lane]:
                self.streaks[lane].clear()
                self.save_journal()

    def save_journal(self) -> None:
        """While the lanes run, er_complete.json names what a resume must archive should this process die now:
        the suspects, and every lane's current streak (as for a stopped batch: not checked yet). Written whenever
        they change, under the lock."""
        with self.lock:
            refs, rows = set(self.suspect_refs), set(self.suspect_rows)
            for streak in self.streaks.values():  # row keys (--ref-run) or record ids
                (rows if self.ref_run else refs).update(streak)
            journal = {"refs": sorted(refs), "rows": sorted(map(list, rows)), "stale_rows": []}
            self.write_complete({"complete": False, "running": True,
                                 **({"archiving": journal} if refs or rows else {})})

    def check_output_streak(self, lane: int, trigger: str) -> None:
        """--ref-run: the lane's rows without a trace in a row are the outputs' own if the host is fine and the
        canaries pass on that lane; else the environment broke: those rows are suspect and the batch stops."""
        streak = self.streaks[lane]
        why = host_problem(self.out_dir)
        if why:
            self.suspect_rows.update(streak)
            raise db.InfraError(f"lane {lane}: {len(streak)} rows in a row without a trace; {why}")
        try:
            self.canary([lane], trigger)
        except db.InfraError:
            self.suspect_rows.update(streak)
            raise
        self.clear_streak(lane)

    def gold_breaker(self, lane: int, rid: str, broken: str | None) -> None:
        """Controls run: GOLD_BREAKER broken references in a row on a lane. A healthy lane (and host) means the golds
        really are broken (bad data: null rows + ref_error, the run goes on); else the environment is suspect: those
        records are archived at the end (rerun on resume) and the batch stops."""
        if broken is None:
            self.clear_streak(lane)
            return
        self.streak_add(lane, rid)  # journaled before the record's reference line and rows are written
        if len(self.streaks[lane]) >= GOLD_BREAKER:
            self.check_gold_streak(lane, f"last {rid}: {broken}")

    def check_gold_streak(self, lane: int, detail: str) -> None:
        """Controls run: the lane's broken references in a row are broken golds if the host and the lane are healthy;
        else those records are suspect and the batch stops."""
        streak = self.streaks[lane]
        why = host_problem(self.out_dir) or (None if self.lane_healthy(lane) else f"lane {lane} is not healthy")
        if why is None:
            self.clear_streak(lane)
            return
        self.suspect_refs.update(streak)
        raise db.InfraError(f"lane {lane}: the gold reference is broken in {len(streak)} records in a row "
                            f"({detail}); {why}")

    def check_trailing_streaks(self) -> None:
        """A normal finish (every lane done): a lane whose last rows form a streak its breaker never reached is checked
        now, as the breaker would: the canaries on that lane (output run), the host and the lane (controls run)."""
        for lane, streak in sorted(self.streaks.items()):
            if streak and self.ref_run:
                self.check_output_streak(lane, "end")
            elif streak:
                self.check_gold_streak(lane, "at the end of the run")

    def controls_ref(self, rid: str, record: dict, todo: list[Entry], lane: int):
        """Controls run: the record's reference (frozen, else computed now) and its gold row first, gated."""
        gold = self.by_record[rid]["gold"]  # prepare checked that every record has one
        info = self.refs_info.get(rid)
        fresh = info is None
        if fresh:
            if gold.skip is not None:
                ref, info = None, self.skip_info(gold)
            else:
                ref, info = self.compute_ref(record, gold, lane)
                self.check_stop()
                self.gold_breaker(lane, rid, info["ref_error"])
        else:
            ref = self.load_ref(record, info)
        before = dict(info)
        gold_row = None
        if gold in todo:
            gold_row, m_trace = self.score(gold, record, ref, info, lane)
            if (info["gold_status"] == "PASS") != (gold_row["er_status"] == "PASS"):
                if fresh:  # computed seconds ago on this lane
                    raise db.InfraError(f"{rid}: gold is {gold_row['er_status']} but its reference run was "
                                        f"{info['gold_status']} (flaky environment)")
                # frozen in another environment (or a flaky gold): the reference line and every row of the record
                # are archived at the end and recomputed on resume; the rest of the record waits for that
                self.suspect_refs.add(rid)  # journaled at once: gold_breaker adds it to the lane's streak
                self.gold_breaker(lane, rid, f"gold is {gold_row['er_status']}, its frozen reference "
                                             f"{info['gold_status']}")
                return None, info, []
            if not fresh and gold_row["er_status"] == "PASS":
                self.gold_breaker(lane, rid, None)  # gold works on this lane
            ref = self.gate(record, ref, info, gold_row, m_trace)
        if fresh or info != before:  # a gate update is a new line: the later line wins
            self.write(self.refs_path, info)
            self.refs_info[rid] = info
        if gold_row is not None:
            self.write(self.results, gold_row)
            todo = [e for e in todo if e is not gold]
        return ref, info, todo

    def run_all(self) -> Path:
        controls = run_lock(self.out / self.ref_run, shared=True) if self.ref_run else nullcontext()
        with run_lock(self.out_dir), controls:
            self.prepare()
            stopped = None
            try:
                if self.ref_run:
                    self.canaries = self.pick_canaries()
                    self.canary(range(1, self.lanes + 1), "start")
                self.run_lanes(self.todo)
                self.check_trailing_streaks()
            except BaseException as exc:
                stopped = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                with sigint_ignored():
                    self.finish(stopped)
        return self.results

    def finish(self, stopped: str | None) -> None:
        """End of run_all, stopped or not. A stopped batch takes every lane's current breaker streak as suspect too:
        the failure that stopped it (a breaker on another lane, INFRA, a lane check, Ctrl-C) may have caused those
        rows. Suspect references and every row of their records, and suspect rows, are archived with "suspect": true
        (a resume reruns them); controls run: rows scored before a gate update of their reference line, with
        "stale_ref": true. Before the first move, the finish journal names every line to move and the archives'
        sizes in er_complete.json ("archiving"); a kill at any step leaves it, and the next prepare replays it
        exactly. er_complete.json then states whether every entry of the manifest has its row. No exception of a
        Ctrl-C enters the join of the lanes (sigint_while_lanes_run), so this runs once every lane has returned (and
        Ctrl-C is ignored while it runs). Should the join fail for another reason, a lane may still run: this holds
        the lock for its whole body (that lane appends only afterwards, and its journal save may then rewrite
        er_complete.json: running: true, the safe direction), folds a snapshot of the streaks, and er_complete.json
        keeps "running": true, as after a SIGKILL."""
        with self.lock:  # a lane still running (the join failed) appends only once this is done
            if stopped:
                for streak in [list(s) for s in list(self.streaks.values())]:  # row keys (--ref-run) or record ids
                    (self.suspect_rows if self.ref_run else self.suspect_refs).update(streak)
            rows = load_results(self.results)
            refs = set() if self.ref_run else set(self.suspect_refs)
            gone = {row_key(r) for r in rows if r["record_id"] in refs or row_key(r) in self.suspect_rows}
            stale = set()
            if not self.ref_run:
                kept = {rid: o for rid, o in load_refs(self.refs_path).items() if rid not in refs}
                stale = {row_key(r) for r in rows if row_key(r) not in gone and off_reference(r, kept)}
            if refs or gone or stale:
                journal = {"refs": sorted(refs), "rows": sorted(map(list, gone)),
                           "stale_rows": sorted(map(list, stale)), "archive_bytes": self.archive_sizes()}
                self.write_complete({"complete": False, "running": True, "archiving": journal})
                self.archive(journal)
                rows = load_results(self.results)
            suspect = sorted({k[:2] for k in gone})
            have = Counter((r["record_id"], r["condition"]) for r in rows)
            done = {(r["record_id"], r["condition"]) for r in rows
                    if self.expected.get((r["record_id"], r["condition"])) == r["overlay_sha256"]}
            missing = sorted(k for k in self.expected if k not in done)
            duplicates = sorted(k for k, n in have.items() if n > 1)
            self.write_complete({
                "expected": len(self.expected), "rows": len(rows), "missing": missing,
                "scope": {"conditions": sorted(self.conditions) if self.conditions else None, "limit": self.limit},
                "scope_missing": sorted(k for k in self.scope if k not in done),
                "extra": sorted(k for k in have if k not in self.expected), "duplicates": duplicates,
                "suspect": suspect, "stopped": stopped, "running": self.lanes_running,
                "complete": not missing and not duplicates and not self.lanes_running})
            if self.lanes_running:
                print("warning: a lane may still be running (the join of the lanes failed): er_complete.json says "
                      "running: true; once this process has exited, rerun the same command to resume", file=sys.stderr)

    def archive_paths(self) -> list[Path]:
        """The archives this run appends to. An output run never touches its controls run's files."""
        return [p.with_name(p.stem + ".superseded.jsonl") for p in [self.results] + ([] if self.ref_run else
                                                                                   [self.refs_path])]

    def archive_sizes(self) -> dict[str, int]:
        """The size of each archive of this run (a cut last line repaired first): where a journal's moves start."""
        sizes = {}
        for archive in self.archive_paths():
            load_jsonl(archive, repair=True)
            sizes[archive.name] = archive.stat().st_size if archive.exists() else 0
        return sizes

    def archive(self, journal: dict) -> None:
        """Move what the journal names: its references and every row of their records, and its rows ("suspect":
        true); its stale rows ("stale_ref": true). Replayable after a kill at any step: a line already moved is no
        longer there, and a line the interrupted move already appended is not appended again."""
        since, refs = journal["archive_bytes"], set(journal["refs"])
        rows, stale = {tuple(k) for k in journal["rows"]}, {tuple(k) for k in journal["stale_rows"]}
        if refs and not self.ref_run:  # the reference lines first, as for a supersede
            move_lines(self.refs_path, lambda o: o["record_id"] in refs, {"suspect": True},
                       since=since["er_refs.superseded.jsonl"])
        if refs or rows:
            move_lines(self.results, lambda r: r["record_id"] in refs or row_key(r) in rows, {"suspect": True},
                       since=since["er_results.superseded.jsonl"])
        if stale:
            move_lines(self.results, lambda r: row_key(r) in stale, {"stale_ref": True},
                       since=since["er_results.superseded.jsonl"])

    def write_complete(self, state: dict) -> None:
        """er_complete.json: whether every entry of the WHOLE manifest (SKIP included) has its row."""
        atomic_write(self.out_dir / "er_complete.json", json.dumps(
            {**state, "manifest_sha256": self.manifest_sha256}, indent=1, ensure_ascii=False))

    def run_lanes(self, todo: list[str]) -> None:
        """Shard the records over the lanes. The first exception (or Ctrl-C) stops every lane at its next run boundary
        (Stopped) and is re-raised once all lanes have returned, so nothing is written after run_all returns. A
        further Ctrl-C (or a first one once a lane failed, or in the final join of a batch whose lanes all finished)
        only waits (sigint_while_lanes_run): no exception enters the join, and the run lock is held until every lane
        returned."""
        def work(lane: int, rids: list[str]) -> None:
            try:
                for rid in rids:
                    if self.stop.is_set():
                        return
                    self.do_record(rid, lane)
            except BaseException:
                self.stop.set()
                raise
        joining = threading.Event()  # not `stop`: the end-of-run canary's retries use it
        with sigint_while_lanes_run(self.stop, joining):
            pool = ThreadPoolExecutor(max_workers=self.lanes)
            try:
                self.lanes_running = True
                futures = [pool.submit(work, lane + 1, todo[lane::self.lanes]) for lane in range(self.lanes)]
                wait(futures, return_when=FIRST_EXCEPTION)
                joining.set()  # every lane is done, or the batch is stopping: a Ctrl-C now only waits
            except BaseException:  # KeyboardInterrupt in the main thread
                self.stop.set()
                raise
            finally:  # every lane returns before finish() and before the run lock is released
                pool.shutdown(wait=True, cancel_futures=True)
                self.lanes_running = False  # not when the join itself failed: a lane may still run
        errors = [f.exception() for f in futures if not f.cancelled() and f.exception() is not None]
        if errors:  # the lane that failed first, not one that merely saw the stop
            raise next((e for e in errors if not isinstance(e, Stopped)), errors[0])


def run_batch(root: Path, run: str, records: dict[str, dict], out: Path, head_root: Path, **kwargs) -> Path:
    return Batch(root, run, records, out, head_root, **kwargs).run_all()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="manifest root (output of scripts/write_overlays.py)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--head-root", type=Path, required=True, help="full clone at c096ca3, for the tamper scan")
    ap.add_argument("--ref-run", help="controls run whose frozen gold references to use")
    ap.add_argument("--lanes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--conditions", help="comma-separated subset of conditions")
    ap.add_argument("--limit", type=int, help="first N records (smoke runs)")
    ap.add_argument("--supersede", action="store_true", help="replace lines scored with a different overlay")
    a = ap.parse_args()
    try:
        print(run_batch(a.root, a.run, load_records(), a.out, a.head_root, lanes=a.lanes,
                        conditions=set(a.conditions.split(",")) if a.conditions else None, limit=a.limit,
                        ref_run=a.ref_run, supersede=a.supersede))
    except db.InfraError as exc:
        print(f"STOPPED (infrastructure): {exc}. Fix the environment (er/docker/init-db.sh) and rerun to resume.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
