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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import er.batch as batch
from er.batch import run_batch
from er.db import InfraError
from er.manifest import write_run

A = "src/Sylius/Behat/Context/Api/A.php"
REC = {"id": "Sylius/Sylius|f.feature|3|api_x", "suite": "api_x", "feature_path": "f.feature", "scenario_line": 3,
       "scenario_name": "s", "t_vis": 1, "background": None, "create": [{"path": A}],
       "scenario": "Scenario: s\n    Given a\n    When b\n    When c\n    Then d\n",
       "steps": ["a", "b", "c", "d"], "step_kinds": ["reuse", "create", "reuse", "create"]}
RECS = {REC["id"]: REC}
BASE = {A: "<?php\n"}
PROV = {"image_id": "sha256:test", "composer_lock_sha256": "lock", "phase2_sha256sums_sha256": "seal",
        "schema_fingerprint": "fp", "results_path": "x", "ref_run": None, "harness_code": "code", "harness_dirty": False}
batch.RETRY_SLEEP = 0


def trace(statuses, keys):
    types = ("Given", "When", "When", "Then")
    return {"steps": [{"i": i, "type": types[i], "text": t, "status": s} for i, (t, s) in enumerate(zip("abcd", statuses))],
            "events": [{"kind": "http", "step": 2, "method": "POST", "path": "/z", "body_keys": keys, "status": 201,
                        "route": "r", "token_values": []}]}


GOLD_N, GOLD_M = trace(["passed"] * 4, ["code", "members"]), trace(["passed", "noop", "passed", "failed"], ["code"])
STUB_N, STUB_M = trace(["passed"] * 4, ["code"]), trace(["passed", "noop", "passed", "passed"], ["code"])


def label(out):
    """What a run was for: '_gold_ref', '_canary' (any trigger) or the condition."""
    return "_canary" if "_canary" in out.parts else out.parts[-2]


class Fake:
    """Canned runs; `script` maps condition -> list of statuses to return in turn (then the default)."""

    def __init__(self, script=None, gold_trace=GOLD_N):
        self.calls, self.script, self.gold_trace = [], {k: list(v) for k, v in (script or {}).items()}, gold_trace

    def __call__(self, record, root, entry, lane, out, noop=()):
        self.calls.append((entry.condition, tuple(noop), label(out)))
        queued = self.script.get(entry.condition)
        status = queued.pop(0) if queued else None
        status, _, source = (status or "").partition(":")  # "INFRA:log" = INFRA with infra_source "log"
        if status in ("INFRA", "TIMEOUT", "ERROR", "OOM"):
            return {"status": status, "trace": None, "trace_sha256": None, "error": status.lower(),
                    **({"infra_source": source or None} if status == "INFRA" else {})}
        return self.emit(out, *self.canned(record, entry, bool(noop), status or None))

    def canned(self, record, entry, m, status):
        if status == "FAIL":  # the normal run fails its last (Then) step
            return trace(["passed", "passed", "passed", "failed"], ["code", "members"]), "FAIL"
        if entry.condition == "gold":
            return (GOLD_M, "FAIL") if m else (self.gold_trace, "PASS")
        if entry.condition == "neg":
            return trace(["passed", "undefined", "passed", "undefined"], ["code"]), "UNDEFINED"
        return (STUB_M, "FAIL") if m else (STUB_N, "PASS")

    @staticmethod
    def emit(out, t, s):
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.json").write_text(json.dumps(t))
        return {"status": s, "seconds": 1.0, "trace": t, "trace_sha256": hashlib.sha256((out / "trace.json").read_bytes()).hexdigest()}


def make(tmp_path, conds=("neg", "gold", "stub"), run="r", extra=()):
    items = [{"record": REC, "condition": c, "slug": "f_3", "base": BASE,
              "files": {} if c == "neg" else {A: f"<?php // {c}\n"}} for c in conds]
    items += [{"record": REC, "condition": c, "slug": "f_3", "base": BASE, "files": {}, "skip": reason}
              for c, reason in extra]
    write_run(tmp_path / "m", run, items)


def recs(n):
    out = {}
    for i in range(n):
        r = {**REC, "id": f"Sylius/Sylius|f{i}.feature|3|api_x", "feature_path": f"f{i}.feature"}
        out[r["id"]] = r
    return out


def make_many(tmp_path, records, conds=("gold", "stub"), run="r", base=BASE):
    items = []
    for i, r in enumerate(records.values()):
        items += [{"record": r, "condition": c, "slug": f"f{i}_3", "base": base,
                   "files": {} if c == "neg" else {A: f"<?php // {c}\n"}} for c in conds]
    write_run(tmp_path / "m", run, items)


def go(tmp_path, fake, run="r", checks=None, **kw):
    lane_check = (lambda lane: checks.append(lane)) if checks is not None else (lambda lane: None)
    kw.setdefault("lane_healthy", lambda lane: True)
    return run_batch(tmp_path / "m", run, RECS, tmp_path / "out", tmp_path, lanes=1, run_fn=fake,
                     lane_check=lane_check, provenance=PROV, **kw)


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def rows(path):
    got = lines(path)
    keys = [(o["record_id"], o["condition"]) for o in got]
    assert len(keys) == len(set(keys)), "duplicate (record_id, condition) rows"
    return {o["condition"]: o for o in got}


def keyed(path):
    """Rows by (record_id, condition); a key twice fails."""
    got = lines(path)
    keys = [(o["record_id"], o["condition"]) for o in got]
    assert len(keys) == len(set(keys)), "duplicate (record_id, condition) rows"
    return dict(zip(keys, got))


def test_controls_are_scored_and_the_reference_is_frozen(tmp_path):
    make(tmp_path)
    fake, checks = Fake(), []
    out = rows(go(tmp_path, fake, checks=checks))
    g, s, n = out["gold"], out["stub"], out["neg"]
    assert (g["er"], g["trace_action"], g["trace_then"], g["er_trace"]) == (1, 1, 1, 1)
    assert (s["er"], s["trace_action"], s["trace_then"], s["er_trace"]) == (1, 0, 0, 0)
    assert n["er_status"] == "UNDEFINED" and n["er_trace"] == 0 and s["n_then_evaluable"] == 1
    assert fake.calls[:2] == [("gold", (), "_gold_ref"), ("gold", (1,), "_gold_ref")]
    assert len(fake.calls) == 2 + 3 * 2 and checks  # lane checked after the non-PASS runs
    ref = json.loads((tmp_path / "out/r/er_refs.jsonl").read_text())
    assert ref["deterministic"] is True and ref["ref_error"] is None


def test_error_timeout_skip_and_failed_m1_score_zero(tmp_path):
    make(tmp_path, conds=("gold", "m.s1.err", "m.s1.slow", "m.s1.badm1"), extra=[("m.s1.none", "extract: none")])
    fake = Fake({"m.s1.err": ["ERROR"], "m.s1.slow": ["TIMEOUT", "TIMEOUT"], "m.s1.badm1": ["PASS", "ERROR"]})
    out = rows(go(tmp_path, fake))
    for c in ("m.s1.err", "m.s1.slow"):
        assert (out[c]["er"], out[c]["trace_action"], out[c]["trace_then"], out[c]["er_trace"]) == (0, 0, 0, 0)
    assert out["m.s1.slow"]["er_status"] == "TIMEOUT" and out["m.s1.slow"]["attempts"] == 2
    skip = out["m.s1.none"]
    assert (skip["er_status"], skip["er"], skip["er_trace"], skip["skip"]) == ("SKIP", 0, 0, "extract: none")
    assert out["m.s1.badm1"]["mutation_status"] == "ERROR" and out["m.s1.badm1"]["trace_then"] == 0


def test_persistent_infra_stops_without_recording_then_resumes(tmp_path):
    make(tmp_path)
    with pytest.raises(InfraError):
        go(tmp_path, Fake({"stub": ["INFRA", "INFRA", "INFRA"]}))
    done = rows(tmp_path / "out/r/er_results.jsonl")
    assert "stub" not in done and "gold" in done
    out = rows(go(tmp_path, Fake()))  # environment fixed: resumes with the stub only
    assert out["stub"]["er_status"] == "PASS" and len(out) == 3


def test_partial_last_line_is_dropped_on_resume(tmp_path):
    make(tmp_path)
    path = go(tmp_path, Fake())
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-1]) + "\n" + text[-1][:40])  # interrupted mid-write
    again = Fake()
    out = rows(go(tmp_path, again))
    assert len(out) == 3 and again.calls  # the cut line is recomputed


def test_conditions_filter_keeps_gold_for_the_reference(tmp_path):
    make(tmp_path)
    out = rows(go(tmp_path, Fake(), conditions={"stub"}))
    assert set(out) == {"gold", "stub"}


def test_nondeterministic_gold_nulls_the_record(tmp_path):
    make(tmp_path)

    class Flaky(Fake):
        n = 0

        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "gold" and not noop:
                Flaky.n += 1
                self.gold_trace = GOLD_N if Flaky.n == 1 else trace(["passed"] * 4, ["code", "members", "extra"])
            return super().__call__(record, root, entry, lane, out, noop)
    out = rows(go(tmp_path, Flaky()))
    assert out["gold"]["ref_error"] == "nondeterministic" and out["stub"]["trace_action"] is None
    assert json.loads((tmp_path / "out/r/er_refs.jsonl").read_text())["deterministic"] is False


def test_ref_run_needs_no_gold_and_a_drifted_canary_stops(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())  # controls run "r": freezes the reference and records its manifest in er_preflight.json
    make(tmp_path, conds=("deepseek.s42.full_aware",), run="o")  # an output run: no gold in its manifest
    fake = Fake()
    out = rows(go(tmp_path, fake, run="o", ref_run="r"))
    assert out["deepseek.s42.full_aware"]["trace_action"] == 0 and out["deepseek.s42.full_aware"]["ref_run"] == "r"
    assert fake.calls[0] == ("gold", (), "_canary") and ("gold", (1,), "_gold_ref") not in fake.calls
    make(tmp_path, conds=("deepseek.s1.x",), run="o2")
    with pytest.raises(InfraError, match="canary"):
        go(tmp_path, Fake(gold_trace=STUB_N), run="o2", ref_run="r")


def test_regenerated_manifest_needs_supersede(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    items = [{"record": REC, "condition": c, "slug": "f_3", "base": BASE, "files": {} if c == "neg" else {A: "<?php // v2\n"}}
             for c in ("neg", "gold", "stub")]
    write_run(tmp_path / "m", "r", items)
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, Fake())
    out = rows(go(tmp_path, Fake(), supersede=True))
    # gold changed: its reference is recomputed, so every row of the record (neg included) is archived
    assert len(out) == 3 and (tmp_path / "out/r/er_results.superseded.jsonl").read_text().count("\n") == 3


def test_a_scoring_exception_keeps_er_and_records_score_error(tmp_path):
    make(tmp_path, conds=("gold", "m.s1.short"))

    class Short(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            r = super().__call__(record, root, entry, lane, out, noop)
            if entry.condition == "m.s1.short" and not noop:
                r["trace"] = {"steps": r["trace"]["steps"][:2], "events": []}  # fewer steps than the record
            return r
    out = rows(go(tmp_path, Short()))
    s = out["m.s1.short"]
    assert (s["er_status"], s["er"], s["error"]) == ("PASS", 1, None)  # ER, the primary metric, is kept
    assert (s["trace_action"], s["trace_then"], s["er_trace"]) == (0, 0, 0) and s["score_error"].startswith("IndexError")
    assert out["gold"]["score_error"] is None


def test_an_exception_in_the_m1_run_stops_the_batch_unrecorded(tmp_path):
    make(tmp_path, conds=("gold", "m.s1.ok"))

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "m.s1.ok" and noop:  # e.g. the docker CLI missing for a moment
                raise FileNotFoundError(2, "No such file or directory: 'docker'")
            return super().__call__(record, root, entry, lane, out, noop)
    with pytest.raises(FileNotFoundError):
        go(tmp_path, F())
    assert set(rows(tmp_path / "out/r/er_results.jsonl")) == {"gold"}  # m.s1.ok is rerun on resume
    assert rows(go(tmp_path, Fake()))["m.s1.ok"]["er_status"] == "PASS"


# ---------- readers: repairs only in Batch.prepare, atomic; refs share the reader ----------

def test_missing_final_newline_is_repaired_on_resume(tmp_path):
    make(tmp_path, conds=("gold", "stub", "neg", "m.s1.x"))
    path = go(tmp_path, Fake(), conditions={"stub"})  # gold + stub done
    path.write_text(path.read_text().rstrip("\n"))  # killed right before the final "\n"
    go(tmp_path, Fake())
    assert len(path.read_text().splitlines()) == 4 and len(rows(path)) == 4


def test_interrupted_mid_write_resumes_each_entry_exactly_once(tmp_path):
    make(tmp_path, extra=[("m.s1.none", "extract: none")])
    with pytest.raises(InfraError):
        go(tmp_path, Fake({"stub": ["INFRA"] * 3}))
    path = tmp_path / "out/r/er_results.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text + text.splitlines()[0][:40], encoding="utf-8")  # killed while writing the next line
    again = Fake()
    go(tmp_path, again)
    assert sorted(o["condition"] for o in lines(path)) == ["gold", "m.s1.none", "neg", "stub"]  # one line each
    assert [c[:2] for c in again.calls] == [("stub", ()), ("stub", (1,))]  # frozen reference, not recomputed
    assert len(lines(tmp_path / "out/r/er_refs.jsonl")) == 1


def test_plain_readers_never_write_the_results_file(tmp_path):
    make(tmp_path)
    path = go(tmp_path, Fake())
    path.write_text(path.read_text() + '{"record_id": "cut')  # a batch is writing its next line
    before = path.read_bytes()
    assert len(batch.load_results(path)) == 3 and path.read_bytes() == before


def test_the_repair_is_atomic(tmp_path, monkeypatch):
    make(tmp_path)
    path = go(tmp_path, Fake())
    path.write_text(path.read_text() + '{"record_id": "cut')
    before = path.read_bytes()

    def killed(*a, **kw):
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        batch.load_results(path, repair=True)
    assert path.read_bytes() == before  # the complete rows are all still there


def test_duplicate_rows_are_refused(tmp_path):
    make(tmp_path)
    path = go(tmp_path, Fake())
    text = path.read_text()
    path.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(SystemExit, match="duplicate"):
        batch.load_results(path)


def test_refs_tolerate_a_partial_last_line_and_later_lines_win(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    refs = tmp_path / "out/r/er_refs.jsonl"
    line = refs.read_text().splitlines()[0]
    later = {**json.loads(line), "ref_error": "nondeterministic", "deterministic": False}
    refs.write_text(line + "\n" + json.dumps(later) + "\n" + line[:60])  # killed while writing a third line
    res = tmp_path / "out/r/er_results.jsonl"
    res.write_text("".join(x + "\n" for x in res.read_text().splitlines() if json.loads(x)["condition"] != "stub"))
    out = rows(go(tmp_path, Fake()))
    assert out["stub"]["ref_error"] == "nondeterministic" and out["stub"]["trace_action"] is None
    assert refs.read_text().endswith("\n") and len(lines(refs)) == 2


# ---------- stopping: one failing lane stops the others; Ctrl-C stops; one batch per run ----------

def test_an_infra_stop_on_lane_1_stops_lane_2_quickly(tmp_path):
    records = recs(8)
    make_many(tmp_path, records)
    first, lane2 = sorted(records)[0], set(sorted(records)[1::2])
    t = {}

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if lane == 1 and record["id"] == first and entry.condition == "stub":
                t["stop"] = time.monotonic()
                return {"status": "INFRA", "infra_source": "docker", "trace": None, "error": "docker: gone"}
            if lane == 2:
                time.sleep(0.05)
            return super().__call__(record, root, entry, lane, out, noop)
    with pytest.raises(InfraError):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=2, run_fn=F(),
                  lane_check=lambda lane: None, provenance=PROV)
    assert time.monotonic() - t["stop"] < 0.5  # lane 2 stopped after its current entry
    done = {r["record_id"] for r in lines(tmp_path / "out/r/er_results.jsonl")}
    assert len(done & lane2) <= 1


def test_ctrl_c_stops_the_lanes_and_propagates(tmp_path, monkeypatch):
    records = recs(8)
    make_many(tmp_path, records)
    res = tmp_path / "out/r/er_results.jsonl"

    class Slow(Fake):
        def __call__(self, *a, **kw):
            time.sleep(0.02)
            return super().__call__(*a, **kw)

    def interrupted(fs, return_when=None):  # Ctrl-C arrives while the main thread waits for the lanes
        deadline = time.monotonic() + 10
        while not (res.exists() and res.read_text()) and time.monotonic() < deadline:
            time.sleep(0.01)
        raise KeyboardInterrupt
    monkeypatch.setattr(batch, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=2, run_fn=Slow(),
                  lane_check=lambda lane: None, provenance=PROV)
    n = len(lines(res))
    time.sleep(0.3)
    assert len(lines(res)) == n < 16  # stopped early, and no lane is still writing


@pytest.mark.parametrize("k", [0, 1, 2, 3])  # lane 2 fails during lane 1's reference, ref M1, gold or gold M1 run
def test_a_stop_is_seen_between_the_reference_and_the_gold_runs(tmp_path, k):
    records = recs(4)
    make_many(tmp_path, records)  # gold, stub; lane 1 has f0 then f2
    inside = threading.Event()
    lane1 = []

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if lane == 2:
                inside.wait(5)
                raise RuntimeError("lane 2 broke")
            lane1.append((entry.condition, label(out), noop))
            if len(lane1) == k + 1:
                inside.set()
                b.stop.wait(5)  # until lane 2's exception sets the batch's stop event
            return super().__call__(record, root, entry, lane, out, noop)
    b = batch.Batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=2, run_fn=F(),
                    lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)
    with pytest.raises(RuntimeError, match="lane 2 broke"):  # not lane 1's own stop
        b.run_all()
    assert len(lane1) == k + 1, lane1  # no run started after the stop: not the next entry, not the next record
    assert len(batch.load_jsonl(tmp_path / "out/r/er_results.jsonl")) == (k == 3)  # only a finished gold row


def test_a_retry_wait_ends_when_the_batch_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "RETRY_SLEEP", 5.0)
    records = recs(2)
    make_many(tmp_path, records)
    waiting = threading.Event()

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if lane == 1:  # the environment hiccups: lane 1 waits before its retry
                waiting.set()
                return {"status": "INFRA", "infra_source": "docker", "trace": None, "error": "x"}
            waiting.wait(5)
            time.sleep(0.05)
            raise RuntimeError("lane 2 broke")
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="lane 2 broke"):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=2, run_fn=F(),
                  lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)
    assert time.monotonic() - t0 < 2


CHILD = """
import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, whatever the parent ignores
from tests.test_er_batch import Fake, PROV, make_many, recs
from er.batch import run_batch
tmp = Path(sys.argv[1]); records = recs(12); make_many(tmp, records)
class Slow(Fake):
    def __call__(self, *a, **kw):
        time.sleep(0.1)
        return super().__call__(*a, **kw)
run_batch(tmp / "m", "r", records, tmp / "out", tmp, lanes=2, run_fn=Slow(), lane_check=lambda l: None, provenance=PROV)
"""


def test_a_real_sigint_stops_a_two_lane_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path)], cwd=root, stderr=subprocess.PIPE, text=True)
    res = tmp_path / "out/r/er_results.jsonl"
    deadline = time.monotonic() + 20
    while not (res.exists() and res.read_text().count("\n") >= 2) and time.monotonic() < deadline:
        time.sleep(0.02)
    p.send_signal(signal.SIGINT)
    t0 = time.monotonic()
    p.wait(timeout=20)
    assert time.monotonic() - t0 < 3 and p.returncode != 0 and "KeyboardInterrupt" in p.stderr.read()
    assert len(lines(res)) < 24


def test_finish_folds_a_snapshot_of_the_streaks(tmp_path):
    make(tmp_path)
    b = batch.Batch(tmp_path / "m", "r", RECS, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    b.prepare()
    b.streaks[1].append(REC["id"])

    class LateLane(set):  # a lane still running (the join failed) starts a streak while finish() folds the others
        def update(self, *others):
            b.streaks[len(b.streaks) + 1].append("Sylius/Sylius|late.feature|3|api_x")
            super().update(*others)
    b.suspect_refs = LateLane()
    b.finish("KeyboardInterrupt: ")  # no "RuntimeError: dictionary changed size during iteration"
    assert REC["id"] in b.suspect_refs


class Reentrant:
    """A sys.stderr whose write fails as a blocked stderr write does when another Ctrl-C interrupts it."""

    def write(self, text):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stderr>'>")

    def flush(self):
        pass


@pytest.mark.parametrize("stderr", ["ok", "reentrant", "broken_pipe"])
@pytest.mark.parametrize("kind", ["ctrl_c", "lane_failure"])
def test_a_real_ctrl_c_during_the_join_keeps_the_run_lock_until_every_lane_returned(tmp_path, monkeypatch, capfd,
                                                                                     kind, stderr):
    """REAL SIGINTs while the main thread joins a lane that is still inside its run. On CPython 3.12 (bpo-45274) a
    KeyboardInterrupt inside Thread.join takes that lane for a stopped one: the join returned, finish() ran and the run
    lock was released while the lane still ran, and its row landed after finish(). ctrl_c: the first Ctrl-C arrives
    while the main thread waits for the lanes, the second inside the join; lane_failure: lane 1 fails, and one Ctrl-C
    arrives inside the join of lane 2. stderr: the handler's notice must not raise into the join either;
    reentrant: sys.stderr fails as a blocked write interrupted by another Ctrl-C does; broken_pipe: fd 2, and a
    sys.stderr over it, is a pipe whose reader is gone (e.g. `| tee`, killed by the first Ctrl-C)."""
    records = recs(2)  # lane 1: f0, lane 2: f1; gold and stub each
    make_many(tmp_path, records)
    run_dir, (f0, f1) = tmp_path / "out/r", sorted(records)
    started, release, joining, lane1_done = (threading.Event() for _ in range(4))
    order, probes, got, late = [], [], [], []

    class Held(Fake):  # lane 2 is inside the last run of its last entry (stub M1): it writes that row right after
        def __call__(self, record, root, entry, lane, out, noop=()):
            if lane == 2 and noop and entry.condition == "stub":
                started.set()
                release.wait(10)
            if lane == 1 and kind == "lane_failure":
                started.wait(10)
                raise RuntimeError("lane 1 broke")
            return super().__call__(record, root, entry, lane, out, noop)

    class Logged(batch.Batch):
        def write(self, path, obj):
            super().write(path, obj)
            if path.name == "er_results.jsonl":
                order.append(("row", obj["record_id"], obj["condition"]))
                if (obj["record_id"], obj["condition"]) == (f0, "stub"):
                    lane1_done.set()

        def finish(self, stopped):  # what is still running when finish() starts
            order.append(("finish", sum(t.name.startswith("ThreadPoolExecutor") for t in threading.enumerate())))
            super().finish(stopped)

    class Pool(ThreadPoolExecutor):  # a real join; it only says when the main thread enters it
        def shutdown(self, wait=True, *, cancel_futures=False):
            joining.set()
            super().shutdown(wait, cancel_futures=cancel_futures)

    def operator():
        started.wait(10)
        time.sleep(0.2)
        if kind == "ctrl_c":
            lane1_done.wait(10)  # lane 1 has written its last row
            os.kill(os.getpid(), signal.SIGINT)  # the first Ctrl-C: the main thread waits for the lanes
        joining.wait(10)
        time.sleep(0.3)  # the main thread is now blocked in Thread.join: lane 2 is held in its run
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.3)
        try:  # an operator resumes at once, while lane 2 still runs
            with batch.run_lock(run_dir):
                probes.append("acquired")
        except SystemExit as exc:
            probes.append(str(exc))
        release.set()
    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    b = Logged(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Held(), **{**KW1, "lanes": 2})
    t = threading.Thread(target=operator)
    real_stderr, fd2, broken = sys.stderr, None, None
    before = signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, whatever pytest inherited
    try:
        if stderr == "reentrant":
            sys.stderr = Reentrant()
        elif stderr == "broken_pipe":
            r, w = os.pipe()
            os.close(r)  # the reader is gone
            fd2 = os.dup(2)
            os.dup2(w, 2)
            os.close(w)
            sys.stderr = broken = open(2, "w", buffering=1, closefd=False)  # line-buffered, as the real stderr
        t.start()
        try:
            b.run_all()
        except BaseException as exc:  # caught here: a KeyboardInterrupt must not reach pytest
            got.append(exc)
        signal.signal(signal.SIGINT, lambda signum, frame: late.append(signum))  # a Ctrl-C after run_all returned
        t.join(20)
    finally:
        signal.signal(signal.SIGINT, before)
        release.set()
        sys.stderr = real_stderr
        if fd2 is not None:
            os.dup2(fd2, 2)
            os.close(fd2)
        if broken is not None:
            try:
                broken.close()
            except OSError:
                pass
    deadline = time.monotonic() + 10  # lane 2's late row, if it comes after finish()
    while ("row", f1, "stub") not in order and time.monotonic() < deadline:
        time.sleep(0.01)
    monkeypatch.undo()
    assert late == [] and isinstance(got[0], KeyboardInterrupt if kind == "ctrl_c" else RuntimeError)
    assert probes == [f"{run_dir}: another batch is running on this run (or an output run is reading its references)"]
    assert order[-1] == ("finish", 0) and ("row", f1, "stub") in order  # no lane thread left, no row after finish()
    assert capfd.readouterr().err.count("still waiting for the lanes' current runs") == (stderr != "broken_pipe")
    done = json.loads((run_dir / "er_complete.json").read_text())
    assert done["running"] is False and done["complete"] is (kind == "ctrl_c")  # every lane had returned
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **{**KW1, "lanes": 2})
    assert len(keyed(tmp_path / "out/r/er_results.jsonl")) == 4  # no late duplicate row
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


CHILD_TEE = """
import json, signal, sys, threading, time, traceback
from pathlib import Path
signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, whatever the parent ignores
import er.batch as batch
from tests.test_er_batch import KW1, Fake, make_many, recs
tmp = Path(sys.argv[1]); events = open(tmp / "events", "a", buffering=1)  # not stderr: its reader will be gone
records = recs(2); make_many(tmp, records)
class Held(Fake):  # lane 2 is inside the last run of its last entry (stub M1) until the parent releases it
    def __call__(self, record, root, entry, lane, out, noop=()):
        if lane == 2 and noop and entry.condition == "stub":
            (tmp / "held").touch()
            deadline = time.monotonic() + 30
            while not (tmp / "release").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
        return super().__call__(record, root, entry, lane, out, noop)
class Logged(batch.Batch):
    def write(self, path, obj):
        super().write(path, obj)
        if path.name == "er_results.jsonl":
            events.write(json.dumps(["row", obj["record_id"], obj["condition"]]) + "\\n")
    def finish(self, stopped):  # what is still running when finish() starts
        events.write(json.dumps(["finish", sum(t.name.startswith("ThreadPoolExecutor") for t in threading.enumerate())])
                     + "\\n")
        super().finish(stopped)
try:
    Logged(tmp / "m", "r", records, tmp / "out", tmp, run_fn=Held(), **{**KW1, "lanes": 2}).run_all()
    events.write(json.dumps(["returned"]) + "\\n")
except BaseException as exc:
    events.write(json.dumps(["raised", type(exc).__name__, [f.name for f in traceback.extract_tb(exc.__traceback__)],
                             type(exc.__context__).__name__ if exc.__context__ else None]) + "\\n")
"""


def test_a_real_ctrl_c_pair_with_stderr_piped_into_a_killed_tee_keeps_the_run_lock(tmp_path):
    """The batch's stdout and stderr go into a tee process, as in `python -m er.batch … 2>&1 | tee
    log`. A real first Ctrl-C during the lane phase; tee is killed (a terminal's Ctrl-C reaches it too); a second real
    Ctrl-C while lane 2 is still inside its run, whose notice meets a broken pipe. Nothing may escape into the join
    (no BrokenPipeError, no KeyboardInterrupt of the handler): the run lock stays held until every lane returned, no
    row lands after finish(), er_complete.json ends running: false."""
    root, (f0, f1) = Path(__file__).resolve().parents[1], sorted(recs(2))
    tee = subprocess.Popen(["tee", str(tmp_path / "log")], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    child = subprocess.Popen([sys.executable, "-c", CHILD_TEE, str(tmp_path)], cwd=root, stdout=tee.stdin,
                             stderr=subprocess.STDOUT)
    tee.stdin.close()  # only the batch writes into tee
    events, probe = tmp_path / "events", []

    def got():
        return [json.loads(x) for x in events.read_text().splitlines() if x.strip()] if events.exists() else []
    try:
        deadline = time.monotonic() + 30
        while not ((tmp_path / "held").exists() and ["row", f0, "stub"] in got()) and time.monotonic() < deadline:
            time.sleep(0.02)  # lane 2 is held in its run; lane 1 has written its last row
        time.sleep(0.2)
        child.send_signal(signal.SIGINT)  # the first Ctrl-C: the main thread waits for the lanes
        tee.kill()
        tee.wait(10)
        time.sleep(0.5)  # the main thread is now in the join; lane 2 is still in its run
        child.send_signal(signal.SIGINT)  # the second Ctrl-C: its notice cannot be written
        time.sleep(0.5)
        with open(tmp_path / "out/r/.er_batch.lock", "a") as fh:  # an operator resumes at once
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                probe.append("free")
            except BlockingIOError:
                probe.append("held")
    finally:
        (tmp_path / "release").touch()
        child.wait(30)
        tee.kill()
    events = got()
    assert probe == ["held"]  # the lock was held while lane 2 still ran
    assert events[-1][:2] == ["raised", "KeyboardInterrupt"] and events[-1][3] is None  # the first Ctrl-C's, from wait()
    assert not {"shutdown", "join", "_wait_for_tstate_lock"} & set(events[-1][2])
    assert events[-2] == ["finish", 0] and ["row", f1, "stub"] in events[:-2]  # no row after finish(), no lane left
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["running"], done["complete"]) == (False, True)


@pytest.mark.parametrize("where", ["in", "after"])
def test_a_first_ctrl_c_in_the_final_join_of_a_finished_batch_only_waits(tmp_path, monkeypatch, capfd, where):
    """Every lane has finished (stop is not set) and the main thread joins the pool's idle workers. A first Ctrl-C
    there (in the join, or right after it returned, before lanes_running is cleared) only prints the notice: the
    batch ends as a finished one (running: false, complete: true), without the "join of the lanes failed" warning."""
    records = recs(2)
    make_many(tmp_path, records)
    got = []

    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            if where == "in":
                os.kill(os.getpid(), signal.SIGINT)  # handled at the next bytecode boundary, in the final join
                time.sleep(0.05)
            super().shutdown(wait, cancel_futures=cancel_futures)
            if where == "after":
                os.kill(os.getpid(), signal.SIGINT)  # after the join returned, before lanes_running = False
                time.sleep(0.05)
    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    before = signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, whatever pytest inherited
    try:
        try:
            run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **{**KW1, "lanes": 2})
        except BaseException as exc:  # caught here: a KeyboardInterrupt must not reach pytest
            got.append(exc)
    finally:
        signal.signal(signal.SIGINT, before)
    monkeypatch.undo()
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    err = capfd.readouterr().err
    assert got == [] and (done["running"], done["complete"], done["stopped"]) == (False, True, None)
    assert err.count("still waiting for the lanes' current runs") == 1 and "join of the lanes failed" not in err


def test_the_end_of_run_canary_can_still_retry_after_the_final_join(tmp_path):
    """The final join keeps a Ctrl-C out without setting the stop event, so the end-of-run canary that follows can
    still retry an INFRA (a set stop would end that retry at once: Stopped)."""
    records = recs(2)
    make_many(tmp_path, records, conds=("gold",))
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    infra = []

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "m.s1.x":  # the run ends on a streak without a trace: the end-of-run canary
                return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "x"}
            if entry.condition == "gold" and out.parent.name.startswith("end") and not infra:
                infra.append(1)  # the canary's first attempt hits a docker hiccup
                return {"status": "INFRA", "infra_source": "docker", "trace": None, "error": "docker: hiccup"}
            return super().__call__(record, root, entry, lane, out, noop)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=F(), ref_run="r", **KW1)
    assert infra and [c["trigger"] for c in lines(tmp_path / "out/o/er_canary.jsonl")] == ["start", "end"]
    assert json.loads((tmp_path / "out/o/er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("handler", ["ignored", "custom"])  # nohup … &; a library caller's own handler
def test_a_batch_leaves_an_ignored_or_a_callers_own_ctrl_c_handler_alone(tmp_path, handler):
    make(tmp_path)
    sent, got, seen = [], [], []

    class F(Fake):
        def __call__(self, *a, **kw):
            if not sent:  # while the lanes run
                sent.append(1)
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.2)
            return super().__call__(*a, **kw)

    def own(signum, frame):
        seen.append(signum)
    mine = signal.SIG_IGN if handler == "ignored" else own
    before = signal.signal(signal.SIGINT, mine)
    try:
        try:
            path = go(tmp_path, F())
        except BaseException as exc:  # caught here: a KeyboardInterrupt must not reach pytest
            got.append(exc)
        after = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, before)
    assert got == [] and sent and len(rows(path)) == 3 and after is mine
    assert seen == ([] if handler == "ignored" else [signal.SIGINT])  # the caller's handler got the Ctrl-C, not ours


def test_ctrl_c_is_ignored_while_finish_archives(tmp_path, monkeypatch):
    records = traceless_output_run(tmp_path)
    real, got = batch.move_lines, []

    def ctrl_c(*a, **kw):  # Ctrl-C pressed while finish() archives the suspect rows
        os.kill(os.getpid(), signal.SIGINT)
        return real(*a, **kw)
    monkeypatch.setattr(batch, "move_lines", ctrl_c)
    before = signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, whatever pytest inherited
    try:
        try:
            run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path,
                      run_fn=CanaryDrift({"m.s1.x": ["ERROR"] * 9}), ref_run="r", **KW1)
        except BaseException as exc:  # caught here: a KeyboardInterrupt must not reach pytest
            got.append(exc)
        after = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, before)
    assert isinstance(got[0], InfraError) and "breaker" in str(got[0])  # the batch's own stop, not the Ctrl-C
    done = json.loads((tmp_path / "out/o/er_complete.json").read_text())
    assert (len(done["suspect"]), done["running"]) == (8, False)  # finish() ran to its end
    assert len(lines(tmp_path / "out/o/er_results.superseded.jsonl")) == 8
    assert after is signal.default_int_handler  # the handler of the run's start is back


def test_er_complete_is_not_complete_while_a_lane_still_runs(tmp_path, monkeypatch):
    make(tmp_path)  # one record, one lane: gold, neg, stub
    written, release, pools = threading.Event(), threading.Event(), []

    class Held(batch.Batch):  # the lane has written its last row but has not returned yet
        def write(self, path, obj):
            super().write(path, obj)
            if path.name == "er_results.jsonl" and len(lines(path)) == 3:
                written.set()
                release.wait(5)

    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            pools.append(self)
            if len(pools) == 1:  # the join of the lanes fails while the lane still runs
                raise RuntimeError("join failed")
            super().shutdown(wait, cancel_futures=cancel_futures)
    monkeypatch.setattr(batch, "wait", lambda fs, return_when=None: written.wait(5))
    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    b = Held(tmp_path / "m", "r", RECS, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    try:
        with pytest.raises(RuntimeError, match="join failed"):
            b.run_all()
        done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    finally:
        release.set()
        pools[0].shutdown()
    assert done["missing"] == [] and (done["running"], done["complete"]) == (True, False)  # every row, lane alive


def test_a_second_batch_on_the_same_run_is_refused(tmp_path):
    make(tmp_path)
    (tmp_path / "out/r").mkdir(parents=True)
    with open(tmp_path / "out/r/.er_batch.lock", "w") as fh:  # another batch holds the run
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="another batch is running on this run"):
            go(tmp_path, Fake())
    assert not (tmp_path / "out/r/er_results.jsonl").exists()
    assert len(rows(go(tmp_path, Fake()))) == 3  # released: runs


# ---------- retries and health gating ----------

def seq_run(*results):
    it = iter(results)
    calls = []

    def run(record, root, entry, lane, out, noop=()):
        calls.append(lane)
        return {"trace": None, **next(it)}
    return run, calls


def test_run_retry_counts_infra_and_timeout_apart():
    infra, timeout, ok = {"status": "INFRA", "infra_source": "docker"}, {"status": "TIMEOUT"}, {"status": "PASS"}
    run, _ = seq_run(infra, infra, timeout, ok)
    r = batch.run_retry(run, REC, None, None, 1, None)
    assert (r["status"], r["attempts"]) == ("PASS", 4)  # the TIMEOUT after two INFRA still gets its retry
    run, _ = seq_run(infra, timeout, infra, timeout, ok)
    assert batch.run_retry(run, REC, None, None, 1, None)["attempts"] == 4  # 2 TIMEOUT attempts at most
    run, calls = seq_run(infra, timeout, infra, infra, ok)
    with pytest.raises(InfraError):
        batch.run_retry(run, REC, None, None, 1, None)
    assert len(calls) == 4  # 3 INFRA attempts at most


@pytest.mark.parametrize("source", ["docker", "timeout", "os", None])
def test_an_environment_failure_is_retried_then_stops(source):
    asked = []
    run, calls = seq_run(*[{"status": "INFRA", "infra_source": source, "error": "x"}] * 4)
    with pytest.raises(InfraError):
        batch.run_retry(run, REC, None, None, 2, None, lane_healthy=lambda lane: asked.append(lane) or True)
    assert len(calls) == 3 and asked == []


def test_infra_like_output_on_a_healthy_lane_twice_is_the_outputs_error(tmp_path):
    records = recs(3)
    make_many(tmp_path, records, conds=("gold", "m.s1.echo"))
    bad = sorted(records)[0]

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if record["id"] == bad and entry.condition == "m.s1.echo":  # the output prints "ConnectionException"
                self.calls.append((entry.condition, tuple(noop), label(out)))
                return {"status": "INFRA", "infra_source": "log", "trace": None, "error": "database connection failed"}
            return super().__call__(record, root, entry, lane, out, noop)
    fake, asked = F(), []
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=1, run_fn=fake,
              lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: asked.append(lane) or True)
    got = {(r["record_id"], r["condition"]): r for r in lines(tmp_path / "out/r/er_results.jsonl")}
    r = got[(bad, "m.s1.echo")]
    assert (r["er_status"], r["er"], r["attempts"]) == ("ERROR", 0, 2) and asked == [1, 1]  # retried once
    assert r["error"] == "infra-like output, environment healthy (log)" and r["er_trace"] == 0
    assert [c[0] for c in fake.calls].count("m.s1.echo") == 6  # 2 normal runs for the bad one, normal + M1 x 2
    assert len(got) == 6  # the records after it ran


def test_infra_like_output_that_goes_away_on_the_retry_is_scored(tmp_path):
    make(tmp_path, conds=("gold", "m.s1.echo"))
    r = rows(go(tmp_path, Fake({"m.s1.echo": ["INFRA:inner"]})))["m.s1.echo"]
    assert (r["er_status"], r["er"], r["attempts"], r["error"]) == ("PASS", 1, 2, None)


def test_run_retry_gates_the_second_infra_like_result_on_a_healthy_lane():
    log, docker = {"status": "INFRA", "infra_source": "log"}, {"status": "INFRA", "infra_source": "docker"}
    run, calls = seq_run(log, log)
    r = batch.run_retry(run, REC, None, None, 1, None, lane_healthy=lambda lane: True)
    assert (r["status"], r["attempts"], len(calls)) == ("ERROR", 2, 2)
    run, calls = seq_run(log, docker, {"status": "INFRA", "infra_source": "inner"})
    r = batch.run_retry(run, REC, None, None, 1, None, lane_healthy=lambda lane: True)
    assert (r["status"], r["attempts"], r["error"]) == ("ERROR", 3, "infra-like output, environment healthy (inner)")


@pytest.mark.parametrize("where", ["reference", "gold condition", "canary"])
def test_gold_runs_are_never_health_gated(tmp_path, where):
    asked = []
    make(tmp_path)
    log3 = ["INFRA:log"] * 3
    if where == "canary":
        go(tmp_path, Fake())
        output_run(tmp_path)
        run, script = dict(run="o", ref_run="r"), {"gold": log3}
    else:
        run, script = {}, {"gold": log3 if where == "reference" else ["PASS", "PASS"] + log3}
    with pytest.raises(InfraError, match="after 3 attempts"):  # trusted code: its INFRA is the environment
        go(tmp_path, Fake(script), lane_healthy=lambda lane: asked.append(lane) or True, **run)
    assert asked == []


@pytest.mark.parametrize("source", ["log", "inner"])
def test_infra_like_output_on_an_unhealthy_lane_stops(tmp_path, source):
    make(tmp_path, conds=("gold", "m.s1.echo"))

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "m.s1.echo":
                self.calls.append((entry.condition, tuple(noop), label(out)))
                return {"status": "INFRA", "infra_source": source, "trace": None, "error": "gone"}
            return super().__call__(record, root, entry, lane, out, noop)
    fake = F()
    with pytest.raises(InfraError):
        go(tmp_path, fake, lane_healthy=lambda lane: False)
    assert [c[0] for c in fake.calls].count("m.s1.echo") == 3
    assert set(rows(tmp_path / "out/r/er_results.jsonl")) == {"gold"}


# ---------- references: frozen, superseded with the gold, gated whenever gold is scored ----------

GOLD_V2_N = trace(["passed"] * 4, ["code", "members", "priority"])  # what a regenerated gold overlay sends


class V2Gold(Fake):
    """The regenerated gold overlay ('v2') sends another body than the old one."""

    def __call__(self, record, root, entry, lane, out, noop=()):
        if entry.condition == "gold" and "v2" in (root / entry.dir / A).read_text():
            self.gold_trace = GOLD_V2_N
        return super().__call__(record, root, entry, lane, out, noop)


def regen(tmp_path, tag="v2", base=BASE, run="r"):
    items = [{"record": REC, "condition": c, "slug": "f_3", "base": base,
              "files": {} if c == "neg" else {A: f"<?php // {c} {tag}\n"}} for c in ("neg", "gold", "stub")]
    write_run(tmp_path / "m", run, items)


def gold_overlay(tmp_path, run="r"):
    from er.manifest import load_manifest
    return {e.condition: e for e in load_manifest(tmp_path / "m" / run)}["gold"].overlay_sha256


def test_a_new_gold_overlay_needs_supersede_and_refreshes_the_reference(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    regen(tmp_path)
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, V2Gold())
    fake = V2Gold()
    out = rows(go(tmp_path, fake, supersede=True))
    assert [c for c in fake.calls if c[2] == "_gold_ref"] == [("gold", (), "_gold_ref"), ("gold", (1,), "_gold_ref")]
    refs = lines(tmp_path / "out/r/er_refs.jsonl")
    assert len(refs) == 1 and refs[0]["gold_overlay_sha256"] == gold_overlay(tmp_path)
    assert len(lines(tmp_path / "out/r/er_refs.superseded.jsonl")) == 1
    assert len(lines(tmp_path / "out/r/er_results.superseded.jsonl")) == 3  # neg too: scored against the old gold
    assert (out["gold"]["trace_action"], out["gold"]["ref_error"]) == (1, None)
    assert all(o["gold_overlay_sha256"] == gold_overlay(tmp_path) for o in out.values())


class OutIsV2(V2Gold):
    """Outputs send exactly what the regenerated gold sends."""

    def canned(self, record, entry, m, status):
        if entry.condition.startswith("m.") and not m:
            return GOLD_V2_N, "PASS"
        return super().canned(record, entry, m, status)


def test_an_output_run_resumed_after_its_controls_were_superseded_needs_supersede(tmp_path):
    make(tmp_path)
    go(tmp_path, V2Gold())
    output_run(tmp_path, conds=("m.s1.a", "m.s1.b"))
    go(tmp_path, OutIsV2(), run="o", ref_run="r", conditions={"m.s1.a"})  # scored against the old reference
    regen(tmp_path)
    go(tmp_path, V2Gold(), supersede=True)  # the controls run recomputed its reference
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, OutIsV2(), run="o", ref_run="r")
    out = rows(go(tmp_path, OutIsV2(), run="o", ref_run="r", supersede=True))
    assert (out["m.s1.a"]["trace_action"], out["m.s1.b"]["trace_action"]) == (1, 1)  # one reference in the run
    assert out["m.s1.a"]["gold_overlay_sha256"] == out["m.s1.b"]["gold_overlay_sha256"] == gold_overlay(tmp_path)
    assert [o["condition"] for o in lines(tmp_path / "out/o/er_results.superseded.jsonl")] == ["m.s1.a"]


@pytest.mark.parametrize("kind, key", [("normal", "gold_trace_sha256"), ("m1", "gold_m1_trace_sha256")])
def test_an_output_row_scored_against_another_frozen_gold_trace_is_stale(tmp_path, kind, key):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path)
    go(tmp_path, Fake(), run="o", ref_run="r")
    refs = tmp_path / "out/r/er_refs.jsonl"
    ref = lines(refs)[0]
    p = tmp_path / "out/r" / ref["ref_dir"] / kind / "trace.json"
    p.write_text(json.dumps(json.loads(p.read_text()), indent=1))  # a recomputed reference: same trace, other bytes
    refs.write_text(json.dumps({**ref, key: hashlib.sha256(p.read_bytes()).hexdigest()}) + "\n")
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, Fake(), run="o", ref_run="r")
    assert rows(go(tmp_path, Fake(), run="o", ref_run="r", supersede=True))["deepseek.s42.full_aware"][key] == \
        lines(refs)[0][key]


def test_a_controls_row_scored_against_another_gold_overlay_needs_supersede(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    res = tmp_path / "out/r/er_results.jsonl"
    res.write_text("".join(json.dumps({**o, "gold_overlay_sha256": "0" * 64} if o["condition"] == "neg" else o) + "\n"
                           for o in lines(res)))
    with pytest.raises(SystemExit, match="supersede"):  # its own overlay is current; its gold is not
        go(tmp_path, Fake())
    assert rows(go(tmp_path, Fake(), supersede=True))["neg"]["gold_overlay_sha256"] == gold_overlay(tmp_path)


def test_a_supersede_cut_between_the_reference_and_the_rows_is_finished_on_resume(tmp_path, monkeypatch):
    make(tmp_path)
    go(tmp_path, V2Gold())
    regen(tmp_path)
    real, calls = batch.move_lines, []

    def crash_second(path, gone, note=None, since=None):
        calls.append(path.name)
        if len(calls) == 2:
            raise KeyboardInterrupt  # killed right after the reference line was archived
        return real(path, gone, note, since)
    monkeypatch.setattr(batch, "move_lines", crash_second)
    with pytest.raises(KeyboardInterrupt):
        go(tmp_path, V2Gold(), supersede=True)
    monkeypatch.setattr(batch, "move_lines", real)
    assert calls == ["er_refs.jsonl", "er_results.jsonl"]
    with pytest.raises(SystemExit, match="supersede"):  # neg's row was scored against the old gold
        go(tmp_path, V2Gold())
    out = rows(go(tmp_path, V2Gold(), supersede=True))
    ref = lines(tmp_path / "out/r/er_refs.jsonl")[-1]
    assert all(o["gold_overlay_sha256"] == ref["gold_overlay_sha256"] == gold_overlay(tmp_path) for o in out.values())


def test_a_fixed_gold_after_supersede_gets_a_working_reference(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake({"gold": ["ERROR"] * 4}))  # a data bug: gold broken, frozen with ref_error
    regen(tmp_path, tag="fixed")
    out = rows(go(tmp_path, Fake(), supersede=True))
    assert (out["gold"]["er_status"], out["gold"]["ref_error"], out["stub"]["trace_action"]) == ("PASS", None, 0)


def test_the_gate_runs_on_resume_when_the_gold_row_was_lost(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    res = tmp_path / "out/r/er_results.jsonl"
    res.write_text("".join(x + "\n" for x in res.read_text().splitlines() if json.loads(x)["condition"] != "gold"))
    out = rows(go(tmp_path, Fake(gold_trace=trace(["passed"] * 4, ["code", "members", "extra"]))))
    assert (out["gold"]["ref_error"], out["gold"]["trace_action"]) == ("nondeterministic", None)
    last = lines(tmp_path / "out/r/er_refs.jsonl")[-1]
    assert (last["ref_error"], last["deterministic"]) == ("nondeterministic", False)


def drop_rows(tmp_path, cond, run="r"):
    res = tmp_path / "out" / run / "er_results.jsonl"
    res.write_text("".join(x + "\n" for x in res.read_text().splitlines() if json.loads(x)["condition"] != cond))


def test_rows_scored_before_a_gate_update_are_recomputed(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    drop_rows(tmp_path, "gold")
    go(tmp_path, Fake(gold_trace=trace(["passed"] * 4, ["code", "members", "extra"])))  # gate: nondeterministic
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["complete"], done["missing"]) == (False, [[REC["id"], "neg"], [REC["id"], "stub"]])
    out = rows(go(tmp_path, Fake()))  # neg and stub rescored against the updated reference line
    assert all(o["ref_error"] == "nondeterministic" and o["trace_action"] is None for o in out.values())


def test_rows_without_a_reference_line_are_recomputed_without_supersede(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    (tmp_path / "out/r/er_refs.jsonl").write_text("")  # e.g. a crash between archiving the reference and its rows
    fake = Fake()
    out = rows(go(tmp_path, fake))
    assert len(out) == 3 and ("gold", (), "_gold_ref") in fake.calls
    assert len(lines(tmp_path / "out/r/er_results.superseded.jsonl")) == 3
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


def test_a_row_disagreeing_with_its_reference_line_is_recomputed_at_once(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    res = tmp_path / "out/r/er_results.jsonl"
    res.write_text("".join(json.dumps({**o, "gold_status": "FAIL"} if o["condition"] == "stub" else o) + "\n"
                           for o in lines(res)))
    fake = Fake()
    out = rows(go(tmp_path, fake))
    assert out["stub"]["gold_status"] == "PASS" and [c[:2] for c in fake.calls] == [("stub", ()), ("stub", (1,))]
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("first, again", [({}, {"gold": ["ERROR"]}),  # frozen PASS, now ERROR
                                          ({"gold": ["ERROR"] * 2}, {})])  # frozen broken, now PASS
def test_a_loaded_reference_contradicted_by_the_gold_row_is_recomputed(tmp_path, first, again):
    make(tmp_path)
    go(tmp_path, Fake(first))
    drop_rows(tmp_path, "gold")
    go(tmp_path, Fake(again))  # no "flaky environment" stop: the record is suspect
    assert lines(tmp_path / "out/r/er_refs.jsonl") == [] and lines(tmp_path / "out/r/er_results.jsonl") == []
    assert all(o["suspect"] is True for o in lines(tmp_path / "out/r/er_refs.superseded.jsonl"))
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is False
    fake = Fake()
    out = rows(go(tmp_path, fake))
    assert ("gold", (), "_gold_ref") in fake.calls and out["gold"]["ref_error"] is None and len(out) == 3


def test_a_gold_row_contradicting_the_reference_just_computed_stops_unrecorded(tmp_path):
    make(tmp_path)
    fake = Fake({"gold": ["PASS", "", "FAIL"]})  # reference PASS (normal, then M1), the gold row right after FAIL
    with pytest.raises(InfraError, match="flaky environment"):
        go(tmp_path, fake)
    for name in ("er_refs.jsonl", "er_results.jsonl"):
        path = tmp_path / "out/r" / name
        assert not path.exists() or lines(path) == []  # nothing frozen, nothing scored: a resume recomputes both


def test_a_crash_in_finish_between_the_reference_and_row_archives_heals_on_resume(tmp_path, monkeypatch):
    records = recs(4)
    make_many(tmp_path, records)
    real, calls = batch.move_lines, []

    def crash_second(path, gone, note=None, since=None):
        calls.append(path.name)
        if len(calls) == 2:
            raise KeyboardInterrupt  # killed between the two archives of finish()
        return real(path, gone, note, since)
    monkeypatch.setattr(batch, "move_lines", crash_second)
    with pytest.raises(KeyboardInterrupt):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Broken(),
                  **{**KW1, "lane_healthy": lambda lane: False})
    monkeypatch.setattr(batch, "move_lines", real)
    assert calls == ["er_refs.jsonl", "er_results.jsonl"]  # the references first
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)  # environment fixed
    refs = lines(tmp_path / "out/r/er_refs.jsonl")
    assert len(refs) == 4 and all(r["ref_error"] is None for r in refs)
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("m1", [trace(["passed", "noop", "passed", "failed"], ["code", "extra"]),  # other M1 events
                                trace(["passed", "noop", "failed", "failed"], ["code"])])  # other M1 statuses
def test_the_gate_also_checks_the_gold_m1_run(tmp_path, m1):
    make(tmp_path)

    class M1Drift(Fake):
        n = 0

        def __call__(self, record, root, entry, lane, out, noop=()):
            r = super().__call__(record, root, entry, lane, out, noop)
            if entry.condition == "gold" and noop:
                self.n += 1
                if self.n == 2:  # the gold condition's own M1 run, not the reference's
                    r["trace"] = m1
            return r
    out = rows(go(tmp_path, M1Drift()))
    assert out["gold"]["ref_error"] == "nondeterministic" and out["stub"]["trace_action"] is None


def test_a_skipped_gold_nulls_the_record_without_stopping(tmp_path):
    make(tmp_path, conds=("neg", "stub"), extra=[("gold", "insert: failed")])
    out = rows(go(tmp_path, Fake()))
    assert set(out) == {"neg", "gold", "stub"} and out["gold"]["er_status"] == "SKIP" and out["stub"]["er"] == 1
    assert all(o["ref_error"] == "gold is SKIP (insert: failed)" and o["er_trace"] is None
               and o["trace_action"] is None and o["act_evaluable"] is None for o in out.values())
    ref = lines(tmp_path / "out/r/er_refs.jsonl")[0]
    assert (ref["ref_error"], ref["deterministic"]) == ("gold is SKIP (insert: failed)", None)  # gated like any gold


@pytest.mark.parametrize("script, error", [({"gold": ["PASS", "ERROR"]}, "gold M1 is ERROR"),
                                           ({"gold": ["FAIL", "FAIL"]}, "gold is FAIL")])
def test_a_broken_gold_reference_nulls_every_row_with_ref_error(tmp_path, script, error):
    make(tmp_path, extra=[("m.s1.none", "extract: none")])
    out = lines(go(tmp_path, Fake(script)))
    assert len(out) == 4 and all(o["ref_error"] == error and o["er_trace"] is None and o["trace_action"] is None
                                 and o["act_evaluable"] is None for o in out)


def test_no_trace_rows_keep_the_record_evaluability(tmp_path):
    make(tmp_path, conds=("gold", "m.s1.err", "m.s1.slow"), extra=[("m.s1.none", "extract: none")])
    out = rows(go(tmp_path, Fake({"m.s1.err": ["ERROR"], "m.s1.slow": ["TIMEOUT"] * 2})))
    for c in ("m.s1.err", "m.s1.slow", "m.s1.none"):
        assert (out[c]["act_evaluable"], out[c]["n_then_evaluable"]) == (True, 1), c
        assert (out[c]["trace_action"], out[c]["trace_then"], out[c]["er_trace"]) == (0, 0, 0), c


def test_no_trace_row_is_null_only_on_the_part_the_record_cannot_evaluate(tmp_path):
    class ThenNotEvaluable(Fake):  # gold's Then still passes under M1: no evaluable Then
        def __call__(self, record, root, entry, lane, out, noop=()):
            r = super().__call__(record, root, entry, lane, out, noop)
            if entry.condition == "gold" and noop:
                r["trace"] = trace(["passed", "noop", "passed", "passed"], ["code"])
            return r
    make(tmp_path, conds=("gold", "m.s1.err"), extra=[("m.s1.none", "extract: none")])
    out = rows(go(tmp_path, ThenNotEvaluable({"m.s1.err": ["ERROR"]})))
    for c in ("m.s1.err", "m.s1.none"):
        assert (out[c]["act_evaluable"], out[c]["n_then_evaluable"]) == (True, 0), c
        assert (out[c]["trace_action"], out[c]["trace_then"], out[c]["er_trace"]) == (0, None, 0), c


# ---------- output runs (--ref-run): consistency with the frozen references, canaries ----------

def output_run(tmp_path, run="o", base=BASE, conds=("deepseek.s42.full_aware",)):
    items = [{"record": REC, "condition": c, "slug": "f_3", "base": base, "files": {A: "<?php // out\n"}} for c in conds]
    write_run(tmp_path / "m", run, items)


def test_ref_run_refuses_an_output_on_another_base(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path, base={A: "<?php // a different ER tree\n"})
    with pytest.raises(SystemExit, match="base"):
        go(tmp_path, Fake(), run="o", ref_run="r")


def test_ref_run_refuses_a_controls_manifest_changed_after_the_freeze(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    regen(tmp_path)  # controls manifest rewritten after its references were frozen
    output_run(tmp_path)
    with pytest.raises(SystemExit, match="manifest"):
        go(tmp_path, Fake(), run="o", ref_run="r")


@pytest.mark.parametrize("key", ["image_id", "phase2_sha256sums_sha256", "schema_fingerprint", "harness_code"])
def test_ref_run_refuses_a_controls_run_of_another_environment(tmp_path, key):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path)
    with pytest.raises(SystemExit, match=f"{key} differs"):
        run_batch(tmp_path / "m", "o", RECS, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), ref_run="r",
                  lane_check=lambda lane: None, provenance={**PROV, key: "other"}, lane_healthy=lambda lane: True)


def test_an_output_run_never_writes_its_controls_run(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    refs = tmp_path / "out/r/er_refs.jsonl"
    refs.write_text(refs.read_text() + '{"record_id": "cut')  # a partial line the controls run must repair itself
    before = {p: p.read_bytes() for p in (tmp_path / "out/r").rglob("*") if p.is_file()}
    output_run(tmp_path)
    go(tmp_path, Fake(), run="o", ref_run="r")
    assert {p: p.read_bytes() for p in (tmp_path / "out/r").rglob("*") if p.is_file()} == before


def test_ref_run_refuses_a_reference_of_another_gold_overlay(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    refs = tmp_path / "out/r/er_refs.jsonl"
    refs.write_text(json.dumps({**lines(refs)[0], "gold_overlay_sha256": "0" * 64}) + "\n")
    output_run(tmp_path)
    with pytest.raises(SystemExit, match="gold overlay"):
        go(tmp_path, Fake(), run="o", ref_run="r")


def test_resume_must_keep_the_ref_run_and_the_manifest_root(tmp_path):
    import shutil
    make(tmp_path)
    go(tmp_path, Fake())
    make(tmp_path, run="r2")
    go(tmp_path, Fake(), run="r2")
    output_run(tmp_path, conds=("m.s1.a", "m.s1.b"))
    go(tmp_path, Fake(), run="o", ref_run="r", conditions={"m.s1.a"})
    with pytest.raises(SystemExit, match="ref_run"):
        go(tmp_path, Fake(), run="o", ref_run="r2")
    shutil.copytree(tmp_path / "m", tmp_path / "m2")
    with pytest.raises(SystemExit, match="manifest_root"):
        run_batch(tmp_path / "m2", "o", RECS, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), ref_run="r",
                  lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)


def trace_path(statuses, path, tokens=()):
    t = trace(statuses, ["code"])
    t["events"][0].update(path=path, token_values=list(tokens))
    return t


class PerRecord(Fake):
    """gold of f0 is broken; f1 sends a numeric id and has an evaluable Then; f2 sends a token, Then not evaluable."""

    def __call__(self, record, root, entry, lane, out, noop=()):
        if entry.condition == "gold" and record["feature_path"] == "f0.feature":
            self.calls.append((entry.condition, tuple(noop), label(out)))
            return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "x"}
        return super().__call__(record, root, entry, lane, out, noop)

    def canned(self, record, entry, m, status):
        f = record["feature_path"]
        if entry.condition == "gold" and f == "f1.feature" and not m:
            return trace_path(["passed"] * 4, "/api/v2/admin/zones/123"), "PASS"
        if entry.condition == "gold" and f == "f2.feature":
            if m:
                return trace_path(["passed", "noop", "passed", "passed"], "/z"), "PASS"
            return trace_path(["passed"] * 4, "/api/v2/shop/orders/AAAAAAAAAA/items", ["AAAAAAAAAA"]), "PASS"
        return super().canned(record, entry, m, status)


def test_the_canaries_are_picked_by_rule_and_run_on_every_lane(tmp_path):
    records = recs(3)
    make_many(tmp_path, records, conds=("gold",))
    kw = dict(lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=2, run_fn=PerRecord(), **kw)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    fake = PerRecord()
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, lanes=2, run_fn=fake, ref_run="r", **kw)
    got = sorted((c["record_id"].split("|")[1], c["lane"], tuple(c["kinds"]), c["ok"])
                 for c in lines(tmp_path / "out/o/er_canary.jsonl"))
    assert got == [("f1.feature", 1, ("n", "then"), True), ("f1.feature", 2, ("n", "then"), True),
                   ("f2.feature", 1, ("tok",), True), ("f2.feature", 2, ("tok",), True)]
    assert [c[:2] for c in fake.calls if c[2] == "_canary"].count(("gold", (1,))) == 2  # M1 of the Then canary


def test_the_canaries_come_from_the_records_of_this_batch(tmp_path):
    records = recs(3)
    make_many(tmp_path, records, conds=("gold",))
    kw = dict(lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=1, run_fn=PerRecord(), **kw)
    mine = {rid: r for rid, r in records.items() if "f2." not in rid}  # the token record f2 is not in this batch
    make_many(tmp_path, mine, conds=("m.s1.x",), run="o")
    run_batch(tmp_path / "m", "o", mine, tmp_path / "out", tmp_path, lanes=1, run_fn=PerRecord(), ref_run="r", **kw)
    got = {(c["record_id"].split("|")[1], tuple(c["kinds"])) for c in lines(tmp_path / "out/o/er_canary.jsonl")}
    assert got == {("f1.feature", ("n", "then"))}


def test_no_canary_candidate_stops_the_output_run(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake({"gold": ["ERROR"] * 4}))  # every reference broken
    output_run(tmp_path)
    with pytest.raises(SystemExit, match="canary"):
        go(tmp_path, Fake(), run="o", ref_run="r")


def test_a_limited_output_run_needs_a_complete_controls_run(tmp_path):
    records = recs(2)
    make_many(tmp_path, records)
    kw = dict(lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), limit=1, **kw)
    make_many(tmp_path, records, conds=("deepseek.s42.full_aware",), run="o")
    with pytest.raises(SystemExit, match="complete"):  # a smoke controls run cannot serve as a reference
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), ref_run="r",
                  limit=1, **kw)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), **kw)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, lanes=1, run_fn=Fake(), ref_run="r", limit=1,
              **kw)
    assert len(lines(tmp_path / "out/o/er_results.jsonl")) == 1


def test_an_output_run_needs_a_complete_controls_run(tmp_path):
    make(tmp_path)
    with pytest.raises(InfraError):
        go(tmp_path, Fake({"stub": ["INFRA"] * 3}))
    output_run(tmp_path)
    with pytest.raises(SystemExit, match="complete"):
        go(tmp_path, Fake(), run="o", ref_run="r")


def test_an_output_run_holds_a_shared_lock_on_its_controls_run(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path)
    lock = tmp_path / "out/r/.er_batch.lock"
    with open(lock, "w") as fh:  # the controls run is running (or superseding)
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="controls run"):
            go(tmp_path, Fake(), run="o", ref_run="r")
    held = []

    class Probe(Fake):  # while the output run works, the controls run cannot start
        def __call__(self, *a, **kw):
            with open(lock, "a") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held.append(False)
                except BlockingIOError:
                    held.append(True)
            return super().__call__(*a, **kw)
    go(tmp_path, Probe(), run="o", ref_run="r")
    assert held and all(held)
    with open(lock, "w") as fh:  # an output run holds it: the controls run is refused
        fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="another batch is running on this run"):
            go(tmp_path, Fake())
        make(tmp_path, conds=("m.s1.c",), run="o2")  # but another output run may share it
        go(tmp_path, Fake(), run="o2", ref_run="r")


@pytest.mark.parametrize("m1", [trace(["passed", "noop", "failed", "failed"], ["code"]),  # other M1 statuses
                                trace(["passed", "noop", "passed", "failed"], ["code", "extra"])])  # other M1 events
def test_a_canary_whose_m1_run_drifted_stops(tmp_path, m1):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path)

    class M1Drift(Fake):
        def canned(self, record, entry, m, status):
            if entry.condition == "gold" and m:
                return m1, "FAIL"
            return super().canned(record, entry, m, status)
    with pytest.raises(InfraError, match="canary"):
        go(tmp_path, M1Drift(), run="o", ref_run="r")
    assert lines(tmp_path / "out/o/er_canary.jsonl")[0]["ok"] is False


# ---------- circuit breakers ----------

KW1 = dict(lanes=1, lane_check=lambda lane: None, provenance=PROV, lane_healthy=lambda lane: True)


def traceless_output_run(tmp_path, n=9):
    records = recs(n)
    make_many(tmp_path, records, conds=("gold",))
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    return records


class CanaryDrift(Fake):
    """The environment breaks after the start canary (normal + M1 of the Then canary): later canaries drift."""

    def canned(self, record, entry, m, status):
        if entry.condition == "gold" and not m and sum(c[0] == "gold" for c in self.calls) > 2:
            return STUB_N, "PASS"
        return super().canned(record, entry, m, status)


def test_eight_traceless_rows_in_a_row_rerun_the_canary(tmp_path):
    records = traceless_output_run(tmp_path)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake({"m.s1.x": ["ERROR"] * 9}),
              ref_run="r", **KW1)
    # the 9th row starts a new streak, which the run ends on: checked at the end
    assert [c["trigger"] for c in lines(tmp_path / "out/o/er_canary.jsonl")] == ["start", "breaker", "end"]
    assert len(lines(tmp_path / "out/o/er_results.jsonl")) == 9  # the canary passed: the outputs' own errors


def test_each_canary_keeps_its_own_run_directory(tmp_path):
    records = traceless_output_run(tmp_path)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake({"m.s1.x": ["ERROR"] * 9}),
              ref_run="r", **KW1)
    base = tmp_path / "out/o/er_runs/f0_3/_canary"  # f0: the first record with an evaluable Then
    assert sorted(p.name for p in base.iterdir()) == ["breaker_1", "end_2", "start_0"]  # seq = canary lines before it
    for d in ("start_0", "breaker_1", "end_2"):
        assert sorted(p.name for p in (base / d).iterdir()) == ["lane1_m1", "lane1_normal"]
        assert (base / d / "lane1_normal/trace.json").is_file()


def test_a_cut_canary_line_is_repaired_before_the_next_canary(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    output_run(tmp_path, conds=("m.s1.a", "m.s1.b"))
    go(tmp_path, Fake(), run="o", ref_run="r", conditions={"m.s1.a"})
    path = tmp_path / "out/o/er_canary.jsonl"
    path.write_text(path.read_text() + '{"trigger": "brea')  # killed while writing a canary line
    go(tmp_path, Fake(), run="o", ref_run="r")
    assert [c["trigger"] for c in lines(path)] == ["start", "start"]  # nothing glued to the cut line
    assert (tmp_path / "out/o/er_runs/f_3/_canary/start_1/lane1_normal").is_dir()


@pytest.mark.parametrize("script", [["ERROR"] * 7 + ["PASS"] + ["ERROR"] * 3,  # a trace breaks the streak
                                    ["FAIL"] * 22])  # FAIL (normal and M1) has a trace: the output's own failure
def test_the_breaker_counts_only_traceless_rows_in_a_row(tmp_path, script):
    records = traceless_output_run(tmp_path, n=11)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake({"m.s1.x": script}),
              ref_run="r", **KW1)
    assert [c["trigger"] for c in lines(tmp_path / "out/o/er_canary.jsonl")] == ["start"]


def test_a_failed_breaker_canary_stops_and_the_suspect_rows_are_rerun(tmp_path):
    records = traceless_output_run(tmp_path)

    class Drift(Fake):  # the environment breaks after the start canary
        def canned(self, record, entry, m, status):
            if entry.condition == "gold" and not m and sum(c[0] == "gold" for c in self.calls) > 2:
                return STUB_N, "PASS"
            return super().canned(record, entry, m, status)
    with pytest.raises(InfraError, match="breaker"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Drift({"m.s1.x": ["ERROR"] * 9}),
                  ref_run="r", **KW1)
    assert lines(tmp_path / "out/o/er_results.jsonl") == []
    suspect = lines(tmp_path / "out/o/er_results.superseded.jsonl")
    assert len(suspect) == 8 and all(r["suspect"] is True for r in suspect)
    done = json.loads((tmp_path / "out/o/er_complete.json").read_text())
    assert (len(done["suspect"]), len(done["missing"]), done["complete"]) == (8, 9, False)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    out = lines(tmp_path / "out/o/er_results.jsonl")
    assert len(out) == 9 and all(r["er_status"] == "PASS" for r in out)


def test_a_kill_inside_the_suspect_archive_is_healed_on_resume(tmp_path, monkeypatch):
    records = traceless_output_run(tmp_path)

    class Drift(Fake):  # the environment breaks after the start canary
        def canned(self, record, entry, m, status):
            if entry.condition == "gold" and not m and sum(c[0] == "gold" for c in self.calls) > 2:
                return STUB_N, "PASS"
            return super().canned(record, entry, m, status)
    real = batch.atomic_write

    def killed(path, text):  # SIGKILL after the suspect archive was synced, before er_results.jsonl is rewritten
        if path.name == "er_results.jsonl":
            raise KeyboardInterrupt
        real(path, text)
    monkeypatch.setattr(batch, "atomic_write", killed)
    with pytest.raises(KeyboardInterrupt):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Drift({"m.s1.x": ["ERROR"] * 9}),
                  ref_run="r", **KW1)
    monkeypatch.undo()
    res, archive = tmp_path / "out/o/er_results.jsonl", tmp_path / "out/o/er_results.superseded.jsonl"
    assert len(lines(res)) == len(lines(archive)) == 8  # the suspect rows are in both files
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    out = lines(res)
    assert len(out) == 9 and all(r["er_status"] == "PASS" for r in out)  # no wrong zeros kept
    assert len(lines(archive)) == 8  # not archived twice
    assert json.loads((tmp_path / "out/o/er_complete.json").read_text())["complete"] is True


def test_a_recomputed_row_is_not_taken_for_an_archived_one(tmp_path):
    records = traceless_output_run(tmp_path)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    res, archive = tmp_path / "out/o/er_results.jsonl", tmp_path / "out/o/er_results.superseded.jsonl"
    first = lines(res)[0]
    archive.write_text(json.dumps({**first, "seconds": 2.0, "suspect": True}) + "\n")  # an earlier run of that row
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    assert lines(res)[0] == first and len(lines(res)) == 9


@pytest.mark.parametrize("kind, raised, lane1_rows", [
    ("breaker", "breaker", 8),  # lane 1's breaker canary drifts after its 8th row
    ("infra", "after 3 attempts", 4),  # lane 1's 5th output is INFRA on all 3 attempts: unrecorded
    ("ctrl_c", KeyboardInterrupt, 5)])  # Ctrl-C while lane 1 runs its 5th output: recorded, then the lanes stop
def test_any_stop_also_archives_the_traceless_streak_of_the_other_lane(tmp_path, monkeypatch, kind, raised,
                                                                        lane1_rows):
    records = recs(16)  # 8 records per lane
    make_many(tmp_path, records, conds=("gold",))
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    mid, lane1_ready, lane1, lane2 = threading.Event(), threading.Event(), [], []
    KW2 = {**KW1, "lanes": 2}

    class Drift(Fake):  # the environment breaks after the start canary: every output ERROR
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "gold" and out.parent.name.startswith("breaker"):  # lane 1's breaker canary
                mid.wait(5)  # lane 2 is 3 rows into its own streak
                return self.emit(out, STUB_N, "PASS")
            if entry.condition == "m.s1.x" and lane == 1:
                lane1.append(record["id"])
                if kind == "infra" and len(lane1) >= 5:
                    mid.wait(5)
                    return {"status": "INFRA", "infra_source": "docker", "trace": None, "error": "docker: gone"}
                if kind == "ctrl_c" and len(lane1) == 5:
                    lane1_ready.set()
                    b.stop.wait(5)
            if entry.condition == "m.s1.x" and lane == 2:
                lane2.append(record["id"])
                if len(lane2) == 4:
                    mid.set()
                    b.stop.wait(5)  # until lane 1's exception (or Ctrl-C) sets the batch's stop event
            return super().__call__(record, root, entry, lane, out, noop)

    def interrupted(fs, return_when=None):  # Ctrl-C arrives while the main thread waits for the lanes
        mid.wait(5)
        lane1_ready.wait(5)
        raise KeyboardInterrupt
    if kind == "ctrl_c":
        monkeypatch.setattr(batch, "wait", interrupted)
    b = batch.Batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, ref_run="r",
                    run_fn=Drift({"m.s1.x": ["ERROR"] * 16}), **KW2)
    with pytest.raises(raised) if kind == "ctrl_c" else pytest.raises(InfraError, match=raised):
        b.run_all()
    monkeypatch.undo()
    assert lines(tmp_path / "out/o/er_results.jsonl") == []  # lane 2's streak: the same broken environment
    suspect = lines(tmp_path / "out/o/er_results.superseded.jsonl")
    assert len(suspect) == lane1_rows + 4 and all(r["suspect"] is True for r in suspect)
    assert set(lane2) <= {r["record_id"] for r in suspect}
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW2)
    out = lines(tmp_path / "out/o/er_results.jsonl")
    assert len(out) == 16 and all(r["er_status"] == "PASS" for r in out)
    assert json.loads((tmp_path / "out/o/er_complete.json").read_text())["complete"] is True


class Broken(Fake):  # e.g. an image rebuilt with a harness bug: every run is ERROR
    def __call__(self, record, root, entry, lane, out, noop=()):
        self.calls.append((entry.condition, tuple(noop), label(out)))
        return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "behat exit 255"}


def test_three_broken_gold_references_in_a_row_on_an_unhealthy_lane_stop_a_controls_run(tmp_path):
    records = recs(4)
    make_many(tmp_path, records)
    asked = []
    with pytest.raises(InfraError, match="3 records in a row"):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Broken(),
                  **{**KW1, "lane_healthy": lambda lane: asked.append(lane) or False})
    assert asked == [1]
    assert lines(tmp_path / "out/r/er_refs.jsonl") == [] and lines(tmp_path / "out/r/er_results.jsonl") == []
    assert len(lines(tmp_path / "out/r/er_refs.superseded.jsonl")) == 2  # frozen as broken: archived, not kept
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    refs = lines(tmp_path / "out/r/er_refs.jsonl")
    assert len(refs) == 4 and all(r["ref_error"] is None for r in refs)


def test_golds_really_broken_on_a_healthy_lane_are_data_and_the_run_goes_on(tmp_path):
    records = recs(5)
    make_many(tmp_path, records)
    bad = set(sorted(records)[:4])

    class F(Fake):  # a data bug: the first three golds fail their Then; the lane itself is fine
        def canned(self, record, entry, m, status):
            if record["id"] in bad and entry.condition == "gold":
                return trace(["passed"] * 3 + ["failed"], ["code"]), "FAIL"
            return super().canned(record, entry, m, status)
    asked = []
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=F(),
              **{**KW1, "lane_healthy": lambda lane: asked.append(lane) or True})
    assert asked == [1]  # asked once, at the third broken reference in a row; the fourth starts a new count
    out = lines(tmp_path / "out/r/er_results.jsonl")
    assert len(out) == 10 and all((o["ref_error"] == "gold is FAIL") == (o["record_id"] in bad) for o in out)
    assert all(o["trace_action"] is None for o in out if o["record_id"] in bad)
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


def test_a_working_reference_resets_the_gold_breaker(tmp_path):
    records = recs(6)
    make_many(tmp_path, records)
    bad = set(sorted(records)[:2] + sorted(records)[3:5])  # broken, broken, ok, broken, broken, ok

    class F(Broken):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if record["id"] in bad:
                return super().__call__(record, root, entry, lane, out, noop)
            return Fake.__call__(self, record, root, entry, lane, out, noop)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=F(),
              **{**KW1, "lane_healthy": lambda lane: False})
    assert len(lines(tmp_path / "out/r/er_results.jsonl")) == 12


@pytest.mark.parametrize("drift", [False, True])
def test_a_trailing_traceless_streak_is_checked_at_the_end_of_an_output_run(tmp_path, drift):
    records = recs(4)  # lane 1: f0, f2 (PASS); lane 2: f1, f3 (ERROR: its last rows, fewer than BREAKER_K)
    make_many(tmp_path, records, conds=("gold",))
    KW2 = {**KW1, "lanes": 2}
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW2)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    lane2 = {rid for rid in sorted(records)[1::2]}

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "m.s1.x" and lane == 2:
                self.calls.append((entry.condition, tuple(noop), label(out)))
                return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "x"}
            if drift and entry.condition == "gold" and out.parent.name.startswith("end"):
                return self.emit(out, STUB_N, "PASS")  # the environment broke under lane 2's last rows
            return super().__call__(record, root, entry, lane, out, noop)
    canary, res = tmp_path / "out/o/er_canary.jsonl", tmp_path / "out/o/er_results.jsonl"
    done = tmp_path / "out/o/er_complete.json"
    if not drift:
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=F(), ref_run="r", **KW2)
        assert [(c["trigger"], c["lane"]) for c in lines(canary)] == [("start", 1), ("start", 2), ("end", 2)]
        assert sorted(r["er_status"] for r in lines(res)) == ["ERROR", "ERROR", "PASS", "PASS"]  # outputs' own errors
        assert json.loads(done.read_text())["complete"] is True
        return
    with pytest.raises(InfraError, match=r"canary .*\(end\)"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=F(), ref_run="r", **KW2)
    assert [r["er_status"] for r in lines(res)] == ["PASS", "PASS"]  # lane 1's rows stand
    suspect = lines(tmp_path / "out/o/er_results.superseded.jsonl")
    assert {r["record_id"] for r in suspect} == lane2 and all(r["suspect"] is True for r in suspect)
    assert json.loads(done.read_text())["complete"] is False
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW2)
    assert [r["er_status"] for r in lines(res)] == ["PASS"] * 4 and json.loads(done.read_text())["complete"] is True


@pytest.mark.parametrize("healthy", [True, False])
def test_a_trailing_broken_reference_streak_is_checked_at_the_end_of_a_controls_run(tmp_path, healthy):
    records = recs(3)
    make_many(tmp_path, records)  # gold, stub
    first, bad = sorted(records)[0], set(sorted(records)[1:])  # the run ends on 2 broken references (< GOLD_BREAKER)

    class F(Broken):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if record["id"] in bad and entry.condition == "gold":
                return super().__call__(record, root, entry, lane, out, noop)
            return Fake.__call__(self, record, root, entry, lane, out, noop)
    asked = []
    kw = {**KW1, "lane_healthy": lambda lane: asked.append(lane) or healthy}
    res, refs = tmp_path / "out/r/er_results.jsonl", tmp_path / "out/r/er_refs.jsonl"
    done = tmp_path / "out/r/er_complete.json"
    if healthy:  # really broken golds (bad data): the rows stand
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=F(), **kw)
        assert asked == [1]  # asked once, at the end of the run
        out = lines(res)
        assert len(out) == 6 and all((o["ref_error"] == "gold is ERROR") == (o["record_id"] in bad) for o in out)
        assert json.loads(done.read_text())["complete"] is True
        return
    with pytest.raises(InfraError, match="2 records in a row"):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=F(), **kw)
    assert asked == [1]
    assert [o["record_id"] for o in lines(refs)] == [first] and {o["record_id"] for o in lines(res)} == {first}
    suspect = lines(tmp_path / "out/r/er_refs.superseded.jsonl")
    assert {o["record_id"] for o in suspect} == bad and all(o["suspect"] is True for o in suspect)
    assert json.loads(done.read_text())["complete"] is False
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)  # environment fixed
    assert len(lines(refs)) == 3 and all(o["ref_error"] is None for o in lines(refs)) and len(lines(res)) == 6


def test_a_tripped_gold_breaker_also_archives_the_broken_references_of_the_other_lane(tmp_path):
    records = recs(6)  # lane 1: f0, f2, f4; lane 2: f1, f3, f5
    make_many(tmp_path, records)
    mid, tripped, lane2 = threading.Event(), threading.Event(), []

    class B(Broken):  # every run ERROR; lane 2 is one broken reference into its own streak when lane 1 trips
        def __call__(self, record, root, entry, lane, out, noop=()):
            if lane == 2 and label(out) == "_gold_ref" and not noop:
                lane2.append(record["id"])
                if len(lane2) == 2:
                    mid.set()
                    tripped.wait(5)
                    b.stop.wait(5)  # until lane 1's InfraError sets the batch's stop event
            return super().__call__(record, root, entry, lane, out, noop)

    def unhealthy(lane):  # asked by lane 1's gold breaker at its third broken reference in a row
        mid.wait(5)
        tripped.set()
        return False
    b = batch.Batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=B(),
                    **{**KW1, "lanes": 2, "lane_healthy": unhealthy})
    with pytest.raises(InfraError, match="3 records in a row"):
        b.run_all()
    assert lines(tmp_path / "out/r/er_refs.jsonl") == [] and lines(tmp_path / "out/r/er_results.jsonl") == []
    assert len(lines(tmp_path / "out/r/er_refs.superseded.jsonl")) == 2 + 1  # lane 1: f0, f2; lane 2: f1
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **{**KW1, "lanes": 2})
    refs = lines(tmp_path / "out/r/er_refs.jsonl")
    assert len(refs) == 6 and all(r["ref_error"] is None for r in refs)
    out = lines(tmp_path / "out/r/er_results.jsonl")
    assert len(out) == 12 and not any(r["ref_error"] for r in out)


# ---------- the finish journal: a kill at any step of finish() is healed exactly on resume ----------

ARCHIVES = ("er_results.superseded.jsonl", "er_refs.superseded.jsonl")
TIMING = ("seconds", "behat_seconds", "cold", "mem_peak_bytes", "log_sha256")


class Kill:
    """Kills the process (KeyboardInterrupt) at one step of Batch.finish, or of the replay in Batch.prepare. A step
    is the nth atomic_write of one file in that call; mode: before (nothing written), during (half of the temporary
    file written, not renamed), after (written), cut (the archive append just before this rewrite lost its second
    half and ends on a cut line; the file is not rewritten; when that append wrote nothing, a replay whose lines were
    all appended already, there is nothing to cut: as before)."""

    def __init__(self, monkeypatch, phase, name, nth, mode):
        real_write, real_phase = batch.atomic_write, getattr(batch.Batch, phase)
        self.fired, armed, seen, sizes = False, [], Counter(), {}

        def measure(run_dir):
            sizes.update({a: (run_dir / a).stat().st_size if (run_dir / a).exists() else 0 for a in ARCHIVES})

        def write(path, text):
            if not armed or self.fired:
                return real_write(path, text)
            seen[path.name] += 1
            if (path.name, seen[path.name]) != (name, nth):
                real_write(path, text)
                return measure(path.parent)
            self.fired = True
            if mode == "during":
                path.with_name(path.name + ".tmp").write_text(text[:len(text) // 2], encoding="utf-8")
            elif mode == "after":
                real_write(path, text)
            elif mode == "cut":
                archive = path.with_name(path.stem + ".superseded.jsonl")
                data, start = archive.read_bytes(), sizes[archive.name]
                new = data[start:].split(b"\n")[:-1]  # the lines this move appended
                k = len(new) // 2
                if new:
                    archive.write_bytes(data[:start] + b"".join(x + b"\n" for x in new[:k]) + new[k][:len(new[k]) // 2])
            raise KeyboardInterrupt(f"killed {mode} {name} #{nth}")

        def phase_fn(b, *a, **kw):
            armed.append(1)
            seen.clear()
            measure(b.out_dir)
            try:
                return real_phase(b, *a, **kw)
            finally:
                armed.clear()
        monkeypatch.setattr(batch, "atomic_write", write)
        monkeypatch.setattr(batch.Batch, phase, phase_fn)


class GoldDrift(Fake):
    """f0's gold now sends another body (its gate: nondeterministic); the golds of f1-f3 are ERROR."""

    def __call__(self, record, root, entry, lane, out, noop=()):
        if entry.condition == "gold" and record["feature_path"] in ("f1.feature", "f2.feature", "f3.feature"):
            self.calls.append((entry.condition, tuple(noop), label(out)))
            return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "behat exit 255"}
        return super().__call__(record, root, entry, lane, out, noop)

    def canned(self, record, entry, m, status):
        if entry.condition == "gold" and record["feature_path"] == "f0.feature" and not m:
            return trace(["passed"] * 4, ["code", "members", "extra"]), "PASS"
        return super().canned(record, entry, m, status)


def crash_setup(tmp_path, kind):
    """(run, stop, resume): a batch of `run` that stops with lines to archive in finish(), and a healthy resume.
    controls: 5 records, one lane; f0 lost its gold row (its gate reruns and changes its reference line: its stub row
    is stale), f1-f3 lost their reference and rows and are broken again on an unhealthy lane (the gold breaker stops
    at f3): finish() moves the reference lines of f1 and f2, their rows, and f0's stub row.
    output: 9 records with m.s1.skip (SKIP) and m.s1.x; every x is ERROR and the breaker canary drifts at the 8th:
    finish() moves those 8 rows."""
    if kind == "controls":
        records = recs(5)
        make_many(tmp_path, records)
        ids = sorted(records)

        def run(fake, **kw):
            return run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=fake, **{**KW1, **kw})
        run(Fake())
        refs, res = tmp_path / "out/r/er_refs.jsonl", tmp_path / "out/r/er_results.jsonl"
        refs.write_text("".join(json.dumps(o) + "\n" for o in lines(refs) if o["record_id"] not in ids[1:4]))
        res.write_text("".join(json.dumps(o) + "\n" for o in lines(res) if o["record_id"] not in ids[1:4]
                               and (o["record_id"], o["condition"]) != (ids[0], "gold")))
        return "r", lambda: run(GoldDrift(), lane_healthy=lambda lane: False), lambda: run(Fake())
    records = recs(9)
    make_many(tmp_path, records, conds=("gold",))
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    with (tmp_path / "out/r/er_refs.superseded.jsonl").open("a") as fh:
        fh.write('{"record_id": "cut')  # a cut line only the controls run itself may repair
    write_run(tmp_path / "m", "o", [it for i, r in enumerate(records.values()) for it in (
        {"record": r, "condition": "m.s1.skip", "slug": f"f{i}_3", "base": BASE, "files": {}, "skip": "extract: none"},
        {"record": r, "condition": "m.s1.x", "slug": f"f{i}_3", "base": BASE, "files": {A: "<?php // x\n"}})])

    def run(fake):
        return run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=fake, ref_run="r", **KW1)
    return "o", lambda: run(CanaryDrift({"m.s1.x": ["ERROR"] * 9})), lambda: run(Fake())


def final_state(tmp_path, run):
    """What a run ends with: its rows (timing fields left out), reference lines, archives (multisets of lines) and
    er_complete.json."""
    d = tmp_path / "out" / run

    def strip(o):
        return {k: v for k, v in o.items() if k not in TIMING}
    rows = sorted((strip(o) for o in lines(d / "er_results.jsonl")), key=lambda o: (o["record_id"], o["condition"]))
    refs = lines(d / "er_refs.jsonl") if (d / "er_refs.jsonl").exists() else []
    archives = {a: sorted(json.dumps(strip(o), sort_keys=True) for o in lines(d / a)) if (d / a).exists() else []
                for a in ARCHIVES}
    return rows, refs, archives, json.loads((d / "er_complete.json").read_text())


@pytest.fixture(scope="module")
def unkilled(tmp_path_factory):
    """The final state of each crash run when it is not killed: it stops, finish() archives, a healthy resume."""
    want = {}
    for kind in ("controls", "output"):
        tmp = tmp_path_factory.mktemp(kind)
        run, stop, resume = crash_setup(tmp, kind)
        with pytest.raises(InfraError):
            stop()
        resume()
        want[kind] = final_state(tmp, run)
    return want


def snapshot(d):
    return {p.relative_to(d): p.read_bytes() for p in d.rglob("*") if p.is_file()}


WRITES = {"journal": ("er_complete.json", 1), "refs": ("er_refs.jsonl", 1), "rows": ("er_results.jsonl", 1),
          "stale": ("er_results.jsonl", 2), "final": ("er_complete.json", 2)}  # the writes of finish(), in order
KILLS = [(kind, f"{step}-{mode}") for kind, steps in (("controls", ("journal", "refs", "rows", "stale", "final")),
                                                     ("output", ("journal", "rows", "final"))) for step in steps
         for mode in {"journal": ("before", "during", "after"), "final": ("before", "during")}.get(
             step, ("cut", "between", "after"))]


@pytest.mark.parametrize("kind, step", KILLS)
def test_a_kill_at_any_step_of_finish_is_healed_on_resume(tmp_path, monkeypatch, unkilled, kind, step):
    run, stop, resume = crash_setup(tmp_path, kind)
    controls = snapshot(tmp_path / "out/r")
    where, mode = step.split("-")
    kill = Kill(monkeypatch, "finish", *WRITES[where], "before" if mode == "between" else mode)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    assert kill.fired
    resume()
    got = final_state(tmp_path, run)
    assert got == unkilled[kind]  # the same rows, reference lines, archives (no line lost, none twice), er_complete
    rows, _, _, done = got
    assert done["complete"] is True and "archiving" not in done
    assert not any(r["er_status"] == "ERROR" for r in rows)  # no row scored in the broken environment survives
    if kind == "output":
        assert snapshot(tmp_path / "out/r") == controls  # an output run never writes its controls run


@pytest.mark.parametrize("kind", ["controls", "output"])
@pytest.mark.parametrize("mode", ["cut", "between", "after"])
@pytest.mark.parametrize("first", ["cut", "between"])  # between: the replay's rows move has nothing left to append
def test_a_second_kill_during_the_replay_in_prepare_is_healed_too(tmp_path, monkeypatch, unkilled, kind, mode, first):
    run, stop, resume = crash_setup(tmp_path, kind)
    Kill(monkeypatch, "finish", "er_results.jsonl", 1, "before" if first == "between" else "cut")  # the rows move
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    kill = Kill(monkeypatch, "prepare", "er_results.jsonl", 1, "before" if mode == "between" else mode)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        resume()  # killed again while prepare() replays the journal
    monkeypatch.undo()
    assert kill.fired
    resume()
    assert final_state(tmp_path, run) == unkilled[kind]


@pytest.mark.parametrize("when", ["run", "row"])
@pytest.mark.parametrize("kind", ["controls", "output"])
def test_a_kill_during_the_run_reruns_every_lanes_current_streak(tmp_path, monkeypatch, kind, when):
    """A SIGKILL while the lanes run, before any breaker: finish() never runs, but each streak was journaled as it
    grew (before the row it adds is written), so the resume archives it, as for a stopped batch, and recomputes it.
    when: in the next run, or right after the last row written (output: the 5th row; controls: the 2nd gold row)."""
    if kind == "output":  # the environment breaks after the start canary
        records, run, kw = traceless_output_run(tmp_path), "o", {"ref_run": "r"}
        victim, last, cond, first = sorted(records)[5], sorted(records)[4], "m.s1.x", CanaryDrift(
            {"m.s1.x": ["ERROR"] * 9})
    else:  # every run ERROR on an unhealthy lane: 2 broken references in a row (< GOLD_BREAKER)
        records, run, kw = recs(4), "r", {"lane_healthy": lambda lane: False}
        make_many(tmp_path, records)
        victim, last, cond, first = sorted(records)[2], sorted(records)[1], "gold", Broken()
    real, real_write = first.__call__, batch.Batch.write

    def killed(record, root, entry, lane, out, noop=()):
        if when == "run" and record["id"] == victim and entry.condition == cond:
            raise KeyboardInterrupt("killed")
        return real(record, root, entry, lane, out, noop)

    def write(self, path, obj):
        real_write(self, path, obj)
        if when == "row" and path.name == "er_results.jsonl" and (obj["record_id"], obj["condition"]) == (last, cond):
            raise KeyboardInterrupt("killed")
    monkeypatch.setattr(batch.Batch, "write", write)
    monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)  # the process is gone: no finish()
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=killed, **{**KW1, **kw})
    monkeypatch.undo()
    assert json.loads((tmp_path / "out" / run / "er_complete.json").read_text())["running"] is True
    run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=Fake(),
              **{**KW1, **({"ref_run": "r"} if kind == "output" else {})})
    out = lines(tmp_path / "out" / run / "er_results.jsonl")
    assert not any(r["er_status"] == "ERROR" or r["ref_error"] for r in out)  # no wrong zero, no wrong null
    assert len(out) == (9 if kind == "output" else 8)
    archived = lines(tmp_path / "out" / run / "er_results.superseded.jsonl")
    assert len(archived) == {"output": 5, "controls": 4 if when == "run" else 3}[kind]
    assert all(r["suspect"] is True for r in archived)
    assert json.loads((tmp_path / "out" / run / "er_complete.json").read_text())["complete"] is True


def test_the_replay_counts_only_the_lines_the_killed_finish_appended(tmp_path, monkeypatch):
    records = traceless_output_run(tmp_path)
    kw = dict(ref_run="r", **KW1)
    archive = tmp_path / "out/o/er_results.superseded.jsonl"
    with pytest.raises(InfraError, match="breaker"):  # 8 suspect rows archived
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path,
                  run_fn=CanaryDrift({"m.s1.x": ["ERROR"] * 9}), **kw)
    Kill(monkeypatch, "finish", "er_results.jsonl", 1, "cut")  # the same 8 rows again, byte for byte; cut at 4.5
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path,
                  run_fn=CanaryDrift({"m.s1.x": ["ERROR"] * 9}), **kw)
    monkeypatch.undo()
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), **kw)
    got = lines(archive)
    assert len(got) == 16 and len({json.dumps(o, sort_keys=True) for o in got}) == 8  # each line archived twice
    assert all(r["er_status"] == "PASS" for r in lines(tmp_path / "out/o/er_results.jsonl"))


def test_the_journal_is_dropped_once_replayed(tmp_path, monkeypatch):
    run, stop, resume = crash_setup(tmp_path, "output")
    Kill(monkeypatch, "finish", "er_results.jsonl", 1, "before")
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    real = Fake.__call__
    done = []

    def killed(self, record, root, entry, lane, out, noop=()):  # the resume dies in its 4th output run (no streak)
        if entry.condition == "m.s1.x":
            done.append(record["id"])
            if len(done) == 4:
                raise KeyboardInterrupt("killed")
        return real(self, record, root, entry, lane, out, noop)
    monkeypatch.setattr(Fake, "__call__", killed)
    monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        resume()  # replays the journal, then writes 3 PASS rows
    monkeypatch.undo()
    assert "archiving" not in json.loads((tmp_path / "out/o/er_complete.json").read_text())
    resume()
    assert len(lines(tmp_path / "out/o/er_results.superseded.jsonl")) == 8  # the 3 new rows were kept, not archived
    assert json.loads((tmp_path / "out/o/er_complete.json").read_text())["complete"] is True


def test_a_kill_in_finish_after_the_end_of_run_check_of_a_controls_run_is_healed(tmp_path, monkeypatch):
    records = recs(3)
    make_many(tmp_path, records)  # gold, stub
    bad = set(sorted(records)[1:])  # the run ends on 2 broken references (< GOLD_BREAKER) on an unhealthy lane

    class F(Broken):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if record["id"] in bad and entry.condition == "gold":
                return super().__call__(record, root, entry, lane, out, noop)
            return Fake.__call__(self, record, root, entry, lane, out, noop)
    Kill(monkeypatch, "finish", "er_refs.jsonl", 1, "before")  # the refs archive synced, er_refs.jsonl not rewritten
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=F(),
                  **{**KW1, "lane_healthy": lambda lane: False})
    monkeypatch.undo()
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)  # environment fixed
    refs, res = lines(tmp_path / "out/r/er_refs.jsonl"), lines(tmp_path / "out/r/er_results.jsonl")
    assert len(refs) == 3 and all(o["ref_error"] is None for o in refs)
    assert len(res) == 6 and not any(o["ref_error"] or o["er_status"] == "ERROR" for o in res)
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


def test_an_idle_resume_reruns_nothing_even_when_rows_equal_archived_lines(tmp_path):
    records = traceless_output_run(tmp_path)
    kw = dict(ref_run="r", **KW1)
    with pytest.raises(InfraError, match="breaker"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path,
                  run_fn=CanaryDrift({"m.s1.x": ["ERROR"] * 9}), **kw)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake({"m.s1.x": ["ERROR"] * 9}),
              **kw)  # the outputs' own errors this time: rows byte-identical to the archived suspect ones
    idle = Fake()
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=idle, **kw)
    assert [c for c in idle.calls if c[0] == "m.s1.x"] == []  # only the journal decides what moves, not identity
    assert len(lines(tmp_path / "out/o/er_results.jsonl")) == 9
    assert json.loads((tmp_path / "out/o/er_complete.json").read_text())["complete"] is True


def test_a_finished_run_has_no_journal_and_a_resume_moves_nothing(tmp_path):
    make(tmp_path)
    path = go(tmp_path, Fake())
    before = path.read_bytes()
    assert "archiving" not in json.loads((tmp_path / "out/r/er_complete.json").read_text())
    again = Fake()
    go(tmp_path, again)
    assert again.calls == [] and path.read_bytes() == before
    assert not any((tmp_path / "out/r" / a).exists() for a in ARCHIVES)


def test_a_refused_resume_keeps_the_journal_and_writes_nothing(tmp_path, monkeypatch):
    run, stop, resume = crash_setup(tmp_path, "output")
    Kill(monkeypatch, "finish", "er_results.jsonl", 1, "before")
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    d = tmp_path / "out/o"
    before = snapshot(d)
    assert len(json.loads((d / "er_complete.json").read_text())["archiving"]["rows"]) == 8
    with pytest.raises(SystemExit, match="image_id changed"):
        run_batch(tmp_path / "m", "o", recs(9), tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r",
                  **{**KW1, "provenance": {**PROV, "image_id": "sha256:other"}})
    assert snapshot(d) == before  # prepare wrote nothing, the journal included
    resume()
    out = lines(d / "er_results.jsonl")
    assert len(out) == 18 and not any(r["er_status"] == "ERROR" for r in out)
    assert json.loads((d / "er_complete.json").read_text())["complete"] is True


# ---------- the journal's details pinned; robustness of prepare and the readers ----------

@pytest.mark.parametrize("step", [f"{s}-{m}" for s in ("refs", "rows", "stale") for m in ("cut", "between")])
def test_a_kill_in_a_second_identical_stop_is_replayed_from_its_own_archive_offsets(tmp_path, monkeypatch, unkilled,
                                                                                   step):
    """Controls: the same stop twice (the results and reference files as before the first stop are put back, the
    archives kept), so the second finish() moves lines byte-identical to archived ones, and the two archives are of
    different sizes. Killed at one of its moves, the replay must count only what that move appended after the offset
    of that move's own archive: every line ends up archived exactly twice."""
    run, stop, resume = crash_setup(tmp_path, "controls")
    d = tmp_path / "out/r"
    before = {name: (d / name).read_bytes() for name in ("er_refs.jsonl", "er_results.jsonl")}
    with pytest.raises(InfraError):
        stop()
    for name, data in before.items():
        (d / name).write_bytes(data)
    where, mode = step.split("-")
    kill = Kill(monkeypatch, "finish", *WRITES[where], "before" if mode == "between" else mode)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    assert kill.fired
    resume()
    rows, refs, archives, done = final_state(tmp_path, run)
    want_rows, want_refs, want_archives, want_done = unkilled["controls"]
    assert (rows, refs, done) == (want_rows, want_refs, want_done)
    assert archives == {a: sorted(want_archives[a] * 2) for a in ARCHIVES}  # no line lost, none archived three times


@pytest.mark.parametrize("kind", ["controls", "output"])
def test_an_in_run_kill_after_an_identical_stop_archives_every_line_again(tmp_path, monkeypatch, kind):
    """A batch that dies in its run leaves a journal without archive sizes: the resume measures them, then replays.
    The lines to move are byte-identical to lines an earlier stop archived: they are archived again (the replay skips
    only what an interrupted move appended after the offset). output: a breaker stop archives 8 rows, then the same
    outputs are ERROR again and the batch dies in the 6th; controls: a gold breaker stop archives 2 broken references
    and their rows, then the same again and the batch dies in the third reference run."""
    if kind == "output":
        records, run, kw = traceless_output_run(tmp_path), "o", {"ref_run": "r"}
        with pytest.raises(InfraError, match="breaker"):
            run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path,
                      run_fn=CanaryDrift({"m.s1.x": ["ERROR"] * 9}), **{**KW1, **kw})
        victim, cond, again, new = sorted(records)[5], "m.s1.x", Fake({"m.s1.x": ["ERROR"] * 9}), {ARCHIVES[0]: 5}
    else:
        records, run, kw = recs(4), "r", {"lane_healthy": lambda lane: False}
        make_many(tmp_path, records)
        with pytest.raises(InfraError, match="3 records in a row"):
            run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=Broken(), **{**KW1, **kw})
        victim, cond, again, new = sorted(records)[2], "gold", Broken(), {ARCHIVES[0]: 4, ARCHIVES[1]: 2}
    d = tmp_path / "out" / run
    old = {a: [json.dumps(o, sort_keys=True) for o in lines(d / a)] for a in ARCHIVES if (d / a).exists()}

    def killed(record, root, entry, lane, out, noop=()):
        if record["id"] == victim and entry.condition == cond:
            raise KeyboardInterrupt("killed")  # SIGKILL: finish() never runs
        return again(record, root, entry, lane, out, noop)
    monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=killed, **{**KW1, **kw})
    monkeypatch.undo()
    assert "archive_bytes" not in json.loads((d / "er_complete.json").read_text())["archiving"]
    run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=Fake(),
              **{**KW1, **({"ref_run": "r"} if kind == "output" else {})})
    for a, n in new.items():
        got = [json.dumps(o, sort_keys=True) for o in lines(d / a)]
        assert got[:len(old[a])] == old[a] and len(got) == len(old[a]) + n  # appended again, nothing lost
        assert set(got[len(old[a]):]) <= set(old[a])  # byte-identical to lines already archived
    out = lines(d / "er_results.jsonl")
    assert not any(r["er_status"] == "ERROR" or r["ref_error"] for r in out)
    assert json.loads((d / "er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("script, streak", [(["ERROR"], ["ERROR"]),  # a streak of one row
                                            (["TIMEOUT", "TIMEOUT", "OOM"], ["TIMEOUT", "OOM"])])  # not only ERROR
def test_every_row_without_a_trace_is_journaled_before_it_is_written(tmp_path, monkeypatch, script, streak):
    records = traceless_output_run(tmp_path, n=4)
    victim, fake = sorted(records)[len(streak)], Fake({"m.s1.x": script})

    def killed(record, root, entry, lane, out, noop=()):
        if record["id"] == victim and entry.condition == "m.s1.x":
            raise KeyboardInterrupt("killed")  # SIGKILL in the next output's run: finish() never runs
        return fake(record, root, entry, lane, out, noop)
    monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=killed, ref_run="r", **KW1)
    monkeypatch.undo()
    d = tmp_path / "out/o"
    written = lines(d / "er_results.jsonl")
    journal = json.loads((d / "er_complete.json").read_text()).get("archiving") or {}
    assert [r["er_status"] for r in written] == streak
    assert sorted(map(tuple, journal.get("rows", []))) == sorted(batch.row_key(r) for r in written)
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    assert [r["er_status"] for r in lines(d / "er_results.superseded.jsonl")] == streak  # rerun, not kept
    assert [r["er_status"] for r in lines(d / "er_results.jsonl")] == ["PASS"] * 4


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_row_with_a_trace_ends_the_journaled_streak_once_it_is_written(tmp_path, monkeypatch, when):
    """Two ERROR rows, then a PASS: the streak (journaled) ends only once that PASS row is written. before: SIGKILL
    right before the PASS row is written: the journal still names the two rows, which a resume reruns; after: SIGKILL
    in the next run: the journal no longer names them (the outputs' own errors: kept)."""
    records = traceless_output_run(tmp_path, n=4)
    ids = sorted(records)
    fake, real_write = Fake({"m.s1.x": ["ERROR", "ERROR"]}), batch.Batch.write

    def killed(record, root, entry, lane, out, noop=()):
        if when == "after" and record["id"] == ids[3] and entry.condition == "m.s1.x":
            raise KeyboardInterrupt("killed")
        return fake(record, root, entry, lane, out, noop)

    def write(self, path, obj):
        if when == "before" and path.name == "er_results.jsonl" and obj["record_id"] == ids[2]:
            raise KeyboardInterrupt("killed")
        real_write(self, path, obj)
    monkeypatch.setattr(batch.Batch, "write", write)
    monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)
    with pytest.raises(KeyboardInterrupt, match="killed"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=killed, ref_run="r", **KW1)
    monkeypatch.undo()
    d = tmp_path / "out/o"
    journal = json.loads((d / "er_complete.json").read_text()).get("archiving") or {}
    errors = sorted(batch.row_key(r) for r in lines(d / "er_results.jsonl") if r["er_status"] == "ERROR")
    assert len(errors) == 2 and sorted(map(tuple, journal.get("rows", []))) == (errors if when == "before" else [])
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", **KW1)
    got = sorted(r["er_status"] for r in keyed(d / "er_results.jsonl").values())
    assert got == (["PASS"] * 4 if when == "before" else ["ERROR", "ERROR", "PASS", "PASS"])


def test_a_resume_refused_by_the_supersede_check_leaves_a_live_journal_untouched(tmp_path, monkeypatch):
    run, stop, resume = crash_setup(tmp_path, "output")
    Kill(monkeypatch, "finish", "er_results.jsonl", 1, "before")  # the finish journal is written, nothing moved
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    records = recs(9)  # the overlay writer regenerates f0's m.s1.skip (its SKIP row, outside the journal, is now stale)
    write_run(tmp_path / "m", "o", [it for i, r in enumerate(records.values()) for it in (
        {"record": r, "condition": "m.s1.skip", "slug": f"f{i}_3", "base": BASE,
         **({"files": {A: "<?php // regenerated\n"}} if i == 0 else {"files": {}, "skip": "extract: none"})},
        {"record": r, "condition": "m.s1.x", "slug": f"f{i}_3", "base": BASE, "files": {A: "<?php // x\n"}})])
    d = tmp_path / "out/o"
    before = snapshot(d)
    with pytest.raises(SystemExit, match="supersede"):
        resume()
    assert snapshot(d) == before  # the journal is replayed only once every check has passed
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r", supersede=True,
              **KW1)
    out = keyed(d / "er_results.jsonl")
    assert len(out) == 18 and not any(r["er_status"] == "ERROR" for r in out.values())
    assert out[(sorted(records)[0], "m.s1.skip")]["er_status"] == "PASS"  # the regenerated output
    assert json.loads((d / "er_complete.json").read_text())["complete"] is True


def test_journaled_lines_skip_the_supersede_check_and_are_recomputed_on_the_current_manifest(tmp_path, monkeypatch):
    """A controls run killed right after its finish journal (nothing moved yet; f0's stub row is a journaled stale row),
    then the overlay writer regenerates f0's stub overlay: the resume needs no --supersede for it (the journal
    archives it anyway, as a stale_ref row) and recomputes it on the current overlay; the operator is not asked."""
    run, stop, resume = crash_setup(tmp_path, "controls")
    Kill(monkeypatch, "finish", *WRITES["journal"], "after")
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    f0 = sorted(recs(5))[0]
    journal = json.loads((tmp_path / "out/r/er_complete.json").read_text())["archiving"]
    assert [tuple(k)[:2] for k in journal["stale_rows"]] == [(f0, "stub")]
    write_run(tmp_path / "m", "r", [{"record": r, "condition": c, "slug": f"f{i}_3", "base": BASE,
                                     "files": {A: f"<?php // {c}{' v2' if (i, c) == (0, 'stub') else ''}\n"}}
                                    for i, r in enumerate(recs(5).values()) for c in ("gold", "stub")])
    resume()  # no --supersede
    stub = keyed(tmp_path / "out/r/er_results.jsonl")[(f0, "stub")]
    from er.manifest import load_manifest
    assert stub["overlay_sha256"] == {(e.record_id, e.condition): e for e in load_manifest(tmp_path / "m/r")}[
        (f0, "stub")].overlay_sha256  # recomputed on the regenerated overlay
    archived = [o for o in lines(tmp_path / "out/r/er_results.superseded.jsonl") if (o["record_id"], o["condition"])
                == (f0, "stub")]
    assert len(archived) == 1 and archived[0]["stale_ref"] is True and archived[0]["overlay_sha256"] != \
        stub["overlay_sha256"]
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


def test_a_late_lane_appends_only_after_finish_when_the_join_fails(tmp_path, monkeypatch):
    """The join of the lanes fails (an error other than Ctrl-C) while lane 2 is still inside its last run: finish()
    holds the lock for its whole body, so that lane's row is appended after finish() rewrote er_results.jsonl, not
    lost in the rewrite."""
    records = recs(4)  # lane 1: f0, f2; lane 2: f1, f3
    make_many(tmp_path, records, conds=("gold",))
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), **KW1)
    make_many(tmp_path, records, conds=("m.s1.x",), run="o")
    f0, f1, f2, _ = sorted(records)
    held, release, attempting, pools = threading.Event(), threading.Event(), threading.Event(), []

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "m.s1.x" and record["id"] == f0:  # lane 1: a row without a trace (its streak)
                return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "x"}
            if entry.condition == "m.s1.x" and record["id"] == f1 and noop:  # lane 2: inside its M1 run
                held.set()
                release.wait(10)
            if entry.condition == "m.s1.x" and record["id"] == f2:  # lane 1: INFRA on every attempt: the batch stops
                held.wait(10)
                return {"status": "INFRA", "infra_source": "docker", "trace": None, "error": "docker: gone"}
            return super().__call__(record, root, entry, lane, out, noop)

    class Late(batch.Batch):
        def write(self, path, obj):
            if path.name == "er_results.jsonl" and obj["record_id"] == f1:
                attempting.set()
            super().write(path, obj)

    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            pools.append(self)
            if len(pools) == 1:  # the join of the lanes fails while lane 2 still runs
                raise RuntimeError("join failed")
            super().shutdown(wait, cancel_futures=cancel_futures)
    real_write = batch.atomic_write

    def write(path, text):  # finish() rewrites er_results.jsonl without the suspect row: lane 2 tries to append now
        if path.name == "er_results.jsonl" and held.is_set() and not release.is_set():
            release.set()
            attempting.wait(5)
            time.sleep(0.2)
        real_write(path, text)
    monkeypatch.setattr(batch, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(batch, "atomic_write", write)
    b = Late(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, ref_run="r", run_fn=F(), **{**KW1, "lanes": 2})
    try:
        with pytest.raises(RuntimeError, match="join failed"):
            b.run_all()
    finally:
        release.set()
        pools[0].shutdown()
    monkeypatch.undo()
    d = tmp_path / "out/o"
    assert (f1, "m.s1.x") in keyed(d / "er_results.jsonl")  # appended after finish(), not lost in its rewrite
    assert [r["record_id"] for r in lines(d / "er_results.superseded.jsonl")] == [f0]
    assert json.loads((d / "er_complete.json").read_text())["running"] is True
    run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=Fake(), ref_run="r",
              **{**KW1, "lanes": 2})
    assert sorted(r["er_status"] for r in lines(d / "er_results.jsonl")) == ["PASS"] * 4
    assert json.loads((d / "er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("kind, where", [("controls", "refs"), ("controls", "rows"), ("output", "rows")])
def test_a_cut_last_line_in_an_archive_does_not_shift_the_journal_offsets(tmp_path, monkeypatch, unkilled, kind,
                                                                          where):
    """Each archive of the run ends on a cut line (an earlier kill) when the batch stops: finish() measures the
    archives only after dropping it, so the replay of a move killed during its append starts at the right byte."""
    run, stop, resume = crash_setup(tmp_path, kind)
    d = tmp_path / "out" / run
    d.mkdir(parents=True, exist_ok=True)
    old = {"record_id": "old", "condition": "c", "overlay_sha256": "x", "suspect": True}
    names = ARCHIVES if kind == "controls" else ARCHIVES[:1]  # an output run has no reference archive of its own
    for a in names:
        (d / a).write_text(json.dumps(old) + "\n" + '{"record_id": "cu')
    Kill(monkeypatch, "finish", *WRITES[where], "cut")
    with pytest.raises(KeyboardInterrupt, match="killed"):
        stop()
    monkeypatch.undo()
    resume()
    rows, refs, archives, done = final_state(tmp_path, run)
    want = unkilled[kind]
    assert (rows, refs, done) == (want[0], want[1], want[3])
    assert archives == {a: sorted(want[2][a] + [json.dumps(old, sort_keys=True)] * (a in names)) for a in ARCHIVES}


@pytest.mark.parametrize("how, archive", [("journal", ARCHIVES[0]), ("supersede", ARCHIVES[0]),
                                          ("supersede", ARCHIVES[1]), ("journal", "er_canary.jsonl")])
def test_a_corrupt_archive_is_refused_before_prepare_writes_anything(tmp_path, monkeypatch, how, archive):
    """A complete archive line that is not JSON (an edit by hand): prepare reads the archives, and er_canary.jsonl
    before it writes anything, so the refusal leaves the run directory as it was (er_preflight.json and
    er_complete.json included). journal: an output run that died in its run (the resume would measure the archives,
    then replay); supersede: a regenerated gold overlay (the resume would archive the reference line and the rows)."""
    if how == "journal":
        records, run, kw = traceless_output_run(tmp_path), "o", {"ref_run": "r"}
        victim, fake = sorted(records)[5], Fake({"m.s1.x": ["ERROR"] * 9})

        def killed(record, root, entry, lane, out, noop=()):
            if record["id"] == victim and entry.condition == "m.s1.x":
                raise KeyboardInterrupt("killed")
            return fake(record, root, entry, lane, out, noop)
        monkeypatch.setattr(batch.Batch, "finish", lambda self, stopped: None)
        with pytest.raises(KeyboardInterrupt, match="killed"):
            run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=killed, **{**KW1, **kw})
        monkeypatch.undo()
    else:
        records, run, kw = RECS, "r", {"supersede": True}
        make(tmp_path)
        go(tmp_path, Fake())
        regen(tmp_path)
    d = tmp_path / "out" / run
    (d / archive).write_text('{"record_id": "a"}\nnot json\n{"record_id": "b"}\n')
    before = snapshot(d)
    with pytest.raises(SystemExit, match=f"{archive}: line 2 is not a JSON object"):
        run_batch(tmp_path / "m", run, records, tmp_path / "out", tmp_path, run_fn=Fake(),
                  **{**KW1, **kw, "provenance": {**PROV, "harness_commit": "later"}})  # a doc commit since
    assert snapshot(d) == before


def test_a_write_cut_inside_a_multibyte_character_is_the_cut_last_line(tmp_path):
    make(tmp_path)
    path = go(tmp_path, Fake())
    good = path.read_bytes()
    cut = json.dumps({"record_id": "Sylius/Sylius|é.feature|3|api_x"}, ensure_ascii=False).encode("utf-8")
    cut = cut[:cut.index("é".encode("utf-8")) + 1]  # killed after the first byte of a two-byte character
    path.write_bytes(good + cut)
    assert len(batch.load_results(path)) == 3 and path.read_bytes() == good + cut  # readers never write
    again = Fake()
    go(tmp_path, again)  # the resume repairs the file and reruns nothing
    assert again.calls == [] and path.read_bytes() == good
    path.write_bytes(good + b'{"record_id": "\xc3"}\n')  # a complete line that is not UTF-8: not a cut, refused
    with pytest.raises(SystemExit, match="not UTF-8"):
        batch.load_results(path)


# ---------- completeness ----------

def test_er_complete_lists_the_missing_rows(tmp_path):
    make(tmp_path)
    with pytest.raises(InfraError):
        go(tmp_path, Fake({"stub": ["INFRA"] * 3}))
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    from er.manifest import sha256_file
    assert (done["expected"], done["rows"], done["missing"], done["complete"]) == (3, 2, [[REC["id"], "stub"]], False)
    assert done["extra"] == [] and done["duplicates"] == []
    assert done["manifest_sha256"] == sha256_file(tmp_path / "m/r/manifest.jsonl")
    go(tmp_path, Fake())
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["rows"], done["missing"], done["complete"]) == (3, [], True)


def test_er_complete_covers_the_whole_manifest_not_only_the_scope(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake(), conditions={"stub"})  # gold + stub; neg never run
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["complete"], done["expected"], done["missing"]) == (False, 3, [[REC["id"], "neg"]])
    assert done["scope_missing"] == []
    assert done["scope"] == {"conditions": ["stub"], "limit": None} and done["running"] is False


def test_a_smaller_rerun_does_not_hide_missing_rows(tmp_path):
    records = recs(3)
    make_many(tmp_path, records, conds=("gold", "stub", "neg"))
    with pytest.raises(InfraError):  # the full run stopped: rows missing
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake({"neg": ["INFRA"] * 3}),
                  **KW1)
    run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Fake(), limit=1, **KW1)  # smoke rerun
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["complete"], len(done["missing"])) == (False, 6)
    assert done["scope_missing"] == []


def test_er_complete_says_running_while_a_batch_runs(tmp_path):
    make(tmp_path, conds=("neg", "gold"))
    go(tmp_path, Fake())
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True
    make(tmp_path)  # the overlay writer adds the stub condition
    seen = []

    class Peek(Fake):  # what a reader sees mid-run (or after a SIGKILL)
        def __call__(self, *a, **kw):
            seen.append(json.loads((tmp_path / "out/r/er_complete.json").read_text()))
            return super().__call__(*a, **kw)
    go(tmp_path, Peek())
    assert seen and all((s.get("running"), s["complete"]) == (True, False) for s in seen)
    done = json.loads((tmp_path / "out/r/er_complete.json").read_text())
    assert (done["running"], done["complete"]) == (False, True)


def test_conditions_must_be_in_the_manifest(tmp_path):
    make(tmp_path)
    with pytest.raises(SystemExit, match="stubb"):
        go(tmp_path, Fake(), conditions={"stubb"})


def test_rows_no_longer_in_the_manifest_need_supersede(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    make(tmp_path, conds=("neg", "gold"))  # the overlay writer regenerated the manifest without stub
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, Fake())
    assert set(rows(go(tmp_path, Fake(), supersede=True))) == {"neg", "gold"}
    assert [r["condition"] for r in lines(tmp_path / "out/r/er_results.superseded.jsonl")] == ["stub"]


def test_a_refused_prepare_writes_nothing(tmp_path):
    make(tmp_path)
    go(tmp_path, Fake())
    run_dir = tmp_path / "out/r"
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    regen(tmp_path)  # the overlay writer regenerated the controls: needs --supersede
    with pytest.raises(SystemExit, match="supersede"):
        go(tmp_path, Fake())
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == before  # er_preflight.json included


# ---------- host side: free space and a writable run directory ----------

def low_disk(monkeypatch, when=lambda: True):
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: real(p)._replace(free=2**20) if when() else real(p))


def test_prepare_needs_free_space_on_the_host(tmp_path, monkeypatch):
    make(tmp_path)
    low_disk(monkeypatch)
    with pytest.raises(InfraError, match="free"):
        go(tmp_path, Fake())
    assert not (tmp_path / "out/r/er_preflight.json").exists()


def test_prepare_needs_a_writable_run_directory(tmp_path):
    make(tmp_path)
    run_dir = tmp_path / "out/r"
    run_dir.mkdir(parents=True)
    (run_dir / ".er_batch.lock").touch()
    run_dir.chmod(0o555)
    try:
        with pytest.raises(InfraError, match="writable"):
            go(tmp_path, Fake())
    finally:
        run_dir.chmod(0o755)


def test_the_output_breaker_trips_when_the_host_runs_out_of_space(tmp_path, monkeypatch):
    records = traceless_output_run(tmp_path)
    started = []

    class F(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            started.append(entry.condition == "m.s1.x")
            return super().__call__(record, root, entry, lane, out, noop)
    low_disk(monkeypatch, when=lambda: any(started))  # the disk fills up once the outputs run
    with pytest.raises(InfraError, match="free"):
        run_batch(tmp_path / "m", "o", records, tmp_path / "out", tmp_path, run_fn=F({"m.s1.x": ["ERROR"] * 9}),
                  ref_run="r", **KW1)
    assert [c["trigger"] for c in lines(tmp_path / "out/o/er_canary.jsonl")] == ["start"]  # tripped without a canary
    assert len(lines(tmp_path / "out/o/er_results.superseded.jsonl")) == 8


def test_the_gold_breaker_trips_on_a_healthy_lane_when_the_host_is_full(tmp_path, monkeypatch):
    records = recs(4)
    make_many(tmp_path, records)
    started = []

    class Broken(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            started.append(1)
            return {"status": "ERROR", "trace": None, "trace_sha256": None, "error": "behat exit 255"}
    low_disk(monkeypatch, when=lambda: bool(started))
    with pytest.raises(InfraError, match="free"):
        run_batch(tmp_path / "m", "r", records, tmp_path / "out", tmp_path, run_fn=Broken(), **KW1)


# ---------- provenance (the real path: provenance=None) ----------

HEAD = "c096ca3fc01545ee25078ba3df914bd2069c7fa6"


def real_prov_batch(tmp_path, head_root, **kw):
    return run_batch(tmp_path / "m", "r", RECS, tmp_path / "out", head_root, lanes=1, run_fn=Fake(),
                     lane_check=lambda lane: None, lane_healthy=lambda lane: True,
                     preflight=lambda lanes: {"schema_fingerprint": "fp"}, **kw)


def test_every_row_carries_the_seal_the_results_path_and_the_head_root(tmp_path, monkeypatch):
    make(tmp_path)
    monkeypatch.setattr(batch, "image_id", lambda image: "sha256:img")
    monkeypatch.setattr(batch, "seal_sha256", lambda: "f6303582")
    monkeypatch.setattr(batch, "harness_code", lambda root: "code")  # hermetic: HARNESS_ROOT need not be a checkout
    monkeypatch.setattr(batch, "head_commit", lambda root: HEAD)
    monkeypatch.setattr(batch, "head_clean", lambda root: True)
    path = real_prov_batch(tmp_path, tmp_path)
    for o in lines(path):
        assert o["phase2_sha256sums_sha256"] == "f6303582" and o["results_path"] == str(path.resolve())
        assert o["head_root"] == str(tmp_path.resolve()) and o["image_id"] == "sha256:img"


def git_repo(path, files):
    """A git checkout with one commit holding `files` (path -> text); returns the git command prefix."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(path)]
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    return git


def harness_setup(tmp_path, monkeypatch):
    """The real provenance path with a clone at HEAD and a harness checkout holding er/ (code and a doc), and code
    and a doc outside er/."""
    make(tmp_path)
    output_run(tmp_path)
    monkeypatch.setattr(batch, "image_id", lambda image: "sha256:img")
    monkeypatch.setattr(batch, "seal_sha256", lambda: "f6303582")
    clone, harness = tmp_path / "clone", tmp_path / "harness"
    git_repo(clone, {"a.php": "<?php\n"})
    git = git_repo(harness, {"er/a.py": "x = 1\n", "er/NOTES.md": "notes\n", "README.md": "readme\n",
                             "tests/x.py": "t = 1\n"})
    monkeypatch.setattr(batch, "HEAD", batch.head_commit(clone))
    monkeypatch.setattr(batch, "HARNESS_ROOT", harness)  # the checkout that holds er/
    return clone, harness, git


def real_prov_output_run(tmp_path, clone):
    return run_batch(tmp_path / "m", "o", RECS, tmp_path / "out", clone, lanes=1, run_fn=Fake(), ref_run="r",
                     lane_check=lambda lane: None, lane_healthy=lambda lane: True,
                     preflight=lambda lanes: {"schema_fingerprint": "fp"})


def test_every_row_carries_the_harness_code_commit_and_dirty_flag(tmp_path, monkeypatch):
    clone, harness, git = harness_setup(tmp_path, monkeypatch)
    path = real_prov_batch(tmp_path, clone)
    tree = subprocess.run(git + ["ls-tree", "-r", "HEAD", "--", "er"], capture_output=True, text=True, check=True)
    entries = sorted((ln.split("\t")[1], ln.split()[2]) for ln in tree.stdout.splitlines())  # (path, blob sha)
    code = hashlib.sha256("".join(f"{p} {s}\n" for p, s in entries if not p.endswith(".md")).encode()).hexdigest()
    assert {(o["harness_code"], o["harness_commit"], o["harness_dirty"]) for o in lines(path)} == \
        {(code, batch.head_commit(harness), False)}  # recomputable from the commit: git ls-tree, *.md left out


def flip_one_byte(path):
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("edit, code", [("er/a.py", True), ("er/NOTES.md", False), ("README.md", False),
                                        ("tests/x.py", False)])  # code outside er/ is not harness code
def test_one_byte_of_harness_code_refuses_the_resume_and_the_ref_run_a_doc_does_not(tmp_path, monkeypatch, edit,
                                                                                    code, commit):
    clone, harness, git = harness_setup(tmp_path, monkeypatch)
    real_prov_batch(tmp_path, clone)  # the controls run "r", complete
    before = batch.head_commit(harness)
    flip_one_byte(harness / edit)
    if commit:
        subprocess.run(git + ["commit", "-q", "-am", "edit"], check=True)
    now = batch.head_commit(harness)
    if code:  # the refusals name both harness commits and say what to do
        with pytest.raises(SystemExit, match="harness_code changed since this run started") as refused:
            real_prov_batch(tmp_path, clone)  # resume of "r"
        assert f"harness_commit {before} -> {now}): the code under er/ changed since this run started: restore it " \
               f"or start a new run" in str(refused.value)
        with pytest.raises(SystemExit, match="--ref-run r: harness_code differs") as refused:
            real_prov_output_run(tmp_path, clone)
        assert f"(controls run harness_commit {before}, this run {now}): the code under er/ changed since the " \
               f"controls run: start a new controls run" in str(refused.value)
    else:
        real_prov_batch(tmp_path, clone)  # docs are not harness code: "r" resumes (nothing left to run)
        assert rows(real_prov_output_run(tmp_path, clone))["deepseek.s42.full_aware"]["er_status"] == "PASS"
    assert json.loads((tmp_path / "out/r/er_complete.json").read_text())["complete"] is True


@pytest.mark.parametrize("new, code", [("er/b.py", True), ("er/b.md", False), ("b.py", False)])
def test_an_untracked_file_under_er_makes_the_harness_dirty_unless_it_is_a_doc(tmp_path, monkeypatch, new, code):
    clone, harness, git = harness_setup(tmp_path, monkeypatch)
    real_prov_batch(tmp_path, clone)
    (harness / new).write_text("y = 1\n")
    assert batch.harness_dirty(harness) is code
    if code:  # an untracked code file is harness code too: both keys change, harness_code is named first
        with pytest.raises(SystemExit, match="harness_code changed since this run started"):
            real_prov_batch(tmp_path, clone)
        with pytest.raises(SystemExit, match="--ref-run r: harness_code differs"):
            real_prov_output_run(tmp_path, clone)
    else:
        real_prov_batch(tmp_path, clone)
        real_prov_output_run(tmp_path, clone)
        assert json.loads((tmp_path / "out/o/er_preflight.json").read_text())["harness_dirty"] is False


def test_an_untracked_code_file_under_er_is_part_of_the_harness_code(tmp_path, monkeypatch):
    clone, harness, git = harness_setup(tmp_path, monkeypatch)
    (harness / "er/b.py").write_text("y = 1\n")  # untracked: a dirty run
    real_prov_batch(tmp_path, clone)
    (harness / "er/b.py").write_text("y = 2\n")  # the dirty run cannot resume once that file changed
    with pytest.raises(SystemExit, match="harness_code changed since this run started"):
        real_prov_batch(tmp_path, clone)
    (harness / "er/b.py").write_text("y = 1\n")  # restored: it resumes
    real_prov_batch(tmp_path, clone)
    assert json.loads((tmp_path / "out/r/er_preflight.json").read_text())["harness_dirty"] is True


def test_the_real_path_refuses_when_the_harness_code_cannot_be_computed(tmp_path, monkeypatch):
    clone, harness, git = harness_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "HARNESS_ROOT", tmp_path / "missing")  # not a git checkout: git fails
    with pytest.raises(SystemExit, match="harness_code"):
        real_prov_batch(tmp_path, clone)
    assert not (tmp_path / "out/r/er_preflight.json").exists()  # no run recorded with an empty identity


@pytest.mark.parametrize("controls, output", [(True, False), (False, True)])
def test_ref_run_needs_a_clean_harness_in_both_runs(tmp_path, controls, output):
    make(tmp_path)
    output_run(tmp_path)
    kw = dict(lanes=1, run_fn=Fake(), lane_check=lambda lane: None, lane_healthy=lambda lane: True)
    run_batch(tmp_path / "m", "r", RECS, tmp_path / "out", tmp_path, provenance={**PROV, "harness_dirty": controls},
              **kw)
    with pytest.raises(SystemExit, match="--ref-run r: harness_dirty"):
        run_batch(tmp_path / "m", "o", RECS, tmp_path / "out", tmp_path, ref_run="r",
                  provenance={**PROV, "harness_dirty": output}, **kw)


def test_the_head_root_must_be_the_pinned_clone(tmp_path, monkeypatch):
    make(tmp_path)
    monkeypatch.setattr(batch, "image_id", lambda image: "sha256:img")
    monkeypatch.setattr(batch, "seal_sha256", lambda: "f6303582")
    with pytest.raises(SystemExit, match="head-root"):
        real_prov_batch(tmp_path, tmp_path / "missing")
    other = tmp_path / "other"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(other)]
    other.mkdir()
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
    with pytest.raises(SystemExit, match="head-root"):
        real_prov_batch(tmp_path, other)  # a git clone, at another commit


# ---------- host health in the gate, lane checks after every output run, prepare checks ----------

def test_an_infra_like_output_on_a_full_host_is_not_the_outputs_error(tmp_path, monkeypatch):
    make(tmp_path)
    full = []  # the host fills up once the stub starts
    monkeypatch.setattr(batch, "host_problem", lambda out_dir: "host full" if full else None)

    class Filling(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            if entry.condition == "stub":
                full.append(1)
            return super().__call__(record, root, entry, lane, out, noop)
    with pytest.raises(InfraError, match="after 3 attempts"):
        go(tmp_path, Filling({"stub": ["INFRA:inner", "INFRA:inner", "INFRA:inner"]}), lane_healthy=lambda lane: True)
    assert "stub" not in rows(tmp_path / "out/r/er_results.jsonl")


class Logged(Fake):
    """A Fake that appends ("run", condition, M1?, status) to `log` after each run (lane checks append the lane)."""

    def __init__(self, log, script=None):
        super().__init__(script)
        self.log = log

    def __call__(self, record, root, entry, lane, out, noop=()):
        r = super().__call__(record, root, entry, lane, out, noop)
        self.log.append(("run", entry.condition, bool(noop), r["status"]))
        return r


def checked_after_each_non_pass_run(log):
    return all(i + 1 < len(log) and log[i + 1] == 1 for i, e in enumerate(log) if isinstance(e, tuple) and e[3] != "PASS")


def test_every_non_pass_output_run_is_followed_by_a_lane_check(tmp_path):
    make(tmp_path)
    log = []
    go(tmp_path, Logged(log), checks=log)
    assert checked_after_each_non_pass_run(log)
    outputs = {(e[1], e[2]) for e in log if isinstance(e, tuple) and e[1] in ("neg", "stub") and e[3] != "PASS"}
    assert outputs == {("neg", False), ("neg", True), ("stub", True)}  # normal and M1 runs


def test_a_failed_gold_reference_run_is_followed_by_a_lane_check(tmp_path):
    make(tmp_path)
    log = []
    go(tmp_path, Logged(log, {"gold": ["FAIL"] * 4}), checks=log)
    assert log[:2] == [("run", "gold", False, "FAIL"), 1]  # the reference's normal run, then the check
    assert checked_after_each_non_pass_run(log)


def test_a_lane_check_failure_after_an_output_run_stops_before_its_row(tmp_path):
    make(tmp_path)
    last = []

    class Last(Fake):
        def __call__(self, record, root, entry, lane, out, noop=()):
            last[:] = [(entry.condition, bool(noop))]
            return super().__call__(record, root, entry, lane, out, noop)

    def check(lane):
        if last == [("stub", True)]:
            raise InfraError("lane 1: schema fingerprint differs")
    with pytest.raises(InfraError, match="fingerprint"):
        run_batch(tmp_path / "m", "r", RECS, tmp_path / "out", tmp_path, lanes=1, run_fn=Last(), lane_check=check,
                  lane_healthy=lambda lane: True, provenance=PROV)
    assert "stub" not in rows(tmp_path / "out/r/er_results.jsonl")


def test_a_controls_record_without_gold_is_refused_before_any_run(tmp_path):
    other = {**REC, "id": REC["id"].replace("|3|", "|9|"), "scenario_line": 9}
    items = [{"record": REC, "condition": c, "slug": "f_3", "base": BASE, "files": {} if c == "neg" else {A: "<?php // x\n"}}
             for c in ("neg", "gold", "stub")]
    items += [{"record": other, "condition": "stub", "slug": "f_9", "base": BASE, "files": {A: "<?php // y\n"}}]
    write_run(tmp_path / "m", "r", items)
    fake = Fake()
    with pytest.raises(SystemExit, match="gold"):
        run_batch(tmp_path / "m", "r", {REC["id"]: REC, other["id"]: other}, tmp_path / "out", tmp_path, lanes=1,
                  run_fn=fake, lane_check=lambda lane: None, lane_healthy=lambda lane: True, provenance=PROV)
    assert fake.calls == []


def test_the_head_root_must_be_a_clean_tree(tmp_path, monkeypatch):
    make(tmp_path)
    monkeypatch.setattr(batch, "image_id", lambda image: "sha256:img")
    monkeypatch.setattr(batch, "seal_sha256", lambda: "f6303582")
    monkeypatch.setattr(batch, "harness_code", lambda root: "code")  # hermetic: HARNESS_ROOT need not be a checkout
    clone = tmp_path / "clone"
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(clone)]
    clone.mkdir()
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["config", "status.showUntrackedFiles", "no"], check=True)  # must not hide untracked files
    (clone / "a.php").write_text("<?php\n")
    subprocess.run(git + ["add", "a.php"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    monkeypatch.setattr(batch, "HEAD", subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    assert batch.head_clean(clone)
    real_prov_batch(tmp_path, clone)  # a clean clone at HEAD is accepted
    (clone / "b.php").write_text("<?php\n")  # untracked
    with pytest.raises(SystemExit, match="clean"):
        real_prov_batch(tmp_path, clone)
    (clone / "b.php").unlink()
    (clone / "a.php").write_text("<?php // edited\n")  # a tracked file changed
    with pytest.raises(SystemExit, match="clean"):
        real_prov_batch(tmp_path, clone)


def test_host_problem_reports_a_vanished_run_directory(tmp_path):
    why = batch.host_problem(tmp_path / "gone")  # e.g. removed while a batch runs: a reason, not a traceback
    assert why and "gone" in why
