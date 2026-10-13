import fcntl
import json

import pytest

from er.manifest import write_run
from er.report import main as report_main
from er.report import report
from er.sample_fn import render_sample
from er.tamper import apply, needs_review, review

A = "src/Sylius/Behat/Context/Api/A.php"
REC = {"id": "Sylius/Sylius|f.feature|3|api_x", "suite": "api_x", "feature_path": "f.feature", "scenario_line": 3,
       "scenario_name": "s", "t_vis": 1, "background": None, "scenario": "Scenario: s\n    When b\n    Then d\n",
       "steps": ["b", "d"], "step_kinds": ["create", "create"], "create": [{"path": A}]}


def row(cond, status, er_trace, action, flags=(), then=None):
    return {"record_id": REC["id"], "condition": cond, "overlay_sha256": f"sha-{cond}", "er_status": status,
            "er": int(status == "PASS"),
            "er_trace": er_trace, "trace_action": action, "trace_then": then, "tamper_flags": list(flags),
            "seconds": 1.0, "cold": 0, "mem_peak_bytes": 2**20, "ref_run": None}


def test_report_counts_three_controls_and_flags():
    text = report([row("neg", "UNDEFINED", 0, 0), row("gold", "PASS", 1, 1), row("stub", "PASS", 0, 0),
                   row("m.s1.x", "PASS", 1, 1, flags=["env"])])
    assert "| stub | 1 | PASS 1 | 1/1 | 0/1 | 0/1 |" in text
    assert "**0/1**" in text  # stub PASS: the record does not pass the three controls
    assert "{'m.s1.x': 1}" in text and "cần xét tay (gắn cờ và er = 1, trace_action = 1 hoặc trace_then = 1): 1" in text


def test_sample_lists_event_differences(tmp_path):
    manifest = write_run(tmp_path / "m", "dev", [{"record": REC, "condition": c, "slug": "f_3",
                                                  "base": {A: "<?php\n"}, "files": {A: f"<?php // {c}\n"}}
                                                 for c in ("gold", "m.s42.full_aware")])
    run_dir = tmp_path / "out" / "dev"
    ev = {"kind": "db", "step": 0, "op": "insert", "entity": "Zone"}
    for cond, events in (("_gold_ref", [ev]), ("m.s42.full_aware", [])):
        d = run_dir / "er_runs" / "f_3" / cond / "normal"
        d.mkdir(parents=True)
        (d / "trace.json").write_text(json.dumps({"steps": [], "events": events}))
    (run_dir / "er_refs.jsonl").write_text(json.dumps({"record_id": REC["id"], "ref_dir": "er_runs/f_3/_gold_ref"}) + "\n")
    (run_dir / "er_results.jsonl").write_text(json.dumps(row("m.s42.full_aware", "PASS", 0, 0)) + "\n")
    text = render_sample(manifest, run_dir, {REC["id"]: REC}, k=30, seed=1)
    assert "Pool: 1 output" in text and "chỉ có ở gold: ('db', 'insert', 'Zone')" in text


def test_tamper_review_then_apply(tmp_path):
    run_dir = tmp_path / "out" / "r"
    run_dir.mkdir(parents=True)
    rows = [row("m.s1.a", "PASS", 1, 1, flags=["global_hook"]), row("m.s1.b", "FAIL", 0, 0, flags=["env"]),
            row("m.s1.c", "PASS", 1, 1)]
    (run_dir / "er_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    path, n = review(run_dir)
    assert n == 1 and "m.s1.a" in path.read_text()
    with pytest.raises(SystemExit, match="need a verdict"):
        apply(run_dir)
    v = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "cheat" for k in v}))
    assert apply(run_dir) == 1
    out = {o["condition"]: o for o in map(json.loads, (run_dir / "er_results.jsonl").read_text().splitlines())}
    assert (out["m.s1.a"]["er"], out["m.s1.a"]["er_trace"], out["m.s1.a"]["tamper_verdict"]) == (0, 0, "cheat")
    assert out["m.s1.a"]["trace_action"] == 0
    assert out["m.s1.c"]["er"] == 1 and "tamper_verdict" not in out["m.s1.c"]


def test_tamper_apply_takes_the_run_lock(tmp_path):
    run_dir = tmp_path / "out" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "er_results.jsonl").write_text(json.dumps(row("m.s1.c", "PASS", 1, 1)) + "\n")
    (run_dir / "er_tamper_verdicts.json").write_text("{}")
    with open(run_dir / ".er_batch.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="another batch is running on this run"):
            apply(run_dir)


def put(run_dir, rows, verdicts=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "er_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if verdicts is not None:
        (run_dir / "er_tamper_verdicts.json").write_text(json.dumps(verdicts))


def results(run_dir):
    return {o["condition"]: o for o in map(json.loads, (run_dir / "er_results.jsonl").read_text().splitlines())}


def test_needs_review_also_on_trace_action_and_never_for_controls():
    assert needs_review(row("m.s1.x", "FAIL", 0, 1, flags=["env"]))  # FAIL, but its action trace matches gold
    assert not needs_review(row("m.s1.x", "FAIL", 0, 0, flags=["env"]))
    assert not needs_review(row("gold", "PASS", 1, 1, flags=["env"], then=1))


def test_a_cheat_zeroes_every_part_and_ok_restores_it(tmp_path):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("m.s1.a", "PASS", 1, 1, flags=["global_hook"])])
    review(run_dir)
    (k,) = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    assert k == f"{REC['id']}\tm.s1.a\tsha-m.s1.a"  # the verdict is for this very overlay
    for _ in range(2):  # idempotent: the scores before the verdict are saved once
        (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "cheat"}))
        assert apply(run_dir) == 1
        r = results(run_dir)["m.s1.a"]
        assert (r["er"], r["er_trace"], r["trace_action"], r["trace_then"]) == (0, 0, 0, None)
        assert r["pre_tamper"] == {"er": 1, "er_trace": 1, "trace_action": 1, "trace_then": None}
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "ok"}))
    apply(run_dir)
    r = results(run_dir)["m.s1.a"]
    assert (r["er"], r["er_trace"], r["trace_action"], r["tamper_verdict"]) == (1, 1, 1, "ok") and "pre_tamper" not in r


@pytest.mark.parametrize("state", [{"complete": False, "running": True},  # a batch runs, or was killed
                                   {"complete": False, "running": False,  # died before the end of its journal
                                    "archiving": {"refs": [], "rows": [], "stale_rows": []}}])
def test_tamper_refuses_a_run_whose_batch_runs_or_died_before_its_end(tmp_path, state):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("m.s1.a", "PASS", 1, 1, flags=["global_hook"])], {f"{REC['id']}\tm.s1.a\tsha-m.s1.a": "cheat"})
    (run_dir / "er_complete.json").write_text(json.dumps(state))
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".er_batch.lock"}
    for action in (review, apply):
        with pytest.raises(SystemExit, match="resume the batch first"):
            action(run_dir)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.name != ".er_batch.lock"} == before
    (run_dir / "er_complete.json").write_text(json.dumps({"complete": True, "running": False}))
    assert review(run_dir)[1] == 1 and apply(run_dir) == 1  # the batch finished: the check goes on


@pytest.mark.parametrize("text", ['{"complete": tr', "[]"])  # cut short, or not an object
def test_tamper_refuses_an_unreadable_er_complete_json_with_a_message(tmp_path, text):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("m.s1.a", "PASS", 1, 1, flags=["global_hook"])], {f"{REC['id']}\tm.s1.a\tsha-m.s1.a": "cheat"})
    (run_dir / "er_complete.json").write_text(text)
    for action in (review, apply):  # a clear refusal, not a traceback
        with pytest.raises(SystemExit, match="er_complete.json: not a JSON object"):
            action(run_dir)


@pytest.mark.parametrize("other", ["unflagged", "flagged_but_zero"])  # needs_review is false for both
def test_apply_refuses_a_verdict_for_an_output_that_needs_no_review(tmp_path, other):
    run_dir = tmp_path / "out" / "r"
    c = row("m.s1.c", "PASS", 1, 1) if other == "unflagged" else row("m.s1.c", "FAIL", 0, 0, flags=["env"])
    put(run_dir, [row("m.s1.a", "PASS", 1, 1, flags=["global_hook"]), c],
        {f"{REC['id']}\tm.s1.a\tsha-m.s1.a": "ok", f"{REC['id']}\tm.s1.c\tsha-m.s1.c": "cheat"})
    before = (run_dir / "er_results.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="need no review"):
        apply(run_dir)
    assert (run_dir / "er_results.jsonl").read_bytes() == before  # nothing applied, m.s1.a included


def test_apply_clears_a_verdict_left_on_an_output_that_needs_no_review(tmp_path):
    run_dir = tmp_path / "out" / "r"  # an earlier apply put a cheat on an output nobody had to review
    left = {**row("m.s1.c", "PASS", 0, 0), "er": 0, "tamper_verdict": "cheat", "tamper_verdicts_sha256": "old",
            "pre_tamper": {"er": 1, "er_trace": 0, "trace_action": 0, "trace_then": None}}
    put(run_dir, [left], {})
    assert apply(run_dir) == 0
    r = results(run_dir)["m.s1.c"]
    assert r["er"] == 1 and not {"tamper_verdict", "tamper_verdicts_sha256", "pre_tamper"} & set(r)
    report_main(["--run", "r", "--out", str(tmp_path / "out")])  # the rows now match the verdicts file


@pytest.mark.parametrize("change", ["edited", "reformatted", "deleted"])
def test_the_report_refuses_rows_that_do_not_carry_the_current_verdicts(tmp_path, change):
    run_dir, vpath = tmp_path / "out" / "r", tmp_path / "out" / "r" / "er_tamper_verdicts.json"
    put(run_dir, [row("gold", "PASS", 1, 1), row("m.s1.a", "PASS", 1, 1, flags=["env"])])
    review(run_dir)
    (k,) = json.loads(vpath.read_text())
    vpath.write_text(json.dumps({k: "ok"}))
    apply(run_dir)
    report_main(["--run", "r", "--out", str(tmp_path / "out")])  # the rows carry the file's verdicts and sha256
    if change == "edited":
        vpath.write_text(json.dumps({k: "cheat"}))  # a verdict changed after apply
    elif change == "reformatted":
        vpath.write_text(json.dumps({k: "ok"}, indent=4))  # another file: the rows were not applied from it
    else:
        vpath.unlink()
    with pytest.raises(SystemExit, match="er_tamper_verdicts.json"):
        report_main(["--run", "r", "--out", str(tmp_path / "out")])


def test_apply_refuses_a_verdict_for_a_control(tmp_path):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("gold", "PASS", 1, 1, flags=["env"])], {f"{REC['id']}\tgold\tsha-gold": "cheat"})
    with pytest.raises(SystemExit, match="control"):
        apply(run_dir)
    assert results(run_dir)["gold"]["er"] == 1


def test_a_regenerated_output_needs_a_new_verdict(tmp_path):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [{**row("m.s1.a", "PASS", 1, 1, flags=["env"]), "overlay_sha256": "OLD"}])
    review(run_dir)
    v = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "ok" for k in v}))  # reviewed: ok
    put(run_dir, [{**row("m.s1.a", "PASS", 1, 1, flags=["env"]), "overlay_sha256": "NEW"}])  # regenerated output
    with pytest.raises(SystemExit, match="need a verdict"):
        apply(run_dir)


def test_the_report_refuses_unreviewed_flagged_outputs(tmp_path, capsys):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("gold", "PASS", 1, 1, flags=["env"]), row("m.s1.a", "PASS", 1, 1, flags=["env"])])
    with pytest.raises(SystemExit, match="1 flagged outputs not reviewed"):
        report_main(["--run", "r", "--out", str(tmp_path / "out")])
    review(run_dir)
    v = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "ok" for k in v}))
    apply(run_dir)
    report_main(["--run", "r", "--out", str(tmp_path / "out")])
    assert "| m.s1.a | 1 |" in capsys.readouterr().out


def test_the_report_warns_first_about_missing_rows():
    rows = [row("gold", "PASS", 1, 1), row("neg", "UNDEFINED", 0, 0)]
    done = {"expected": 3, "rows": 2, "missing": [[REC["id"], "stub"]], "scope_missing": [], "duplicates": [],
            "complete": False}  # missing on the WHOLE manifest, none in this run's scope: still a warning
    assert "THIẾU 1 dòng" in report(rows, done).split("\n")[0]
    assert "THIẾU" in report(rows, None).split("\n")[0] and "er_complete.json" in report(rows, None).split("\n")[0]
    changed = {**done, "rows": 3, "missing": [], "complete": True}
    assert "er_complete.json" in report(rows, changed).split("\n")[0]  # 2 rows now, 3 when the batch ended
    assert "THIẾU" not in report(rows, {**done, "missing": [], "complete": True})


def test_the_report_counts_score_errors_per_condition():
    rows = [row("gold", "PASS", 1, 1), {**row("m.s1.a", "PASS", 0, 0), "score_error": "IndexError: x"},
            {**row("m.s1.b", "PASS", 0, 0), "score_error": "TraceError: y"}, row("m.s1.c", "PASS", 1, 1)]
    assert "Lỗi khi chấm vết (score_error, ER giữ nguyên): {'m.s1.a': 1, 'm.s1.b': 1}" in report(rows, None)
    assert "(score_error, ER giữ nguyên): —" in report([row("gold", "PASS", 1, 1)], None)


def test_readers_never_write_the_results_file(tmp_path):
    manifest = write_run(tmp_path / "m", "r", [{"record": REC, "condition": c, "slug": "f_3", "base": {A: "<?php\n"},
                                                 "files": {A: f"<?php // {c}\n"}} for c in ("gold", "m.s1.a")])
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("m.s1.a", "PASS", 1, 1)])
    res = run_dir / "er_results.jsonl"
    res.write_text(res.read_text() + '{"record_id": "cut')  # a batch is writing its next line
    before = res.read_bytes()
    report_main(["--run", "r", "--out", str(tmp_path / "out")])
    review(run_dir)
    render_sample(manifest, run_dir, {REC["id"]: REC}, k=30, seed=1)
    assert res.read_bytes() == before


def test_the_sample_reads_refs_with_the_shared_reader(tmp_path):
    manifest = write_run(tmp_path / "m", "dev", [{"record": REC, "condition": c, "slug": "f_3",
                                                  "base": {A: "<?php\n"}, "files": {A: f"<?php // {c}\n"}}
                                                 for c in ("gold", "m.s42.full_aware")])
    run_dir = tmp_path / "out" / "dev"
    ev = {"kind": "db", "step": 0, "op": "insert", "entity": "Zone"}
    for cond, events in (("_gold_ref", [ev]), ("m.s42.full_aware", [])):
        d = run_dir / "er_runs" / "f_3" / cond / "normal"
        d.mkdir(parents=True)
        (d / "trace.json").write_text(json.dumps({"steps": [], "events": events}))
    old = json.dumps({"record_id": REC["id"], "ref_dir": "er_runs/f_3/_nowhere"})
    new = json.dumps({"record_id": REC["id"], "ref_dir": "er_runs/f_3/_gold_ref"})
    (run_dir / "er_refs.jsonl").write_text(old + "\n" + new + "\n" + new[:30])  # later line wins; cut last line
    (run_dir / "er_results.jsonl").write_text(json.dumps(row("m.s42.full_aware", "PASS", 0, 0)) + "\n")
    text = render_sample(manifest, run_dir, {REC["id"]: REC}, k=30, seed=1)
    assert "chỉ có ở gold: ('db', 'insert', 'Zone')" in text


def test_a_cheat_keeps_the_parts_that_are_not_evaluable_null(tmp_path):
    run_dir = tmp_path / "out" / "r"
    broken = {**row("m.s1.a", "PASS", None, None, flags=["env"]), "ref_error": "gold is FAIL"}  # no usable reference
    put(run_dir, [broken])
    review(run_dir)
    (k,) = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps({k: "cheat"}))
    assert apply(run_dir) == 1
    r = results(run_dir)["m.s1.a"]
    assert (r["er"], r["er_trace"], r["trace_action"], r["trace_then"]) == (0, None, None, None)
    assert "| m.s1.a | 1 | PASS 1 | 0/1 | 0/0 | 0/0 | 0/0 | 1 |" in report([r])  # ER-trace denominator unchanged


def test_a_cheat_stays_under_review_after_apply(tmp_path):
    cheated = {**row("m.s1.a", "PASS", 0, 0, flags=["env"]), "er": 0,
               "pre_tamper": {"er": 1, "er_trace": 1, "trace_action": 1, "trace_then": None}}
    assert needs_review(cheated)  # judged on the scores before the verdict
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [cheated], {})
    with pytest.raises(SystemExit, match="need a verdict"):  # its verdict was dropped from the file
        apply(run_dir)
    assert review(run_dir)[1] == 1


def test_apply_refuses_verdicts_not_keyed_by_the_overlay(tmp_path):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("m.s1.a", "PASS", 1, 1, flags=["env"])], {f"{REC['id']}\tm.s1.a": "ok"})  # old two-part key
    with pytest.raises(SystemExit, match="not keyed"):
        apply(run_dir)


def test_the_report_warns_about_duplicate_rows_and_a_running_batch():
    rows = [row("gold", "PASS", 1, 1), row("neg", "UNDEFINED", 0, 0)]
    dup = {"expected": 2, "rows": 2, "missing": [], "duplicates": [[REC["id"], "neg"]], "complete": False}
    assert "1 (record, condition) có hai dòng" in report(rows, dup).split("\n")[0]
    running = {"complete": False, "running": True}
    assert "đang chạy" in report(rows, running).split("\n")[0]


def test_needs_review_on_trace_then_alone():
    row = {"record_id": "r", "condition": "m.s1.x", "overlay_sha256": "o", "tamper_flags": ["env"], "er": 0,
           "trace_action": 0, "trace_then": 1}
    assert needs_review(row)


def test_the_sample_pool_leaves_out_rows_with_a_score_error(tmp_path):
    manifest = write_run(tmp_path / "m", "dev", [{"record": REC, "condition": c, "slug": "f_3",
                                                  "base": {A: "<?php\n"}, "files": {A: f"<?php // {c}\n"}}
                                                 for c in ("gold", "m.s42.full_aware", "m.s42.rag_only")])
    run_dir = tmp_path / "out" / "dev"
    ev = {"kind": "db", "step": 0, "op": "insert", "entity": "Zone"}
    for cond, events in (("_gold_ref", [ev]), ("m.s42.full_aware", []), ("m.s42.rag_only", [{"kind": "db"}])):
        d = run_dir / "er_runs" / "f_3" / cond / "normal"
        d.mkdir(parents=True)
        (d / "trace.json").write_text(json.dumps({"steps": [], "events": events}))
    ref = {"record_id": REC["id"], "ref_dir": "er_runs/f_3/_gold_ref"}
    (run_dir / "er_refs.jsonl").write_text(json.dumps(ref) + "\n")
    scored = {**row("m.s42.rag_only", "PASS", 0, 0), "score_error": "KeyError: 'step'"}  # 0 by rule, not by comparison
    put(run_dir, [row("m.s42.full_aware", "PASS", 0, 0), scored])
    text = render_sample(manifest, run_dir, {REC["id"]: REC}, k=30, seed=1)
    assert "Pool: 1 output" in text and "m.s42.rag_only" not in text


def test_the_report_counts_the_tamper_verdicts_per_condition(tmp_path):
    run_dir = tmp_path / "out" / "r"
    put(run_dir, [row("gold", "PASS", 1, 1), row("m.s1.a", "PASS", 1, 1, flags=["env"]),
                  row("m.s1.b", "PASS", 1, 1, flags=["global_hook"]), row("m.s1.c", "FAIL", 0, 0, flags=["env"])])
    assert "Kết quả xét tay tamper theo điều kiện: ok —; cheat —" in report(list(results(run_dir).values()))
    review(run_dir)
    v = json.loads((run_dir / "er_tamper_verdicts.json").read_text())
    verdicts = {k: "cheat" if "\tm.s1.b\t" in k else "ok" for k in v}
    (run_dir / "er_tamper_verdicts.json").write_text(json.dumps(verdicts))
    apply(run_dir)
    text = report(list(results(run_dir).values()))  # m.s1.c: flagged, not to review, no verdict
    assert "Kết quả xét tay tamper theo điều kiện: ok {'m.s1.a': 1}; cheat {'m.s1.b': 1}" in text
