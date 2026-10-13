import json
from pathlib import Path

from v2tools.records import build, hidden_for, render, snapshot_fns

def fn(fn_id, times, suites=("default",), scope=(), catch_all=False, first_line=1, dead=False, contains_fn=False):
    anns = [{"def_id": f"{fn_id.split(':')[0]}:{first_line + i}", "line": first_line + i, "keyword": "When",
             "pattern": f"p{i}", "time": t} for i, t in enumerate(times)]
    code = "\n".join([f"#[When('p{i}')]" for i in range(len(times))] + ["public function f() {}"])
    return {"fn_id": fn_id, "path": fn_id.split(":")[0], "first_line": first_line, "body_end": first_line + len(times),
            "code": code, "annotations": anns, "suites": list(suites), "scope_tags": [list(s) for s in scope],
            "catch_all": catch_all, "dead": dead, "contains_fn": contains_fn}

def test_render_keeps_old_patterns_and_the_records_own_create_patterns_only():
    f = fn("a.php:9", [10, 30, 20])
    shown = render(f, 20, frozenset({"a.php:3"}))
    assert shown.splitlines() == ["#[When('p0')]", "#[When('p2')]", "public function f() {}"]
    assert render(f, None).count("#[When") == 3

def test_snapshot_rules():
    fns = {"a:1": fn("a:1", [10], suites=("s",)), "b:1": fn("b:1", [10], suites=("t",)), "c:1": fn("c:1", [60], suites=("s",)),
           "d:1": fn("d:1", [10], suites=("s",), catch_all=True), "e:1": fn("e:1", [10], suites=("s",))}
    rec = {"tags": [], "scoping": "behat_suites", "suite": "s", "scenario_time": 50, "partition": "train"}
    forb = {"test": ["e:1"], "dev": ["a:1"], "aux": ["x:1"]}
    assert hidden_for("train", forb) == {"e:1", "a:1", "x:1"} and hidden_for("dev", forb) == {"e:1", "x:1"}
    assert hidden_for("test", forb) == hidden_for("aux_test", forb) == set() and hidden_for("train", {}) == set()
    assert snapshot_fns(rec, fns, hidden_for("train", forb)) == []
    assert snapshot_fns({**rec, "partition": "dev"}, fns, hidden_for("dev", forb)) == ["a:1"]
    assert snapshot_fns({**rec, "partition": "aux_test"}, fns, hidden_for("aux_test", forb)) == ["a:1", "e:1"]
    assert snapshot_fns(rec, fns, {"e:1"}) == ["a:1"]   # snapshot_fns hides exactly what it is given, whatever the partition
    flagged = {**fns, "g:1": fn("g:1", [10], suites=("s",), dead=True), "h:1": fn("h:1", [10], suites=("s",), contains_fn=True)}
    assert snapshot_fns({**rec, "partition": "test"}, flagged, set()) == ["a:1", "e:1"]   # dead and span-swallowing: never shown
    try:
        hidden_for("Train", forb)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown partition name must raise, not hide nothing")
    scoped = {"m:1": fn("m:1", [10], scope=[("@manual",)]), "u:1": fn("u:1", [10])}
    sf = {"tags": ["@Other"], "scoping": "specflow_scope", "suite": "default", "scenario_time": 50, "partition": "test"}
    assert snapshot_fns(sf, scoped, set()) == ["u:1"]
    assert snapshot_fns({**sf, "tags": ["@Manual"]}, scoped, set()) == ["m:1", "u:1"]

def test_build_writes_one_record_per_member_pair(tmp_path: Path):
    (tmp_path / "hist" / "demo").mkdir(parents=True)
    (tmp_path / "hist" / "demo" / "f.feature").write_text(
        "Feature: F\n  Background:\n    Given old\n\n  @x\n  Scenario: s\n    When new\n      | a |\n")
    (tmp_path / "sources.json").write_text(json.dumps([{"repo": "r", "dir": "demo", "language": "php"}]))
    pair = {"repo": "r", "commit": "c0", "feature_path": "f.feature", "scenario_line": 6, "scenario_name": "s", "suite": "default",
            "tags": ["@x"], "steps": ["old", "new"], "scenario_min_time": 50, "new_target_fns": ["s.php:9"]}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    p2 = tmp_path / "p2"
    p2.mkdir()
    old, new = fn("s.php:3", [10], first_line=1), fn("s.php:9", [50], first_line=7)
    (p2 / "catalog.jsonl").write_text("".join(json.dumps({"repo": "r", **f}) + "\n" for f in (old, new)))
    (p2 / "labels.jsonl").write_text(json.dumps({"repo": "r", "feature_path": "f.feature", "scenario_line": 6, "suite": "default",
        "step_def_ids": [["s.php:1"], ["s.php:7"]], "step_kinds": ["reuse", "create"], "create_def_ids": ["s.php:7"],
        "create_fns": ["s.php:9"], "reuse_fns": ["s.php:3"]}) + "\n")
    (p2 / "split_members.jsonl").write_text(json.dumps({"repo": "r", "feature_path": "f.feature", "scenario_line": 6,
        "suite": "default", "partition": "test", "subsets": [], "exec_core": False}) + "\n")
    (p2 / "forbidden.json").write_text(json.dumps({"r": {"test": ["s.php:9"], "dev": []}}))
    assert build(tmp_path / "sources.json", tmp_path / "hist", tmp_path / "pairs.jsonl", p2) == {"train": 0, "dev": 0, "test": 1, "aux_test": 0}
    rec = json.loads((p2 / "records" / "test.jsonl").read_text())
    assert rec["scenario"] == "@x\nScenario: s\n  When new\n    | a |" and rec["background"] == "Background:\n  Given old"
    assert rec["step_fns"] == ["s.php:3", "s.php:9"] and rec["framework"] == "Behat" and rec["snapshot_size"] == 1
    assert rec["create"] == [{"fn_id": "s.php:9", "path": "s.php", "code": "#[When('p0')]\npublic function f() {}",
                              "preexisting": False, "post_cutoff_code": True}]
    assert rec["t_vis"] == 50 and rec["neighbours"] is None and rec["create_def_ids"] == ["s.php:7"]

def test_build_hides_answers_by_partition(tmp_path: Path):
    (tmp_path / "hist" / "demo").mkdir(parents=True)
    (tmp_path / "hist" / "demo" / "f.feature").write_text("Feature: F\n  Scenario: s\n    When new\n")
    (tmp_path / "sources.json").write_text(json.dumps([{"repo": "r", "dir": "demo", "language": "php"}]))
    pair = {"repo": "r", "commit": "c0", "feature_path": "f.feature", "scenario_line": 2, "scenario_name": "s", "suite": "default",
            "tags": [], "steps": ["new"], "scenario_min_time": 50, "new_target_fns": []}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(pair) + "\n")
    old, new = fn("s.php:3", [10], first_line=1), fn("s.php:9", [50], first_line=7)
    sizes = {}
    for part in ("train", "dev", "test", "aux_test"):
        for hidden in ("test", "dev", "aux"):
            p2 = tmp_path / f"p2_{part}_{hidden}"
            p2.mkdir()
            (p2 / "catalog.jsonl").write_text("".join(json.dumps({"repo": "r", **f}) + "\n" for f in (old, new)))
            (p2 / "labels.jsonl").write_text(json.dumps({"repo": "r", "feature_path": "f.feature", "scenario_line": 2,
                "suite": "default", "step_def_ids": [["s.php:7"]], "step_kinds": ["create"], "create_def_ids": ["s.php:7"],
                "create_fns": ["s.php:9"], "reuse_fns": []}) + "\n")
            (p2 / "split_members.jsonl").write_text(json.dumps({"repo": "r", "feature_path": "f.feature", "scenario_line": 2,
                "suite": "default", "partition": part, "subsets": [], "exec_core": False}) + "\n")
            (p2 / "forbidden.json").write_text(json.dumps({"r": {"test": [], "dev": [], "aux": [], hidden: ["s.php:3"]}}))
            build(tmp_path / "sources.json", tmp_path / "hist", tmp_path / "pairs.jsonl", p2)
            sizes[(part, hidden)] = json.loads((p2 / "records" / f"{part}.jsonl").read_text())["snapshot_size"]
    assert sizes == {("train", "test"): 0, ("train", "dev"): 0, ("train", "aux"): 0,
                     ("dev", "test"): 0, ("dev", "dev"): 1, ("dev", "aux"): 0,
                     ("test", "test"): 1, ("test", "dev"): 1, ("test", "aux"): 1,
                     ("aux_test", "test"): 1, ("aux_test", "dev"): 1, ("aux_test", "aux"): 1}

def test_train_records_list_their_direct_neighbours(tmp_path: Path):
    (tmp_path / "hist" / "demo").mkdir(parents=True)
    (tmp_path / "sources.json").write_text(json.dumps([{"repo": "r", "dir": "demo", "language": "php"}]))
    p2 = tmp_path / "p2"
    p2.mkdir()
    pairs, labels, members = [], [], []
    # a and b CREATE the same function; b and c share a feature file; d stands alone. Not transitive: a is not c's neighbour.
    for name, path, create, times in (("a", "a.feature", "s.php:9", (40,)), ("b", "bc.feature", "s.php:9", (40,)),
                                      ("c", "bc.feature", "t.php:9", (40, 60)), ("d", "d.feature", "u.php:9", (40,))):
        line = {"a": 2, "b": 2, "c": 5, "d": 2}[name]
        where = {"repo": "r", "feature_path": path, "scenario_line": line, "suite": "default"}
        pairs.append({**where, "commit": "c0", "scenario_name": "s", "tags": [], "steps": ["new"], "scenario_min_time": 30,
                      "new_target_fns": [create]})
        ids = [create.replace(":9", f":{7 - i}") for i in range(len(times))]   # c has two CREATE patterns: :7 and :6
        labels.append({**where, "step_def_ids": [ids], "step_kinds": ["create"], "create_def_ids": ids, "create_fns": [create],
                       "reuse_fns": []})
        members.append({**where, "partition": "train", "subsets": [], "exec_core": False})
    for path, text in (("a.feature", "Feature: F\n  Scenario: s\n    When new\n"),
                       ("bc.feature", "Feature: F\n  Scenario: s\n    When new\n\n  Scenario: s\n    When new\n"),
                       ("d.feature", "Feature: F\n  Scenario: s\n    When new\n")):
        (tmp_path / "hist" / "demo" / path).write_text(text)
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in pairs))
    (p2 / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    (p2 / "split_members.jsonl").write_text("".join(json.dumps(x) + "\n" for x in members))
    t_fn = fn("t.php:9", [60, 40], first_line=6)   # annotations t.php:6 (time 60) and t.php:7 (time 40)
    (p2 / "catalog.jsonl").write_text("".join(json.dumps({"repo": "r", **f}) + "\n"
                                              for f in (fn("s.php:9", [40], first_line=7), t_fn, fn("u.php:9", [40], first_line=7))))
    (p2 / "forbidden.json").write_text(json.dumps({"r": {"test": [], "dev": [], "aux": []}}))
    build(tmp_path / "sources.json", tmp_path / "hist", tmp_path / "pairs.jsonl", p2)
    recs = {r["id"].split("|")[1] + ":" + r["id"].split("|")[2]: r
            for r in map(json.loads, (p2 / "records" / "train.jsonl").read_text().splitlines())}
    ids = {k: r["id"] for k, r in recs.items()}
    assert recs["a.feature:2"]["neighbours"] == [ids["bc.feature:2"]]
    assert recs["bc.feature:2"]["neighbours"] == sorted([ids["a.feature:2"], ids["bc.feature:5"]])
    assert recs["bc.feature:5"]["neighbours"] == [ids["bc.feature:2"]] and recs["d.feature:2"]["neighbours"] == []
    assert recs["bc.feature:5"]["t_vis"] == 60 and recs["bc.feature:5"]["create_def_ids"] == ["t.php:6", "t.php:7"]
