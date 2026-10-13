import json
from pathlib import Path

from v2tools.final_split import run, select

CFG = {"cutoff_utc": "1970-01-01T00:01:40Z", "seed": 1, "test_quota": {"r": 1}, "exec_suite_prefix": {},
       "dev_total": 1, "learning_curve": [1], "excluded_tags": {"r": ["@skip"]},
       "minimum_unique_scenarios": {"train": 600, "dev": 150, "test": 150}}

def pair(path, line, t, key, suite="default", tags=(), new=(), fns=("s:1",), amb=0, repo="r"):
    return {"repo": repo, "feature_path": path, "scenario_line": line, "suite": suite, "status": "BOUND",
            "ambiguous_steps": amb, "tags": list(tags), "structure_key": key, "scenario_min_time": t,
            "new_target_ids": [f"{f}0" for f in new], "new_target_fns": list(new),
            "target_fns": sorted({*fns, *new}), "targets_new_count": len(new)}

def lab(p, create=()):
    return {"create_fns": list(create), "create_def_ids": [f"{f}0" for f in create]}

def labels_for(rows, creates):
    return {(r["repo"], r["feature_path"], r["scenario_line"], r["suite"]): lab(r, c) for r, c in zip(rows, creates)}

def lines(rows):
    return sorted((r["feature_path"], r["scenario_line"]) for r in rows)

def test_split_selection_rules():
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"]),            # the only test candidate
            pair("t.feature", 5, 50, "k2"),                           # same file as the test item
            pair("a.feature", 1, 50, "k3", fns=["n:1", "s:1"]),       # uses the test answer n:1
            pair("b.feature", 1, 50, "k1"),                           # repeats the structure of the test item
            pair("c.feature", 1, 50, "k5", amb=1),
            pair("d.feature", 1, 50, "k6", tags=["@SKIP"]),           # excluded tags match case-insensitively
            pair("e.feature", 1, 50, "k7"),                           # nothing to create
            pair("f.feature", 1, 50, "k8"), pair("g.feature", 1, 50, "k9")]
    creates = [["n:1"], ["c:1"], ["c:2"], ["c:3"], ["c:4"], ["c:5"], [], ["c:6"], ["c:7"]]
    res = select(rows, labels_for(rows, creates), CFG)
    assert lines(res["parts"]["test"]) == [("t.feature", 1)]
    assert {k: lines(v) for k, v in res["aside"].items() if v} == {
        "same_file_as_test": [("t.feature", 5)], "uses_test_answer": [("a.feature", 1)],
        "duplicate_structure": [("b.feature", 1)], "ambiguous": [("c.feature", 1)],
        "excluded_tag": [("d.feature", 1)], "no_create": [("e.feature", 1)]}
    assert len(res["parts"]["dev"]) == 1 and len(res["parts"]["train"]) == 1
    assert list(res["subsets"].values()) == [["lc1"]] and res["answer_fns"] == [("r", "n:1")]

def test_exec_prefix_limits_test_candidates_to_runnable_suites():
    rows = [pair("x.feature", 1, 200, "a", suite="ui_x", new=["n:1"], repo="s"),
            pair("y.feature", 1, 200, "b", suite="ui_y", new=["n:2"], repo="s"),
            pair("y.feature", 1, 200, "b", suite="api_y", new=["n:2"], repo="s")]
    cfg = {**CFG, "test_quota": {"s": 5}, "exec_suite_prefix": {"s": "api"}, "excluded_tags": {}}
    res = select(rows, labels_for(rows, [["n:1"], ["n:2"], ["n:2"]]), cfg)
    assert sorted((r["feature_path"], r["suite"]) for r in res["parts"]["test"]) == [("y.feature", "api_y"), ("y.feature", "ui_y")]

def test_run_writes_phase1_compatible_files(tmp_path: Path):
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"]), pair("f.feature", 1, 50, "k8"), pair("g.feature", 1, 50, "k9")]
    labels = [{"repo": "r", "feature_path": r["feature_path"], "scenario_line": r["scenario_line"], "suite": "default",
               **lab(r, c)} for r, c in zip(rows, [["n:1"], ["c:6"], ["c:7"]])]
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in labels))
    (tmp_path / "cfg.json").write_text(json.dumps(CFG))
    (tmp_path / "catalog.jsonl").write_text("")
    (tmp_path / "sources.json").write_text(json.dumps([{"repo": "r", "language": "php"}]))
    report = run(tmp_path / "pairs.jsonl", tmp_path / "labels.jsonl", tmp_path / "cfg.json", tmp_path / "out",
                 tmp_path / "catalog.jsonl", tmp_path / "sources.json")
    assert report["verdict"] == "SHORT" and report["input_status"] == {"BOUND": 3}
    members = [json.loads(l) for l in (tmp_path / "out" / "split_members.jsonl").read_text().splitlines()]
    assert [(m["feature_path"], m["partition"], m["subsets"]) for m in members] == [
        ("f.feature", "train", ["lc1"]), ("g.feature", "dev", []), ("t.feature", "test", [])]
    assert json.loads((tmp_path / "out" / "forbidden.json").read_text()) == {"r": {"test": ["n:1"], "dev": ["c:7"], "aux": []}}

def test_required_tags_any_keeps_only_scenarios_the_ci_selects():
    rows = [pair("a.feature", 1, 50, "k1", tags=["@GOV"], repo="p"), pair("b.feature", 1, 50, "k2", repo="p"),
            pair("c.feature", 1, 50, "k3", tags=["@other"], repo="p")]
    cfg = {**CFG, "test_quota": {}, "excluded_tags": {}, "required_tags_any": {"p": ["@gov", "@e2e"]},
           "dev_total": 0, "learning_curve": []}
    res = select(rows, labels_for(rows, [["c:1"], ["c:2"], ["c:3"]]), cfg)
    assert lines(res["aside"]["excluded_tag"]) == [("b.feature", 1), ("c.feature", 1)]
    assert lines(res["parts"]["train"]) == [("a.feature", 1)]

def test_dev_is_drawn_in_whole_co_creation_groups():
    # b and c sit in different files but CREATE the same function x:1 (added in one commit): same partition, always
    rows = [pair("a.feature", 1, 50, "k1"), pair("b.feature", 1, 50, "k2"), pair("c.feature", 1, 50, "k3"),
            pair("d.feature", 1, 50, "k4")]
    creates = [["a:1"], ["x:1"], ["x:1", "c:1"], ["d:1"]]
    for seed in range(8):
        res = select(rows, labels_for(rows, creates), {**CFG, "seed": seed, "test_quota": {}, "dev_total": 2})
        part_of = {r["feature_path"]: part for part in ("train", "dev") for r in res["parts"][part]}
        assert part_of["b.feature"] == part_of["c.feature"]
        assert ("r", "x:1") not in set(res["dev_answer_fns"]) or part_of["b.feature"] == "dev"

def test_aux_holdout_takes_the_least_depended_on_scenarios_before_dev():
    # p1 is used by q1 (P) and shares p.feature with r1; q1 has no dependants; r1's only dependant is p1 (same file)
    rows = [pair("p.feature", 1, 50, "k1", repo="h"), pair("q.feature", 1, 50, "k2", repo="h", fns=["P", "s:1"]),
            pair("p.feature", 5, 50, "k3", repo="h")]
    creates = [["P"], ["Q"], ["R"]]
    cfg = {**CFG, "test_quota": {}, "excluded_tags": {}, "dev_total": 0, "learning_curve": []}
    one = select(rows, labels_for(rows, creates), {**cfg, "aux_holdout": {"h": 1}})
    assert lines(one["aux"]) == [("q.feature", 1)] and lines(one["parts"]["train"]) == [("p.feature", 1), ("p.feature", 5)]
    two = select(rows, labels_for(rows, creates), {**cfg, "aux_holdout": {"h": 2}})
    assert lines(two["aux"]) == [("p.feature", 5), ("q.feature", 1)]
    assert lines(two["aside"]["uses_aux_answer"]) == [("p.feature", 1)] and two["aux_answer_fns"] == [("h", "Q"), ("h", "R")]

def test_train_scenario_reusing_a_dev_answer_is_set_aside():
    # d1 CREATEs D; t1 (another file, another group: it only REUSES D) must not stay in train once d1 is in dev
    rows = [pair("d.feature", 1, 50, "k1"), pair("t.feature", 1, 60, "k2", fns=["D", "s:1"])]
    creates = [["D"], ["T"]]
    for seed in range(12):
        res = select(rows, labels_for(rows, creates), {**CFG, "seed": seed, "test_quota": {}, "dev_total": 1, "learning_curve": []})
        if lines(res["parts"]["dev"]) == [("d.feature", 1)]:
            assert lines(res["aside"]["uses_dev_answer"]) == [("t.feature", 1)] and not res["parts"]["train"]
            break
    else:
        raise AssertionError("no seed put d.feature in dev")

def test_post_cutoff_twin_of_a_test_answer_joins_the_answer():
    body = "public function filter(string $p): void { $this->page->filterByCode($p); $this->page->filter(); }"
    cfn = lambda t: {"first_line": 1, "code": f"#[When('p')]\n{body}", "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn(200), "twin:1": cfn(150), "old:1": cfn(50)}}   # cutoff is t = 100
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"]),                       # the test item; its answer is n:1
            pair("a.feature", 1, 160, "k2", fns=["twin:1", "s:1"]),             # uses the post-cutoff twin: set aside
            pair("b.feature", 1, 60, "k3", fns=["old:1", "s:1"]),               # uses the pre-cutoff twin: kept
            pair("c.feature", 1, 60, "k4")]
    creates = [["n:1"], ["a:1"], ["b:1"], ["c:1"]]
    cfg = {**CFG, "dev_total": 0}
    res = select(rows, labels_for(rows, creates), cfg, catalog)
    assert res["test_answer_twins"] == [("r", "twin:1")] and ("r", "twin:1") in set(res["answer_fns"])
    assert lines(res["aside"]["uses_test_answer"]) == [("a.feature", 1)]
    assert lines(res["parts"]["train"]) == [("b.feature", 1), ("c.feature", 1)]
    assert select(rows, labels_for(rows, creates), cfg)["test_answer_twins"] == []   # no catalog, no twins

def test_twin_dating_uses_the_earliest_dated_annotation_and_catches_renamed_copies():
    body = "{ $this->page->filterByCode($p); $this->page->filter(); $this->page->waitUntilLoaded(); }"
    cfn = lambda name, *times: {"first_line": 1, "annotations": [{"line": 1 + i, "time": t} for i, t in enumerate(times)],
                                "code": "\n".join(["#[When('p')]"] * len(times) + [f"public function {name}(string $p): void", body])}
    catalog = {"r": {"n:1": cfn("answer", 200),
                     "renamed:1": cfn("copyWithAnotherName", 150),   # post-cutoff copy under another name: joins
                     "late:1": cfn("answer", None, 150),              # undated annotations are ignored: 150 > 100 joins
                     "mixed:1": cfn("answer", 150, 50),               # earliest dated is 50, before the cutoff: kept
                     "undated:1": cfn("answer", None)}}               # no dated annotation: not after the cutoff: kept
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"])]
    res = select(rows, labels_for(rows, [["n:1"]]), {**CFG, "dev_total": 0}, catalog, {"r": "php"})
    assert res["test_answer_twins"] == [("r", "late:1"), ("r", "renamed:1")]

def test_post_cutoff_near_duplicate_of_a_test_answer_joins_it():
    body = ("{ $total = $this->cart->getTotal(); Assert::same($expected, $total->getAmount()); "
            "$this->sharedStorage->set('total', $total); }")
    near_body = body.replace("'total'", "'cart_total'")
    cfn = lambda name, t, b: {"first_line": 1, "code": f"#[Then('p')]\npublic function {name}(int $expected): void\n{b}",
                              "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn("answer", 200, body), "near:1": cfn("close", 150, near_body),
                     "old:1": cfn("closeButOld", 50, near_body),
                     "far:1": cfn("far", 150, "{ $this->client->delete(Resources::LOCALES, $code); $this->client->flush(); }")}}
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"]), pair("a.feature", 1, 160, "k2", fns=["near:1", "s:1"]),
            pair("b.feature", 1, 60, "k3", fns=["old:1", "s:1"])]
    res = select(rows, labels_for(rows, [["n:1"], ["a:1"], ["b:1"]]), {**CFG, "dev_total": 0}, catalog, {"r": "php"})
    (near,) = res["test_answer_near_duplicates"]
    assert near[:3] == ("r", "near:1", "n:1") and 0.9 <= near[3] < 1.0 and res["test_answer_twins"] == []
    assert lines(res["aside"]["uses_test_answer"]) == [("a.feature", 1)] and lines(res["parts"]["train"]) == [("b.feature", 1)]

def test_languages_reach_the_twin_rule_so_ruby_keeps_its_braces():
    tail = "{ name: 'Jo', born: '1970-01-01', ward: 'Renal', consultant: 'Dr Who' }, family: 'Blogs')\nend"
    cfn = lambda head, t: {"first_line": 1, "code": f"Given(/^x$/) do\n  {head} {tail}", "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn("create(:patient, attributes:", 200),
                     "c:1": cfn("build_a_completely_different_record_with_a_long_helper_name(:doctor, :nurse, attributes:", 150)}}
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"])]
    labels = labels_for(rows, [["n:1"]])
    ruby = select(rows, labels, {**CFG, "dev_total": 0}, catalog, {"r": "ruby"})
    assert ruby["test_answer_twins"] == [] and ruby["test_answer_near_duplicates"] == []
    assert select(rows, labels, {**CFG, "dev_total": 0}, catalog, {"r": "php"})["test_answer_twins"] == [("r", "c:1")]

def test_post_cutoff_container_of_a_long_test_answer_joins_it():
    body = "{ " + "; ".join(f"$this->page->step{i}($value)" for i in range(12)) + "; }"   # a long answer
    bigger = body[:-1] + "; ".join(f"$this->other->more{i}($value)" for i in range(14)) + "; }"
    cfn = lambda name, t, b: {"first_line": 1, "code": f"#[When('p')]\npublic function {name}($value): void\n{b}",
                              "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn("answer", 200, body), "big:1": cfn("bigger", 150, bigger), "old:1": cfn("older", 50, bigger)}}
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"]), pair("a.feature", 1, 160, "k2", fns=["big:1", "s:1"]),
            pair("b.feature", 1, 60, "k3", fns=["old:1", "s:1"])]
    res = select(rows, labels_for(rows, [["n:1"], ["a:1"], ["b:1"]]), {**CFG, "dev_total": 0}, catalog, {"r": "php"})
    assert [x[:3] for x in res["test_answer_containers"]] == [("r", "big:1", "n:1")] and res["test_answer_near_duplicates"] == []
    assert lines(res["aside"]["uses_test_answer"]) == [("a.feature", 1)] and lines(res["parts"]["train"]) == [("b.feature", 1)]

def test_answers_include_new_target_fns_and_ratios_are_rounded():
    body = "{ $order = $this->orders->find($id); Assert::same($expected, $order->getTotalPaid()); $this->flush(); }"
    near = body.replace("getTotalPaid", "getTotalPaidTaxIncl")
    cfn = lambda t, b: {"first_line": 1, "code": f"#[Then('p')]\npublic function f($expected): void\n{b}",
                        "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn(200, body), "c:1": cfn(200, "{ $this->somethingCompletelyUnrelated(); $this->andMore(); }"),
                     "copy:1": cfn(150, near)}}
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1", "c:1"])]   # n:1 is a REUSE-type new target, not a CREATE
    res = select(rows, labels_for(rows, [["c:1"]]), {**CFG, "dev_total": 0}, catalog, {"r": "php"})
    (row,) = res["test_answer_near_duplicates"]
    import difflib
    raw = difflib.SequenceMatcher(None, near[2:], body[2:], autojunk=False).ratio()   # nosig_text drops "... {"
    assert row == ("r", "copy:1", "n:1", round(raw, 4)) and raw != round(raw, 4)

def test_copies_count_only_within_their_repo():
    body = "{ $this->page->filterByCode($p); $this->page->filter(); $this->page->waitUntilLoaded(); }"
    cfn = lambda t: {"first_line": 1, "code": f"#[When('p')]\npublic function f(string $p): void\n{body}",
                     "annotations": [{"line": 1, "time": t}]}
    catalog = {"r": {"n:1": cfn(200)}, "q": {"copy:1": cfn(150)}}   # an identical post-cutoff copy, but in another repo
    rows = [pair("t.feature", 1, 200, "k1", new=["n:1"])]
    res = select(rows, labels_for(rows, [["n:1"]]), {**CFG, "dev_total": 0}, catalog, {"r": "php", "q": "php"})
    assert res["test_answer_twins"] == res["test_answer_near_duplicates"] == res["test_answer_containers"] == []
