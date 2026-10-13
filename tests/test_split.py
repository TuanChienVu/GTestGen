import json
import subprocess
import sys
from pathlib import Path

from v2tools.split import assign

def row(line, path, t, new=(), key=None, status="BOUND", suite="default", amb=0, old=("s.php:1",), repo="r"):
    target_ids, new_target_ids = sorted({*old, *new}), sorted(new)
    return {"repo": repo, "feature_path": path, "scenario_line": line, "suite": suite, "status": status,
            "ambiguous_steps": amb, "structure_key": key or f"k{line}", "scenario_min_time": t,
            "target_ids": target_ids, "new_target_ids": new_target_ids, "targets_new_count": len(new),
            "target_fns": target_ids, "new_target_fns": new_target_ids}

def lines(pairs):
    return sorted({p["scenario_line"] for p in pairs})

def test_rules_in_order():
    rows = [row(1, "a.feature", 200, new=["n.php:1"]),                 # test
            row(2, "a.feature", 50),                                   # same file as a test item
            row(3, "b.feature", 50), row(4, "b.feature", 50, key="k1"),  # 4 repeats the structure of 1
            row(5, "c.feature", 200),                                  # new scenario, old targets -> not test
            row(6, "d.feature", 50, status="PARTIAL"),                 # not bound -> ignored
            row(7, "e.feature", 50, old=["n.php:1"])]                  # reuses the new target of test item 1
    r = assign(rows, 100, dev_target=1)
    assert lines(r["test"]) == [1] and lines(r["dev"] + r["train"]) == [3, 5]
    assert {k: lines(v) for k, v in r["set_aside"].items() if v} == {
        "same_file_as_test": [2], "duplicate_structure": [4], "uses_test_new_target": [7]}
    assert r["verdict"] == "SHORT" and r["shortfall"]["train"] == 600 - r["counts"]["train"]["unique_scenarios"]

def test_pairs_of_a_scenario_stay_together_and_ambiguous_pairs_are_set_aside():
    rows = [row(1, "a.feature", 50, suite="ui"), row(1, "a.feature", 50, suite="api"),
            row(2, "t.feature", 200, suite="ui", new=["n.php:1"]), row(2, "t.feature", 200, suite="api"),
            row(3, "m.feature", 50, suite="ui"), row(3, "m.feature", 50, suite="api", amb=2),
            row(4, "z.feature", 50, suite="api", amb=1)]
    r = assign(rows, 100, dev_target=0)
    assert [(p["scenario_line"], p["suite"]) for p in r["test"]] == [(2, "ui")]
    assert sorted((p["scenario_line"], p["suite"]) for p in r["train"]) == [(1, "api"), (1, "ui"), (3, "ui")]
    assert r["counts"]["train"] == {"unique_scenarios": 2, "pairs": 3}
    assert sorted((p["scenario_line"], p["suite"]) for p in r["set_aside"]["ambiguous"]) == [(3, "api"), (4, "api")]
    assert [(p["scenario_line"], p["suite"]) for p in r["set_aside"]["test_sibling_old_targets"]] == [(2, "api")]

def test_duplicate_structures_count_once():
    r = assign([row(i, f"f{i}.feature", 50, key="same") for i in range(5)], 100, dev_target=0)
    assert r["counts"]["train"]["unique_scenarios"] == 1 and len(r["set_aside"]["duplicate_structure"]) == 4

def test_identity_includes_repo():
    rows = [row(1, "features/a.feature", 200, new=["n.php:1"], repo="r1"),
            row(2, "features/a.feature", 50, repo="r2"), row(3, "features/b.feature", 50, old=["n.php:1"], repo="r2")]
    r = assign(rows, 100, dev_target=0)
    assert lines(r["test"]) == [1] and lines(r["train"]) == [2, 3] and not any(r["set_aside"].values())

def test_verdict_counts_unique_scenarios_not_pairs():
    def rows(n_rest):
        test = [row(i, f"t{i}.feature", 200, new=[f"n{i}.php:1"]) for i in range(150)]
        rest = [row(1000 + i, f"f{i}.feature", 50, suite=s) for i in range(n_rest) for s in ("ui", "api")]
        return test + rest
    go = assign(rows(750), 100)
    assert go["verdict"] == "GO" and go["counts"]["dev"]["unique_scenarios"] == 150
    short = assign(rows(749), 100)
    assert short["counts"]["train"] == {"unique_scenarios": 599, "pairs": 1198}
    assert short["verdict"] == "SHORT" and short["shortfall"] == {"train": 1, "dev": 0, "test": 0}

def test_cli_writes_report_and_members(tmp_path: Path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in [
        row(1, "a.feature", 2_000_000_000, new=["n.php:1"]), row(2, "b.feature", 50, suite="ui"),
        row(2, "b.feature", 50, suite="api", amb=1), row(3, "c.feature", 50, status="NO_SUITE", suite=None)]))
    out = tmp_path / "split.json"
    subprocess.run([sys.executable, "-m", "v2tools.split", "--pairs", str(pairs), "--out", str(out)],
                   check=True, cwd=Path(__file__).resolve().parents[1], capture_output=True)
    report = json.loads(out.read_text())
    assert report["counts"]["test"] == {"unique_scenarios": 1, "pairs": 1}
    assert report["input_status"] == {"BOUND": 3, "NO_SUITE": 1}
    assert {k: v["pairs"] for k, v in report["set_aside"].items()} == {
        "ambiguous": 1, "duplicate_structure": 0, "test_sibling_old_targets": 0, "same_file_as_test": 0, "uses_test_new_target": 0}
    members = [json.loads(l) for l in (tmp_path / "split_members.jsonl").read_text().splitlines()]
    assert [(m["scenario_line"], m["suite"], m["partition"]) for m in members] == [
        (1, "default", "test"), (2, "api", "set_aside:ambiguous"), (2, "ui", "dev")]

def test_uses_test_new_target_rule_compares_functions_not_annotation_lines():
    # a.php:10 and a.php:11 are two different annotations on the SAME function a.php:12 (887 such functions
    # were found across the real repos). The train pair's target_ids id (a.php:11) never appears in
    # any new_target_ids, so an id-only comparison would miss the leak; comparing target_fns/new_target_fns
    # catches it because both annotations collapse to the same function a.php:12.
    test_pair = {**row(1, "t.feature", 200, new=["a.php:10"]), "new_target_fns": ["a.php:12"]}
    train_pair = {**row(2, "b.feature", 50, old=["a.php:11"]), "target_fns": ["a.php:12"]}
    r = assign([test_pair, train_pair], 100, dev_target=1)
    assert lines(r["test"]) == [1]
    assert lines(r["set_aside"]["uses_test_new_target"]) == [2]
    assert not r["train"] and not r["dev"]

def test_check_split_catches_a_function_level_leak_hand_edited_into_split_members(tmp_path: Path):
    # Same function-sharing setup as above, at the pairs.jsonl / split_members.jsonl level check_split.py
    # actually reads. split.py (fixed) correctly sets the leaking pair aside; hand-edit split_members.jsonl to
    # simulate a corrupted/buggy split that let it slip into train, and confirm check_split.py's independent
    # re-check (which must not import v2tools.split) still flags it via target_fns, not target_ids.
    pairs_path = tmp_path / "pairs.jsonl"
    test_pair = {**row(1, "t.feature", 2_000_000_000, new=["a.php:10"]), "new_target_fns": ["a.php:12"]}
    leak_pair = {**row(2, "b.feature", 50, old=["a.php:11"]), "target_fns": ["a.php:12"]}
    pairs_path.write_text("".join(json.dumps(r) + "\n" for r in [test_pair, leak_pair]))
    split_out = tmp_path / "split.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "v2tools.split", "--pairs", str(pairs_path), "--out", str(split_out)],
                   check=True, cwd=root, capture_output=True)
    members_path = tmp_path / "split_members.jsonl"
    members = [json.loads(l) for l in members_path.read_text().splitlines()]
    assert next(m for m in members if m["scenario_line"] == 2)["partition"] == "set_aside:uses_test_new_target"
    members = [{**m, "partition": "train"} if m["scenario_line"] == 2 else m for m in members]
    members_path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in members))
    result = subprocess.run([sys.executable, "scripts/check_split.py", "--pairs", str(pairs_path),
                             "--split", str(split_out)], cwd=root, capture_output=True, text=True)
    checks = json.loads(result.stdout)
    assert checks["train_dev_target_leaks"] == 1
    assert result.returncode == 1
