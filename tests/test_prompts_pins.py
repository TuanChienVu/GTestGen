"""Pinning tests of the prompt-file layer (Phase 3A): build() and held_out_exclusions() (v2tools/prompts.py), the parent
gate and dirty_code() (v2tools/phase3io.py) and the independent acceptance scan (scripts/scan_prompts.py)."""
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.phase3_fixture import T2022, T2024, _git, catalog, make_repo, record
from v2tools import phase3io
from v2tools.compose import FORMATS, KNOWLEDGE_HEAD, Composer, Params
from v2tools.prompts import LIMIT_HELD_OUT, build, control_strings
from v2tools.state import RepoContext
from v2tools.symcache import load as load_symtimes

ROOT = Path(__file__).resolve().parents[1]
REPO, OTHER = "Acme/Shop", "Acme/Other"
STEP = {"iAddToCart": "I add \"x\" to the cart", "iOpenTheShop": "I open the shop"}


def words(text: str) -> int:
    return len(text.split())


def short(msgs: list[dict], gold: str) -> int:
    return 10


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("repo"))


def example(cat: dict, rid: str, fn_name: str = "iAddToCart", t_vis: int = T2022, **kw) -> dict:
    """A train record (feature path from its id) whose one new CREATE function is `fn_name`; older than the test record
    (T2024), so that it is an eligible ③ example for it."""
    fn = next(f for f in cat.values() if f"function {fn_name}(" in f["code"])
    r = record(cat, "train", rid)
    r.update(feature_path=rid.split("|")[1], steps=[STEP[fn_name]], step_kinds=["create"], neighbours=[], t_vis=t_vis,
             create=[{"fn_id": fn["fn_id"], "path": fn["path"], "code": fn["code"], "preexisting": False}],
             create_def_ids=[fn["annotations"][0]["def_id"]])
    r.update(kw)
    return r


def composer(repo, *recs: dict, params: Params = Params(), other: bool = False) -> Composer:
    bare, head = repo
    cat = catalog()
    contexts, cats = {REPO: RepoContext(REPO, "php", bare, head, cat)}, {REPO: cat}
    if other:  # a second repo over the same history, so that a pool may hold ids of two repos
        cat_o = {k: dict(copy.deepcopy(v), repo=OTHER) for k, v in cat.items()}
        contexts[OTHER], cats[OTHER] = RepoContext(OTHER, "php", bare, head, cat_o), cat_o
    return Composer(contexts, cats, {}, {r["id"]: r for r in recs}, params)


def rows(out: Path, name: str) -> list[dict]:
    return [json.loads(l) for l in (out / name).read_text(encoding="utf-8").splitlines()]


def ids_in(out: Path, pattern: str) -> set[str]:
    return {r["record_id"] for f in sorted(out.glob(pattern)) for r in rows(out, f.name)}


def dumps(msgs: list[dict]) -> str:
    return json.dumps(msgs, ensure_ascii=False, sort_keys=True)


# ----- 1. the output directory ---------------------------------------------------------------------------------------
def test_build_refuses_a_non_empty_directory_and_accepts_an_empty_or_missing_one(repo, tmp_path):
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    args = ("v0", ["m1"], ["train", "test"], {"all": [tr["id"]]}, {"m1": words}, {"m1": short}, {"m1": []}, {})
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "m1.all.test.F9.jsonl").write_text("old\n")
    with pytest.raises(FileExistsError):
        build(composer(repo, q, tr), stale, *args)
    assert [p.name for p in stale.iterdir()] == ["m1.all.test.F9.jsonl"]  # refused before writing anything
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build(composer(repo, q, tr), empty, *args) == {"files": 6, "excluded": 0}
    missing = tmp_path / "a" / "b"
    assert build(composer(repo, q, tr), missing, *args) == {"files": 6, "excluded": 0}
    assert (missing / "PROMPTS_SHA256SUMS").exists()


# ----- 2. build() runs the held-out length rule itself ---------------------------------------------------------------
def test_build_applies_the_held_out_length_rule_under_every_tokenizer(repo, tmp_path):
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    common = ("v0", ["m1", "m2"], ["test"], {"all": [tr["id"]]}, {"m1": words, "m2": words})
    at_limit = {"m1": short, "m2": lambda msgs, gold: LIMIT_HELD_OUT}
    build(composer(repo, q, tr), tmp_path / "ok", *common, at_limit, {"m1": [], "m2": []}, {})
    assert rows(tmp_path / "ok", "excluded.jsonl") == []  # exactly LIMIT_HELD_OUT tokens is allowed
    assert ids_in(tmp_path / "ok", "*.test.*.jsonl") == {q["id"]}
    too_long = {"m1": short, "m2": lambda msgs, gold: LIMIT_HELD_OUT + 1}
    res = build(composer(repo, q, tr), tmp_path / "long", *common, too_long, {"m1": [], "m2": []}, {})
    assert rows(tmp_path / "long", "excluded.jsonl") == [
        {"record_id": q["id"], "model": None, "fmt": None, "reason": "too_long_union"}] and res["excluded"] == 1
    for m in ("m1", "m2"):  # too long for m2 only: gone from m1's files as well, in every format
        for fmt in FORMATS:
            assert (tmp_path / "long" / f"{m}.all.test.{fmt}.jsonl").read_text() == ""


def test_build_measures_held_out_prompts_on_the_clean_pool(repo, tmp_path):
    """A pool record with a control token of m2 leaves the pool (clean_pool); had the rule measured the raw pool, the
    test record's F2 would hold it as ③ example and the record would be dropped for a token it is never shown."""
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    tok = example(cat, "Acme/Shop|features/tok.feature|3|s",
                  scenario="Scenario: Total <PRE>\n  Then I should see the total \"5\"")
    comp = composer(repo, q, tr, tok)
    raw = comp.compose_prompt(q, "F2", words, [tr["id"], tok["id"]])[0][0]["content"]
    assert "features/tok.feature" in raw and "<PRE>" in raw  # the raw pool would show it (non-vacuous)
    build(comp, tmp_path, "v0", ["m1", "m2"], ["test"], {"all": [tr["id"], tok["id"]]}, {"m1": words, "m2": words},
          {"m1": short, "m2": short}, {"m1": [], "m2": ["\u2581<PRE>"]}, {})
    assert rows(tmp_path, "excluded.jsonl") == []
    assert json.loads((tmp_path / "pools.json").read_text())["all"]["ids"] == [tr["id"]]
    for m in ("m1", "m2"):
        (f2,) = rows(tmp_path, f"{m}.all.test.F2.jsonl")
        assert f2["provenance"]["examples"] == [tr["id"]] and "<PRE>" not in f2["messages"][0]["content"]


# ----- 3. prompt measured = prompt written ---------------------------------------------------------------------------
def test_the_measured_f2_prompt_is_the_written_one_with_two_repos_two_counters_and_two_pools(repo, tmp_path):
    cat = catalog()
    q = record(cat)
    e1 = example(cat, "Acme/Shop|features/e1.feature|3|s", scenario="Scenario: Totals\n  Then the total is shown",
                 steps=["the total is shown"])
    e2 = example(cat, "Acme/Shop|features/e2.feature|3|s", scenario="Scenario: Seeing\n  Then I see it", steps=["I see it"])
    others = [example(cat, f"Acme/Other|features/o{i}.feature|3|s", repo=OTHER,
                      scenario="Scenario: Seen\n  Then I see I see it", steps=["I see it"]) for i in range(3)]
    comp = composer(repo, q, e1, e2, *others, other=True)
    pools = {"all": sorted([e1["id"], e2["id"], *(o["id"] for o in others)]), "lc600": [e2["id"], others[0]["id"]]}
    # m2's tokenizer cannot afford any text holding "indexPage": its ④ differs from m1's
    counters = {"m1": words, "m2": lambda t: 10 ** 6 if "indexPage" in t else words(t)}
    seen: dict[str, list] = {"m1": [], "m2": []}

    def measure(model):
        def f(msgs, gold):
            seen[model].append((dumps(msgs), gold))
            return 10
        return f
    build(comp, tmp_path, "v0", ["m1", "m2"], ["test"], pools, counters, {m: measure(m) for m in seen},
          {"m1": [], "m2": []}, {})
    written = {m: sorted(dumps(r["messages"]) for p in pools for r in rows(tmp_path, f"{m}.{p}.test.F2.jsonl")) for m in seen}
    assert {m: sorted(x for x, _ in v) for m, v in seen.items()} == written
    assert all(g == comp.gold(q) for v in seen.values() for _, g in v)
    assert len({x for v in written.values() for x in v}) == 4  # every (model, pool) prompt differs: non-vacuous
    (f2,) = rows(tmp_path, "m1.all.test.F2.jsonl")
    # BM25 over this repo's records alone ranks e2 first; with the Other repo's "I see" records in the statistics, e1
    assert f2["provenance"]["examples"] == [e2["id"], e1["id"]]


# ----- 4. control tokens ---------------------------------------------------------------------------------------------
def test_a_token_made_only_of_word_markers_excludes_nothing(repo, tmp_path):
    assert control_strings(["\u2581<PRE>", "\u2581", "\u2581\u2581", "<s>"]) == ["<PRE>", "<s>"]
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    res = build(composer(repo, q, tr), tmp_path, "v0", ["m1"], ["train", "test"], {"all": [tr["id"]]}, {"m1": words},
                {"m1": short}, {"m1": ["\u2581", "\u2581\u2581"]}, {})
    assert res["excluded"] == 0 and rows(tmp_path, "excluded.jsonl") == []
    assert ids_in(tmp_path, "*.train.*.jsonl") == {tr["id"]} and ids_in(tmp_path, "*.test.*.jsonl") == {q["id"]}
    assert json.loads((tmp_path / "pools.json").read_text())["all"]["ids"] == [tr["id"]]


def test_held_out_records_use_the_union_of_every_models_control_tokens(repo, tmp_path):
    """m2's token "indexPage" is in m1's F2 (④) but not in m2's (its tokenizer cannot afford it): a per-model check
    would keep the record; the union drops it for every model."""
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    counters = {"m1": words, "m2": lambda t: 10 ** 6 if "indexPage" in t else words(t)}
    common = ("v0", ["m1", "m2"], ["test"], {"all": [tr["id"]]}, counters, {"m1": short, "m2": short})
    build(composer(repo, q, tr), tmp_path / "ok", *common, {"m1": [], "m2": ["notInAnyPrompt"]}, {})
    assert "indexPage" in rows(tmp_path / "ok", "m1.all.test.F2.jsonl")[0]["messages"][0]["content"]
    assert "indexPage" not in rows(tmp_path / "ok", "m2.all.test.F2.jsonl")[0]["messages"][0]["content"]
    build(composer(repo, q, tr), tmp_path / "tok", *common, {"m1": [], "m2": ["indexPage"]}, {})
    assert rows(tmp_path / "tok", "excluded.jsonl") == [
        {"record_id": q["id"], "model": None, "fmt": None, "reason": "control_token"}]
    assert ids_in(tmp_path / "tok", "*.test.*.jsonl") == set()


def test_a_train_record_with_one_models_token_is_excluded_for_that_model_only(repo, tmp_path):
    cat = catalog()
    tr = example(cat, "Acme/Shop|features/cart.feature|3|s")
    build(composer(repo, record(cat), tr), tmp_path, "v0", ["m1", "m2"], ["train"], {"all": [tr["id"]]},
          {"m1": words, "m2": words}, {"m1": short, "m2": short}, {"m1": ["iAddToCart"], "m2": []}, {})
    assert rows(tmp_path, "excluded.jsonl") == [
        {"record_id": tr["id"], "model": "m1", "fmt": f, "reason": "control_token"} for f in FORMATS]
    assert ids_in(tmp_path, "m1.*.jsonl") == set()
    for fmt in FORMATS:
        assert [r["record_id"] for r in rows(tmp_path, f"m2.all.train.{fmt}.jsonl")] == [tr["id"]]


def test_a_control_token_exclusion_is_listed_once_under_two_pools(repo, tmp_path):
    cat = catalog()
    tr = example(cat, "Acme/Shop|features/cart.feature|3|s")
    build(composer(repo, record(cat), tr), tmp_path, "v0", ["m1"], ["train"], {"all": [tr["id"]], "lc600": [tr["id"]]},
          {"m1": words}, {"m1": short}, {"m1": ["iAddToCart"]}, {})
    ex = rows(tmp_path, "excluded.jsonl")
    assert sorted((e["record_id"], e["model"], e["fmt"]) for e in ex) == [(tr["id"], "m1", f) for f in FORMATS]


# ----- 5. users of test-answer infrastructure ------------------------------------------------------------------------
def test_infrastructure_users_have_no_train_rows_and_leave_every_pool(repo, tmp_path):
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    user = example(cat, "Acme/Shop|features/infra.feature|3|s")
    user["create"] = [dict(user["create"][0], code="#[When('x')]\npublic function x(): void\n{\n    $this->checkout->go();\n}")]
    comp = composer(repo, q, tr, user)
    assert comp.infra()["users"] == {user["id"]}
    both = sorted([tr["id"], user["id"]])
    build(comp, tmp_path, "v0", ["m1"], ["train", "test"], {"all": both, "lc600": both}, {"m1": words}, {"m1": short},
          {"m1": []}, {})
    assert ids_in(tmp_path, "*.train.*.jsonl") == {tr["id"]}
    for p in ("all", "lc600"):
        for fmt in FORMATS:
            assert [r["record_id"] for r in rows(tmp_path, f"m1.{p}.train.{fmt}.jsonl")] == [tr["id"]]
    assert rows(tmp_path, "excluded.jsonl") == [
        {"record_id": user["id"], "model": None, "fmt": None, "reason": "uses_test_infra"}]
    assert {k: v["ids"] for k, v in json.loads((tmp_path / "pools.json").read_text()).items()} == {
        "all": [tr["id"]], "lc600": [tr["id"]]}


# ----- 6. the parent gate --------------------------------------------------------------------------------------------
def gate(tmp: Path, **unresolved: int) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "PARENTS.json").write_text(json.dumps({repo: {"resolved": 3, "external": 1, "unresolved": n, "unresolved_list": []}
                                                  for repo, n in unresolved.items()}))
    return tmp / "symtimes.jsonl"


def test_the_parent_gate_needs_parents_json_with_zero_unresolved_and_load_checks_it_first(tmp_path):
    assert phase3io.parent_gate_ok(gate(tmp_path / "ok", A=0, B=0)) is None
    with pytest.raises(RuntimeError, match=r"unresolved parents: \{'B': 2\}"):
        phase3io.parent_gate_ok(gate(tmp_path / "bad", A=0, B=2))
    with pytest.raises(FileNotFoundError, match="PARENTS.json"):
        phase3io.parent_gate_ok(tmp_path / "none" / "symtimes.jsonl")
    with pytest.raises(RuntimeError, match=r"no parent gate for \['C'\]"):  # a composed repo the gate never saw
        phase3io.parent_gate_ok(gate(tmp_path / "ok2", A=0, B=0), {"A", "C"})
    # load() stops at the gate before it reads any other input (none of them exists here)
    with pytest.raises(RuntimeError, match="unresolved parents"):
        phase3io.load(tmp_path / "sources.json", tmp_path / "phase2", tmp_path / "bad" / "symtimes.jsonl", tmp_path)
    with pytest.raises(FileNotFoundError, match="PARENTS.json"):
        phase3io.load(tmp_path / "sources.json", tmp_path / "phase2", tmp_path / "none" / "symtimes.jsonl", tmp_path)


# ----- 7. dirty_code ---------------------------------------------------------------------------------------------------
def test_dirty_code_reports_code_and_config_changes_only(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for p in ("v2tools/a.py", "scripts/b.py", "config/c.json", "notes.md", "out/o.txt"):
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c")
    assert phase3io.dirty_code(root) == ""
    (root / "notes.md").write_text("y\n")
    (root / "notes2.md").write_text("new\n")
    (root / "out" / "o.txt").write_text("y\n")
    (root / "out" / "new").mkdir()
    (root / "out" / "new" / "p.jsonl").write_text("{}\n")
    assert phase3io.dirty_code(root) == ""  # outside v2tools/, scripts/, config/
    (root / "v2tools" / "sub").mkdir()
    (root / "v2tools" / "sub" / "new.py").write_text("x\n")
    assert phase3io.dirty_code(root).split() == ["??", "v2tools/sub/new.py"]  # untracked, named file by file
    (root / "v2tools" / "sub" / "new.py").unlink()
    (root / "scripts" / "b.py").write_text("y\n")
    assert phase3io.dirty_code(root).split() == ["M", "scripts/b.py"]
    (root / "scripts" / "b.py").write_text("x\n")
    (root / "config" / "c.json").write_text("y\n")
    _git(root, "add", "config/c.json")
    assert phase3io.dirty_code(root).split() == ["M", "config/c.json"]  # staged
    _git(root, "reset", "-q", "config/c.json")
    (root / "config" / "c.json").write_text("x\n")
    (root / "v2gpu").mkdir()
    (root / "v2gpu" / "serialize.py").write_text("x\n")
    assert phase3io.dirty_code(root).split() == ["??", "v2gpu/serialize.py"]  # build_prompts imports it


def test_pools_hold_one_pool_per_learning_curve_subset():
    recs = {"a": {"partition": "train", "subsets": ["lc600", "lc1200"]}, "b": {"partition": "train", "subsets": ["lc1200"]},
            "c": {"partition": "train", "subsets": []}, "d": {"partition": "dev", "subsets": []}}
    assert phase3io.pools(recs) == {"all": ["a", "b", "c"], "lc600": ["a"], "lc1200": ["a", "b"]}


def test_a_learning_curve_pool_writes_only_its_subset_of_train_records(repo, tmp_path):
    cat = catalog()
    q = record(cat)
    a = example(cat, "Acme/Shop|features/a.feature|3|s")
    b = example(cat, "Acme/Shop|features/b.feature|3|s")
    build(composer(repo, q, a, b), tmp_path / "p", "v0", ["m1"], ["train"], {"all": [a["id"], b["id"]], "lc600": [a["id"]]},
          {"m1": words}, {"m1": short}, {"m1": []}, {})
    assert ids_in(tmp_path / "p", "m1.all.train.*.jsonl") == {a["id"], b["id"]}
    assert ids_in(tmp_path / "p", "m1.lc600.train.*.jsonl") == {a["id"]}


# ----- 8.–9. scan_prompts --------------------------------------------------------------------------------------------
LONG_TAIL = ("    $this->checkout->assertCurrencyShownNextToTheGrandTotal('EUR', 'with two decimals after the separator',"
             " 'and a thin space before the currency symbol', 'on the order summary page of the checkout');\n")


def scan_world(repo) -> tuple[Composer, dict]:
    """The test record's new answer is longer than 200 normalised characters, so that its last 200 hold neither its
    method name nor its pattern; one pool record mentions both answer names and is dropped (rule "name") from the test
    prompts and from tr2's."""
    cat = catalog()
    q = record(cat)
    q["create"] = [dict(q["create"][0], code=q["create"][0]["code"].replace("\n}", "\n" + LONG_TAIL + "}"))]
    tr = example(cat, "Acme/Shop|features/cart.feature|3|s")
    named = example(cat, "Acme/Shop|features/named.feature|3|s", background="Background:\n  # iShouldSeeTotal iOpenTheShop")
    tr2 = example(cat, "Acme/Shop|features/open.feature|3|s", fn_name="iOpenTheShop", t_vis=T2024)
    comp = composer(repo, q, tr, named, tr2)
    return comp, comp.records


@pytest.fixture(scope="module")
def honest(repo, tmp_path_factory):
    comp, recs = scan_world(repo)
    out = tmp_path_factory.mktemp("honest") / "p"
    pool = sorted(i for i, r in recs.items() if r["partition"] == "train")
    build(comp, out, "v0", ["m"], ["train", "test"], {"all": pool}, {"m": words}, {"m": short}, {"m": []}, {})
    return out, recs, {REPO: comp.catalog[REPO]}


@pytest.fixture(scope="module")
def scanner():
    return script("scan_prompts")


def copy_of(honest, tmp_path: Path) -> Path:
    shutil.copytree(honest[0], tmp_path / "p")
    return tmp_path / "p"


def reseal(out: Path) -> None:
    names = sorted(p.name for p in out.iterdir() if p.name != "PROMPTS_SHA256SUMS")
    (out / "PROMPTS_SHA256SUMS").write_text("".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in names))


def edit_first_row(out: Path, name: str, fn) -> None:
    """Change the first row of one prompt file and reseal, so that only the planted finding remains."""
    f = out / name
    lines = f.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    fn(row)
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reseal(out)


def into_knowledge(leak: str):
    def fn(row):
        text = row["messages"][0]["content"]
        assert KNOWLEDGE_HEAD in text
        row["messages"][0]["content"] = text.replace(KNOWLEDGE_HEAD, KNOWLEDGE_HEAD + leak + "\n", 1)
    return fn


def answer_tail(honest) -> str:
    code = honest[1][record(catalog())["id"]]["create"][0]["code"]
    tail = " ".join(code.split())[-200:]
    assert "iShouldSeeTotal" not in tail and "see the total" not in tail
    return tail


def test_scan_accepts_the_honest_files(honest, scanner):
    out, recs, cat = honest
    assert scanner.scan(out, recs, cat) == {}
    assert rows(out, "m.all.test.F2.jsonl")[0]["provenance"]["dropped"]  # the honest files do carry drops


@pytest.mark.parametrize("leak, expected", [
    ("$s = 'I  SHOULD see the total :total';", {"create_pattern_quoted": 1}),
    ('$s = "I should see the total :total";', {"create_pattern_quoted": 1}),
    ("$re = '#' . /I should see the total :total/;", {"create_pattern_quoted": 1}),
    ("     * @Then I should see the total :total", {"create_pattern_docblock": 1}),
    ("    #[Then('I should see the total :total')]", {"create_annotation_line": 1, "create_pattern_quoted": 1}),
    (None, {"short_answer_slice": 1}),
    ("$this->iShouldSeeTotal('5');", {"answer_method_name": 1}),
], ids=["quoted_single", "quoted_double", "quoted_slash", "docblock", "annotation_line", "short_answer_slice",
        "answer_method_name"])
def test_scan_catches_each_answer_leak_planted_in_block_4(honest, scanner, tmp_path, leak, expected):
    out = copy_of(honest, tmp_path)
    edit_first_row(out, "m.all.test.F2.jsonl", into_knowledge(answer_tail(honest) if leak is None else leak))
    assert scanner.scan(out, honest[1], honest[2]) == expected


def test_scan_reads_only_blocks_3_and_4(honest, scanner, tmp_path):
    out = copy_of(honest, tmp_path)
    leak = "$this->iShouldSeeTotal('5'); 'I should see the total :total'"

    def into_scenario(row):
        row["messages"][0]["content"] = row["messages"][0]["content"].replace("Undefined steps:\n", leak + "\nUndefined steps:\n", 1)
    edit_first_row(out, "m.all.test.F0.jsonl", into_scenario)  # F0 has neither ③ nor ④
    edit_first_row(out, "m.all.test.F2.jsonl", into_scenario)  # ② of F2, before ③
    assert scanner.scan(out, honest[1], honest[2]) == {}
    near = "$iShouldSeeTotal = 1; iShouldSeeTotalX(); 'I should see the total :total!'"
    edit_first_row(out, "m.all.test.F2.jsonl", into_knowledge(near))  # not the identifier, not the quoted pattern
    assert scanner.scan(out, honest[1], honest[2]) == {}


def test_scan_counts_a_file_outside_the_sums_and_a_changed_file(honest, scanner, tmp_path):
    out = copy_of(honest, tmp_path / "extra")
    (out / "notes.txt").write_text("x\n")
    assert scanner.scan(out, honest[1], honest[2]) == {"sums_file_set": 1}
    out = copy_of(honest, tmp_path / "gone")
    (out / "infra.json").unlink()
    assert scanner.scan(out, honest[1], honest[2]) == {"sums_file_set": 1}
    out = copy_of(honest, tmp_path / "changed")
    (out / "infra.json").write_text((out / "infra.json").read_text() + "\n")
    assert scanner.scan(out, honest[1], honest[2]) == {"sums_mismatch": 1}


def test_scan_counts_rows_that_are_not_one_user_turn_or_carry_a_held_out_answer(honest, scanner, tmp_path):
    out = copy_of(honest, tmp_path / "turns")
    edit_first_row(out, "m.all.train.F1.jsonl", lambda row: row["messages"].append({"role": "assistant", "content": "ok"}))
    assert scanner.scan(out, honest[1], honest[2]) == {"not_one_user_turn": 1}
    out = copy_of(honest, tmp_path / "answer")
    edit_first_row(out, "m.all.test.F0.jsonl", lambda row: row.update(answer=""))  # "" is an answer too: null only
    edit_first_row(out, "m.all.test.F2.jsonl", lambda row: row.update(answer="leak"))
    assert scanner.scan(out, honest[1], honest[2]) == {"held_out_answer": 2}


def test_scan_counts_drops_by_partition_kind_and_rule_and_main_writes_the_report(honest, scanner, tmp_path):
    out, recs, cat = honest
    drops: Counter = Counter()
    assert scanner.scan(out, recs, cat, drops) == {}
    direct: Counter = Counter()
    for f in sorted(out.glob("m.*.jsonl")):
        for r in rows(out, f.name):
            direct.update((recs[r["record_id"]]["partition"], d["kind"], d["rule"]) for d in r["provenance"]["dropped"])
    assert drops == direct == Counter({("test", "example", "name"): 2, ("train", "example", "name"): 2})
    p2 = tmp_path / "phase2"
    (p2 / "records").mkdir(parents=True)
    for part in phase3io.PARTS:
        (p2 / "records" / f"{part}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs.values() if r["partition"] == part))
    (p2 / "catalog.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cat[REPO].values()))
    report = tmp_path / "drops.json"
    # run from elsewhere with a bare environment: the script needs no v2tools on the path
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "scan_prompts.py"), "--prompts", str(out),
                          "--phase2", str(p2), "--report", str(report)], cwd=tmp_path, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"})
    assert (res.returncode, res.stdout, res.stderr) == (0, '{"findings": 0}\n', "")
    assert json.loads(report.read_text()) == {"test": {"example": {"name": 2}}, "train": {"example": {"name": 2}}}


# ----- 10. META and SUMS ---------------------------------------------------------------------------------------------
def test_meta_holds_the_contract_keys_compose_and_files_and_the_sums_cover_every_other_file(repo, tmp_path):
    cat = catalog()
    q, tr = record(cat), example(cat, "Acme/Shop|features/cart.feature|3|s")
    params = Params(k=5, n_examples=2, ratio=0.25, budget_examples=1000, budget_knowledge=900, mode="4b")
    contract = {"v2tools_commit": "c0ffee", "phase2_sums_sha256": "p" * 64, "symtimes_sha256": "s" * 64,
                "special_tokens_sha256": "t" * 64, "tokenizer_revisions": {"m1": "rev1", "m2": "rev2"}}
    build(composer(repo, q, tr, params=params), tmp_path, "v7", ["m1", "m2"], ["train", "test"],
          {"all": [tr["id"]], "lc600": [tr["id"]]}, {"m1": words, "m2": words}, {"m1": short, "m2": short},
          {"m1": [], "m2": []}, contract)
    prompt_files = sorted(f"{m}.{p}.{part}.{fmt}.jsonl" for m in ("m1", "m2") for p in ("all", "lc600")
                          for part in ("train", "test") for fmt in FORMATS)
    assert sorted(p.name for p in tmp_path.glob("m*.jsonl")) == prompt_files
    meta = json.loads((tmp_path / "PROMPTS_META.json").read_text())
    assert meta == {**contract, "compose": {"variant": "v7", **asdict(params)}, "files": prompt_files}
    assert meta["compose"] == {"variant": "v7", "k": 5, "n_examples": 2, "ratio": 0.25, "budget_examples": 1000,
                               "budget_knowledge": 900, "mode": "4b"}
    lines = (tmp_path / "PROMPTS_SHA256SUMS").read_text().splitlines()
    sums = dict(reversed(l.split("  ", 1)) for l in lines)
    assert len(sums) == len(lines)
    assert set(sums) == {p.name for p in tmp_path.iterdir()} - {"PROMPTS_SHA256SUMS"}
    assert {"PROMPTS_META.json", "excluded.jsonl", "infra.json", "pools.json"} <= set(sums)
    assert all(hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() == h for n, h in sums.items())


# ----- 11. Line separators and variant labels ------------------------------------------------------------------------
SEPARATORS = "\u2028\u2029\u0085\x1c\x1d\x1e\x0b\x0c"  # splitlines() breaks at each; json.dumps writes the first 3 raw


def test_the_records_reader_splits_on_newlines_only(tmp_path):
    rec = {"id": "Acme/Shop|features/a.feature|1|s", "partition": "train", "text": "x" + SEPARATORS + "y"}
    (tmp_path / "records").mkdir()
    for part in phase3io.PARTS:
        (tmp_path / "records" / f"{part}.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n" if part == "train" else "", encoding="utf-8")
    assert phase3io.read_records(tmp_path) == {rec["id"]: rec}
    row = {"repo": "Acme/Shop", "fn_id": "f1", "code": "a" + SEPARATORS + "b"}
    (tmp_path / "catalog.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert phase3io.read_catalog(tmp_path) == {"Acme/Shop": {"f1": row}}
    sym = {"repo": "Acme/Shop", "path": "src/A.php", "file": "src/A.php", "names": ["a" + SEPARATORS], "params": {},
           "versions": [], "vnames": [], "vprivate": [], "present": {}}
    (tmp_path / "symtimes.jsonl").write_text(json.dumps(sym, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_symtimes(tmp_path / "symtimes.jsonl")["Acme/Shop"]["src/A.php"]["names"] == ["a" + SEPARATORS]
    f = tmp_path / "rows.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")  # no final newline
    assert phase3io.jsonl_rows(f) == [{"a": 1}, {"a": 2}]
    f.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):  # a blank line inside the file still fails loudly
        phase3io.jsonl_rows(f)


def test_the_scan_reads_prompt_rows_records_and_catalog_whose_strings_hold_line_separators(honest, scanner, tmp_path):
    out = copy_of(honest, tmp_path)
    f = out / "m.all.test.F2.jsonl"
    rows_ = [json.loads(line) for line in f.read_text(encoding="utf-8").split("\n") if line]
    text = rows_[0]["messages"][0]["content"]
    rows_[0]["messages"][0]["content"] = text.replace("Undefined steps:\n", "Undefined steps:" + SEPARATORS + "\n", 1)
    assert SEPARATORS in rows_[0]["messages"][0]["content"]
    f.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows_), encoding="utf-8")
    reseal(out)
    assert scanner.scan(out, honest[1], honest[2]) == {}
    p2 = tmp_path / "phase2"
    (p2 / "records").mkdir(parents=True)
    for part in phase3io.PARTS:
        (p2 / "records" / f"{part}.jsonl").write_text("".join(
            json.dumps({**r, "note": SEPARATORS}, ensure_ascii=False) + "\n" for r in honest[1].values()
            if r["partition"] == part), encoding="utf-8")
    (p2 / "catalog.jsonl").write_text("".join(json.dumps({**row, "note": SEPARATORS}, ensure_ascii=False) + "\n"
                                              for row in honest[2][REPO].values()), encoding="utf-8")
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "scan_prompts.py"), "--prompts", str(out),
                          "--phase2", str(p2)], cwd=tmp_path, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert (res.returncode, res.stdout, res.stderr) == (0, '{"findings": 0}\n', "")


def test_a_variant_other_than_v0_must_name_every_compose_parameter(monkeypatch, tmp_path, capsys):
    bp = script("build_prompts")
    monkeypatch.setattr(bp, "dirty_code", lambda root: pytest.fail("the code check ran before the refusal"))
    assert bp.compose_params("v0", None, None, None, None) == Params()
    assert bp.compose_params("v0", 8, 3, 0.5, "full") == Params()
    with pytest.raises(ValueError, match="v0 means"):
        bp.compose_params("v0", 5, None, None, None)
    with pytest.raises(ValueError, match=r"\['k', 'n_examples', 'ratio', 'mode'\]"):
        bp.compose_params("dev_pilot_v1", None, None, None, None)
    with pytest.raises(ValueError, match=r"\['mode'\]"):
        bp.compose_params("dev_grid_k5_n2_r0.25", 5, 2, 0.25, None)
    assert bp.compose_params("v1", 5, 2, 0.25, "4b") == Params(k=5, n_examples=2, ratio=0.25, mode="4b")
    monkeypatch.setattr(sys, "argv", ["build_prompts.py", "--variant", "v1", "--k", "5", "--out", str(tmp_path / "p")])
    assert bp.main() == 2  # refused before the code check and the tokenizers; nothing is written
    assert "'v1' needs explicit ['n_examples', 'ratio', 'mode']" in capsys.readouterr().err
    assert not (tmp_path / "p").exists()


def test_both_jsonl_readers_decode_bytes_so_a_bare_carriage_return_is_not_a_newline(scanner, tmp_path):
    """read_text() turns a bare "\\r" into "\\n" (universal newlines); v2gpu's jsonl_lines decodes the bytes and splits
    on "\\n" only. Both 3A readers (phase3io.jsonl_rows, scan_prompts.jsonl_rows) must read exactly as v2gpu does."""
    f = tmp_path / "rows.jsonl"
    for read in (phase3io.jsonl_rows, scanner.jsonl_rows):
        f.write_bytes(b'{"a":\r 1}\r\n{"b": 2}\n')  # "\r" is JSON whitespace; CRLF still ends a row
        assert read(f) == [{"a": 1}, {"b": 2}]
        f.write_bytes(b'{"a": 1}\r{"a": 2}\r')  # a bare "\r" does not end a row
        with pytest.raises(json.JSONDecodeError):
            read(f)
        f.write_bytes(b'{"a": 1}\n\n{"a": 2}\n')  # a blank line inside the file still fails loudly
        with pytest.raises(json.JSONDecodeError):
            read(f)


def test_main_hands_the_named_compose_parameters_to_load(monkeypatch, tmp_path):
    """build_prompts.main composes with exactly the parameters it checked. The tokenizers are stubbed, and load() stops
    the run once it has seen the parameters."""
    bp = script("build_prompts")
    seen = {}

    class Seen(Exception):
        pass

    def fake_load(config, phase2, symtimes, full_root, params):
        seen["params"] = params
        raise Seen

    models, serialize = types.ModuleType("v2gpu.models"), types.ModuleType("v2gpu.serialize")
    models.MODELS, models.load_tokenizer = {}, lambda spec: None
    serialize.answer_ids = serialize.prompt_ids = lambda *a: []
    monkeypatch.setitem(sys.modules, "v2gpu", types.ModuleType("v2gpu"))
    monkeypatch.setitem(sys.modules, "v2gpu.models", models)
    monkeypatch.setitem(sys.modules, "v2gpu.serialize", serialize)
    monkeypatch.setattr(bp, "dirty_code", lambda root: "")
    monkeypatch.setattr(bp, "load", fake_load)
    special = tmp_path / "special_tokens.json"
    special.write_text('{"models": {}}')
    monkeypatch.setattr(sys, "argv", ["build_prompts.py", "--variant", "dev_grid_k5_n2_r0.25", "--k", "5", "--n-examples",
                                      "2", "--ratio", "0.25", "--mode", "4b", "--special-tokens", str(special),
                                      "--out", str(tmp_path / "p")])
    with pytest.raises(Seen):
        bp.main()
    assert seen["params"] == Params(k=5, n_examples=2, ratio=0.25, mode="4b")
