"""The leak and exclusion rules of the prompt composer that test_compose.py does not pin: temporal order of examples,
direct neighbours, hidden answers inside examples, twins in ③ and ④, the placebo size, the gold of a preexisting item
and the strict Java member references."""
import copy

import pytest

from tests.phase3_fixture import SHOP, T2020, T2022, T2024, catalog, make_repo, record
from v2tools.compose import Composer, Params, gold_answer
from v2tools.state import RepoContext, member_refs, render_answer


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    bare, head = make_repo(tmp_path_factory.mktemp("repo"))
    return bare, head


def build(base, extra_train=(), forbidden=None, params=Params(budget_examples=400, budget_knowledge=400)):
    bare, head = base
    cat = catalog()
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    q = record(cat)
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    recs = {q["id"]: q}
    for rid, t, feature, kw in extra_train:
        e = record(cat, "train", rid)
        e.update(feature_path=feature, steps=["I add \"x\" to the cart"], step_kinds=["create"], t_vis=t, scenario_time=t,
                 neighbours=[])
        e.pop("create")
        e.pop("create_def_ids")
        e.update(kw)
        e.setdefault("create", [{"fn_id": add["fn_id"], "path": add["path"], "code": add["code"], "preexisting": False}])
        e.setdefault("create_def_ids", [add["annotations"][0]["def_id"]])
        recs[rid] = e
    return Composer({"Acme/Shop": ctx}, {"Acme/Shop": cat}, forbidden or {}, recs, params), q


def pool(comp):
    return [i for i, r in comp.records.items() if r["partition"] == "train"]


def test_examples_must_be_older_than_the_query(base):
    comp, q = build(base, [("Acme/Shop|features/a.feature|1|s", T2022, "features/a.feature", {}),
                           ("Acme/Shop|features/b.feature|1|s", T2024, "features/b.feature", {})])
    _, prov = comp.compose_prompt(q, "F1", words, pool(comp))
    assert prov["examples"] == ["Acme/Shop|features/a.feature|1|s"]  # b has t_vis == the query's t_vis: future code


def test_direct_neighbours_are_excluded_for_a_train_query(base):
    comp, _ = build(base, [("Acme/Shop|features/a.feature|1|s", T2020, "features/a.feature", {}),
                           ("Acme/Shop|features/b.feature|1|s", T2022, "features/b.feature",
                            {"neighbours": ["Acme/Shop|features/a.feature|1|s"]})])
    b = comp.records["Acme/Shop|features/b.feature|1|s"]
    b["create"] = [dict(b["create"][0], fn_id="other")]  # no shared CREATE function: only the neighbour rule applies
    comp.catalog["Acme/Shop"]["other"] = dict(copy.deepcopy(comp.catalog["Acme/Shop"][next(iter(comp.catalog["Acme/Shop"]))]),
                                              fn_id="other")
    b["create_def_ids"] = []
    _, prov = comp.compose_prompt(b, "F1", words, pool(comp))
    assert "Acme/Shop|features/a.feature|1|s" not in prov["examples"]


def test_an_example_whose_create_function_is_hidden_is_excluded(base):
    add_id = next(f for f, r in catalog().items() if "iAddToCart" in r["code"])
    comp, _ = build(base, [("Acme/Shop|features/a.feature|1|s", T2020, "features/a.feature", {}),
                           ("Acme/Shop|features/b.feature|1|s", T2024 + 10, "features/b.feature", {})],
                    forbidden={"Acme/Shop": {"test": [add_id], "dev": [], "aux": []}})
    b = comp.records["Acme/Shop|features/b.feature|1|s"]
    b["create"] = [dict(b["create"][0], fn_id="x")]
    comp.catalog["Acme/Shop"]["x"] = dict(copy.deepcopy(comp.catalog["Acme/Shop"][add_id]), fn_id="x", code="x() {}")
    comp.catalog["Acme/Shop"]["x"]["annotations"] = []
    b["create_def_ids"] = []
    _, prov = comp.compose_prompt(b, "F1", words, pool(comp))
    assert prov["examples"] == []  # a's CREATE function is a test answer, hidden from a train query


def test_a_twin_of_the_answer_is_dropped_from_knowledge(base):
    comp, q = build(base)
    cat = comp.catalog["Acme/Shop"]
    ans = next(f for f in cat.values() if "iShouldSeeTotal" in f["code"])
    opened = next(f for f in cat.values() if "iOpenTheShop" in f["code"])
    # the same (long enough) body under two names: body_key_nosig makes them twins
    body = "{\n    $this->indexPage->open(['a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5]);\n}"
    opened["code"] = opened["code"].split("{", 1)[0] + body
    ans["code"] = ans["code"].split("{", 1)[0] + body
    comp._keys.clear()
    comp.contexts["Acme/Shop"].unchanged = lambda fid, t: True  # bodies edited in memory only: rule 5 would show the repo's
    _, prov = comp.compose_prompt(q, "F2", words, pool(comp))
    assert {"kind": "knowledge", "id": opened["fn_id"], "rule": "twin"} in prov["dropped"]


def test_placebo_uses_another_repo_and_no_more_tokens_than_the_real_knowledge(base):
    comp, q = build(base, params=Params(budget_knowledge=400, mode="placebo"))
    other = {f"x{i}": {"fn_id": f"x{i}", "path": f"Other{i}.php", "code": f"public function f{i}() {{ return {i}; }}"}
             for i in range(50)}
    comp.catalog["PrestaShop/PrestaShop"] = other
    import v2tools.compose as c
    c.PLACEBO_REPO["Acme/Shop"] = "PrestaShop/PrestaShop"
    try:
        msgs, prov = comp.compose_prompt(q, "F2", words, pool(comp))
        real = Composer(comp.contexts, comp.catalog, comp.forbidden, comp.records, Params(budget_knowledge=400))
        target = words(real.knowledge(q, real.answer(q), words, [])[0])
        text = msgs[0]["content"].split("Project knowledge:", 1)[1]
        assert "Other" in text and "ShopContext" not in text and prov["knowledge"] == []
        assert words("Project knowledge:" + text) <= target + words("Project knowledge: Existing step definitions: Now write the code for the undefined steps.") + 4
    finally:
        del c.PLACEBO_REPO["Acme/Shop"]


def test_gold_of_a_preexisting_item_keeps_the_annotations_shown_at_t_vis():
    fn = {"first_line": 10, "code": "#[Given('old')]\n#[When('mid')]\n#[When('new')]\n#[When('later')]\npublic function f(): void\n{\n}",
          "annotations": [{"def_id": "d1", "line": 10, "time": 1, "pattern": "old"},
                          {"def_id": "d2", "line": 11, "time": 5, "pattern": "mid"},
                          {"def_id": "d3", "line": 12, "time": 9, "pattern": "new"},
                          {"def_id": "d4", "line": 13, "time": 20, "pattern": "later"}]}
    got = render_answer(fn, 10, frozenset({"d3"}))
    assert "old" in got and "mid" in got and "new" in got and "later" not in got
    r = {"language": "php", "t_vis": 10, "create_def_ids": ["d3"],
         "create": [{"fn_id": "f", "path": "a.php", "code": "PHASE2", "preexisting": True}]}
    assert "#[When('mid')]" in gold_answer(r, {"f": fn}) and "PHASE2" not in gold_answer(r, {"f": fn})
    with pytest.raises(ValueError):
        gold_answer(r)  # without the catalog a preexisting item would get Phase 2 code, the wrong gold


def test_java_member_references_ignore_parameters_and_locals():
    code = ('@Then("x {int}")\npublic void check(int loanId) {\n    String status = helper.get(loanId);\n'
            '    assertEquals(EXPECTED, status);\n    this.lastId = loanId;\n    retry();\n}')
    refs = member_refs("java", code)
    assert {"helper", "EXPECTED", "lastId", "retry", "assertEquals"} <= refs
    assert "loanId" not in refs and "status" not in refs
    lam = 'Then("y", (String name) -> {\n    client.send(name);\n});'
    assert "client" in member_refs("java", lam) and "name" not in member_refs("java", lam)


def test_strict_context_refuses_a_path_missing_from_the_sealed_symtimes(base):
    bare, head = base
    ctx = RepoContext("Acme/Shop", "php", bare, head, catalog(), {}, strict=True)
    with pytest.raises(KeyError):
        ctx.times(SHOP)


def test_a_twin_of_the_answer_is_dropped_from_the_class_outline_too(base):
    comp, q = build(base)
    cat = comp.catalog["Acme/Shop"]
    ans = next(f for f in cat.values() if "iShouldSeeTotal" in f["code"])
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    body = "{\n    $this->indexPage->open(['a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5]);\n}"
    add["code"] = add["code"].split("{", 1)[0] + body
    ans["code"] = ans["code"].split("{", 1)[0] + body
    comp._keys.clear()
    msgs, prov = comp.compose_prompt(q, "F2", words, pool(comp))
    assert any(d["kind"] == "frame_line" and d["rule"] == "twin" for d in prov["dropped"])
    assert add["fn_id"] not in prov["skeleton"] and "iAddToCart" not in msgs[0]["content"]


def test_the_records_own_preexisting_function_is_dropped_when_it_holds_new_answer_code(base):
    comp, q = build(base)
    cat = comp.catalog["Acme/Shop"]
    ans = next(f for f in cat.values() if "iShouldSeeTotal" in f["code"])
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    body = "{\n" + "".join(f"    $this->cart->add('item-{i:03d}', 'a fairly long argument number {i:03d}');\n" for i in range(8)) + "}"
    add["code"] = add["code"].split("{", 1)[0] + body
    ans["code"] = ans["code"].split("{", 1)[0] + body  # the new answer is a copy of the preexisting function
    comp._keys.clear()
    comp.contexts["Acme/Shop"].unchanged = lambda fid, t: True  # bodies edited in memory only: rule 5 would show the repo's
    r = dict(q, create=[{"fn_id": add["fn_id"], "path": SHOP, "code": add["code"], "preexisting": True},
                        {"fn_id": ans["fn_id"], "path": SHOP, "code": ans["code"], "preexisting": False}],
             create_def_ids=[add["annotations"][0]["def_id"], ans["annotations"][0]["def_id"]])
    _, prov = comp.compose_prompt(r, "F2", words, pool(comp))
    assert add["fn_id"] not in prov["knowledge"]
    assert {"kind": "knowledge", "id": add["fn_id"], "rule": "twin"} in prov["dropped"]
