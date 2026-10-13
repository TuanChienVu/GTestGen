"""Pins for the Phase 3A prompt composer (v2tools/compose.py): one test per leak and exclusion rule (③ examples, ④
knowledge, the text rules), per budget rule and per structural rule, each with a positive control where one makes sense
(without the trigger the item is shown), so that deleting the rule turns its test red. One of them pins that the HEAD
body of a preexisting item makes its record a user of the test-answer infrastructure.

The catalog code of the shared fixture is the real repo content. A test that edits a function in memory and targets
another rule sets `ctx.unchanged` to always True, because rule 5 would otherwise show the function as the repo had it
before T (state.pre_t_code), or drop it, instead of the edited code; the rule-5 tests never do. Two rules cannot be
isolated on the shared fixture, where every change of a function also declares the members it uses: they run on a
second history of ShopContext.php (`reparented`, below)."""
from __future__ import annotations

import copy
import subprocess

import pytest

import v2tools.compose as compose
from tests.phase3_fixture import (BASE, COMPOSER, SHOP, SHOP_2020, T2020, T2022, T2024, _fn, _git, catalog, make_repo,
                                  record)
from v2tools.body import body_key
from v2tools.compose import EXAMPLES_HEAD, KNOWLEDGE_HEAD, TICKS, Composer, Params, aligned_slices, gold_answer, norm_ws
from v2tools.state import RepoContext, future_ref, render_answer, self_contained

REPO = "Acme/Shop"
BASE_PATH = "src/Behat/Context/BaseContext.php"
UI = "src/Behat/Context/UiShopContext.php"
END = "Now write the code for the undefined steps."


def words(text: str) -> int:
    return len(text.split())


def chars(text: str) -> int:  # every character costs a token: headers, fences and separators included
    return len(text)


def lumpy(text: str) -> int:  # superadditive: a joined block costs more than the sum of its parts
    return len(text) + (len(text) // 64) ** 2


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return make_repo(tmp_path_factory.mktemp("repo"))


def shop(base):
    """A fresh catalog, context and test query (the fixture's answer iShouldSeeTotal at T2024): tests edit them."""
    bare, head = base
    cat = catalog()
    ctx = RepoContext(REPO, "php", bare, head, cat)
    fn = {n: next(f for f in cat.values() if f"function {n}(" in f["code"])
          for n in ("iOpenTheShop", "iAddToCart", "iShouldSeeTotal")}
    return cat, ctx, record(cat), fn["iOpenTheShop"], fn["iAddToCart"], fn["iShouldSeeTotal"]


def rec(cat, name, fn, t, steps=("I open the shop",), partition="train", preexisting=False, **kw):
    """A record of `partition` in features/<name>.feature, t_vis = scenario_time = t, one CREATE item: `fn`."""
    rid = f"{REPO}|features/{name}.feature|1|s"
    e = dict(record(cat, partition, rid), feature_path=f"features/{name}.feature", t_vis=t, scenario_time=t,
             steps=list(steps), step_kinds=["create"] * len(steps), step_fns=[fn["fn_id"]] * len(steps),
             scenario="Scenario: x\n" + "\n".join(f"  Given {s}" for s in steps),
             neighbours=[] if partition == "train" else None,
             create=[{"fn_id": fn["fn_id"], "path": fn["path"], "code": fn["code"], "preexisting": preexisting}],
             create_def_ids=[fn["annotations"][0]["def_id"]])
    e.update(kw)
    return e


def comp_of(cat, ctx, recs, forbidden=None, **params):
    return Composer({REPO: ctx}, {REPO: cat}, forbidden or {}, {r["id"]: r for r in recs},
                    Params(**{"budget_examples": 400, "budget_knowledge": 400, **params}))


def step_fn(fid: str, first_line: int, pattern: str, code: str, time: int) -> dict:
    """A catalog row for a function that exists in memory only (not in the repo)."""
    path = fid.rsplit(":", 1)[0]
    return {"repo": REPO, "fn_id": fid, "path": path, "first_line": first_line, "body_end": first_line + 4,
            "code": code, "annotations": [{"def_id": f"{path}:{first_line}", "line": first_line, "keyword": "When",
                                           "pattern": pattern, "time": time}],
            "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}


# A second history of ShopContext.php, for the two rules the shared fixture cannot isolate:
#   2020-01-01  ShopContext without a parent; iOpenTheShop opens the index page
#   2022-01-01  iOpenTheShop's body changes (it also refreshes the page): no new member
#   2024-06-01  `extends BaseContext` arrives with the test answer iShouldSeeTotal, which calls the parent's helperOld,
#               declared in 2020
R2020 = SHOP_2020.replace(" extends BaseContext", "")
R2022 = R2020.replace("        $this->indexPage->open();\n",
                      "        $this->indexPage->open();\n        $this->indexPage->refresh();\n")
R2024 = R2022.replace("final class ShopContext\n", "final class ShopContext extends BaseContext\n").replace(
    "        $this->indexPage->refresh();\n    }\n",
    "        $this->indexPage->refresh();\n    }\n\n    #[Then('I should see the total :total')]\n"
    "    public function iShouldSeeTotal(string $total): void\n    {\n        $this->helperOld();\n"
    "        $this->indexPage->assertTotal($total);\n    }\n")


@pytest.fixture(scope="module")
def reparented(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reparented")
    work = tmp / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in [(T2020, {"composer.json": COMPOSER, BASE_PATH: BASE, SHOP: R2020}), (T2022, {SHOP: R2022}),
                        (T2024, {SHOP: R2024})]:
        for p, text in files.items():
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp / "repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    return bare, _git(bare, "rev-parse", "HEAD").strip()


def reparented_shop(reparented):
    bare, head = reparented
    lines = R2024.splitlines()
    body = {n: next(i for i, l in enumerate(lines, 1) if f"function {n}(" in l) + 1
            for n in ("iOpenTheShop", "iShouldSeeTotal")}
    rows = [_fn(body["iOpenTheShop"], "iOpenTheShop", "Given", "I open the shop", T2020, R2024),
            _fn(body["iShouldSeeTotal"], "iShouldSeeTotal", "Then", "I should see the total :total", T2024, R2024)]
    cat = {r["fn_id"]: r for r in rows}
    return cat, RepoContext(REPO, "php", bare, head, cat), record(cat), rows[0], rows[1]


# ----- ③ examples: silent filters (eligible_example) and recorded rules (example_rule) --------------------------------
def test_direct_neighbour_is_left_out_of_the_pool(base):
    """The example collides with the query by no other rule (another name, pattern, body and function): only the
    neighbour link keeps it out, silently."""
    cat, ctx, q, opn, add, ans = shop(base)
    a = rec(cat, "a", opn, T2020)
    b = rec(cat, "b", add, T2022, steps=['I add "x" to the cart'], neighbours=[a["id"]])
    free = dict(b, neighbours=[])
    comp = comp_of(cat, ctx, [q, a, b])
    assert comp.example_rule(free, comp.answer(free), a, comp.hidden(free)) is None
    assert Composer.text_rule(comp.answer(free), comp.example_text(a, 1)) is None
    _, prov = comp.compose_prompt(b, "F1", words, [a["id"]])
    assert prov["examples"] == [] and prov["dropped"] == []
    _, prov = comp.compose_prompt(free, "F1", words, [a["id"]])
    assert prov["examples"] == [a["id"]]


def test_example_whose_create_function_is_hidden_is_dropped_as_hidden(base):
    cat, ctx, q, opn, add, ans = shop(base)
    a = rec(cat, "a", opn, T2020)
    b = rec(cat, "b", add, T2022, steps=['I add "x" to the cart'])
    hide = {REPO: {"test": [opn["fn_id"]], "dev": [], "aux": []}}  # a's CREATE function answers a test record
    _, prov = comp_of(cat, ctx, [q, a, b], hide).compose_prompt(b, "F1", words, [a["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": a["id"], "rule": "hidden"}]
    _, prov = comp_of(cat, ctx, [q, a, b]).compose_prompt(b, "F1", words, [a["id"]])
    assert prov["examples"] == [a["id"]]


def ui_twin(ans: dict, pattern: str) -> dict:
    fn = copy.deepcopy(ans)
    fn.update(fn_id=f"{UI}:9", path=UI, first_line=7,
              code=f"#[Then('{pattern}')]\npublic function uiTotal(string $t): void\n{{\n    $this->page->x($t);\n}}")
    fn["annotations"] = [dict(fn["annotations"][0], def_id=f"{UI}:7", line=7, pattern=pattern, time=T2020)]
    return fn


def test_example_sharing_the_create_pattern_through_another_function_is_dropped_as_pattern(base):
    """The Api/Ui pair: the example creates ANOTHER function (another file) whose annotation carries the query's CREATE
    pattern, up to spaces and case."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # UiShopContext.php is not in the fixture repo: rule 5 would drop it first
    ui = cat.setdefault(f"{UI}:9", ui_twin(ans, "I  should see the TOTAL :total"))
    a = rec(cat, "a", ui, T2020, steps=['I should see the total "1"'])
    msgs, prov = comp_of(cat, ctx, [q, a]).compose_prompt(q, "F1", words, [a["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": a["id"], "rule": "pattern"}]
    assert "uiTotal" not in msgs[0]["content"]
    cat[ui["fn_id"]] = ui_twin(ans, "I should see the Ui sum :total")  # control: another pattern
    _, prov = comp_of(cat, ctx, [q, a]).compose_prompt(q, "F1", words, [a["id"]])
    assert prov["examples"] == [a["id"]]


TWIN_BODY = "{\n    $this->indexPage->open(['a' => 1, 'b' => 2, 'c' => 3, 'd' => 4, 'e' => 5]);\n}"


def test_example_twin_of_a_new_answer_is_dropped_as_twin(base):
    """The example creates the answer's body under another name (body_key_nosig equal, body_key not)."""
    cat, ctx, q, opn, add, ans = shop(base)
    a = rec(cat, "a", opn, T2020)
    _, prov = comp_of(cat, ctx, [q, a]).compose_prompt(q, "F1", words, [a["id"]])
    assert prov["examples"] == [a["id"]]  # control: the fixture's bodies differ
    cat, ctx, _, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # bodies edited in memory: rule 5 would show the repo's
    for fn in (opn, ans):
        fn["code"] = fn["code"].split("{", 1)[0] + TWIN_BODY
    q, a = record(cat), rec(cat, "a", opn, T2020)
    comp = comp_of(cat, ctx, [q, a])
    assert {k for k, _ in comp.fn_keys(REPO, opn["fn_id"]) & comp.answer(q).keys} == {"nosig"}
    _, prov = comp.compose_prompt(q, "F1", words, [a["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": a["id"], "rule": "twin"}]


def test_example_whose_create_function_changed_at_or_after_the_query_t_vis_is_dropped(base):
    """Rule 5 on ③, from the fixture's real history: iShouldSeeTotal's HEAD code was written at T2024, so for a query
    whose t_vis is at or before T2024 an older example creating it is code the query's V(r) never had."""
    cat, ctx, q, opn, add, ans = shop(base)
    e = rec(cat, "e", ans, T2022, steps=['I should see the total "1"'])
    for t in (T2022 + 1, T2024):
        r = rec(cat, "r", add, t, steps=['I add "x" to the cart'], partition="test")
        assert not ctx.unchanged(ans["fn_id"], t)
        _, prov = comp_of(cat, ctx, [r, e]).compose_prompt(r, "F1", words, [e["id"]])
        assert prov["examples"] == []
        assert prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "changed_after_t"}]
    r = rec(cat, "r", add, T2024 + 1, steps=['I add "x" to the cart'], partition="test")
    _, prov = comp_of(cat, ctx, [r, e]).compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [e["id"]] and prov["dropped"] == []  # control: HEAD code is the code before T


def test_example_using_a_member_first_declared_at_or_after_the_query_t_vis_is_dropped(base):
    """Rule 3 on ③: iAddToCart uses $cart, first declared at T2022."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # iAddToCart's code dates from T2022 too: rule 5 would drop it first
    e = rec(cat, "e", add, T2020, steps=['I add "x" to the cart'])
    r = dict(q, t_vis=T2022)
    _, prov = comp_of(cat, ctx, [r, e]).compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "future_ref"}]
    r = dict(q, t_vis=T2022 + 1)
    _, prov = comp_of(cat, ctx, [r, e]).compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [e["id"]] and prov["dropped"] == []


def test_example_from_the_same_feature_file_is_left_out(base):
    cat, ctx, q, opn, add, ans = shop(base)
    same = rec(cat, "s", opn, T2020, feature_path=q["feature_path"])
    other = rec(cat, "o", opn, T2020)
    _, prov = comp_of(cat, ctx, [q, same]).compose_prompt(q, "F1", words, [same["id"]])
    assert prov["examples"] == [] and prov["dropped"] == []
    _, prov = comp_of(cat, ctx, [q, other]).compose_prompt(q, "F1", words, [other["id"]])
    assert prov["examples"] == [other["id"]]


def test_example_sharing_a_create_function_is_left_out(base):
    """The query adds a new pattern to iOpenTheShop (a preexisting item: no new code, names or body keys), so no leak
    rule sees the older example that created iOpenTheShop: only the shared CREATE fn_id keeps it out."""
    cat, ctx, q, opn, add, ans = shop(base)
    opn["annotations"].append({"def_id": f"{SHOP}:15", "line": 15, "keyword": "Given", "pattern": "I visit the shop",
                               "time": T2024})
    add["annotations"].append({"def_id": f"{SHOP}:21", "line": 21, "keyword": "When", "pattern": "I put :product in",
                               "time": T2024})
    e = rec(cat, "e", opn, T2020)

    def query(fn, def_id):
        return dict(q, steps=["I visit the shop"], step_kinds=["create"], create_def_ids=[def_id],
                    create=[{"fn_id": fn["fn_id"], "path": SHOP, "code": fn["code"], "preexisting": True}])
    r = query(opn, f"{SHOP}:15")
    comp = comp_of(cat, ctx, [r, e])
    assert comp.example_rule(r, comp.answer(r), e, comp.hidden(r)) is None
    assert Composer.text_rule(comp.answer(r), comp.example_text(e, 1)) is None
    _, prov = comp.compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == []
    r = query(add, f"{SHOP}:21")  # control: the same kind of query on another function
    _, prov = comp_of(cat, ctx, [r, e]).compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [e["id"]]


# ----- ④ knowledge --------------------------------------------------------------------------------------------------
def test_knowledge_function_whose_old_annotation_carries_the_create_pattern_is_dropped_as_pattern(base):
    cat, ctx, q, opn, add, ans = shop(base)
    comp = comp_of(cat, ctx, [q])
    _, prov = comp.compose_prompt(q, "F2", words, [])
    assert opn["fn_id"] in prov["knowledge"]  # control: the fixture's iOpenTheShop
    cat, ctx, q, opn, add, ans = shop(base)
    # an older second annotation, on its own line, with the query's CREATE pattern up to spaces and case
    opn["code"] = opn["code"].replace("\n", "\n#[Then('I  should see the TOTAL :total')]\n", 1)
    opn["annotations"].append({"def_id": f"{SHOP}:17", "line": 17, "keyword": "Then",
                               "pattern": "I  should see the TOTAL :total", "time": T2020})
    msgs, prov = comp_of(cat, ctx, [q]).compose_prompt(q, "F2", words, [])
    assert opn["fn_id"] not in prov["knowledge"]
    assert {"kind": "knowledge", "id": opn["fn_id"], "rule": "pattern"} in prov["dropped"]
    assert "TOTAL" not in msgs[0]["content"]


def test_knowledge_function_using_a_member_first_declared_at_or_after_t_is_not_a_candidate(base):
    """Rule 3 in ④(b), silent: iAddToCart uses $cart, first declared at T2022 (its pattern is older than the
    scenario)."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # iAddToCart's code dates from T2022 too: rule 5 would drop it first
    for t, candidate in ((T2022, False), (T2022 + 1, True)):
        r = dict(q, t_vis=t, scenario_time=T2022 + 1)
        comp, dropped = comp_of(cat, ctx, [r]), []
        ids = [f for f, _ in comp.knowledge_candidates(r, comp.answer(r), dropped)]
        assert (add["fn_id"] in ids) == candidate and opn["fn_id"] in ids and dropped == []


def test_knowledge_function_whose_body_changed_at_or_after_t_is_shown_as_it_was_before_t(reparented):
    """Rule 5 in ④(b), on real history: iOpenTheShop's HEAD body dates from T2022 (no new member: rule 3 keeps it), so
    a query at T2022 sees its 2020 body (state.pre_t_code), a query after T2022 its HEAD body."""
    cat, ctx, q, opn, ans = reparented_shop(reparented)
    for t, head in ((T2022, False), (T2022 + 1, True)):
        r = dict(q, t_vis=t, scenario_time=t)
        assert not future_ref(ctx, opn, t)
        comp, dropped = comp_of(cat, ctx, [r]), []
        code = dict(comp.knowledge_candidates(r, comp.answer(r), dropped))
        assert opn["fn_id"] in code and dropped == []
        assert ("$this->indexPage->refresh();" in code[opn["fn_id"]]) == head
        msgs, _ = comp.compose_prompt(r, "F2", words, [])
        assert ("$this->indexPage->refresh();" in msgs[0]["content"]) == head


# ----- the answer infrastructure of held-out records ----------------------------------------------------------------
INFRA = {(SHOP, "checkout"), (SHOP, "computeTotal")}


def test_test_answer_infrastructure_is_masked_in_later_train_and_dev_prompts(base):
    """The test answer (T2024) uses $checkout and computeTotal, written with it. A later record of train or dev must
    not see them in ④(a) (the outline) nor in ④(b) (iShouldSeeTotal's body; forbidden.json is left empty so that
    only the mask keeps it out); a test record at the same time sees them."""
    cat, ctx, q, opn, add, ans = shop(base)
    for part, masked in (("train", True), ("dev", True), ("test", False)):
        t = rec(cat, "t", opn, T2024 + 100, partition=part)
        comp = comp_of(cat, ctx, [q, t])
        assert comp.infra()["members"] == {REPO: INFRA}
        msgs, prov = comp.compose_prompt(t, "F2", words, [])
        know = msgs[0]["content"].split(KNOWLEDGE_HEAD, 1)[1]
        assert ("private function computeTotal(" in know) == (not masked)
        assert ("private Checkout $checkout" in know) == (not masked)
        assert ("$this->checkout->assertTotal" in know) == (not masked)
        assert (ans["fn_id"] in prov["knowledge"]) == (not masked)


def infra_user() -> dict:
    code = ("#[When('I total :product')]\npublic function iTotal(string $product): void\n{\n"
            "    $this->computeTotal($product);\n}")
    return step_fn(f"{SHOP}:900", 898, "I total :product", code, T2020)


def test_train_record_whose_new_code_uses_the_infrastructure_leaves_every_pool(base):
    """The query is a test record after T2024 (no mask, computeTotal is old for it): only the pool rule keeps the user
    out, silently. Control: without the held-out answer there is no infrastructure and no user."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # the user's function exists in memory only: rule 5 would drop it first
    user = cat.setdefault(f"{SHOP}:900", infra_user())
    u = rec(cat, "u", user, T2020, steps=['I total "x"'])
    tq = rec(cat, "tq", opn, T2024 + 100, partition="test")
    comp = comp_of(cat, ctx, [q, u, tq])
    assert comp.infra() == {"members": {REPO: INFRA}, "users": {u["id"]}}
    _, prov = comp.compose_prompt(tq, "F1", words, [u["id"]])
    assert prov["examples"] == [] and prov["dropped"] == []
    comp = comp_of(cat, ctx, [u, tq])
    assert comp.infra() == {"members": {}, "users": set()}
    _, prov = comp.compose_prompt(tq, "F1", words, [u["id"]])
    assert prov["examples"] == [u["id"]]


def test_train_record_whose_preexisting_item_uses_the_infrastructure_at_head_leaves_every_pool(base):
    """uses() reads preexisting items too: an example shows the HEAD code of its preexisting item as gold, so a record
    whose preexisting item calls computeTotal at HEAD is a user too."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # edited in memory: rule 5 would show the repo's body
    opn["code"] = opn["code"].replace("$this->indexPage->open();", "$this->indexPage->open($this->computeTotal('1'));")
    e = rec(cat, "e", opn, T2022, preexisting=True)
    tq = rec(cat, "tq", add, T2024 + 100, steps=['I add "x" to the cart'], partition="test")
    comp = comp_of(cat, ctx, [q, e, tq])
    assert comp.infra()["members"] == {REPO: INFRA} and comp.infra()["users"] == {e["id"]}
    msgs, prov = comp.compose_prompt(tq, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [] and "computeTotal" not in msgs[0]["content"]
    comp = comp_of(cat, ctx, [e, tq])  # control: no held-out answer, no infrastructure
    msgs, prov = comp.compose_prompt(tq, "F1", words, [e["id"]])
    assert prov["examples"] == [e["id"]] and "$this->computeTotal('1')" in msgs[0]["content"]


def test_held_out_infrastructure_comes_from_new_code_only(base):
    """Infrastructure is what the NEW code of a held-out answer uses. A test record whose only use of computeTotal
    (added at its t_vis) is the HEAD body of a preexisting item defines none."""
    cat, ctx, q, opn, add, ans = shop(base)
    ctx.unchanged = lambda fid, t: True  # edited in memory: rule 5 would show the repo's body
    opn["code"] = opn["code"].replace("$this->indexPage->open();", "$this->indexPage->open($this->computeTotal('1'));")
    h = rec(cat, "h", opn, T2024, preexisting=True, partition="test")
    comp = comp_of(cat, ctx, [h])
    assert (SHOP, "computeTotal") in comp.uses(h, preexisting=True) and comp.infra()["members"] == {}


def test_aux_test_answers_define_infrastructure_too(base):
    """Held-out answers (test and aux_test) define infrastructure, dev and train answers do not; and only with members
    that no file of the class chain declared before the held-out record's t_vis (a record at T2024 + 1 uses the same
    members, old for it)."""
    cat, ctx, q, opn, add, ans = shop(base)
    t = rec(cat, "t", opn, T2024 + 100)
    for part, t_vis, members in (("aux_test", T2024, {REPO: INFRA}), ("test", T2024, {REPO: INFRA}),
                                 ("test", T2024 + 1, {}), ("dev", T2024, {}), ("train", T2024, {})):
        h = dict(q, partition=part, t_vis=t_vis, neighbours=[] if part == "train" else None)
        comp = comp_of(cat, ctx, [h, t])
        assert comp.uses(h) == INFRA and comp.infra()["members"] == members
        msgs, _ = comp.compose_prompt(t, "F2", words, [])
        assert ("computeTotal" in msgs[0]["content"]) == (not members)


def test_old_member_reached_through_a_parent_added_with_the_answer_is_not_infrastructure(reparented):
    """`extends BaseContext` arrives in the answer's commit and the answer calls BaseContext::helperOld (2020). The
    rule of compose.uses (new_since_t): such an old member is NOT infrastructure (old helpers of many classes, e.g.
    PrestaShop referenceToId, would be masked in every train prompt of the repo); the test record's own ④(a) does not
    show it (chain_before) and the record is not self-contained. So a later train record sees helperOld as an
    ordinary inherited member and a train record using it stays in the pools."""
    cat, ctx, q, opn, ans = reparented_shop(reparented)
    assert "helperOld" in ctx.declared_before(SHOP, q["t_vis"]) and ctx.chain_before(SHOP, q["t_vis"]) == [SHOP]
    msgs, _ = comp_of(cat, ctx, [q]).compose_prompt(q, "F2", words, [])
    assert "helperOld" not in msgs[0]["content"] and not self_contained(ctx, q, {ans["fn_id"]})
    ctx.unchanged = lambda fid, t: True  # the user's function exists in memory only: rule 5 would drop it first
    user = cat.setdefault(f"{SHOP}:900", step_fn(f"{SHOP}:900", 898, "I prepare :x",
                                                 "#[When('I prepare :x')]\npublic function iPrepare(string $x): void\n"
                                                 "{\n    $this->helperOld();\n}", T2024 + 10))
    u = rec(cat, "u", user, T2024 + 10, steps=['I prepare "x"'])
    t = rec(cat, "t", opn, T2024 + 100)
    hide = {REPO: {"test": [ans["fn_id"]], "dev": [], "aux": []}}
    comp = comp_of(cat, ctx, [q, u, t], hide)
    assert comp.uses(q) == {(BASE_PATH, "helperOld"), (SHOP, "indexPage")}
    assert comp.infra() == {"members": {}, "users": set()}
    msgs, prov = comp.compose_prompt(t, "F2", words, [u["id"]])
    assert "    protected function helperOld(): void" in msgs[0]["content"] and prov["examples"] == [u["id"]]


class RubyOnly:
    """A Ruby repo context that fails if uses() consults anything beyond its language."""
    language = "ruby"

    def __getattr__(self, name):
        raise AssertionError(f"uses() read {name!r} of a Ruby context")


def test_uses_is_empty_for_ruby(base):
    bare, head = base
    fid = "features/step_definitions/shop_steps.rb:1"
    code = "Then(/^I see the total (\\d+)$/) do |total|\n  expect(@checkout.total).to eq(compute_total(total))\nend"
    r = {"id": "Acme/Rb|features/t.feature|1|s", "repo": "Acme/Rb", "language": "ruby", "partition": "test",
         "t_vis": T2024, "create": [{"fn_id": fid, "path": SHOP, "code": code, "preexisting": False},
                                    {"fn_id": fid + "0", "path": SHOP, "code": code, "preexisting": True}]}
    comp = Composer({"Acme/Rb": RubyOnly()}, {"Acme/Rb": {}}, {}, {r["id"]: r})
    for kw in ({}, {"preexisting": True}, {"new_since_t": True}):
        assert comp.uses(r, **kw) == set()
    assert comp.infra() == {"members": {}, "users": set()}
    real = Composer({"Acme/Rb": RepoContext("Acme/Rb", "ruby", bare, head, {})}, {"Acme/Rb": {}}, {}, {r["id"]: r})
    assert real.uses(r, preexisting=True) == set() and real.uses(r, new_since_t=True) == set()


# ----- text rules ---------------------------------------------------------------------------------------------------
def test_short_slices_catch_the_tail_of_a_new_answer_and_aligned_slices_of_its_body_key(base):
    cat, ctx, q, opn, add, ans = shop(base)
    ans["code"] = ans["code"].split("{", 1)[0] + "{\n" + "".join(
        f"    $this->indexPage->assertLine('line {i:02d}', 'a fairly long argument that fills the slice {i:02d}');\n"
        for i in range(6)) + "}"
    q = record(cat)
    comp = comp_of(cat, ctx, [q])
    a = comp.answer(q)
    tail, key_slice = norm_ws(ans["code"])[-200:], aligned_slices(body_key(ans))[1]
    assert tail != key_slice and not any(s in tail for s in aligned_slices(body_key(ans)))
    for item in (tail, key_slice):
        assert not any(s in item for s in a.slices)  # no aligned slice of the code itself: only the short rule sees it
        assert Composer.text_rule(a, f"// copied\n{item}\n// end") == "short_slice"
        assert Composer.text_rule(a, item[1:]) is None  # control: 199 characters of it
    # end to end: a ④(b) function that ends like the answer
    ctx.unchanged = lambda fid, t: True  # edited in memory: rule 5 would show the repo's body
    opn["code"] = "#[Given('I open the shop')]\npublic function iOpenTheShop(): void\n{\n    " + tail
    msgs, prov = comp.compose_prompt(q, "F2", words, [])
    assert {"kind": "knowledge", "id": opn["fn_id"], "rule": "short_slice"} in prov["dropped"]
    assert tail not in norm_ws(msgs[0]["content"])


def test_exact_signature_line_is_caught_as_signature_before_name(base):
    cat, ctx, q, opn, add, ans = shop(base)
    a = comp_of(cat, ctx, [q]).answer(q)
    assert a.sig_lines == {"public function iShouldSeeTotal(string $total): void"}
    assert Composer.text_rule(a, "x\n    public function   iShouldSeeTotal(string $total):  void\ny") == "signature"
    assert Composer.text_rule(a, "public function iShouldSeeTotal(string $total): void {") == "name"  # not the line
    assert Composer.text_rule(a, "$this->iShouldSeeTotal('5');") == "name"
    assert Composer.text_rule(a, "$this->iShouldSeeTotalLater('5');") is None


# ----- budgets and structure ----------------------------------------------------------------------------------------
def budget_shop(base):
    cat, ctx, q, opn, add, ans = shop(base)
    pool = [rec(cat, f"e{i}", fn, t, steps=[s]) for i, (fn, t, s) in enumerate([
        (opn, T2020, "I open the shop"), (add, T2022, 'I add "x" to the cart'),
        (opn, T2020 + 1, "I open the shop again"), (add, T2022 + 1, 'I add "y" to the cart')])]
    return cat, ctx, q, [q] + pool, [e["id"] for e in pool]


def test_whole_example_and_knowledge_blocks_stay_within_budget_headers_included(base):
    cat, ctx, q, recs, ids = budget_shop(base)
    big = comp_of(cat, ctx, recs, budget_examples=10 ** 6, budget_knowledge=10 ** 6)
    a = big.answer(q)
    chosen, block = big.examples(q, a, ids, chars, [])
    texts = [big.example_text(big.records[x], i + 1) for i, x in enumerate(chosen)]
    assert len(chosen) == 3 and block == EXAMPLES_HEAD + "\n\n".join(texts)
    content = sum(map(len, texts))  # a budget that holds the examples but not the header and the separators
    few, small = comp_of(cat, ctx, recs, budget_examples=content).examples(q, a, ids, chars, [])
    assert len(few) < len(chosen) and len(small) <= content
    kblock, kfns, _ = comp_of(cat, ctx, recs, budget_knowledge=10 ** 6, mode="4b").knowledge(q, a, chars, [])
    code = dict(big.knowledge_candidates(q, a, []))
    content = sum(len(f"// {cat[f]['path']}\n{code[f]}") for f in kfns)  # ④(b) functions without header and fences
    assert len(kfns) == 2 and kblock == KNOWLEDGE_HEAD + "Existing step definitions:\n" + TICKS + "php\n" + \
        "\n\n".join(f"// {cat[f]['path']}\n{code[f]}" for f in kfns) + "\n" + TICKS
    kblock, fewer, _ = comp_of(cat, ctx, recs, budget_knowledge=content, mode="4b").knowledge(q, a, chars, [])
    assert len(fewer) < len(kfns) and len(kblock) <= content
    shown = set()
    for tok in (chars, lumpy):
        for b in range(0, tok(block) + 40, 9):
            c = comp_of(cat, ctx, recs, budget_examples=b, budget_knowledge=b)
            ca = c.answer(q)
            ex, kn = c.examples(q, ca, ids, tok, [])[1], c.knowledge(q, ca, tok, [])[0]
            assert tok(ex) <= b and tok(kn) <= b
            msgs, _ = c.compose_prompt(q, "F2", tok, ids)
            parts = [c.instructions(q), c.scenario_block(q)] + [x for x in (ex, kn) if x] + [END]
            assert msgs[0]["content"] == "\n\n".join(parts)  # the blocks measured are the prompt's blocks
            shown |= {("ex", ex.count("Example ")), ("kn", kn.count(TICKS))}
    assert ("ex", 3) in shown and ("ex", 1) in shown and ("kn", 4) in shown  # the sweep sees full and partial blocks


def test_ratio_caps_the_class_outline_and_modes_select_one_part(base):
    cat, ctx, q, recs, ids = budget_shop(base)

    def know(budget, **kw):
        c = comp_of(cat, ctx, recs, budget_knowledge=budget, **kw)
        return c.knowledge(q, c.answer(q), chars, [])
    whole, fns, skel = know(10 ** 6)
    only_a = know(10 ** 6, mode="4a")
    only_b = know(10 ** 6, mode="4b")
    assert only_a[1] == [] and only_a[2] == skel
    assert "Class outline of" in only_a[0] and "Existing step" not in only_a[0]
    assert only_b[2] == [] and only_b[1] == fns
    assert "Existing step" in only_b[0] and "Class outline" not in only_b[0]
    budget = len(whole)
    counts = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        block, f, s = know(budget, ratio=ratio)
        part_a = block.split("\n\nExisting step definitions:", 1)[0] if "Class outline of" in block else ""
        assert len(part_a) <= int(ratio * budget)
        counts.append(len(s))
    assert counts[0] == 0 and counts[-1] == len(skel) and 0 < counts[1] < len(skel)  # the ratio binds
    assert know(budget, ratio=0.25, mode="4a")[2] == skel  # "4a" gives ④(a) the whole budget
    with pytest.raises(ValueError):
        Params(mode="4c")


def test_placebo_knowledge_costs_no_more_tokens_than_the_real_knowledge_exactly(base, monkeypatch):
    cat, ctx, q, recs, ids = budget_shop(base)
    other = {f"x{i}": {"fn_id": f"x{i}", "path": f"src/Other{i}.php",
                       "code": f"public function f{i}(): int\n{{\n    return {i};\n}}"} for i in range(80)}
    monkeypatch.setitem(compose.PLACEBO_REPO, REPO, "PrestaShop/PrestaShop")
    cats = {REPO: cat, "PrestaShop/PrestaShop": other}
    filled = False
    by_id = {r["id"]: r for r in recs}
    for tok in (chars, words):
        for budget in range(0, 1600, 13):
            real = Composer({REPO: ctx}, cats, {}, by_id, Params(budget_knowledge=budget))
            fake = Composer({REPO: ctx}, cats, {}, by_id, Params(budget_knowledge=budget, mode="placebo"))
            target = tok(real.knowledge(q, real.answer(q), tok, [])[0])
            block, fns, skel = fake.knowledge(q, fake.answer(q), tok, [])
            assert tok(block) <= target and fns == [] and skel == []
            assert "ShopContext" not in block and "iOpenTheShop" not in block
            filled |= "src/Other" in block
    assert filled


def test_instructions_mark_a_target_file_absent_at_t_as_new_with_its_namespace(base):
    cat, ctx, q, opn, add, ans = shop(base)
    comp = comp_of(cat, ctx, [q])
    new_at_t = "class ShopContext, new file, namespace App\\Behat\\Context"
    for t, desc in ((T2020 - 1, new_at_t), (T2020, new_at_t), (T2020 + 1, "class ShopContext")):
        assert comp.instructions(dict(q, t_vis=t)).splitlines()[-1] == f"- {SHOP} ({desc})"
    new = "src/Behat/Context/CheckoutContext.php"  # absent at HEAD too: no class to name, no namespace to give
    lines = comp.instructions(dict(q, create=[dict(q["create"][0], path=new), q["create"][0]])).splitlines()
    assert lines[-2:] == [f"- {new} (new file)", f"- {SHOP} (class ShopContext)"]


def test_pool_of_every_repo_composes_the_prompt_of_the_query_repo_pool(base):
    """Another repo's train records change the BM25 statistics (they are about "see"): without the filter the query
    would get another example (n_examples = 1)."""
    bare, head = base
    cat, ctx, q, opn, add, ans = shop(base)
    a = rec(cat, "a", opn, T2020, steps=["I see the cart"])
    b = rec(cat, "b", add, T2022, steps=["I check the order total"])
    others = [dict(rec(cat, f"z{i}", opn, T2020, steps=["I see the page", "I see the menu"]), repo="Other/Rb",
                   id=f"Other/Rb|features/z{i}.feature|1|s", language="ruby") for i in range(3)]
    recs = {r["id"]: r for r in [q, a, b] + others}
    comp = Composer({REPO: ctx, "Other/Rb": RepoContext("Other/Rb", "ruby", bare, head, {})},
                    {REPO: cat, "Other/Rb": {}}, {}, recs,
                    Params(n_examples=1, budget_examples=400, budget_knowledge=400))
    mine = [a["id"], b["id"]]
    every = [r["id"] for r in [a, b] + others]
    query = compose.tokens("\n".join(compose.undefined_steps(q)))
    order = lambda pool: [x for x, _ in comp.pool_index(pool).rank(query) if x in mine]
    assert order(mine) == mine and order(every) == mine[::-1]  # the statistics would change the choice
    got = comp.compose_prompt(q, "F2", words, every)
    assert got == comp.compose_prompt(q, "F2", words, mine) and got[1]["examples"] == [a["id"]]


def test_gold_answer_needs_the_catalog_only_for_preexisting_items():
    cat = catalog()
    q = record(cat)
    code = q["create"][0]["code"]
    assert gold_answer(q) == gold_answer(q, cat) == f"{TICKS}php\n// FILE: {SHOP}\n{code}\n{TICKS}"
    add = next(f for f in cat.values() if "iAddToCart" in f["code"])
    pre_item = {"fn_id": add["fn_id"], "path": SHOP, "code": "PHASE 2 CODE", "preexisting": True}
    r = dict(q, create=[pre_item] + q["create"],
             create_def_ids=[add["annotations"][0]["def_id"]] + q["create_def_ids"])
    with pytest.raises(ValueError, match="needs the catalog"):
        gold_answer(r)
    got = gold_answer(r, cat)
    pre = render_answer(add, r["t_vis"], frozenset(r["create_def_ids"]))
    assert got == f"{TICKS}php\n// FILE: {SHOP}\n{pre}\n\n{code}\n{TICKS}" and "PHASE 2 CODE" not in got
