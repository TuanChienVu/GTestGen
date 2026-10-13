import random
from collections import Counter

import pytest

from v2gpu import leakcheck as lc

PHP_NEW = """#[When('I add the product :name to the wishlist')]
    public function iAddTheProductToTheWishlist(string $name): void
    {
        $product = $this->productRepository->findOneByCode($name);
        $this->wishlistPage->addProduct($product);
    }"""


def fn(fn_id, code, patterns=(), first_line=10, ann_lines=None, times=None, suites=("s",), catch_all=False,
       scope_tags=()):
    """A catalog entry: annotation lines default to the first len(patterns) lines of the code; annotation times default
    to 50, i.e. before the default scenario_time 100 (an old, visible pattern)."""
    lines = ann_lines if ann_lines is not None else list(range(first_line, first_line + len(patterns)))
    times = times if times is not None else [50] * len(patterns)
    return {"fn_id": fn_id, "code": code, "first_line": first_line, "suites": list(suites), "catch_all": catch_all,
            "scope_tags": [list(t) for t in scope_tags],
            "annotations": [{"def_id": f"{fn_id}@{l}", "line": l, "pattern": p, "time": t}
                            for l, p, t in zip(lines, patterns, times)]}


NEW = fn("Ui/Wish.php:10", PHP_NEW, ["I add the product :name to the wishlist"], times=[150])
# the same function copied into the Api context: other annotation, same signature and body
TWIN = fn("Api/Wish.php:10", PHP_NEW.replace("I add the product :name to the wishlist", "I add :name via the API"),
          ["I add :name via the API"])


def rec(rid, feature="f1.feature", create=(), partition="train", repo="Sylius/Sylius", create_def_ids=(), **kw):
    r = {"id": rid, "repo": repo, "feature_path": feature, "partition": partition, "create": list(create),
         "create_def_ids": list(create_def_ids), "suite": "s", "tags": [], "scoping": None, "scenario_time": 100,
         "t_vis": 150, "background": "", "scenario": "", "steps": []}
    if partition == "train":
        r["neighbours"] = []
    return {**r, **kw}


def item(fn_id, code="", preexisting=False):
    return {"fn_id": fn_id, "path": fn_id.split(":")[0], "code": code, "preexisting": preexisting}


def test_normalize_ws():
    assert lc.normalize_ws("  a\n\t b   c \n") == "a b c"


def test_every_400_char_window_contains_an_aligned_200_slice():
    rng = random.Random(0)
    code = "".join(rng.choice("abcdefgh(){};$ \n") for _ in range(1500))
    norm, slices = lc.normalize_ws(code), lc.code_slices(code)
    for p in range(len(norm) - 400 + 1):
        assert any(s in norm[p:p + 400] for s in slices)


def test_short_code_is_one_whole_slice():
    assert lc.code_slices("function x() { }") == ["function x() { }"]


JAVA_LAMBDA = fn("BatchStepDef.java:40", 'When("/^I call the batch API (.*)$/", (String body) -> {\n'
                 '    Response r = batchApi.call(body);\n    assertEquals(200, r.code());\n});',
                 ["^I call the batch API (.*)$"], 40)


def test_method_names_come_from_the_declaration():
    assert lc.method_name(NEW, "php") == "iAddTheProductToTheWishlist"
    java = fn("S.java:5", '@When("I pay {int}")\n@SuppressWarnings("unused")\npublic void iPay(int amount) throws Exception {\n'
              '  pay(amount);\n}', ["I pay {int}"], 5)
    csharp = fn("S.cs:5", '[Given(@"a patient")]\npublic async Task GivenAPatient(Table t)\n{\n  Setup();\n}', ["a patient"])
    doc = fn("S.php:5", "/**\n * @Given a shop\n */\npublic function aShop(): void\n{\n}", ["a shop"], 5, [6])
    ruby = fn("s.rb:5", "Given(/^a patient$/) do\n  visit(root_path)\nend", ["^a patient$"])
    commented = fn("C.cs:187", '//[Then(@"it works")]\n//public void ThenItWorks()\n//{\n//}', ["it works"], 187)
    assert (lc.method_name(java, "java"), lc.method_name(csharp, "csharp"), lc.method_name(doc, "php"),
            lc.method_name(ruby, "ruby"), lc.method_name(commented, "csharp")) == ("iPay", "GivenAPatient", "aShop",
                                                                                 None, None)
    assert lc.method_name(JAVA_LAMBDA, "java") is None          # cucumber-java8 lambda: no method name
    nested = fn("COB.java:163", 'Then("throw exception", () -> {\n    assertThrows(BusinessStepException.class, () -> {\n'
                '        service.run(map, item);\n    });\n});', ["throw exception"], 163)
    assert lc.method_name(nested, "java") is None               # a lambda inside the lambda is not a declaration
    multi_line = fn("Sql.java:39", 'Given("/^A criteria (.*) and arguments (.*)$/", (String criteria,\n'
                    '        String argument) -> {\n    sqlBuilder = new SqlBuilder();\n});', ["^A criteria (.*)$"], 39)
    assert lc.is_lambda(JAVA_LAMBDA, "java") and lc.is_lambda(nested, "java") and lc.is_lambda(multi_line, "java")
    assert not lc.is_lambda(java, "java") and lc.method_name(multi_line, "java") is None
    php_default = fn("D.php:5", "#[Given('d')]\npublic function withDefaults(array $o = ['a' => 1]): void\n{\n}", ["d"], 5)
    assert lc.method_name(php_default, "php") == "withDefaults"   # `=>` in a PHP default is not a lambda


def test_a_new_function_whose_name_cannot_be_found_is_a_problem_not_a_silent_skip():
    """Every function that is neither a lambda nor Ruby must have a method name."""
    same_line = fn("S.java:5", '@When("I pay") public void iPay() {\n  pay();\n}', ["I pay"], 5)
    q = rec("q", "q.feature", [item(same_line["fn_id"], same_line["code"])], create_def_ids=["S.java:5@5"])
    c = lc.Context(pool={"q": q}, catalog={"Sylius/Sylius": {same_line["fn_id"]: same_line}},
                   hidden={"Sylius/Sylius": set()}, language={"Sylius/Sylius": "java"}, neighbours=lc.neighbours([q]))
    assert any("no method name found for S.java:5" in p for p in lc.check(q, msgs("..."), prov(), c))


def test_body_key_keeps_the_signature_and_drops_annotations_and_comments():
    assert lc.body_key(NEW) == lc.body_key(TWIN) is not None
    assert lc.body_key(NEW).startswith("public function iAddTheProductToTheWishlist(string $name)")
    other_name = fn("X.php:1", PHP_NEW.replace("iAddTheProductToTheWishlist", "iPutItOnTheWishlist"), ["x"])
    assert lc.body_key(other_name) != lc.body_key(NEW)          # same idiom, other method: not a copy by body_key
    commented = fn("C.php:1", "#[When('c')]\n// helper\n# note\npublic function c(): void\n{\n}", ["c"])
    assert lc.body_key(commented) is None                       # under 60 characters once cleaned


def test_body_key_nosig_catches_a_renamed_copy():
    a = fn("Batch.java:673", '@Then("the batch {string} succeeds")\npublic void theBatchSucceeds(String id) {\n'
           '    BatchResponse r = batchApi.get(id);\n    assertThat(r.getStatusCode()).isEqualTo(200);\n}',
           ["the batch {string} succeeds"], 673)
    b = fn("Batch.java:713", a["code"].replace("theBatchSucceeds", "theBatchRunsOk").replace("succeeds\"", "runs ok\""),
           ["the batch {string} runs ok"], 713)
    assert lc.body_key(a) != lc.body_key(b) and lc.body_key_nosig(a, "java") == lc.body_key_nosig(b, "java") is not None
    assert lc.body_key_nosig(a, "java").startswith("BatchResponse r =")
    assert lc.body_key_nosig(JAVA_LAMBDA, "java") == lc.body_key(JAVA_LAMBDA) is not None   # no declaration: kept whole
    ruby = fn("s.rb:3", "Given(/^a patient named (.*)$/) do |name|\n  visit patients_path(name: name, extra: 1, more: 2)\n"
                        "  expect(page).to have_content(name)\nend", ["^a patient named (.*)$"], 3)
    assert lc.body_key_nosig(ruby, "ruby") == lc.body_key(ruby)


def test_ruby_step_line_is_an_annotation_and_the_block_is_kept():
    ruby = fn("s.rb:3", "Given(/^a patient named (.*)$/) do |name|\n  visit patients_path(name: name, extra: 1, more: 2)\n"
                        "  expect(page).to have_content(name)\nend", ["^a patient named (.*)$"], 3)
    assert lc.body_key(ruby).startswith("visit patients_path") and lc.body_key(ruby).endswith("end")


def test_direct_neighbours_are_not_transitive():
    a, b = rec("a", "f1.feature", [item("F.php:1")]), rec("b", "f1.feature", [item("F.php:2")])
    c, d = rec("c", "f9.feature", [item("F.php:2")]), rec("d", "f7.feature", [item("F.php:3")])
    nb = lc.neighbours([a, b, c, d])
    assert nb == {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}, "d": set()}


def test_neighbours_field_must_equal_the_direct_links():
    a = rec("a", "f1.feature", [item("F.php:1")], neighbours=["b"])
    b = rec("b", "f1.feature", [item("F.php:2")], neighbours=["a", "c"])
    c = rec("c", "f9.feature", [item("F.php:2")], neighbours=["b"])
    assert lc.neighbour_problems([a, b, c], []) == []
    wrong = dict(c, neighbours=["a", "b"])                        # transitive, as the old co-creation group was
    assert any("c: neighbours field" in p for p in lc.neighbour_problems([a, b, wrong], []))
    assert any("no neighbours field" in p for p in lc.neighbour_problems([{k: v for k, v in a.items() if k != "neighbours"}], []))


def test_no_query_outside_train_may_have_a_train_neighbour():
    t = rec("t", "shared.feature", [item("F.php:1")])
    d = rec("d", "shared.feature", [item("F.php:9")], partition="dev")
    x = rec("x", "x.feature", [item("F.php:1")], partition="test")
    probs = lc.neighbour_problems([t], [d, x])
    assert any("dev record d" in p for p in probs) and any("test record x" in p for p in probs)
    assert lc.neighbour_problems([t], [dict(d, feature_path="other.feature")]) == []


def test_hidden_sets_by_partition():
    f = {"test": ["t"], "dev": ["d"], "aux": ["x"]}
    assert lc.hidden_for("train", f) == {"t", "d", "x"}
    assert lc.hidden_for("dev", f) == {"t", "x"}
    assert lc.hidden_for("test", f) == set() == lc.hidden_for("aux_test", f)


def test_snapshot_membership_and_future_functions():
    q = rec("q")
    old = fn("O.php:1", "#[When('o')]\npublic function o(): void\n{\n}", ["o"])
    assert lc.snapshot_reason(q, old, set()) is None
    assert "hidden" in lc.snapshot_reason(q, old, {"O.php:1"})
    assert "before the scenario" in lc.snapshot_reason(q, dict(old, annotations=[dict(old["annotations"][0], time=120)]), set())
    assert "suite" in lc.snapshot_reason(q, dict(old, suites=["other"]), set())
    assert "catch-all" in lc.snapshot_reason(q, dict(old, catch_all=True), set())
    assert "dead" in lc.snapshot_reason(q, dict(old, dead=True), set())
    assert "contains" in lc.snapshot_reason(q, dict(old, contains_fn=["O.php:9"]), set())
    scoped = rec("q2", scoping="specflow_scope", tags=["@Smoke"])
    assert lc.snapshot_reason(scoped, dict(old, scope_tags=[["@smoke"]]), set()) is None
    assert "scope" in lc.snapshot_reason(scoped, dict(old, scope_tags=[["@manual"]]), set())
    future = dict(old, annotations=[dict(old["annotations"][0], time=150)])
    assert lc.is_future(future, 150) and not lc.is_future(old, 150) and not lc.is_future(future, None)


QUERY = rec("q", "q.feature", [item(NEW["fn_id"], PHP_NEW)], create_def_ids=[NEW["annotations"][0]["def_id"]])


def ctx(pool, catalog=(), hidden=(), language="php", queries=(QUERY,)):
    cat = {f["fn_id"]: f for f in [NEW, *catalog]}
    return lc.Context(pool={r["id"]: r for r in pool}, catalog={"Sylius/Sylius": cat},
                      hidden={"Sylius/Sylius": set(hidden)}, language={"Sylius/Sylius": language},
                      neighbours=lc.neighbours(list({r["id"]: r for r in [*pool, *queries]}.values())))


def msgs(text):
    return [{"role": "user", "content": text}]


def prov(examples=(), knowledge=(), skeleton=()):
    return {"examples": list(examples), "knowledge": list(knowledge), "skeleton": list(skeleton), "dropped": []}


def test_clean_prompt_passes():
    shop = fn("S.php:1", "#[Given('a shop')]\npublic function aShop(): void\n{\n}", ["a shop"])
    ex = rec("e", "e.feature", [item("S.php:1")], create_def_ids=["S.php:1@10"], t_vis=120)
    assert lc.check(QUERY, msgs("Scenario ... Example: a shop"), prov(["e"], ["S.php:1"]), ctx([QUERY, ex], [shop])) == []


def test_answer_code_with_other_whitespace_is_caught():
    reindented = "\n".join("  " + l.strip() for l in PHP_NEW.splitlines())
    probs = lc.check(QUERY, msgs(reindented), prov(), ctx([QUERY]))
    assert any("answer code" in p for p in probs)


def test_preexisting_functions_may_be_shown_without_their_create_annotation():
    q = dict(QUERY, create=[item(NEW["fn_id"], PHP_NEW, preexisting=True)])
    body = PHP_NEW.split("\n", 1)[1]                                        # the annotation line removed
    old_new = dict(NEW, first_line=9, code="#[When('old one')]\n" + PHP_NEW,
                   annotations=[{"def_id": "Ui/Wish.php:10@9", "line": 9, "pattern": "old one", "time": 50},
                                NEW["annotations"][0]])
    assert lc.check(q, msgs(body), prov(knowledge=[NEW["fn_id"], TWIN["fn_id"]], skeleton=[NEW["fn_id"]]),
                    ctx([q], [old_new, TWIN], queries=[q])) == []


# a preexisting function: an old pattern (line 10) and the pattern this record's step created (line 11)
OLD_AND_NEW = fn("Ui/Cart.php:10", "#[When('I add :product to the cart')]\n#[When('I put :product in my cart')]\n"
                 "public function iAddToTheCart(ProductInterface $product): void\n{\n    $this->cart->add($product);\n}",
                 ["I add :product to the cart", "I put :product in my cart"], times=[50, 150])
PRE_Q = rec("q", "q.feature", [item(OLD_AND_NEW["fn_id"], OLD_AND_NEW["code"], preexisting=True)],
            create_def_ids=["Ui/Cart.php:10@11"])


def test_preexisting_function_in_knowledge_that_keeps_its_create_annotation_is_caught():
    c = ctx([PRE_Q], [OLD_AND_NEW], queries=[PRE_Q])
    probs = lc.check(PRE_Q, msgs(OLD_AND_NEW["code"]), prov(knowledge=[OLD_AND_NEW["fn_id"]]), c)
    assert any("CREATE pattern 'i put :product in my cart'" in p for p in probs), probs
    rendered = OLD_AND_NEW["code"].replace("#[When('I put :product in my cart')]\n", "")
    assert lc.check(PRE_Q, msgs(rendered), prov(knowledge=[OLD_AND_NEW["fn_id"]]), c) == []


def test_create_annotation_line_of_a_twin_outside_the_provenance_is_caught():
    api = fn("Api/Cart.php:40", "#[When('I put :product in my cart')]\npublic function putInCart($p): void\n{\n}",
             ["I put :product in my cart"], 40)
    frame = "final class CartContext\n{\n    #[When('I put :product in my cart')]\n    public function putInCart($p): void;\n}"
    probs = lc.check(PRE_Q, msgs(frame), prov(), ctx([PRE_Q], [OLD_AND_NEW, api], queries=[PRE_Q]))
    assert any("Api/Cart.php:40" in p for p in probs), probs


def test_example_sharing_a_create_pattern_with_a_preexisting_item_is_caught():
    twin = fn("Api/Cart.php:40", "#[When('I put :product in my cart')]\npublic function putInCart($p): void\n{\n}",
              ["I put :product in my cart"], 40)
    ex = rec("e", "e.feature", [item(twin["fn_id"])], create_def_ids=["Api/Cart.php:40@40"])
    probs = lc.check(PRE_Q, msgs("..."), prov(["e"]), ctx([PRE_Q, ex], [OLD_AND_NEW, twin], queries=[PRE_Q]))
    assert any("e: CREATE step pattern" in p for p in probs), probs


def test_create_patterns_match_whole_not_as_substrings():
    q = rec("q", "q.feature", [item("S.php:1", preexisting=True)], create_def_ids=["S.php:1@1"])
    short = fn("S.php:1", "#[When('I add the product')]\npublic function s(): void\n{\n}", ["I add the product"], 1)
    longer = fn("L.php:1", "#[When('I add the product :name to the wishlist')]\npublic function l(): void\n{\n}",
                ["I add the product :name to the wishlist"], 1)
    shown = "#[When('I add the product :name to the wishlist')]\n* @When I add the product to it"
    assert lc.check(q, msgs(shown), prov(knowledge=["L.php:1"]), ctx([q], [short, longer], queries=[q])) == []


def test_create_def_ids_are_required_and_must_belong_to_the_create_functions():
    no_field = {k: v for k, v in QUERY.items() if k != "create_def_ids"}
    assert any("no create_def_ids" in p for p in lc.check(no_field, msgs("..."), prov(), ctx([QUERY])))
    stray = dict(QUERY, create_def_ids=["Other.php:1@1"])
    assert any("Other.php:1@1" in p for p in lc.check(stray, msgs("..."), prov(), ctx([QUERY])))


def test_example_rules():
    other = fn("O.php:1", "#[When('other')]\npublic function other(): void\n{\n}", ["other"])
    same_feature = rec("s", "q.feature", [item("O.php:1")], create_def_ids=["O.php:1@1"])
    shared_fn = rec("h", "h.feature", [item(NEW["fn_id"])], create_def_ids=[])
    twin = rec("t", "t.feature", [item(TWIN["fn_id"])], create_def_ids=[TWIN["annotations"][0]["def_id"]])
    same_pattern_fn = fn("P.php:1", "#[When('I add the product :name to the wishlist')]\npublic function p() {}",
                         ["I add the product :name to the wishlist"], 1)
    same_pattern = rec("p", "p.feature", [item("P.php:1")], create_def_ids=["P.php:1@1"])
    renamed = fn("R.php:1", PHP_NEW.replace("iAddTheProductToTheWishlist", "iWishlistIt").replace(
        "I add the product :name to the wishlist", "I wishlist :name"), ["I wishlist :name"], 1)
    renamed_copy = rec("r", "r.feature", [item("R.php:1")], create_def_ids=["R.php:1@1"])
    hidden_fn = fn("Hid.php:1", "#[When('hid')]\npublic function hid(): void\n{\n}", ["hid"], 1)
    shows_hidden = rec("x", "x.feature", [item("Hid.php:1")], create_def_ids=["Hid.php:1@1"])
    no_ids = {k: v for k, v in rec("n", "n.feature", [item("O.php:1")]).items() if k != "create_def_ids"}
    pool = [QUERY, same_feature, shared_fn, twin, same_pattern, renamed_copy, shows_hidden, no_ids]
    c = ctx(pool, [other, TWIN, same_pattern_fn, renamed, hidden_fn], hidden=["Hid.php:1"])
    probs = "\n".join(lc.check(QUERY, msgs("..."), prov(["s", "h", "t", "p", "r", "x", "n", "zz"]), c))
    for needle in ("s: same .feature", "h: direct neighbour", "h: shares CREATE function", "t: twin of a new CREATE",
                   "p: CREATE step pattern", "r: twin of a new CREATE", "x: CREATE function Hid.php:1 is hidden",
                   "n: example record has no create_def_ids", "zz: not in the retrieval pool"):
        assert needle in probs, needle


def test_example_listed_in_the_neighbours_field_is_caught():
    ex = rec("e", "e.feature", [item("E.php:1")], create_def_ids=["E.php:1@1"])
    e_fn = fn("E.php:1", "#[Given('e')]\npublic function e(): void\n{\n}", ["e"], 1)
    q = dict(QUERY, neighbours=["e"])                        # the Phase 2 build linked them through something not visible here
    assert any("e: direct neighbour" in p for p in lc.check(q, msgs("..."), prov(["e"]), ctx([q, ex], [e_fn], queries=[q])))


def test_knowledge_rules():
    by_name = fn("K.php:2", "public function iAddTheProductToTheWishlist() { }", [], ann_lines=[])
    by_pattern = fn("K.php:3", "#[When('I add the product :name to the wishlist')]\npublic function k() { }",
                    ["I add the product :name to the wishlist"])
    hidden = fn("K.php:4", "#[When('fine')]\npublic function fine() { }", ["fine"])
    future = fn("K.php:5", "#[When('later')]\npublic function later() { }", ["later"], times=[160])
    other_suite = fn("K.php:6", "#[When('elsewhere')]\npublic function elsewhere() { }", ["elsewhere"], suites=["x"])
    c = ctx([QUERY], [TWIN, by_name, by_pattern, hidden, future, other_suite], hidden=["K.php:4"])
    probs = "\n".join(lc.check(QUERY, msgs("..."), prov(knowledge=[
        TWIN["fn_id"], "K.php:2", "K.php:3", "K.php:4", "K.php:5", "K.php:6", "K.php:9"]), c))
    for needle in (f"{TWIN['fn_id']}: twin of a new CREATE", "K.php:2: method name", "K.php:3: CREATE step pattern",
                   "K.php:4: hidden", "K.php:5: future step function", "K.php:6: not in the snapshot (other suite)",
                   "K.php:9: not in the catalog"):
        assert needle in probs, needle


def test_a_contains_fn_knowledge_entry_is_not_in_the_snapshot():
    """A span that swallows another function's body is never in the snapshot, so an independent
    ④(b) check must reject it rather than let the hidden code through."""
    swallows = dict(fn("K.php:9", "#[When('swallow')]\npublic function swallow() { }", ["swallow"]),
                    contains_fn=["Hidden.php:1"])
    probs = "\n".join(lc.check(QUERY, msgs("..."), prov(knowledge=[swallows["fn_id"]]), ctx([QUERY], [swallows])))
    assert "K.php:9: not in the snapshot" in probs, probs


def test_a_new_create_function_of_the_query_is_never_in_knowledge_or_skeleton():
    probs = "\n".join(lc.check(QUERY, msgs("..."), prov(knowledge=[NEW["fn_id"]], skeleton=[NEW["fn_id"]]), ctx([QUERY])))
    assert f"knowledge {NEW['fn_id']}: a new CREATE function of the query" in probs
    assert f"skeleton {NEW['fn_id']}: a new CREATE function of the query" in probs


def test_the_knowledge_pattern_rule_looks_only_at_annotations_v_r_keeps():
    """An annotation at or after t_vis (or undated) is not shown by V(r), so it cannot leak."""
    later = fn("K.php:7", "#[When('seen')]\n#[When('I add the product :name to the wishlist')]\npublic function k() { }",
               ["seen", "I add the product :name to the wishlist"], times=[50, 160])
    undated = dict(later, fn_id="K.php:8", annotations=[dict(a, def_id=a["def_id"].replace("K.php:7", "K.php:8"))
                                                        for a in later["annotations"][:1]] +
                   [dict(later["annotations"][1], def_id="K.php:8@11", time=None)])
    probs = lc.check(QUERY, msgs("..."), prov(knowledge=["K.php:7", "K.php:8"]), ctx([QUERY], [later, undated]))
    assert not any("CREATE step pattern" in p for p in probs), probs


def test_skeleton_rules():
    hidden = fn("K.php:4", "#[When('fine')]\npublic function fine() { }", ["fine"])
    future = fn("K.php:5", "#[When('later')]\npublic function later() { }", ["later"], times=[160])
    same_name = fn("K.php:2", "#[When('x')]\npublic function iAddTheProductToTheWishlist() { }", ["x"])
    other_suite = fn("K.php:6", "#[When('elsewhere')]\npublic function elsewhere() { }", ["elsewhere"], suites=["x"])
    c = ctx([QUERY], [TWIN, hidden, future, same_name, other_suite], hidden=["K.php:4"])
    probs = "\n".join(lc.check(QUERY, msgs("..."), prov(skeleton=[TWIN["fn_id"], "K.php:2", "K.php:4", "K.php:5",
                                                                   "K.php:6", "K.php:9"]), c))
    for needle in (f"skeleton {TWIN['fn_id']}: twin of a new CREATE", "skeleton K.php:2: method name",
                   "skeleton K.php:4: hidden", "skeleton K.php:5: future step function", "skeleton K.php:9: not in the catalog"):
        assert needle in probs, needle
    assert "K.php:6" not in probs                           # a class frame may show members of other suites


def test_method_name_of_a_new_answer_anywhere_in_the_prompt_is_caught():
    caller = "public function iSeeIt(): void\n{\n    $this->iAddTheProductToTheWishlist('x');\n}"
    assert any("method name iAddTheProductToTheWishlist" in p
               for p in lc.check(QUERY, msgs(caller), prov(), ctx([QUERY])))
    assert not any("method name" in p for p in lc.check(QUERY, msgs("... iAddTheProductToTheWishlistLater ..."), prov(),
                                                         ctx([QUERY])))       # whole tokens only
    in_gherkin = dict(QUERY, scenario="Scenario: When I call iAddTheProductToTheWishlist")
    assert not any("method name" in p for p in lc.check(in_gherkin, msgs(in_gherkin["scenario"]), prov(), ctx([in_gherkin],
                                                                                                           queries=[in_gherkin])))


def test_lambda_answers_have_no_method_name_rule():
    q = rec("q", "q.feature", [item(JAVA_LAMBDA["fn_id"], JAVA_LAMBDA["code"])],
            create_def_ids=[JAVA_LAMBDA["annotations"][0]["def_id"]])
    shown = "public void check() {\n    assertEquals(1, 1);\n}"
    c = lc.Context(pool={"q": q}, catalog={"Sylius/Sylius": {JAVA_LAMBDA["fn_id"]: JAVA_LAMBDA}},
                   hidden={"Sylius/Sylius": set()}, language={"Sylius/Sylius": "java"}, neighbours=lc.neighbours([q]))
    assert lc.check(q, msgs(shown), prov(), c) == []


def test_signature_line_of_a_new_answer_is_caught():
    for frame in ("final class WishContext\n{\n    public function iAddTheProductToTheWishlist(string $name): void\n}",
                  "    public function iAddTheProductToTheWishlist(string $name): void;",
                  # compose's ④(a) form (decision of 24/09): the signature up to `{`, then ` /* … */ }`
                  "    public function iAddTheProductToTheWishlist(string $name): void { /* \u2026 */ }",
                  "    public function iAddTheProductToTheWishlist(string $name): void\n    { /* \u2026 */ }"):
        assert any("signature line of Ui/Wish.php:10" in p for p in lc.check(QUERY, msgs(frame), prov(), ctx([QUERY]))), frame


def test_the_first_line_of_a_multi_line_signature_is_compared():
    code = ("#[Given('a review by :customer')]\npublic function aReviewBy(\n    ProductInterface $product,\n"
            "    CustomerInterface $customer,\n): void {\n    $this->factory->create($product, $customer, 'text');\n}")
    f = fn("Rev.php:10", code, ["a review by :customer"], times=[150])
    q = rec("q", "q.feature", [item(f["fn_id"], code)], create_def_ids=[f["annotations"][0]["def_id"]])
    frame = "    public function aReviewBy(\n        ProductInterface $product,\n        CustomerInterface $customer,\n    ): void { /* \u2026 */ }"
    assert any("signature line of Rev.php:10" in p for p in lc.check(q, msgs(frame), prov(), ctx([q], [f], queries=[q])))


def test_render_answer_keeps_create_and_old_annotations_only():
    code = "#[When('old')]\n#[When('later')]\n#[When('created')]\n#[When('undated')]\npublic function x(): void\n{\n}"
    f = fn("X.php:10", code, ["old", "later", "created", "undated"], times=[50, 160, 150, None])
    out = lc.render_answer(f, 150, {"X.php:10@12"})
    assert out == "#[When('old')]\n#[When('created')]\npublic function x(): void\n{\n}"
    assert lc.render_answer(f, None, {"X.php:10@12"}).splitlines()[:3] == ["#[When('old')]", "#[When('later')]",
                                                                         "#[When('created')]"]


def test_an_example_must_be_visible_before_the_query():
    ex = rec("e", "e.feature", [item("E.php:1")], create_def_ids=["E.php:1@1"], t_vis=150)
    e_fn = fn("E.php:1", "#[Given('e')]\npublic function e(): void\n{\n}", ["e"], 1)
    for t, bad in ((149, False), (150, True), (None, True)):
        probs = lc.check(QUERY, msgs("..."), prov(["e"]), ctx([QUERY, dict(ex, t_vis=t)], [e_fn]))
        assert any("e: t_vis" in p for p in probs) is bad, (t, probs)


def test_test_infrastructure_is_counted_per_path_and_name_for_train_and_dev_only():
    """A (path, name) pair counts when the name is a token of the prompt AND its file is shown: a target file of the
    record, or the file of a ④ function (bare names are mostly noise)."""
    shown = "public function iSeeIt(): void\n{\n    $this->sharedStorage->lastOrderNumberFor('x');\n}"
    seen = fn("Setup/Order.php:5", "#[Given('o')]\npublic function o(): void\n{\n}", ["o"], 5)
    c = ctx([QUERY], [seen])
    c.infra = {"Sylius/Sylius": {("Setup/Order.php", "lastOrderNumberFor"), ("Other/Ctx.php", "lastOrderNumberFor"),
                                 ("Ui/Wish.php", "sharedStorage"), ("Setup/Order.php", "neverShown")}}
    counts = Counter()
    assert lc.check(QUERY, msgs(shown), prov(knowledge=["Setup/Order.php:5"]), c, counts) == []
    assert counts["test_infra_name"] == 2            # Setup/Order.php via ④(b), Ui/Wish.php as the target file
    counts = Counter()
    lc.check(dict(QUERY, partition="test"), msgs(shown), prov(knowledge=["Setup/Order.php:5"]), c, counts)
    assert counts["test_infra_name"] == 0


def test_short_answer_slices_are_refused():
    counts = Counter()
    body = lc.body_key(NEW)                                  # the answer shown without its annotation line
    probs = lc.check(QUERY, msgs(body), prov(), ctx([QUERY]), counts)
    assert any("short answer slice of Ui/Wish.php:10" in p for p in probs)
    assert "short_answer_slice" not in counts
    counts = Counter()
    probs = lc.check(QUERY, msgs("unrelated"), prov(), ctx([QUERY]), counts)
    assert not any("short" in p for p in probs)
    assert "short_answer_slice" not in counts


def test_short_answer_slice_is_not_a_second_problem_when_the_full_code_appears():
    probs = lc.check(QUERY, msgs(PHP_NEW), prov(), ctx([QUERY]))
    assert any("answer code of Ui/Wish.php:10" in p for p in probs)
    assert not any("short answer slice" in p for p in probs)


def test_functions_without_code_are_only_checked_by_id():
    stripped = {k: v for k, v in fn("T.php:1", "#[When('t')]\npublic function t() { }", ["t"]).items() if k != "code"}
    probs = lc.check(QUERY, msgs("..."), prov(knowledge=["T.php:1"]), ctx([QUERY], [stripped], hidden=["T.php:1"]))
    assert probs == ["q knowledge T.php:1: hidden for partition train"]


# The text check against the partition's hidden functions (laptop audit; fixed 25/09)


def test_hidden_answer_code_in_the_text_is_found_unless_a_visible_function_holds_the_same_body():
    body = "public function checkout(): void { $this->cart->checkout(); $this->page->open(); }"
    hidden = fn("Ui/Pay.php:1", body)                       # a test answer, hidden for train
    copy = fn("Api/Pay.php:1", body)                        # a visible byte-identical copy in another file
    other = fn("Ui/Ship.php:1", body.replace("checkout", "ship"))
    short = fn("Ui/Short.php:1", "public function a(): void { }")          # under 60 chars: no evidence
    cat = {f["fn_id"]: f for f in (hidden, copy, other, short)}
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": cat}, hidden={}, language={"Sylius/Sylius": "php"},
                     neighbours={})
    q, hide = rec("q"), {hidden["fn_id"], other["fn_id"], short["fn_id"]}
    shows = [{"role": "user", "content": f"Given x\n{body}\n{other['code']}\n  {short['code']}"}]
    assert lc.hidden_code(q, shows, ctx, hide) == [other["fn_id"]]
    assert lc.hidden_code(q, shows, ctx, hide | {copy["fn_id"]}) == [copy["fn_id"], hidden["fn_id"], other["fn_id"]]
    assert lc.hidden_code(q, shows, ctx, set()) == []                        # a test query hides nothing
    # a pool record's answer at its own commit (③ shows it) holds the other function's body: no evidence either
    ctx.pool = {"e": rec("e", create=[item("Ui/Old.php:1", other["code"])])}
    ctx._hidden_slices.clear()
    assert lc.hidden_code(q, shows, ctx, hide) == []
    ctx.pool = {"e": rec("e", create=[item(other["fn_id"], other["code"])])}   # unless its function is the hidden one
    ctx._hidden_slices.clear()
    assert lc.hidden_code(q, shows, ctx, hide) == [other["fn_id"]]


def test_a_hidden_body_without_its_declaration_is_no_evidence():
    """Real v0: an example's pre-T body shares 200-char boilerplate slices with two dev answers' current code, without
    their signatures; the leak rule needs the declaration too."""
    steps = "\n".join(f"    $this->step{i}->run('{i}');" for i in range(20))
    code = "public function longOne(): void\n{\n" + steps + "\n}"
    long = fn("Ui/Long.php:1", code)
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": {long["fn_id"]: long}}, hidden={},
                     language={"Sylius/Sylius": "php"}, neighbours={})
    body = "\n".join(code.splitlines()[1:])
    assert lc.hidden_code(rec("q"), [{"role": "user", "content": body}], ctx, {long["fn_id"]}) == []
    shown = [{"role": "user", "content": "public function longOne(): void {\n" + body}]
    assert lc.hidden_code(rec("q"), shown, ctx, {long["fn_id"]}) == [long["fn_id"]]


def test_a_hidden_ruby_step_is_declared_by_its_annotation_line():
    """Ruby step definitions have no method name: the annotation line is the declaration mark."""
    code = "Given(/^the patient has a prescription for (.+)$/) do |drug|\n" + \
        "  prescriptions.create!(drug: Drug.find_by!(name: drug), dose_amount: 5, dose_unit: 'mg', route: 'PO')\nend"
    step = {**fn("features/step_definitions/rx.rb:1", code, ["the patient has a prescription for (.+)"]),
            "first_line": 1}
    step["annotations"][0]["line"] = 1
    ctx = lc.Context(pool={}, catalog={"R": {step["fn_id"]: step}}, hidden={}, language={"R": "ruby"},
                     neighbours={})
    q = {**rec("q"), "repo": "R"}
    assert lc.hidden_code(q, [{"role": "user", "content": code}], ctx, {step["fn_id"]}) == [step["fn_id"]]
    body = [{"role": "user", "content": "\n".join(code.splitlines()[1:])}]
    assert lc.hidden_code(q, body, ctx, {step["fn_id"]}) == []


def test_a_declaration_a_visible_function_also_carries_is_no_evidence():
    """Api and Ui contexts declare the same step method: a shown visible twin's signature does not make a shared
    boilerplate slice of the hidden one a leak."""
    lines = [f"    $this->step{i}->run('{i}');" for i in range(20)]
    hidden = fn("Ui/Shop.php:1", "public function iSeeIt(): void\n{\n" + "\n".join(lines) + "\n}")
    twin = fn("Api/Shop.php:1", "public function iSeeIt(): void\n{\n    $this->client->get('/shop');\n}")
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": {f["fn_id"]: f for f in (hidden, twin)}}, hidden={},
                     language={"Sylius/Sylius": "php"}, neighbours={})
    shown = [{"role": "user", "content": twin["code"] + "\n" + "\n".join(lines)}]
    assert lc.hidden_code(rec("q"), shown, ctx, {hidden["fn_id"]}) == []


# After the 11 positive-control misses (25/09): spans compose never renders are no visible code, the anchored
# slice is evidence of its own, the declaration rule is kept; a test below pins the lambda declaration mark


@pytest.mark.parametrize("flag", ["dead", "contains_fn", "catch_all"])
def test_code_held_only_by_a_span_compose_never_renders_is_still_evidence(flag):
    """Never-rendered spans (gpconnect StructuredMultipleSteps.cs:187, a commented method whose span swallows the methods
    after it): a dead, contains_fn or catch_all entry is never shown (snapshot_reason), so the hidden
    method's code it also holds is still the hidden method's own. Here the span's own declaration is the hidden
    method's, so the declaration rule drops that mark and the anchored slice finds it."""
    steps = "\n".join(f"    $this->step{i}->run('{i}');" for i in range(12))
    hidden = fn("Steps/Multi.php:302", "public function iCheckTheBundle(): void\n{\n" + steps + "\n}")
    span = fn("Steps/Multi.php:187", "// public function old(): void\n// {\n" + hidden["code"] + "\n")
    span[flag] = ["Steps/Multi.php:302"] if flag == "contains_fn" else True
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": {f["fn_id"]: f for f in (hidden, span)}}, hidden={},
                     language={"Sylius/Sylius": "php"}, neighbours={})
    shown = [{"role": "user", "content": "Given x\n" + hidden["code"]}]
    assert lc.hidden_code(rec("q"), shown, ctx, {hidden["fn_id"]}) == [hidden["fn_id"]]
    span[flag] = False if flag != "contains_fn" else []                      # a rendered span: visible code again
    ctx._hidden_slices.clear()
    assert lc.hidden_code(rec("q"), shown, ctx, {hidden["fn_id"]}) == []


@pytest.mark.parametrize("flag", ["dead", "contains_fn", "catch_all"])
def test_the_never_rendered_rule_alone_gives_the_slice_rule_back_its_evidence(flag):
    """The never-rendered rule without the anchored slice, as on the real gpconnect span (its own declaration is the
    first method after the commented one, not the hidden method's): the hidden method's signature line and a later
    aligned slice, without its anchored slice, are evidence once the never-rendered span no longer counts as visible code."""
    steps = "\n".join(f"    $this->step{i}->run('{i}');" for i in range(20))
    hidden = fn("Steps/Multi.php:302", "public function iCheckTheBundle(): void\n{\n" + steps + "\n}")
    first = fn("Steps/Multi.php:190", "public function iCheckTheFirst(): void\n{\n    $this->first->run();\n}")
    span = fn("Steps/Multi.php:187", "// public function old(): void\n// {\n// }\n" + first["code"] + "\n" +
              hidden["code"] + "\n")
    span[flag] = [first["fn_id"], hidden["fn_id"]] if flag == "contains_fn" else True
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": {f["fn_id"]: f for f in (hidden, first, span)}}, hidden={},
                     language={"Sylius/Sylius": "php"}, neighbours={})
    assert lc.signature_line(span, "php") == "public function iCheckTheFirst(): void"          # not the declaration rule's case
    later = hidden["code"].replace("step0", "stepA")                  # breaks the anchored slice, keeps slice 2
    assert lc.normalize_ws(hidden["code"])[:lc.SLICE] not in lc.normalize_ws(later)
    shown = [{"role": "user", "content": "Given x\n" + later}]
    assert lc.hidden_code(rec("q"), shown, ctx, {hidden["fn_id"]}) == [hidden["fn_id"]]
    span[flag] = False if flag != "contains_fn" else []                      # a rendered span: visible code again
    ctx._hidden_slices.clear()
    assert lc.hidden_code(rec("q"), shown, ctx, {hidden["fn_id"]}) == []


def test_a_hidden_twin_is_caught_by_its_anchored_slice():
    """The anchored slice (Sylius Api/Ui pairs): a visible twin carries the same signature line, so the declaration mark is
    no evidence; the hidden method's own code from its signature line on (its first 200 chars, at least 60,
    held by no visible code) is."""
    lines = [f"    $this->step{i}->run('{i}');" for i in range(20)]
    hidden = fn("Ui/Shop.php:1", "#[Then('I see it')]\npublic function iSeeIt(): void\n{\n" + "\n".join(lines) + "\n}",
                ["I see it"])
    twin = fn("Api/Shop.php:1", "public function iSeeIt(): void\n{\n    $this->client->get('/shop');\n}")
    short = fn("Ui/Short.php:1", "public function iSee(): void\n{\n    $this->a->b();\n}")      # under 60 chars
    short_twin = fn("Api/Short.php:1", "public function iSee(): void\n{\n    $this->c->d();\n}")
    cat = {f["fn_id"]: f for f in (hidden, twin, short, short_twin)}
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": cat}, hidden={}, language={"Sylius/Sylius": "php"},
                     neighbours={})
    hide = {hidden["fn_id"], short["fn_id"]}
    whole = [{"role": "user", "content": "Given x\n" + hidden["code"] + "\n" + short["code"]}]
    assert lc.hidden_code(rec("q"), whole, ctx, hide) == [hidden["fn_id"]]
    moved = hidden["code"].replace("#[Then('I see it')]", "#[Then('I see it again')]")   # anchored at the signature
    assert lc.hidden_code(rec("q"), [{"role": "user", "content": moved}], ctx, hide) == [hidden["fn_id"]]
    shared = [{"role": "user", "content": twin["code"] + "\n" + "\n".join(lines)}]     # the declaration rule kept
    assert lc.hidden_code(rec("q"), shared, ctx, hide) == []


def test_a_hidden_java_lambda_step_is_declared_by_its_annotation_line():
    """Annotation lines are the declaration mark of every step without a method name, not of Ruby only; the 19
    fineract lambda dev answers depend on it."""
    ctx = lc.Context(pool={}, catalog={"F": {JAVA_LAMBDA["fn_id"]: JAVA_LAMBDA}}, hidden={}, language={"F": "java"},
                     neighbours={})
    q = {**rec("q"), "repo": "F"}
    code = JAVA_LAMBDA["code"]
    assert lc.hidden_code(q, [{"role": "user", "content": "Given x\n" + code}], ctx, {JAVA_LAMBDA["fn_id"]}) == [
        JAVA_LAMBDA["fn_id"]]
    body = [{"role": "user", "content": "\n".join(code.splitlines()[1:])}]
    assert lc.hidden_code(q, body, ctx, {JAVA_LAMBDA["fn_id"]}) == []


def test_an_annotation_line_a_visible_step_also_carries_is_no_evidence():
    """The declaration rule for steps without a method name (a Ruby block): two steps with the same annotation line;
    the shown
    visible one's line does not make a slice of the hidden one's body a leak (this half of the rule was not pinned
    before)."""
    line = "Given(/^the patient has a prescription for (.+)$/) do |drug|"
    body = "\n".join(f"  prescriptions.create!(drug: Drug.find_by!(name: drug), dose: {i})" for i in range(12))
    hidden = {**fn("features/step_definitions/rx.rb:1", line + "\n" + body + "\nend",
                   ["the patient has a prescription for (.+)"]), "first_line": 1}
    visible = {**fn("features/step_definitions/old_rx.rb:1", line + "\n  Prescription.make(drug)\nend",
                    ["the patient has a prescription for (.+)"]), "first_line": 1}
    for f in (hidden, visible):
        f["annotations"][0]["line"] = 1
    ctx = lc.Context(pool={}, catalog={"R": {f["fn_id"]: f for f in (hidden, visible)}}, hidden={},
                     language={"R": "ruby"}, neighbours={})
    q = {**rec("q"), "repo": "R"}
    shown = [{"role": "user", "content": visible["code"] + "\n" + body}]
    assert any(s in lc.normalize_ws(body) for s in lc.code_slices(hidden["code"])[1:])      # a slice of its own body
    assert lc.hidden_code(q, shown, ctx, {hidden["fn_id"]}) == []


# The anchored slice's width, and the never-rendered half of the declaration rule


def test_the_anchored_slice_is_exactly_the_first_200_chars_from_the_signature():
    """The published threshold rests on SLICE = 200. The hidden method shares its signature with a visible
    twin (no declaration mark) and its first 150 chars from the signature line too, then differs. A 120-char
    anchor would be visible code (never evidence); a 400-char anchor would miss a prompt that holds only its first 300
    chars."""
    common = [f"    $this->shared{i}->run();" for i in range(5)]
    hidden = fn("Ui/Shop.php:1", "public function iSeeIt(): void\n{\n" + "\n".join(
        common + [f"    $this->hidden{i}->check('{i}');" for i in range(12)]) + "\n}")
    twin = fn("Api/Shop.php:1", "public function iSeeIt(): void\n{\n" + "\n".join(
        common + [f"    $this->visible{i}->go();" for i in range(3)]) + "\n}")
    code = lc.normalize_ws(hidden["code"])
    assert code[:120] in lc.normalize_ws(twin["code"]) and code[:200] not in lc.normalize_ws(twin["code"])
    assert len(code) > 400 and lc.signature_line(hidden, "php") == lc.signature_line(twin, "php")
    ctx = lc.Context(pool={}, catalog={"Sylius/Sylius": {f["fn_id"]: f for f in (hidden, twin)}}, hidden={},
                     language={"Sylius/Sylius": "php"}, neighbours={})
    for shown in (hidden["code"], code[:300]):                 # the whole method; only its first 300 chars
        assert lc.hidden_code(rec("q"), [{"role": "user", "content": "Given x\n" + shown}], ctx,
                              {hidden["fn_id"]}) == [hidden["fn_id"]]


@pytest.mark.parametrize("flag", ["dead", "contains_fn", "catch_all"])
@pytest.mark.parametrize("language", ["php", "ruby"])
def test_a_declaration_only_a_never_rendered_function_carries_is_no_evidence(flag, language):
    """The declaration rule's other half: ④(a) may show the declaration of a catch-all (compose filters class members
    by id
    only), so a declaration mark a never-rendered function carries is no evidence either. The prompt holds the hidden
    function's declaration and a hidden-only slice, but not its anchored slice: nothing is flagged."""
    steps = "\n".join(f"    $this->step{i}->run('{i}');" for i in range(20))
    if language == "php":
        decl, other = "public function iCheckTheBundle(): void", "\n{\n    $this->other->run();\n}"
        hidden = fn("Steps/Multi.php:302", decl + "\n{\n" + steps + "\n}")
        span = fn("Steps/Old.php:10", decl + other)
    else:
        decl = "Given(/^the bundle is checked$/) do"
        body = "\n".join(f"  bundle.step{i}.run!('{i}')" for i in range(20))
        hidden = {**fn("features/step_definitions/b.rb:1", decl + "\n" + body + "\nend", ["the bundle is checked"]),
                  "first_line": 1}
        span = {**fn("features/step_definitions/old.rb:1", decl + "\n  Bundle.check\nend", ["the bundle is checked"]),
                "first_line": 1}
        for f in (hidden, span):
            f["annotations"][0]["line"] = 1
    span[flag] = [hidden["fn_id"]] if flag == "contains_fn" else True
    repo = "R"
    ctx = lc.Context(pool={}, catalog={repo: {f["fn_id"]: f for f in (hidden, span)}}, hidden={},
                     language={repo: language}, neighbours={})
    shown = hidden["code"].replace("step0", "stepX")              # the declaration and slice 2, no anchored slice
    q = {**rec("q"), "repo": repo}
    assert any(s in lc.normalize_ws(shown) for s in lc.code_slices(hidden["code"])[1:])
    assert lc.hidden_code(q, [{"role": "user", "content": shown}], ctx, {hidden["fn_id"]}) == []
    span[flag] = [] if flag == "contains_fn" else False            # a rendered twin: the same, by the rendered half
    ctx._hidden_slices.clear()
    assert lc.hidden_code(q, [{"role": "user", "content": shown}], ctx, {hidden["fn_id"]}) == []
    other = {**span, "fn_id": "Steps/Unrelated.php:1", "code": span["code"].replace(decl, decl.replace("heck", "ount"))}
    ctx.catalog[repo] = {f["fn_id"]: f for f in (hidden, other)}   # no carrier of the declaration: evidence
    ctx._hidden_slices.clear()
    assert lc.hidden_code(q, [{"role": "user", "content": shown}], ctx, {hidden["fn_id"]}) == [hidden["fn_id"]]
