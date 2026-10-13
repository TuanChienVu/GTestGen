import dataclasses
from collections import Counter

import pytest

from tests.phase3_fixture import SHOP
from tests.phase3c_fixture import JPATH, make_jprep, make_prep
from v2tools.controls import (Shown4, body_span, call_body, controls, frequent_call, generalize, param_names, shown4, stubs,
                               swap, top4_calls, visible_calls)
from v2tools.evaluate import ca


@pytest.fixture()
def prep(tmp_path):
    return make_prep(tmp_path)


def test_generalize_turns_values_and_numbers_into_groups():
    assert generalize('I add "apple" 3 times') == ('^I add "([^"]*)" (-?\\d+(?:[.,]\\d+)*) times$', 2)


def test_v3m_picks_the_most_frequent_visible_call_alphabetical_on_ties(prep):
    assert frequent_call(prep, SHOP) == ("cart", "add")
    blocks, fallback = stubs(prep, "v3m")
    assert fallback == 0 and "$this->cart->add();" in blocks[SHOP]


def test_v3all_makes_every_visible_call_self_calls_included(prep):
    seen = sorted(visible_calls(prep, SHOP))
    blocks, fallback = stubs(prep, "v3all")
    assert fallback == 0 and ("cart", "add") in seen and all(f"->{f}();" in blocks[SHOP] for _, f in seen)
    assert call_body("php", ("self", "helper")) == "    $this->helper();\n" and call_body("java", ("self", "go")) == "    go();\n"


def test_lazy_references_read_visible_functions_in_their_pre_t_form(prep, monkeypatch):
    """Rule 5: the lazy writer reads what 3A shows, here through state.pre_t_code: a visible function changed
    since t_vis is read as it was before t_vis (control: None, the HEAD form); one whose pre-T form cannot be located
    ("") is not counted."""
    import v2tools.controls as c
    add = next(fid for fid, f in prep.repo.fns.items() if "function iAddToCart(" in f["code"])
    assert ("cart", "add") in visible_calls(prep, SHOP)
    pre = "public function iAddToCart(string $product)\n{\n    $this->cart->remove($product);\n}"
    monkeypatch.setattr(c, "pre_t_code", lambda ctx, fid, t: pre if fid == add else None)
    seen = visible_calls(prep, SHOP)
    assert ("cart", "remove") in seen and ("cart", "add") not in seen
    monkeypatch.setattr(c, "pre_t_code", lambda ctx, fid, t: "" if fid == add else None)
    assert not {("cart", "add"), ("cart", "remove")} & set(visible_calls(prep, SHOP))


def test_lazy_references_leave_out_what_3a_keeps_out_of_block_4(prep):
    """As in prompt building, a function the composer's leak rules drop, or one using a masked member, is not read."""
    add = next(fid for fid, f in prep.repo.fns.items() if "function iAddToCart(" in f["code"])
    assert ("cart", "add") not in visible_calls(prep, SHOP, Shown4(dropped=frozenset({add})))
    assert ("cart", "add") not in visible_calls(prep, SHOP, Shown4(masked=frozenset({(SHOP, "cart")})))
    blocks, _ = stubs(prep, "v3m", Shown4(dropped=frozenset({add})))
    assert "$this->cart->add();" not in blocks[SHOP]


def test_shown4_reads_the_composer(prep):
    from v2tools.compose import Composer
    comp = Composer({prep.repo.name: prep.repo.ctx}, {prep.repo.name: prep.repo.fns}, {}, {prep.record["id"]: prep.record})
    k = shown4(comp, prep.record)
    assert k.masked == frozenset() and ("cart", "add") in visible_calls(prep, SHOP, k)


def test_v3c_reads_the_members_as_t_view_shows_them(prep, monkeypatch):
    """Rule 5: the lazy writer reads what 3A shows, here through state.t_view: a file new at t_vis (t_view
    None) has no injected service to call."""
    import v2tools.controls as c
    monkeypatch.setattr(c, "t_view", lambda *a, **k: None)
    assert c.first_service_call(prep, SHOP) is None


def test_v3c_falls_back_when_no_injected_service_resolves(prep):
    blocks, fallback = stubs(prep, "v3c")
    assert fallback == 1 and "$x = 1;" in blocks[SHOP]


def test_controls_report_bv_and_ca(prep):
    res = controls(prep)
    got = {k: (s.bv, ca(prep, s)[1]) for k, (s, _, _) in res.items()}
    assert got["gold"] == (1, 1) and got["stub-v3"] == (1, 0) and got["stub-v3m"] == (1, 0) and got["stub-v3c"] == (0, 0)
    assert got["stub-v3all"][0] == 1
    assert "swap" not in got


def test_body_span_and_param_names():
    text = "#[Then('x')]\npublic function t(string $a, int $b): void\n{\n    $this->go($a);\n}"
    a, b = body_span("php", text)
    assert text[a:b] == "{\n    $this->go($a);\n}" and param_names("php", text) == ["a", "b"]
    lam = 'Then("^x$", (String s) -> {\n    go(s);\n});'
    a, b = body_span("java", lam)
    assert lam[a:b] == "{\n    go(s);\n}" and param_names("java", lam) == ["s"]


def test_swap_takes_a_donor_body_and_renames_parameters(prep):
    pool = {SHOP: [("other", "#[Then('x :v')]\npublic function other(string $v): void\n{\n    $this->cart->add($v);\n}")]}
    blocks = swap(prep, pool)
    assert "$this->cart->add($total);" in blocks[SHOP] and "I should see the total :total" in blocks[SHOP]
    res = controls(prep, pool)
    s = res["swap"][0]
    assert s.bv == 1 and ca(prep, s) == (True, 0)
    assert swap(prep, {}) is None


def test_controls_gold_passes_stub_v1_v2_fail_v3_passes(prep):
    res = {k: v[0] for k, v in controls(prep).items()}
    assert res["gold"].bv == 1 and all(res["gold"].conds.values())
    assert res["stub-v1"].bv == 0 and not res["stub-v1"].conds["grounded"]
    assert res["stub-v2"].bv == 0 and [c for c, v in res["stub-v2"].conds.items() if not v] == ["grounded"]
    assert res["stub-v3"].bv == 1


def test_ca_gold_agrees_with_itself_and_stub_v3_does_not(prep):
    res = {k: v[0] for k, v in controls(prep).items()}
    assert prep.gold().calls and ca(prep, res["gold"]) == (True, 1)
    assert ca(prep, res["stub-v3"]) == (True, 0)


# ----- The helper controls (expansion of helper calls) ----------------------------------------------------------------

D10_NAMES = {"gold-helpers", "gold-helpers-renamed", "stub-v3m-h", "stub-v3all-h"}


def test_controls_return_the_helper_controls(prep):
    assert D10_NAMES <= set(controls(prep))


def test_ruby_has_no_helper_controls():
    """Ruby has no helper expansion (no class; fixed 25/09/2026):
    none of the four helper controls is emitted for a Ruby record."""
    from types import SimpleNamespace
    from v2tools.controls import d10_controls
    assert d10_controls(SimpleNamespace(repo=SimpleNamespace(language="ruby"))) == {}


def test_gold_helpers_append_the_future_helpers_the_gold_reads(prep):
    """gold-helpers = the gold blocks plus, verbatim from the ER tree, every future helper the gold's expansion reads
    (computeTotal, private, written with the answer): it replaces the ER tree's copy and scores as gold."""
    from v2tools.evaluate import expanded_futures
    res = controls(prep)
    s, fallback, blocks = res["gold-helpers"]
    gold = res["gold"][2]
    helper = expanded_futures(prep, prep.gold())[SHOP][0]
    assert helper.name == "computeTotal" and helper.origin == SHOP
    assert blocks[SHOP] == gold[SHOP] + "\n\n" + helper.text and blocks[SHOP].count("function computeTotal(") == 1
    assert fallback == 0 and s.insert_status == "ok" and s.detail["replaced"] == 1
    assert (s.bv, ca(prep, s)) == (prep.gold().bv, ca(prep, prep.gold())) == (1, (True, 1))
    assert s.calls[s.binders['I should see the total "5"']] == prep.gold().calls[prep.gold().binders['I should see the total "5"']]


def test_gold_helpers_renamed_rename_every_helper_and_call_site(prep):
    s, fallback, blocks = controls(prep)["gold-helpers-renamed"]
    assert "computeTotal" not in blocks[SHOP] and "$this->probeHelper1($total)" in blocks[SHOP]
    assert blocks[SHOP].count("function probeHelper1(") == 1 and fallback == 0
    assert (s.bv, ca(prep, s)) == (prep.gold().bv, ca(prep, prep.gold())) == (1, (True, 1))
    assert s.calls[s.binders['I should see the total "5"']] == prep.gold().calls[prep.gold().binders['I should see the total "5"']]


def test_gold_helpers_equal_the_gold_when_it_reads_no_future_helper(prep, monkeypatch):
    """No future helper read: both controls are the gold answer itself, flagged by fallback 1."""
    import v2tools.controls as c
    monkeypatch.setattr(c, "expanded_futures", lambda p, s: {})
    res = controls(prep)
    for name in ("gold-helpers", "gold-helpers-renamed"):
        s, fallback, blocks = res[name]
        assert blocks == res["gold"][2] and fallback == 1 and (s.bv, ca(prep, s)) == (1, (True, 1))


def test_rename_calls_touches_only_self_calls_and_the_definition():
    from v2tools.controls import rename_calls
    php = ("public function t(): void\n{\n    $this->ComputeTotal(1);\n    self::computeTotal (2);\n    static::computetotal(3);\n"
           "    $this\n        ->computeTotal(4);\n    $this->cart->computeTotal(5);\n    $this->computeTotalX(6);\n"
           "    parent::computeTotal(7);\n    $this->computeTotal;\n}\n\nprivate function computeTotal(): int\n{\n    return 1;\n}")
    got = rename_calls("php", php, {"computeTotal": "probeHelper1"})
    assert got.count("probeHelper1") == 5 and "function probeHelper1()" in got and "$this->probeHelper1(1)" in got
    for kept in ("$this->cart->computeTotal(5)", "$this->computeTotalX(6)", "parent::computeTotal(7)", "$this->computeTotal;"):
        assert kept in got
    java = ("public void t() {\n    computeTotal(1);\n    this.computeTotal(2);\n    this . computeTotal (3);\n"
            "    cart.computeTotal(4);\n    new computeTotal(5);\n    xcomputeTotal(6);\n    computeTotalX(7);\n    super.computeTotal(8);\n}\n\n"
            "private String computeTotal(int n) {\n    return \"computeTotal\";\n}")
    got = rename_calls("java", java, {"computeTotal": "probeHelper1"})
    assert got.count("probeHelper1") == 4 and "private String probeHelper1(int n)" in got
    for kept in ("cart.computeTotal(4)", "new computeTotal(5)", "xcomputeTotal(6)", "computeTotalX(7)", "super.computeTotal(8)",
                 '"computeTotal"'):
        assert kept in got
    assert rename_calls("java", "a(); b();", {"a": "probeHelper1", "b": "probeHelper2"}) == "probeHelper1(); probeHelper2();"


@pytest.mark.parametrize("base", ["v3m", "v3all"])
def test_helper_stubs_score_as_their_stub(prep, base):
    """stub-v3m-h / stub-v3all-h: the step only calls one self-written helper holding the stub's body; it contributes
    exactly its own calls, so the binder's call set, BV and CA are the stub's."""
    res = controls(prep)
    (s0, f0, b0), (s1, f1, b1) = res[f"stub-{base}"], res[f"stub-{base}-h"]
    step = 'I should see the total "5"'
    assert b1[SHOP].count("function probeHelper(") == 1 and f1 == f0 == 0
    assert "$this->probeHelper();" in b1[SHOP] and b1[SHOP].count("$this->cart->add();") == 1
    assert s1.calls[s1.binders[step]] == s0.calls[s0.binders[step]]
    assert (s1.bv, ca(prep, s1)) == (s0.bv, ca(prep, s0))


def test_helper_stubs_fall_back_like_their_stub(prep, monkeypatch):
    """When v3m (v3all) falls back to a meaningless statement, the -h variant falls back the same way, with no helper."""
    import v2tools.controls as c
    from collections import Counter
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter())
    res = controls(prep)
    for base in ("v3m", "v3all"):
        (s0, f0, b0), (s1, f1, b1) = res[f"stub-{base}"], res[f"stub-{base}-h"]
        assert f0 == f1 == 1 and b1 == b0 and "probeHelper" not in b1[SHOP] and "$x = 1;" in b1[SHOP]
        assert (s1.bv, ca(prep, s1)) == (s0.bv, ca(prep, s0))


def test_import_lines_and_the_names_they_bind():
    """The top-level single-line imports of a file (a trait `use` in a class body is not one; a multi-line group use is
    left out), and the short name each binds (None: a group, wildcard or namespace import)."""
    from v2tools.controls import import_lines, import_name
    php = ("<?php\n\nnamespace A;\n\nuse X\\Clock;\nuse Y\\Foo as Bar;\nuse function Z\\f;\nuse G\\{\n    H,\n    I\n};\n\n"
           "abstract class B\n{\n    use SomeTrait;\n}\n")
    assert import_lines("php", php) == ["use X\\Clock;", "use Y\\Foo as Bar;", "use function Z\\f;"]
    assert [import_name("php", x) for x in import_lines("php", php)] == ["Clock", "Bar", "f"]
    assert import_name("php", "use G\\{H, I};") is None
    java = "package p;\n\nimport a.b.C;\nimport static a.b.C.m;\nimport a.b.*;\n\nclass K {\n}\n"
    assert [import_name("java", x) for x in import_lines("java", java)] == ["C", "m", None]
    cs = "namespace N\n{\n    using System.Linq;\n    using Foo = A.B;\n\n    class K\n    {\n    }\n}\n"
    assert import_lines("csharp", cs) == ["using System.Linq;", "using Foo = A.B;"]
    assert [import_name("csharp", x) for x in import_lines("csharp", cs)] == [None, "Foo"]


def test_needed_imports_skip_what_the_target_has_binds_or_the_helper_never_names():
    from v2tools.controls import needed_imports
    origin = ("<?php\n\nnamespace A;\n\nuse X\\Clock;\nuse Y\\Unused;\nuse W\\Dup;\nuse R\\Other;\nuse V\\Lower;\n\n"
              "abstract class B\n{\n}\n")
    helper = "protected function h(): void\n{\n    Clock::tick(Dup::A, Other::B);\n    $lower = $this->lower;\n}"
    have = ["use W\\Dup;", "use Q\\Other;"]
    assert needed_imports("php", origin, helper, have) == ["use X\\Clock;"]
    assert needed_imports("php", origin, helper.replace("Clock::tick", "\\clock::tick"), have) == []
    java = "package p;\n\nimport a.Cart;\nimport a.Other;\nimport a.*;\n\nclass K {\n}\n"
    assert needed_imports("java", java, "void h() {\n    new Cart();\n}", ["import a.*;"]) == ["import a.Cart;"]


# A variant of the PHP fixture: BaseContext gains, with the answer, the protected helper checkTotal, which
# calls the in-repo class Clock that only BaseContext imports; iShouldSeeTotal's body is given per test.
CLOCK = "<?php\n\nnamespace App\\Behat\\Service;\n\nfinal class Clock\n{\n    public static function check(string $t): void\n    {\n    }\n}\n"
TRAIT = ("<?php\n\nnamespace App\\Behat;\n\ntrait StorageTrait\n{\n    protected function keepTotal(string $total): void\n    {\n"
         "        SameNsClass::f($total);\n    }\n}\n")
SAME_NS = "<?php\n\nnamespace App\\Behat;\n\nfinal class SameNsClass\n{\n    public static function f(string $t): void\n    {\n    }\n}\n"


def anc_prep(tmp_path, body: list[str], partition: str = "test", hide_add: bool = False, trait: bool = False):
    """`trait`: ShopContext also uses StorageTrait (namespace App\\Behat, the parent of the target's), added with the
    answer, whose protected keepTotal calls SameNsClass::f() of its own namespace, without an import."""
    import subprocess
    from tests.phase3_fixture import BASE, SHOP_2020, SHOP_2022, SHOP_2024, T2020, T2022, T2024, _fn, _git, record
    from v2tools.evaluate import Repo, prepare
    from v2tools.state import RepoContext
    base = BASE.replace("namespace App\\Behat\\Context;\n", "namespace App\\Behat\\Context;\n\nuse App\\Behat\\Service\\Clock;\n") \
        .replace("    private function secret", "    protected function checkTotal(string $total): void\n    {\n"
                 "        Clock::check($total);\n    }\n\n    private function secret")
    shop = SHOP_2024.replace("        $this->checkout->assertTotal($this->computeTotal($total));\n", "".join(f"        {l}\n" for l in body))
    extra = {}
    if trait:
        shop = shop.replace("use App\\Page\\IndexPage;\n", "use App\\Page\\IndexPage;\nuse App\\Behat\\StorageTrait;\n") \
            .replace("final class ShopContext extends BaseContext\n{\n", "final class ShopContext extends BaseContext\n{\n    use StorageTrait;\n\n")
        extra = {"src/Behat/StorageTrait.php": TRAIT, "src/Behat/SameNsClass.php": SAME_NS}
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in [(T2020, {"composer.json": '{"autoload": {"psr-4": {"App\\\\Behat\\\\": "src/Behat/"}}}\n',
                                 "src/Behat/Context/BaseContext.php": BASE, "src/Behat/Service/Clock.php": CLOCK, SHOP: SHOP_2020}),
                        (T2022, {SHOP: SHOP_2022}), (T2024, {SHOP: shop, "src/Behat/Context/BaseContext.php": base, **extra})]:
        for path, text in files.items():
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp_path / "repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    head = _git(bare, "rev-parse", "HEAD").strip()
    lines = shop.splitlines()
    line = {n: next(i for i, l in enumerate(lines, 1) if f"function {n}(" in l) + 1 for n in ("iOpenTheShop", "iAddToCart", "iShouldSeeTotal")}
    rows = [_fn(line["iOpenTheShop"], "iOpenTheShop", "Given", "I open the shop", T2020, shop),
            _fn(line["iAddToCart"], "iAddToCart", "When", "I add :product to the cart", T2022, shop),
            _fn(line["iShouldSeeTotal"], "iShouldSeeTotal", "Then", "I should see the total :total", T2024, shop)]
    cat = {f["fn_id"]: f for f in rows}
    def_of = lambda text: next(a["def_id"] for f in cat.values() for a in f["annotations"] if text in a["pattern"])
    suite = {"s": [("I open the shop", (def_of("open the shop"),)), ('I should see the total "5"', (def_of("total"),)),
                   ('I add "apple" to the cart', (def_of("cart"),))]}
    forbidden = {"Acme/Shop": {"test": [rows[1]["fn_id"]]}} if hide_add else {}
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, RepoContext("Acme/Shop", "php", bare, head, cat),
                frozenset(), suite, forbidden)
    r = record(cat, partition)
    r["commit"] = head
    return prepare(r, repo, {"step_def_ids": [[def_of("open the shop")], [def_of("total")]]})


def test_gold_helpers_bring_the_imports_an_ancestor_helper_needs(tmp_path):
    """An ancestor's helper is copied with the import lines of its file that it needs (as at HEAD, with its imports):
    insertion merges them into the target file, so the helper's class resolves there and condition 5 holds as for
    gold."""
    p = anc_prep(tmp_path, ["$this->checkTotal($total);"])
    g, res = p.gold(), controls(p)
    assert g.bv == 1
    for name, fn in (("gold-helpers", "checkTotal"), ("gold-helpers-renamed", "probeHelper1")):
        s, fallback, blocks = res[name]
        assert blocks[SHOP].startswith("use App\\Behat\\Service\\Clock;\n") and f"protected function {fn}(" in blocks[SHOP]
        assert "use App\\Behat\\Service\\Clock;" in s.after[SHOP] and fallback == 0
        assert (s.bv, ca(p, s)) == (g.bv, ca(p, g)), name


def test_gold_helpers_import_a_class_of_the_helpers_own_namespace(tmp_path):
    """A helper from a file of another PHP namespace (StorageTrait, App\\Behat) names a class of its own
    namespace (SameNsClass) that needs no import there; the copy brings `use App\\Behat\\SameNsClass;`, so the class
    resolves in the target file (App\\Behat\\Context) and the helper's call counts as in gold."""
    p = anc_prep(tmp_path, ["$this->keepTotal($total);"], trait=True)
    g, res = p.gold(), controls(p)
    assert (g.bv, ca(p, g)) == (1, (True, 1)) and g.calls[g.binders['I should see the total "5"']] == {("SameNsClass", "f")}
    for name, fn in (("gold-helpers", "keepTotal"), ("gold-helpers-renamed", "probeHelper1")):
        s, _, blocks = res[name]
        assert blocks[SHOP].startswith("use App\\Behat\\SameNsClass;\n") and f"protected function {fn}(" in blocks[SHOP]
        assert blocks[SHOP].count("use ") == 1 and "use App\\Behat\\SameNsClass;" in s.after[SHOP]
        assert (s.bv, ca(p, s), s.calls[s.binders['I should see the total "5"']]) == (1, (True, 1), {("SameNsClass", "f")}), name


def test_renamed_helpers_drop_their_step_annotations(tmp_path):
    """A future helper that is also a step function of the ER tree (iAddToCart, hidden from a dev record): its renamed
    copy loses its step annotation, so it never binds the step a second time (condition 3); gold-helpers keeps it, as
    its copy replaces the ER tree's."""
    p = anc_prep(tmp_path, ["$this->checkout->assertTotal($total);", "$this->iAddToCart($total);"], "dev", hide_add=True)
    g, res = p.gold(), controls(p)
    assert (g.bv, ca(p, g)) == (1, (True, 1)) and ("cart", "add") in g.calls[g.binders['I should see the total "5"']]
    s, _, blocks = res["gold-helpers"]
    assert "#[When('I add :product to the cart')]" in blocks[SHOP] and (s.bv, ca(p, s)) == (1, (True, 1))
    s, _, blocks = res["gold-helpers-renamed"]
    assert "#[When(" not in blocks[SHOP] and "#[Then('I should see the total :total')]" in blocks[SHOP]
    assert "public function probeHelper1(string $product)" in blocks[SHOP] and "$this->probeHelper1($total);" in blocks[SHOP]
    assert (s.bv, ca(p, s)) == (1, (True, 1)), s.detail


def test_helper_definition_per_language():
    from v2tools.controls import helper_def, helper_name
    assert helper_def("php", "probeHelper", "    $this->a->b();\n") == "private function probeHelper(): void\n{\n    $this->a->b();\n}"
    assert helper_def("java", "probeHelper", "    a.b();\n") == "private void probeHelper() {\n    a.b();\n}"
    assert helper_def("csharp", "ProbeHelper", "    a.B();\n") == "private void ProbeHelper()\n{\n    a.B();\n}"
    assert [helper_name(lang) for lang in ("php", "java", "csharp")] == ["probeHelper", "probeHelper", "ProbeHelper"]
    assert helper_name("java", 2) == "probeHelper2" and helper_name("csharp", 1) == "ProbeHelper1"


# ----- the same on the Java twin of the fixture (tests/test_score_independent.py, jpair, plus a class chain):
# ShopSteps extends BaseSteps (field cart); the gold iShouldSeeTotal calls the future private helper computeTotal, which
# calls BaseSteps.baseNew, future too; the constructor added with them is future but never a helper. The
# fixture itself (make_jprep) lives in tests/phase3c_fixture.py so that tests/test_describe.py can share it too.


@pytest.fixture()
def jprep(tmp_path):
    return make_jprep(tmp_path)


def test_helper_controls_on_the_java_fixture(jprep):
    """Java: gold-helpers appends computeTotal (replacing the ER tree's) and BaseSteps.baseNew (an override), never the
    future constructor; gold-helpers-renamed names them probeHelper1, probeHelper2 at every definition and call site;
    both score as gold. stub-v3m-h and stub-v3all-h score as their stubs, with one helper each."""
    p = jprep
    step = 'I should see the total "5"'
    res = controls(p)
    g = p.gold()
    assert D10_NAMES <= set(res) and (g.bv, ca(p, g)) == (1, (True, 1))
    gold_calls = g.calls[g.binders[step]]
    assert gold_calls == {("cart", "assertTotal"), ("cart", "total")}
    s, fallback, blocks = res["gold-helpers"]
    assert "private String computeTotal(String total) {" in blocks[JPATH] and "protected void baseNew() {" in blocks[JPATH]
    assert "ShopSteps()" not in blocks[JPATH] and fallback == 0
    assert (s.bv, ca(p, s), s.calls[s.binders[step]]) == (1, (True, 1), gold_calls)
    s, fallback, blocks = res["gold-helpers-renamed"]
    assert "computeTotal" not in blocks[JPATH] and "baseNew" not in blocks[JPATH]
    assert "cart.assertTotal(probeHelper1(total));" in blocks[JPATH] and "private String probeHelper1(String total) {" in blocks[JPATH]
    assert "    probeHelper2();\n" in blocks[JPATH] and "protected void probeHelper2() {" in blocks[JPATH]
    assert (s.bv, ca(p, s), s.calls[s.binders[step]]) == (1, (True, 1), gold_calls)
    for base in ("v3m", "v3all"):
        (s0, f0, b0), (s1, f1, b1) = res[f"stub-{base}"], res[f"stub-{base}-h"]
        assert b1[JPATH].count("private void probeHelper() {") == 1 and "    probeHelper();\n" in b1[JPATH] and f0 == f1 == 0
        assert s1.calls[s1.binders[step]] == s0.calls[s0.binders[step]] == {("cart", "open")}
        assert (s1.bv, ca(p, s1)) == (s0.bv, ca(p, s0))


# ----- D11 (a), fixed 25/09/2026: stub-v3names, the third lazy reference (design specification at 3739dd6) ----

def test_v3names_adds_a_call_to_every_method_v_r_shows_to_each_v3all_body(prep):
    """D11 (a): stub-v3names is stub-v3all with, at the top of every stub step body, a call without arguments on the
    class itself to every METHOD name that V(r) shows for the file (evaluate.shown_methods: state.t_view, constructors
    and every non-method member left out), sorted; it never reads the step text. Scored like any control."""
    from v2tools.controls import shown_methods
    assert shown_methods(prep, SHOP) == ["helperOld", "iAddToCart", "iOpenTheShop"]
    (s3, f3, b3), (s, fallback, blocks) = controls(prep)["stub-v3all"], controls(prep)["stub-v3names"]
    names = "    $this->helperOld();\n    $this->iAddToCart();\n    $this->iOpenTheShop();\n"
    assert blocks[SHOP] == b3[SHOP].replace("): void\n{\n", "): void\n{\n" + names) and fallback == f3 == 0
    step = 'I should see the total "5"'
    assert s.calls[s.binders[step]] == s3.calls[s3.binders[step]] | {("self", "helperOld"), ("self", "iAddToCart"),
                                                                     ("self", "iOpenTheShop")}
    assert s.bv == 1 and ca(prep, s) == (True, 0)


def test_v3names_on_a_new_file_calls_what_its_ancestors_show(tmp_path):
    """D11 (a) on a target file NEW at t_vis (the prompt's class skeleton shows nothing): the names are the METHODS of
    the shown set of a new file (evaluate.shown_names: what t_view shows for each in-repo ancestor, less its constructor
    and the privates of files outside RepoContext.access): here BaseContext.helperOld, the own trait's private ownOld
    and the parent trait's parOld; never BaseContext.baseNew (added at t_vis), the private secret, nor the property
    sharedStorage."""
    from tests.phase3_fixture import SHOP_2024, T2020, T2024
    from tests.test_evaluate import BASE, OWN, PAR, base_text, member, trait_text, twin_prep, with_own_trait
    from v2tools.controls import shown_methods
    own = trait_text("OwnTrait", member("private", "ownOld", "o0"))
    par = trait_text("ParentTrait", member("protected", "parOld", "p0"))
    p = twin_prep(tmp_path, [(T2020, {BASE: base_text(), OWN: own, PAR: par}),
                             (T2024, {BASE: base_text(member("protected", "baseNew", "b1")), OWN: own, PAR: par,
                                      SHOP: with_own_trait(SHOP_2024)})])
    assert p.repo.ctx.is_new_file(SHOP, p.record["t_vis"])
    assert shown_methods(p, SHOP) == ["helperOld", "ownOld", "parOld"]
    s, fallback, blocks = controls(p)["stub-v3names"]
    assert "    $this->helperOld();\n    $this->ownOld();\n    $this->parOld();\n" in blocks[SHOP]


def test_v3names_on_java_and_not_for_ruby(jprep):
    """D11 (a): Java calls a bare `f();` (C#: `F();`, as named); Ruby has no method list in the prompt's class skeleton
    (3A's state.skeleton_parts is None for a Ruby file), so no stub-v3names row and ca_ref_v3names None."""
    from types import SimpleNamespace
    from v2tools.controls import V3NAMES_LANGUAGES, d11_controls
    s, fallback, blocks = controls(jprep)["stub-v3names"]
    want = "public void stepStub1(String p1) {\n    baseOld();\n    iOpenTheShop();\n    cart.open();\n"
    assert want in blocks[JPATH]
    assert V3NAMES_LANGUAGES == ("php", "java", "csharp")
    assert d11_controls(SimpleNamespace(repo=SimpleNamespace(language="ruby"))) == {}


# ----- D11 (d), fixed 26/09/2026 (§2.2b of the design specification): stub-v3top4, the fourth lazy reference ---

def top4_block(calls: str) -> str:
    return ("use Behat\\Step\\When;\n\n"  # the stubs' own import (design specification §2.2 at 8e4405a)
            "#[When('/^I should see the total \"([^\"]*)\"$/')]\npublic function stepStub1($p1): void\n{\n" + calls + "}")


def shop_prior(golds: dict):
    from v2tools.controls import DevPrior
    return DevPrior("Acme/Shop", "php", tuple(sorted((rid, frozenset(a)) for rid, a in golds.items())))


GOLDS = {"d1": {("checkout", "assertTotal"), ("Ghost", "f"), ("cart", "ADD")},
         "d2": {("checkout", "assertTotal"), ("self", "HELPEROLD"), ("nothere", "x")},
         "d3": {("BaseContext", "make"), ("self", "secret"), ("self", "iOpenTheShop")}}


def test_v3top4_fills_the_visible_calls_up_from_the_dev_prior(prep):
    """D11 (d), fixed 26/09/2026: stub-v3top4 keeps stub-v3all's skeleton and never reads the step text; every stub of a
    target file gets one body of at most 4 calls: the file's visible calls (visible_calls, self calls included) most
    frequent first, ties by sorted atom, then the dev prior (DevPrior.ranked: the atoms of the dev golds of the repo by
    the number of dev records holding them, ties by sorted atom; PHP names casefolded), a call already taken in any case
    skipped. Scored like any control, after stub-v3names. `checkout` is FUTURE for this record (promoted alongside its
    own answer, at t_vis), so `writable` excludes the prior's ("checkout", "assertTotal") and the 4th slot falls to
    BaseContext::make()."""
    s, fallback, blocks = controls(prep, priors={"Acme/Shop": shop_prior(GOLDS)})["stub-v3top4"]
    body = "    $this->cart->add();\n    $this->indexPage->open();\n    BaseContext::make();\n    $this->helperOld();\n"
    assert blocks[SHOP] == top4_block(body) and fallback == 0
    assert s.bv == 1 and ca(prep, s) == (True, 0)
    assert list(controls(prep, priors={}))[-2:] == ["stub-v3names", "stub-v3top4"]


def test_v3top4_ca_prec_equals_its_ca(tmp_path):
    """D11 (d), fixed 27/09/2026: ca_ref_top4 is stub-v3top4's CA; its body has at most 4 distinct calls none of which
    expands (visible names stay ("self", f)), so CA = 1 implies call_precision >= 1/4 = τ: CA_prec = CA. The
    gold here calls $this->cart->add() (established well before t_vis, never future), so top4_calls can reproduce it.
    call_precision is a local copy of describe.call_overlap (tests/test_evaluate.py); tests/test_describe.py pins
    the copy to it."""
    from tests.test_evaluate import call_precision
    p = anc_prep(tmp_path, ["$this->cart->add($total);"])
    s, _, blocks = controls(p, priors={"Acme/Shop": shop_prior(GOLDS)})["stub-v3top4"]
    ov = call_precision(p, s)
    assert ov["n_calls"] <= 4 and ca(p, s) == (True, 1) and ov["call_precision"] >= 0.25


def test_v3top4_prior_keeps_only_calls_the_record_can_write(prep):
    """D11 (d): a prior call enters only when the record can write it in the file: a call on the class itself to a
    METHOD the prompt's class skeleton shows (shown_methods; any case in PHP, written as shown), a call on an available
    member of the class (Scope.available) that is not FUTURE or masked (ctx.future(path, t_vis, masked), as
    visible_calls applies), a PHP static call on a class that resolves in the file (a project call, calls.php_calls);
    never Ghost::f (no such class), $this->nothere, the ancestor's private secret, nor $this->checkout->assertTotal()
    (promoted alongside this record's own answer, at t_vis: FUTURE). With no visible call (both visible step
    functions dropped by 3A, Shown4), the body is the prior's first 4 such calls."""
    fids = frozenset(fid for fid, f in prep.repo.fns.items() if "iShouldSeeTotal" not in f["code"])
    s, fallback, blocks = controls(prep, k=Shown4(dropped=fids), priors={"Acme/Shop": shop_prior(GOLDS)})["stub-v3top4"]
    body = "    BaseContext::make();\n    $this->cart->ADD();\n    $this->helperOld();\n    $this->iOpenTheShop();\n"
    assert blocks[SHOP] == top4_block(body) and fallback == 0 and ca(prep, s) == (True, 0)
    s, fallback, blocks = controls(prep, k=Shown4(dropped=fids), priors={})["stub-v3top4"]
    assert blocks[SHOP] == top4_block("    $x = 1;\n") and fallback == 1 and s.bv == 0


def test_v3top4_never_reads_a_dev_records_own_gold(tmp_path):
    """D11 (d): the prior of a DEV record leaves its own gold out (DevPrior.ranked(exclude=its id)); dev_prior builds
    the priors from the dev partition's prepared records (Prepared.gold: the CA call sets of their gold functions) and
    refuses any other partition (a test gold must never reach a prior). Here the differentiator is $this->cart->add()
    (established well before t_vis, so writable regardless of future status; both visible step functions are dropped
    so the prior is the only source): the dev record leaves it out of its own prior, the test record does not."""
    from tests.phase3c_fixture import make_prep
    from v2tools.controls import DevPrior, dev_prior
    (tmp_path / "d").mkdir()
    (tmp_path / "t").mkdir()
    dev, test = make_prep(tmp_path / "d", "dev"), make_prep(tmp_path / "t")
    rid = dev.record["id"]
    test.record = dict(test.record, id=rid.replace("total.feature", "total_test.feature"))  # ids are unique (Phase 2)
    want = DevPrior("Acme/Shop", "php", ((rid, frozenset({("checkout", "assertTotal")})),))
    assert dev_prior([dev]) == {"Acme/Shop": want}
    with pytest.raises(ValueError):
        dev_prior([test])
    fids = frozenset(fid for fid, f in dev.repo.fns.items() if "iShouldSeeTotal" not in f["code"])
    priors = {"Acme/Shop": shop_prior({rid: {("cart", "add")}, "other": {("self", "helperOld")}})}
    assert controls(dev, k=Shown4(dropped=fids), priors=priors)["stub-v3top4"][2][SHOP] == top4_block(
        "    $this->helperOld();\n")
    assert controls(test, k=Shown4(dropped=fids), priors=priors)["stub-v3top4"][2][SHOP] == top4_block(
        "    $this->cart->add();\n    $this->helperOld();\n")


def test_devprior_excludes_every_dev_record_of_the_same_scenario(tmp_path):
    """A dev record can read its own gold through a SIBLING SUITE of the SAME scenario
    (same repo, feature_path, scenario_line, another `suite`): DevPrior.counts/.ranked exclude every dev record of
    that scenario from the record's own prior, not only the record itself (no D11 (d) prior leakage)."""
    from v2tools.controls import DevPrior
    rid = "Acme/Shop|features/total.feature|3|s1"
    sib = "Acme/Shop|features/total.feature|3|s2"  # same scenario (repo, feature_path, scenario_line), another suite
    other = "Acme/Shop|features/other.feature|9|s1"  # a different scenario: still counted
    prior = DevPrior("Acme/Shop", "php", ((other, frozenset({("checkout", "assertTotal")})),
                                          (rid, frozenset({("cart", "add")})), (sib, frozenset({("cart", "add")}))))
    assert prior.counts(rid) == {("checkout", "asserttotal"): 1}  # PHP: counts key on the casefolded method
    assert prior.ranked(rid) == [("checkout", "assertTotal")]
    assert prior.counts(None) == {("cart", "add"): 2, ("checkout", "asserttotal"): 1}
    assert prior.ranked(None) == [("cart", "add"), ("checkout", "assertTotal")]


def test_v3top4_on_a_new_file_ranks_what_its_ancestors_show_by_the_dev_prior(tmp_path):
    """D11 (d), stub-v3top4 on a target file NEW at t_vis (no visible step function): the METHODS of the new file's
    shown set (shown_methods, as stub-v3names) ranked by the dev prior of ("self", name), ties and names the prior never
    holds by name, top 4; no other prior call (here $this->sharedStorage->get(), the most frequent)."""
    from tests.phase3_fixture import SHOP_2024, T2020, T2024
    from tests.test_evaluate import BASE, OWN, PAR, base_text, member, trait_text, twin_prep, with_own_trait
    from v2tools.controls import shown_methods
    own = trait_text("OwnTrait", member("private", "ownOld", "o0"))
    par = trait_text("ParentTrait", member("protected", "parOld", "p0"))
    extra = (member("protected", "aOld", "a"), member("protected", "zOld", "z"))
    p = twin_prep(tmp_path, [(T2020, {BASE: base_text(*extra), OWN: own, PAR: par}),
                             (T2024, {BASE: base_text(*extra, member("protected", "baseNew", "b1")), OWN: own, PAR: par,
                                      SHOP: with_own_trait(SHOP_2024)})])
    assert p.repo.ctx.is_new_file(SHOP, p.record["t_vis"]) and not visible_calls(p, SHOP)
    assert shown_methods(p, SHOP) == ["aOld", "helperOld", "ownOld", "parOld", "zOld"]
    get = ("sharedStorage", "get")
    golds = {"d1": {("self", "ZOLD"), ("self", "parOld"), get}, "d2": {("self", "zOld"), ("self", "parold"), get},
             "d3": {("self", "zold"), ("self", "ownOld"), get, ("self", "baseNew")}}
    s, fallback, blocks = controls(p, priors={"Acme/Shop": shop_prior(golds)})["stub-v3top4"]
    body = "    $this->zOld();\n    $this->parOld();\n    $this->ownOld();\n    $this->aOld();\n"
    assert blocks[SHOP] == top4_block(body) and fallback == 0


def future_property_prep(tmp_path, partition: str = "test"):
    """As phase3c_fixture.make_prep, but ctx (RepoContext) is pinned to ONE MORE commit after the record's own t_vis
    (T2024): that later commit adds `private LoggerService $logger` to ShopContext's constructor, a member
    symbols.scope reads straight from the file's CURRENT text into Scope.available, though the record's t_vis never
    saw it (top4_calls.writable must exclude a future/masked receiver, as visible_calls does)."""
    from tests.phase3_fixture import SHOP, SHOP_2024, T2024, _git, catalog, make_repo, record
    from v2tools.evaluate import Repo, prepare
    from v2tools.state import RepoContext
    bare, _ = make_repo(tmp_path)
    work = tmp_path / "work"
    future_shop = SHOP_2024.replace("        private Checkout $checkout,\n",
                                    "        private Checkout $checkout,\n        private LoggerService $logger,\n")
    (work / SHOP).write_text(future_shop)
    _git(work, "commit", "-q", "-am", "future logger property", date=T2024 + 30 * 86400)
    _git(work, "push", "-q", str(bare), "main")
    head = _git(bare, "rev-parse", "HEAD").strip()
    cat = catalog()
    def_of = lambda text: next(a["def_id"] for f in cat.values() for a in f["annotations"] if text in a["pattern"])
    suite = {"s": [("I open the shop", (def_of("open the shop"),)), ('I should see the total "5"', (def_of("total"),)),
                   ('I add "apple" to the cart', (def_of("cart"),))]}
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), suite)
    r = record(cat, partition)
    r["commit"] = head
    return prepare(r, repo, {"step_def_ids": [[def_of("open the shop")], [def_of("total")]]})


def test_top4_calls_writable_excludes_a_future_or_masked_receiver(tmp_path):
    """top4_calls.writable used to apply the plain "m in Scope.available" test to a prior
    call's receiver; it must apply the SAME future/masked filter visible_calls applies (ctx.future(path, t_vis,
    masked)): a member the class gains AFTER t_vis is not writable, even though symbols.scope reads it straight from
    the file's current text into `available` (the lazy writer reads only what 3A shows, rule 5)."""
    from v2tools.symbols import scope as sym_scope
    p = future_property_prep(tmp_path)
    text = p.er_files.get(SHOP, p.repo.ctx.text(SHOP))
    sc = sym_scope(p.repo.ctx, SHOP, text, text, p.er_files)
    assert "logger" in sc.available  # sanity: the future property IS in Scope.available (the bug's premise)
    fids = frozenset(fid for fid, f in p.repo.fns.items() if "iShouldSeeTotal" not in f["code"])
    prior = shop_prior({"d1": {("logger", "log"), ("cart", "add")}})  # cart: established well before t_vis, not future
    calls = top4_calls(p, SHOP, Shown4(dropped=fids), prior)
    assert ("logger", "log") not in calls
    assert ("cart", "add") in calls


def test_v3top4_on_java_and_not_for_ruby_and_only_with_priors(jprep):
    """D11 (d): in stub-v3top4, Java writes `field.f();` and a bare `f();`; Ruby has no stub-v3top4 row (ca_ref_top4
    None), and neither has a record scored without priors (controls(priors=None), the default)."""
    from types import SimpleNamespace
    from v2tools.controls import DevPrior, d11_controls
    atoms = frozenset({("cart", "assertTotal"), ("self", "baseOld"), ("other", "x")})
    prior = DevPrior("Acme/Java", "java", (("d", atoms),))
    s, fallback, blocks = controls(jprep, priors={"Acme/Java": prior})["stub-v3top4"]
    want = "public void stepStub1(String p1) {\n    cart.open();\n    cart.assertTotal();\n    baseOld();\n}"
    assert want in blocks[JPATH]
    assert s.bv == 1 and ca(jprep, s) == (True, 1) and fallback == 0
    assert "stub-v3top4" not in controls(jprep)
    assert d11_controls(SimpleNamespace(repo=SimpleNamespace(language="ruby")), priors={}) == {}


def test_top4_calls_orders_visible_calls_by_frequency_not_alphabetically(prep, monkeypatch):
    """Closes a test gap found by mutation testing: top4_calls' own visible-calls ordering is by
    FREQUENCY first, alphabetical only to break a tie (§2.2b of the design specification): "zzz" is alphabetically last
    but the most frequent visible call, so it must come first."""
    import v2tools.controls as c
    from collections import Counter
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter({("aaa", "z"): 1, ("zzz", "a"): 5}))
    calls = c.top4_calls(prep, SHOP)
    assert calls[:2] == [("zzz", "a"), ("aaa", "z")]


def test_top4_calls_does_not_repeat_a_call_the_prior_also_ranks(prep, monkeypatch):
    """Closes a test gap found by mutation testing: a prior call already taken (as visible_calls
    gave it) is skipped, not counted twice against the TOP4 budget."""
    import v2tools.controls as c
    from collections import Counter
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter({("cart", "add"): 5}))
    prior = shop_prior({"d1": {("cart", "add"), ("indexPage", "open")}})  # indexPage: established well before t_vis
    calls = c.top4_calls(prep, SHOP, Shown4(), prior)
    assert calls.count(("cart", "add")) == 1
    assert ("indexPage", "open") in calls


def test_the_gold_control_is_the_gold_answer_scored_as_an_output(tmp_path):
    """D11 (e), fixed 26/09/2026: the gold control is the gold answer scored exactly like an output
    (Prepared.gold_output): its call set is the shell's own, while the CA reference (Prepared.gold) also holds
    ("self", shell); its BV and CA stay 1."""
    from tests.test_evaluate import shell_prep
    p = shell_prep(tmp_path)
    s, fallback, blocks = controls(p)["gold"]
    assert s is p.gold_output() and s.calls[(SHOP, "iaddtocart")] == {("cart", "add")} and fallback == 0
    assert (s.bv, ca(p, s)) == (1, (True, 1)) and ("self", "iAddToCart") in p.gold().calls[(SHOP, "iaddtocart")]


@pytest.mark.parametrize("seen, names, want", [(False, False, 1), (False, True, 0), (True, False, 0), (True, True, 0)])
def test_v3names_falls_back_only_with_no_visible_call_and_no_shown_method(prep, monkeypatch, seen, names, want):
    """D11 (a): a stub-v3names function counts as a fallback only when
    stub-v3all's body fell back (no visible call) AND V(r) shows no method; then its body is only the meaningless
    statement. A stub that gets a name (its calls, then stub-v3all's fallen-back statement), or a visible call, is no
    fallback (fallback += fell and not names)."""
    import v2tools.controls as c
    from collections import Counter
    if not seen:
        monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter())
    if not names:
        monkeypatch.setattr(c, "shown_methods", lambda *a, **k: [])
    blocks, fallback = stubs(prep, "v3names")
    assert fallback == want and ("$this->helperOld();" in blocks[SHOP]) == names
    assert ("$this->" not in blocks[SHOP]) == bool(want) and ("$x = 1;" in blocks[SHOP]) == (not seen)



def test_stubs_open_with_the_import_of_their_step_annotation(prep, jprep):
    """As §2.2 of the design specification states at 8e4405a (fixed 27/09/2026), a stub is written
    as a valid output, so each of its blocks opens with the import of the annotation it uses
    (`When`), and binds whatever the target file imports: the Java twin's ShopSteps imports Given
    and Then only. Without that line the reader, which resolves names as Java does, reads no step
    and the stub's BV falls to 0 for a reason that is not laziness (the pin is not blind)."""
    from v2tools.evaluate import score_blocks
    for p, path, head in ((prep, SHOP, "use Behat\\Step\\When;\n\n"), (jprep, JPATH, "import io.cucumber.java.en.When;\n\n")):
        for version in ("v1", "v3", "v3m", "v3all"):
            blocks, _ = stubs(p, version)
            assert blocks[path].startswith(head), (version, blocks[path][:80])
            assert score_blocks(p, blocks).conds["binding"], version
    blocks, _ = stubs(jprep, "v3")
    assert not score_blocks(jprep, {JPATH: blocks[JPATH][len("import io.cucumber.java.en.When;\n\n"):]}).conds["binding"]


# ----- Pins of the stub forms, bodies, swap and the dev prior, each killing mutants that survived the earlier
# tests; swap renames parameters in one pass -----------------------------------------------------------------------------

TICKS = "`" * 3


def test_pin_stub_forms_per_style():          # docblock / Java / C# / Ruby stub forms
    from v2tools.controls import stub_item
    assert stub_item("php_docblock", 2, "^a\\.b$", 1, "    $x = 1;\n") == \
        "/**\n * @When /^a\\.b$/\n */\npublic function stepStub2($p1): void\n{\n    $x = 1;\n}"
    assert stub_item("java_annotation", 1, '^a\\.b "q"$', 1, "") == '@When("^a\\\\.b \\"q\\"$")\npublic void stepStub1(String p1) {\n}'
    assert stub_item("csharp_attribute", 1, '^a "q"$', 1, "") == '[When(@"^a ""q""$")]\npublic void StepStub1(string p1)\n{\n}'
    assert stub_item("ruby_block", 1, "^a$", 2, "  x = 1\n") == "When(/^a$/) do |p1, p2|\n  x = 1\nend"
    assert stub_item("ruby_block", 1, "^a$", 0, "") == "When(/^a$/) do\nend"


def test_pin_bodies_per_language():
    from v2tools.controls import call_body, meaningless, touch, step_import
    assert call_body("csharp", ("cart", "Add")) == "    cart.Add();\n" and call_body("csharp", ("self", "Go")) == "    Go();\n"
    assert call_body("ruby", ("self", "go")) == "  go\n" and call_body("php", ("Clock", "check")) == "    Clock::check();\n"
    assert meaningless("csharp") == "    var x = 1;\n" and meaningless("ruby") == "  x = 1\n" and meaningless("java") == "    int x = 1;\n"
    assert touch("csharp", "m") == "    var o = this.m;\n" and touch("ruby", "m") == "  m\n" and touch("java", "m") == "    Object o = this.m;\n"
    assert step_import("csharp_attribute") == "using TechTalk.SpecFlow;" and step_import("php_docblock") is None
    assert step_import("ruby_block") is None


def test_pin_table_argument_adds_a_parameter(prep):
    from v2tools.controls import stubs
    p = dataclasses.replace(prep, create_steps=[(s, True) for s, _ in prep.create_steps])
    blocks, _ = stubs(p, "v2")
    assert "public function stepStub1($p1, $p2): void" in blocks[SHOP]


def test_pin_v1_is_anchored(prep):
    from v2tools.controls import stubs
    blocks, _ = stubs(prep, "v1")
    assert "#[When('/^I should see the total \"5\"$/')]" in blocks[SHOP]


def test_pin_swap_never_borrows_from_its_own_record(prep):
    from v2tools.controls import swap
    other = "#[Then('x :v')]\npublic function other(string $v): void\n{\n    $this->cart->add($v);\n}"
    assert swap(prep, {SHOP: [(prep.record["id"], other)]}) is None


def test_pin_swap_renames_by_position_simultaneously(prep, monkeypatch):   # pins a fixed defect
    import v2tools.controls as c
    own = "#[Then('I see :value of :other')]\npublic function own(string $value, string $other): void\n{\n    $this->x($value, $other);\n}"
    monkeypatch.setattr(c, "gold_answer", lambda r, fns: TICKS + "php\n// FILE: " + SHOP + "\n" + own + "\n" + TICKS)
    don = "#[Then('x :name :value')]\npublic function don(string $name, string $value): void\n{\n    $this->cart->add($name, $value);\n}"
    assert "$this->cart->add($value, $other);" in c.swap(prep, {SHOP: [("other", don)]})[SHOP]


def test_pin_swap_renames_whole_identifiers_only(prep, monkeypatch):
    import v2tools.controls as c
    own = "#[Then('I see :value of :other')]\npublic function own(string $value, string $other): void\n{\n    $this->x($value);\n}"
    monkeypatch.setattr(c, "gold_answer", lambda r, fns: TICKS + "php\n// FILE: " + SHOP + "\n" + own + "\n" + TICKS)
    don = "#[Then('x :name')]\npublic function don(string $name, string $z): void\n{\n    $this->cart->add($name, $subname, $names);\n}"
    assert "$this->cart->add($value, $subname, $names);" in c.swap(prep, {SHOP: [("other", don)]})[SHOP]


def test_pin_devprior_writes_the_smallest_spelling():
    from v2tools.controls import DevPrior
    pr = DevPrior("R", "php", (("a", frozenset({("cart", "add")})), ("b", frozenset({("cart", "ADD")}))))
    assert pr.ranked() == [("cart", "ADD")] and pr.counts() == {("cart", "add"): 2}


def test_pin_top4_prior_static_call_on_a_global_class(prep, monkeypatch):
    import v2tools.controls as c
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter())
    seen = {}
    monkeypatch.setattr(c, "class_status", lambda ctx, o, m: seen.setdefault(m, "global" if m == "Glob" else "missing"))
    got = c.top4_calls(prep, SHOP, c.Shown4(), shop_prior({"d1": {("Glob", "f"), ("Ghost", "f")}}))
    assert ("Glob", "f") in got and ("Ghost", "f") not in got


def test_pin_shown4_returns_the_composers_masked(prep):
    from v2tools.controls import shown4
    class Comp:
        def answer(self, r): return None
        def knowledge_candidates(self, r, a, dropped): dropped.append({"id": "f1", "kind": "knowledge"}); dropped.append({"id": "f2", "kind": "other"})
        def masked(self, r): return frozenset({(SHOP, "cart")})
    k = shown4(Comp(), prep.record)
    assert k.masked == frozenset({(SHOP, "cart")}) and k.dropped == frozenset({"f1"})


def test_pin_v3all_body_is_sorted(prep):
    from v2tools.controls import stubs, visible_calls, call_body
    seen = visible_calls(prep, SHOP)
    want = "".join(call_body("php", c) for c in sorted(seen))
    assert want in stubs(prep, "v3all")[0][SHOP]


def test_pin_v3m_excludes_self_calls_and_takes_the_first_of_a_tie(prep, monkeypatch):
    import v2tools.controls as c
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter({("self", "zz"): 9, ("b", "x"): 2, ("a", "y"): 2}))
    assert c.frequent_call(prep, SHOP) == ("a", "y")
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter({("self", "go"): 3, ("a", "y"): 1}))
    assert c.frequent_call(prep, SHOP) == ("a", "y")


def test_pin_v3c_positive_path_on_the_java_twin(tmp_path):    # the other fixtures never make v3c write a call
    from tests.phase3c_fixture import JPATH, make_jprep
    from v2tools.controls import first_service_call, stubs
    j = make_jprep(tmp_path)
    assert first_service_call(j, JPATH) == ("cart", "open")
    blocks, fallback = stubs(j, "v3c")
    assert fallback == 0 and "    cart.open();\n" in blocks[JPATH]


def test_pin_swap_needs_a_method_or_lambda_function(prep, monkeypatch):
    import v2tools.controls as c
    own = "private $x;\n\n#[Then('I see :v')]\npublic function own(string $v): void\n{\n    $this->x($v);\n}"
    monkeypatch.setattr(c, "gold_answer", lambda r, fns: TICKS + "php\n// FILE: " + SHOP + "\n" + own + "\n" + TICKS)
    don = "#[Then('x :n')]\npublic function don(string $n): void\n{\n    $this->cart->add($n);\n}"
    assert c.swap(prep, {SHOP: [("other", don)]}) is None


def test_pin_member_type():
    from types import SimpleNamespace as NS
    from v2tools.controls import member_type
    assert member_type(NS(text="private ?Cart $cart;", names=["cart"])) == "Cart"
    assert member_type(NS(text="private string $cart;", names=["cart"])) is None
    assert member_type(NS(text="private $cart;", names=["cart"])) is None


def test_pin_dev_prior_skips_ruby_records():
    from types import SimpleNamespace as NS
    from v2tools.controls import dev_prior
    p = NS(record={"partition": "dev", "id": "x", "repo": "r"}, repo=NS(language="ruby"))
    assert dev_prior([p]) == {}


def test_pin_controls_row_order(prep):           # controls.jsonl row order feeds SHA256SUMS
    from v2tools.controls import controls
    pool = {SHOP: [("other", "#[Then('x :v')]\npublic function other(string $v): void\n{\n    $this->cart->add($v);\n}")]}
    assert list(controls(prep, pool, priors={"Acme/Shop": shop_prior({})})) == [
        "gold", "stub-v1", "stub-v2", "stub-v3", "stub-v3c", "stub-v3m", "stub-v3all", "swap", "gold-helpers",
        "gold-helpers-renamed", "stub-v3m-h", "stub-v3all-h", "stub-v3names", "stub-v3top4"]


def test_pin_visible_calls_count_static_calls_on_project_classes(prep, monkeypatch):    # o_after reaches php_calls
    import v2tools.controls as c
    add = next(fid for fid, f in prep.repo.fns.items() if "function iAddToCart(" in f["code"])
    pre = "public function iAddToCart(string $product)\n{\n    BaseContext::make($product);\n    Nowhere::open($product);\n}"
    monkeypatch.setattr(c, "pre_t_code", lambda ctx, fid, t: pre if fid == add else None)
    seen = c.visible_calls(prep, SHOP)
    assert ("BaseContext", "make") in seen and ("Nowhere", "open") not in seen


def test_pin_escape_covers_the_php_delimiter():
    from v2tools.controls import _escape
    assert _escape("a/b#c-d.e") == r"a\/b\#c\-d\.e"


# ----- Lazy references and swap: members the ER tree lost, the function's own body, donor order -----------------------

def gold_item_text(prep) -> str:
    import v2tools.controls as c
    return c.split_code("php", c.extract(c.gold_answer(prep.record, prep.repo.fns), prep.targets).by_path()[SHOP])[1][0].text


def test_lazy_references_skip_php_calls_to_members_the_er_tree_lost(prep, monkeypatch):
    """A visible function's pre-T call to a property or a method the ER tree no longer has is not
    counted (it would make the lazy stub fail condition 6, the level non-conservative); a call to a member or an
    inherited method still there counts, and so does a class call."""
    import v2tools.controls as c
    add = next(fid for fid, f in prep.repo.fns.items() if "function iAddToCart(" in f["code"])
    pre = ("public function iAddToCart(string $product)\n{\n    $this->cart->add($product);\n    $this->goneMember->x();\n"
           "    $this->goneMethod();\n    $this->helperOld();\n    BaseContext::make($product);\n}")
    monkeypatch.setattr(c, "pre_t_code", lambda ctx, fid, t: pre if fid == add else None)
    seen = c.visible_calls(prep, SHOP)
    assert {("cart", "add"), ("self", "helperOld"), ("BaseContext", "make")} <= set(seen)
    assert ("goneMember", "x") not in seen and ("self", "goneMethod") not in seen


def test_swap_never_takes_the_functions_own_body(prep):
    """A donor whose body, after renaming, IS the function's own body (the same function reached
    through another record's gold with other annotations, or a twin) is skipped; with no other donor, no swap."""
    own = gold_item_text(prep)
    twin = "#[Given('another phrasing :total')]\n" + own
    assert swap(prep, {SHOP: [("other|x|1|s", twin)]}) is None


def test_swap_chooses_its_donor_by_hash_not_by_pool_order(prep):
    """The donor is the first in sha256 order, whatever the order of the pool."""
    a = ("o1", "#[Then('x :v')]\npublic function other(string $v): void\n{\n    $this->cart->add($v);\n}")
    b = ("o2", "#[Then('y :v')]\npublic function more(string $v): void\n{\n    $this->checkout->assertTotal($v);\n}")
    assert swap(prep, {SHOP: [a, b]}) == swap(prep, {SHOP: [b, a]})


def test_v3all_keeps_self_calls(prep, monkeypatch):
    """stub-v3all keeps self calls, `$this->f()` included (without them, test v3all CA 183 → 98)."""
    import v2tools.controls as c
    monkeypatch.setattr(c, "visible_calls", lambda *a, **k: Counter({("self", "helperOld"): 1, ("cart", "add"): 2}))
    blocks, fallback = c.stubs(prep, "v3all")
    assert fallback == 0 and "    $this->cart->add();\n    $this->helperOld();\n" in blocks[SHOP]


def test_one_stub_per_generalised_pattern(prep):
    """Two CREATE steps with one generalised pattern get one stub (without this rule, v3 BV 429 → 346)."""
    r = dict(prep.record)
    r["steps"], r["step_fns"] = r["steps"] + ['I should see the total "7"'], r["step_fns"] + [r["step_fns"][-1]]
    r["step_kinds"] = r["step_kinds"] + ["create"]
    p = dataclasses.replace(prep, record=r, create_steps=prep.create_steps + [('I should see the total "7"', False)])
    for v in ("v2", "v3", "v3m", "v3all"):
        assert stubs(p, v)[0][SHOP].count("function stepStub") == 1, v


def test_generalize_numbers_and_php_quote_escape():
    """Negative and decimal numbers (9 records) and a `'` escaped in a PHP attribute (4 records)."""
    from v2tools.controls import stub_item
    g = r"(-?\d+(?:[.,]\d+)*)"
    assert generalize("I pay -1.5 or 2,000 euros") == (f"^I pay {g} or {g} euros$", 2)
    assert stub_item("php_attribute", 1, "^it's$", 0, "") == "#[When('/^it\\'s$/')]\npublic function stepStub1(): void\n{\n}"
