"""Pins for rule 5 as the PRE-T FORM (state.pre_t_code): a ④(b) function, or the CREATE function of a ③ example,
whose HEAD code changed at or after T is shown as it was in its file before T, its annotation lines filtered by pattern
(rule 1), instead of being dropped; it is dropped (③: rule "changed_after_t"; ④(b): silently) only when that form
cannot be located. Rule 3 looks at the text shown. Fixture: tests/phase3_pre_t_fixture.py; each test has a control
where one makes sense (rule 5 off: `ctx.unchanged` always True, the HEAD form)."""
import pytest

from tests import phase3_pre_t_fixture as fx
from tests.phase3_chain_fixture import COMPOSER, T2021, build
from tests.phase3_fixture import T2020, T2022, T2024
from v2tools.compose import Composer, Params
from v2tools.outline import parse
from v2tools.state import RepoContext, annotation_pattern, pre_t_code, skeleton


@pytest.fixture(scope="module")
def php_git(tmp_path_factory):
    return fx.php_repo(tmp_path_factory.mktemp("pre_t_php"))


@pytest.fixture(scope="module")
def java_git(tmp_path_factory):
    return fx.java_repo(tmp_path_factory.mktemp("pre_t_java"))


def cart(php_git):
    cat = fx.php_catalog()
    return cat, RepoContext(fx.REPO, "php", *php_git, cat)


def apples(java_git):
    cat = fx.java_catalog()
    return cat, RepoContext(fx.JREPO, "java", *java_git, cat)


def comp_of(ctx, cat, recs):
    return Composer({ctx.repo: ctx}, {ctx.repo: cat}, {}, {r["id"]: r for r in recs},
                    Params(budget_examples=10 ** 6, budget_knowledge=10 ** 6))


def words(text: str) -> int:
    return len(text.split())


def shown(comp, r, dropped=None) -> dict[str, str]:
    return dict(comp.knowledge_candidates(r, comp.answer(r), [] if dropped is None else dropped))


def answer_of(comp, r, e) -> str:
    msgs, prov = comp.compose_prompt(r, "F1", words, [e["id"]])
    assert prov["examples"] == [e["id"]] and prov["dropped"] == []
    return msgs[0]["content"].split("Answer:\n", 1)[1]


HAVE_PRE_T = "public function iHaveACart()\n{\n    $this->cart->create();\n}"


# ----- ④(b) ---------------------------------------------------------------------------------------------------------
def test_knowledge_function_changed_after_t_by_a_cosmetic_edit_is_shown_in_its_pre_t_form(php_git):
    """iHaveACart's only change at T2024 is `: void`: the query at T2022 sees it as it was in the 2021 version (both
    annotations were there before T)."""
    cat, ctx = cart(php_git)
    q, have = fx.query(cat), fx.fid(cat, "function iHaveACart(")
    comp = comp_of(ctx, cat, [q])
    assert not ctx.unchanged(have, q["t_vis"])
    assert shown(comp, q)[have] == "#[Given('I have a cart')]\n#[Given('there is a cart')]\n" + HAVE_PRE_T
    ctx.unchanged = lambda fid, t: True  # control: rule 5 off, the HEAD form
    assert "public function iHaveACart(): void\n" in shown(comp, q)[have]


def test_docblock_annotated_function_keeps_its_docblock_in_the_pre_t_form(php_git):
    cat, ctx = cart(php_git)
    q, pay = fx.query(cat), fx.fid(cat, "function iPay(")
    comp = comp_of(ctx, cat, [q])
    assert shown(comp, q)[pay] == ("/**\n * @When I pay :amount\n */\npublic function iPay(string $amount)\n{\n"
                                   "    $this->cart->pay($amount);\n}")


def test_records_own_preexisting_item_is_shown_in_its_pre_t_form_without_its_create_annotation(php_git):
    """The record adds 'there is a cart' (2021) to iHaveACart and creates iCheckOut (T = 2022): the 2021 version holds
    the CREATE annotation line, which rule 1 takes out of the text shown."""
    cat, ctx = cart(php_git)
    q, have = fx.query_with_preexisting(cat), fx.fid(cat, "function iHaveACart(")
    assert "#[Given('there is a cart')]" in pre_t_code(ctx, have, q["t_vis"])
    comp, dropped = comp_of(ctx, cat, [q]), []
    assert shown(comp, q, dropped)[have] == "#[Given('I have a cart')]\n" + HAVE_PRE_T
    assert [d["id"] for d in dropped] == [fx.fid(cat, "function iSeeTheCart(")]  # it carries 'I check out' before T


def test_function_whose_pre_t_form_carries_a_create_pattern_of_the_query_is_dropped_as_pattern(php_git):
    """'I check out' moved from iSeeTheCart to the answer at T: iSeeTheCart's HEAD annotations do not carry it, its
    pre-T form does."""
    cat, ctx = cart(php_git)
    q, see = fx.query(cat), fx.fid(cat, "function iSeeTheCart(")
    assert "#[Then('I check out')]" in pre_t_code(ctx, see, q["t_vis"])
    comp, dropped = comp_of(ctx, cat, [q]), []
    assert see not in shown(comp, q, dropped)
    assert {"kind": "knowledge", "id": see, "rule": "pattern"} in dropped
    ctx.unchanged = lambda fid, t: True  # control: the HEAD form no longer carries the pattern
    assert see in shown(comp, q)


def test_rule_3_looks_at_the_pre_t_form_that_is_shown(php_git):
    """iEmptyTheCart's HEAD body calls fresh(), declared at T2024; its pre-T body does not: it is shown in ④(b) and as
    a ③ example. Control: the HEAD form would be dropped by rule 3."""
    cat, ctx = cart(php_git)
    q, e, empty = fx.query(cat), fx.example(cat, "iEmptyTheCart", "I empty the cart"), fx.fid(cat, "iEmptyTheCart(")
    assert "fresh" in ctx.future(fx.CART, q["t_vis"]) and "$this->fresh()" in cat[empty]["code"]
    comp = comp_of(ctx, cat, [q, e])
    assert "$this->cart->clear();" in shown(comp, q)[empty]
    assert "$this->cart->clear();" in answer_of(comp, q, e)
    ctx.unchanged = lambda fid, t: True
    assert empty not in shown(comp, q)
    _, prov = comp.compose_prompt(q, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "future_ref"}]


# ----- ③ examples ---------------------------------------------------------------------------------------------------
def test_example_whose_create_function_changed_after_t_is_kept_in_its_pre_t_form(php_git):
    cat, ctx = cart(php_git)
    q, e = fx.query(cat), fx.example(cat, "iHaveACart", "I have a cart")
    comp = comp_of(ctx, cat, [q, e])
    assert "#[Given('I have a cart')]\n" + HAVE_PRE_T + "\n" in answer_of(comp, q, e)
    ctx.unchanged = lambda fid, t: True  # control: rule 5 off, the HEAD form
    assert "public function iHaveACart(): void\n" in answer_of(comp, q, e)


def test_example_answer_leaves_out_an_annotation_added_after_its_scenario_and_before_t(php_git):
    """'there is a cart' (2021) is in the pre-T form (the 2021 version) but not in the gold of an example of 2020;
    control: the example of 2021 + 1 shows it."""
    cat, ctx = cart(php_git)
    q, have = fx.query(cat), fx.fid(cat, "function iHaveACart(")
    assert "#[Given('there is a cart')]" in pre_t_code(ctx, have, q["t_vis"])
    e = fx.example(cat, "iHaveACart", "I have a cart")
    assert "there is a cart" not in answer_of(comp_of(ctx, cat, [q, e]), q, e)
    later = fx.example(cat, "iHaveACart", "I have a cart", T2021 + 1)
    assert "#[Given('I have a cart')]\n#[Given('there is a cart')]\n" + HAVE_PRE_T in \
        answer_of(comp_of(ctx, cat, [q, later]), q, later)


def test_example_whose_pre_t_form_lacks_its_own_create_annotation_is_dropped(php_git):
    """The example's step text at HEAD differs from the one in the pre-T form (renamed after T): its answer would show no
    annotation for its step, so the example is dropped with rule 5."""
    cat, ctx = cart(php_git)
    q, e = fx.query(cat), fx.example(cat, "iHaveACart", "I have a cart")
    have = fx.fid(cat, "function iHaveACart(")
    ann = next(a for a in cat[have]["annotations"] if a["def_id"] in e["create_def_ids"])
    ann["pattern"] = "I own a cart"
    _, prov = comp_of(ctx, cat, [q, e]).compose_prompt(q, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "changed_after_t"}]


def _twin_of_the_answer_before_t(cat, ctx, q):
    """Give the answer a long body and make the pre-T form of iHaveACart a renamed copy of it (a twin before T only)."""
    out, have = fx.fid(cat, "function iCheckOut("), fx.fid(cat, "function iHaveACart(")
    body = "\n".join(f"    $this->cart->step('item-{i}', 'a fairly long argument number {i}');" for i in range(4))
    cat[out]["code"] = cat[out]["code"].split("{", 1)[0] + "{\n" + body + "\n}"
    ctx.pre_t[(have, ctx.version_index(cat[have]["path"], q["t_vis"]))] = \
        "#[Given('I have a cart')]\npublic function iHaveACart()\n{\n" + body + "\n}"
    return have


def test_knowledge_function_whose_pre_t_form_is_a_twin_of_a_new_answer_is_dropped(php_git):
    cat, ctx = cart(php_git)
    q = fx.query(cat)
    have = _twin_of_the_answer_before_t(cat, ctx, q)
    dropped: list = []
    assert have not in shown(comp_of(ctx, cat, [q]), q, dropped)
    assert {"kind": "knowledge", "id": have, "rule": "twin"} in dropped


def test_example_whose_pre_t_form_is_a_twin_of_a_new_answer_is_dropped(php_git):
    cat, ctx = cart(php_git)
    q, e = fx.query(cat), fx.example(cat, "iHaveACart", "I have a cart")
    _twin_of_the_answer_before_t(cat, ctx, q)
    _, prov = comp_of(ctx, cat, [q, e]).compose_prompt(q, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "twin"}]


def test_an_overload_never_takes_the_pre_t_form_of_a_hidden_step(tmp_path):
    """fineract WorkingCapitalStepDef: with the step overload hidden (a test answer, hidden from train), the helper
    overload left in the view takes its own pre-T declaration (same parameter names), never the step's, so the
    answer's signature is not shown."""
    path = "src/test/java/org/x/WcSteps.java"
    java = ("package org.x;\n\npublic class WcSteps {\n    @Given(\"I create a product\")\n    public void make() {\n"
            "        run();\n    }\n\n    public void make(String name) {\n        run();\n    }\n}\n")
    bare, head = build(tmp_path, [(T2020, T2020, {path: java})])
    step = fx.row("Acme/J", path, java, fx._decl(java, "public void make()"), [T2020])
    ctx = RepoContext("Acme/J", "java", bare, head, {step["fn_id"]: step})
    s = skeleton(ctx, path, T2021, {step["fn_id"]})
    assert "public void make(String name) { /* … */ }" in s and "public void make() {" not in s
    assert "public void make() { /* … */ }" in skeleton(ctx, path, T2021, set())  # control: not hidden, shown


def test_a_helper_overload_with_the_same_parameter_names_never_takes_a_hidden_step_s_pre_t_form(tmp_path):
    """The step guard on its own: step make(String name) and helper make(int name) share parameter names, so only
    "a non-step never takes a declaration that carries a step annotation" keeps the step's signature out."""
    path = "src/test/java/org/x/WcSteps.java"
    java = ("package org.x;\n\npublic class WcSteps {\n    @Given(\"I create {string}\")\n    public void make(String name) {\n"
            "        run();\n    }\n\n    public void make(int name) {\n        run();\n    }\n}\n")
    bare, head = build(tmp_path, [(T2020, T2020, {path: java})])
    step = fx.row("Acme/J", path, java, fx._decl(java, "public void make(String name)"), [T2020])
    ctx = RepoContext("Acme/J", "java", bare, head, {step["fn_id"]: step})
    s = skeleton(ctx, path, T2021, {step["fn_id"]})
    assert "public void make(int name) { /* … */ }" in s and "make(String name)" not in s


def test_an_overload_takes_the_pre_t_declaration_with_its_own_parameter_names(tmp_path):
    """Two helper overloads before T; go(int a) is removed after T. The one left at HEAD, go(String s), is the first of
    its name at HEAD but the second before T: it must be shown as go(String s), not by position as go(int a)."""
    path = "src/test/java/org/x/GoHelp.java"
    before = ("package org.x;\n\npublic class GoHelp {\n    public void go(int a) {\n        run();\n    }\n\n"
              "    public void go(String s) {\n        run();\n    }\n}\n")
    after = "package org.x;\n\npublic class GoHelp {\n    public void go(String s) {\n        run();\n    }\n}\n"
    bare, head = build(tmp_path, [(T2020, T2020, {path: before}), (T2024, T2024, {path: after})])
    s = skeleton(RepoContext("Acme/J", "java", bare, head, {}), path, T2022, set())
    assert "public void go(String s) { /* … */ }" in s and "go(int a)" not in s


def test_the_namespace_of_the_outline_is_the_one_before_t(tmp_path):
    path = "src/Ctx/Moved.php"
    body = "final class Moved\n{\n    public function a(): void\n    {\n    }\n}\n"
    bare, head = build(tmp_path, [(T2020, T2020, {"composer.json": COMPOSER, path: "<?php\n\nnamespace App\\Old;\n\n" + body}),
                                   (T2024, T2024, {path: "<?php\n\nnamespace App\\Ctx;\n\n" + body})])
    ctx = RepoContext("Acme/P", "php", bare, head, {})
    assert skeleton(ctx, path, T2022, set()).startswith("namespace App\\Old;")
    assert skeleton(ctx, path, T2024 + 1, set()).startswith("namespace App\\Ctx;")


def test_a_helper_after_a_kept_step_overload_keeps_its_plain_pre_t_declaration(tmp_path):
    """fineract LoanStepDef.createCustomizedLoan: the helper's parameter names changed after T, so it falls back to its
    position among ALL overloads of the name before T, i.e. the plain helper, not dropped."""
    path = "src/test/java/org/x/LoanSteps.java"
    step = "    @Given(\"I create a loan on {string}\")\n    public void make(String date) {\n        run();\n    }\n\n"
    before = "package org.x;\n\npublic class LoanSteps {\n" + step + "    private void make(String data, boolean emi) {\n        run();\n    }\n}\n"
    after = "package org.x;\n\npublic class LoanSteps {\n" + step + "    private void make(String loanData, String emiStr) {\n        run();\n    }\n}\n"
    bare, head = build(tmp_path, [(T2020, T2020, {path: before}), (T2024, T2024, {path: after})])
    row = fx.row("Acme/J", path, after, fx._decl(after, "public void make(String date)"), [T2020])
    s = skeleton(RepoContext("Acme/J", "java", bare, head, {row["fn_id"]: row}), path, T2022, set())
    assert "public void make(String date) { /* … */ }" in s and "private void make(String data, boolean emi) { /* … */ }" in s


def test_the_step_guard_reads_the_pre_t_text_of_the_declaration(tmp_path):
    """HEAD adds lines at the top of the class, so the HEAD text at a pre-T span is not the pre-T annotation: the guard
    must read the pre-T version (same-parameter-name overloads, as in the WorkingCapitalStepDef pin)."""
    path = "src/test/java/org/x/WcSteps.java"
    body = ("    @Given(\"I create {string}\")\n    public void make(String name) {\n        run();\n    }\n\n"
            "    public void make(int name) {\n        run();\n    }\n}\n")
    before = "package org.x;\n\npublic class WcSteps {\n" + body
    after = "package org.x;\n\npublic class WcSteps {\n    private int one;\n    private int two;\n    private int three;\n\n" + body
    bare, head = build(tmp_path, [(T2020, T2020, {path: before}), (T2024, T2024, {path: after})])
    row = fx.row("Acme/J", path, after, fx._decl(after, "public void make(String name)"), [T2020])
    s = skeleton(RepoContext("Acme/J", "java", bare, head, {row["fn_id"]: row}), path, T2022, {row["fn_id"]})
    assert "public void make(int name) { /* … */ }" in s and "make(String name)" not in s


def test_a_constructor_takes_the_pre_t_declaration_with_its_own_parameter_names(tmp_path):
    path = "src/test/java/org/x/Ctor.java"
    before = ("package org.x;\n\npublic class Ctor {\n    public Ctor(int a) {\n        run();\n    }\n\n"
              "    public Ctor(String s) {\n        run();\n    }\n}\n")
    after = "package org.x;\n\npublic class Ctor {\n    public Ctor(String s) {\n        run();\n    }\n}\n"
    bare, head = build(tmp_path, [(T2020, T2020, {path: before}), (T2024, T2024, {path: after})])
    s = skeleton(RepoContext("Acme/J", "java", bare, head, {}), path, T2022, set())
    assert "public Ctor(String s) { /* … */ }" in s and "Ctor(int a)" not in s


def test_the_step_guard_also_vetoes_the_positional_pick(tmp_path):
    """gpconnect OrganizationSearch: the hidden step store() and the helper store(siteCode), itself a step before T.
    The name pick is vetoed, and the position (k = 0) lands on the hidden step's pre-T declaration, which the guard
    must veto too."""
    path = "src/test/java/org/x/OrgSteps.java"
    s0 = "    @Given(\"I store the Organization\")\n    public void store() {\n        run();\n    }\n\n"
    before = ("package org.x;\n\npublic class OrgSteps {\n" + s0 + "    @Given(\"I store the Organization with site code {string}\")\n"
              "    public void store(String siteCode) {\n        run();\n    }\n}\n")
    after = "package org.x;\n\npublic class OrgSteps {\n" + s0 + "    public void store(String siteCode) {\n        run();\n    }\n}\n"
    bare, head = build(tmp_path, [(T2020, T2020, {path: before}), (T2024, T2024, {path: after})])
    row = fx.row("Acme/J", path, after, fx._decl(after, "public void store()"), [T2020])
    ctx = RepoContext("Acme/J", "java", bare, head, {row["fn_id"]: row})
    assert "public void store()" not in skeleton(ctx, path, T2022, {row["fn_id"]})
    assert "public void store() { /* … */ }" in skeleton(ctx, path, T2022, set())  # control: not hidden, shown


# ----- Java: lambda steps and overloads -----------------------------------------------------------------------------
def test_java_lambda_step_changed_after_t_is_left_out_and_shown_when_unchanged(java_git):
    """A cucumber-java8 step has no method to find in the pre-T file: ④(b) skips it silently, ③ drops its example as
    changed_after_t. Control: a query after its last change sees its HEAD code."""
    cat, ctx = apples(java_git)
    q, lam = fx.java_query(cat), fx.fid(cat, 'Given("I have {int} apples"')
    e = fx.record(cat, "have", [(lam, False, ["I have {int} apples"])], T2020, T2020, "train", ["I have 3 apples"], "java")
    assert pre_t_code(ctx, lam, q["t_vis"]) == ""
    comp, dropped = comp_of(ctx, cat, [q, e]), []
    assert lam not in shown(comp, q, dropped) and dropped == []
    _, prov = comp.compose_prompt(q, "F1", words, [e["id"]])
    assert prov["examples"] == [] and prov["dropped"] == [{"kind": "example", "id": e["id"], "rule": "changed_after_t"}]
    later = dict(q, t_vis=T2024 + 1, scenario_time=T2024 + 1)
    assert pre_t_code(ctx, lam, later["t_vis"]) is None and shown(comp, later)[lam] == cat[lam]["code"]


def test_pre_t_code_picks_the_overload_with_the_head_parameter_names_and_keeps_its_javadoc(java_git):
    cat, ctx = apples(java_git)
    one, many = sorted((f for f in cat if f.startswith(fx.GO)), key=lambda f: int(f.rsplit(":", 1)[1]))
    assert pre_t_code(ctx, many, T2022) == '@When("I go {int} times")\npublic void go(int n) {\n    run(n);\n}'
    assert pre_t_code(ctx, one, T2022) == '/**\n * Goes once.\n */\n@When("I go")\npublic void go() {\n    run(0);\n}'


def test_pre_t_code_is_none_when_unchanged_and_empty_when_the_function_cannot_be_located(php_git, java_git):
    cat, ctx = cart(php_git)
    out = fx.fid(cat, "function iCheckOut(")
    assert pre_t_code(ctx, out, T2024 + 1) is None
    assert pre_t_code(ctx, out, T2021) == ""  # the file had no method of that name before T
    assert pre_t_code(ctx, out, T2020) == ""  # the file did not exist before T
    jcat, jctx = apples(java_git)
    assert pre_t_code(jctx, fx.fid(jcat, "iEatApples"), T2021) == ""
    ruby = RepoContext("Acme/Rb", "ruby", *php_git, {"features/step_definitions/a.rb:1": {"path": "a.rb", "code": ""}})
    ruby.unchanged = lambda fid, t: False
    assert pre_t_code(ruby, "features/step_definitions/a.rb:1", T2022) == ""


# ----- Member.span and annotation_pattern --------------------------------------------------------------------------
PHP_SRC = """<?php
class A
{
    private $x;

    /**
     * Doc.
     */
    #[Given('a')]
    public function f(): void
    {
        $y = 1;
    }

    // a comment
    public function g() { }

    public function h() { } // trailing
    public function k() { }
}
"""
JAVA_SRC = """class B {
    private int x;

    /**
     * Doc.
     */
    @Given("b")
    public void f() {
    }

    // a comment
    public B() {
    }
}
"""
CS_SRC = """namespace N
{
    public class C
    {
        /// <summary>Doc.</summary>
        [Given(@"c")]
        public void F()
        {
        }
    }
}
"""


@pytest.mark.parametrize("language,source,spans,lines", [
    ("php", PHP_SRC, {"x": (0, 0), "f": (6, 13), "g": (15, 16), "h": (18, 18), "k": (19, 19)},
     {"x": 4, "f": 10, "g": 16, "h": 18, "k": 19}),
    ("java", JAVA_SRC, {"x": (0, 0), "f": (4, 9), "B": (11, 13)}, {"x": 2, "f": 8, "B": 12}),
    ("csharp", CS_SRC, {"F": (5, 9)}, {"F": 7}),
])
def test_member_span_covers_attributes_and_the_docblock_or_comments_directly_above(language, source, spans, lines):
    """A comment that ends another member's line (h's `// trailing`) does not belong to the next one; `line` and
    `text` still start at the signature."""
    o = parse(language, source)
    assert {m.names[0]: m.span for m in o.members} == spans
    assert {m.names[0]: m.line for m in o.members} == lines
    assert all(m.text.startswith(("public", "private")) for m in o.members)


@pytest.mark.parametrize("language,line,pattern", [
    ("php", "    #[Given('I have a cart')]", "I have a cart"),
    ("php", '    #[When("I pay \'x\'")]', "I pay 'x'"),
    ("php", "    #[Then('it\\'s /^fine$/')]", "it's /^fine$/"),
    ("php", "    #[Then('/^I (?:see|view) it$/')]", "^I (?:see|view) it$"),
    ("php", "    #[given('lower case')]", "lower case"),
    ("php", "     * @When I pay :amount", "I pay :amount"),
    ("php", "    /** @Then I see :count products */", "I see :count products"),
    ("php", "     * @param string $amount", None),
    ("php", "    public function iPay(string $amount)", None),
    ("java", '    @Given("I have {int} apples")', "I have {int} apples"),
    ("java", '    @Then("^the \\"x\\" is (\\\\d+)$")', '^the "x" is (\\d+)$'),
    ("java", '    @And("/^I go$/")', "^I go$"),
    ("java", '        Given("I have {int} apples", (Integer n) -> {', None),
    ("csharp", '        [Given(@"I say ""hi"" to (.*)")]', 'I say "hi" to (.*)'),
    ("csharp", '        [Then("a \\"b\\"")]', 'a "b"'),
    ("csharp", "        public void F()", None),
    ("ruby", "Given(/^I x$/) do", None),
])
def test_annotation_pattern_reads_the_pattern_as_the_catalog_stores_it(language, line, pattern):
    assert annotation_pattern(language, line) == pattern
