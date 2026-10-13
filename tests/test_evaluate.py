from types import SimpleNamespace

import pytest

from tests.phase3_fixture import SHOP
from tests.phase3c_fixture import block, make_prep
from v2tools.evaluate import Prepared, Repo, ca, expanded_futures, future_keys, future_methods, score_output

TICKS = "`" * 3


@pytest.fixture()
def prep(tmp_path):
    return make_prep(tmp_path)


def test_hallucinated_member_fails_resolved(prep):
    s = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($this->ghostService->x($total));\n}"))
    assert s.bv == 0 and not s.conds["resolved"] and s.conds["grounded"] and s.detail["unresolved"] == ["ghostService"]


def test_hardcoded_value_fails_arity(prep):
    s = score_output(prep, block("#[Then('I should see the total \"5\"')]\npublic function t(): void\n{\n    $this->checkout->assertTotal('5');\n}"))
    assert not s.conds["arity"] and s.conds["binding"]


def test_behat_n_is_read_on_the_step(prep):
    """Condition 4 for PHP reads N on the CREATE step itself (§2.2 of the design specification at 25ba873): the step
    `I should see the total "5"` leaves the trailing optional group of the pattern unmatched, so Behat hands over one
    argument and a method that requires two fails (Behat: "Can not find a matching value for an argument"), while
    one whose second parameter has a default, or that takes one, passes."""
    head = "#[Then('/^I should see the total \"(\\d+)\"(?: in (\\w+))?$/')]\npublic function t("
    tail = "): void\n{\n    $this->checkout->assertTotal($total);\n}"
    got = {sig: score_output(prep, block(head + sig + tail)).conds["arity"]
           for sig in ("string $total, string $currency", "string $total, string $currency = ''", "string $total")}
    assert got == {"string $total, string $currency": False, "string $total, string $currency = ''": True,
                   "string $total": True}

def test_reimplementing_a_reuse_step_breaks_no_ambiguity(prep):
    s = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}\n\n#[Given('I open the shop')]\npublic function again(): void\n{\n    $this->indexPage->open();\n}"))
    assert not s.conds["no_ambiguity"] and s.detail["duplicate_reuse"]


def test_no_block_and_collision(prep):
    assert score_output(prep, "no code here").extract_status == "none"
    s = score_output(prep, block("#[Then('I should see the total :total')]\npublic function iAddToCart(string $total): void\n{\n}"))
    assert s.insert_status == "collision" and s.bv == 0


def test_gold_scores_bv_1_and_agrees_with_itself(prep):
    g = prep.gold()
    assert g.bv == 1 and all(g.conds.values()) and g.calls and ca(prep, g) == (True, 1)


def test_ca_needs_a_shared_project_call(prep):
    right = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}"))
    wrong = score_output(prep, block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->cart->add($total);\n}"))
    assert right.bv == wrong.bv == 1
    assert ca(prep, right) == (True, 1) and ca(prep, wrong) == (True, 0)


def test_a_whole_class_in_the_block_is_not_repaired(prep):
    s = score_output(prep, block("<?php\n\nclass ShopContext\n{\n    #[Then('I should see the total :total')]\n    public function t(string $total): void\n    {\n        $this->checkout->assertTotal($total);\n    }\n}"))
    assert not s.conds["syntax"] and s.bv == 0 and ca(prep, s)[1] == 0


def test_the_same_method_twice_is_a_collision(prep):
    fn = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}"
    s = score_output(prep, block(fn + "\n\n" + fn))
    assert s.insert_status == "collision" and s.bv == 0 and s.detail["duplicate_keys"] == ["t"]
    assert score_output(prep, block(fn) + "\n\n" + block(fn.replace("#[Then", "#[When"))).insert_status == "collision"


def test_calls_that_cannot_run_do_not_count_for_ca(prep):
    head = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->cart->add($total);\n"
    for dead in ("    // $this->checkout->assertTotal($total);\n", "    $s = \"{$this->checkout->assertTotal($total)}\";\n",
                 "    return;\n    $this->checkout->assertTotal($total);\n"):
        s = score_output(prep, block(head + dead + "}"))
        assert s.bv == 1 and ca(prep, s) == (True, 0), dead


def test_other_suites_count_only_where_the_target_file_is_loaded(prep):
    """A step of another suite that a function of the target file serves must keep its binding."""
    cart = next(ids for s, ids in prep.repo.suite_steps["s"] if "cart" in s)
    prep.repo.suite_steps["other"] = [("I pay now", cart)]
    out = block("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}\n\n"
                "#[When('I pay now')]\npublic function pay(): void\n{\n    $this->cart->add('x');\n}")
    for served, ok in ((set(), True), ({"other"}, False)):
        p = make_prep_with(prep, served)
        assert score_output(p, out).conds["no_ambiguity"] is ok


def make_prep_with(prep, served: set):
    from v2tools.evaluate import prepare
    fns = {fid: {**f, "suites": sorted({"s"} | served)} if f["path"] == SHOP else f for fid, f in prep.repo.fns.items()}
    repo = prep.repo
    repo.fns = fns
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in repo.suite_steps["s"][:2]]}
    return prepare(prep.record, repo, labels_row)


JPATH = "src/test/java/p/Steps.java"
JAVA_ER = ('package p;\n\nimport io.cucumber.java.en.*;\n\npublic class Steps {\n    private Api api;\n\n'
           '    @Given("I open the shop")\n    public void iOpenTheShop() {\n        api.open();\n    }\n}\n')
JGOLD = '@Then("the total is {int}")\npublic void theTotalIs(int n) {\n    api.check(n);\n}'


def java_prep() -> Prepared:
    """One Java target file on a fake repo context (no ancestors, no static import); CREATE step "the total is 5"."""
    resolver = SimpleNamespace(by_suffix={}, resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n))
    ctx = SimpleNamespace(language="java", text={JPATH: JAVA_ER}.get, parents=lambda p: [], ancestors=lambda p: [],
                          resolver=resolver)
    repo = Repo("Acme/J", "java", ["java_annotation"], {}, ctx, frozenset(), {})
    return Prepared({"suite": "s"}, repo, None, {JPATH: JAVA_ER}, [JPATH], {JPATH: set()}, [("the total is 5", False)],
                    [], [], {JPATH: {"s"}}, _future={JPATH: {}}, _shown={JPATH: set()})  # no history for the skeleton


def jblock(code: str) -> str:
    return f"{TICKS}java\n// FILE: {JPATH}\n{code}\n{TICKS}"


def test_a_step_hidden_in_an_anonymous_class_fails_syntax():
    """A probe: the gold method moved into an anonymous class of a field initializer."""
    p = java_prep()
    assert score_output(p, jblock(JGOLD)).bv == 1
    hidden = "private Object f = new Object() {\n" + "\n".join("    " + l for l in JGOLD.split("\n")) + "\n};"
    s = score_output(p, jblock(hidden))
    assert not s.conds["syntax"] and s.bv == 0 and s.detail["raw"]
    idiom = "private TypeReference<List<Foo>> t = new TypeReference<List<Foo>>() {};"  # a member
    assert score_output(p, jblock(JGOLD + "\n\n" + idiom)).bv == 1


def test_a_field_initializer_is_judged_by_resolved():
    """The names a field initializer uses must resolve like body references;
    condition 5 and the CA call sets still read the step functions only."""
    p = java_prep()
    gold = score_output(p, jblock(JGOLD))
    ghost = score_output(p, jblock(JGOLD + "\n\nprivate Object f = ghostFieldZ.make();"))
    assert gold.bv == 1 and ghost.bv == 0 and not ghost.conds["resolved"] and ghost.conds["grounded"]
    assert ghost.detail["unresolved"] == ["ghostFieldZ"] and ghost.calls == gold.calls
    real = score_output(p, jblock(JGOLD + "\n\nprivate Object f = api.snapshot(), g = f;"))
    assert real.bv == 1 and real.calls == gold.calls


def test_root_object_members_resolve_in_a_field_and_a_body():
    """`getClass()` of java.lang.Object resolves in a field initializer and in a body, is no
    open path, and adds no call to the CA call sets."""
    p = java_prep()
    gold = score_output(p, jblock(JGOLD))
    log = "private final org.slf4j.Logger probeLog = org.slf4j.LoggerFactory.getLogger(getClass());"
    field = score_output(p, jblock(JGOLD + "\n\n" + log))
    body = score_output(p, jblock(JGOLD.replace("api.check(n);", "api.check(n + hashCode());")))
    for s in (field, body):
        assert s.bv == 1 and not s.detail["open_resolution"] and s.calls == gold.calls


def test_a_step_hidden_in_a_member_never_binds(prep):
    """Step definitions come from method, lambda and Ruby block items only, as in the independent
    scorer; a step annotation inside a member (a class nested in an anonymous class, a PHP `new class`) never binds."""
    p, indent = java_prep(), lambda code, pad: "\n".join(pad + l for l in code.split("\n"))
    nested = "private Object probeX = new Object() {\n    class In {\n" + indent(JGOLD, " " * 8) + "\n    }\n};"
    s = score_output(p, jblock(nested))
    assert s.bv == 0 and not s.conds["binding"] and s.conds["syntax"]
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    php = score_output(prep, block("private $probeX = new class {\n" + indent(gold, " " * 4) + "\n};"))
    assert php.bv == 0 and not php.conds["binding"] and ca(prep, php) == (True, 0)


# ----- Future methods, replacement, helper expansion ------------------------------------------------------------------

BASE = "src/Behat/Context/BaseContext.php"
STEP = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{{\n    {}\n}}"
T = (SHOP, "t")


def er_edit(p, path: str, old: str, new: str) -> None:
    """Edit the record's ER tree (before anything is scored): a future helper with project calls, or a new one."""
    text = p.er_files.get(path) or p.repo.ctx.text(path)
    assert old in text
    p.er_files[path] = text.replace(old, new)


def php_fn(name: str, *lines: str) -> str:
    return f"private function {name}(): void\n{{\n" + "".join(f"    {x}\n" for x in lines) + "}"


def test_future_methods_are_the_er_tree_methods_3a_does_not_show(prep):
    """A method of the target class or an in-repo ancestor in the ER tree whose name the prompt's class skeleton
    (state.t_view) does not show; only the target class's own ones may be replaced (future_keys). An ancestor's private
    method is never future (BaseContext::secret); the target's own private computeTotal is."""
    er_edit(prep, BASE, "    private function secret", php_fn("baseHelper", "$this->sharedStorage->get('x');").replace(
        "private", "    protected", 1).replace("\n", "\n    ") + "\n\n    private function secret")
    fut = future_methods(prep)[SHOP]
    assert {k: (f.name, f.origin) for k, f in fut.items()} == {
        "computetotal": ("computeTotal", SHOP), "basehelper": ("baseHelper", BASE)}
    assert fut["computetotal"].text == "private function computeTotal(string $total): string\n{\n    return $total;\n}"
    assert future_keys(prep, SHOP) == {"computetotal"}


def test_gold_expands_a_future_helper_from_its_er_tree_body(prep):
    """The gold's ("self", computeTotal) becomes the project calls of computeTotal's ER-tree body."""
    er_edit(prep, SHOP, "        return $total;\n", "        $this->cart->add($total);\n        return $total;\n")
    g = prep.gold()
    assert g.bv == 1 and g.calls == {(SHOP, "ishouldseetotal"): {("checkout", "assertTotal"), ("cart", "add")}}
    assert g.cuts == {"depth": 0, "cycle": 0}


def test_a_visible_method_is_not_expanded(prep):
    s = score_output(prep, block(STEP.format("$this->iAddToCart($total);")))
    assert s.bv == 1 and s.calls[T] == {("self", "iAddToCart")}


def test_replacing_a_future_helper_counts_only_the_outputs_calls(prep):
    """An output method with a future method's key replaces it (no collision); only its calls count."""
    er_edit(prep, SHOP, "        return $total;\n", "        $this->cart->add($total);\n        return $total;\n")
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    mine = ("private function computeTotal(string $total): string\n{\n    $this->indexPage->open();\n"
            "    return $total;\n}")
    s = score_output(prep, block(gold + "\n\n" + mine))
    assert s.insert_status == "ok" and s.detail["replaced"] == 1 and s.bv == 1
    assert s.calls[(SHOP, "ishouldseetotal")] == {("checkout", "assertTotal"), ("indexPage", "open")}
    assert ca(prep, s) == (True, 1)


def test_a_self_written_helper_contributes_exactly_its_own_calls(prep):
    """Calling a self-written helper counts as writing its calls in the step, for CA and for
    condition 5 (stub-v3m-h = stub-v3m); the helper's name never grounds on its own."""
    direct = score_output(prep, block(STEP.format("$this->checkout->assertTotal('x');")))
    via = score_output(prep, block(STEP.format("$this->probeHelper();") + "\n\n" +
                                   php_fn("probeHelper", "$this->checkout->assertTotal('x');")))
    assert direct.bv == via.bv == 1 and via.calls[T] == direct.calls[T] == {("checkout", "assertTotal")}
    assert ca(prep, via) == ca(prep, direct) == (True, 1)
    empty = score_output(prep, block(STEP.format("$this->probeHelper();") + "\n\n" + php_fn("probeHelper", "$x = 1;")))
    assert not empty.conds["grounded"] and empty.bv == 0 and empty.calls[T] == set()


def test_cycles_and_calls_deeper_than_3_stay_unexpanded(prep):
    """A cycle, and a helper reached at depth > 3, stay ("self", f); Score.cuts counts them."""
    h1 = php_fn("h1", "$this->cart->a1();", "$this->h2();")
    cyc = score_output(prep, block(STEP.format("$this->h1();") + "\n\n" + h1 + "\n\n" + php_fn("h2", "$this->h1();")))
    assert cyc.calls[T] == {("cart", "a1"), ("self", "h1")} and cyc.cuts == {"depth": 0, "cycle": 1}
    chain = "\n\n".join(php_fn(f"h{i}", f"$this->cart->a{i}();", f"$this->h{i + 1}();") for i in range(1, 5))
    chain += "\n\n" + php_fn("h5", "$this->cart->a5();")
    deep = score_output(prep, block(STEP.format("$this->h1();") + "\n\n" + chain))
    assert deep.calls[T] == {("cart", "a1"), ("cart", "a2"), ("cart", "a3"), ("self", "h4")}
    assert deep.cuts == {"depth": 1, "cycle": 0}


def test_an_ancestor_future_method_is_overridden_in_the_target(prep):
    """An output method with the key of an ANCESTOR's future method is a normal override,
    inserted in the target class; the ancestor's file is untouched, and expansion reads the output's version."""
    helper = php_fn("baseHelper", "$this->sharedStorage->get('x');").replace("private", "protected")
    er_edit(prep, BASE, "    private function secret",
            "    " + helper.replace("\n", "\n    ") + "\n\n    private function secret")
    inherited = score_output(prep, block(STEP.format("$this->baseHelper();")))
    assert inherited.calls[T] == {("sharedStorage", "get")}
    mine = php_fn("baseHelper", "$this->cart->add('y');").replace("private", "protected")
    s = score_output(prep, block(STEP.format("$this->baseHelper();") + "\n\n" + mine))
    assert s.insert_status == "ok" and s.detail["replaced"] == 0 and s.after[BASE] == prep.er_files[BASE]
    assert s.calls[T] == {("cart", "add")}


def test_gold_with_its_future_helpers_verbatim_or_renamed_keeps_ca(prep):
    """Positive controls at the level of evaluate: gold plus its future helper copied from the ER tree
    (a replacement, not a collision), or renamed at its definition and call site, reaches CA 1."""
    er_edit(prep, SHOP, "        return $total;\n", "        $this->cart->add($total);\n        return $total;\n")
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    helper = future_methods(prep)[SHOP]["computetotal"].text
    for out in (gold + "\n\n" + helper, (gold + "\n\n" + helper).replace("computeTotal", "probeHelper1")):
        s = score_output(prep, block(out))
        gid = (SHOP, "ishouldseetotal")
        assert s.bv == 1 and ca(prep, s) == (True, 1) and s.calls[gid] == prep.gold().calls[gid], out


def test_a_java_self_written_helper_is_expanded_by_name():
    """Helper expansion on Java keys (name, arity): a bare call to an output-defined helper becomes the helper's
    calls, and the helper's name alone does not ground."""
    p, step = java_prep(), '@Then("the total is {int}")\npublic void theTotalIs(int n) {\n    probeHelper(n);\n}'
    s = score_output(p, jblock(step + "\n\nprivate void probeHelper(int n) {\n    api.check(n);\n}"))
    assert s.bv == 1 and s.calls[(JPATH, ("theTotalIs", 1))] == {("api", "check")}
    empty = score_output(p, jblock(step + "\n\nprivate void probeHelper(int n) {\n    int x = n;\n}"))
    assert not empty.conds["grounded"] and empty.calls[(JPATH, ("theTotalIs", 1))] == set()


def test_expanded_futures_lists_the_future_helpers_the_gold_reads(prep):
    """The input of the helper controls: the FUTURE methods the gold's expansion reads, to depth 3, per target path,
    in order."""
    er_edit(prep, SHOP, "        return $total;\n", "        $this->cart->add($total);\n        $this->deeper();\n"
            "        return $total;\n    }\n\n    private function deeper(): void\n    {\n"
            "        $this->indexPage->open();\n")
    got = expanded_futures(prep, prep.gold())
    assert [(f.name, f.origin) for f in got[SHOP]] == [("computeTotal", SHOP), ("deeper", SHOP)]
    want = {("checkout", "assertTotal"), ("cart", "add"), ("indexPage", "open")}
    assert prep.gold().calls[(SHOP, "ishouldseetotal")] == want


def test_a_visible_method_stays_self_even_when_rewritten(prep):
    """A method visible in V(r) stays ("self", h) even when the gold or the output rewrites it
    (a preexisting CREATE item): visible -> keep; else output-defined -> the output's version; else future -> the ER
    tree's; the visible name grounds as a project member."""
    from v2tools.evaluate import prepare
    cart = next(fid for fid, f in prep.repo.fns.items() if "iAddToCart" in f["code"])
    r = {**prep.record, "create": [{**prep.record["create"][0], "code": prep.record["create"][0]["code"].replace(
        "$this->checkout->assertTotal(", "$this->iAddToCart($total);\n    $this->checkout->assertTotal(")},
        {"fn_id": cart, "path": SHOP, "code": prep.repo.fns[cart]["code"], "preexisting": True,
         "post_cutoff_code": False}]}
    p = prepare(r, prep.repo, {"step_def_ids": [[i[0]] for _, i in prep.repo.suite_steps["s"][:2]]})
    gold = p.gold()
    assert gold.bv == 1 and ("self", "iAddToCart") in gold.calls[(SHOP, "ishouldseetotal")]
    mine = ("#[When('I add :product to the cart')]\npublic function iAddToCart(string $product): void\n{\n"
            "    $this->indexPage->open();\n}")
    s = score_output(p, block(STEP.format("$this->iAddToCart($total);") + "\n\n" + mine))
    assert s.detail["replaced"] == 1 and s.calls[T] == {("self", "iAddToCart")} and ca(p, s) == (True, 1)


def test_a_future_constructor_is_replaced_but_never_expanded(prep):
    """Constructors can be future methods: an output constructor replaces a future one (here: the file shows
    nothing, as a file new at t_vis), not a collision; a constructor is never expanded."""
    prep._shown = {SHOP: set()}
    ctor = ("public function __construct(\n    private IndexPage $indexPage,\n    private Cart $cart,\n"
            "    private Checkout $checkout,\n) {\n    $this->cart->add('ctor');\n}")
    step = STEP.format("$this->checkout->assertTotal($total);\n    parent::__construct();")
    s = score_output(prep, block(step + "\n\n" + ctor))
    assert s.insert_status == "ok" and s.detail["replaced"] == 1
    assert s.calls[T] == {("checkout", "assertTotal"), ("self", "__construct")}


def test_a_helper_body_is_read_with_the_file_that_declares_it(prep):
    """Design specification at 1b287d0: an ancestor's future helper is read with the ancestor file's outline and
    Scope, so a static call to a repo class that only the ancestor imports (`Registry`, an alias `use` of
    BaseContext) survives the expansion, as `Language::getIdByLocale` does in PrestaShop's parseLocalizedRow; an
    output-defined helper is read with the target file, where `Registry` is not a repo class."""
    er_edit(prep, BASE, "namespace App\\Behat\\Context;\n",
            "namespace App\\Behat\\Context;\n\nuse App\\Behat\\Context\\ShopContext as Registry;\n")
    helper = php_fn("baseHelper", "Registry::make();", "$this->sharedStorage->get('x');")
    helper = helper.replace("private", "protected")
    er_edit(prep, BASE, "    private function secret",
            "    " + helper.replace("\n", "\n    ") + "\n\n    private function secret")
    inherited = score_output(prep, block(STEP.format("$this->baseHelper();")))
    assert inherited.calls[T] == {("Registry", "make"), ("sharedStorage", "get")}
    mine = score_output(prep, block(STEP.format("$this->baseHelper();") + "\n\n" + helper))
    assert mine.calls[T] == {("sharedStorage", "get")}


def test_an_output_method_replacing_a_future_one_grounds_like_it(prep):
    """An output method with the key of a FUTURE method grounds condition 5 by its name (§2.2b of the design
    specification), like the ER-tree member it
    replaces (Sylius ValidationTrait::getMaxCodeLength, `return 255;`); a helper with a NEW name still counts only
    through its expanded body (its name alone never grounds); calling the hidden future helper by name grounds."""
    er_edit(prep, BASE, "    private function secret",  # an ancestor's future helper, before anything is scored
            "    protected function baseHelper(): int\n    {\n        return 255;\n    }\n\n"
            "    private function secret")
    call = STEP.format("$this->computeTotal($total);")
    mine = "private function computeTotal(string $total): string\n{\n    return $total;\n}"
    same = score_output(prep, block(call + "\n\n" + mine))
    assert same.detail["replaced"] == 1 and same.conds["grounded"] and same.bv == 1
    new = score_output(prep, block(STEP.format("$this->probeHelper();") + "\n\n" + php_fn("probeHelper", "$x = 1;")))
    assert not new.conds["grounded"] and new.bv == 0
    assert score_output(prep, block(call)).conds["grounded"]
    override = "protected function baseHelper(): int\n{\n    return 255;\n}"
    over = score_output(prep, block(STEP.format("$this->baseHelper();") + "\n\n" + override))
    assert over.detail["replaced"] == 0 and over.conds["grounded"] and over.bv == 1


# ----- Bodyless and private methods, new files, C# access, overloads (fixed 25/09/2026), and the mutation pins --

OWN, PAR = "src/Behat/Context/OwnTrait.php", "src/Behat/Context/ParentTrait.php"


def trait_text(name: str, *methods: str) -> str:
    return "<?php\n\nnamespace App\\Behat\\Context;\n\ntrait " + name + "\n{\n" + "\n".join(methods) + "}\n"


def member(vis: str, name: str, arg: str) -> str:
    """A PHP method of a class or trait, indented, whose body makes one project call."""
    return f"    {vis} function {name}(): void\n    {{\n        $this->sharedStorage->get('{arg}');\n    }}\n"


def with_own_trait(shop: str) -> str:
    return shop.replace("extends BaseContext\n{\n", "extends BaseContext\n{\n    use OwnTrait;\n\n")


def base_text(*extra: str) -> str:
    """The fixture's BaseContext using ParentTrait, with a constructor, helperOld making a project call, and `extra`."""
    from tests.phase3_fixture import BASE as text
    text = text.replace("{\n    protected $sharedStorage;\n",
                        "{\n    use ParentTrait;\n\n    protected $sharedStorage;\n\n"
                        "    public function __construct()\n    {\n    }\n")
    text = text.replace("    protected function helperOld(): void\n    {\n    }\n",
                        member("protected", "helperOld", "old"))
    return text.replace("    private function secret", "".join(m + "\n" for m in extra) + "    private function secret")


def twin_prep(tmp_path, commits: list) -> Prepared:
    """The PHP fixture rebuilt from `commits` = [(time, {path: text})] (plus composer.json): the same steps, suite and
    test record (t_vis 2024-06-01, CREATE iShouldSeeTotal), ShopContext as the last commit that touches it writes it."""
    import subprocess
    from tests.phase3_fixture import COMPOSER, T2020, T2022, T2024, _fn, _git, record
    from v2tools.evaluate import prepare
    from v2tools.state import RepoContext
    work = tmp_path / "twin"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in commits:
        for fp, text in {"composer.json": COMPOSER, **files}.items():
            (work / fp).parent.mkdir(parents=True, exist_ok=True)
            (work / fp).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp_path / "twin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    head = _git(bare, "rev-parse", "HEAD").strip()
    shop = next(files[SHOP] for _, files in reversed(commits) if SHOP in files)
    line = lambda n: next(i for i, x in enumerate(shop.splitlines(), 1) if f"function {n}(" in x) + 1
    rows = [_fn(line("iOpenTheShop"), "iOpenTheShop", "Given", "I open the shop", T2020, shop),
            _fn(line("iAddToCart"), "iAddToCart", "When", "I add :product to the cart", T2022, shop),
            _fn(line("iShouldSeeTotal"), "iShouldSeeTotal", "Then", "I should see the total :total", T2024, shop)]
    cat = {f["fn_id"]: f for f in rows}
    ids = [f["annotations"][0]["def_id"] for f in rows]
    suite = {"s": [("I open the shop", (ids[0],)), ('I should see the total "5"', (ids[2],)),
                   ('I add "apple" to the cart', (ids[1],))]}
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), suite)
    return prepare(dict(record(cat), commit=head), repo, {"step_def_ids": [[ids[0]], [ids[2]]]})


def test_a_future_method_without_a_body_is_never_expanded(prep):
    """A method without a body (abstract, or an interface method) is never expanded (§2.2b of the design specification,
    fixed 25/09/2026): the call stays ("self", f), as for a constructor; Sylius ValidationTrait::resolveCurrentPage. It
    is still future (an output method may override it, and then its body expands), and the name of a bodyless method the
    output declares still never grounds on its own."""
    er_edit(prep, BASE, "    private function secret",
            "    abstract protected function resolvePage(): string;\n\n    private function secret")
    assert future_methods(prep)[SHOP]["resolvepage"].origin == BASE
    s = score_output(prep, block(STEP.format("$this->resolvePage();\n    $this->checkout->assertTotal($total);")))
    assert s.bv == 1 and s.calls[T] == {("self", "resolvePage"), ("checkout", "assertTotal")}
    abstract = "\n\nabstract protected function probeAbs(): void;"
    mine = score_output(prep, block(STEP.format("$this->probeAbs();") + abstract))
    assert mine.calls[T] == {("self", "probeAbs")} and not mine.conds["grounded"]
    body = "protected function resolvePage(): string\n{\n    $this->cart->add('p');\n    return 'p';\n}"
    over = score_output(prep, block(STEP.format("$this->resolvePage();") + "\n\n" + body))
    assert over.calls[T] == {("cart", "add")}


def test_a_private_method_of_a_parents_trait_is_not_future(tmp_path):
    """Fixed 25/09/2026: a private method of a trait that the PARENT uses is the parent's private (calling it from the
    child is a PHP runtime error), so it is never future; a private method of a trait the TARGET class uses is callable
    there and can be future (a gap found by mutation testing). The files whose privates count are those of
    RepoContext.access(target): the target and the traits it uses, through traits."""
    from tests.phase3_fixture import SHOP_2020, SHOP_2022, SHOP_2024, T2020, T2022, T2024
    own0 = trait_text("OwnTrait", member("private", "ownOld", "o0"))
    own1 = trait_text("OwnTrait", member("private", "ownOld", "o0"), member("private", "ownNew", "o1"))
    par0 = trait_text("ParentTrait", member("protected", "parOld", "p0"))
    par1 = trait_text("ParentTrait", member("protected", "parOld", "p0"), member("private", "parNew", "p1"),
                      member("protected", "parNewProt", "p2"))
    p = twin_prep(tmp_path, [(T2020, {BASE: base_text(), OWN: own0, PAR: par0, SHOP: with_own_trait(SHOP_2020)}),
                             (T2022, {SHOP: with_own_trait(SHOP_2022)}),
                             (T2024, {SHOP: with_own_trait(SHOP_2024), OWN: own1, PAR: par1})])
    assert {k: f.origin for k, f in future_methods(p)[SHOP].items()} == {
        "computetotal": SHOP, "ownnew": OWN, "parnewprot": PAR}
    s = score_output(p, block(STEP.format("$this->parNew();\n    $this->ownNew();")))
    assert s.calls[T] == {("self", "parNew"), ("sharedStorage", "get")}


def test_a_new_target_file_keeps_the_old_helpers_of_its_ancestors(tmp_path):
    """Future is defined by time (§2.2b of the design specification, fixed 25/09/2026). For a target file NEW at t_vis
    (state.t_view None) the prompt's class skeleton shows nothing, yet what its ancestors had before t_vis is not
    future: the shown names are those state.t_view shows for each in-repo ancestor (same gone and masked), less the
    ancestors' constructors and the privates of files outside RepoContext.access(target), which class_view leaves out
    too. Every method of the new file is future (its constructor too), and so is an ancestor's method added at or after
    t_vis. Before, the old PrestaShop getQueryBus expanded, and a stub calling only $this->getQueryBus() had CA 1 on
    8/19 new-file records."""
    from tests.phase3_fixture import SHOP_2024, T2020, T2024
    from v2tools.evaluate import shown_names
    own0 = trait_text("OwnTrait", member("private", "ownOld", "o0"))
    own1 = trait_text("OwnTrait", member("private", "ownOld", "o0"), member("private", "ownNew", "o1"))
    olds = (member("protected", "parOld", "p0"), member("private", "parOldPriv", "p3"))
    par0 = trait_text("ParentTrait", *olds)
    par1 = trait_text("ParentTrait", *olds, member("private", "parNew", "p1"), member("protected", "parNewProt", "p2"))
    flips = (member("private", "wasPrivate", "w"), member("protected", "nowPrivate", "n"))  # private before t_vis, ...
    flipped = (member("protected", "wasPrivate", "w"), member("private", "nowPrivate", "n"))  # ... or at HEAD
    p = twin_prep(tmp_path, [(T2020, {BASE: base_text(*flips), OWN: own0, PAR: par0}),
                             (T2024, {BASE: base_text(member("protected", "baseNew", "b1"), *flipped), OWN: own1,
                                      PAR: par1, SHOP: with_own_trait(SHOP_2024)})])
    assert p.repo.ctx.is_new_file(SHOP, p.record["t_vis"])
    shown = shown_names(p, SHOP)
    assert {"sharedStorage", "helperOld", "parOld", "ownOld"} <= shown
    assert not shown & {"secret", "parOldPriv", "__construct", "baseNew", "ownNew", "parNew", "parNewProt",
                        "wasPrivate", "nowPrivate"}
    assert set(future_methods(p)[SHOP]) == {"__construct", "iopentheshop", "iaddtocart", "computetotal", "ownnew",
                                            "parnewprot", "basenew", "wasprivate"}
    assert future_keys(p, SHOP) == {"__construct", "iopentheshop", "iaddtocart", "computetotal"}
    old = score_output(p, block(STEP.format("$this->helperOld();")))
    new = score_output(p, block(STEP.format("$this->baseNew();")))
    assert old.calls[T] == {("self", "helperOld")} and new.calls[T] == {("sharedStorage", "get")}


def test_a_csharp_member_without_an_access_modifier_is_private():
    """A C# member of a class with no access modifier is private (fixed 25/09/2026), so an
    ancestor's is never future (aux only); `internal` and `protected` ones are callable from the subclass, an interface
    member is public, and the target's own member without a modifier can be future."""
    base = ("namespace P\n{\n    public class BaseSteps\n    {\n        void Hidden()\n        {\n"
            "            _api.X();\n        }\n\n        protected void Shared()\n        {\n"
            "            _api.Y();\n        }\n\n"
            "        internal void Near()\n        {\n            _api.Z();\n        }\n    }\n}\n")
    iface = "namespace P\n{\n    public interface IShop\n    {\n        void Run();\n    }\n}\n"
    er = CS_ER.replace("Api _api;\n", "Api _api;\n\n        void Mine()\n        {\n            _api.M();\n        }\n")
    p = mini_prep("csharp", ["csharp_attribute"], CPATH, er)
    parent = lambda path, rel: (SimpleNamespace(path=path, relation=rel, status="resolved"), 1)
    p.repo.ctx.text = {CPATH: er, "Steps/BaseSteps.cs": base, "Steps/IShop.cs": iface}.get
    p.repo.ctx.ancestors = lambda path: [parent("Steps/BaseSteps.cs", "extends"),
                                         parent("Steps/IShop.cs", "implements")]
    p.repo.ctx.access = lambda path: {path}
    p._future = None
    assert {k: f.origin.split("/")[1] for k, f in future_methods(p)[CPATH].items()} == {
        ("Mine", 0): "ShopSteps.cs", ("Shared", 0): "BaseSteps.cs", ("Near", 0): "BaseSteps.cs", ("Run", 0): "IShop.cs"}


def jfuture_prep(er: str) -> Prepared:
    """java_prep over the ER text `er`; the prompt's class skeleton shows nothing, so every method of `er` is
    future."""
    p = java_prep()
    p.er_files[JPATH] = er
    p.repo.ctx.text = {JPATH: er}.get
    p.repo.ctx.access = lambda path: {path}
    p._future = None
    return p


JSTEP = '@Then("the total is {int}")\npublic void theTotalIs(int n) {\n    make(n);\n}'
JT = (JPATH, ("theTotalIs", 1))


def test_same_key_overloads_of_a_future_helper_all_expand():
    """A call has no arity (§2.2b of the design specification, fixed 25/09/2026): it resolves by
    name to the union of the overloads. Two future Java overloads with one key (name, arity) both
    expand (fineract WorkingCapitalLoanAccount.feature|106). expanded_futures (gold-helpers)
    lists one declaration per key, the first: an output that declares a key twice is a collision
    (3C). insert replaces only the FIRST method of the file with that key, so the other overload
    stays in the file after insertion, and it still expands: the copy keeps the gold's calls."""
    er = JAVA_ER.replace("\n}\n", "\n\n    private void make(String a) {\n        api.one(a);\n    }\n\n"
                         "    private void make(Integer a) {\n        api.two(a);\n    }\n}\n")
    p = jfuture_prep(er)
    s = score_output(p, jblock(JSTEP))
    assert s.bv == 1 and s.calls[JT] == {("api", "one"), ("api", "two")}
    helpers = expanded_futures(p, s)[JPATH]
    assert [f.text.split("(")[1].split(")")[0] for f in helpers] == ["String a"]
    copy = score_output(p, jblock(JSTEP + "\n\n" + helpers[0].text))
    assert copy.detail["replaced"] == 1 and copy.bv == 1 and copy.calls[JT] == {("api", "one"), ("api", "two")}


def test_masking_by_the_output_is_by_key_not_by_name():
    """Replacement is by KEY (§2.2b of the design specification, fixed 25/09/2026): a ("self", f) call that is not
    visible expands to the output's methods named f and to the future methods named f of the ER tree after insertion
    (an ancestor's, unless the target now declares its key; the target's own, except the one each output method
    replaced). On the expansion path is the function's own BODY, not its name (fineract WorkingCapital_COB|154: the
    step createWorkingCapitalLoanProduct() calls its future overload createWorkingCapitalLoanProduct(req))."""
    er = JAVA_ER.replace("\n}\n", "\n\n    private void make(Integer a, String b) {\n        api.two(a);\n    }\n}\n")
    p = jfuture_prep(er)
    step = JSTEP.replace("make(n);", 'make(n, "x");')
    other = score_output(p, jblock(step + "\n\nprivate void make() {\n    api.one(1);\n}"))
    assert other.bv == 1 and other.calls[JT] == {("api", "one"), ("api", "two")}
    same = score_output(p, jblock(step + "\n\nprivate void make(Integer a, String b) {\n    api.three(a);\n}"))
    assert same.detail["replaced"] == 1 and same.calls[JT] == {("api", "three")}
    own = score_output(p, jblock(step.replace("theTotalIs(int n)", "make(int n)")))
    assert own.calls[(JPATH, ("make", 1))] == {("self", "make"), ("api", "two")}
    assert own.cuts == {"depth": 0, "cycle": 1}
    assert [b.text.split("(")[1].split(")")[0] for _, b in own.expansions[(JPATH, ("make", 1))].helpers] == [
        "Integer a, String b"]  # the step's own body is on the path from the start: it is never re-read


def test_the_most_derived_declaration_of_a_future_key_wins(prep):
    """A gap found by mutation testing: when the target class and an ancestor both have a future method with one
    key, the target's (most derived) is the future method, and only its body expands."""
    er_edit(prep, BASE, "    private function secret", "    protected function computeTotal(string $total): string\n"
            "    {\n        $this->cart->add('base');\n        return $total;\n    }\n\n    private function secret")
    assert future_methods(prep)[SHOP]["computetotal"].origin == SHOP
    assert prep.gold().calls[(SHOP, "ishouldseetotal")] == {("checkout", "assertTotal")}


def test_a_replaced_future_step_function_leaves_its_er_bindings(prep):
    """A gap found by mutation testing: a future method the output replaces leaves the ER bindings of conditions 2
    and 3 (the `replaced` set), like a replaced shell. When the replacement no longer registers a step the ER-tree
    method bound (here its annotation, followed by a comment, is not one insert keeps), that step's binding changes."""
    prep._shown = {SHOP: set()}  # as for a file new at t_vis: every method of the file is future
    er_edit(prep, SHOP, "    #[Given('I open the shop')]\n", "    #[Given('I open the shop')] // entry\n")
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    helper = "public function iOpenTheShop(): void\n{\n    $this->indexPage->open();\n}"
    s = score_output(prep, block(gold + "\n\n" + helper))
    assert s.detail["replaced"] == 1 and not s.conds["no_ambiguity"] and s.bv == 0
    kept = score_output(prep, block(gold + "\n\n#[Given('I open the shop')]\n" + helper))
    assert kept.detail["replaced"] == 1 and kept.conds["no_ambiguity"] and kept.bv == 1


def test_a_logger_call_in_an_expanded_helper_never_counts(prep):
    """A gap found by mutation testing: a helper body is read like a step body, the logger exclusion
    included: a call on a field whose declared type is a logger (Scope.loggers) never enters a CA call set."""
    er_edit(prep, SHOP, "    private function computeTotal", "    private LoggerInterface $audit;\n\n"
            "    private function audited(): void\n    {\n        $this->audit->info('x');\n"
            "        $this->cart->add('y');\n    }\n\n    private function computeTotal")
    s = score_output(prep, block(STEP.format("$this->audited();")))
    assert s.calls[T] == {("cart", "add")}


def test_a_preexisting_shell_is_never_expanded_from_the_er_tree(prep):
    """D11 and §2.2b of the design specification at 3739dd6 (fixed 25/09/2026): the ER tree cuts
    only the annotations of the record's preexisting CREATE functions (p.shells), so their HEAD
    body is the answer; a call to one that the prompt's class skeleton does not show stays ("self",
    f), never its ER-tree body (a thin new step whose only line calls the hidden shell). An output
    that redefines the shell still replaces it, and then the OUTPUT's version expands."""
    from v2tools.evaluate import prepare
    r = {**prep.record, "create": [{**prep.record["create"][0], "preexisting": True}]}
    p = prepare(r, prep.repo, {"step_def_ids": [[i[0]] for _, i in prep.repo.suite_steps["s"][:2]]})
    assert p.shells[SHOP] == {"ishouldseetotal"} and "ishouldseetotal" in future_methods(p)[SHOP]
    gold = p.gold()
    assert gold.bv == 1 and gold.calls[(SHOP, "ishouldseetotal")] == {("checkout", "assertTotal")}
    thin = score_output(p, block(STEP.format("$this->iShouldSeeTotal($total);")))
    assert thin.bv == 1 and thin.calls[T] == {("self", "iShouldSeeTotal")} and ca(p, thin) == (True, 0)
    mine = "public function iShouldSeeTotal(string $total): void\n{\n    $this->cart->add($total);\n}"
    redefined = score_output(p, block(STEP.format("$this->iShouldSeeTotal($total);") + "\n\n" + mine))
    assert redefined.detail["replaced"] == 1 and redefined.calls[T] == {("cart", "add")}


# ----- PHP names compare case-insensitively (fixed 25/09/2026) -------------------------------------------------

def test_an_upper_case_php_constructor_is_the_constructor(prep):
    """`__CONSTRUCT`, in any case, is THE constructor (Sylius filtering_payment_requests|23): it replaces a future
    constructor (insert.is_constructor) and its promoted parameters are properties for condition 6
    (symbols.php_promoted; 3A's outline, left unchanged, only knows `__construct`)."""
    prep._shown = {SHOP: set()}  # as for a file new at t_vis: the constructor is future
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    for name in ("__construct", "__CONSTRUCT", "__Construct"):
        ctor = (f"public function {name}(\n    private IndexPage $indexPage,\n    private Cart $cart,\n"
                "    private Checkout $checkout,\n) {\n}")
        s = score_output(prep, block(gold + "\n\n" + ctor))
        assert s.detail["replaced"] == 1 and s.conds["resolved"] and s.bv == 1 and ca(prep, s) == (True, 1), name


def test_php_method_names_compare_in_any_case(prep):
    """PHP call matching ignores case (§2.2b of the design specification): CA compares PHP METHOD names
    case-insensitively, for gold and output alike (`->AddRequestData(` against the gold's `addRequestData`); condition 6
    resolves, and condition 5 grounds on, a method called in another case (`$this->PROBEFILL()`); a
    property name stays case-sensitive (`$this->Checkout` is another property). `SELF::CLASS` names no member."""
    up = score_output(prep, block(STEP.format("$this->checkout->AssertTotal($total);")))
    assert up.bv == 1 and ca(prep, up) == (True, 1)
    prop = score_output(prep, block(STEP.format("$this->Checkout->assertTotal($total);")))
    assert not prop.conds["resolved"] and prop.detail["unresolved"] == ["Checkout"]
    helper = "private function probeFill(string $t): void\n{\n    $this->checkout->assertTotal($t);\n}"
    own = score_output(prep, block(STEP.format("$this->PROBEFILL($total);") + "\n\n" + helper))
    assert own.conds["resolved"] and own.bv == 1 and ca(prep, own) == (True, 1)
    visible = score_output(prep, block(STEP.format("$this->IADDTOCART($total);")))
    assert visible.conds["resolved"] and visible.conds["grounded"] and visible.calls[T] == {("self", "IADDTOCART")}
    klass = score_output(prep, block(STEP.format("$x = SELF::CLASS;\n    $this->checkout->assertTotal($total);")))
    assert klass.bv == 1


# ----- Where a step definition is registered -------------------------------------------------------------------------

CPATH, RPATH = "Steps/ShopSteps.cs", "features/step_definitions/shop_steps.rb"
CS_ER = ("using TechTalk.SpecFlow;\n\nnamespace P\n{\n    [Binding]\n    public class ShopSteps\n    {\n"
         "        private readonly Api _api;\n    }\n}\n")
CGOLD = '[Then(@"^the total is (\\d+)$")]\npublic void TheTotalIs(int n)\n{\n    _api.Check(n);\n}'
J8_ER = ('package p;\n\npublic class Steps implements En {\n    private Api api;\n\n    public Steps() {\n'
         '        Given("I open the shop", () -> {\n            api.open();\n        });\n    }\n}\n')
J8GOLD = 'Then("the total is {int}", (Integer n) -> {\n    api.check(n);\n});'
RGOLD = "Then(/^the total is (\\d+)$/) do |n|\n  check_total(n)\nend"


def mini_prep(lang: str, styles: list[str], path: str, er: str) -> Prepared:
    """One target file of `lang` on a fake repo context (no ancestors, no static import, nothing shown in the
    prompt's class skeleton); one
    CREATE step, "the total is 5"."""
    resolver = SimpleNamespace(by_suffix={}, resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n))
    ctx = SimpleNamespace(language=lang, text={path: er}.get, parents=lambda p: [], ancestors=lambda p: [],
                          resolver=resolver, tree=[path])
    repo = Repo("Acme/M", lang, styles, {}, ctx, frozenset(), {})
    return Prepared({"suite": "s"}, repo, None, {path: er}, [path], {path: set()}, [("the total is 5", False)],
                    [], [], {path: {"s"}}, _future={path: {}}, _shown={path: set()})


def fence(lang: str, path: str, code: str) -> str:
    return f"{TICKS}{lang}\n{'#' if lang == 'ruby' else '//'} FILE: {path}\n{code}\n{TICKS}"


def pad(code: str, n: int) -> str:
    return "\n".join(" " * n + x if x.strip() else "" for x in code.split("\n"))


def test_a_step_nested_in_a_method_body_never_binds(prep):
    """A step annotation binds only when it lies before the `{` that
    opens its method item's body; one on a method of an anonymous class, of a local class or on a local function in
    the body is never registered (Behat, Cucumber-JVM, SpecFlow), though stepdefs reads the whole item text."""
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    doc = gold.replace("#[Then('I should see the total :total')]", "/**\n * @Then I should see the total :total\n */")
    for inner in (gold, doc):
        pa2 = ("private $probeX;\n\npublic function probeInit(string $total): void\n{\n"
               "    $this->probeX = new class {\n" + pad(inner, 8) + "\n    };\n}")
        s = score_output(prep, block(pa2))
        assert s.conds["syntax"] and not s.conds["binding"] and s.bv == 0 and ca(prep, s) == (True, 0), inner
        assert score_output(prep, block(inner)).bv == 1  # an attribute or a docblock before `function` binds
    p = java_prep()
    anon = "public void probeHolder(int n) {\n    Object probeX = new Object() {\n" + pad(JGOLD, 8) + "\n    };\n}"
    local = "public void probeHolder(int n) {\n    class In {\n" + pad(JGOLD, 8) + "\n    }\n}"
    for code in (anon, local):
        s = score_output(p, jblock(code))
        assert s.conds["syntax"] and not s.conds["binding"] and s.bv == 0, code
    assert score_output(p, jblock(JGOLD.replace(")\npublic", ") public"))).bv == 1  # annotation on the signature line
    cs = mini_prep("csharp", ["csharp_attribute"], CPATH, CS_ER)
    assert score_output(cs, fence("csharp", CPATH, CGOLD)).bv == 1
    fn = "public void ProbeHolder(int n)\n{\n" + pad(CGOLD.replace("public void", "void"), 4) + "\n}"
    s = score_output(cs, fence("csharp", CPATH, fn))
    assert s.conds["syntax"] and not s.conds["binding"] and s.bv == 0


def test_a_step_nested_in_a_lambda_or_a_ruby_block_never_binds():
    """A java8 lambda item or a Ruby block item registers its own step (the registration call that opens
    it) and nothing inside its body."""
    jl = mini_prep("java", ["java_annotation", "java_lambda"], JPATH, J8_ER)
    assert score_output(jl, jblock(J8GOLD)).bv == 1
    s = score_output(jl, jblock('Then("probe outer", () -> {\n' + pad(J8GOLD, 4) + "\n});"))
    assert s.conds["syntax"] and not s.conds["binding"]
    rb = mini_prep("ruby", ["ruby_block"], RPATH, "Given(/^I open the shop$/) do\n  open_shop\nend\n")
    assert score_output(rb, fence("ruby", RPATH, RGOLD)).conds["binding"]
    s = score_output(rb, fence("ruby", RPATH, "Given(/^probe outer$/) do\n" + pad(RGOLD, 2) + "\nend"))
    assert s.conds["syntax"] and not s.conds["binding"]


def test_a_java8_lambda_outside_a_top_level_statement_never_binds():
    """A cucumber-java8 lambda is a step only as a top-level
    statement of the block or of a constructor body; in a `static {}` block, a method body or an anonymous class it
    stays where it is (Cucumber-java8 never registers it there) and never binds."""
    jl = mini_prep("java", ["java_annotation", "java_lambda"], JPATH, J8_ER)
    for code in ("static {\n" + pad(J8GOLD, 4) + "\n}",
                 "static {\n    ghostStaticZ.init();\n" + pad(J8GOLD, 4) + "\n}",
                 "public void probeHolder() {\n" + pad(J8GOLD, 4) + "\n}",
                 "private Object probeX = new Object() {\n    {\n" + pad(J8GOLD, 8) + "\n    }\n};"):
        s = score_output(jl, jblock(code))
        assert not s.conds["binding"] and s.bv == 0 and not any(k == "lambda" for _, k, _ in s.items.values()), code


def test_a_php_step_with_a_table_needs_a_parameter_that_takes_it(prep):
    """Condition 4, as Behat 3.26.0 runs it (design specification at fe73dc9): required <= N with no upper bound, but
    a table or doc string that no parameter takes stops the run (shown by a real run, "fewer_table"): a TableNode
    parameter, or total >= N."""
    prep.create_steps = [(s, True) for s, _ in prep.create_steps]
    body = "{\n    $this->checkout->assertTotal($total);\n}"
    attr = "#[Then('I should see the total :total')]\npublic function t("
    assert not score_output(prep, block(attr + "string $total): void\n" + body)).conds["arity"]
    assert score_output(prep, block(attr + "string $total, TableNode $table): void\n" + body)).conds["arity"]
    assert score_output(prep, block(attr + "string $total, $rows): void\n" + body)).conds["arity"]


# ----- Lambda steps in a constructor, scoped calls in another case, the PHP tag (fixed 26/09/2026) ------------

def test_java8_lambdas_in_the_outputs_own_constructor_bind():
    """Lambda steps (fineract WorkingCapital_COB|154) written inside the output's own
    no-argument constructor, on a class that declares none, bind as the top-level form does; on a class whose visible
    constructor the output rewrites, it stays a collision (dev commands.service|23)."""
    bare = mini_prep("java", ["java_annotation", "java_lambda"], JPATH,
                     "package p;\n\npublic class Steps implements En {\n    private Api api;\n}\n")
    top = score_output(bare, jblock(J8GOLD))
    for head in ("public Steps() {", "@Deprecated\npublic Steps() {"):
        s = score_output(bare, jblock(head + "\n" + pad(J8GOLD, 4) + "\n}"))
        got, want = s.calls.get(next(iter(s.binders.values()))), top.calls[next(iter(top.binders.values()))]
        assert s.insert_status == "ok" and s.bv == 1 == top.bv and got == want, head
    held = mini_prep("java", ["java_annotation", "java_lambda"], JPATH, J8_ER)
    assert score_output(held, jblock(J8GOLD)).bv == 1
    assert score_output(held, jblock("public Steps() {\n" + pad(J8GOLD, 4) + "\n}")).insert_status == "collision"


def test_php_scoped_calls_to_a_method_in_another_case(prep):
    """`self::PROBEFILL()` and `STATIC::PrObEfIlL()` call the output's helper
    `probeFill` (PHP method names ignore case): condition 6 resolves them, the call set expands the helper, and the
    helper's name alone never grounds."""
    helper = "private function probeFill(string $t): void\n{\n    $this->checkout->assertTotal($t);\n}"
    for call in ("self::PROBEFILL($total);", "STATIC::PrObEfIlL($total);", "static::probeFill($total);"):
        s = score_output(prep, block(STEP.format(call) + "\n\n" + helper))
        assert s.conds["resolved"] and s.bv == 1 and s.calls[T] == {("checkout", "assertTotal")}, call
        assert ca(prep, s) == (True, 1), call
    empty = "private function probeFill(string $t): void\n{\n    $x = $t;\n}"
    assert not score_output(prep, block(STEP.format("self::PROBEFILL($total);") + "\n\n" + empty)).conds["grounded"]
    fut = score_output(prep, block(STEP.format("$this->checkout->assertTotal(self::COMPUTETOTAL($total));")))
    assert fut.conds["resolved"] and fut.conds["grounded"] and fut.bv == 1


def test_a_php_block_that_opens_with_the_php_tag(prep):
    """A PHP block (Sylius channel_validation|100) whose code starts with the `<?php`
    open tag (on its own line or before code) or ends with `?>` scores as the same block without them, for every
    condition (insert.php_open_tag, applied by split_code)."""
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    plain = score_output(prep, block(gold))
    for code in ("<?php\n\n" + gold, "<?php " + gold, "  <?PHP\n" + gold + "\n?>", "<?php\n" + gold + "\n\n?>\n"):
        s = score_output(prep, block(code))
        assert s.bv == plain.bv == 1 and s.conds == plain.conds and s.calls == plain.calls, code
        assert ca(prep, s) == (True, 1), code
    assert score_output(prep, block("<?php\nnamespace App;\n\n" + gold)).bv == 0  # only the tag is tolerated


# ----- D11 (e), fixed 26/09/2026 (§2.2b of the design specification): delegation to a visible preexisting shell ---

PUT = "I put :product in the cart"


def shell_prep(tmp_path, body: str = "$this->cart->add($product);") -> Prepared:
    """The PHP fixture where the test record's answer only adds `#[When('I put :product in the cart')]` (2024-06-01) to
    iAddToCart, a PREEXISTING shell that the prompt's class skeleton shows (in the file since 2022-01-01, body
    `body`); record steps: "I open
    the shop" (REUSE), 'I put "apple" in the cart' (CREATE)."""
    from tests.phase3_fixture import SHOP_2020, SHOP_2022, SHOP_2024, T2020, T2022, T2024, _fn, record
    old = "        $this->cart->add($product);\n"
    s2022, s2024 = SHOP_2022.replace(old, f"        {body}\n"), SHOP_2024.replace(old, f"        {body}\n")
    s2024 = s2024.replace("    #[When('I add :product to the cart')]\n",
                          f"    #[When('I add :product to the cart')]\n    #[When('{PUT}')]\n")
    p = twin_prep(tmp_path, [(T2020, {BASE: base_text(), SHOP: SHOP_2020}), (T2022, {SHOP: s2022}),
                             (T2024, {SHOP: s2024})])
    lines = s2024.splitlines()
    line = lambda n: next(i for i, x in enumerate(lines, 1) if f"function {n}(" in x) + 1
    add = _fn(line("iAddToCart"), "iAddToCart", "When", "I add :product to the cart", T2022, s2024)
    put = next(i for i, x in enumerate(lines, 1) if f"'{PUT}'" in x)
    add["annotations"].append({"def_id": f"{SHOP}:{put}", "line": put, "keyword": "When", "pattern": PUT,
                               "time": T2024})
    rows = [_fn(line("iOpenTheShop"), "iOpenTheShop", "Given", "I open the shop", T2020, s2024), add,
            _fn(line("iShouldSeeTotal"), "iShouldSeeTotal", "Then", "I should see the total :total", T2024, s2024)]
    cat = {f["fn_id"]: f for f in rows}
    d = {f["annotations"][-1]["pattern"]: f["annotations"][-1]["def_id"] for f in rows}
    suite = {"s": [("I open the shop", (d["I open the shop"],)), ('I put "apple" in the cart', (d[PUT],)),
                   ('I add "apple" to the cart', (add["annotations"][0]["def_id"],)),
                   ('I should see the total "5"', (d["I should see the total :total"],))]}
    from v2tools.evaluate import prepare
    from v2tools.state import RepoContext
    ctx = RepoContext("Acme/Shop", "php", tmp_path / "twin.git", p.record["commit"], cat)
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), suite)
    r = dict(record(cat), id="Acme/Shop|features/put.feature|3|s", feature_path="features/put.feature",
             commit=p.record["commit"], steps=["I open the shop", 'I put "apple" in the cart'],
             scenario='Scenario: Put\n  Given I open the shop\n  When I put "apple" in the cart',
             step_fns=[rows[0]["fn_id"], add["fn_id"]], create_def_ids=[d[PUT]],
             create=[{"fn_id": add["fn_id"], "path": SHOP, "code": add["code"], "preexisting": True,
                      "post_cutoff_code": False}])
    return prepare(r, repo, {"step_def_ids": [[d["I open the shop"]], [d[PUT]]]})


PUT_STEP = "#[When('I put :product in the cart')]\npublic function iPut(string $product): void\n{{\n    {}\n}}"


def call_precision(p: Prepared, s) -> dict:
    """A local reimplementation of describe.call_overlap (n_calls = |O|, call_precision = |O ∩ G| / |O|): O the
    project calls of the output's CREATE-step binders after helper expansion (Score.calls), G the gold's; PHP keys
    as (receiver, casefold(method)) so a differing case is the same call. Kept local so this test file (and
    tests/test_controls.py) need not import v2tools.describe; tests/test_describe.py pins it to call_overlap."""
    g = p.gold()
    key = (lambda cs: {(r, f.casefold()) for r, f in cs}) if p.repo.language == "php" else set
    o = key(set().union(*(s.calls.get(f, set()) for f in set(s.binders.values()))))
    gold = key(set().union(*(g.calls.get(f, set()) for f in set(g.binders.values()))))
    return {"n_calls": len(o), "call_precision": round(len(o & gold) / len(o), 4) if o else None}


def test_a_step_calling_a_visible_preexisting_shell_shares_its_gold_call(tmp_path):
    """D11 (e), fixed 26/09/2026: when a gold binder is a preexisting shell that the prompt's class skeleton shows (its
    name in shown_names of its file; the gold only adds an annotation), its gold call set also holds
    ("self", shell name), so an output that writes a new step calling the existing function reaches CA like the gold;
    PHP compares the name in any case. Only the gold's call set changes: the output's is scored as before, and the gold
    answer scored as an output (gold_output, the gold control) keeps its own call set."""
    import re
    from v2tools.evaluate import shown_names
    p = shell_prep(tmp_path)
    ident = (SHOP, "iaddtocart")
    assert p.shells[SHOP] == {"iaddtocart"} and "iAddToCart" in shown_names(p, SHOP)
    g = p.gold()
    assert g.bv == 1 and g.binders == {'I put "apple" in the cart': ident}
    assert g.calls[ident] == {("cart", "add"), ("self", "iAddToCart")}
    for call in ("$this->iAddToCart($product);", "$this->IADDTOCART($product);", "self::iAddToCart($product);"):
        s = score_output(p, block(PUT_STEP.format(call)))
        assert s.bv == 1 and s.calls[(SHOP, "iput")] == {("self", re.search(r"(\w+)\(", call).group(1))}, call
        assert ca(p, s) == (True, 1) and call_precision(p, s)["call_precision"] == 1.0, call
    other = score_output(p, block(PUT_STEP.format("$this->iOpenTheShop();")))
    assert other.bv == 1 and ca(p, other) == (True, 0)
    raw = p.gold_output()
    assert raw.bv == 1 and raw.binders == g.binders and raw.calls[ident] == {("cart", "add")}
    assert ca(p, raw) == (True, 1) and call_precision(p, raw) == {"n_calls": 1, "call_precision": 1.0}


def test_a_visible_shell_with_no_project_call_adds_no_condition(tmp_path):
    """D11 (e), the shell delegation, on an empty gold call set (the specification is silent): nothing is added, so
    D11 (e) never adds a CA condition and never changes ca_defined (2 such binders in test and 5 in dev, all in
    records whose gold fails BV)."""
    p = shell_prep(tmp_path, "$this->cart;")
    g = p.gold()
    assert g.bv == 1 and g.calls[(SHOP, "iaddtocart")] == set() and ca(p, g) == (False, 0)


# ----- The two equivalent step-annotation forms (fixed 26/09/2026) --------------------------------------------

def test_a_function_scores_the_same_in_either_annotation_form(prep):
    """Fixed 26/09/2026: the step functions the prompt shows keep their annotations as written at HEAD (Sylius moved
    from docblock `@Then x` to PHP 8 attributes `#[Then('x')]` on 11-13/02/2026; renalware dropped the parentheses of
    `Then(/x/)`), so an output may copy either form. The SAME function in the two forms scores identically: conditions
    1-6, binders, items, the CA call sets and their helper expansion (helpers read, cuts), CA and every other
    Score/evaluate detail except step_styles; only what measures the form itself differs, by design: detail step_styles
    (php_docblock_steps, php_attribute_steps; §2.3 of the design specification: the ER image registers attributes only).
    Turnip and regex patterns (N = every capture group, Behat), a `\\'` escape of the attribute literal included; Ruby
    `Then(/re/) do`, `Then /re/ do` and the `%r{}` forms too. The describe-level half
    (call_overlap/n_calls/call_precision, conv_pattern, conv_naming, hallucinated, and conv_style itself) is
    tests.test_describe.test_a_function_in_either_annotation_form_agrees_on_conventions_and_hallucination, split out
    so this file need not import v2tools.describe."""

    def view(p, s):
        exp = {i: (e.calls, [(n, b.origin) for n, b in e.helpers], e.depth_cuts, e.cycle_cuts)
               for i, e in s.expansions.items()}
        detail = {k: v for k, v in s.detail.items() if k != "step_styles"}
        return (s.extract_status, s.insert_status, s.conds, s.bv, s.binders, s.calls, s.cuts, detail, exp,
                sorted((str(i), k) for i, (_, k, _) in s.items.items()), ca(p, s))
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    attr = "#[Then('I should see the total :total')]"
    body, g = gold.split(attr, 1)[1], (SHOP, "ishouldseetotal")
    forms = [(attr, "/**\n * @Then I should see the total :total\n */")]
    for pattern in ("I should see the (customer's) total :total", "/^I should see the (?:grand )?total \"(\\d+)\"$/"):
        forms.append(("#[Then('" + pattern.replace("'", "\\'") + "')]", "/**\n * @Then " + pattern + "\n */"))
    for a, d in forms:  # a `'` is written `\'` in the attribute literal
        sa, sd = score_output(prep, block(a + body)), score_output(prep, block(d + body))
        assert sa.bv == 1 and ca(prep, sa) == (True, 1) and sa.calls[g] == {("checkout", "assertTotal")}, a
        assert [n for n, _ in sa.expansions[g].helpers] == ["computeTotal"], a  # the future helper is expanded
        assert view(prep, sa) == view(prep, sd), a
        assert (sa.detail["step_styles"], sd.detail["step_styles"]) == ({"php_attribute": 1}, {"php_docblock": 1})
    rb = mini_prep("ruby", ["ruby_block"], RPATH, "Given(/^I open the shop$/) do\n  open_shop\nend\n")
    for re_ in ("/^the total is (\\d+)$/", "%r{^the total is (\\d+)$}"):  # the gold in renalware HEAD's form
        rb._gold = bare = score_output(rb, fence("ruby", RPATH, f"Then {re_} do |n|\n  check_total(n)\nend"))
        paren = score_output(rb, fence("ruby", RPATH, f"Then({re_}) do |n|\n  check_total(n)\nend"))
        assert paren.conds["binding"] and paren.conds["arity"] and view(rb, paren) == view(rb, bare), re_


def test_two_fences_of_one_file_score_as_one_whatever_their_php_tags(prep):
    """The code extraction rule (§1.4) applies per block (design specification at c6999dc, fixed 27/09/2026): the gold
    function and the helper it calls, written as two fences of the same file, score as the same code in one fence
    whether each fence opens with `<?php`, closes with `?>`, both, or only one does (before: BV 0, CA 0 when every fence
    opened with `<?php`)."""
    gold = next(f["code"] for f in prep.repo.fns.values() if "iShouldSeeTotal" in f["code"])
    helper = "private function computeTotal(string $total): string\n{\n    return $total;\n}"
    view = lambda s: (s.bv, s.conds, ca(prep, s), s.binders, s.calls, s.cuts)
    one = score_output(prep, block(gold + "\n\n" + helper))
    assert one.bv == 1 and ca(prep, one) == (True, 1)
    for a, b in (("<?php\n\n" + gold, "<?php\n\n" + helper), (gold + "\n?>", helper + "\n?>"),
                 ("<?php\n" + gold, helper), (gold, "<?php\n" + helper + "\n?>")):
        assert view(score_output(prep, block(a) + "\n" + block(b))) == view(one), (a, b)


# ----- Rules no earlier test pinned -----------------------------------------------------------------------------

OK = "$this->checkout->assertTotal($total);"


def test_ca_is_zero_when_bv_is_zero_even_if_the_calls_agree(prep):
    """CA = BV × agreement (§2.2b of the design specification): an output whose calls meet the gold's but that fails
    condition 6."""
    s = score_output(prep, block(STEP.format("$this->checkout->assertTotal($this->ghostService->x($total));")))
    assert s.bv == 0 and ca(prep, s) == (True, 0)


def test_binding_needs_exactly_one_generated_function_for_every_create_step(prep):
    """Condition 2: two generated functions for one step, an ER-tree function that still binds the
    CREATE step, an output with no step at all, and a CREATE step nothing binds all fail binding."""
    two = STEP.format(OK) + "\n\n" + STEP.replace("function t(", "function t2(").format(OK)
    assert not score_output(prep, block(two)).conds["binding"]
    head = next(ids for su, s, ids in prep.bound if s == "I open the shop")
    prep.bound.append(("s", 'I should see the total "5"', head))
    assert not score_output(prep, block(STEP.format(OK))).conds["binding"]
    assert not score_output(prep, block("public function helperOnly(): void\n{\n    $this->checkout->x();\n}")).conds["binding"]
    prep.bound.pop()
    prep.create_steps += [("I pay now", False)]
    assert not score_output(prep, block(STEP.format(OK))).conds["binding"]


def test_a_repeated_create_step_counts_once(prep):
    """A CREATE step written twice in the scenario is one step to bind (gold BV: 43 test records, 10 dev)."""
    from v2tools.evaluate import prepare
    r = dict(prep.record)
    r["steps"] = r["steps"] + [r["steps"][-1]]
    r["step_kinds"] = r["step_kinds"] + [r["step_kinds"][-1]]
    ids = {s: [i[0]] for s, i in prep.repo.suite_steps["s"]}
    p = prepare(r, prep.repo, {"step_def_ids": [ids[s] for s in r["steps"]]})
    assert len(p.create_steps) == 1 and p.gold().bv == 1


def test_an_unparsable_output_is_not_judged_further(prep):
    """When condition 1 fails, every condition is 0: gold plus a function that does not parse."""
    s = score_output(prep, block(prep.record["create"][0]["code"] + "\n\npublic function broken(): void\n{\n    $x = ;\n}"))
    assert s.conds == {c: False for c in s.conds}


def test_condition_1_reads_the_tree_of_the_whole_file():
    """A broken import line (it joins the file's import block, so split_code never sees it as raw) fails syntax."""
    assert not score_output(java_prep(), jblock("import a.b..C;\n\n" + JGOLD)).conds["syntax"]


def test_a_tablenode_parameter_takes_the_table(prep):
    """A step with a table needs a method that can take it; a TableNode parameter does, even when the total
    number of parameters is below N."""
    prep.create_steps = [('I should see the total "5"', True)]
    out = "#[Then('I should see the total :total')]\npublic function t(TableNode $table): void\n{\n    " + OK + "\n}"
    assert score_output(prep, block(out)).conds["arity"]


def test_an_er_tree_constructor_is_never_expanded(prep):
    """A constructor never expands, on the ER side too: a call to the ER tree's own constructor stays
    ("self", "__construct")."""
    prep._shown = {SHOP: set()}
    s = score_output(prep, block(STEP.format(OK + "\n    $this->__construct();")))
    assert ("self", "__construct") in s.calls[T]


def test_condition_5_needs_a_binder(prep):
    """An output that binds no step grounds nothing (bv_grounded = 0)."""
    s = score_output(prep, block("public function helperOnly(): void\n{\n    $this->checkout->x();\n}"))
    assert not s.conds["grounded"]


def test_a_style_whose_read_fails_gives_nothing_and_the_others_still_bind(prep):
    """When one style's reader raises (an attribute on an arrow function, with no `function` after it), that style
    gives no definition and the other style (the docblock) still binds the step."""
    out = ("/**\n * @Then I should see the total :total\n */\npublic function t(string $total): void\n{\n    " + OK +
           "\n    $f = #[Then('trailing')] fn() => 1;\n}")
    assert score_output(prep, block(out)).bv == 1


def test_future_methods_keep_the_first_declaration_of_a_key():
    """Of two same-key overloads, future_methods (and future_methods.jsonl) keeps the first declaration."""
    er = JAVA_ER.replace("\n}\n", "\n\n    private void make(String a) {\n        api.one(a);\n    }\n\n"
                         "    private void make(Integer a) {\n        api.two(a);\n    }\n}\n")
    fm = future_methods(jfuture_prep(er))[JPATH]
    assert "String a" in fm[("make", 1)].text


def test_a_new_function_counts_only_in_the_suites_its_own_file_serves(prep):
    """With two target files, a step of suite "other" (served only by the
    first) that a new function of the SECOND file matches keeps its binding: no ambiguity (gold BV: 34 test
    records, 25 dev)."""
    cart = next(ids for s, ids in prep.repo.suite_steps["s"] if "cart" in s)
    prep.repo.suite_steps["other"] = [("I pay now", cart)]
    p = make_prep_with(prep, {"other"})
    p.targets = p.targets + [BASE]
    p.served[BASE], p.shells[BASE] = {"s"}, set()
    out = (block(STEP.format(OK)) + "\n" + f"{TICKS}php\n// FILE: {BASE}\nuse Behat\\Step\\When;\n\n#[When('I pay now')]\n"
           "public function payBase(): void\n{\n    $this->x();\n}\n" + TICKS)
    s = score_output(p, out)
    assert s.detail["step_styles"] == {"php_attribute": 2} and s.conds["no_ambiguity"]


def test_the_own_suite_binding_decides_a_create_step(prep):
    """The CREATE step text bound in ANOTHER served suite to a HEAD function does
    not stop the gold from binding it in its own suite (gold BV: 33 test records, 25 dev)."""
    head = next(ids for su, s, ids in prep.bound if s == "I open the shop")
    prep.bound.append(("other", 'I should see the total "5"', head))
    assert score_output(prep, block(prep.record["create"][0]["code"])).conds["binding"]


def test_open_resolution_needs_condition_6():
    """An open path (an external static import) counts only when
    condition 6 passes (swap open_resolution on test: +63)."""
    imp = "import static org.assertj.core.api.Assertions.assertThat;\n\n"
    ok = JGOLD.replace("api.check(n);", "api.check(n);\n    assertThat(n);")
    assert score_output(java_prep(), jblock(imp + ok)).detail["open_resolution"]
    ghost = ok.replace("assertThat(n);", "assertThat(n);\n    ghostField.go();")
    s = score_output(java_prep(), jblock(imp + ghost))
    assert not s.conds["resolved"] and not s.detail["open_resolution"]
