"""One test per rule of the visible state V(r) (rules 1–5 of the design specification, rendering, fail-closed git
reads)."""
import pytest

from tests import phase3_chain_fixture as fx
from tests.phase3_chain_fixture import BASE, CHILD, LATE, LATE_PARENT, OWN, T2020, T2021, T2022, T2023, T2024
from tests.phase3_fixture import SHOP, catalog, make_repo, record
from v2tools import gitio
from v2tools.inherit import Resolver
from v2tools.state import (MAX_DEPTH, RepoContext, class_view, future_ref, member_refs, member_use, param_names,
                           self_contained, skeleton, t_view)


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    bare, head = make_repo(tmp_path_factory.mktemp("shop"))
    return RepoContext("Acme/Shop", "php", bare, head, catalog())


@pytest.fixture(scope="module")
def chain_git(tmp_path_factory):
    return fx.chain_repo(tmp_path_factory.mktemp("chain"))


@pytest.fixture(scope="module")
def chain(chain_git):
    return RepoContext(fx.REPO, "php", *chain_git, fx.chain_catalog())


@pytest.fixture(scope="module")
def rebased(tmp_path_factory):
    return RepoContext("Acme/Rebase", "php", *fx.rebase_repo(tmp_path_factory.mktemp("rebase")), {})


@pytest.fixture(scope="module")
def cs(tmp_path_factory):
    return RepoContext("Acme/Cs", "csharp", *fx.csharp_repo(tmp_path_factory.mktemp("cs")), {})


@pytest.fixture(scope="module")
def java(tmp_path_factory):
    return RepoContext("Acme/Java", "java", *fx.java_repo(tmp_path_factory.mktemp("java")), {})


@pytest.fixture(scope="module")
def deep(tmp_path_factory):
    return RepoContext("Acme/Deep", "php", *fx.deep_repo(tmp_path_factory.mktemp("deep")), {})


def fid(ctx, name):
    return next(f for f, row in ctx.fns.items() if f"function {name}(" in row["code"])


def own(s):
    return s.split("    // from ", 1)[0]


def group(s, cls):
    return s.split(f"    // from {cls} (", 1)[1].split("    // from ", 1)[0]


def answer_record(call):
    code = f"#[Then('I answer')]\npublic function iAnswer(): void\n{{\n    {call}\n}}"
    return {"t_vis": T2024, "create": [{"fn_id": f"{CHILD}:x", "path": CHILD, "code": code}]}


def test_answer_with_old_annotation_and_name_is_hidden_by_gone_alone(shop):
    answer = next(f for f, row in shop.fns.items() if "iShouldSeeTotal" in row["code"])
    assert "iShouldSeeTotal" not in skeleton(shop, SHOP, T2024 + 1, {answer})
    assert "public function iShouldSeeTotal(string $total): void" in skeleton(shop, SHOP, T2024 + 1, set())


def test_step_with_an_old_name_is_hidden_until_its_first_annotation(chain):
    assert chain.chain_times(CHILD)["stepLater"] == T2020
    assert "stepLater" not in skeleton(chain, CHILD, T2021, set())
    assert "public function stepLater(): void" in skeleton(chain, CHILD, T2022 + 1, set())


def test_member_moved_between_class_and_parent_keeps_its_first_time(chain):
    times = chain.chain_times(CHILD)
    assert times["movedUp"] == T2020 and times["movedDown"] == T2020
    s = skeleton(chain, CHILD, T2021, set())
    assert "protected function movedUp(): void" in own(s)  # in the child before T, in the parent at HEAD
    assert "protected function movedDown(): void" in group(s, "BaseContext")  # the other way round


def test_member_first_added_to_the_parent_at_t_is_masked_by_rule_3(chain):
    removed = []
    class_view(chain, CHILD, T2024, set(), removed=removed)
    assert ("addedAtT", BASE, "rule3") in {(m.names[0], origin, why) for m, origin, why in removed}
    assert "addedAtT" in chain.future(CHILD, T2024) and "addedAtT" not in chain.future(CHILD, T2024 + 1)
    assert "addedAtT" not in skeleton(chain, CHILD, T2024, set())
    assert "protected function addedAtT(): void" in group(skeleton(chain, CHILD, T2024 + 1, set()), "BaseContext")


def test_masked_infrastructure_member_is_hidden_and_makes_a_body_that_uses_it_future(chain):
    masked = frozenset({(BASE, "helperInfra")})
    user = chain.fns[fid(chain, "untouchedStep")]
    assert "helperInfra" not in skeleton(chain, CHILD, T2024 + 1, set(), masked)
    assert future_ref(chain, user, T2024 + 1, masked)
    assert "protected function helperInfra(): void" in skeleton(chain, CHILD, T2024 + 1, set())
    assert not future_ref(chain, user, T2024 + 1)


def test_php_plain_constructor_parameter_added_at_t_is_left_out_of_the_signature(chain):
    assert chain.times(CHILD)["params"]["newDep"] == T2024  # recorded for audit; the pre-T constructor is what is shown
    s = skeleton(chain, CHILD, T2024, {fid(chain, "iAnswer")})
    assert "public function __construct(OldDep $oldDep)\n" in s and "NewDep" not in s
    assert "public function __construct(OldDep $oldDep, NewDep $newDep)\n" in skeleton(chain, CHILD, T2024 + 1, set())


def test_csharp_plain_constructor_parameter_and_its_field_added_at_t_are_left_out(cs):
    assert cs.times(fx.CS)["params"]["newDep"] == T2024
    s = skeleton(cs, fx.CS, T2024, set())
    assert "public Steps(OldDep oldDep)\n" in s and "NewDep" not in s and "_newDep" not in s
    later = skeleton(cs, fx.CS, T2024 + 1, set())
    assert "public Steps(OldDep oldDep, NewDep newDep)\n" in later and "private readonly NewDep _newDep;" in later


def test_private_members_come_only_from_traits_the_class_uses_itself(chain):
    s = skeleton(chain, CHILD, T2024 + 1, set())
    assert chain.access(CHILD) == {CHILD, OWN}
    assert "private function ownTraitSecret(): void" in group(s, "OwnTrait")
    assert "public function parentTraitPublic(): void" in group(s, "ParentTrait")
    assert "parentTraitSecret" not in s and "baseSecret" not in s


def test_private_member_of_a_parents_trait_is_not_shown_for_a_member_that_moved_out_of_it(chain):
    assert "tock" not in skeleton(chain, CHILD, T2023, set())


def test_parent_promoted_property_is_rendered_as_a_property_declaration(chain):
    s = skeleton(chain, CHILD, T2024 + 1, set())
    assert "    protected Foo $bar;" in group(s, "BaseContext").splitlines()


def test_parent_method_overridden_by_the_class_is_shown_once_from_the_class(chain):
    s = skeleton(chain, CHILD, T2024 + 1, set())
    assert s.count("function sharedName(") == 1 and "function sharedName(" in own(s)


def test_java_overloads_in_one_class_are_all_shown(java):
    lines = skeleton(java, fx.JAVA, T2022, set()).splitlines()
    assert "    public void go(String a) { /* … */ }" in lines and "    public void go(String a, int b) { /* … */ }" in lines


def test_helper_whose_parameters_change_with_the_answer_is_shown_as_it_was_before_t(chain):
    s = skeleton(chain, CHILD, T2024, {fid(chain, "iAnswer")})
    assert "public function helper(string $a): void" in s and "int $b" not in s
    assert "public function helper(string $a, int $b): void" in skeleton(chain, CHILD, T2024 + 1, set())


def test_self_contained_needs_the_pre_t_parameter_names_of_a_called_method(chain):
    gone = {fid(chain, "iAnswer")}
    changed, same = answer_record("$this->helper('x', 2);"), answer_record("$this->stable('x');")
    assert member_use(chain, changed, gone)[0][2] == {"helper"}
    assert not self_contained(chain, changed, gone)
    assert self_contained(chain, same, gone)


def test_file_before_t_ends_at_the_oldest_commit_authored_at_or_after_t_in_history_order(rebased):
    assert [at for _, at, _ in rebased.times(fx.REB)["versions"]] == [T2022, T2024, T2020]
    assert rebased.version_index(fx.REB, T2023) == 2  # A, not C: C (authored 2022) was committed after B (2024)
    before = rebased.text_before(fx.REB, T2023)
    assert "function a(" in before and "function b(" not in before and "function c(" not in before
    assert rebased.version_index(fx.REB, T2024 + 1) == 0 and rebased.version_index(fx.REB, T2020) is None


def test_unchanged_holds_only_when_the_head_code_is_in_the_file_before_t(chain):
    edited, kept, annotated = (fid(chain, n) for n in ("editedStep", "untouchedStep", "stepLater"))
    assert not chain.unchanged(edited, T2024) and chain.unchanged(edited, T2024 + 1)
    assert chain.unchanged(kept, T2024) and chain.unchanged(kept, T2021)
    assert chain.unchanged(annotated, T2021)  # only its annotation line is newer


def test_php_member_refs_include_static_properties():
    assert "counter" in member_refs("php", "public function f(): void\n{\n    self::$counter++;\n}")


def test_java_member_refs_ignore_string_literals_and_a_brace_inside_an_annotation():
    refs = member_refs("java", '@Then("the total is {int} words")\npublic void total(int n) {\n    check(n, "hidden");\n}')
    assert refs == {"check"}


def test_csharp_member_refs_include_a_bare_capitalised_property():
    assert "Patients" in member_refs("csharp", "public void SaveAll()\n{\n    Patients.ForEach(p => p.Save());\n}")


def test_php_param_names_are_the_dollar_names_in_order():
    sig = "public function f(string $a, array $opts = ['x' => 1, 'y' => 2], ?int ...$rest): void\n    {"
    assert param_names("php", sig) == ("a", "opts", "rest")


def test_php_param_names_ignore_commas_inside_an_array_default():
    assert param_names("php", "public function f(array $a = [1, 2], $b): void {") == ("a", "b")


def test_java_param_names_ignore_commas_inside_generics():
    assert param_names("java", "public void f(Map<String, List<Integer>> m, int x) {") == ("m", "x")


def test_csharp_param_names_ignore_default_values():
    assert param_names("csharp", "public void F(string a, int b = 3, bool c = false)") == ("a", "b", "c")


def test_csharp_param_names_ignore_a_comma_inside_a_default_string():
    assert param_names("csharp", 'public void F(string s = "a,b", int n = 1)') == ("s", "n")


def test_show_raises_on_a_path_absent_at_the_commit(chain):
    with pytest.raises(RuntimeError):
        gitio.show(chain.git_dir, chain.commit, "src/Ctx/Nope.php")


def test_cat_batch_raises_when_one_object_is_missing(chain):
    with pytest.raises(RuntimeError):
        gitio.cat_batch(chain.git_dir, [f"{chain.commit}:{CHILD}", f"{chain.commit}:src/Ctx/Nope.php"])


def test_text_is_none_only_for_a_path_outside_the_head_tree(chain_git, monkeypatch):
    ctx = RepoContext(fx.REPO, "php", *chain_git, {})
    assert ctx.text("src/Ctx/Nope.php") is None and "class ChildContext" in ctx.text(CHILD)

    def broken(*_):
        raise RuntimeError("missing blob")
    monkeypatch.setattr(gitio, "show", broken)
    with pytest.raises(RuntimeError):
        ctx.text(BASE)


def test_php_resolver_without_composer_json_raises():
    with pytest.raises(ValueError):
        Resolver("php", [CHILD], None)


def test_java_constructors_keep_their_own_parameters(java):
    o = java.outline(fx.JAVA)
    assert [[n for _, _, n in o.params[m.line]] for m in o.members if m.kind == "constructor"] == [["other"], ["dep", "size"]]
    lines = skeleton(java, fx.JAVA, T2022, set(), frozenset({(fx.JAVA, "size")})).splitlines()
    assert "    public Steps(Other other) { /* … */ }" in lines and "    public Steps(Dep dep) { /* … */ }" in lines
    assert not any("size" in l for l in lines)


def test_class_chain_deeper_than_max_depth_raises(deep):
    assert MAX_DEPTH == 5 and len(deep.ancestors(fx.DEEP[1])) == 5
    with pytest.raises(RuntimeError, match="deeper than 5"):
        deep.ancestors(fx.DEEP[0])


def test_csharp_skeleton_has_a_file_scoped_namespace_and_balanced_braces(cs):
    s = skeleton(cs, fx.CS, T2024 + 1, set())
    assert s.startswith("namespace A.B;\n\npublic class Steps\n")
    assert s.count("{") == s.count("}")
    assert "        public void Go(string a)\n        { /* … */ }" in s


def test_versions_stop_at_a_weak_rename_and_follow_an_exact_one(tmp_path):
    weak = fx.rename_repo(tmp_path / "weak", keep=11)
    log = gitio.git(weak[0], "log", "--follow", "--format=", "--name-status", weak[1], "--", "B.php")
    assert 50 <= int(log.split()[0][1:]) < gitio.MIN_RENAME_SCORE  # git reports the weak rename (R054)
    assert [(at, p) for _, at, p in gitio.versions(*weak, "B.php")] == [(T2022, "B.php")]
    exact = fx.rename_repo(tmp_path / "exact", keep=20)
    assert [(at, p) for _, at, p in gitio.versions(*exact, "B.php")] == [(T2022, "B.php"), (T2020, "A.php")]


def test_parent_added_at_a_later_commit_is_not_searched_before_it(chain):
    assert chain.chain_before(LATE, T2021) == [LATE] and chain.chain_before(LATE, T2022 + 1) == [LATE, LATE_PARENT]
    before = skeleton(chain, LATE, T2021, set())
    header = next(l for l in before.splitlines() if " class " in f" {l}")  # header and namespace come from before T
    assert "extends" not in header and "extends LateParent" in skeleton(chain, LATE, T2022 + 1, set())
    assert "public function own(): void" in before and "function old(" not in before
    assert "public function old(): void" in group(skeleton(chain, LATE, T2022 + 1, set()), "LateParent")


def test_versions_stop_at_a_copy_whose_source_lives_on(tmp_path):
    bare, head = fx.copy_repo(tmp_path)
    assert gitio.git(bare, "log", "--follow", "--format=", "--name-status", head, "--", "B.php").split()[0].startswith("C")
    assert [(at, p) for _, at, p in gitio.versions(bare, head, "B.php")] == [(T2022, "B.php")]
    assert [(at, p) for _, at, p in gitio.versions(bare, head, "A.php")] == [(T2020, "A.php")]


def test_an_old_version_with_two_classes_is_read_for_the_head_class(tmp_path):
    ctx = RepoContext(fx.REPO, "php", *fx.two_class_repo(tmp_path), {})
    assert ctx.outline_before(fx.TWO, T2021).name == "TwoContext"
    assert "helperOnly" not in ctx.times(fx.TWO)["vnames"][-1] and "stepOne" in ctx.times(fx.TWO)["vnames"][-1]


def test_self_contained_counts_a_member_that_the_answer_itself_defines(shop):
    """computeTotal is first declared at T (with the answer): used and defined by the answer, it does not break the flag;
    control: the same call without the definition does."""
    r = record(shop.fns)
    call = "#[Then('I should see the total :total')]\npublic function iShouldSeeTotal(string $total): void\n{\n" \
        "    $this->cart->add($this->computeTotal($total));\n}"
    define = "\n\nprivate function computeTotal(string $total): string\n{\n    return $total;\n}"
    gone = {next(f for f, row in shop.fns.items() if "iShouldSeeTotal" in row["code"])}
    assert self_contained(shop, dict(r, create=[dict(r["create"][0], code=call + define)]), gone)
    assert not self_contained(shop, dict(r, create=[dict(r["create"][0], code=call)]), gone)


def test_the_pre_t_constructor_keeps_a_parameter_whose_member_was_removed_before_head(tmp_path):
    """The constructor is shown as it was before T, with every parameter it had then, even one whose property no
    longer exists at HEAD."""
    step = "    #[Given('x')]\n    public function x(): void\n    {\n    }\n"
    two = "    public function __construct(\n        private A $a,\n        private B $b,\n    ) {\n    }\n\n"
    one = "    public function __construct(\n        private A $a,\n    ) {\n    }\n\n"
    bare, head = fx.build(tmp_path, [(T2020, T2020, {"composer.json": fx.COMPOSER, fx.TWO: fx._php("final class TwoContext", two + step)}),
                                      (T2023, T2023, {fx.TWO: fx._php("final class TwoContext", one + step)})])
    ctx = RepoContext(fx.REPO, "php", bare, head, {})
    before = skeleton(ctx, fx.TWO, T2021, set())
    assert "private A $a," in before and "private B $b," in before
    assert "private B $b" not in skeleton(ctx, fx.TWO, T2024, set())


def test_a_one_line_constructor_keeps_its_promoted_members_usable(tmp_path):
    """Promoted parameters share the line of a one-line constructor; the pre-T declarations used by t_view are told
    apart by kind and names, not by line, so they stay in the outline and count as usable."""
    step = "    #[Given('x')]\n    public function x(): void\n    {\n        $this->a->go();\n    }\n"
    ctor = "    public function __construct(private A $a, private B $b) {}\n\n"
    bare, head = fx.build(tmp_path, [(T2020, T2020, {"composer.json": fx.COMPOSER, fx.TWO: fx._php("final class TwoContext", ctor + step)})])
    ctx = RepoContext(fx.REPO, "php", bare, head, {})
    assert "public function __construct(private A $a, private B $b) { /* … */ }" in skeleton(ctx, fx.TWO, T2021, set())
    assert {n for s in t_view(ctx, fx.TWO, T2021, set()) for n in s.member.names} >= {"a", "b", "__construct", "x"}


def test_a_file_created_at_or_after_t_is_new_even_if_a_rebased_commit_with_an_older_date_follows(tmp_path):
    """is_new_file reads the version choice of rule 5, not the oldest author time: B creates the file at T2024, then C
    (authored T2022, committed after B) edits it; at T2023 the file did not exist."""
    one = fx._php("final class TwoContext", "    public function a(): void\n    {\n    }\n")
    two = fx._php("final class TwoContext", "    public function a(): void\n    {\n    }\n\n    public function b(): void\n    {\n    }\n")
    bare, head = fx.build(tmp_path, [(T2020, T2020, {"composer.json": fx.COMPOSER}), (T2024, T2024, {fx.TWO: one}),
                                      (T2022, T2024 + 3600, {fx.TWO: two})])
    ctx = RepoContext(fx.REPO, "php", bare, head, {})
    assert ctx.is_new_file(fx.TWO, T2023) and not ctx.is_new_file(fx.TWO, T2024 + 1)
