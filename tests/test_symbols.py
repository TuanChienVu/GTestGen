from dataclasses import replace
from types import SimpleNamespace

from tests.phase3_fixture import SHOP, catalog, make_repo
from v2tools.outline import parse
from v2tools.state import RepoContext
from v2tools.symbols import (Refs, RubyWorld, Scope, class_status, grounded, initializer_refs, logger_fields,
                             open_paths, refs, ruby_grounded, ruby_unresolved, ruby_world, scope, snake, unresolved)


def test_php_refs_members_calls_and_classes():
    r = refs("php", "public function a(): void\n{\n    $x = new Foo();\n    $this->cart->add(Bar::X);\n    self::helper();\n    return static::class;\n}")
    assert r.members == {"cart", "helper"} and r.calls == {"helper"} and r.classes == {"Foo", "Bar"}
    assert refs("php", "function a() { $this->go(); }").calls == {"go"}


def test_refs_skip_comments_strings_and_dead_code():
    php = ('public function a(): void\n{\n    // $this->c1->x();\n    /* $this->c2->x(); */\n    $s = "{$this->c3->x()}";\n'
           '    $h = <<<EOT\n    $this->c4\n    EOT;\n    $this->live->x();\n    return;\n    $this->dead->x();\n}')
    assert refs("php", php).members == {"live"}
    java = 'public void a() {\n    // helperA();\n    String s = "helperB()";\n    helperC();\n    throw new IllegalStateException();\n}'
    assert refs("java", java).calls == {"helperC"}
    java2 = "public void a() {\n    if (x) { return; }\n    after();\n    return;\n    dead();\n}"
    assert refs("java", java2).calls == {"after"}
    java3 = "public void a() {\n    switch (k) {\n        case 1:\n            return;\n        default:\n            live();\n    }\n    lbl: return;\n    dead();\n}"
    assert refs("java", java3).calls == {"live"}


def test_process_exits_end_the_flow():
    """A statement that ends the process cuts the rest of its block like `return`; an exit inside an expression
    (`?: exit(1)`, `or die()`) does not."""
    php = "public function a(): void\n{{\n    $this->before->x();\n    {}\n    $this->after->x();\n}}"
    for stmt in ("exit;", "exit(1);", "die('x');", "die;", "EXIT;", "Die(2);"):
        assert refs("php", php.format(stmt)).members == {"before"}, stmt
    for expr in ("$x = $a ?: exit(1);", "foo() or die();"):
        assert refs("php", php.format(expr)).members == {"before", "after"}, expr
    assert refs("php", php.format("switch ($k) { case 1: exit; $this->c1->x(); default: $this->c2->x(); }")).members \
        == {"before", "c2", "after"}
    java = "public void a() {{\n    before();\n    {}\n    after();\n}}"
    for stmt in ("System.exit(0);", "Runtime.getRuntime().halt(1);", "Runtime.getRuntime().exit(2);",
                 "java.lang.System.exit(3);"):
        assert refs("java", java.format(stmt)).calls == {"before"}, stmt
    assert refs("java", java.format("int v = f(System.exit(0));")).calls == {"before", "f", "after"}
    cs = "public void A()\n{{\n    Before();\n    {}\n    After();\n}}"
    for stmt in ("Environment.Exit(0);", "System.Environment.Exit(1);", 'Environment.FailFast("x");'):
        assert refs("csharp", cs.format(stmt)).calls == {"Before"}, stmt



def test_a_php_exit_under_the_error_suppression_operator_ends_the_flow():
    """`@exit`/`@die` as the whole statement end the process like `exit`/`die`; `@` on any other call, or an `@exit`
    inside an expression, does not."""
    php = "public function a(): void\n{{\n    $this->before->x();\n    {}\n    $this->after->x();\n}}"
    for stmt in ("@exit;", "@exit(1);", "@die('x');", "@DIE;", "@ exit ;", "@@die();"):
        assert refs("php", php.format(stmt)).members == {"before"}, stmt
    for stmt in ("@$this->probe->x();", "@f();", "$v = @exit(1);", "@f() or die();"):
        assert {"before", "after"} <= refs("php", php.format(stmt)).members, stmt


def test_a_fully_qualified_java_runtime_exit_ends_the_flow():
    """`java.lang.Runtime.getRuntime().exit(…)` and `.halt(…)` end the process like the unqualified forms."""
    java = "public void a() {{\n    before();\n    {}\n    after();\n}}"
    for stmt in ("java.lang.Runtime.getRuntime().exit(0);", "java.lang.Runtime.getRuntime().halt(1);",
                 "java.lang.Runtime . getRuntime ( ) . halt ( 2 ) ;"):
        assert refs("java", java.format(stmt)).calls == {"before"}, stmt
    assert refs("java", java.format("other.lang.Runtime.getRuntime().exit(0);")).calls == {"before", "after"}


def test_a_csharp_global_environment_exit_ends_the_flow():
    """`global::System.Environment.Exit(…)` and `.FailFast(…)` end the process like the `System.`-qualified and
    unqualified forms."""
    cs = "public void A()\n{{\n    Before();\n    {}\n    After();\n}}"
    for stmt in ("global::System.Environment.Exit(0);", 'global::System.Environment.FailFast("x");',
                 "global :: System . Environment . Exit ( 1 ) ;"):
        assert refs("csharp", cs.format(stmt)).calls == {"Before"}, stmt
    assert refs("csharp", cs.format("global::Other.Environment.Exit(0);")).calls == {"Before", "After"}

def test_java_refs_skip_locals_params_and_package_roots():
    code = ('public void a(String p) {\n    var b = foo.bar(p, SECONDS);\n    this.q = baz(b);\n    list.forEach(x -> x.go());\n'
            '    java.util.List.of(1);\n    Utils.call(CONST);\n    UUID.randomUUID();\n}')
    r = refs("java", code)
    assert r.members == {"foo", "SECONDS", "q", "baz", "list", "CONST"}
    assert r.calls == {"baz"} and r.classes == {"Utils", "UUID"}


def test_java_lambda_step_uses_its_parameters():
    r = refs("java", 'Given("^x (\\\\d+)$", (Integer n) -> {\n    total = n + helper();\n});')
    assert r.members == {"total", "helper"} and r.calls == {"helper"}


def test_csharp_lambda_parameter_is_local():
    r = refs("csharp", "public void A()\n{\n    Items.ForEach(item => { item.Go(); _log.Write(1); });\n}")
    assert "item" not in r.members and "_log" in r.members


def test_ruby_refs_calls_and_constants():
    r = refs("ruby", "Given(/^x (\\d+)$/) do |n|\n  y = n\n  create_thing(Renalware::Patients::Patient.first, y)\nend")
    assert r.calls == {"create_thing"} and "Renalware::Patients::Patient" in r.classes and "n" not in r.calls


def test_ruby_lines_after_return_raise_or_fail_cannot_run():
    """A line whose first token is return, raise or fail (no trailing modifier) cuts every following line up to the
    end or } that closes its innermost block, or up to an else/elsif/when/in/rescue/ensure line at the same depth;
    nested blocks in the cut are dropped whole."""
    step = ("Given(/^x$/) do\n  before_call\n  {}\n  after_call(Renalware::Ghost.first)\n  items.each do |i|\n"
            "    dead_block(i)\n  end\nend")
    for stmt in ("return", "return nil", "raise 'boom if bad'", "fail ArgumentError", "raise # comment if x"):
        r = refs("ruby", step.format(stmt))
        assert r.calls - {"raise", "fail"} == {"before_call"} and "Renalware::Ghost" not in r.classes, stmt  # line runs
    for stmt in ("return if done?", "raise Oops unless ok?", "fail while x", "returned_value", "raise_error"):
        assert {"before_call", "after_call", "dead_block"} <= refs("ruby", step.format(stmt)).calls, stmt
    branches = ("When(/^y$/) do\n  if a?\n    raise Oops\n    dead_a(1)\n  elsif b?\n    live_b(1)\n  else\n"
                "    return\n    dead_c(1)\n  end\n  case k\n  when 1\n    fail\n    dead_d(1)\n  when 2\n"
                "    live_e(1)\n  end\n  begin\n    raise Oops\n    dead_f(1)\n  rescue\n    live_g(1)\n  ensure\n"
                "    live_h(1)\n  end\n  [1].each { |v|\n    return v\n    dead_i(1)\n  }\n  live_j\nend")
    got = refs("ruby", branches).calls
    assert {"live_b", "live_e", "live_g", "live_h", "live_j"} <= got
    assert not {"dead_a", "dead_c", "dead_d", "dead_f", "dead_i"} & got


def test_scope_resolves_own_inherited_and_output_members(tmp_path):
    bare, head = make_repo(tmp_path)
    ctx = RepoContext("Acme/Shop", "php", bare, head, catalog())
    text = ctx.text(SHOP)
    after = text.replace("    private function computeTotal", "    public function mine(): void\n    {\n    }\n\n    private function computeTotal")
    s = scope(ctx, SHOP, text, after, {})
    assert {"indexPage", "cart", "checkout", "sharedStorage", "helperOld", "mine"} <= s.available
    assert "secret" not in s.available and "mine" not in s.project and not s.open_all
    assert unresolved(Refs(frozenset({"cart", "ghost"}), frozenset(), frozenset()), s) == {"ghost"}
    o = parse("php", after)
    assert grounded(ctx, o, Refs(frozenset({"cart"}), frozenset(), frozenset()), s)
    assert not grounded(ctx, o, Refs(frozenset({"mine"}), frozenset({"mine"}), frozenset()), s)
    assert class_status(ctx, o, "BaseContext") == "resolved"
    root = frozenset({"getClass", "toString"})  # PHP has no root object
    assert unresolved(Refs(root, frozenset(), frozenset()), s) == root


def test_php_class_constant_in_any_case_is_no_member():
    """PHP's `::class` is case-insensitive (25/09/2026), so `SELF::CLASS` names no member and `Foo::Class` is no class
    reference, as `static::class` and `Foo::class`."""
    r = refs("php", "public function a(): void\n{\n    $x = SELF::CLASS;\n    $y = Foo::Class;\n"
                    "    $z = static::class;\n}")
    assert r.members == set() and r.classes == set()
    assert refs("php", "public function a(): void\n{\n    $y = Foo::X;\n}").classes == {"Foo"}


def test_php_constructor_and_method_names_in_any_case(tmp_path):
    """PHP names are case-insensitive: a constructor written `__CONSTRUCT` promotes its parameters to properties, as
    `__construct` does (3A's outline only knows the lower-case name, so symbols.scope adds them); a METHOD call resolves
    (condition 6) and grounds on a project method (condition 5) whatever its case; a property name stays
    case-sensitive."""
    bare, head = make_repo(tmp_path)
    ctx = RepoContext("Acme/Shop", "php", bare, head, catalog())
    text = ctx.text(SHOP)
    upper = text.replace("public function __construct(", "public function __CONSTRUCT(")
    s = scope(ctx, SHOP, upper, upper, {})
    assert {"indexPage", "cart", "checkout"} <= s.available and {"indexPage", "cart", "checkout"} <= s.project
    calls = Refs(frozenset({"IADDTOCART", "HelperOld", "CART"}), frozenset({"IADDTOCART", "HelperOld"}), frozenset())
    assert unresolved(calls, s) == {"CART"}
    o = parse("php", upper)
    assert grounded(ctx, o, Refs(frozenset({"IADDTOCART"}), frozenset({"IADDTOCART"}), frozenset()), s)
    assert not grounded(ctx, o, Refs(frozenset({"CART"}), frozenset(), frozenset()), s)
    mine = upper.replace("    private function computeTotal", "    public function mine(): void\n    {\n    }\n\n"
                         "    private function computeTotal")
    s2 = scope(ctx, SHOP, upper, mine, {})
    assert unresolved(Refs(frozenset({"MINE"}), frozenset({"MINE"}), frozenset()), s2) == set()
    assert not grounded(ctx, o, Refs(frozenset({"MINE"}), frozenset({"MINE"}), frozenset()), s2)
    base = "src/Behat/Context/BaseContext.php"  # an ancestor's `__CONSTRUCT`: a private promoted property stays its own
    ctor = "    public function __CONSTRUCT(private Foo $hidden, protected Bar $shared)\n    {\n    }\n\n"
    s3 = scope(ctx, SHOP, text, text, {base: ctx.text(base).replace("    protected function helperOld", ctor +
                                                                    "    protected function helperOld")})
    assert "shared" in s3.available and "hidden" not in s3.available


def test_root_object_members_resolve_but_are_not_project_members():
    """What every class inherits from java.lang.Object / System.Object resolves for condition 6, in a body or a field
    initializer, and is never an open path; it never grounds (condition 5) and never enters a call set (CA)."""
    resolver = SimpleNamespace(by_suffix={}, resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n))
    ctx = {lang: SimpleNamespace(language=lang, text={}.get, parents=lambda p: [], ancestors=lambda p: [],
                                 resolver=resolver) for lang in ("java", "csharp")}
    s = scope(ctx["java"], "src/x/Steps.java", "public class Steps {\n    private Api api;\n}\n",
              "public class Steps {\n    private Api api;\n}\n", {})
    body = "public void a() {\n    api.go(getClass(), hashCode(), equals(api), toString());\n}"
    log = "private final org.slf4j.Logger probeLog = org.slf4j.LoggerFactory.getLogger(getClass());"
    assert unresolved(refs("java", body), s) == set() and unresolved(initializer_refs("java", log), s) == set()
    for opened in (replace(s, open_all=True), replace(s, open_calls=True)):
        assert open_paths(ctx["java"], None, refs("java", body), opened) == set()
    assert not {"getClass", "hashCode", "equals", "toString"} & (s.available | s.project)
    assert not grounded(ctx["java"], None, refs("java", "public void a() {\n    getClass();\n}"), s)
    c = scope(ctx["csharp"], "Steps/Steps.cs", "public class Steps\n{\n    private Api _api;\n}\n",
              "public class Steps\n{\n    private Api _api;\n}\n", {})
    cs = "public void A()\n{\n    _api.Go(GetType(), GetHashCode(), ReferenceEquals(_api, null), MemberwiseClone());\n}"
    assert unresolved(refs("csharp", cs), c) == set()
    assert unresolved(initializer_refs("csharp", "private readonly string _n = GetType().Name;"), c) == set()
    assert unresolved(refs("java", body.replace("getClass", "ghostCall")), s) == {"ghostCall"}


def test_open_scopes_tolerate_unknown_members():
    s = Scope(frozenset({"a"}), frozenset({"a"}), frozenset(), True, False)
    assert unresolved(Refs(frozenset({"zzz"}), frozenset(), frozenset()), s) == set()
    s2 = Scope(frozenset({"a"}), frozenset({"a"}), frozenset(), False, True)
    assert unresolved(Refs(frozenset({"call", "field"}), frozenset({"call"}), frozenset()), s2) == {"field"}


def test_ruby_world_resolution():
    w = RubyWorld(frozenset({"create_thing"}), frozenset({"renalware", "renalware/patients", "renalware/patients/patient"}),
                  frozenset({"renalware"}))
    assert snake("PrimaryCarePhysician") == "primary_care_physician" and snake("HTTPClient") == "http_client"
    assert ruby_unresolved(Refs(frozenset(), frozenset(), frozenset({"Renalware::Patients::Ghost", "FactoryBot"})), w) == {"Renalware::Patients::Ghost"}
    assert ruby_grounded(Refs(frozenset({"create_thing"}), frozenset({"create_thing"}), frozenset()), w)
    assert ruby_grounded(Refs(frozenset(), frozenset(), frozenset({"Renalware::Patients::Patient"})), w)
    assert not ruby_grounded(Refs(frozenset({"visit"}), frozenset({"visit"}), frozenset({"FactoryBot"})), w)


def test_open_paths_name_what_passes_only_through_a_heuristic():
    java = SimpleNamespace(language="java")
    r = Refs(frozenset({"a", "call"}), frozenset({"call"}), frozenset())
    assert open_paths(java, None, r, Scope(frozenset(), frozenset(), frozenset(), True, False)) == {"a", "call"}
    assert open_paths(java, None, r, Scope(frozenset({"a", "call"}), frozenset(), frozenset(), True, False)) == set()
    assert open_paths(java, None, r, Scope(frozenset({"a"}), frozenset(), frozenset(), False, True)) == {"call"}
    assert open_paths(java, None, r, Scope(frozenset({"a", "call"}), frozenset(), frozenset(), False, True)) == set()
    lombok = Scope(frozenset({"log"}), frozenset(), frozenset(), False, False, True)
    assert open_paths(java, None, Refs(frozenset({"log"}), frozenset(), frozenset()), lombok) == {"log"}
    ext = Refs(frozenset({"assertThat"}), frozenset({"assertThat"}), frozenset())
    s = Scope(frozenset(), frozenset(), frozenset(), False, False, open_statics=frozenset({"assertThat"}))
    assert open_paths(java, None, ext, s) == {"assertThat"} and unresolved(ext, s) == set()


def test_static_imports_count_only_from_the_er_tree_and_the_repo():
    files = {"src/com/acme/Util.java": "package com.acme;\npublic class Util {\n    public static void helper() {}\n    public static void other() {}\n}\n"}
    ctx = SimpleNamespace(language="java", text=files.get, parents=lambda p: [], ancestors=lambda p: [],
                          resolver=SimpleNamespace(by_suffix={"com/acme/Util": "src/com/acme/Util.java"}))
    er = ("package x;\nimport static com.acme.Util.helper;\nimport static org.assertj.core.api.Assertions.assertThat;\n"
          "public class Steps {\n    private static final org.slf4j.Logger AUDIT = null;\n    private Api api;\n}\n")
    after = er.replace("import static com.acme.Util.helper;", "import static com.acme.Util.helper;\nimport static com.acme.Util.other;")
    s = scope(ctx, "src/x/Steps.java", er, after, {})
    assert s.statics == {"helper"} and s.open_statics == {"assertThat", "other"} and "AUDIT" in s.loggers
    assert grounded(ctx, None, Refs(frozenset({"helper"}), frozenset({"helper"}), frozenset()), s)
    assert not grounded(ctx, None, Refs(frozenset({"assertThat", "other"}), frozenset({"assertThat", "other"}), frozenset()), s)
    assert not grounded(ctx, None, Refs(frozenset({"AUDIT", "log"}), frozenset(), frozenset()), s)


def test_java_upper_case_field_receiver_grounds():
    ctx = SimpleNamespace(language="java", resolver=SimpleNamespace(resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n)))
    r = refs("java", "public void s(String date) {\n    FORMATTER.format(date);\n}")
    assert "FORMATTER" in r.classes and not r.members
    s = Scope(frozenset({"FORMATTER"}), frozenset({"FORMATTER"}), frozenset(), False, False)
    assert grounded(ctx, None, r, s) and not grounded(ctx, None, r, Scope(frozenset(), frozenset(), frozenset(), False, False))


def test_logger_fields_by_declared_type():
    java = "import org.slf4j.Logger;\nclass A {\n    private static final Logger TRACE = LoggerFactory.getLogger(A.class);\n    private Api api;\n}"
    assert logger_fields("java", java) == {"TRACE"} and logger_fields("java", java.replace("import org.slf4j.Logger;", "")) == set()
    assert logger_fields("php", "class A {\n    public function __construct(private readonly LoggerInterface $audit) {}\n}") == {"audit"}


def test_php_scoped_method_calls_resolve_in_any_case(tmp_path):
    """PHP method names are case-insensitive: a SCOPED method call, `self::f()`, `static::f()` or `parent::f()` (scope
    keyword in any case), is a method call like `$this->f()`: it resolves (condition 6) and grounds on a project method
    (condition 5) whatever its case; a scoped property `self::$p` or constant `self::C` stays case-sensitive; a class
    receiver `Foo::f()` is unchanged."""
    r = refs("php", "public function a(): void\n{\n    self::PROBEFILL();\n    STATIC::PrObEfIlL();\n"
                    "    Parent::HELPEROLD();\n    $x = self::$Prop;\n    $y = static::LIMIT;\n    Foo::Bar();\n}")
    assert r.calls == {"PROBEFILL", "PrObEfIlL", "HELPEROLD"} and r.classes == {"Foo"}
    assert r.members == {"PROBEFILL", "PrObEfIlL", "HELPEROLD", "Prop", "LIMIT"}
    bare, head = make_repo(tmp_path)
    ctx = RepoContext("Acme/Shop", "php", bare, head, catalog())
    text = ctx.text(SHOP)
    mine = text.replace("    private function computeTotal", "    public function probeFill(): void\n    {\n    }\n\n"
                        "    private function computeTotal")
    s = scope(ctx, SHOP, text, mine, {})
    assert unresolved(r, s) == {"Prop", "LIMIT"}
    o = parse("php", mine)
    assert grounded(ctx, o, refs("php", "public function a(): void\n{\n    parent::HELPEROLD();\n}"), s)
    assert not grounded(ctx, o, refs("php", "public function a(): void\n{\n    self::PROBEFILL();\n}"), s)


def test_a_comment_inside_an_exit_statement_still_ends_the_flow():
    """Comments inside a process-exit statement (in its callee, or after `@`) do not hide it: the callee is compared
    without its comments and without whitespace."""
    php = "public function a(): void\n{{\n    $this->before->x();\n    {}\n    $this->after->x();\n}}"
    for stmt in ("@/*c*/exit;", "@ /* c */ die('x');", "@/*a*/@/*b*/exit(1);"):
        assert refs("php", php.format(stmt)).members == {"before"}, stmt
    java = "public void a() {{\n    before();\n    {}\n    after();\n}}"
    for stmt in ("System./*c*/exit(1);", "System // c\n        .exit(1);", "System.exit/*c*/(0);",
                 "java/*x*/.lang.Runtime.getRuntime()./*y*/halt(2);"):
        assert refs("java", java.format(stmt)).calls == {"before"}, stmt
    assert refs("java", java.format("System./*exit*/out.println(1);")).calls == {"before", "after"}
    cs = "public void A()\n{{\n    Before();\n    {}\n    After();\n}}"
    for stmt in ("Environment/*c*/.Exit(1);", 'global::System./*c*/Environment.FailFast("x");'):
        assert refs("csharp", cs.format(stmt)).calls == {"Before"}, stmt


def test_an_any_case_method_call_is_no_open_path():
    """On an open class (an external ancestor or trait), a PHP method called in another case that the class has
    resolves by name, so it is not an open path; a name the class does not have is one."""
    php = SimpleNamespace(language="php")
    s = Scope(frozenset({"baseOld"}), frozenset({"baseOld"}), frozenset(), True, False,
              methods=frozenset({"baseold"}), project_methods=frozenset({"baseold"}))
    assert open_paths(php, None, Refs(frozenset({"BASEOLD"}), frozenset({"BASEOLD"}), frozenset()), s) == set()
    ghost = Refs(frozenset({"probeGhost"}), frozenset({"probeGhost"}), frozenset())
    assert open_paths(php, None, ghost, s) == {"probeGhost"}


# ----- further reference, scope, dead-code and logger rules (28/09/2026 and 04/10/2026) ----------------------------

def jvm_ctx(language: str, files: dict | None = None, parents: dict | None = None, ancestors: dict | None = None):
    """A RepoContext stand-in: file texts, HEAD parents/ancestors per path, every class reference external."""
    files = files or {}
    return SimpleNamespace(language=language, text=files.get, tree=list(files), parents=lambda p: (parents or {}).get(p, []),
                           ancestors=lambda p: (ancestors or {}).get(p, []),
                           resolver=SimpleNamespace(by_suffix={}, resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n)))


def test_this_super_and_base_calls_are_member_references():
    """Java `this.f()`, `super.f()` (and `super.x`) and C# `this.F()`, `base.F()` name a member of the class as a bare
    `f()` does, so condition 5 (§2.2: methods of the class or of a parent) does not depend on the spelling."""
    s = Scope(frozenset({"helper", "Helper", "f"}), frozenset({"helper", "Helper", "f"}), frozenset(), False, False)
    ctx = jvm_ctx("java")
    for body in ("this.helper();", "super.helper();", "helper();"):
        r = refs("java", "public void a() {\n    " + body + "\n}")
        assert r.members == {"helper"} and r.calls == {"helper"} and grounded(ctx, None, r, s), body
    assert refs("java", "public void a() {\n    super.f = 1;\n}").members == {"f"}
    for body in ("this.Helper();", "base.Helper();"):
        r = refs("csharp", "public void A()\n{\n    " + body + "\n}")
        assert r.members == {"Helper"} and r.calls == {"Helper"} and grounded(jvm_ctx("csharp"), None, r, s), body


def test_pattern_and_query_variables_are_locals():
    """A variable a pattern or a query declares is a local, never an unresolved field: Java `instanceof T x`; C#
    `is T x`, `is var x`, `var (a, b)`, `foreach (var (k, v) in …)`, LINQ `from`/`let`/`join`/`into` (otherwise
    condition 6 fails code that compiles)."""
    assert refs("java", "public void a(Object o) {\n    if (o instanceof String sx) {\n        api.go(sx);\n    }\n}").members == {"api"}
    cs = ("public void A(object o)\n{\n    if (o is string sx) { _api.Go(sx); }\n    if (o is var w) { _api.Go(w); }\n"
          "    var (aa, bb) = _api.Pair();\n    foreach (var (k, v) in _api.Pairs) { _api.Go(k, v, aa, bb); }\n"
          "    var q1 = from r in _api.Rows let t = r.V select t;\n"
          "    var q2 = from r in _api.Rows join s in _api.Others on r.K equals s.K select s;\n"
          "    var q3 = from r in _api.Rows group r by r.K into g select g;\n}")
    assert refs("csharp", cs).members == {"_api"}


def test_a_bare_call_in_an_anonymous_class_body_is_no_member_of_the_step_class():
    """28/09/2026: `put(…)` or `this.put(…)` inside `new HashMap<>() {{ … }}` (ph-ee's KongStepDef double-brace
    initialisation, 6/409 catalog functions) is the anonymous class's own call, resolved against its supertype, never a
    member of the step class; a field of the step class used there still counts."""
    java = ("public void a() {\n    Map<String, String> m = new HashMap<>() {{\n        put(\"k\", value);\n"
            "        this.put(\"x\", \"y\");\n    }};\n    List<String> l = new ArrayList<>() {{ add(\"z\"); }};\n    api.go(m, l);\n}")
    r = refs("java", java)
    assert r.members == {"value", "api"} and r.calls == set()
    s = Scope(frozenset({"api", "value"}), frozenset({"api", "value"}), frozenset(), False, False)
    assert unresolved(r, s) == set()
    assert unresolved(r, replace(s, available=frozenset({"api"}))) == {"value"}


def test_scope_rules_that_move_sealed_gold_results():
    """The scope rules that decide gold results (28/09/2026), each through scope(), ruby_world() or class_status():
    Lombok `log` (28 fineract golds), the World under features/support/** (33 renalware golds; also without
    features/step_definitions/), a trait target opens every member (15 Sylius golds), a PHP class without a namespace
    outside vendor/ is global (6 PrestaShop golds), a C# PascalCase non-member is an open path (aux open_resolution)."""
    lombok = "package x;\n@Slf4j\npublic class Steps {\n    private Api api;\n}\n"
    s = scope(jvm_ctx("java"), "src/x/Steps.java", lombok, lombok, {})
    r = refs("java", "public void a() {\n    log.info(\"x\");\n    api.go();\n}")
    assert s.lombok_log and unresolved(r, s) == set() and open_paths(jvm_ctx("java"), None, r, s) == {"log"}
    plain = lombok.replace("@Slf4j\n", "")
    assert unresolved(r, scope(jvm_ctx("java"), "src/x/Steps.java", plain, plain, {})) == {"log"}
    files = {"features/support/world.rb": "module W\n  def seed_patient\n  end\nend\n",
             "features/step_definitions/x_steps.rb": "def record_x\nend\n",
             "app/models/renalware/patient.rb": "module Renalware\n  class Patient\n  end\nend\n"}
    w = ruby_world(SimpleNamespace(tree=list(files), text=files.get))
    assert {"seed_patient", "record_x"} <= w.methods and "renalware/patient" in w.paths
    trait = "<?php\nnamespace App;\ntrait ValidationTrait\n{\n    public function check(): void\n    {\n    }\n}\n"
    t = scope(jvm_ctx("php"), "src/ValidationTrait.php", trait, trait, {})
    assert t.open_all and unresolved(Refs(frozenset({"client"}), frozenset(), frozenset()), t) == set()
    php = jvm_ctx("php", {"classes/Hook.php": "<?php\nclass Hook {}\n", "vendor/acme/Db.php": "<?php\nclass Db {}\n"})
    assert class_status(php, None, "Hook") == "global" and class_status(php, None, "Db") == "external"
    hook = Scope(frozenset(), frozenset(), frozenset(), False, False)
    assert grounded(php, None, refs("php", "public function a(): void\n{\n    Hook::exec('x');\n}"), hook)
    cs = jvm_ctx("csharp")
    c = Scope(frozenset({"_api"}), frozenset({"_api"}), frozenset(), False, False)
    assert open_paths(cs, None, Refs(frozenset({"_api"}), frozenset(), frozenset({"GhostHelper"})), c) == {"GhostHelper"}


def test_dead_code_cuts_follow_each_branch_kind():
    """Code after a return is dead up to the end of ITS branch only: a PHP `:` block and `default:`, a Java case group,
    a C# switch section; a local declared in dead code does not hide a member; PHP backtick strings are no code."""
    php = ("public function a(): void\n{\n    if ($x):\n        return;\n        $this->dead1->x();\n    else:\n"
           "        $this->live1->x();\n    endif;\n    switch ($k) {\n        default:\n            return;\n"
           "            $this->dead2->x();\n    }\n    $this->live2->x();\n    $s = `ls {$this->ghost}`;\n}")
    assert refs("php", php).members == {"live1", "live2"}
    java = ("public void a() {\n    switch (k) {\n        case 1:\n        case 2:\n            return;\n            dead();\n"
            "        case 3:\n            live();\n    }\n    if (x) {\n        return;\n        String api = \"\";\n    }\n    api.go();\n}")
    assert refs("java", java).calls == {"live"} and "api" in refs("java", java).members
    cs = ("public void A()\n{\n    switch (k)\n    {\n        case 1:\n            return;\n            Dead();\n        case 2:\n"
          "            Live();\n            break;\n    }\n}")
    assert refs("csharp", cs).calls == {"Live"}


def test_logger_rules_by_type_name_and_ancestor():
    """Loggers are no project knowledge: Log4j and JUL typed fields are loggers like SLF4J; `_log`/`_logger` names
    never ground; a PSR-3 `\\Psr\\Log\\LoggerInterface` field is one; an in-repo ancestor's logger field is the class's
    logger too."""
    assert logger_fields("java", "import org.apache.logging.log4j.Logger;\nclass A {\n    private static final Logger a4 = null;\n}") == {"a4"}
    assert logger_fields("java", "class A {\n    private java.util.logging.Logger jul = null;\n}") == {"jul"}
    assert logger_fields("php", "class A {\n    private \\Psr\\Log\\LoggerInterface $psr;\n}") == {"psr"}
    s = Scope(frozenset({"_log", "_logger"}), frozenset({"_log", "_logger"}), frozenset(), False, False)
    assert not grounded(jvm_ctx("java"), None, Refs(frozenset({"_log", "_logger"}), frozenset(), frozenset()), s)
    base = "src/x/Base.java"
    files = {base: "package x;\nimport org.slf4j.Logger;\npublic class Base {\n    protected Logger audit;\n}\n"}
    ctx = jvm_ctx("java", files, ancestors={"src/x/Steps.java": [(SimpleNamespace(path=base, relation="extends"), 1)]})
    steps = "package x;\npublic class Steps extends Base {\n    private Api api;\n}\n"
    assert "audit" in scope(ctx, "src/x/Steps.java", steps, steps, {}).loggers


def test_open_paths_of_external_parents_and_in_repo_statics():
    """Open paths come only through something outside the repo: an external interface leaves only bare calls open
    (open_calls, not open_all); an in-repo ancestor's external parent opens every member; a `.*` static import of an
    in-repo class brings its members and an explicit one resolves for condition 6; a trait's private members are the
    using class's own."""
    steps = "package x;\npublic class Steps implements Api {\n    private Api api;\n}\n"
    ext = SimpleNamespace(status="external", relation="implements")
    s = scope(jvm_ctx("java", parents={"src/x/Steps.java": [ext]}), "src/x/Steps.java", steps, steps, {})
    assert s.open_calls and not s.open_all
    assert unresolved(Refs(frozenset({"go", "field"}), frozenset({"go"}), frozenset()), s) == {"field"}
    base = "src/x/Base.java"
    ctx = jvm_ctx("java", {base: "package x;\npublic class Base extends Lib {\n}\n"},
                  parents={base: [SimpleNamespace(status="external", relation="extends")]},
                  ancestors={"src/x/Steps.java": [(SimpleNamespace(path=base, relation="extends"), 1)]})
    plain = steps.replace(" implements Api", " extends Base")
    assert scope(ctx, "src/x/Steps.java", plain, plain, {}).open_all
    util = {"src/com/acme/Util.java": "package com.acme;\npublic class Util {\n    public static void helper() {}\n    public static void other() {}\n}\n"}
    sctx = SimpleNamespace(language="java", text=util.get, parents=lambda p: [], ancestors=lambda p: [],
                           resolver=SimpleNamespace(by_suffix={"com/acme/Util": "src/com/acme/Util.java"}))
    star = "package x;\nimport static com.acme.Util.*;\npublic class Steps {\n}\n"
    assert {"helper", "other"} <= scope(sctx, "src/x/Steps.java", star, star, {}).statics
    one = star.replace("Util.*", "Util.helper")
    assert unresolved(Refs(frozenset({"helper"}), frozenset({"helper"}), frozenset()), scope(sctx, "src/x/Steps.java", one, one, {})) == set()
    trait = "src/T.php"
    pctx = jvm_ctx("php", {trait: "<?php\ntrait T\n{\n    private function tp(): void\n    {\n    }\n}\n"},
                   ancestors={"src/C.php": [(SimpleNamespace(path=trait, relation="trait"), 1)]})
    cls = "<?php\nclass C\n{\n    use T;\n}\n"
    assert "tp" in scope(pctx, "src/C.php", cls, cls, {}).available


def test_locals_of_catch_resources_foreach_out_and_initializers():
    """Java catch and try-with-resources variables, C# `foreach` and `out var` variables are locals; a C# property
    initializer and a PHP class-constant initializer are read like a body."""
    java = ("public void a() {\n    try (Reader rd = open()) {\n        rd.read();\n    } catch (IOException ex) {\n"
            "        ex.printStackTrace();\n    }\n}")
    assert refs("java", java).members == {"open"} and refs("java", java).calls == {"open"}
    cs = "public void A()\n{\n    foreach (var it in _api.Items) { it.Go(); }\n    _api.Try(out var res);\n    res.Go();\n}"
    assert refs("csharp", cs).members == {"_api"}
    assert initializer_refs("csharp", "public string Name { get; } = _ghost.Name;").members == {"_ghost"}
    assert "ghost" in initializer_refs("php", "const C = self::ghost;").members
