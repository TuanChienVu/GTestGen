from v2tools.calls import CUT_DEPTH, agreement, calls, expand, jvm_calls


def test_java_calls_fields_self_and_this():
    code = ("public void a(String p) {\n    var x = loanApi.create(p);\n    this.repo.save(x);\n    helper(x);\n"
            "    this.go();\n    local.skip();\n    Utils.make();\n}")
    got = jvm_calls("java", code.replace("local.skip();", "Object local = null; local.skip();"), {"loanApi", "repo", "helper", "go"})
    assert got == {("loanApi", "create"), ("repo", "save"), ("self", "helper"), ("self", "go")}


def test_java_lambda_calls():
    got = jvm_calls("java", 'When("^x$", () -> {\n    counter.incrementAndGet();\n});', {"counter"})
    assert got == {("counter", "incrementAndGet")}


def test_csharp_calls():
    got = calls("csharp", "public void A()\n{\n    _http.Send(1);\n    this.Check();\n    Items.ForEach(i => i.Go());\n}", members={"_http", "Check", "Items"})
    assert got == {("_http", "Send"), ("self", "Check"), ("Items", "ForEach")}


def test_php_calls_from_the_tree_only_where_code_runs():
    code = ('public function a(): void\n{\n    $this->repo->find(1); // $this->c1->x();\n    /* $this->c2->x(); */\n'
            '    $s = "{$this->c3->x()}";\n    $this->go();\n    self::st();\n    $this->logger->info("x");\n'
            '    if ($s) { return; }\n    $this->m?->n();\n    throw new \\Exception();\n    $this->dead->x();\n}')
    assert calls("php", code) == {("repo", "find"), ("self", "go"), ("self", "st"), ("m", "n")}
    assert ("audit", "log") not in calls("php", code.replace("$this->go();", "$this->audit->log(1);"), loggers=frozenset({"audit"}))


def test_java_calls_skip_dead_code_and_loggers():
    code = "public void a() {\n    api.create();\n    log.info(\"x\");\n    audit.warn(\"y\");\n    return;\n    api.dead();\n}"
    assert jvm_calls("java", code, {"api", "log", "audit"}, frozenset({"audit"})) == {("api", "create")}


def test_calls_after_a_process_exit_do_not_count():
    """The exit call itself is an ordinary call; what follows it in the block cannot run."""
    php = "public function a(): void\n{{\n    $this->repo->find(1);\n    {}\n    $this->repo->dead();\n}}"
    for stmt in ("exit;", "exit(1);", "die('x');"):
        assert calls("php", php.format(stmt)) == {("repo", "find")}, stmt
    for expr in ("$x = $a ?: exit(1);", "foo() or die();"):
        assert calls("php", php.format(expr)) == {("repo", "find"), ("repo", "dead")}, expr
    java = "public void a() {{\n    api.create();\n    {}\n    api.dead();\n}}"
    for stmt in ("System.exit(0);", "Runtime.getRuntime().halt(1);"):
        assert jvm_calls("java", java.format(stmt), {"api"}) == {("api", "create")}, stmt
    cs = "public void A()\n{\n    _api.Create();\n    Environment.Exit(0);\n    _api.Dead();\n}"
    assert calls("csharp", cs, members={"_api", "Environment"}) == {("_api", "Create"), ("Environment", "Exit")}


def test_root_object_members_are_never_calls():
    """A bare getClass()/hashCode()/equals()/toString() of java.lang.Object resolves for condition 6 (test_symbols)
    but never enters a CA call set."""
    from dataclasses import replace
    from types import SimpleNamespace
    from v2tools.symbols import scope
    resolver = SimpleNamespace(by_suffix={}, resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n))
    ctx = SimpleNamespace(language="java", text={}.get, parents=lambda p: [], ancestors=lambda p: [], resolver=resolver)
    s = scope(ctx, "src/x/Steps.java", "public class Steps {\n    private Api api;\n}\n",
              "public class Steps {\n    private Api api;\n}\n", {})
    body = "public void a() {\n    api.go(getClass(), hashCode(), equals(api), toString());\n}"
    assert jvm_calls("java", body, s.available) == {("api", "go")}


def test_ruby_calls_only_world_methods():
    assert calls("ruby", "Given(/^x$/) do\n  create_patient(1)\n  visit root_path\nend", world={"create_patient"}) == {("self", "create_patient")}


def test_ruby_calls_after_return_do_not_count():
    code = "Given(/^x$/) do\n  create_patient(1)\n  return\n  create_letter(2)\nend"
    assert calls("ruby", code, world={"create_patient", "create_letter"}) == {("self", "create_patient")}


def test_agreement_by_step():
    gb, gc = {"s1": "G1", "s2": "G2", "s3": "G3"}, {"G1": {("a", "f")}, "G2": {("b", "g")}, "G3": set()}
    ob = {"s1": "O1", "s2": "O1", "s3": "O2"}
    assert agreement(gb, gc, ob, {"O1": {("a", "f"), ("b", "g")}}) == (True, True)
    assert agreement(gb, gc, ob, {"O1": {("a", "f")}}) == (True, False)
    assert agreement({"s": "G"}, {"G": set()}, {"s": "O"}, {"O": {("a", "f")}}) == (False, True)


def test_agreement_folds_php_method_names_only():
    """PHP call matching ignores case (§2.2b of the design specification): with `fold` (evaluate.ca passes str.casefold
    for PHP), the METHOD names of gold and output calls compare case-insensitively; the receiver, a PHP property name,
    stays case-sensitive."""
    gb, gc, ob = {"s": "G"}, {"G": {("client", "addRequestData")}}, {"s": "O"}
    assert agreement(gb, gc, ob, {"O": {("client", "AddRequestData")}}) == (True, False)
    assert agreement(gb, gc, ob, {"O": {("client", "AddRequestData")}}, str.casefold) == (True, True)
    assert agreement(gb, gc, ob, {"O": {("Client", "addRequestData")}}, str.casefold) == (True, False)


def expansion(bodies: dict, own: set, root=None):
    """expand() over helper bodies given directly as call sets: `bodies` = helper name -> its project calls."""
    return expand(own, lambda f: [f] if f in bodies else None, lambda text: set(bodies[text]), root)


def test_expand_replaces_a_helper_call_by_the_calls_of_its_body():
    """Helper expansion (§2.2b of the design specification): an expandable ("self", f) becomes the project calls of f's
    body, recursively; a call that resolve() does not expand (a visible method) is kept; a helper contributes exactly
    its own calls."""
    bodies = {"h1": {("api", "a"), ("self", "h2")}, "h2": {("repo", "b"), ("self", "visible")}, "empty": set()}
    e = expansion(bodies, {("self", "h1"), ("self", "empty"), ("cart", "add")})
    assert e.calls == {("api", "a"), ("repo", "b"), ("self", "visible"), ("cart", "add")}
    assert [h for h, _ in e.helpers] == ["empty", "h1", "h2"] and (e.depth_cuts, e.cycle_cuts) == (0, 0)
    assert expansion(bodies, {("self", "h2")}).calls == expansion({}, {("repo", "b"), ("self", "visible")}).calls


def test_expand_keeps_cycles_and_calls_deeper_than_3():
    """A call already on the expansion path, or reached at depth > 3, stays ("self", f), counted."""
    assert CUT_DEPTH == 3
    cyc = expansion({"h1": {("self", "h2"), ("api", "a")}, "h2": {("self", "h1")}}, {("self", "h1")})
    assert cyc.calls == {("api", "a"), ("self", "h1")} and (cyc.depth_cuts, cyc.cycle_cuts) == (0, 1)
    assert expansion({"step": {("api", "a")}}, {("self", "step")}, root="step").calls == {("self", "step")}
    chain = {f"h{i}": {("self", f"h{i + 1}"), (f"m{i}", "f")} for i in range(1, 6)}
    deep = expansion(chain, {("self", "h1")})
    assert deep.calls == {("m1", "f"), ("m2", "f"), ("m3", "f"), ("self", "h4")}
    assert (deep.depth_cuts, deep.cycle_cuts) == (1, 0)


def test_a_call_that_is_both_a_cycle_and_too_deep_counts_as_a_cycle():
    """A gap found by mutation testing: for each body, a body already on the expansion path is a cycle before it
    is a depth cut: h3 -> h1, reached at depth 3, counts once, as a cycle."""
    e = expansion({"h1": {("self", "h2"), ("api", "a")}, "h2": {("self", "h3")}, "h3": {("self", "h1")}},
                  {("self", "h1")})
    assert e.calls == {("api", "a"), ("self", "h1")} and (e.depth_cuts, e.cycle_cuts) == (0, 1)


def test_the_expansion_path_holds_bodies_not_names():
    """A call resolves to every body of its name (the union of the overloads; fixed 25/09/2026); only a
    body already on the expansion path, the function's own body `root` included, is cut as a cycle, and the call then
    stays ("self", f) besides the other overloads, which expand (fineract WorkingCapital_COB|154: the step
    createWorkingCapitalLoanProduct() calls its future overload createWorkingCapitalLoanProduct(req))."""
    bodies = {"f/0": {("self", "f"), ("api", "a")}, "f/1": {("api", "b")}}
    e = expand(bodies["f/0"], lambda f: ["f/0", "f/1"] if f == "f" else None, lambda b: set(bodies[b]), root="f/0")
    assert e.calls == {("api", "a"), ("self", "f"), ("api", "b")} and (e.depth_cuts, e.cycle_cuts) == (0, 1)
    assert [b for _, b in e.helpers] == ["f/1"]


def test_a_call_with_several_bodies_on_the_path_counts_one_cycle():
    """A counter counts a cut CALL once, however many of its bodies
    are cut: f/0 -> g -> f/1 -> f, where both overloads of f are then on the path, adds one cycle, not two."""
    bodies = {"f/0": {("self", "g")}, "g": {("self", "f"), ("api", "g")}, "f/1": {("self", "f"), ("api", "b")}}
    resolve = lambda f: {"f": ["f/0", "f/1"], "g": ["g"]}.get(f)
    e = expand(bodies["f/0"], resolve, lambda b: set(bodies[b]), root="f/0")
    assert e.calls == {("self", "f"), ("api", "g"), ("api", "b")} and (e.depth_cuts, e.cycle_cuts) == (0, 2)


def test_agreement_takes_the_union_over_every_step_a_gold_function_binds():
    """Matching is step by step (§2.2b of the design specification): a gold function binding several steps agrees when
    the union of the output functions bound to THOSE steps meets its calls, not only the first step's."""
    gb, gc = {"s1": "G", "s2": "G"}, {"G": {("api", "go")}}
    assert agreement(gb, gc, {"s1": "O1", "s2": "O2"}, {"O1": {("api", "other")}, "O2": {("api", "go")}}) == (True, True)
    assert agreement(gb, gc, {"s1": "O1", "s2": "O2"}, {"O1": {("api", "other")}, "O2": set()}) == (True, False)


def test_php_class_calls_follow_global_classes_and_drop_the_leading_backslash():
    """`X::f(` is a call when X is in the repo or a PHP global class (a PHP file of that name outside vendor/,
    PrestaShop's legacy classes), with its leading `\\` dropped; a vendor class is no call."""
    from types import SimpleNamespace
    ctx = SimpleNamespace(language="php", tree=["classes/Hook.php", "vendor/acme/Db.php"],
                          resolver=SimpleNamespace(resolve=lambda o, k, n: SimpleNamespace(status="external", fqn=n.lstrip("\\"))))
    code = "public function a(): void\n{\n    Hook::exec('x');\n    \\Hook::exec('y');\n    Db::query();\n}"
    assert calls("php", code, ctx=ctx) == {("Hook", "exec")}


def test_locals_hide_fields_only_where_declared_in_live_code():
    """A local, a catch variable or a lambda parameter with a field's name hides the field (no field call); one
    declared in dead code does not (one of these cases alone changes 3 ph-ee test records)."""
    members = {"api", "list"}
    assert jvm_calls("java", "public void a() {\n    Api api = new Api();\n    api.go();\n}", members) == set()
    assert jvm_calls("java", "public void a() {\n    try { x(); } catch (Exception api) { api.go(); }\n}", members) == set()
    assert jvm_calls("java", "public void a() {\n    list.forEach(api -> api.go());\n}", members) == {("list", "forEach")}
    assert jvm_calls("java", "public void a() {\n    api.go();\n    return;\n    Api api = null;\n}", members) == {("api", "go")}


def test_csharp_calls_need_a_member_except_through_this():
    """A C# bare call or a `field.` call counts only when the
    name is a member; `this.field.F()` counts as that field's call."""
    assert calls("csharp", "public void A()\n{\n    UrlEncode(1);\n    _x.Go();\n}", members=set()) == set()
    assert calls("csharp", "public void A()\n{\n    this._x.Go();\n}", members=set()) == {("_x", "Go")}
    assert calls("csharp", "public void A()\n{\n    UrlEncode(1);\n    _x.Go();\n}", members={"UrlEncode", "_x"}) == \
        {("self", "UrlEncode"), ("_x", "Go")}


def test_pattern_variables_are_locals_and_an_anonymous_class_calls_its_own_methods():
    """The call set reads a function's locals as symbols.refs does (fixed 04/10/2026): a call on a pattern variable
    (`instanceof`, a switch type or record pattern; C# `is T x`, a tuple, a LINQ range variable) named like a field is a
    call on the local, not on the field; and, as symbols.refs does for the member references, a bare or `this.` call
    inside a Java anonymous class body is that class's own (the field of the step class used there is still the step
    class's). Before: (cart, length) and (self, put) were counted."""
    members = {"cart", "put", "store"}
    for stmt in ("if (total instanceof String cart) { cart.length(); }",
                 "switch (total) { case String cart -> cart.length(); default -> {} }",
                 "switch (total) { case Entry(var cart, var v) -> cart.length(); default -> {} }"):
        code = f"public void a(Object total) {{\n    {stmt}\n    store.total();\n}}"
        assert jvm_calls("java", code, members) == {("store", "total")}, stmt
    anon = ("public void a() {\n    Map m = new HashMap<>() {{ put(\"a\", \"b\"); this.put(\"c\", \"d\"); this.store.total();"
            " cart.add(); }};\n}")
    assert jvm_calls("java", anon, members) == {("cart", "add")}  # this.store there is the anonymous class's own
    assert jvm_calls("java", "public void a() {\n    put(\"a\", \"b\");\n    this.put(\"c\", \"d\");\n}", members) == \
        {("self", "put")}  # outside an anonymous class: the step class's own method
    for stmt in ("if (o is string cart) { cart.Trim(); }", "var (cart, n) = (o, 1);\n    cart.Trim();",
                 "var q = from cart in items select cart.Trim();",
                 "var q = from x in items group x by x.K into cart select cart.Trim();"):
        code = f"public void A(object o)\n{{\n    {stmt}\n    store.Total();\n}}"
        assert calls("csharp", code, members={"cart", "store", "items"}) == {("store", "Total")}, stmt
