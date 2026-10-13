import math

import pytest

from v2tools.argcheck import all_groups, arity_ok, hardcoded, params_of, placeholders, step_args, top_groups
from v2tools.patterns import compile_pattern

TICKS = "`" * 3


def test_top_groups_counts_outermost_capturing_groups_only():
    assert top_groups(r'^(the customer) belongs to (group "([^"]+)")$') == 2
    assert top_groups(r"^a (?:b|c) (?P<x>\d+) \((y)\) [(]z[)]$") == 2
    assert top_groups(r"^(?<=a)(?<name>b)$") == 1


def test_placeholders_by_pattern_kind():
    assert placeholders("turnip", "I add :product to :cart", "php") == 2
    assert placeholders("cucumber", "I pay {int} EUR on {string}", "java") == 2
    assert placeholders("regex_full", '^I add "([^"]*)" (\\d+) times$', "php") == 2


def test_hardcoded_values_only_when_written_literally():
    assert not hardcoded("regex_full", r'^the "[^"]+" channel keeps (\d+) days$', 'the "Mobile" channel keeps 30 days')
    assert hardcoded("regex_full", "^it is called 2 times$", "it is called 2 times")
    assert hardcoded("regex_full", r'^I pick "Red\.Blue"$', 'I pick "Red.Blue"')
    assert not hardcoded("turnip", "I have :n items in :cart", 'I have 3 items in "main"')
    assert not hardcoded("cucumber", "I pay {int} EUR", "I pay 20 EUR")


def test_step_args_marks_tables_and_doc_strings():
    r = {"background": "Background:\n  Given a", "scenario": 'Scenario: s\n  When b:\n    | x |\n  Then c\n    """\n    t\n    """',
         "steps": ["a", "b:", "c"]}
    assert step_args(r) == [False, True, True]


def test_params_of_counts_required_total_and_variadic():
    assert params_of("php", "method", "public function a(string $a, int $b = 1): void {}") == (1, 2)
    assert params_of("php", "method", "public function a(string ...$xs): void {}") == (0, math.inf)
    assert params_of("java", "method", "public void a(String a, String... rest) {}") == (1, 2)
    assert params_of("ruby", "block", "Given(/^x$/) do |a, b|\nend") == (2, 2)
    assert params_of("java", "lambda", "", 3) == (3, 3)


def test_ruby_params_are_read_on_the_first_line_only():
    """The `|…|` of an inner block (`items.each do |i|`) is not a step parameter."""
    assert params_of("ruby", "block", "Given(/^x$/) do\n  items.each do |i|\n    go(i)\n  end\nend") == (0, 0)
    assert params_of("ruby", "block", "Given(/^x$/) { |a, *rest|\n  [1].map { |i| i }\n}") == (1, math.inf)


def test_csharp_params_array_is_one_parameter():
    """C# params (fixed 25/09/2026): tree-sitter-c-sharp splits `params string[] rest` into a `params` token, an
    array_type and an identifier; SpecFlow passes one argument per group, so the array is ONE parameter."""
    assert params_of("csharp", "method", "public void A(string a, params string[] rest) { }") == (1, 2)
    assert params_of("csharp", "method", "public void A(params object[] xs) { }") == (0, 1)
    assert params_of("csharp", "method", "public void A(string a, Table t, int n = 1) { }") == (2, 3)


def test_ruby_params_come_from_the_step_block_in_the_tree():
    """The block parameters of the do_block / block attached to the step call (Given, When, Then, And, But, Step),
    read from the tree: never a `do|` inside the step's regex, and on the next line too; an inner block's |i| never
    counts. Each parameter as before: `=` optional, `*` variadic."""
    assert params_of("ruby", "block", "Given(/^I (do|don't) have (\\d+)$/) do |a, n|\n  go(a, n)\nend") == (2, 2)
    assert params_of("ruby", "block", "When(/^x (\\d+)$/) do\n  |n|\n  go(n)\nend") == (1, 1)
    assert params_of("ruby", "block", "Then(/^y$/) do\n  [1].each do |i|\n    go(i)\n  end\nend") == (0, 0)
    assert params_of("ruby", "block", "Step \"z\" do |q, r = 1, *rest|\nend") == (1, math.inf)


def test_a_value_a_placeholder_captures_is_not_hardcoded():
    """A quoted value or number the pattern captures is never hard-coded, even when the same word is
    written elsewhere in the pattern (turnip :type, Cucumber {string}, a regex group)."""
    assert not hardcoded("turnip", "the main image should be of type :type", 'the main image should be of type "main"')
    assert not hardcoded("cucumber", "I activate the budget account with command {string}",
                         'I activate the budget account with command "activate"')
    assert not hardcoded("regex_full", '^"(Partner offers)" is "(required|not required)"$',
                         '"Partner offers" is "required"')


def test_argcheck_pins():
    """Test gaps (no rule change): an escaped parenthesis; Cucumber optional text, alternation and
    an anonymous {}; single quotes never count; no match or a broken pattern is hard-coded; step_args skips comments,
    reads a backtick doc string and `*` steps, and falls back when the counts differ; a Ruby default is optional; the
    first method is read, not a nested one."""
    assert top_groups(r"^a \(b\) (c)$") == 1 and top_groups(r"^\\(x)$") == 1
    assert placeholders("cucumber", "I have {int} cucumber(s) in my belly/stomach {}", "java") == 2
    assert not hardcoded("regex_full", "^I pick 'Red'$", "I pick 'Red'")
    assert hardcoded("regex_full", "^I pick red$", "I pick blue")
    assert hardcoded("regex_full", "^I pick (red$", "I pick red")
    r = {"background": "", "steps": ["a", "b", "c"],
         "scenario": f"Scenario: s\n  Given a\n  # note\n  | x |\n  * b\n    {TICKS}\n    t\n    {TICKS}\n  Then c"}
    assert step_args(r) == [True, True, False]
    assert step_args({**r, "steps": ["a", "b"]}) == [False, False]
    assert params_of("ruby", "block", "Given(/^(\\d+)$/) do |n = 1|\nend") == (0, 1)
    nested = "public void a(int x) {\n    new Runnable() { public void run(int y, int z) {} };\n}"
    assert params_of("java", "method", nested) == (1, 1)


def test_top_groups_measure_depth_through_capture_groups_only():
    """The outermost capture groups: a capture group counts unless another CAPTURE group
    holds it; a non-capturing group, a branch, a repeat, an atomic group or a lookaround does not hide it."""
    session = r'^a session for (?:customer named "(.*)"|the employee) is created (\d+) hours? ago$'
    assert top_groups(session) == 2  # PrestaShop SecurityFeatureContext::createSession(string $name, string $hours)
    assert top_groups(r"^(a (b) c)$") == 1 and top_groups(r"^(?:a|b)$") == 0 and top_groups(r"^(?:(?:(a)))+$") == 1
    assert top_groups(r"^(?P<x>a) (?<y>b)$") == 2 and top_groups(r"^(?=(\d+))\d+ items$") == 1
    assert top_groups(r"^(?>(a))b|(c)$") == 2
    assert top_groups(r"^a\K (b) (?:(c))$") == 0 == all_groups(r"^a\K (b) (?:(c))$")  # re rejects PCRE \K: no count,
    assert compile_pattern("regex_full", r"^a\K (b) (?:(c))$") is None  # and such a pattern never binds a step



# BV condition 4 per framework (§2.2 of the design specification, fixed 25/09/2026): the patterns and signatures
# below ran for real in regex_args/ (Behat 3.26.0/3.33.0/4.0.0-alpha1, cucumber-jvm 7.34.3, cucumber-ruby 11.1.1,
# SpecFlow 3.0.213).
ROWS = {  # row: (pattern, N for PHP and C# = every capture group, N for Java and Ruby = the outermost ones); PHP's N
    # is read per step (test_behat_hands_the_groups_up_to_the_last_one_the_step_matched): these are its
    # counts without a step. "named" ran for real under Behat only: cucumber-jvm, SpecFlow and cucumber-ruby refuse a
    # named group at load (a documented limit)
    "A": (r'^a session for (?:customer named "(.*)"|the employee) is created (\d+) hours? ago$', 2, 2),
    "B": (r"^(a|b) and ((c)|d)$", 3, 2),
    "D": (r'^I add a shop group "(.+)" with name "(.+?)"(?: and color "(.+)")?$', 3, 3),
    "E": (r"^x(?: (\d+))? y (\d+)$", 2, 2),
    "F": (r"^(?=(\d+))\d+ items$", 1, 1),
    "G": (r"^((\d+)-(\d+)) range$", 3, 1),
    "named": (r"^the price is (?P<amount>\d+) (?<currency>[A-Z]+)$", 2, 2),
}


def test_regex_arguments_per_framework():
    """N = the arguments a framework hands over: Behat (PHP) and SpecFlow (C#) every capture group, a named one once;
    cucumber-jvm (Java) and cucumber-ruby the outermost ones. Turnip :tokens and Cucumber {…} count 1 each."""
    for row, (pattern, every, outer) in ROWS.items():
        assert [placeholders("regex_full", pattern, lang) for lang in ("php", "csharp", "java", "ruby")] == \
            [every, every, outer, outer], row
    assert all_groups(ROWS["B"][0]) == 3 and top_groups(ROWS["B"][0]) == 2
    assert placeholders("turnip", "I have :count green apple(s) in basket/bag :name", "php") == 2


def test_parameters_compare_with_n_per_framework():
    """PHP (Behat drops the arguments a method does not take): required <= N, no upper bound (the table case:
    test_behat_takes_fewer_parameters_but_not_an_unused_table). C# (SpecFlow): total, optional ones included, == N; a
    `params` array is one parameter. Java (cucumber-jvm): total == N; varargs is one parameter. Ruby (cucumber-ruby,
    Proc#arity): required == N, or N >= required with a splat; a default, &blk or **kw is not required, a required
    keyword set counts one."""
    php = lambda sig, n: arity_ok("php", n, *params_of("php", "method", f"public function {sig} {{}}"))
    assert php("p1($name, $hours)", 2) and not php("p1($name, $hours)", 1)
    assert php("p4($g, $n, $c = null)", 2) and php("p4($g, $n, $c = null)", 4) and not php("p4($g, $n, $c = null)", 1)
    assert php("v($a, ...$rest)", 5) and not php("v($a, ...$rest)", 0)
    cs = lambda sig, n: arity_ok("csharp", n, *params_of("csharp", "method", f"public void {sig} {{ }}"))
    assert cs("P4(string g, string n, string c = null)", 3) and not cs("P4(string g, string n, string c = null)", 2)
    assert cs("A(string a, params string[] rest)", 2) and not cs("A(string a, params string[] rest)", 3)
    java = lambda sig, n: arity_ok("java", n, *params_of("java", "method", f"public void {sig} {{}}"))
    assert java("p1(String name, String hours)", 2) and not java("p1(String name, String hours)", 1)
    assert java("a(String a, String... rest)", 2) and not java("a(String a, String... rest)", 3)
    assert not java("a(String a, String... rest)", 1)  # equality, not required <= N <= total
    rb = lambda block, n: arity_ok("ruby", n, *params_of("ruby", "block", f"Given(/^x$/) do {block}\nend"))
    assert rb("|a, b = nil|", 1) and not rb("|a, b = nil|", 2)
    assert rb("|a, *r|", 1) and rb("|a, *r|", 3) and not rb("|a, *r|", 0)
    assert rb("|a, &blk|", 1) and rb("|a, **kw|", 1) and rb("|a, k:|", 2) and rb("|(a, b)|", 1)



def test_behat_takes_fewer_parameters_but_not_an_unused_table():
    """PHP, BV condition 4 (§2.2 of the design specification, fixed 25/09/2026), from regex_args (Behat 3.26.0): Behat
    drops the arguments a method does not take ("fewer": `p1($name)` on the 2-group session pattern passes), but stops
    when a table or doc string is not taken ("fewer_table": `p7()` on a step with a table). A table is taken by a
    parameter type-hinted TableNode or PyStringNode, or when the method has at least N parameters (DefaultsContext
    `p7($t = 'DEFAULT')`)."""
    session = ROWS["A"][0]
    fewer = params_of("php", "method", "public function p1($name) { }")
    assert placeholders("regex_full", session, "php") == 2 and arity_ok("php", 2, *fewer)
    table = lambda sig, n: arity_ok("php", n, *(q := params_of("php", "method", f"public function {sig} {{ }}")),
                                    table=True, takes_table=q.takes_table)
    assert not table("p7()", 1)  # fewer_table: 0 groups + the table
    assert table("p7(TableNode $t)", 1) and table("p7($t = 'DEFAULT')", 1)  # exact, defaults
    assert not table("f($a)", 2) and table("f(TableNode $t)", 2)
    assert table("f($a, ?\\Behat\\Gherkin\\Node\\PyStringNode $d)", 2)
    assert arity_ok("php", 2, *params_of("php", "method", "public function f($a) { }"))  # no table: extra dropped



def test_behat_hands_the_groups_up_to_the_last_one_the_step_matched():
    """BV condition 4 (§2.2 of the design specification at 25ba873, fixed 28/09/2026): PHP's preg_match drops
    the TRAILING groups that take no part in the match, so Behat hands a method the groups up to the highest-numbered
    one the STEP matched; N is read on the step (placeholders(…, step)). The real Behat runs (ExactContext,
    Behat 3.26.0, 3.33.0 and 4.0.0-alpha1): p2($x, $y, $z) on "a and d" and p4($g, $n, $c) without
    the colour FAIL ("Can not find a matching value for an argument"), while "b and c", p4 with the colour and p5
    (a MIDDLE group missing: ["", "7"]) pass. A table after such a step joins after N: an untyped f($g, $rows) takes
    it when the trailing group is absent (N = 1 + 1). Without a step, or for the other frameworks, N is as before."""
    method = lambda sig: params_of("php", "method", f"public function {sig} {{ }}")
    n = lambda row, step: placeholders("regex_full", ROWS[row][0], "php", step)
    assert (n("B", "b and c"), n("B", "a and d")) == (3, 2)
    assert (n("D", 'I add a shop group "G2" with name "N2" and color "red"'), n("D", 'I add a shop group "G1" with name "N1"')) \
        == (3, 2)
    assert n("E", "x y 7") == 2 and n("named", "the price is 42 EUR") == 2 and n("F", "12 items") == 1
    assert placeholders("regex_full", ROWS["B"][0], "php") == 3  # no step: every group, as before
    assert [placeholders("regex_full", ROWS["B"][0], lang, "a and d") for lang in ("csharp", "java", "ruby")] == [3, 2, 2]
    assert arity_ok("php", n("B", "b and c"), *method("p2($x, $y, $z)"))
    assert not arity_ok("php", n("B", "a and d"), *method("p2($x, $y, $z)"))  # Behat: no value for $z
    assert arity_ok("php", n("B", "a and d"), *method("p2($x, $y, $z = 'DEFAULT')"))  # DefaultsContext
    assert not arity_ok("php", n("D", 'I add a shop group "G1" with name "N1"'), *method("p4($g, $n, $c)"))  # no $c
    assert arity_ok("php", n("E", "x y 7"), *method("p5($a, $b)"))
    rows = r'^I add a shop group "(.+)"(?: named "(.+)")? with rows:$'
    n_rows = placeholders("regex_full", rows, "php", 'I add a shop group "G1" with rows:')
    table = lambda sig: arity_ok("php", n_rows + 1, *(q := method(sig)), table=True, takes_table=q.takes_table)
    assert n_rows == 1 and table("f($g, $rows)") and table("f($g, TableNode $t)") and not table("f($g, $n, TableNode $t)")
    assert not arity_ok("php", all_groups(rows) + 1, *method("f($g, $rows)"), table=True)  # every group: a false fail


def test_mutant_pins_for_hard_coded_values_and_tables():
    """Pins for four mutants that survived the earlier tests: an empty quoted value is no hard-coded
    value; a value is hard-coded only as a whole word of the pattern, not inside a longer number; a table after an
    `And` or a `But` step is that step's; a Turnip step matches in any case, so its values are captured all the same."""
    assert not hardcoded("regex_full", r'^I type "([^"]*)" in field$', 'I type "" in field')  # empty value
    assert not hardcoded("regex_full", r"^I see (\d+) of \d+ in lot12$", "I see 5 of 1 in lot12")  # whole word only
    record = {"background": "", "scenario": "Scenario: s\n  Given a\n  And b\n    | k |\n  But c\n    | v |",
              "steps": ["a", "b", "c"]}
    assert step_args(record) == [False, True, True]  # And/But table
    assert not hardcoded("turnip", "I pick :color apple", "i pick red apple")  # Turnip ignores case
    # a mutant here flipped one gold answer (fineract WorkingCapitalLoanOrigination.feature:320)
    assert not hardcoded("cucumber", "Working Capital loan breach schedule has {int} period(s)",
                         "Working Capital loan breach schedule has 1 period")
    assert not hardcoded("regex_full", '^I change order "(.*)" note to "(.*)"$', 'I change order "bo_order1" note to ""')
    assert not hardcoded("turnip", "The default category of product :productName has no group reduction",
                         'the default category of product "Mug" has no group reduction')
    assert top_groups(r"^(?i:(a)) (?:x (\d+))?? (b)$") == 3


def test_mutant_pins_for_conditional_groups_and_ruby_block_locals():
    """Pins for mutants that survived: a conditional group `(?(1)…|…)` is looked through, so a capture in it counts
    for every-group and outermost counting; Ruby block-local variables after `;` are not parameters, as Proc#arity
    ignores them (`|n; tmp|` has arity 1; the Phase 2 catalog has none)."""
    assert all_groups(r"^(a)?(?(1)(b)|(c))$") == 3 == top_groups(r"^(a)?(?(1)(b)|(c))$")
    assert all_groups(r"^((a)|b)(?(2)(c)|d)$") == 3 and top_groups(r"^((a)|b)(?(2)(c)|d)$") == 2
    locals_ = params_of("ruby", "block", "Given(/^(\\d+)$/) do |n; tmp, other|\nend")
    assert locals_ == (1, 1) and arity_ok("ruby", 1, *locals_)


# Regex forms Python's `re` cannot read, or reads its own way (checked since 26/09/2026). `re` parses a pattern for both
# bind.Library (patterns.compile_pattern) and argcheck, after patterns._ruby_to_python. The Phase 2 catalog, gold and
# control outputs hold none of them (0 of 52,512 patterns on 26/09/2026).
REJECTED = {  # .NET/Ruby/PCRE named group; Unicode properties (Java, .NET, Ruby, PCRE): re raises re.error
    "quote_named": r"^I have (?'n'\d+) apples$",
    "unicode_prop": r"^I have (\p{N}+) apples$",
    "unicode_prop_negated": r"^I have (\P{L}+) apples$",
    "unicode_short": r"^I have (\pN+) apples$",
}
ACCEPTED = {  # re reads these as PCRE, .NET and Onigmo do (java.util.regex rejects `(?#`: a documented limit)
    "comment": (r"^I have (?#count)(\d+) apples$", "I have 3 apples", ("3",)),
    "bracket_first": (r"^I have ([]a]+) apples$", "I have ]a] apples", ("]a]",)),
    "bracket_negated_first": (r"^I have ([^]a]+) apples$", "I have 3 apples", ("3",)),
}


@pytest.mark.parametrize("form", sorted(REJECTED))
def test_a_pattern_python_cannot_read_never_binds(form):
    """The step library cannot compile it (bind.Library skips an uncompilable pattern), so the step does not bind:
    condition 2 fails and condition 4 is not reached; N is 0 in every framework's count and the value check does not
    crash (it reports hard-coded)."""
    pat = REJECTED[form]
    assert compile_pattern("regex_full", pat) is None and compile_pattern("regex", pat) is None
    assert all_groups(pat) == top_groups(pat) == 0
    assert [placeholders("regex_full", pat, lang) for lang in ("php", "csharp", "java", "ruby")] == [0, 0, 0, 0]
    assert hardcoded("regex_full", pat, "I have 3 apples")


@pytest.mark.parametrize("form", sorted(ACCEPTED))
def test_a_pattern_python_reads_like_the_framework(form):
    """`(?#…)` is a comment and a `]` right after `[` or `[^` is a member of the class, in Python as in PCRE, .NET and
    Onigmo: the pattern compiles, binds its step and hands over its one group."""
    pat, step, groups = ACCEPTED[form]
    rx, mode = compile_pattern("regex_full", pat)
    assert mode == "fullmatch" and rx.fullmatch(step).groups() == groups
    assert all_groups(pat) == top_groups(pat) == 1 and not hardcoded("regex_full", pat, step)



def test_condition_4_pins_for_table_types_and_java_lambdas():
    """An intersection or DNF type takes a table. `tablenode` in any case takes it
    (PHP class names are case-insensitive; the last `\\` segment of each alternative); a java8 lambda counts its
    parameters from the tree, so `(java.util.Map<String, String> m) ->` has one; a `=` inside a parameter's attribute
    does not make it optional (a PHP default is the parameter's `default_value`). Kept as limits: TableNode on a step
    with no table, PyStringNode on a table step, any class named …\\TableNode."""
    takes = lambda sig: params_of("php", "method", f"public function f({sig}) {{ }}").takes_table
    assert takes("TableNode&\\Countable $t") and takes("(TableNode&\\Countable)|null $t")
    assert takes("tablenode $t") and takes("\\behat\\gherkin\\node\\PYSTRINGNODE $d")
    assert takes("\\PrestaShop\\TableNode $t") and not takes("\\Behat\\TableNode\\Other $t")
    assert params_of("php", "method", "public function f(string $field, #[Foo(a: 'x=y')] string $other) { }") == (2, 2)
    assert params_of("php", "method", "public function f($a, $b = 'x=y') { }") == (1, 2)
    lam = lambda params: params_of("java", "lambda", f'When("x", {params} -> {{\n}});', 9)
    assert lam("(java.util.Map<String, String> m)") == (1, 1)
    assert lam("(String a, Map<String, List<Integer>> b)") == (2, 2)
    assert lam("(a, b)") == (2, 2) and lam("x") == (1, 1) and lam("()") == (0, 0)



def test_mutant_pins_for_doc_strings_and_nested_groups():
    """Pins for two mutants that survived: a doc-string step whose method takes only a PyStringNode (1 placeholder
    + the doc string: N = 2 > total 1, so it passes only because the parameter takes it); capture groups nested three
    deep all count for Behat and SpecFlow, while the outermost count stays 1."""
    par = params_of("php", "method", "public function a(PyStringNode $d): void {}")
    assert par == (1, 1) and par.takes_table and arity_ok("php", 2, *par, table=True, takes_table=par.takes_table)
    assert all_groups(r"^(((a)))$") == 3 and top_groups(r"^(((a)))$") == 1
    got = [placeholders("regex_full", r"^(((a)))$", lang) for lang in ("php", "csharp", "java", "ruby")]
    assert got == [3, 3, 1, 1]


def test_a_trailing_group_that_matches_the_empty_string_still_counts():
    """PHP's preg_match drops only the trailing groups that take NO part in the match; one that matches the empty
    string is handed over (a mutant that also drops it flips 5 labelled PrestaShop pairs)."""
    assert placeholders("regex_full", '^I change order "(.*)" note to "(.*)"$', "php", 'I change order "bo_order1" note to ""') == 2
    assert placeholders("regex_full", r"^x(?: (\d+))?$", "php", "x") == 0
