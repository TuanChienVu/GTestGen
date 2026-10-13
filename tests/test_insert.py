from collections import Counter

from v2tools.ertree import parser
from v2tools.insert import (STEP_ANNOTATION, class_body, class_methods, indented, insert, node_name, php_open_tag,
                            shell_keys, split_code)
from v2tools.stepdefs import extract as extract_defs
from v2tools.stepdefs import file_imports

PHP = '''<?php

namespace App\\Behat\\Context;

use App\\Page\\IndexPage;

final class ShopContext
{
    public function __construct(private IndexPage $indexPage)
    {
    }

    /**
     * @Given I open the shop
     */
    public function iOpenTheShop(): void
    {
        $this->indexPage->open();
    }

    #[When('I add :product')]
    #[When('I put :product')]
    public function iAdd(string $product): void
    {
    }

    private function helper(): void
    {
    }
}
'''


def ok(language, text):
    return not parser(language).parse(text.encode()).root_node.has_error


def test_split_code_imports_methods_and_members():
    imports, items = split_code("php", "use App\\X;\n\n#[Then('a')]\npublic function a(): void\n{\n}\n\nprivate $p;")
    assert imports == ["use App\\X;"]
    assert [(i.kind, i.name, i.arity) for i in items] == [("method", "a", 0), ("member", None, None)]


def test_add_new_method_before_class_close_and_merge_import():
    out = insert("php", PHP, "use App\\Y;\n#[Then('b')]\npublic function b(): void\n{\n}")
    assert out.added == 1 and not out.replaced and ok("php", out.text)
    assert "use App\\Page\\IndexPage;\nuse App\\Y;\n" in out.text
    assert out.text.rstrip().endswith("public function b(): void\n    {\n    }\n}")


def test_duplicate_import_is_dropped():
    assert insert("php", PHP, "use App\\Page\\IndexPage;\n#[Then('b')]\npublic function b(): void {}").text.count("IndexPage;") == 1


def test_replace_preexisting_shell_case_insensitive_and_keep_old_annotations():
    shells = shell_keys("php", ["#[When('I add :product')]\npublic function iAdd(string $product): void\n{\n}"])
    out = insert("php", PHP, "#[When('I add :product')]\n#[When('I append :product')]\npublic function IADD(string $p): void\n{\n    $x = $p;\n}", shells)
    assert out.replaced == ("IADD",) and out.kept_annotations == 1 and ok("php", out.text)
    body = out.text[out.text.index("#[When('I add :product')]"):]
    assert body.index("I append") < body.index("I put :product") < body.index("function IADD")
    assert out.text.count("function iAdd") + out.text.count("function IADD") == 1


def test_clash_with_non_shell_method_is_a_collision():
    out = insert("php", PHP, "#[Then('h')]\npublic function helper(): void {}")
    assert out.collisions == ("helper",) and out.text == PHP


def test_docblock_annotation_kept_inside_new_docblock():
    shells = shell_keys("php", ["public function iOpenTheShop(): void {}"])
    out = insert("php", PHP, "/**\n * @Given I enter the shop\n */\npublic function iOpenTheShop(): void\n{\n}", shells)
    assert out.kept_annotations == 1 and ok("php", out.text)
    assert " * @Given I enter the shop\n     * @Given I open the shop\n     */" in out.text


def test_java_overload_is_added_not_replaced():
    java = 'package p;\n\npublic class S {\n    @When("a {int}")\n    public void a(int x) {\n    }\n}\n'
    out = insert("java", java, '@When("a {int} {int}")\npublic void a(int x, int y) {\n}', shell_keys("java", ['public void a(int x) {\n}']))
    assert out.added == 1 and not out.replaced and not out.collisions and ok("java", out.text)


def test_java_lambda_goes_into_constructor_holding_lambdas():
    java = 'public class S implements En {\n    private int n;\n\n    public S() {\n        Given("^a$", () -> {\n        });\n    }\n}\n'
    out = insert("java", java, 'Then("^b (\\\\d+)$", (Integer k) -> {\n    n = k;\n});')
    assert out.added == 1 and ok("java", out.text)
    ctor = out.text[out.text.index("public S()"):]
    assert ctor.index('Then("^b') < ctor.index("    }\n}")



def test_a_java8_lambda_is_taken_only_as_a_top_level_statement():
    """A cucumber-java8 step lambda becomes a lambda item only as a top-level statement of the block or of a
    constructor body (where Cucumber-java8 registers it); anywhere else (a `static {}` or instance initializer, a
    method body, a block nested in a constructor, an anonymous class, a comment) it stays in the code around it."""
    lam = 'Given("^a$", () -> {\n    go();\n});'
    kinds = lambda code: [(it.kind, it.text.count('Given("^a$"')) for it in split_code("java", code)[1]]
    assert kinds(lam + "\n\n" + lam.replace("^a$", "^b$")) == [("lambda", 1), ("lambda", 0)]
    ctor = "/** Steps. */\n@Inject\npublic Steps(Api api) throws IOException {\n    this.api = api;\n" + lam + "\n}"
    assert kinds(ctor) == [("lambda", 1), ("method", 0)]
    ctor = 'private int n;\n\n@SuppressWarnings({"a", "b"})\npublic Steps() {\n' + lam + "\n}"
    assert kinds(ctor) == [("lambda", 1), ("member", 0), ("method", 0)]
    for code in ("static {\n" + lam + "\n}", "{\n" + lam + "\n}", "public void holder() {\n" + lam + "\n}",
                 "public Steps() {\n    if (on) {\n" + lam + "\n    }\n}", "public <T> T holder() {\n" + lam + "\n}"):
        assert kinds(code) == [("member" if code.startswith(("static", "{")) else "method", 1)], code
    anon = "private Object probeX = new Object() {\n    public void f() {\n" + lam + "\n    }\n};"
    assert kinds(anon) == [("raw", 1)]
    assert kinds("/**\n" + lam + "\n*/\nprivate int n;") == [("member", 1)]  # a docblock of the member

def test_java_lambda_creates_constructor_when_none():
    out = insert("java", "public class S implements En {\n}\n", 'Given("^a$", () -> {\n});')
    assert "public S() {" in out.text and ok("java", out.text)


def test_ruby_blocks_are_appended_one_item_each():
    imports, items = split_code("ruby", 'Given(/^a$/) do\n  x\nend\n\nWhen(/^b (\\d+)$/) do |n|\n  y(n)\nend')
    assert [i.kind for i in items] == ["block", "block"]
    out = insert("ruby", "Given(/^z$/) do\nend\n", 'Given(/^a$/) do\nend')
    assert out.text == "Given(/^z$/) do\nend\n\nGiven(/^a$/) do\nend\n"


def test_code_that_does_not_parse_is_inserted_verbatim():
    out = insert("php", PHP, "public function broken( {")
    assert out.raw == 1 and "public function broken( {" in out.text and not ok("php", out.text)


def test_multibyte_text_before_the_insertion_point():
    text = PHP.replace("    private function helper(): void", "    // prix en €, café\n    private function helper(): void")
    out = insert("php", text, "#[Then('b')]\npublic function b(): void\n{\n}")
    assert ok("php", out.text) and "// prix en €, café" in out.text and out.text.rstrip().endswith("{\n    }\n}")


def test_same_new_method_twice_in_one_block_collides():
    out = insert("php", PHP, "#[Then('b')]\npublic function b(): void {}\n\n#[Then('c')]\npublic function b(): void {}")
    assert out.added == 1 and out.collisions == ("b",)


def test_a_type_declared_in_a_block_is_raw():
    code = 'public class Wrapper {\n    @Given("^x$")\n    public void a() {\n    }\n}'
    assert [it.kind for it in split_code("java", code)[1]] == ["raw"]
    assert [it.kind for it in split_code("java", "private Api api;\n\npublic void a() {\n}")[1]] == ["member", "method"]


def test_a_struct_or_an_anonymous_class_in_a_field_is_raw():
    """A C# struct or record struct is a type declaration; a Java field whose initializer holds an anonymous class body
    hides methods that can never bind as steps of the class."""
    cs = 'public struct S\n{\n    [Given(@"^x$")]\n    public void A() { }\n}'
    assert [it.kind for it in split_code("csharp", cs)[1]] == ["raw"]
    rec = "public record struct R(int X);\n\nprivate int _f = 1;"
    assert [it.kind for it in split_code("csharp", rec)[1]] == ["raw", "member"]
    anon = 'private Object f = new Object() {\n    @Given("^x$")\n    public void a() {\n    }\n};'
    assert [it.kind for it in split_code("java", anon)[1]] == ["raw"]
    assert [it.kind for it in split_code("java", "private List<String> l = new ArrayList<>(), m = List.of();\n"
                                                 "private Runnable r = () -> { go(); };")[1]] == ["member", "member"]


def test_an_anonymous_class_without_a_method_stays_a_member():
    """Only an anonymous class body that declares a method makes the field raw; the Jackson idiom
    `new TypeReference<…>() {}` or a body with fields only is an ordinary member."""
    for code in ("private TypeReference<List<Foo>> t = new TypeReference<List<Foo>>() {};",
                 "private Object o = new Object() {\n    int x = 1;\n};"):
        assert [it.kind for it in split_code("java", code)[1]] == ["member"], code
    nested = ("private Object o = new Object() {\n    Runnable r = new Runnable() {\n        public void run() {\n"
              "        }\n    };\n};")
    assert [it.kind for it in split_code("java", nested)[1]] == ["raw"]


def test_a_ruby_block_that_never_closes_is_raw():
    assert [it.kind for it in split_code("ruby", "Given(/^x$/) {\n  visit root_path\n")[1]] == ["raw"]


def test_a_method_with_the_key_of_a_future_method_replaces_it():
    """An output method whose key is that of a FUTURE method of the class replaces it, like a preexisting shell; a
    clash with any other method of the class is still a collision."""
    out = insert("php", PHP, "#[Then('h')]\npublic function HELPER(): void\n{\n    $this->x();\n}", future={"helper"})
    assert out.replaced == ("HELPER",) and not out.collisions and ok("php", out.text)
    assert "function helper(" not in out.text and out.text.count("$this->x();") == 1
    assert insert("php", PHP, "public function helper(): void {}", future={"other"}).collisions == ("helper",)
    java = 'public class S {\n    private void help(int a) {\n        old();\n    }\n}\n'
    out = insert("java", java, "private void help(int b) {\n    fresh();\n}", future={("help", 1)})
    assert out.replaced == ("help",) and "old();" not in out.text and "fresh();" in out.text
    assert insert("java", java, "private void help(int b) {\n}", future={("help", 2)}).collisions == ("help",)


def test_class_methods_lists_name_key_and_text():
    """The methods of a file's class, with their insertion key and their text (docblock and annotations included)."""
    got = class_methods("php", PHP)
    assert [(n, k) for n, k, _ in got] == [("__construct", "__construct"), ("iOpenTheShop", "iopentheshop"),
                                           ("iAdd", "iadd"), ("helper", "helper")]
    assert got[1][2].startswith("/**\n * @Given I open the shop\n */\npublic function iOpenTheShop(): void\n{")
    java = 'public class S {\n    @Given("x")\n    public void a(int n) {\n        go(n);\n    }\n}\n'
    assert class_methods("java", java) == [("a", ("a", 1), '@Given("x")\npublic void a(int n) {\n    go(n);\n}')]


def test_an_output_constructor_holds_its_java8_lambdas_when_the_class_has_none():
    """fineract's standard cucumber-java8 form: when the target class declares no constructor, the output's
    constructor is inserted first and receives the block's lambda steps (its own and the top-level ones), whatever its
    annotations or type parameters; no `X() {}` is created, so nothing collides. When the class has a constructor, the
    lambdas go into it and an output constructor with its key is a collision, unless that constructor may be replaced
    (a shell or a FUTURE method): then the output's constructor replaces it first and keeps the lambdas."""
    lam = 'When("^a$", () -> {\n    api.go();\n});'
    body = "\n    int probe = 1;\n" + indented(lam, "    ") + "\n}"
    bare = "public class S implements En {\n    private Api api;\n}\n"
    for head in ("public S() {", "@Deprecated\npublic S() {", '@SuppressWarnings({"a", "b"})\npublic S() {',
                 "public <T extends java.util.Map<String, java.util.List<Integer>>> S() {"):
        out = insert("java", bare, head + body)
        assert not out.collisions and ok("java", out.text) and out.text.count(" S() {") == 1, head
        ctor = out.text[out.text.index(" S() {"):]
        assert ctor.index("int probe = 1;") < ctor.index('When("^a$"') < ctor.index("    }\n}"), head
        assert [k for k, _, _ in out.items] == ["method", "lambda"], head
    out = insert("java", bare, lam + "\n\npublic S() {\n}")  # top-level lambdas and an empty constructor
    assert not out.collisions and ok("java", out.text) and out.text.count(" S() {") == 1
    assert out.text.index(" S() {") < out.text.index('When("^a$"')
    held = "public class S implements En {\n    public S() {\n        old();\n    }\n}\n"
    assert insert("java", held, "public S() {" + body).collisions == ("S",)
    out = insert("java", held, "public S() {" + body, future={("S", 0)})
    assert out.replaced == ("S",) and not out.collisions and ok("java", out.text) and "old();" not in out.text
    assert out.text.index("int probe = 1;") < out.text.index('When("^a$"') < out.text.rindex("    }\n}")


def test_constructor_stays_after_lambdas_when_the_class_has_an_unreplaceable_constructor_of_another_key():
    """_constructors_first moves the output's constructor before the lambdas only when the class declares NO
    constructor, or one with the SAME key it may replace (a shell or a FUTURE method). Here the class already has
    `S(String a)`, of a DIFFERENT key than the output's `S()`, which the output may not replace: the order must stay
    as split_code gave it (lambda, then the output's own new constructor), so _add_lambda still finds only the class's
    OWN constructor `S(String a)` for the block's lambdas, not a brand new `S()` the output happens to add under an
    unrelated key."""
    lam = 'When("^a$", () -> {\n    api.go();\n});'
    body = "\n    int probe = 1;\n" + indented(lam, "    ") + "\n}"
    held = ("public class S implements En {\n    private String s;\n\n    public S(String a) {\n"
            "        this.s = a;\n    }\n}\n")
    out = insert("java", held, "public S() {" + body)
    assert not out.collisions and ok("java", out.text) and out.text.count(" S() {") == 1
    before, after = out.text.split(" S() {", 1)
    assert "this.s = a;" in before and 'When("^a$"' in before and "int probe = 1;" not in before
    assert "int probe = 1;" in after and 'When("^a$"' not in after


def head_annotations(language: str, text: str, name: str) -> list[str]:
    """The attributes/annotations tree-sitter attaches to method `name` of a file (whitespace squeezed)."""
    src = text.encode()
    body = class_body(parser(language).parse(src).root_node)
    m = next(c for c in body.named_children if c.type == "method_declaration" and node_name(src, c) == name)
    out, stack = [], [m]
    while stack:
        n = stack.pop()
        if n.type in ("attribute", "annotation", "marker_annotation"):
            out.append(" ".join(n.text.decode().split()))
        elif n.type not in ("compound_statement", "block", "formal_parameters", "parameter_list"):
            stack.extend(n.children)
    return out


def test_kept_annotations_go_before_the_head_of_the_method():
    """An old step annotation kept by a replacement (the kept-annotation rule) goes on its own line right before the
    method's first token after its attributes/annotations when that token starts a line, else before the method's
    first line (the whole attribute list): never inside a multi-line attribute, never between the signature and the
    body."""
    shells = shell_keys("php", ["#[When('I add :product')]\npublic function iAdd(string $product): void\n{\n}"])
    for new in ("#[When(\n    'I append :product'\n)]\npublic function iAdd(string $p): void\n{\n}",
                "#[When('I append :product')] public function iAdd(string $p): void\n{\n}",
                "#[When(\n    'I append :product'\n)] public function iAdd(string $p): void\n{\n}",
                "/**\n * Adds.\n */\n#[When('I append :product')] #[Other] public function iAdd(string $p): void\n"
                "{\n}"):
        out = insert("php", PHP, new, shells)
        assert out.replaced == ("iAdd",) and ok("php", out.text), new
        assert {"When('I add :product')", "When('I put :product')"} <= set(head_annotations("php", out.text, "iAdd"))
    java = 'package p;\n\npublic class S {\n    @When("a {int}")\n    public void a(int x) {\n    }\n}\n'
    for new in ('@When(\n    "b {int}"\n)\npublic void a(int x) {\n}', '@When("b {int}") public void a(int x) {\n}'):
        out = insert("java", java, new, shell_keys("java", ["public void a(int x) {\n}"]))
        assert out.kept_annotations == 1 and ok("java", out.text), new
        assert '@When("a {int}")' in head_annotations("java", out.text, "a"), new
    cs = "public class S\n{\n    [When(@\"a\")]\n    public void A()\n    {\n    }\n}\n"
    for new in ('[When(\n    @"b")]\npublic void A()\n{\n}', '[When(@"b")] public void A()\n{\n}'):
        out = insert("csharp", cs, new, shell_keys("csharp", ["public void A()\n{\n}"]))
        assert out.kept_annotations == 1 and ok("csharp", out.text), new
        assert 'When(@"a")' in head_annotations("csharp", out.text, "A"), new


def test_the_php_open_tag_of_a_block_is_dropped():
    """Sylius channel_validation|100: a leading `<?php` open tag (any case; on its own line, or before code on its
    line) and a trailing `?>` of a PHP block are dropped before anything else, in split_code, so every PHP insertion
    (and write_overlays) sees the block without them; php_open_tag says whether it happened (a descriptive column)."""
    fn = "#[Then('a')]\npublic function a(): void\n{\n}"
    assert php_open_tag("<?php\n" + fn) == (fn, True)
    assert php_open_tag("<?php use App\\X;\n" + fn) == ("use App\\X;\n" + fn, True)
    assert php_open_tag("  <?PHP  \n\n" + fn + "\n?>\n") == ("\n" + fn, True)
    assert php_open_tag(fn + "\n\n?>") == (fn + "\n", True)
    for code in (fn, "<?phpx\n" + fn, fn + "\n$x = '?>';", "$a = 1; ?>\n<b>html</b>"):
        assert php_open_tag(code) == (code, False), code
    use = "use App\\X;\n\n" + fn
    assert split_code("php", "<?php\n" + use + "\n?>") == split_code("php", use)
    assert insert("php", PHP, "<?php\n\n" + use).text == insert("php", PHP, use).text
    assert split_code("java", "<?php\n" + fn)[1][0].kind == "raw"  # PHP only


def test_the_php_open_tag_strips_blank_lines_before_and_after_it():
    """§1.4: `<?php` is "at the start" of a PHP block after ANY leading whitespace, blank lines included (PHP_OPEN's
    `\\A\\s*`), and `?>` is "at the end" before ANY trailing whitespace, blank lines included (PHP_CLOSE's `\\s*\\Z`);
    the regexes implement both, each direction on its own and together."""
    fn = "#[Then('a')]\npublic function a(): void\n{\n}"
    assert php_open_tag("\n\n<?php\n" + fn) == (fn, True)  # blank lines before <?php
    assert php_open_tag(fn + "\n?>\n\n") == (fn, True)  # blank lines after ?>
    assert php_open_tag("\n\n<?php\n" + fn + "\n?>\n\n") == (fn, True)  # both


def test_braces_in_strings_comments_and_earlier_members_never_hide_a_java8_lambda():
    """A `{` in a string, a character literal or a comment before a top-level lambda is not code (the brace depth stays
    0); a member's `}` ends its header, so a constructor after a method is still recognised as a constructor and its
    lambdas are taken."""
    lam = 'When("^a$", () -> {\n    api.go();\n});'
    kinds = lambda code: [it.kind for it in split_code("java", code)[1]]
    for before in ('private String probeS = "{";', "private char probeC = '{';", "// {", "/* { */",
                   'private String probeT = "\\"{";'):
        assert kinds(before + "\n\n" + lam) == ["lambda"] + kinds(before), before
    ctor = "public void probeHelper() {\n}\n\npublic S() {\n    super();\n" + indented(lam, "    ") + "\n}"
    assert kinds(ctor) == ["lambda", "method", "method"]
    field = 'private int[] probeA = {1, 2};\n\npublic S() {\n' + indented(lam, "    ") + "\n}"
    assert kinds(field) == ["lambda", "member", "method"]


ATTRS = ["#[Given('the store has :x')]", "#[Given('the shop\\'s :x')]", "#[Given('the store also has :x')]"]
DOCS = ["/**", " * @Given the store has :x", " * @Given the shop's :x", " * @Given the store also has :x", " */"]
BODY = ["public function f($x): void", "{", "    $this->a->b($x);", "}"]
OLD3 = {"the store has :x", "the shop's :x", "the store also has :x"}


def forms(text: str) -> dict[str, list[str]]:
    """pattern -> the forms that register it in a PHP file ("attribute", "docblock"), Phase 1's extractor."""
    out: dict[str, list[str]] = {}
    for st in ("php_attribute", "php_docblock"):
        for d in extract_defs("C.php", text, st):
            out.setdefault(d.pattern, []).append(st[4:])
    return out


def test_an_old_pattern_the_output_rewrites_keeps_the_head_annotation():
    """The kept-annotation rule as refined on 26/09/2026 after a Behat 4.0.0-alpha1 run of the ER harness, branch (a): a
    pattern of the replaced method's OLD annotations (never a CREATE step: the ER tree drops those) is compared with the
    output's by the pattern Behat registers, whatever the form (attribute or docblock) and the keyword, the attribute
    literal unescaped. When the output also writes it, the HEAD line is kept as HEAD wrote it and the output's
    annotation for it is dropped (on its own line, or inline before the signature), so no pattern is registered twice on
    the method (two attributes of one pattern stop Behat 4: RedundantStepException); when it does not, the HEAD line is
    kept."""
    cases = ((ATTRS, [DOCS[0], DOCS[1], DOCS[2].replace("@Given", "@Then"), DOCS[4]] + BODY, "attribute", 3),
             (DOCS, [ATTRS[0], ATTRS[1].replace("Given", "When")] + BODY, "docblock", 3),
             (ATTRS, [ATTRS[0].replace("Given", "When"), "#[Given(\"the shop's :x\")]"] + BODY, "attribute", 3),
             (ATTRS, ["#[Then('the store has :x')] #[Given('the store also has :x')] " + BODY[0]] + BODY[1:],
              "attribute", 2))  # the output's own copy of a HEAD line stays where it is
    for old, new, form, kept in cases:
        shell = "<?php\nclass C\n{\n" + "\n".join("    " + l for l in old + BODY) + "\n}\n"
        ins = insert("php", shell, "\n".join(new), {"f"})
        assert ins.replaced == ("f",) and ins.kept_annotations == kept and ok("php", ins.text), ins.text
        assert forms(ins.text) == {p: [form] for p in OLD3}, ins.text
    same = "<?php\nclass C\n{\n" + "\n".join("    " + l for l in ATTRS[:1] + BODY) + "\n}\n"
    ins = insert("php", same, "\n".join(ATTRS[:1] + BODY), {"f"})  # the HEAD line itself: nothing to keep or drop
    assert ins.kept_annotations == 0 and ins.text == same


def test_a_create_pattern_is_never_changed_by_the_kept_annotation_rule():
    """The kept-annotation rule, branch (b): an annotation of the output whose pattern is not among the replaced
    method's OLD annotations (the answer's CREATE steps) is never dropped or converted: a CREATE step written as a
    docblock stays a docblock (ER then records it UNDEFINED under Behat 4, §2.3) next to the HEAD attributes that are
    kept."""
    shell = "<?php\nclass C\n{\n" + "\n".join("    " + l for l in ATTRS + BODY) + "\n}\n"
    new = [DOCS[0], DOCS[1], " * @When a new step :x", DOCS[4], "#[Then('another new step :x')]"] + BODY
    ins = insert("php", shell, "\n".join(new), {"f"})
    assert ins.kept_annotations == 3 and ok("php", ins.text)
    assert forms(ins.text) == {**{p: ["attribute"] for p in OLD3}, "a new step :x": ["docblock"],
                               "another new step :x": ["attribute"]}, ins.text
    assert "\n    /**\n     * @When a new step :x\n     */\n    #[Then('another new step :x')]\n" in ins.text
    java = 'public class S {\n    @Given("a {int}")\n    public void a(int x) {\n    }\n}\n'  # Java: the line rule
    assert insert("java", java, '@When("a {int}")\npublic void a(int x) {\n}', {("a", 1)}).kept_annotations == 1
    assert insert("java", java, '@Given( "a {int}" )\npublic void a(int x) {\n}', {("a", 1)}).kept_annotations == 1
    assert insert("java", java, '  @Given("a {int}")\npublic void a(int x) {\n}', {("a", 1)}).kept_annotations == 0


# ----- §2.2 (27/09/2026): the reader resolves names as the language does -------------------------------------------

def read(language: str, path: str, text: str, style: str, imports: bool = True) -> list[tuple]:
    """(keyword, kind, pattern, line) of the step definitions the reader reads in `text`: through the file's imports
    (stepdefs.file_imports; 3C's scoring path), or without them (None: Phase 1, the catalog)."""
    return [(d.keyword, d.kind, d.pattern, d.line)
            for d in extract_defs(path, text, style, file_imports(language, text) if imports else None)]


PHP_FORMS = """<?php
namespace App;

use Behat\\Step\\Then;
use Behat\\Step\\Given as G;

class C
{
    #[Then('a ' . "b :x")]
    public function a($x) {}

    #[G(pattern: 'c :x'), \\Behat\\Step\\When('d :x')]
    public function b($x) {}

    #[then('e :x')]
    public function c($x) {}

    #[\\behat\\step\\Then('f :x')]
    public function d($x) {}

    #[When('g :x')]
    public function e($x) {}

    #[Then('h :x')]
    private function f($x) {}

    #[Then('it\\'s :x')]
    public function g($x) {}
}
"""


def test_the_reader_resolves_php_attribute_names_as_php_does():
    """Behat 4.0.0-alpha1, shown by a real run: an attribute is a step when its name resolves to
    Behat\\Step\\{Given,When,Then}: a short name through its `use` in any case, an alias (`use ... as G`), the fully
    qualified name as written (`\\behat\\step\\Then` is not registered); `pattern:`, a concatenation of literals,
    several attributes in one `#[...]`; on a public method only. `When` without its `use` is App\\When, no step.
    Without imports (Phase 1: the catalog, the same on the sealed repositories) a short name is read without its
    `use`, as before, and an alias is not."""
    assert read("php", "C.php", PHP_FORMS, "php_attribute") == [
        ("Then", "turnip", "a b :x", 9), ("Given", "turnip", "c :x", 12), ("When", "turnip", "d :x", 12),
        ("Then", "turnip", "e :x", 15), ("Then", "turnip", "it's :x", 27)]
    assert read("php", "C.php", PHP_FORMS, "php_attribute", imports=False) == [
        ("Then", "turnip", "a b :x", 9), ("When", "turnip", "d :x", 12), ("Then", "turnip", "e :x", 15),
        ("When", "turnip", "g :x", 21), ("Then", "turnip", "it's :x", 27)]


JAVA_FORMS = """package p;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.*;

public class S {
    @Given("a {int}")
    public void a(int x) {}

    @ Then ( value = "b " + "{int}" )
    public void b(int x) {}

    @io.cucumber.java.en.When("c {int}")
    @io.cucumber.java.When("d {int}")
    public void c(int x) {}

    @Given("e {int}")
    void d(int x) {}

    @Given("f \\"{int}\\"")
    public void e(int x) {}

    @Given(PATTERN)
    public void f(int x) {}

    @Given("g {int}")
    @Deprecated(since = "{1}")
    public void g(int x) {}
}
"""


def test_the_reader_resolves_java_annotation_names_as_java_does():
    """Cucumber-JVM 7.34.3 and 7.8.1, shown by real runs: an annotation is a step when its name resolves to
    io.cucumber.java.en.{Given,When,Then,And,But} (fully qualified, a single-type or an on-demand import), any
    whitespace, `value =`, a concatenation of literals; on a public method only (a brace inside a string between the
    annotations and the body does not hide `public`). io.cucumber.java.When is no step; a constant as the pattern is
    not read (a documented limit). A file without the import reads nothing; Phase 1 reads the simple name without it,
    as before."""
    want = [("Given", "cucumber", "a {int}", 7), ("Then", "cucumber", "b {int}", 10), ("When", "cucumber", "c {int}", 13),
            ("Given", "cucumber", 'f "{int}"', 20), ("Given", "cucumber", "g {int}", 26)]
    assert read("java", "S.java", JAVA_FORMS, "java_annotation") == want
    assert read("java", "S.java", JAVA_FORMS, "java_annotation", imports=False) == want
    bare = 'package p;\n\npublic class S {\n    @Given("a {int}")\n    public void a(int x) {}\n}\n'
    assert read("java", "S.java", bare, "java_annotation") == []
    assert read("java", "S.java", bare, "java_annotation", imports=False) == [("Given", "cucumber", "a {int}", 4)]


CS_FORMS = """using TechTalk.SpecFlow;
using T = TechTalk.SpecFlow.ThenAttribute;

namespace N;

[Binding]
public class S
{
    [Given(@"^a ""(\\d+)""$")]
    void A(int x) {}

    [T("^b (\\\\d+)$"), When(Regex = "^c " + "(\\\\d+)$", Culture = "en")]
    public void B(int x) {}

    [TechTalk.SpecFlow.StepDefinitionAttribute("^d$")]
    private void C() {}

    [then("^e$")]
    public void D() {}

    [Given]
    public void E() {}
}
"""


def test_the_reader_resolves_csharp_attribute_names_as_csharp_does():
    """SpecFlow 3.0.213, shown by a real run: an attribute is a step when its name, with or without the Attribute
    suffix, resolves to TechTalk.SpecFlow.{Given,When,Then,StepDefinition}Attribute (fully qualified, a using alias,
    `using TechTalk.SpecFlow;`), case-sensitively; `Regex =` and other named arguments, a concatenation, verbatim and
    regular literals unescaped, several attributes in one `[...]`; on a method of any access level (SpecFlow scans
    non-public methods). `[Given]` without a regex is not read (a documented limit). Phase 1 reads the names without
    the usings, as before (so not the alias)."""
    assert read("csharp", "S.cs", CS_FORMS, "csharp_attribute") == [
        ("Given", "regex_full", '^a "(\\d+)"$', 9), ("Then", "regex_full", "^b (\\d+)$", 12),
        ("When", "regex_full", "^c (\\d+)$", 12), ("StepDefinition", "regex_full", "^d$", 15)]
    assert read("csharp", "S.cs", CS_FORMS, "csharp_attribute", imports=False) == [
        ("Given", "regex_full", '^a "(\\d+)"$', 9), ("When", "regex_full", "^c (\\d+)$", 12),
        ("StepDefinition", "regex_full", "^d$", 15)]


def test_a_ruby_string_pattern_is_a_cucumber_expression_even_with_anchors():
    """cucumber-ruby 11.1.1, shown by a real run: a String pattern is always a Cucumber expression (its `^` and `$`
    are literal text there); only a Regexp literal is a regex. Ruby has no imports: the same in both modes."""
    rb = "Given('a {int}') do |x|\nend\n\nThen(\"^b (\\\\d+)$\") do |x|\nend\n\nWhen(/^c (\\d+)$/) do |x|\nend\n"
    want = [("Given", "cucumber", "a {int}", 1), ("Then", "cucumber", "^b (\\d+)$", 4), ("When", "regex", "^c (\\d+)$", 7)]
    assert read("ruby", "s.rb", rb, "ruby_block") == read("ruby", "s.rb", rb, "ruby_block", imports=False) == want


def test_output_annotation_lines_follow_the_reader():
    """insert.STEP_ANNOTATION (the kept-annotation line rule) knows the reader's forms line by line (without the file's
    imports: an alias is not recognised on one line, a documented limit): a PHP group of several attributes, the fully
    qualified name, whitespace, `pattern:`; Java's fully qualified name and whitespace; C#'s suffix and namespace."""
    php = STEP_ANNOTATION["php"].match
    for line in ("#[Then('x'), Given('y')]", "#[\\Behat\\Step\\Then('x')]", " * @Then x", "#[ then ( pattern: 'x' ) ]"):
        assert php(line), line
    for line in ("#[Route('/x')]", "#[Then('x')] public function f()", "#[Other\\Then('x')]"):
        assert not php(line), line
    java = STEP_ANNOTATION["java"].match
    assert all(java(l) for l in ('@io.cucumber.java.en.Then("x")', '@ Then ( "x" )', '@And("x")', '@Then(value = "x")'))
    assert not java("@Override")
    cs = STEP_ANNOTATION["csharp"].match
    assert all(cs(l) for l in ('[ThenAttribute("x")]', '[TechTalk.SpecFlow.Then("x")]', '[StepDefinition(@"x")]'))
    assert not cs("[Binding]")


def test_php_kept_rewrites_a_group_of_several_attributes():
    """The kept-annotation rule on a group: an OLD pattern the output also writes inside a `#[...]` of several
    attributes is dropped from that group alone (the group is rewritten with the others, or removed when none stays),
    and the HEAD line is kept; no pattern is registered twice (Behat 4: RedundantStepException)."""
    shell = "<?php\nclass C\n{\n" + "\n".join("    " + l for l in ATTRS + BODY) + "\n}\n"
    cases = ((["#[Given('the store has :x'), Then('a new step :x')]"] + BODY, {"a new step :x"},
             "    #[Then('a new step :x')]\n    #[Given('the store has :x')]\n"),
             (["#[Given('the store has :x'), Given('the shop\\'s :x')]"] + BODY, set(),
              "{\n    #[Given('the store has :x')]\n"),
             (["#[Then('a new step :x'), Given('the store also has :x'), When('another :x')]"] + BODY,
              {"a new step :x", "another :x"}, "    #[Then('a new step :x'), When('another :x')]\n"))
    for new, created, where in cases:
        ins = insert("php", shell, "\n".join(new), {"f"})
        assert ins.replaced == ("f",) and ins.kept_annotations == 3 and ok("php", ins.text), ins.text
        assert forms(ins.text) == {p: ["attribute"] for p in OLD3 | created}, ins.text
        assert where in ins.text, ins.text



def test_a_byte_order_mark_hides_no_import():
    """A byte-order mark (U+FEFF) is no text to the compilers; 105 of gpconnect's 128 .cs files start with one, just
    before `using TechTalk.SpecFlow;` (Steps/ManualSteps.cs @ c592582: the scoring path read 0 of its 9 definitions).
    file_imports drops it wherever it stands, as when the callers join an output block and the file: `block + "\n" +
    file`."""
    cs = "﻿using TechTalk.SpecFlow;\n\nnamespace N\n{\n    [Binding]\n    public class S\n    {\n" \
         "        [Given(\".*\"), When(\".*\"), Then(\".*\")]\n        public void E() {}\n    }\n}\n"
    assert [d[0] for d in read("csharp", "S.cs", cs, "csharp_attribute")] == ["Given", "When", "Then"]
    assert file_imports("csharp", "[Then(\"^x$\")]\npublic void T() {}\n" + "\n" + cs)["namespaces"] == ["TechTalk.SpecFlow"]
    java = "﻿import io.cucumber.java.en.Given;\n\npublic class S {\n    @Given(\"a\")\n    public void a() {}\n}\n"
    assert [d[:3] for d in read("java", "S.java", java, "java_annotation")] == [("Given", "cucumber", "a")]



def rejected(language: str, text: str, style: str) -> list[str]:
    """The names of the step-looking candidates of `text` that the reader does not take (extract's `rejected`)."""
    out: list = []
    extract_defs("F", text, style, file_imports(language, text) if language != "ruby" else None, rejected=out)
    return [name for _, name in out]


def test_the_reader_reports_what_it_does_not_take():
    """Since 28/09/2026 extract's `rejected` lists every candidate of the style that looks like a step (its last name,
    an alias resolved, is a keyword in any case) and that the reader itself does not take, so that describe's
    unread_step_annotations counts with this reader: a foreign class under an alias or a qualified name, a constant
    pattern, a private method, a docblock tag in the wrong case; Java without the import, a constant, a package-private
    method; C# `[Then]` without a regex, another namespace's Then; a Ruby step of no readable form. A "then" in a
    docblock's prose, `#[Route]`, `@Override` and `[Binding]` are no candidates; what is read is none."""
    php = ("<?php\nuse Behat\\Step\\Then as T;\nuse Foo\\Bar\\Then as X;\nclass C {\n    #[T('a')]\n    public function a() {}\n\n"
           "    #[\\Foo\\Then('b')]\n    public function b() {}\n\n    #[X('c')]\n    public function c() {}\n\n"
           "    #[Route('/d'), Then(self::P)]\n    public function d() {}\n\n    #[T('e')]\n    private function e() {}\n\n"
           "    /**\n     * If x, then (y) we go\n     * @then lower\n     */\n    public function f() {}\n}\n")
    assert rejected("php", php, "php_attribute") == ["\\Foo\\Then", "X", "Then", "T"]
    assert rejected("php", php, "php_docblock") == []  # Behat 3 registers `@then` (a real run of 3.26.0)
    java = ('import io.cucumber.java.en.Given;\npublic class S {\n  @Given(P)\n  public void a() {}\n  @And("x")\n  public void b() {}\n'
            '  @Override\n  public void c() {}\n  @Given("ok")\n  void d() {}\n  @Given("read")\n  public void e() {}\n}\n')
    assert rejected("java", java, "java_annotation") == ["Given", "And", "Given"]
    cs = ('using TechTalk.SpecFlow;\n[Binding]\npublic class S {\n  [Then]\n  public void A() {}\n'
          '  [Other.Then("x"), Scope(Tag="t")]\n  public void B() {}\n  [Given("ok")]\n  public void C() {}\n}\n')
    assert rejected("csharp", cs, "csharp_attribute") == ["Then", "Other.Then"]
    rb = "Given(:sym) do\nend\n\nThen('ok') do\nend\n\nWhen(/re/) do\n"  # the last block never ends,
    assert rejected("ruby", rb, "ruby_block") == ["Given"]  # cut off, so it is not recorded (was ["Given", "When"])
    assert rejected("php", "<?php\nclass C {\n    #[Route('/x')]\n    public function r() {}\n}\n", "php_attribute") == []


def test_the_reader_records_nothing_of_a_cut_method():
    """§2.2 (28/09/2026): a method cut mid-body (an output stopped at the token cap) is never judged by the reader, so
    none of its candidates is recorded as declined, in any language and whatever the order of its annotations. Three
    cases used to differ: a Ruby `do` block that never ends was recorded where the same block with `{` was not; a
    declined form on a cut body was recorded (it needs no body), in PHP, Java, C# and Ruby; and `#[\\Foo\\Then] #[Then]`
    on a cut body was recorded while the reverse order was not. What comes before the cut method still is, and the
    definitions read before the reader stops are kept in `read`; the definitions returned and the ValueError raised do
    not change."""
    def seen(language: str, text: str, style: str):
        out, read = [], []
        try:
            extract_defs("F", text, style, file_imports(language, text) if language != "ruby" else None,
                         rejected=out, read=read)
            stopped = False
        except ValueError:
            stopped = True
        return [name for _, name in out], [d.pattern for d in read], stopped
    php = "<?php\nuse Behat\\Step\\Then;\nclass C {\n"
    done, cut = "\n    public function a() {}\n\n", "\n    public function t() {\n        $x = 1;\n"
    assert seen("php", php + "    #[\\Foo\\Then('x')]" + cut, "php_attribute") == ([], [], False)
    assert seen("php", php + "    #[\\Foo\\Then('x')]\n    #[Then('y')]" + cut, "php_attribute") == ([], [], True)
    assert seen("php", php + "    #[Then('y')]\n    #[\\Foo\\Then('x')]" + cut, "php_attribute") == ([], [], True)
    assert seen("php", php + "    #[\\Foo\\Then('a')]" + done + "    #[Then('y')]\n    #[\\Foo\\Then('x')]" + cut,
                "php_attribute") == (["\\Foo\\Then"], [], True)
    assert seen("php", php + "    #[Then('a')]" + done + "    #[\\Foo\\Then('x')]\n    #[Then('y')]" + cut,
                "php_attribute") == ([], ["a"], True)
    assert seen("php", php + "    /**\n     * @And other\n     */" + cut, "php_docblock") == ([], [], False)  # Behat 3:
    assert seen("php", php + "    /**\n     * @then lower\n     */" + cut, "php_docblock") == ([], [], True)  # @then reads
    assert seen("php", php + "    /**\n     * @Then y\n     */" + cut, "php_docblock") == ([], [], True)
    java = "import io.cucumber.java.en.Then;\npublic class S {\n"
    jdone, jcut = "\n  public void a() {}\n", "\n  public void t() {\n    cart.x("
    assert seen("java", java + '  @foo.Then("x")' + jcut, "java_annotation") == ([], [], False)
    assert seen("java", java + '  @foo.Then("x")\n  @Then("y")' + jcut, "java_annotation") == ([], [], True)
    assert seen("java", java + '  @Then("y")\n  @foo.Then("x")' + jcut, "java_annotation") == ([], [], True)
    assert seen("java", java + '  @And("a")' + jdone + '  @Then("y")' + jcut, "java_annotation") == (["And"], [], True)
    inner = '  @Then("y")\n  public void t() {\n    new Object() {\n      @Then("inner")\n      public void g() {}\n    };\n    x('
    assert seen("java", java + inner, "java_annotation") == ([], [], True)  # inside the cut body: never judged
    assert seen("java", 'Given(P, (String a) -> {\n  cart.x(', "java_lambda") == ([], [], False)
    assert seen("java", 'Given(P, (String a) -> {\n  cart.x();\n});\nThen("y", () -> {\n  x(', "java_lambda") == \
        (["Given"], [], True)
    assert seen("java", 'Then("y", () -> {\n  Given("inner", () -> {\n  });\n  x(', "java_lambda") == ([], [], True)
    cs = "using TechTalk.SpecFlow;\npublic class S {\n"
    ccut = "\n  public void T() {\n    x = 1;\n"
    assert seen("csharp", cs + '  [Other.Then("x")]' + ccut, "csharp_attribute") == ([], [], False)
    assert seen("csharp", cs + '  [Other.Then("x")]\n  [Then("y")]' + ccut, "csharp_attribute") == ([], [], True)
    assert seen("csharp", cs + '  [Then("y")]\n  [Other.Then("x")]' + ccut, "csharp_attribute") == ([], [], True)
    assert seen("ruby", "When(/re/) do\n  x = 1\n", "ruby_block") == ([], [], False)  # was ["When"]
    assert seen("ruby", "When(/re/) {\n  x = 1\n", "ruby_block") == ([], [], True)
    assert seen("ruby", "Given(:sym) do\n  x = 1\n", "ruby_block") == ([], [], False)
    assert seen("ruby", "Given(:sym) do\nend\n\nWhen(/re/) do\n  x = 1\n", "ruby_block") == (["Given"], [], False)
    # a brace in a LATER annotation's pattern ({string}, {int}, {2}) is not the method's own `{`,
    # so it no longer passes for the body of a cut method (unread 1 before); complete code reads as before
    two = '  @Then("I should see the total {string}")\n  @Then("I see the sum {string}")'
    assert seen("java", java + two + jcut, "java_annotation") == ([], [], True)
    assert seen("java", java + '  @Then("I see the sum {string}")\n  @Then("I should see the total {string}")' + jcut,
                "java_annotation") == ([], [], True)
    assert seen("java", java + '  @Then("I should see {int} items")' + jcut, "java_annotation") == ([], [], True)
    assert seen("java", java + two + "\n  public void t(String a) {\n    x();\n  }\n", "java_annotation") == \
        ([], ["I should see the total {string}", "I see the sum {string}"], False)
    assert seen("csharp", cs + '  [Then("^a$")]\n  [Then("^b{2}$")]' + ccut, "csharp_attribute") == ([], [], True)
    assert seen("ruby", "Then(/^x{2}$/) {\n  x = 1\n", "ruby_block") == ([], [], True)


# ----- 28/09/2026: the reader reads code as the compiler does --------------------------------------------------------

def test_a_word_or_a_brace_inside_a_string_or_a_comment_is_no_code():
    """On complete code, `private` in a later attribute's pattern or in a comment is no modifier (the public check);
    `function` or `{` inside an annotation's string is not the method's (the body read); an annotation, an attribute, a
    doc comment or a java8 call inside a comment, a string, a text block or a heredoc is none, as for the compiler (so
    for the framework). On the pinned repos the scoring path thus drops four commented-out definitions (fineract
    LoanStepDef.java:5446, :5458; gpconnect AccessStructuredMedicationSteps.cs:670, StructuredMultipleSteps.cs:185) and
    reads the body of fineract WorkingCapitalNearBreachActionStepDef.java:78 from its own `{`. Phase 1 (no imports: the
    sealed catalog) keeps the sealed path, ghosts included (_extract_sealed; test_catalog pins its `dead` flags on
    them)."""
    php = ("<?php\nuse Behat\\Step\\{Given, Then, When};\nclass C {\n"
           "    #[Given('I am logged in as :user')]\n    #[Then('I should see the private channel :name')]\n"
           "    public function a($user, $name) {}\n\n"
           "    #[Then('y')]\n    // protected by the firewall\n    public function b() {}\n\n"
           "    #[Then('z')]\n    #[When('I call function {x}')]\n    public function c($x) {\n        $a = 1;\n    }\n\n"
           "    // #[Then('ghost 1')]\n    /* #[Then('ghost 2')] */\n    # #[Then('ghost 3')]\n"
           "    public function d() {\n        $s = \"#[Then('ghost 4')] public function\";\n"
           "        $h = <<<EOT\n#[Then('ghost 5')]\npublic function e() {}\nEOT;\n    }\n}\n")
    want = [("Given", "turnip", "I am logged in as :user", 4), ("Then", "turnip", "I should see the private channel :name", 5),
            ("Then", "turnip", "y", 8), ("Then", "turnip", "z", 12), ("When", "turnip", "I call function {x}", 13)]
    assert read("php", "C.php", php, "php_attribute") == want
    assert ("Then", "turnip", "ghost 1", 18) in read("php", "C.php", php, "php_attribute", imports=False)  # Phase 1: sealed
    c = [d for d in extract_defs("C.php", php, "php_attribute", file_imports("php", php)) if d.pattern == "z"][0]
    assert (c.body_start, c.body_end) == (14, 16), c
    hidden = ("<?php\nuse Behat\\Step\\Then;\nclass C {\n    #[Then('y')]\n    #[Route('call function x')]\n"
              "    private function t() {}\n}\n")  # the method's own `function` is the private one
    assert read("php", "C.php", hidden, "php_attribute") == []
    ident = hidden.replace("#[Route('call function x')]", "#[Attr(Foo::functionName)]")  # the keyword, not a name
    assert read("php", "C.php", ident, "php_attribute") == []
    doc = ("<?php\nclass C {\n    /**\n     * @Given x\n     */\n    #[Deprecated('private api')]\n    public function a() {}\n\n"
           "    public function b() {\n        $s = '/** @Given ghost 1 */';\n        // /** @Given ghost 2 */\n    }\n\n"
           "    /**@Given ghost 3 */\n    public function c() {}\n\n"
           "    /**\n     * @Then y\n     */\n    # protected by the firewall\n    public function d() {}\n\n"
           "    /**\n     * @Then z\n     */\n    #[Route('call function x')]\n    private function e() {}\n\n"
           "    // /** @Given ghost 4 */\n    public function f() {}\n\n"
           "    #[Route('/** @Given ghost 5 */')]\n    public function g() {}\n}\n")
    assert read("php", "C.php", doc, "php_docblock") == [("Given", "turnip", "x", 4), ("Then", "turnip", "y", 18)]
    java = ('import io.cucumber.java.en.*;\npublic class S {\n  @Then("y")\n  @Given("x {string}")\n'
            '  public void t(String a) {\n    char q = \'"\';\n  }\n\n  @When("w")\n  public void w() {\n  }\n\n'
            '  // @Given("ghost 1")\n  /* @Given("ghost 2") */\n  public void u() {\n    String s = """\n'
            '      say "hi\n      @Given("ghost 3")\n      """;\n  }\n\n  @When("v")\n  public void v() {\n  }\n}\n')
    assert read("java", "S.java", java, "java_annotation") == [("Then", "cucumber", "y", 3),
        ("Given", "cucumber", "x {string}", 4), ("When", "cucumber", "w", 9), ("When", "cucumber", "v", 22)]
    assert [(d.body_start, d.body_end) for d in extract_defs("S.java", java, "java_annotation", file_imports("java", java))] == \
        [(5, 7), (5, 7), (10, 11), (23, 24)]
    lam = 'public S() {\n  /*\n  Given("ghost", () -> {\n  });\n  */\n  Given("a", () -> {\n  });\n}\n'
    assert read("java", "S.java", lam, "java_lambda") == [("Given", "cucumber", "a", 6)]
    cs = ('using TechTalk.SpecFlow;\npublic class S {\n  [Then("^y$")]\n  [Given("^x{2}$")]\n  public void T() {\n'
          '    var c = \'"\';\n  }\n\n  //[Then(@"ghost 1")]\n  /*\n  [Then(@"ghost 2")]\n  public void G() {}\n  */\n'
          '  [When("^w$")]\n  public void W() {\n  }\n}\n')
    assert read("csharp", "S.cs", cs, "csharp_attribute") == [
        ("Then", "regex_full", "^y$", 3), ("Given", "regex_full", "^x{2}$", 4), ("When", "regex_full", "^w$", 14)]
    assert [(d.body_start, d.body_end) for d in extract_defs("S.cs", cs, "csharp_attribute", file_imports("csharp", cs))] == \
        [(5, 7), (5, 7), (15, 16)]
    rb = "Given('I have {int} items') { |n|\n  x = n\n}\n"
    d = extract_defs("s.rb", rb, "ruby_block")
    assert [(x.pattern, x.body_start, x.body_end, x.code) for x in d] == [("I have {int} items", 1, 3, "{ |n|\n  x = n\n}")]
    assert [it.kind for it in split_code("ruby", rb)[1]] == ["block"]


def test_a_brace_or_a_function_inside_a_later_pattern_never_hides_a_cut_method():
    """§2.2: a cut method records nothing, whatever its form and the order of its annotations: the method's own `{` (or
    `function`) is the first one outside strings and comments, so a `{string}`, `{2}`, `{int}` or the word `function` in
    a later annotation's pattern does not pass for it, in Java, C#, PHP and Ruby."""
    def seen(language: str, text: str, style: str):
        out, read_ = [], []
        try:
            extract_defs("F", text, style, file_imports(language, text) if language != "ruby" else None,
                         rejected=out, read=read_)
            stopped = False
        except ValueError:
            stopped = True
        return [name for _, name in out], [d.pattern for d in read_], stopped
    java = "import io.cucumber.java.en.Then;\nimport io.cucumber.java.en.Given;\npublic class S {\n"
    jcut = "\n  public void t() {\n    cart.x("
    assert seen("java", java + '  @Then("y")\n  @Given("x {string}")' + jcut, "java_annotation") == ([], [], True)
    assert seen("java", java + '  @Then("y")\n  @foo.Then("x {string}")' + jcut, "java_annotation") == ([], [], True)
    cs = "using TechTalk.SpecFlow;\npublic class S {\n"
    ccut = "\n  public void T() {\n    x = 1;\n"
    assert seen("csharp", cs + '  [Then("^y$")]\n  [Other.Then("^x{2}$")]' + ccut, "csharp_attribute") == ([], [], True)
    php = "<?php\nuse Behat\\Step\\{Then, When};\nclass C {\n"
    pcut = "\n    public function t($x) {\n        $a = 1;\n"
    assert seen("php", php + "    #[Then('y')]\n    #[When('I call function {x}')]" + pcut, "php_attribute") == ([], [], True)
    assert seen("php", php + "    #[\\Foo\\Then('y')]\n    #[When('call function {x}')]" + pcut, "php_attribute") == \
        ([], [], True)
    assert seen("php", php + "    /**\n     * @Then y\n     */\n    #[Route('function {x}')]" + pcut, "php_docblock") == \
        ([], [], True)
    assert seen("ruby", "Given('I have {int} items') { |n|\n  x = n\n", "ruby_block") == ([], [], True)


def test_a_php_import_already_in_use_is_not_added_again():
    """Duplicate imports are dropped. PHP refuses a second `use` of an alias in use (a Fatal: an ERROR of the output
    under Behat 4), whatever the form; a name imported as the same class (case-insensitive, a leading `\\`, blanks
    before `;`, a group) is dropped, a group keeps its other names; a name whose alias is another class stays (the
    output's own clash). The stubs' `use Behat\\Step\\When;` into Sylius's grouped import is the case of the ER
    overlays. The results pass `php -l` (PHP 8.4)."""
    head = ("<?php\nnamespace A;\n\nuse Behat\\Step\\{Given, Then, When};\nuse \\Foo\\Bar ;\n\nfinal class C\n{\n"
            "    #[Given('x')]\n    public function x(): void {}\n}\n")
    base = ["use Behat\\Step\\{Given, Then, When};", "use \\Foo\\Bar ;"]
    for block, added in (("use Behat\\Step\\When;", []), ("use \\Behat\\Step\\Then;", []), ("use Behat\\Step\\Then ;", []),
                         ("use behat\\step\\when;", []), ("use Foo\\Bar;", []),
                         ("use Behat\\Step\\{Given, Other};", ["use Behat\\Step\\{Other};"]),
                         ("use Baz\\When;", ["use Baz\\When;"]),
                         ("use Baz\\Qux;\nuse Baz\\{Qux, Quux as Q};", ["use Baz\\Qux;", "use Baz\\{Quux as Q};"]),
                         ("use function Behat\\f;", ["use function Behat\\f;"])):
        out = insert("php", head, block + "\n\n#[When('y')]\npublic function y(): void {}\n")
        assert [l for l in out.text.split("\n") if l.startswith("use ")] == base + added, out.text
        assert ok("php", out.text) and out.added == 1


def test_php_kept_compares_every_pattern_of_a_head_group():
    """A HEAD attribute group of several step attributes (`#[Given('a :x'), When('b :x')]`) is compared with the output
    by EVERY pattern it holds, not its first: an output annotation of any of them is dropped and the HEAD group kept,
    so no pattern is registered twice (Behat 4: RedundantStepException); a CREATE pattern stays."""
    shell = "<?php\nclass C\n{\n    #[Given('a :x'), When('b :x')]\n    public function f($x) {}\n}\n"
    for new, created in (("#[When('b :x')]", set()), ("#[Given('a :x')]\n#[When('b :x')]", set()),
                         ("#[Given('a :x'), When('b :x')]", set()), ("#[When('b :x'), Then('c :x')]", {"c :x"}),
                         ("/**\n * @When b :x\n */", set())):
        ins = insert("php", shell, new + "\npublic function f($x) { echo 1; }", {"f"})
        pats = [d.pattern for st in ("php_attribute", "php_docblock") for d in extract_defs("C.php", ins.text, st)]
        assert Counter(pats) == Counter({"a :x", "b :x"} | created), ins.text
        assert "#[Given('a :x'), When('b :x')]" in ins.text and ok("php", ins.text), ins.text


def test_insertion_rules_pinned_by_their_counter_examples():
    """A lambda goes into the constructor that HOLDS the lambdas, not the first one; a java8 call that is no whole
    `(...);` statement does not stop the lambdas after it; a one-line doc comment `/** @Kw p */` is a PHP step
    annotation; a group of a step and a non-step attribute loses only the step; a class whose closing brace shares its
    line with code gets the method inside it."""
    java = ("public class S {\n    public S(int a) {\n        x();\n    }\n\n    public S() {\n"
            "        Given(\"a\", () -> {\n        });\n    }\n}\n")
    out = insert("java", java, 'When("b", () -> {\n});').text
    assert out.index('When("b"') > out.index('Given("a"') > out.index("public S()"), out
    code = 'Given("a", () -> {\n});\nWhen("b", () -> {\n}).toString();\nThen("c", () -> {\n});'
    assert [it.kind for it in split_code("java", code)[1]].count("lambda") == 2
    from v2tools.insert import php_steps
    assert [(src, pat) for _, _, src, pat in php_steps("/** @Given a :x */\npublic function f() {}")] == \
        [("@Given a :x", "a :x")]
    shell = "<?php\nclass C\n{\n    #[Given('a :x')]\n    public function f($x) {}\n}\n"
    ins = insert("php", shell, "#[Route('/x'), Given('a :x')]\npublic function f($x) { echo 1; }", {"f"})
    assert "#[Route('/x')]" in ins.text and ins.text.count("a :x") == 1 and ok("php", ins.text), ins.text
    one = "<?php\nclass C\n{\n    public function a() {} }\n"
    ins = insert("php", one, "#[Then('b')]\npublic function b() {}")
    assert ok("php", ins.text) and [m[0] for m in class_methods("php", ins.text)] == ["a", "b"], ins.text


def test_reader_rules_pinned_by_their_counter_examples():
    """A PHP doc comment on a private method is not read and what the reader reads is never recorded as declined; an
    attribute with no method after it raises, as the reader's contract says; a C# verbatim string's `""` and `]` do not
    end the group; `…Attribute` looks like a step; whitespace inside a Java qualified name is read. The names the
    frameworks do NOT register stay unread in both modes: PHP attribute And/But/Step, Java io.cucumber.java.en.Step, C#
    And/But/Step (shown by real runs); and a cucumber-java8 or cucumber-ruby `Step`/`StepDefinition` call is declined
    (also shown by real runs)."""
    doc = "<?php\nclass C {\n    /**\n     * @Given a\n     */\n    private function a() {}\n\n" \
          "    /**\n     * @Given b\n     */\n    public function b() {}\n}\n"
    assert read("php", "C.php", doc, "php_docblock") == \
        [("Given", "turnip", "b", 9)]
    assert rejected("php", doc, "php_docblock") == ["@Given"]
    try:
        head = "<?php\nuse Behat\\Step\\Then;\nclass C {\n    #[Then('x')]\n"
        extract_defs("C.php", head, "php_attribute", file_imports("php", head))
        raised = False
    except ValueError:
        raised = True
    assert raised
    cs = ('using TechTalk.SpecFlow;\npublic class S {\n  [Given(@"a ""]"" b")]\n  public void A() {}\n'
          '  [Given(@"^the folder is ""C:\\temp\\""$"), When("^w$")]\n  public void B() {}\n}\n')
    assert read("csharp", "S.cs", cs, "csharp_attribute") == [
        ("Given", "regex_full", 'a "]" b', 3), ("Given", "regex_full", '^the folder is "C:\\temp\\"$', 5),
        ("When", "regex_full", "^w$", 5)]
    cs2 = 'using TechTalk.SpecFlow;\npublic class S {\n  [Other.ThenAttribute("x")]\n  public void A() {}\n}\n'
    assert rejected("csharp", cs2, "csharp_attribute") == ["Other.ThenAttribute"]
    java = 'public class S {\n  @io . cucumber . java . en . Then ( "x" )\n  public void a() {}\n}\n'
    assert read("java", "S.java", java, "java_annotation") == [("Then", "cucumber", "x", 2)]
    php = ("<?php\nuse Behat\\Step\\{And, But, Step};\nclass C {\n    #[And('a')]\n    public function a() {}\n"
           "    #[\\Behat\\Step\\But('b')]\n    public function b() {}\n    #[Step('c')]\n    public function c() {}\n}\n")
    jv = ("import io.cucumber.java.en.*;\npublic class S {\n  @io.cucumber.java.en.Step(\"a\")\n  public void a() {}\n"
          "  @Step(\"b\")\n  public void b() {}\n}\n")
    cn = ('using TechTalk.SpecFlow;\npublic class S {\n  [And("^a$")]\n  public void A() {}\n  [But("^b$")]\n'
          '  public void B() {}\n  [Step("^c$"), StepAttribute("^d$")]\n  public void C() {}\n}\n')
    lam = 'Given("a", () -> {\n});\nStep("b", () -> {\n});\nStepDefinition("c", () -> {\n});\n'
    assert read("java", "S.java", lam, "java_lambda") == [("Given", "cucumber", "a", 1)]
    assert rejected("java", lam, "java_lambda") == ["Step", "StepDefinition"]
    rb = "And(/^a$/) do\nend\n\nStep(/^b$/) do\nend\n\nStepDefinition(/^c$/) do\nend\n"
    assert read("ruby", "s.rb", rb, "ruby_block") == [("And", "regex", "^a$", 1)]
    assert rejected("ruby", rb, "ruby_block") == ["Step", "StepDefinition"]
    for language, path, text, style in (("php", "C.php", php, "php_attribute"), ("java", "S.java", jv, "java_annotation"),
                                        ("csharp", "S.cs", cn, "csharp_attribute")):
        assert read(language, path, text, style) == read(language, path, text, style, imports=False) == [], language


def test_the_docblock_reader_reads_what_behat_3_registers():
    """§2.2: only forms the framework registers, each shown by a real run: Behat 3.26.0 and 3.33.0 (PrestaShop's
    docblocks) register `@given`/`@when`/`@then` in any case at a doc line's start, blanks, then the pattern; a line
    continued by a backslash; one-line and multi-line doc comments; the method's doc comment only (the last one before
    it, not one a property or a constant took), public or static. Not `@And`, `@But`, `@Step`, `@StepDefinition`,
    `@Given:`, `@Given` alone, a tag after prose or after a second `*`, `/**@Given` (no doc comment for PHP), a private
    method. Phase 1 (no imports: the sealed catalog) keeps the sealed path (_extract_sealed)."""
    doc = ("<?php\nclass C {\n    /** @then I do a */\n    public function a() {}\n\n"
           "    /**\n     * @Given I have a very long \\\n     *        pattern b\n     */\n    public static function b() {}\n\n"
           "    /** @Given I do c1 */\n    /** @Given I do c2 */\n    public function c() {}\n\n"
           "    /** @Given I do d */\n    public $p;\n\n    /** @Given I do e */\n    const E = 1;\n\n"
           "    /**\n     * @And I do f1\n     * @Step I do f2\n     * @Given: I do f3\n     * Use @Given f4 here\n"
           "     *      * @Then I do f5\n     *@WHEN I do f6\n     */\n    public function f() {}\n\n"
           "    /**@Given I do g */\n    public function g() {}\n\n"
           "    /** @Given I do h */\n    public $q;\n    public function h() {}\n\n"
           "    /**\n     * @Given I have a long \\\n     *   pattern i1\n     * @When I do i2\n     */\n    public function i() {}\n}\n")
    want = [("Then", "turnip", "I do a", 3), ("Given", "turnip", "I have a very long pattern b", 7),
            ("Given", "turnip", "I do c2", 13), ("When", "turnip", "I do f6", 28),
            ("Given", "turnip", "I have a long pattern i1", 40), ("When", "turnip", "I do i2", 42)]
    assert read("php", "C.php", doc, "php_docblock") == want
    assert rejected("php", doc, "php_docblock") == ["@Given", "@Given", "@Given", "@And", "@Step", "@Given", "@Then",
                                                    "@Given"]
