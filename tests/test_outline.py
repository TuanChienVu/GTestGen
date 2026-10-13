from v2tools.outline import has_syntax_error, parse

PHP = '''<?php
namespace App\\Behat;

use Behat\\Behat\\Context\\Context;
use App\\Page\\IndexPageInterface as Index;

final class ShopContext extends BaseContext implements Context
{
    use HelperTrait;

    private const LIMIT = 3;

    /** @var string */
    protected ?string $name = null;

    public function __construct(
        private readonly Index $indexPage,
        $plain,
    ) {
    }

    /**
     * @Given I open the shop
     */
    #[When('I browse')]
    public function iOpenTheShop(): void
    {
        $this->indexPage->open();
    }

    private function helper(): int { $this->counter = 1; return 1; }
}
'''

JAVA = '''package org.x.stepdef;

import org.springframework.beans.factory.annotation.Autowired;
import io.cucumber.java.en.Then;

public class LoanStepDef extends AbstractStepDef {
    @Autowired
    private LoanHelper loanHelper;
    @Autowired private ErrorHelper errorHelper;

    @Then("Loan has {int} status")
    public void loanHasStatus(int s) throws Exception {
        loanHelper.check(s);
    }
}
'''


def test_php_outline_has_namespace_imports_parents_and_members_without_annotations():
    o = parse("php", PHP)
    assert (o.namespace, o.name, o.kind) == ("App\\Behat", "ShopContext", "class")
    assert o.imports == {"Context": "Behat\\Behat\\Context\\Context", "Index": "App\\Page\\IndexPageInterface"}
    assert o.parents == (("extends", "BaseContext"), ("implements", "Context"), ("trait", "HelperTrait"))
    by = {m.names[0]: m for m in o.members}
    assert by["LIMIT"].kind == "const" and by["name"].kind == "property" and by["name"].visibility == "protected"
    assert by["indexPage"].kind == "promoted" and by["__construct"].kind == "constructor"
    step = by["iOpenTheShop"]
    assert step.text == "public function iOpenTheShop(): void\n    {"  # no attribute, no docblock
    assert step.body_line == 27 and step.line == 26
    assert o.header.startswith("final class ShopContext extends BaseContext implements Context")
    assert [p for p, _, _ in o.params[by["__construct"].line]] == ["private readonly Index $indexPage", "$plain"]
    assert [(promoted, name) for _, promoted, name in o.params[by["__construct"].line]] == [("indexPage", "indexPage"), (None, "plain")]


def test_java_outline_drops_annotations_inside_modifiers():
    o = parse("java", JAVA)
    assert (o.namespace, o.name) == ("org.x.stepdef", "LoanStepDef")
    assert o.parents == (("extends", "AbstractStepDef"),)
    by = {m.names[0]: m for m in o.members}
    assert by["loanHelper"].text == "private LoanHelper loanHelper;"
    assert by["errorHelper"].text == "private ErrorHelper errorHelper;"
    assert by["loanHasStatus"].text == "public void loanHasStatus(int s) throws Exception {"
    assert "@Then" not in by["loanHasStatus"].text and by["loanHasStatus"].body_line == 12


def test_no_class_and_ruby_give_none():
    assert parse("php", "<?php\nfunction f() {}\n") is None
    assert parse("ruby", "Given(/^x$/) do\nend\n") is None


def test_syntax_error_detects_error_and_missing_nodes():
    assert not has_syntax_error("php", PHP)
    assert has_syntax_error("php", PHP.replace("return 1; }", "return 1;"))
    assert has_syntax_error("java", JAVA.replace("check(s);", "check(s)"))
    assert not has_syntax_error("ruby", "Given(/^x$/) do\n  1\nend\n")
    assert has_syntax_error("ruby", "Given(/^x$/) do\n  1\n")
