from v2tools.stepdefs import extract

PHP_ATTR = '''<?php
final class C {
    #[When('I set the position of :productVariant to :position')]
    public function setPosition(string $v, int $p): void
    {
        // don't break on this apostrophe { brace in comment
        $this->x = ['a' => '{'];
    }
}
'''
PHP_DOC = '''<?php
class D {
    /**
     * @Given the current currency is :currencyIsoCode
     * @When I switch currency to :code
     */
    public function currency($c) { return "}"; }
    /** @Then /^I should see (\\d+) items$/ */
    public function items($n) { }
}
'''
JAVA = '''class S {
    @Given("any year-end close for fiscal year ending {string} is removed")
    public void remove(String y) throws Exception {
        if (y != null) { run("\\"{"); }
    }
    @Then("^the total is (\\\\d+)$")
    public void total(int t) { }
}
'''
CS = '''public class Steps {
    [Given(@"I set the Parameter name ""(.*)"" to ""(.*)""")]
    public void SetParam(string n, string v) { _x = "}"; }
}
'''
RUBY = '''Given /^Clinics$/ do |clinics|
  if clinics
    x = 1
  end
end

Given(/the following consultants:/) do |table|
  table.hashes.each { |h| p h }
end
'''

def test_php_attribute():
    [d] = extract("C.php", PHP_ATTR, "php_attribute")
    assert d.kind == "turnip" and d.pattern == "I set the position of :productVariant to :position"
    assert d.line == 3 and d.body_end == 8 and "$this->x" in d.code

def test_php_docblock_multiple_annotations_and_regex_form():
    defs = extract("D.php", PHP_DOC, "php_docblock")
    assert [(d.keyword, d.kind) for d in defs] == [("Given", "turnip"), ("When", "turnip"), ("Then", "regex_full")]
    assert defs[0].body_start == defs[1].body_start
    assert defs[2].pattern == r"^I should see (\d+) items$"

def test_java_annotation_cucumber_and_anchored_regex():
    defs = extract("S.java", JAVA, "java_annotation")
    assert defs[0].kind == "cucumber" and "{string}" in defs[0].pattern and defs[0].body_end == 5
    assert defs[1].kind == "regex_full" and defs[1].pattern == r"^the total is (\\d+)$".replace("\\\\", "\\")

def test_csharp_verbatim_string():
    [d] = extract("Steps.cs", CS, "csharp_attribute")
    assert d.kind == "regex_full" and d.pattern == 'I set the Parameter name "(.*)" to "(.*)"'

def test_ruby_do_block_ends_at_matching_indent():
    defs = extract("x_steps.rb", RUBY, "ruby_block")
    assert [(d.pattern, d.line, d.body_end) for d in defs] == [("^Clinics$", 1, 5), ("the following consultants:", 7, 9)]

JAVA8 = '''public class S implements En {
    public S() {
        Given("the tenant {string} exists", (String t) -> {
            create(t, "}");
        });
        Then("/^The step service should have a result$/",
                () -> {
                    check();
                });
    }
}
'''
RUBY_MORE = '''observation_regex =
  /^request (\\w+) requires (\\w+)$/
Given(observation_regex) do |a, b|
  x = 1
end

Given(%r{^the dates (yes|no)$}) do |w|
end
'''

def test_java_lambda_and_slash_delimited_regex():
    defs = extract("S.java", JAVA8, "java_lambda")
    assert [(d.keyword, d.kind, d.pattern, d.line, d.body_end) for d in defs] == [
        ("Given", "cucumber", "the tenant {string} exists", 3, 5),
        ("Then", "regex_full", "^The step service should have a result$", 6, 9)]

def test_ruby_variable_and_percent_r_patterns():
    defs = extract("x_steps.rb", RUBY_MORE, "ruby_block")
    assert [(d.pattern, d.line, d.body_end) for d in defs] == [
        (r"^request (\w+) requires (\w+)$", 3, 5), ("^the dates (yes|no)$", 7, 8)]
