from v2tools.bind import Library, bind_scenario
from v2tools.gherkin_steps import Scenario, Step
from v2tools.stepdefs import StepDef

def sd(pattern, kind="turnip", line=1):
    return StepDef("C.php", "php_attribute", kind, "Given", pattern, line, line, line + 2, "")

LIB = Library([sd("there is a product :name", line=10), sd("I check out", line=20),
               sd("I check out", line=30), sd("(broken", kind="regex", line=40)])

def scenario(*texts):
    return Scenario("s", 1, [Step("Given", t, i + 2) for i, t in enumerate(texts)])

def test_bound_and_ambiguous():
    r = bind_scenario(scenario('there is a product "Mug"', "I check out"), LIB)
    assert r["status"] == "BOUND" and r["ambiguous_steps"] == 1
    assert r["target_ids"] == ["C.php:10", "C.php:20", "C.php:30"]

def test_partial_and_unbound():
    assert bind_scenario(scenario("I check out", "I fly"), LIB)["status"] == "PARTIAL"
    r = bind_scenario(scenario("I fly"), LIB)
    assert r["status"] == "UNBOUND" and r["unbound_steps"] == ["I fly"]

def test_uncompilable_patterns_are_counted_not_raised():
    assert LIB.uncompilable == 1

def test_index_does_not_hide_valid_matches():
    lib = Library([sd("is enabled", kind="regex", line=50),   # unanchored Ruby regex: searched anywhere
                   sd("product(s) are listed", line=60),      # optional suffix on the first word
                   sd("I/we see the cart", line=70),          # alternation on the first word
                   sd(r"^foo bar$|baz qux$", kind="regex_full", line=80)])  # top-level regex alternation
    assert [d.line for d in lib.matches("the some feature is enabled")] == [50]
    assert [d.line for d in lib.matches("products are listed")] == [60]
    assert [d.line for d in lib.matches("we see the cart")] == [70]
    assert [d.line for d in lib.matches("baz qux")] == [80]
