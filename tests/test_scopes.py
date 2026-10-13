from pathlib import Path

from v2tools.patterns import compile_pattern
from v2tools.scopes import (NEGATIVE_STEPS, active_in_scope, is_catch_all, load_behat_suites, negative_hits,
                            pick_wrong_suite, specflow_scope_tags, tag_filter_matches)
from v2tools.stepdefs import extract

BROKEN_WITHCONTEXTS = """<?php
(new Suite('broken'))->withContexts('x', # don't
 'y');
"""

def test_unparseable_withcontexts_call_becomes_one_raw_missing_argument(tmp_path: Path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "broken.php").write_text(BROKEN_WITHCONTEXTS)
    [suite] = load_behat_suites(tmp_path, ["cfg/broken.php"], [], {})
    assert suite.context_args_total == 1 and suite.context_args_resolved == 0
    assert "<unparsed withContexts call>" in suite.missing_contexts

def test_negative_steps():
    assert len(NEGATIVE_STEPS) == 20 and len(set(NEGATIVE_STEPS)) == 20

def test_catch_all_requires_matching_every_negative_step():
    broad = compile_pattern("regex", "the (.*) is")       # legitimate unanchored Ruby regex, searched
    total = compile_pattern("regex_full", "^.*$")
    assert 0 < negative_hits(*broad) < len(NEGATIVE_STEPS) and not is_catch_all(*broad)
    assert negative_hits(*total) == len(NEGATIVE_STEPS) and is_catch_all(*total)

def test_tag_filter_expressions():
    assert tag_filter_matches("@a&&@b", ["@a", "@b", "@c"])
    assert not tag_filter_matches("@a&&@b", ["@a"])
    assert tag_filter_matches("@a,@b", ["@b"])
    assert tag_filter_matches("@a&&~@b", ["@a"]) and not tag_filter_matches("@a&&~@b", ["@a", "@b"])
    assert tag_filter_matches("@A", ["@a"])
    assert tag_filter_matches("", [])
    # Behat's TagFilter::isTagsMatchCondition: split on '&&' first (all parts must hold), ',' inside a part
    # is OR; a leading '@' is optional on both sides.
    assert not tag_filter_matches("@a,@b&&@c", ["@a"])
    assert tag_filter_matches("a", ["@a"])

SUITE = """<?php
use App\\Behat\\Context\\Ui\\Shop\\CountryContext as ShopCountryContext;
return (new Config())->withProfile((new Profile('default'))->withSuite(
    (new Suite('ui_countries', ['javascript' => false]))
        ->withContexts('app.context.setup.country', 'app.context.ui.country')
        ->withContexts('app.context.missing', ShopCountryContext::class, $extra, new Foo('x'))
        ->withFilter(new TagFilter('@managing_countries&&@ui')),
));
"""
SERVICES = """<?php
use App\\Behat\\Context\\Setup\\CountryContext;
use App\\Behat\\Context\\Ui\\ManagingCountriesContext as UiCountry;
return static function (ContainerConfigurator $c) {
    $c->services()
        ->set('app.context.setup.country', CountryContext::class)
        ->set('app.context.ui.country', UiCountry::class)->args([]);
};
"""
UI_CONTEXT = """<?php
namespace App\\Behat\\Context\\Ui;
use App\\Behat\\Context\\Ui\\Helper\\ValidationTrait;
final class ManagingCountriesContext
{
    use ValidationTrait;
}
"""

def test_load_behat_suites_resolves_contexts_and_traits(tmp_path: Path):
    files = {"cfg/suites/ui/countries.php": SUITE, "cfg/services/contexts.php": SERVICES,
             "src/Context/Setup/CountryContext.php": "<?php\nnamespace App\\Behat\\Context\\Setup;\nclass CountryContext {}\n",
             "src/Context/Ui/ManagingCountriesContext.php": UI_CONTEXT,
             "src/Context/Ui/Helper/ValidationTrait.php": "<?php\ntrait ValidationTrait {}\n",
             "src/Context/Ui/Shop/CountryContext.php": "<?php\nnamespace App\\Behat\\Context\\Ui\\Shop;\nclass CountryContext {}\n"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    [suite] = load_behat_suites(tmp_path, ["cfg/suites/ui/countries.php"], ["cfg/services/contexts.php"], {"App\\Behat\\": "src/"})
    assert suite.name == "ui_countries" and suite.tag_filter == "@managing_countries&&@ui"
    assert suite.files == frozenset({"src/Context/Setup/CountryContext.php", "src/Context/Ui/ManagingCountriesContext.php",
                                     "src/Context/Ui/Helper/ValidationTrait.php", "src/Context/Ui/Shop/CountryContext.php"})
    # 'app.context.missing' (unregistered service id), '$extra' and 'new Foo('x')' (neither a string nor
    # ::class) are all reported verbatim rather than silently dropped.
    assert suite.missing_contexts == ("app.context.missing", "$extra", "new Foo('x')")
    # 2 (first withContexts) + 4 (second: missing, ::class, $extra, new Foo('x')) = 6 raw arguments;
    # resolved = the 2 service ids + the one ::class reference = 3. The nested-paren argument proves the
    # raw splitter uses balanced-paren matching, not the old `[^)]*` regex (which would stop at Foo('x')'s
    # own closing paren).
    assert (suite.context_args_total, suite.context_args_resolved) == (6, 3)

MANUAL = '''using TechTalk.SpecFlow;
namespace X.Steps
{
    [Binding, Scope(Tag = "Manual")]
    public class ManualSteps
    {
        [Given(".*"), When(".*"), Then(".*")]
        public void EmptyStep()
        {
        }
    }

    [Binding]
    public class OtherSteps
    {
        [Scope(Tag = "slow")]
        [Given(@"a slow thing")]
        public void Slow() { }
    }
}
'''

def test_specflow_scope_spans():
    spans = specflow_scope_tags("Steps.cs", MANUAL)
    assert (4, 11, frozenset({"@manual"})) in spans
    assert (16, 18, frozenset({"@slow"})) in spans

SCOPE_NESTING = '''using TechTalk.SpecFlow;
namespace X.Steps
{
    [Binding]
    public class AboveSteps
    {
        [Given(@"x")]
        [Scope(Tag = "slow")]
        public void X()
        {
        }
    }

    [Binding, Scope(Tag = "a")]
    public class NestedSteps
    {
        [Scope(Tag = "b")]
        [Given(@"y")]
        public void Y()
        {
        }
    }
}
'''

def test_active_in_scope_uses_body_start_and_ors_nested_spans():
    spans = specflow_scope_tags("Steps.cs", SCOPE_NESTING)
    defs = {d.pattern: d for d in extract("Steps.cs", SCOPE_NESTING, "csharp_attribute")}
    x, y = defs["x"], defs["y"]
    # x's own [Given] line sits ABOVE its [Scope(Tag="slow")]: only body_start (inside the method), not the
    # attribute's own line, correctly places it inside the (8, 11, {"@slow"}) span.
    assert active_in_scope(x, spans, {"@slow"})
    assert not active_in_scope(x, spans, {"@a"})
    assert not active_in_scope(x, spans, set())
    # y is inside both a class-level Scope("a") and a method-level Scope("b") span: active on EITHER tag
    # (SpecFlow creates one binding per [Scope] attribute -- OR across nested spans, not AND).
    assert active_in_scope(y, spans, {"@a"})
    assert active_in_scope(y, spans, {"@b"})
    assert not active_in_scope(y, spans, set())
    # scenario tags come from the .feature file verbatim (any case); span tags are already lowercased --
    # the comparison must be case-insensitive regardless of the caller.
    assert active_in_scope(x, spans, ["@SLOW"])

def test_pick_wrong_suite():
    assert pick_wrong_suite(["b"], ["a", "b", "c"]) == "c"
    assert pick_wrong_suite(["c"], ["a", "b", "c"]) == "a"
    assert pick_wrong_suite(["a", "b", "c"], ["a", "b", "c"]) is None
    assert pick_wrong_suite([], ["a"]) is None
