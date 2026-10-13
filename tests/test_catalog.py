import os
import subprocess
from datetime import datetime
from pathlib import Path

from v2tools.catalog import build_catalog, line_keys, pattern_key, render

def ts(day):
    return int(datetime.fromisoformat(day + "T00:00:00+00:00").timestamp())

def commit(repo, date):
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", date], cwd=repo, check=True, env=env)

OLD = "<?php\nclass BrowseContext {\n    /**\n     * @When I want to browse\n     */\n    public function browse(): void\n    {\n    }\n}\n"
NEW = ("<?php\nclass BrowseContext {\n    #[When('I am browsing')]\n    #[When('I want to browse')]\n    public function browse(): void\n"
       "    {\n    }\n    #[Given('/^.*$/')]\n    public function anything(): void\n    {\n    }\n"
       "    #[Then('I see it')]\n    public function see(): void\n    {\n    }\n}\n")

def test_pattern_keys_agree_across_syntaxes():
    k = pattern_key("^I associate \"(.+)\"$")
    assert line_keys("* @Given /^I associate \"(.+)\"$/") >= {k}
    assert line_keys("#[When('/^I associate \"(.+)\"$/')]") >= {k}
    assert line_keys('[Given(@"I set ""(.*)"" to ""(.*)""")]') >= {pattern_key('I set "(.*)" to "(.*)"')}
    assert line_keys("Given /the following appointments:/ do |table|") >= {"the following appointments:"}
    assert line_keys("/** @When I want to browse */") >= {"I want to browse"}

def test_catalog_dates_each_pattern_even_across_a_syntax_migration(tmp_path: Path):
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "BrowseContext.php").write_text(OLD)
    commit(repo, "2022-01-01T00:00:00Z")
    (repo / "src" / "BrowseContext.php").write_text(NEW)
    commit(repo, "2025-01-01T00:00:00Z")
    cat = build_catalog({"repo": "x/demo", "styles": ["php_attribute"], "step_globs": ["src/*.php"]}, repo)
    browse = cat.fns["src/BrowseContext.php:6"]
    assert [(a.line, a.pattern, a.time) for a in browse.annotations] == [
        (3, "I am browsing", ts("2025-01-01")), (4, "I want to browse", ts("2022-01-01"))]
    assert browse.time == ts("2022-01-01") and browse.suites == ("default",)
    snapshot = render(browse, ts("2024-01-01"))
    assert "I am browsing" not in snapshot and "I want to browse" in snapshot and "public function browse" in snapshot
    assert "I am browsing" in render(browse, None)
    assert cat.fns["src/BrowseContext.php:10"].catch_all
    assert cat.bind_steps(["I am browsing", "I see it", "nothing here"], "default", []) == [
        ["src/BrowseContext.php:3"], ["src/BrowseContext.php:12"], []]
    assert cat.fn_of("src/BrowseContext.php:4") == "src/BrowseContext.php:6"

DOCBLOCK = "<?php\nclass C {\n    /**\n     * Creates it.\n     * @Given a thing exists\n     */\n    public function thing()\n    {\n    }\n}\n"

def test_docblock_code_starts_at_the_docblock_and_is_dedented(tmp_path: Path):
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "C.php").write_text(DOCBLOCK)
    commit(repo, "2025-01-01T00:00:00Z")
    cat = build_catalog({"repo": "x/demo", "styles": ["php_docblock"], "step_globs": ["src/*.php"]}, repo)
    [fn] = cat.fns.values()
    assert fn.first_line == 3 and fn.code.splitlines()[:3] == ["/**", " * Creates it.", " * @Given a thing exists"]
    assert fn.code.splitlines()[-3:] == ["public function thing()", "{", "}"]
    assert render(fn, ts("2024-01-01")).splitlines() == ["/**", " * Creates it.", " */", "public function thing()", "{", "}"]
    assert not fn.dead and not fn.contains_fn   # a Behat docblock is live code, not a commented-out definition

JAVA = ('class S {\n    @Given("live one")\n    public void live() {\n    }\n'
        '    // @Given("commented out")\n    // public void dead() {\n    // }\n'
        '    /* @When("inside a block")\n    public void blocked() {\n    } */\n'
        '    @Then("a /* not a comment */ string")\n    public void str() {\n        String s = "/* still a string";\n    }\n}\n')

def test_dead_and_span_swallowing_definitions_are_flagged(tmp_path: Path):
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "S.java").write_text(JAVA)
    commit(repo, "2025-01-01T00:00:00Z")
    cat = build_catalog({"repo": "x/demo", "styles": ["java_annotation"], "step_globs": ["src/*.java"], "language": "java"}, repo)
    flags = {f: (fn.dead, fn.contains_fn) for f, fn in cat.fns.items()}
    # the `//` definition is dead and, its braces being commented out, its span (5..15) swallows the functions after it
    assert flags == {"src/S.java:3": (False, False), "src/S.java:6": (True, True), "src/S.java:9": (True, False),
                     "src/S.java:12": (False, False)}   # `/*` inside a string literal opens no comment

STACKED = ('class T {\n    @Then("history has {int} entry")\n    @Then("history has {int} entries")\n'
           '    public void size(final int n) {\n        check(n);\n    }\n}\n')

def test_a_stacked_annotation_split_off_its_function_is_flagged_dead(tmp_path: Path):
    # the Phase-1 extractor gives the first of two stacked Java annotations its own "function" whose only other line is
    # the second annotation: an entry without a body line, which snapshots must not show
    repo = tmp_path / "demo"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src" / "T.java").write_text(STACKED)
    commit(repo, "2025-01-01T00:00:00Z")
    cat = build_catalog({"repo": "x/demo", "styles": ["java_annotation"], "step_globs": ["src/*.java"], "language": "java"}, repo)
    assert {f: (fn.first_line, fn.body_end, fn.dead) for f, fn in cat.fns.items()} == {
        "src/T.java:3": (2, 3, True), "src/T.java:4": (3, 6, False)}
