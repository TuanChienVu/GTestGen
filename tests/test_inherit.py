import os
import subprocess

from v2tools.inherit import Resolver
from v2tools.outline import parse
from v2tools.symtime import first_times

COMPOSER = '{"autoload": {"psr-4": {"App\\\\": "src/"}}, "autoload-dev": {"psr-4": {"Tests\\\\": "tests/"}}}'
TREE = ["src/Behat/BaseContext.php", "tests/Ctx/AbstractCtx.php", "tests/Ctx/ShopCtx.php"]


def php(body: str) -> str:
    return "<?php\nnamespace Tests\\Ctx;\nuse App\\Behat\\BaseContext;\nuse Vendor\\Lib\\Thing;\n" + body


def test_php_parents_resolve_through_autoload_and_autoload_dev():
    r = Resolver("php", TREE, COMPOSER)
    o = parse("php", php("class ShopCtx extends AbstractCtx implements Thing { use BaseContext; }"))
    got = {(p.relation, p.status, p.path) for p in (r.resolve(o, rel, w) for rel, w in o.parents)}
    assert got == {("extends", "resolved", "tests/Ctx/AbstractCtx.php"), ("implements", "external", None),
                   ("trait", "resolved", "src/Behat/BaseContext.php")}


def test_php_parent_in_a_repo_namespace_without_a_file_is_unresolved():
    r = Resolver("php", TREE, COMPOSER)
    o = parse("php", php("class ShopCtx extends MissingCtx {}"))
    assert r.resolve(o, "extends", "MissingCtx").status == "unresolved"
    only_autoload = Resolver("php", TREE, '{"autoload": {"psr-4": {"App\\\\": "src/"}}}')
    assert only_autoload.resolve(o, "extends", "MissingCtx").status == "external"  # why autoload-dev must be read


def test_java_parents_resolve_by_import_or_package():
    tree = ["mod/src/test/java/org/x/stepdef/AbstractStepDef.java", "mod/src/test/java/org/x/stepdef/loan/LoanStepDef.java",
            "mod/src/test/java/org/x/util/Helper.java"]
    r = Resolver("java", tree)
    o = parse("java", "package org.x.stepdef.loan;\nimport org.x.stepdef.AbstractStepDef;\nimport org.x.util.Gone;\n"
                      "import io.cucumber.java8.En;\npublic class LoanStepDef extends AbstractStepDef implements En {}\n")
    assert r.resolve(o, "extends", "AbstractStepDef").path == "mod/src/test/java/org/x/stepdef/AbstractStepDef.java"
    assert r.resolve(o, "implements", "En").status == "external"
    assert r.resolve(o, "extends", "Gone").status == "unresolved"  # imported from a repo package, file missing


def test_first_times_follow_a_rename(tmp_path):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*a, date=None):
        e = dict(env, **({"GIT_AUTHOR_DATE": f"@{date} +0000", "GIT_COMMITTER_DATE": f"@{date} +0000"} if date else {}))
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True, env=e).stdout
    git("init", "-q", "-b", "main")
    (tmp_path / "A.php").write_text("<?php\nclass A {\n    public function a(): void {}\n}\n")
    git("add", "-A"); git("commit", "-q", "-m", "1", date=1000)
    git("mv", "A.php", "B.php")
    (tmp_path / "B.php").write_text("<?php\nclass A {\n    public function a(): void {}\n    public function b(): void {}\n}\n")
    git("add", "-A"); git("commit", "-q", "-m", "2", date=2000)
    t = first_times(tmp_path, git("rev-parse", "HEAD").strip(), "B.php", "php")
    assert (t["file"], t["names"], t["params"], t["present"]) == (1000, {"a": 1000, "b": 2000}, {}, {})
    assert [(at, p) for _, at, p in t["versions"]] == [(2000, "B.php"), (1000, "A.php")]  # newest first, old path kept
