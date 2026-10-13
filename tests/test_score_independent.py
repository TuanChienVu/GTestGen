import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, namedtuple
from pathlib import Path

import pytest

from tests.phase3_fixture import SHOP, SHOP_2024, T2020, T2024, _git, catalog, record
from tests.phase3c_fixture import block, gen_plan, make_phase2, make_prep
from v2tools.controls import controls
from v2tools.describe import rouge_l, row
from v2tools.evaluate import CONDITIONS, Repo, ca, prepare, score_output
from v2tools.extract import extract
from v2tools.state import RepoContext

ROOT = Path(__file__).resolve().parents[1]
TICKS = "`" * 3
spec = importlib.util.spec_from_file_location("score_independent", ROOT / "scripts" / "score_independent.py")
ind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ind)

OUTPUTS = [
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->assertTotal($total);\n}",
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->ghost->x($total);\n}",
    "#[Then('I should see the total \"5\"')]\npublic function t(): void\n{\n    $this->checkout->assertTotal('5');\n}",
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->cart->add($total);\n}\n\n#[Given('I open the shop')]\npublic function again(): void\n{\n}",
    "#[Then('I should see the total :total')]\npublic function iAddToCart(string $total): void\n{\n}",
    "public function broken( {",
]
GOLD = "#[Then('I should see the total :total')]\npublic function iShouldSeeTotal(string $total): void\n{\n    $this->checkout->assertTotal($this->computeTotal($total));\n}"
HEAD = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->cart->add($total);\n"
# the CA gaming probes (focused.py, probe_ca.py) and the 9 output shapes of make_unusual.py, on the fixture
PROBES = [
    HEAD + "    // $this->checkout->assertTotal($total);\n}",
    HEAD + "    /* $this->checkout->assertTotal($total); */\n}",
    HEAD + "    $s = \"{$this->checkout->assertTotal($total)}\";\n}",
    HEAD + "    $h = <<<EOT\n    $this->checkout\n    EOT;\n}",
    HEAD + "    return;\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "    throw new \\RuntimeException('x');\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "    if ($total) { return; }\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "}\n\nprivate function neverCalled(): void\n{\n    $this->checkout->assertTotal('1');\n}",
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->logger->info($total);\n}",
    GOLD + "\n\n" + GOLD,  # U3: the same method twice
    "<?php\n\nclass Wrapper\n{\n" + GOLD + "\n}",  # U1: the whole class wrapped again
    GOLD + "\n\n#[Given('I open the shop')]\npublic function probeReuse(): void\n{\n}",  # U4: re-implements a REUSE step
    GOLD.replace("iShouldSeeTotal", "probeRenamed1"),  # U11: renamed
    GOLD.split("{")[0] + "{\n}",  # U12: emptied body
    HEAD + "    return;\n    if ($total) {\n        $a = 1;\n    } else {\n        $this->checkout->assertTotal($total);\n    }\n}",
    HEAD + "    $x = $total ?: throw new \\RuntimeException('x');\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "    switch ($total) {\n        case 1:\n            $a = 1;\n            return;\n        default:\n            $this->checkout->assertTotal($total);\n    }\n}",
    HEAD + "    if ($total):\n        $a = 1;\n        return;\n    else:\n        $this->checkout->assertTotal($total);\n    endif;\n}",
    # X7 (`exit;` before the gold lines), `die(…);`, an exit inside an expression (not a flow end), and a property
    # whose initializer uses an unknown member (the PHP form of X5)
    HEAD + "    exit;\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "    @exit;\n    $this->checkout->assertTotal($total);\n}",  # a process exit under `@` (error suppression)
    HEAD + "    die('stop');\n    $this->checkout->assertTotal($total);\n}",
    HEAD + "    $x = $total ?: exit(1);\n    $this->checkout->assertTotal($total);\n}",
    "private $probeF = self::GHOST_Z;\n\n" + GOLD,
    "private $probeX = new class {\n" + GOLD + "\n};",  # X10: a step method in an anonymous class never binds
]

# helper-call expansion on the PHP fixture: (output, BV, CA, n_calls, expansion cuts (depth, cycle)). computeTotal, the
# private helper written with the answer at t_vis, is a future method of the ER tree: the gold's (self, computeTotal)
# opens into its body, which makes no project call, so the gold's calls are {(checkout, assertTotal)}. iAddToCart
# (2022) is visible.
HEAD2 = "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n"


def php_fn(name: str, lines: list[str], params: str = "string $total", ret: str = "void") -> str:
    return f"private function {name}({params}): {ret}\n{{\n" + "".join(f"    {l}\n" for l in lines) + "}"


def php_step(lines: list[str]) -> str:
    return HEAD2 + "".join(f"    {l}\n" for l in lines) + "}"


GPROBES = [
    (GOLD, 1, 1, 1, 0),
    (GOLD + "\n\n" + php_fn("computeTotal", ["return $total;"], ret="string"), 1, 1, 1, 0),  # gold-helpers (override)
    (GOLD.replace("computeTotal", "probeHelper1") + "\n\n" + php_fn("probeHelper1", ["return $total;"], ret="string"),
     1, 1, 1, 0),  # gold-helpers-renamed
    (php_step(["$this->computeTotal($total);"]) + "\n\n" + php_fn("computeTotal", ["$this->cart->add($total);", "return $total;"],
                                                                   ret="string"), 1, 0, 1, 0),  # a future name, another body
    (php_step(["$this->computeTotal($total);"]) + "\n\n" + php_fn("computeTotal", ["$this->checkout->assertTotal($total);",
                                                                                   "return $total;"], ret="string"), 1, 1, 1, 0),
    (php_step(["$this->probeHelper($total);"]) + "\n\n" + php_fn("probeHelper", ["$this->cart->add($total);"]), 1, 0, 1, 0),  # v3m-h
    (php_step(["$this->cart->add($total);"]), 1, 0, 1, 0),  # v3m
    (php_step(["$this->probeHelper($total);"]) + "\n\n" + php_fn("probeHelper", ["$this->indexPage->open();", "$this->cart->add($total);"]),
     1, 0, 2, 0),  # v3all-h
    (php_step(["$this->indexPage->open();", "$this->cart->add($total);"]), 1, 0, 2, 0),  # v3all
    (php_step(["$this->probeA($total);"]) + "\n\n" + php_fn("probeA", ["$this->checkout->assertTotal($total);", "$this->probeB($total);"])
     + "\n\n" + php_fn("probeB", ["$this->probeA($total);"]), 1, 1, 2, (0, 1)),  # a cycle: (self, probeA) kept
    (php_step(["$this->probeH1($total);"]) + "".join("\n\n" + php_fn(f"probeH{k}", (["$this->indexPage->open();"] if k == 1 else [])
                                                                     + [f"$this->probeH{k + 1}($total);"]) for k in (1, 2, 3))
     + "\n\n" + php_fn("probeH4", ["$this->checkout->assertTotal($total);"]), 1, 0, 2, (1, 0)),  # a chain of 4: (self, probeH4) kept
    (php_step(["$this->iAddToCart($total);", "$this->checkout->assertTotal($total);"]), 1, 1, 2, 0),  # visible: not expanded
    (php_step(["$this->probeHelper();"]) + "\n\n" + php_fn("probeHelper", ["$x = 1;"], params=""), 0, 0, 0, 0),  # a name never grounds
    (php_step(["$this->probeHelper();"]) + "\n\n" + php_fn("probeHelper", ["$this->cart->add('x');"], params=""), 1, 0, 1, 0),
    (php_step(["$this->secret();", "$this->checkout->assertTotal($total);"]), 0, 0, 2, 0),  # BaseContext's private
    # secret() is not future, so (self, secret) stays (and condition 6 fails: a private member is not inherited)
    (php_step(["$this->cart->add($total);", "$this->helperOld();"]) + "\n\nprotected function helperOld(): void\n{\n"
     "    $this->checkout->assertTotal('5');\n}", 1, 0, 2, 0),  # the output rewrites the visible helperOld: kept
    # condition 5: an output method that replaces a future method of the same key grounds by its name
    # like the member it replaces; a new name does not (the probeHelper with `$x = 1;` above); calling the hidden future
    # helper without defining it grounds by its name too
    (php_step(["$this->computeTotal($total);"]) + "\n\n" + php_fn("computeTotal", ["return $total;"], ret="string"), 1, 0, 0, 0),
    (php_step(["$this->computeTotal($total);"]), 1, 0, 0, 0),
]
PROBES += [g[0] for g in GPROBES]


GHOSTS = "".join(f"    $this->ghost{i}->x($total);\n" for i in range(12))
# the descriptive columns: open paths, the unresolved cap of 10, REUSE re-bound, conventions broken, a C# params array
EXTRA = [
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->ghost->x($total);\n}\n\n"
    "public function __call($n, $a)\n{\n}",  # a magic __call opens every member (open_resolution)
    "#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n" + GHOSTS + "}",
    GOLD + "\n\n#[Given('I open the shop')]\npublic function i_open_again(): void\n{\n    $this->cart->add('x');\n}",
    "/**\n * @Then /^I should see the total \"([^\"]*)\"$/\n */\npublic function i_see_total(string $t): void\n{\n"
    "    $this->checkout->assertTotal($t);\n    $this->cart->add($t);\n}",
    "private $probeState;\n\n" + GOLD.replace("{\n    $this", "{\n    $this->probeState = 1;\n    $this"),
    "use App\\Page\\IndexPage;\n\n" + GOLD,
    GOLD + "\n\npublic function iOpenTheShop(): void\n{\n}",  # a clash with a method of the file
    "public function broken( {\n\n#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n}",
]


def shapes() -> list[str]:
    """The text-level shapes of make_unusual.py: U2 (a block per function), U6 (unclosed), U7 (extra blocks), U8 (./)."""
    t = TICKS
    one = f"{t}php\n// FILE: {SHOP}\n{GOLD}\n{t}"
    return [one + "\n\nSome prose.\n\n" + f"{t}php\n// FILE: {SHOP}\n{GOLD.replace('iShouldSeeTotal', 'other')}\n{t}",
            one[:-3].rstrip() + "\n",
            one + f"\n\nRun:\n{t}bash\nvendor/bin/behat\n{t}\n{t}php\n// FILE: src/Other/NotATarget.php\nfoo();\n{t}",
            one.replace("// FILE: ", "// FILE: ./"),
            # U13, a second FILE line in the same fence (multi_file_fence); U14,
            # a four-backtick fence keeps a three-backtick line (in a comment) as code; the same in a three-backtick
            # fence closes early and is detected (fence_truncated)
            one[:-3] + f"// FILE: {SHOP}\nprivate function probeOther(): void\n{{\n}}\n{t}",
            f"{t}`php\n// FILE: {SHOP}\n{GOLD}\n\n/*\n{t}\n*/\n{t}`",
            f"{t}php\n// FILE: {SHOP}\n{GOLD}\n\n/*\n{t}\n*/\n{t}"]


@pytest.fixture()
def pair(tmp_path):
    p = make_prep(tmp_path)
    scorer = ind.Scorer("Acme/Shop", "php", ["php_attribute", "php_docblock"], catalog(),
                        ind.Git(p.repo.ctx.git_dir, p.record["commit"]), p.repo.suite_steps)
    return p, scorer


def compare(p, scorer, text, labels_row: dict | None = None) -> dict:
    """BV, CA, the six conditions and the static columns of the score row (describe.row) agree; `labels_row` = the
    record's labels (default: the fixture record's two steps, the first two of the suite)."""
    theirs = score_output(p, text)
    labels_row = labels_row or {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    mine = scorer.score(p.record, labels_row, text)
    d, v = scorer.ca(p.record, labels_row, mine)
    mine.update(scorer.overlap(p.record, labels_row, mine))
    assert (mine["bv"], d, v) == (theirs.bv, *ca(p, theirs))
    assert {c: mine[f"bv_{c}"] for c in CONDITIONS} == {c: int(theirs.conds[c]) for c in CONDITIONS}
    ruby = p.record["language"] == "ruby"  # D11: ca_ref.jsonl also has ca_ref_v3names and ca_ref_top4, None for Ruby
    refs_ = {"ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": None if ruby else 0, "ca_ref_top4": None if ruby else 0}
    want = row(p, {"text": text}, theirs, extract(text, p.targets).by_path(), refs_,
               {"excl_bv": False, "excl_ca": False}, {"post_cutoff_all": [], "time_coherent_30d": [], "residual_clean": []})
    assert {k: mine[k] for k in ind.ROW} == {k: want[k] for k in ind.ROW}, text
    assert mine["cuts"] == dict(theirs.cuts), text  # the expansion cuts (depth, cycle) of the step-binding functions
    return mine


# A Java twin of the PHP fixture, with a class chain: ShopSteps extends BaseSteps, which holds the
# field `cart` of the in-repo class Cart. iOpenTheShop (2020, REUSE); the answer iShouldSeeTotal (2024, CREATE) calls the
# private helper computeTotal, written with it, which calls baseNew, added to BaseSteps with it: both are future methods
# of the ER tree, and so is the constructor added with them (a constructor never expands); BaseSteps.baseOld (2020) is
# visible. baseNew calls a field private to BaseSteps: its body is read with the Scope of BaseSteps (1b287d0).
JPATH = "src/test/java/acme/steps/ShopSteps.java"
JBASE = "src/test/java/acme/steps/BaseSteps.java"
CART = ("package acme;\n\npublic class Cart {\n    public void open() {\n    }\n\n    public void assertTotal(String total) {\n    }\n\n"
        "    public String total() {\n        return \"\";\n    }\n}\n")
BASE_2020 = """package acme.steps;

import acme.Cart;

public abstract class BaseSteps {

    protected final Cart cart = new Cart();

    protected void baseOld() {
        cart.open();
    }
}
"""
BASE_2024 = BASE_2020.replace("""    protected final Cart cart = new Cart();
""", """    protected final Cart cart = new Cart();

    private final Cart store = new Cart();
""").replace("""        cart.open();
    }
""", """        cart.open();
    }

    protected void baseNew() {
        store.total();
    }
""")
JAVA_2020 = """package acme.steps;

import io.cucumber.java.en.Given;
import io.cucumber.java.en.Then;

public class ShopSteps extends BaseSteps {

    @Given("I open the shop")
    public void iOpenTheShop() {
        cart.open();
    }
}
"""
JAVA_2024 = JAVA_2020.replace("""        cart.open();
    }
""", """        cart.open();
    }

    public ShopSteps() {
        super();
    }

    @Then("I should see the total {string}")
    public void iShouldSeeTotal(String total) {
        cart.assertTotal(computeTotal(total));
    }

    private String computeTotal(String total) {
        baseNew();
        return total;
    }
""")
JGOLD = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(computeTotal(total));\n}'
JHEAD = '@Then("I should see the total {string}")\npublic void t(String total) {\n    cart.open();\n'
LOGGER_FIELD = "private final org.slf4j.Logger probeLog = org.slf4j.LoggerFactory.getLogger(getClass());\n\n"
# (output, expected independent BV, CA): shapes that hide, wrap or cut the gold on the Java fixture
JPROBES = [
    (JGOLD, 1, 1),
    (LOGGER_FIELD + JGOLD, 1, 1),  # X6: getClass() of java.lang.Object resolves in an initializer
    ("private Object probeF = ghostFieldZ.make();\n\n" + JGOLD, 0, 0),  # X5: condition 6 fails
    ("private Object probeX = new Object() {\n" + JGOLD + "\n};", 0, 0),  # X3: the gold method in an anonymous class
    ("private TypeReference<List<String>> probeT = new TypeReference<List<String>>() {};\n\n" + JGOLD, 1, 1),  # no method
    ("private Object probeT = new Object() {\n    int n = 1;\n};\n\n" + JGOLD, 1, 1),  # anonymous class, fields only
    ("private Object probeT = new Object() {\n    Object o = new Object() {\n        public String toString() {\n"
     "            return \"\";\n        }\n    };\n};\n\n" + JGOLD, 0, 0),  # a method in a nested anonymous class
    ("private Object probeX = new Object() {\n    class In {\n" + JGOLD + "\n    }\n};", 0, 0),  # X9: a step
    # method in a class nested in an anonymous class stays a member but never binds
    (JHEAD + "    System.exit(0);\n    cart.assertTotal(total);\n}", 1, 0),  # X8: System.exit first
    (JGOLD.replace("(total));\n", "(total));\n    getClass();\n    hashCode();\n    this.toString();\n"), 1, 1),  # Object calls
    (FO_ONLY := JGOLD.replace("cart.assertTotal(computeTotal(total));", "getClass().getName();"), 0, 0),  # never grounds
]
# helper-call expansion on the Java fixture: (output, BV, CA, n_calls); the gold's calls after expansion are
# (cart, assertTotal) and (store, total), through computeTotal (future, own class) and baseNew (future, BaseSteps, read
# in its own Scope)
JCOMPUTE = "private String computeTotal(String total) {\n    baseNew();\n    return total;\n}"
JGPROBES = [
    (JGOLD, 1, 1, 2),
    (JGOLD + "\n\n" + JCOMPUTE, 1, 1, 2),  # the output's computeTotal replaces the future one, no collision
    (JGOLD + "\n\nprotected void baseNew() {\n    cart.open();\n}", 1, 1, 2),  # an ancestor's future method: an override,
    # and the output's body wins in the expansion of the ER tree's computeTotal: (cart, open) instead of (cart, total)
    ('@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(probeHelper1(total));\n}'
     "\n\nprivate String probeHelper1(String total) {\n    probeHelper2();\n    return total;\n}\n\nprivate void probeHelper2() {\n"
     "    cart.total();\n}", 1, 1, 2),  # gold-helpers-renamed
    (JHEAD.replace("cart.open();", "baseOld();") + "    cart.assertTotal(total);\n}", 1, 1, 2),  # visible: (self, baseOld) kept
    (JGOLD + "\n\npublic ShopSteps() {\n    cart.open();\n}", 1, 1, 2),  # replaces the future constructor, no collision
    (JHEAD.replace("cart.open();", "computeTotal(total);") + "}\n\nprivate String computeTotal(String total) {\n    return total;\n}",
     1, 0, 0),  # the replacement of the future computeTotal grounds by its name
    (JHEAD + "    baseOld();\n}\n\nprotected void baseOld() {\n    cart.assertTotal(\"5\");\n}", 1, 0, 2),  # the output
    # rewrites the visible baseOld, which stays (self, baseOld): {(cart, open), (self, baseOld)}, CA 0
]
JPROBES += [(c, b, a) for c, b, a, _ in JGPROBES]


def jblock(code: str) -> str:
    return f"{TICKS}java\n// FILE: {JPATH}\n{code}\n{TICKS}"


def step_fn(repo: str, path: str, text: str, keyword: str, pattern: str, time: int, body_line: int) -> dict:
    """A catalog row for the step method annotated with `pattern` in `text` (fn_id: the line of its body's brace)."""
    lines = text.splitlines()
    ann = next(i for i, l in enumerate(lines, 1) if f"('{pattern}')" in l or f'("{pattern}")' in l)
    end = next(i for i in range(ann, len(lines) + 1) if lines[i - 1] == "    }")
    return {"repo": repo, "fn_id": f"{path}:{ann + body_line}", "path": path, "first_line": ann, "body_end": end,
            "code": "\n".join(l[4:] for l in lines[ann - 1:end]),
            "annotations": [{"def_id": f"{path}:{ann}", "line": ann, "keyword": keyword, "pattern": pattern, "time": time}],
            "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}


def build_pair(tmp_path, repo: str, lang: str, styles: list[str], commits: list, path: str, open_pattern: str, total_pattern: str,
               body_line: int):
    """(Prepared, independent Scorer) of a twin of the PHP fixture: `commits` = [(time, {path: text})], the REUSE step
    "I open the shop" (2020) and the CREATE answer "I should see the total ..." (2024) in `path`."""
    work = tmp_path / f"{lang}work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in commits:
        for fp, text in files.items():
            (work / fp).parent.mkdir(parents=True, exist_ok=True)
            (work / fp).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp_path / f"{lang}.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    head = _git(bare, "rev-parse", "HEAD").strip()
    text = next(files[path] for _, files in reversed(commits) if path in files)
    opn = step_fn(repo, path, text, "Given", open_pattern, T2020, body_line)
    tot = step_fn(repo, path, text, "Then", total_pattern, T2024, body_line)
    cat = {f["fn_id"]: f for f in (opn, tot)}
    r = record(cat)
    r.update(id=f"{repo}|features/total.feature|3|s", repo=repo, commit=head, language=lang, step_fns=[opn["fn_id"], tot["fn_id"]],
             create_def_ids=[tot["annotations"][0]["def_id"]],
             create=[{"fn_id": tot["fn_id"], "path": path, "code": tot["code"], "preexisting": False, "post_cutoff_code": True}])
    d_open, d_tot = opn["annotations"][0]["def_id"], tot["annotations"][0]["def_id"]
    suite = {"s": [("I open the shop", (d_open,)), ('I should see the total "5"', (d_tot,))]}
    rp = Repo(repo, lang, styles, cat, RepoContext(repo, lang, bare, head, cat), frozenset(), suite)
    return prepare(r, rp, {"step_def_ids": [[d_open], [d_tot]]}), ind.Scorer(repo, lang, styles, cat, ind.Git(bare, head), suite)


@pytest.fixture()
def jpair(tmp_path):
    return build_pair(tmp_path, "Acme/Java", "java", ["java_annotation"],
                      [(T2020, {"src/main/java/acme/Cart.java": CART, JBASE: BASE_2020, JPATH: JAVA_2020}),
                       (T2024, {JBASE: BASE_2024, JPATH: JAVA_2024})], JPATH, "I open the shop", "I should see the total {string}", 1)


# A second PHP twin (1b287d0): the parent context imports App\Util\Language and gets, with the answer, a
# helper langId() calling Language::getIdByLocale(); the answer calls $this->langId(). The child never imports Language:
# the helper's body resolves the class with the parent file's imports.
PPATH, PBASE = "src/Behat/ShopContext.php", "src/Behat/BaseContext.php"
PLANG = "<?php\n\nnamespace App\\Util;\n\nfinal class Language\n{\n    public static function getIdByLocale(string $l): int\n    {\n        return 1;\n    }\n}\n"
PBASE_2020 = """<?php

namespace App\\Behat;

use App\\Util\\Language;

abstract class BaseContext
{
    protected function baseOld(): void
    {
    }
}
"""
PBASE_2024 = PBASE_2020.replace("""    protected function baseOld(): void
    {
    }
""", """    protected function baseOld(): void
    {
    }

    protected function langId(string $locale): int
    {
        return Language::getIdByLocale($locale);
    }
""")
PSHOP_2020 = """<?php

namespace App\\Behat;

use Behat\\Step\\Given;
use Behat\\Step\\Then;

final class ShopContext extends BaseContext
{
    #[Given('I open the shop')]
    public function iOpenTheShop(): void
    {
        $this->baseOld();
    }
}
"""
PSHOP_2024 = PSHOP_2020.replace("""        $this->baseOld();
    }
""", """        $this->baseOld();
    }

    #[Then('I should see the total :total')]
    public function iShouldSeeTotal(string $total): void
    {
        $this->langId($total);
    }
""")
PGOLD = "#[Then('I should see the total :total')]\npublic function iShouldSeeTotal(string $total): void\n{\n    $this->langId($total);\n}"
# (output, BV, CA, n_calls): the gold's calls are {(Language, getIdByLocale)}, found only through the parent's imports
IPROBES = [
    (PGOLD, 1, 1, 1),
    (PGOLD + "\n\nprotected function langId(string $locale): int\n{\n    return Language::getIdByLocale($locale);\n}", 1, 0, 0),  # the
    # output's copy of the helper is read with the child's imports, where Language is not imported: no call; its name,
    # the key of the parent's future langId, grounds like the member it overrides
    ("use App\\Util\\Language;\n\n" + PGOLD + "\n\nprotected function langId(string $locale): int\n{\n    return Language::getIdByLocale($locale);\n}",
     1, 1, 1),  # the same copy with the import the output adds
]


@pytest.fixture()
def ppair(tmp_path):
    composer = '{"autoload": {"psr-4": {"App\\\\": "src/"}}}\n'
    return build_pair(tmp_path, "Acme/Imports", "php", ["php_attribute", "php_docblock"],
                      [(T2020, {"composer.json": composer, "src/Util/Language.php": PLANG, PBASE: PBASE_2020, PPATH: PSHOP_2020}),
                       (T2024, {PBASE: PBASE_2024, PPATH: PSHOP_2024})], PPATH, "I open the shop", "I should see the total :total", 2)


def pblock(code: str) -> str:
    return f"{TICKS}php\n// FILE: {PPATH}\n{code}\n{TICKS}"


def test_an_ancestor_helper_is_read_with_its_own_file(ppair, jpair):
    """A future helper of an ancestor is read with the ancestor file's outline and Scope (1b287d0): its
    imports (PHP: Language::getIdByLocale imported by the parent only) and its private members (Java: BaseSteps.store);
    an output-defined helper with the target file's. Both scorers agree."""
    p, scorer = ppair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    g = scorer.gold(p.record, labels_row)
    assert set().union(*g["calls"].values()) == {("Language", "getIdByLocale")}
    for code, bv, ca_, n in IPROBES:
        mine = compare(p, scorer, pblock(code))
        assert (mine["bv"], scorer.ca(p.record, labels_row, mine)[1], scorer.overlap(p.record, labels_row, mine)["n_calls"]) == (bv, ca_, n), code
    p, scorer = jpair
    g = scorer.gold(p.record, {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]})
    assert ("store", "total") in set().union(*g["calls"].values())


def test_independent_scorer_agrees_on_outputs(pair):
    p, scorer = pair
    for code in OUTPUTS:
        compare(p, scorer, block(code))
    compare(p, scorer, "no code")


def test_independent_scorer_agrees_on_probes_and_shapes(pair):
    p, scorer = pair
    for code in PROBES:
        compare(p, scorer, block(code))
    for text in shapes():
        compare(p, scorer, text)
    assert score_output(p, block(PROBES[0])).bv == 1 and ca(p, score_output(p, block(PROBES[0]))) == (True, 0)
    assert score_output(p, block(PROBES[9])).insert_status == "collision"


def test_independent_scorer_agrees_on_the_descriptive_columns(pair):
    p, scorer = pair
    got = [compare(p, scorer, block(code)) for code in EXTRA]
    assert (got[0]["open_resolution"], got[0]["unresolved"], got[1]["unresolved"], got[1]["bv_resolved"]) == (True, 0, 10, 0)
    assert got[2]["duplicate_reuse"] and not got[0]["duplicate_reuse"]
    assert (got[3]["conv_style"], got[3]["conv_pattern"], got[3]["conv_naming"], got[3]["n_calls"], got[3]["call_precision"]) == \
        (0.0, 0.0, 0.0, 2, 0.5)
    # conventions() reads the output's own blocks (own_items; fixed 26/09/2026), available even on a collision (the
    # clash is against the OLD file, not within the output): iShouldSeeTotal alone gives conv_style/pattern/naming 1.0
    assert (got[6]["insert_status"], got[6]["conv_style"], got[7]["insert_status"], got[7]["bv_syntax"]) \
        == ("collision", 1.0, "ok", 0)
    outs = [compare(p, scorer, block(c)) for c in OUTPUTS] + [compare(p, scorer, t) for t in shapes()]
    assert {x["extract_status"] for x in outs} == {"ok", "unclosed", "malformed", "multi_file_fence"} and any(x["exact"] for x in outs)
    assert [x["fence_truncated"] for x in outs].count(True) == 1  # the three-backtick fence closed early
    assert compare(p, scorer, "no code")["extract_status"] == "none"


def test_blocks_of_follows_the_extractor():
    """The independent block reader gives the extractor's status, code per target and truncation flag: lines split on
    \\n only (CRLF trimmed; U+2028 and friends stay in the code), any info string without a backtick opens a fence, a
    line of at least as many backticks closes it, every FILE line starts a file (text before the first one is a block
    without a path), and a bare fence without a FILE line right after a FILE fence is a detected early close."""
    t = TICKS
    cases = [f"{t}php\r\n// FILE: a.php\r\nf();\u2028g();\r\n{t}\r\n", f"{t} php\n// FILE: a.php\nf();\n{t}",
             f"{t}php:src/a.php\n// FILE: a.php\nf();\n{t}", f"{t}php {{1,3}}\n// FILE: a.php\nf();\n{t}", f"{t}\n// FILE: a.php\nf();\n{t}",
             f"{t}`php\n// FILE: a.php\n{t}\nf();\n{t}`", f"{t}php\n// FILE: a.php\nf();\n{t}``  ", f"{t}php\n// FILE: a.php\nf();\n  {t}  \nx",
             f"{t}php\n// FILE: a.php\nf();\n// FILE: b.php\ng();\n{t}", f"{t}php\nnote\n// FILE: a.php\nf();\n{t}",
             f"{t}php\n\n// FILE: a.php\nf();\n\n   // FILE: ./b.php\n\n{t}", f"{t}php\n// FILE: a.php\n/*\n{t}\n*/\n{t}",
             f"{t}php\n// FILE: a.php\nf();\n{t}\n{t}\ng();\n", f"{t}php\n// FILE: a.php\nf();", "no fence", f"{t}php\n{t}",
             f"{t}php\n// FILE: c.php\nf();\n{t}", f"{t}ruby\n# FILE: a.php\nf\n{t}\ntext {t}php",
             # the block-reading rule of design specification §1.4 (a FILE line opens a segment): the open tag before
             # the FILE line, a fence reopened right before a FILE line, a closed shell fence, a reopened and unclosed
             # rest, prose then a FILE fence, a bare FILE fence
             f"{t}php\n<?php\n// FILE: a.php\nf();\n{t}", f"{t}php\n// FILE: a.php\nf();\n\n{t}php\n// FILE: b.php\ng();\n{t}",
             f"{t}php\n// FILE: a.php\nf();\n{t}\nRun:\n{t}\nvendor/bin/behat\n{t}\n", f"{t}php\n// FILE: a.php\n/*\n{t}\n*/\n{t}\nx",
             f"{t}php\n// FILE: a.php\n/*\n{t}\n*/\n{t}\nB:\n{t}php\n// FILE: b.php\ng();\n{t}",
             f"{t}php\n// FILE: a.php\nf();\n{t}\n{t}\n// FILE: b.php\ng();\n{t}",
             # the open-tag case takes a line of the open tag ALONE, not `<?php echo 1;`; a fence-open line is dropped
             # only RIGHT before a FILE line, not with a blank line between
             f"{t}php\n<?php echo 1;\n// FILE: a.php\nf();\n{t}", f"{t}php\n// FILE: a.php\nf();\n{t}php\n\n// FILE: b.php\ng();\n{t}",
             # an unclosed non-bare fence after a FILE fence is no truncation; a
             # `<?php`-headed fence whose first block is off target; a bare opener right before FILE in a longer fence
             f"{t}php\n// FILE: a.php\nf();\n{t}\n\nRun:\n{t}bash\nvendor/bin/behat",
             f"{t}php\n<?php\n// FILE: c.php\nx();\n// FILE: a.php\nf();\n{t}", f"{t}`php\n// FILE: a.php\nf();\n{t}\n// FILE: b.php\ng();\n{t}`"]
    for text in cases:
        e = extract(text, ["a.php", "b.php"])
        assert ind.blocks_of(text, ["a.php", "b.php"]) == (e.status, e.by_path(), e.truncated), repr(text)


def test_truncated_output_is_scored(pair):
    """A step definition whose function or block never closes (an output cut at max_new_tokens) is one raw item:
    condition 1 fails and no step definition is judged for the conventions (describe.conventions and, for a Ruby `{`
    block, insert.split_code raise ValueError on such an output)."""
    p, scorer = pair
    for code in ("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n    $this->checkout->x(",
                 "#[Then('I should see the total :total')]\n"):
        theirs = score_output(p, block(code))
        mine = scorer.score(p.record, {"step_def_ids": [[i[0]] for _, i in p.repo.suite_steps["s"][:2]]}, block(code))
        assert (mine["insert_status"], mine["bv_syntax"]) == (theirs.insert_status, int(theirs.conds["syntax"])) == ("ok", 0)
        assert (mine["conv_style"], mine["conv_pattern"], mine["conv_naming"]) == (None, None, None)
    assert [i["kind"] for i in ind.items_of("ruby", "Given(/^a$/) do\n  x\nend\n\nWhen(/^b$/) {\n  y\n")] == ["raw"]


def test_items_and_annotations_follow_the_insertion_rule():
    from v2tools.insert import php_kept, split_code, with_annotations
    from v2tools.outline import parse
    cases = [("php", "use A\\B;\n\n/** doc */\n#[Given('x')]\npublic function a(int $x, ...$y) {}\n\nprivate $f;\n\n// c\nclass K {}"),
             ("php", "public function a( {"), ("csharp", "[Given(@\"x\")]\npublic void A(int a, params string[] b) { }\n\nprivate int _f;"),
             ("java", '  Given("a", (String x) -> {\n    f(x);\n  });\n  When("b", () -> {\n  // c\n  @Given("x")\n  public void m(int a /* c */) {}'),
             ("java", 'Given("a", () -> {\n});\n\n@Given("x")\npublic void m(int a, String... b) {}\n// c\nprivate int f;'),
             ("ruby", "Given(/^a$/) do\n  Given(/^b$/) do\n  end\nend\n\ndef helper(x)\n  x\nend"),
             # a C# struct (X4), and a Java field whose initializer holds an anonymous class (X3), are raw
             ("csharp", '[Given(@"x")]\npublic void A() { }\n\npublic struct Wrapper\n{\n    [Given(@"y")]\n    public void B() { }\n}'),
             ("java", 'private Object probeX = new Object() {\n    @Given("y")\n    public void b() {}\n};\n\n@Given("x")\npublic void a() {}'),
             # a java8 lambda registration is a lambda item only at the top level or in a constructor body
             ("java", 'static {\nGiven("a", () -> {\n    f();\n});\n}\n\npublic void holder() {\nWhen("b", () -> {\n});\n}'),
             ("java", '@Inject\npublic Steps(Foo foo) throws Exception {\nGiven("a", () -> {\n    f();\n});\n    if (x) {\nWhen("b", () -> {\n});\n    }\n}'),
             ("java", 'String s = "\nGiven(\\"a\\", () -> {\n});";\n/*\nGiven("b", () -> {\n});\n*/\nThen("c", (String x) -> {\n    g(x);\n});')]
    for lang, code in cases:
        mine = [(i["kind"], i["name"], i["arity"]) for i in ind.items_of(lang, code)]
        assert mine == [(i.kind, i.name, i.arity if lang != "ruby" else None) for i in split_code(lang, code)[1]] or lang == "ruby", code
        if lang == "ruby":
            assert [i["kind"] for i in ind.items_of(lang, code)] == [i.kind for i in split_code(lang, code)[1]]
        else:
            assert [i["text"] for i in ind.items_of(lang, code)] == [i.text for i in split_code(lang, code)[1]]
    # attribute lines kept before the method head (fixed 26/09/2026): main's with_old is php_kept (by pattern, not by
    # normalised line text) + with_annotations; ind.with_old mirrors the whole pipeline (php_pattern_kept + placement)
    old = "/**\n * @Given /^a$/\n * @When /^b$/\n */\n#[Then('c')]\n#[Given('d')]\npublic function f(): void\n{\n}"
    for new in ("public function f(): void\n{\n}", "/** @When /^b$/ */\npublic function f(): void\n{\n}",
                "/**\n * Doc.\n */\n#[Then('c')]\npublic function f(): void\n{\n}", "/***/\npublic function f(): void\n{\n}"):
        want_new, want_missing = php_kept(new, old)
        assert ind.with_old("php", new, old) == with_annotations("php", want_new, want_missing), new
    assert ind.method_names("php", SHOP_2024) == [m.names[0] for m in parse("php", SHOP_2024).members if m.kind == "method"]


def test_ruby_calls_and_constants_match_symbols():
    from v2tools.symbols import ruby_refs
    for code in ('Given(/^a (\\d+)$/) do |n|\n  visit root_path\n  expect(page).to have_content("#{n} x")\n  Renalware::Patient.find(n)\nend',
                 "def helper(x, y = 1)\n  x + y\n  go_to(x)\nend\n\nlambda { |z| z }\n-> (w) { w + q }",
                 "When(/^b$/) { fill_in 'x', with: Foo::Bar.new(user) }\n# call_me\nx = 3\nputs x",
                 # R1: what follows return/raise/fail up to the end of the block is dead
                 "Given(/^a$/) do\n  go_a(1)\n  return\n  go_b(1)\n  Renalware::Ghost.find(1)\nend",
                 "Given(/^a$/) do\n  go_a(1)\n  return if ready\n  go_b(1)\nend",
                 "When(/^b$/) do\n  if ready\n    fail 'no'\n    go_c(1)\n  else\n    go_d(1)\n  end\n  go_e(1)\nend",
                 "When(/^b$/) do |n|\n  return\n  [n].each do |i|\n    go_c(i)\n  end\n  go_d(n)\nend",
                 "When(/^b$/) do\n  begin\n    raise Boom\n    go_c(1)\n  rescue Boom\n    go_d(1)\n  ensure\n    go_e(1)\n  end\nend",
                 "When(/^b$/) {\n  return\n  go_c(1)\n}\ngo_d(1)",
                 "When(/^b$/) do |n|\n  case n\n  when 1\n    return\n    go_c(1)\n  when 2\n    go_d(1)\n  end\nend",
                 # the literal line rule: only comments, strings and heredocs are blanked, so an `end` in a regex, a
                 # symbol or a method name closes a block (a shared, documented limit)
                 "When(/^b$/) do\n  return\n  go_c(/ end x/, :end, range.end)\n  go_d(1)\nend\ngo_e(1)",
                 "When(/^b$/) do\n  fail = true\n  go_c(fail)\n  raise Foo.new(\n    go_d(1)\n  )\n  go_e(1)\nend"):
        _, calls, classes = ind.Scorer._ruby(code)
        assert (calls, classes) == (set(ruby_refs(code).calls), set(ruby_refs(code).classes)), code


def test_ruby_step_parameters_match_argcheck():
    """R2: condition 4 reads the |…| of the step's first line only, on both sides."""
    from v2tools.argcheck import params_of
    for code in ("Given(/^a$/) do\n  [1].each do |i|\n    go(i)\n  end\nend",
                 "Given(/^(\\d+) a$/) { |n, *rest|\n  [n].each { |i| go(i) }\n}",
                 "Given(/^(\\d+) a$/) do |n, m = 1|\n  n\nend"):
        u = ind.units("ruby", code)[1][0]
        assert (u["required"], u["total"]) == params_of("ruby", "block", u["text"]), code


def test_lcs_and_rouge_match_the_definition():
    rnd = random.Random(7)
    for _ in range(300):
        a = [rnd.choice("abcde") for _ in range(rnd.randrange(0, 40))]
        b = [rnd.choice("abcdef") for _ in range(rnd.randrange(0, 40))]
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
            prev = cur
        n = ind.lcs(a, b)
        assert n == prev[-1]
        f1 = 0.0 if n == 0 else 2 * (n / len(a)) * (n / len(b)) / (n / len(a) + n / len(b))
        assert round(f1, 4) == round(rouge_l(a, b), 4)
    assert (ind.majority(Counter("aab")), ind.majority(Counter("ab")), ind.majority(Counter())) == ("a", None, None)


def test_drop_dead_cuts_to_the_end_of_the_block_only():
    cases = [("php", "{ return; if ($a) { $x = 1; } else { $this->c->f(); } }", "{ return;}"),
             ("php", "{ switch ($x) { case 1: $a = 1; return; $d(); default: $this->c->f(); } }",
              "{ switch ($x) { case 1: $a = 1; return;default: $this->c->f(); } }"),
             ("php", "{ if ($x): return; $d(); else: $l(); endif; $a(); }", "{ if ($x): return;else: $l(); endif; $a(); }"),
             ("php", "{ $x = $t ?: throw new E(); $this->c->f(); }", "{ $x = $t ?: throw new E(); $this->c->f(); }"),
             ("php", "{ if ($a): f(); endif; return; if ($b) { x(); } else { y(); } }", "{ if ($a): f(); endif; return;}"),
             ("java", "{ lbl: return; repo.save(); }", "{ lbl: return;}"),
             ("csharp", "{ return; int d = default; _repo.Save(); }", "{ return;}"),
             ("java", "{ int v = a ? 1 : 2; repo.save(); }", "{ int v = a ? 1 : 2; repo.save(); }"),
             # a braced if/else, or a nested `:` block, inside the dead part of a PHP `:` block or case
             ("php", "{ if ($t): return; if ($a) { $x = 1; } else { $this->c->f(); } endif; }", "{ if ($t): return;endif; }"),
             ("php", "{ if ($t): return; if ($a) { x(); } elseif ($b) { $this->c->f(); } endif; }", "{ if ($t): return;endif; }"),
             ("php", "{ switch ($x) { case '1': return; switch ($a): case 2: $this->c->f(); endswitch; case 3: g(); } }",
              "{ switch ($x) { case '1': return;case 3: g(); } }"),
             ("php", "{ if ($t): return; if ($a): x(); else: $this->c->f(); endif; else: y(); endif; }",
              "{ if ($t): return;else: y(); endif; }"),
             ("php", "{ foreach ($l as $i): $x = $i ? 1 : 2; return; $d(); endforeach; $this->c->f(); }",
              "{ foreach ($l as $i): $x = $i ? 1 : 2; return;endforeach; $this->c->f(); }"),
             # the `;` of a `for (…; …; …):` head does not end a statement
             ("php", "{ if ($t): for ($i = 0; $i < 1; $i++): $a = 1; endfor; return; else: $this->c->f(); endif; }",
              "{ if ($t): for ($i = 0; $i < 1; $i++): $a = 1; endfor; return;else: $this->c->f(); endif; }"),
             ("php", "{ for ($i = 0; $i < 1; $i++): return; $d(); endfor; $this->c->f(); }",
              "{ for ($i = 0; $i < 1; $i++): return;endfor; $this->c->f(); }"),
             ("php", "{ $f(function () { $a = 1; return; $d(); }); $this->c->f(); }",
              "{ $f(function () { $a = 1; return;}); $this->c->f(); }"),
             ("php", "{ if (f(function ($y) { return $y; })): $a = 1; return; $d(); endif; $this->c->f(); }",
              "{ if (f(function ($y) { return $y;})): $a = 1; return;endif; $this->c->f(); }"),
             # after a nested `:` switch closes, the outer case body goes on up to the next outer case label
             ("php", "{ switch ($x) { case 1: switch ($a): case 2: f(); endswitch; return; $d(); case 3: $this->c->f(); } }",
              "{ switch ($x) { case 1: switch ($a): case 2: f(); endswitch; return;case 3: $this->c->f(); } }"),
             # a statement that ends the process is a flow end, with the stop points of return/throw
             ("php", "{ exit; $this->c->f(); }", "{ exit;}"),
             ("php", "{ EXIT(1); $this->c->f(); }", "{ EXIT(1);}"),
             ("php", "{ die(''); $this->c->f(); }", "{ die('');}"),
             ("php", "{ Die ; $this->c->f(); }", "{ Die ;}"),
             ("php", "{ exit(f($a)); $this->c->f(); }", "{ exit(f($a));}"),
             ("php", "{ switch ($x) { case 1: exit; $d(); default: $this->c->f(); } }",
              "{ switch ($x) { case 1: exit;default: $this->c->f(); } }"),
             ("php", "{ if ($x): die(); $d(); else: $this->c->f(); endif; }", "{ if ($x): die();else: $this->c->f(); endif; }"),
             ("java", "{ System.exit(0); repo.save(); }", "{ System.exit(0);}"),
             ("java", "{ java.lang.System.exit(1); repo.save(); }", "{ java.lang.System.exit(1);}"),
             ("java", "{ Runtime.getRuntime().halt(f(2)); repo.save(); }", "{ Runtime.getRuntime().halt(f(2));}"),
             ("java", "{ if (a) { Runtime.getRuntime().exit(0); repo.save(); } repo.load(); }",
              "{ if (a) { Runtime.getRuntime().exit(0);} repo.load(); }"),
             ("java", "{ switch (x) { case 1: System.exit(1); repo.save(); default: repo.load(); } }",
              "{ switch (x) { case 1: System.exit(1);default: repo.load(); } }"),
             ("csharp", "{ Environment.Exit(0); _repo.Save(); }", "{ Environment.Exit(0);}"),
             ("csharp", '{ System.Environment.FailFast(""); _repo.Save(); }', '{ System.Environment.FailFast("");}'),
             # not a flow end: an exit inside an expression, a longer statement, another name, or another language's exit
             ("php", "{ $x = $a ?: exit(1); $this->c->f(); }", "{ $x = $a ?: exit(1); $this->c->f(); }"),
             ("php", "{ foo() or die(); $this->c->f(); }", "{ foo() or die(); $this->c->f(); }"),
             ("php", "{ die() or f(); $this->c->f(); }", "{ die() or f(); $this->c->f(); }"),
             ("php", "{ exitNow(); $exit = 1; if ($a) exit; $this->c->f(); }", "{ exitNow(); $exit = 1; if ($a) exit; $this->c->f(); }"),
             ("java", "{ MySystem.exit(0); Runnable r = () -> System.exit(0); repo.save(); }",
              "{ MySystem.exit(0); Runnable r = () -> System.exit(0); repo.save(); }"),
             ("csharp", "{ System.exit(0); Environment.ExitCode = 1; _repo.Save(); }",
              "{ System.exit(0); Environment.ExitCode = 1; _repo.Save(); }"),
             ("java", "{ Environment.Exit(0); repo.save(); }", "{ Environment.Exit(0); repo.save(); }"),
             # the same under PHP error suppression, java.lang.Runtime, global::System
             ("php", "{ @exit; $this->c->f(); }", "{ @exit;}"), ("php", "{ @@Die(''); $this->c->f(); }", "{ @@Die('');}"),
             ("php", "{ $v = @exit(1); @f() or die(); @$this->x(); $this->c->f(); }", "{ $v = @exit(1); @f() or die(); @$this->x(); $this->c->f(); }"),
             ("java", "{ java.lang.Runtime.getRuntime().halt(1); repo.save(); }", "{ java.lang.Runtime.getRuntime().halt(1);}"),
             ("csharp", "{ global::System.Environment.Exit(0); _repo.Save(); }", "{ global::System.Environment.Exit(0);}"),
             ("csharp", "{ global :: System . Environment . FailFast(\"\"); _repo.Save(); }", "{ global :: System . Environment . FailFast(\"\");}")]
    for lang, code, want in cases:
        assert ind.drop_dead(code, lang) == want, code


def test_ruby_dead_lines_and_step_parameters():
    """A line whose first token is return/raise/fail (no trailing modifier) kills the following lines up to
    the end/} of its innermost block, stopping at else/elsif/when/in/rescue/ensure of the same depth; dead lines give no
    call and no constant. Step parameters are the |…| of the first line only."""
    cases = [("Given(/^a$/) do\n  go_a(1)\n  return\n  go_b(1)\nend", {"go_a"}, set()),
             ("Given(/^a$/) do\n  go_a(1)\n  return if ready\n  go_b(1)\nend", {"go_a", "go_b"}, set()),
             ("Given(/^a$/) do\n  raise Boom unless ready\n  go_b(1)\nend", {"raise", "go_b"}, {"Boom"}),
             ("When(/^b$/) do\n  if ready\n    fail 'no'\n    go_c(1)\n  else\n    go_d(1)\n  end\n  go_e(1)\nend",
              {"fail", "go_d", "go_e"}, set()),
             ("When(/^b$/) do |n|\n  return\n  [n].each do |i|\n    go_c(i)\n  end\n  go_d(n)\nend", set(), set()),
             ("When(/^b$/) do\n  begin\n    raise Boom\n    go_c(1)\n  rescue Boom\n    go_d(1)\n  ensure\n    go_e(1)\n  end\nend",
              {"raise", "go_d", "go_e"}, {"Boom"}),
             ("When(/^b$/) {\n  return\n  go_c(1)\n}\ngo_d(1)", {"go_d"}, set()),
             ("When(/^b$/) do |n|\n  case n\n  when 1\n    return\n    go_c(1)\n  when 2\n    go_d(1)\n  end\nend", {"go_d"}, set()),
             ("When(/^b$/) do\n  fail = true\n  go_c(fail)\nend", set(), set()),  # literal: the first token is `fail`
             ("When(/^b$/) do\n  return\n  Renalware::Ghost.find(1)\nend", set(), set()),
             ("When(/^b$/) do\n  x = \"return\"\n  # return\n  go_c(x)\nend", {"go_c"}, set()),
             ("When(/^b$/) do\n  return\n  x = { a: go_c(1) }\n  go_d(1)\nend", set(), set()),
             ("When(/^b$/) {\n  [1].each { |i|\n    return }\n  go_c(1)\n}", {"go_c"}, set())]
    for code, calls, classes in cases:
        got = ind.Scorer._ruby(code)
        assert (got[1], got[2]) == (calls, classes), code
    inner = ind.units("ruby", "Given(/^a$/) do\n  [1].each do |i|\n    go(i)\n  end\nend")[1][0]
    one = ind.units("ruby", "Given(/^(\\d+) a$/) { |n, *rest|\n  [n].each { |i| go(i) }\n}")[1][0]
    assert (inner["arity"], inner["required"], inner["total"]) == (0, 0, 0)
    assert (one["arity"], one["required"], one["total"]) == (2, 1, math.inf)


def test_a_struct_or_an_anonymous_class_in_a_member_is_raw():
    """A C# struct (record struct too) declared in a block, and a Java field whose initializer holds an anonymous
    class body that declares a method, fail condition 1 like any type declared in a block; an anonymous class without
    a method, or inside a method, does not."""
    struct = '[Given(@"x")]\npublic void A() { }\n\npublic struct Wrapper\n{\n    [Given(@"y")]\n    public void B() { }\n}'
    rec = 'public readonly record struct P(int A);\n\n[Given(@"x")]\npublic void A() { }'
    anon = 'private Object probeX = new Object() {\n    @Given("y")\n    public void b() {}\n};\n\n@Given("x")\npublic void a() {}'
    fine = ('private Object f = new Object();\nprivate int[] g = new int[] {1};\n\n@Given("x")\npublic void a() {\n'
            '    Runnable r = new Runnable() { public void run() {} };\n}')
    empty = 'private TypeReference<List<Foo>> t = new TypeReference<List<Foo>>() {};\n\n@Given("x")\npublic void a() {}'
    fields = 'private Object t = new Object() { int n = 1; };\n\n@Given("x")\npublic void a() {}'
    for lang, code, ok, kinds in (("csharp", struct, False, ["method", "raw"]), ("csharp", rec, False, ["raw", "method"]),
                                  ("java", anon, False, ["raw", "method"]), ("java", fine, True, ["member", "member", "method"]),
                                  ("java", empty, True, ["member", "method"]), ("java", fields, True, ["member", "method"])):
        assert ind.units(lang, code)[2] is ok, code
        assert [i["kind"] for i in ind.items_of(lang, code)] == kinds, code


def test_initializers_are_judged_by_condition_6(pair):
    """The initializer of a field, property or constant of the output is judged by condition 6 like a body
    (declared names resolve; a lambda parameter is local); a property with an arrow body or no initializer is not."""
    java = ('private Object probeF = ghostFieldZ.make();\nprivate Function<String, String> g = s -> helper(s);\nprivate int n;\n\n'
            '@Given("x")\npublic void a() {}')
    cs = ('private object _probeF = _ghostFieldZ.Make();\npublic int P { get; set; } = Compute();\npublic int Q => _other;\n'
          'private IFoo _foo;\n\n[Given(@"x")]\npublic void A() { }')
    php = "private $a = self::GHOST_Z, $b;\nconst C = self::D;\nprivate $c;\n\n" + GOLD
    for lang, code, want in (("java", java, [{"ghostFieldZ"}, {"helper"}]), ("csharp", cs, [{"_ghostFieldZ"}, {"Compute"}]),
                             ("php", php, [{"GHOST_Z"}, {"D"}])):
        assert [ind.member_uses(lang, t)[0] for t in ind.init_texts(lang, code)] == want, code
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    ghost = scorer.score(p.record, labels_row, block("private $probeF = self::GHOST_Z;\n\n" + GOLD))
    assert (ghost["bv_resolved"], ghost["unresolved"], ghost["bv_grounded"], ghost["bv"]) == (0, 1, 1, 0)
    own = scorer.score(p.record, labels_row, block("private const PROBE_A = 1;\nprivate $probeF = self::PROBE_A;\n\n" + GOLD))
    assert (own["bv_resolved"], own["bv"]) == (1, 1)


def test_independent_shapes_on_the_java_fixture(jpair):
    """The independent scorer alone on the Java fixture: X6 (a logger field whose initializer calls getClass()) scores
    as gold, a Java Object member resolves but never grounds, an anonymous class is raw only when it declares a
    method, plus X3, X5 and X8."""
    p, scorer = jpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    for code, bv, ca_ in JPROBES:
        mine = scorer.score(p.record, labels_row, jblock(code))
        assert (mine["bv"], scorer.ca(p.record, labels_row, mine)) == (bv, (True, ca_)), code
    fo = scorer.score(p.record, labels_row, jblock(FO_ONLY))
    assert (fo["bv_resolved"], fo["bv_grounded"], fo["open_resolution"]) == (1, 0, False)


def test_independent_scorer_agrees_on_the_java_fixture(jpair):
    p, scorer = jpair
    for code, _, _ in JPROBES:
        compare(p, scorer, jblock(code))


def test_java_shape_probe_outputs_on_the_java_fixture(jpair):
    """probe_outputs builds X3, X5, X6 and X8 from the Java fixture's gold answer and stub-v3; both scorers agree on
    them, X6 scores as gold, X3 and X5 fail BV, X8 has CA 0."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    p, scorer = jpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    gold = extract(scorer.gold_text(p.record), p.targets).by_path()
    out = po.probes(p.record, "java", "java_annotation", gold, controls(p)["stub-v3"][2])
    got = {}
    for name in ("X3_anon_class_field", "X5_field_ghost_init", "X6_logger_field", "X8_system_exit"):
        mine = compare(p, scorer, out[name])
        got[name] = (mine["bv"], scorer.ca(p.record, labels_row, mine)[1])
    assert got == {"X3_anon_class_field": (0, 0), "X5_field_ghost_init": (0, 0), "X6_logger_field": (1, 1), "X8_system_exit": (1, 0)}


def test_helper_expansion_probe_outputs_on_both_fixtures(pair, jpair):
    """probe_outputs' helper-expansion probes, built from each fixture's controls: both scorers agree on them; a cycle
    of helpers around the gold lines scores as gold (H3), the gold lines beyond depth 3 give CA 0 (H4), and a stub
    whose body moved into a helper scores as the stub (H6 = stub-v3m, H7 = stub-v3all)."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    for (p, scorer), lang, style in ((pair, "php", "php_attribute"), (jpair, "java", "java_annotation")):
        labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
        ctl = controls(p)
        more = {name: ctl[name][2] for name in ("stub-v3m", "stub-v3all")}
        out = po.probes(p.record, lang, style, extract(scorer.gold_text(p.record), p.targets).by_path(), ctl["stub-v3"][2], more)
        ca_of = lambda text: scorer.ca(p.record, labels_row, compare(p, scorer, text))[1]
        stub = lambda name: "\n\n".join(f"{TICKS}{lang}\n// FILE: {path}\n{code}\n{TICKS}" for path, code in ctl[name][2].items())
        assert (ca_of(out["H3_helper_cycle"]), ca_of(out["H4_helper_chain4"])) == (1, 0), lang
        assert (ca_of(out["H6_v3m_helper"]), ca_of(out["H7_v3all_helper"])) == (ca_of(stub("stub-v3m")), ca_of(stub("stub-v3all"))), lang


def test_masked_infrastructure_is_future_for_dev_records(tmp_path):
    """The shown names with 3A's masked answer infrastructure (infra.json, Composer.infra()["members"]):
    for a train or dev record a masked member of the class chain is not shown, so a masked method is future on both
    scorers (a call to it opens into its body); for a test record the mask does not apply."""
    masked = frozenset({(SHOP, "iAddToCart")})
    for part, future in (("dev", True), ("test", False)):
        (tmp_path / part).mkdir()
        p0 = make_prep(tmp_path / part, part)
        labels_row = {"step_def_ids": [[ids[0]] for _, ids in p0.repo.suite_steps["s"][:2]]}
        p = prepare(p0.record, p0.repo, labels_row, masked if part == "dev" else frozenset())  # as score_run: Composer.masked
        scorer = ind.Scorer("Acme/Shop", "php", ["php_attribute", "php_docblock"], catalog(), ind.Git(p.repo.ctx.git_dir, p.record["commit"]),
                            p.repo.suite_steps, infra=masked)
        from v2tools.evaluate import future_methods
        assert ("iaddtocart" in future_methods(p)[SHOP]) is future
        assert ("iaddtocart" in scorer.future(SHOP, p.record, p.er_files)) is future
        mine = compare(p, scorer, block(php_step(["$this->iAddToCart($total);", "$this->checkout->assertTotal($total);"])))
        assert set().union(*mine["calls"].values()) == ({("cart", "add")} if future else {("self", "iAddToCart")}) | {("checkout", "assertTotal")}


def test_resolver_keeps_visible_names_and_never_opens_a_constructor(jpair):
    """A ("self", f) call to a name V(r) shows stays, even when the output rewrites f; else the
    output's version opens, else the future one; a constructor never opens (Java: the class name; PHP: __construct)."""
    _, scorer = jpair
    fut = {("baseNew", 0): [("baseNew", JBASE, "void baseNew() {}", False, True)], ("ShopSteps", 0): [("ShopSteps", JPATH, "ShopSteps() {}", True, True)]}
    own = {"baseOld": [(JPATH, "void baseOld() {}")], "probeHelper": [(JPATH, "void probeHelper() {}")], "ShopSteps": [(JPATH, "ShopSteps() {}")]}
    resolve = scorer.resolver(JPATH, {"baseOld"}, own, {"ShopSteps"}, {("baseOld", 0), ("probeHelper", 0), ("ShopSteps", 0)}, fut)
    assert [resolve(f) for f in ("baseOld", "probeHelper", "baseNew", "ShopSteps", "other")] == \
        [None, [(JPATH, "void probeHelper() {}")], [(JBASE, "void baseNew() {}")], None, None]  # a body with its declaring file
    # for the output's own methods too: an abstract one never opens
    resolve = scorer.resolver(JPATH, set(), {"probeAbstract": [(JPATH, "abstract void probeAbstract();")]}, set(), {("probeAbstract", 0)}, {})
    assert resolve("probeAbstract") is None


def test_fence_probes_of_probe_outputs(pair):
    """U13 (a second FILE line in the gold's fence) and U14 (the gold in a four-backtick fence holding a three-backtick
    line in a comment) from probe_outputs: both scorers agree; U13 is multi_file_fence, U14 reads as the gold."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    p, scorer = pair
    gold = extract(scorer.gold_text(p.record), p.targets).by_path()
    out = po.probes(p.record, "php", "php_attribute", gold, controls(p)["stub-v3"][2])
    u13, u14 = compare(p, scorer, out["U13_multi_file_fence"]), compare(p, scorer, out["U14_long_fence"])
    g = compare(p, scorer, scorer.gold_text(p.record))
    assert (u13["extract_status"], u14["extract_status"], u14["bv"], u14["fence_truncated"]) == ("multi_file_fence", "ok", g["bv"], False)


def test_capture_groups_are_read_with_a_regex_parser():
    """Groups are counted from the regex parse tree, so a capture group inside a non-capturing group is never missed
    (PrestaShop SecurityFeatureContext.php:130: 2 outermost groups, 2 in all); a lookaround, an atomic group or a repeat
    hides nothing; escaped and bracketed parentheses are no group. Ruby step parameters are read from the step call's
    block in the tree."""
    pins = {r'^a session for (?:customer named "(.*)"|the employee) is created (\d+) hours? ago$': (2, 2),
            r"^((a)(b))$": (1, 3), r"^(?:(a)|(b))$": (2, 2), r"^(?<name>x) (?P<y>y)$": (2, 2), r"^(?=(a))a(b)?$": (2, 2),
            r"^\((a)\) [(]x(?>(c))$": (2, 2), r"^\h+ (\d+)$": (1, 1), r"^(a(?:b(c)))$": (1, 2),
            # a pattern the parser rejects (an inline flag not at the start, PCRE-only syntax) hands over nothing, as it
            # never binds: no hand scanning of parentheses (design specification §2.2 at 448165c)
            r"^(?i)(a)$": (0, 0), r"^(?<a>x)(?'b'y)$": (0, 0), r"^(a)\K(b)$": (0, 0)}
    for pat, (outer, every) in pins.items():
        assert (ind.regex_args(pat, "java"), ind.regex_args(pat, "ruby"), ind.regex_args(pat, "php"), ind.regex_args(pat, "csharp")) == \
            (outer, outer, every, every), pat
    for code, n in (('Given(/^I (do|don\'t) have (\\d+)$/) do |a, n|\n  go(a, n)\nend', 2), ('Given(/^x (\\d+)$/) do\n  |n|\n  go(n)\nend', 1),
                    ('Given(/^x$/) do\n  [1].each do |i|\n    go(i)\n  end\nend', 0), ("When(/^b (\\d+)$/) { |n, *m|\n  go(n)\n}", 2)):
        assert ind.units("ruby", code)[1][0]["arity"] == n, code


def test_condition_4_follows_each_framework(pair, jpair):
    """Condition 4 per framework (design specification §2.2 at 448165c, fixed 25/09/2026): N is every capture group,
    nested ones included (a named group once), for PHP and C#, and the outermost ones (non-capturing constructs looked
    through) for Java and Ruby; PHP passes when required <= N <= total (variadic: no upper bound); C# when every
    declared parameter, optional ones and a `params` array (one) included, makes N; Java when the parameters make N
    (varargs count one); Ruby when the required block parameters make N, or N >= required with a splat. Groups are read
    with a regex parser."""
    from collections import namedtuple
    D = namedtuple("D", "kind pattern")
    nested = r'^I should see the total "((\d+)(?:\.(\d+))?)"$'
    counts = {"php": 3, "csharp": 3, "java": 1, "ruby": 1}
    for lang, n in counts.items():
        assert ind.regex_args(nested, lang) == n, lang
    assert ind.regex_args(r"^(?<a>x)(?P<b>y)$", "php") == 2 and ind.regex_args(r"^(?:(a)|(b))$", "java") == 2
    unit = lambda lang, code: ind.units(lang, code)[1][0]
    cases = [("php", "#[Then('/x/')]\npublic function t(string $a, string $b = '', ...$c): void\n{\n}", [1, 2, 5], [0]),
             ("csharp", '[Then(@"x")]\npublic void T(string a, string b = "", params string[] c)\n{\n}', [3], [1, 2, 4]),
             ("java", '@Then("x")\npublic void t(String a, String... b) {\n}', [2], [1, 3]),
             ("ruby", "Given(/^x$/) do |a, b = 1|\n  a\nend", [1], [0, 2]),
             ("ruby", "Given(/^x$/) do |a, *b|\n  a\nend", [1, 2, 5], [0])]
    for lang, code, ok, bad in cases:
        u = unit(lang, code)
        assert [ind.arity_ok(lang, u, n) for n in ok + bad] == [True] * len(ok) + [False] * len(bad), code
    # on the fixtures, against the main scorer: a PHP step whose nested groups each make an argument; a Java one where
    # only the outer group does, and where varargs are no exception
    p, scorer = pair
    for code, want in (('#[Then(\'/^I should see the total "((\\d+))"$/\')]\npublic function t(string $a, string $b): void\n{\n    $this->checkout->assertTotal($a);\n}', 1),):
        assert compare(p, scorer, block(code))["bv_arity"] == want, code
    p, scorer = jpair
    for code, want in (('@Then("^I should see the total \\"((\\\\d+))\\"$")\npublic void t(String total) {\n    cart.assertTotal(total);\n}', 1),
                       ('@Then("^I should see the total \\"(\\\\d+)\\"$")\npublic void t(String total, String... more) {\n    cart.assertTotal(total);\n}', 0)):
        assert compare(p, scorer, jblock(code))["bv_arity"] == want, code


def test_step_definitions_count_only_before_the_body(tmp_path):
    """An item registers only the step definitions whose annotation starts before the `{` that opens its
    body (C# expression body: `=>`; a method without a body keeps them all); a lambda item only its own registration; a
    Ruby block only the step call on its first line. The same definitions as evaluate.item_defs."""
    from v2tools.evaluate import item_defs
    cases = [("php", "method", ["php_attribute", "php_docblock"], "#[Then('/^a$/')]\npublic function a(): void\n{\n    $x = new class {\n"
              "        #[Then('/^b$/')]\n        public function b(): void {}\n    };\n}"),
             ("php", "method", ["php_attribute", "php_docblock"], "/**\n * @When /^a$/\n */\npublic function a(): void\n{\n}"),
             ("java", "method", ["java_annotation", "java_lambda"], '@Then("^a$") public void a() {\n    Object o = new Object() {\n'
              '        @Then("^b$")\n        public void b() {}\n    };\n}'),
             ("java", "lambda", ["java_annotation", "java_lambda"], 'Given("^a$", () -> {\n    When("^b$", () -> {\n    });\n});'),
             ("csharp", "method", ["csharp_attribute"], '[Then(@"^a$")]\npublic void A() => F();'),
             ("csharp", "method", ["csharp_attribute"], '[Then(@"^a$")]\npublic void A()\n{\n    [Then(@"^b$")]\n    void B() { }\n}'),
             ("csharp", "method", ["csharp_attribute"], '[Then(@"^a$")]\npublic abstract void A();'),
             ("ruby", "block", ["ruby_block"], "Given(/^a$/) do\n  Given(/^b$/) do\n  end\nend")]
    for lang, kind, styles, text in cases:
        mine = sorted(d.pattern for d in ind.unit_defs(lang, kind, "f", text, styles))
        assert mine == sorted(d.pattern for d in item_defs(lang, kind, "f", text, styles)), text
        assert "b" not in "".join(mine).replace("^", "").replace("$", ""), text  # a C# expression body: stepdefs reads none


def test_condition_4_php_arity_rule_and_ruby_arity():
    """The arity rule (design specification at fe73dc9, fixed 25/09/2026): a PHP step method passes when required <= N,
    with no upper bound (Behat drops the arguments it does not take); a step with a table or doc string also needs a
    parameter that takes it: one type-hinted TableNode/PyStringNode, or N parameters in all. Ruby block parameters count
    as Proc#arity does: a plain or destructured one is required, the required keywords together count one, a default,
    **kw or &blk does not count, a splat leaves no upper bound. As argcheck.params_of and arity_ok."""
    from v2tools.argcheck import arity_ok, params_of
    php = [("#[Then('/x/')]\npublic function t(string $a): void\n{\n}", [(1, False), (2, False), (1, True)], [(0, False), (2, True)]),
           ("#[Then('/x/')]\npublic function t(string $a, ?\\Behat\\Gherkin\\Node\\TableNode $t): void\n{\n}", [(2, True), (5, True)], [(1, True)]),
           ("#[Then('/x/')]\npublic function t(string $a, $b): void\n{\n}", [(2, True)], [(3, True)])]
    for code, ok, bad in php:
        u = ind.units("php", code)[1][0]
        p = params_of("php", "method", code)
        for (n, table), want in [(x, True) for x in ok] + [(x, False) for x in bad]:
            assert ind.arity_ok("php", u, n, table) == arity_ok("php", n, p[0], p[1], table, p.takes_table) == want, (code, n, table)
    for code in ("Given(/^x$/) do |a, b = 1, *c, d:, e: 2, **f, &g|\n  a\nend", "Given(/^x$/) do |a, (b, c), d:, k:, &g|\n  a\nend",
                 "Given(/^x$/) do |a, b = 1|\n  a\nend", "Given(/^x$/) do |n; tmp, u|\n  n\nend"):  # block-locals are no parameters
        u = ind.units("ruby", code)[1][0]
        p = params_of("ruby", "block", code)
        assert (u["required"], u["total"]) == (p[0], p[1]), code
        for n in range(5):
            assert ind.arity_ok("ruby", u, n, False) == arity_ok("ruby", n, p[0], p[1]), (code, n)


def test_php_arity_rule_table_clause_on_the_fixture(pair):
    """The arity rule through both scorers: with a table after the CREATE step (N = 1 placeholder + 1), `t(string
    $total)` fails, while a TableNode parameter or a second parameter takes the table."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    r = dict(p.record, scenario=p.record["scenario"] + "\n    | a |\n    | 1 |")
    p2 = prepare(r, p.repo, labels_row)
    head = "#[Then('I should see the total :total')]\npublic function t("
    tail = "): void\n{\n    $this->checkout->assertTotal($total);\n}"
    got = [compare(p2, scorer, block(head + params + tail))["bv_arity"]
           for params in ("string $total", "string $total, TableNode $table", "string $total, $rows", "string $total, ?\\Behat\\Gherkin\\Node\\PyStringNode $d")]
    assert got == [0, 1, 1, 1]


def test_behat_n_is_read_on_the_step_in_both_scorers(pair, jpair):
    """Condition 4 (§2.2 of the design specification at 25ba873, fixed 28/09/2026): for PHP, N is the step's groups up
    to the last one that matched (preg_match drops the trailing unmatched ones), in both scorers, each its own way: the
    ExactContext patterns and steps of real Behat runs, a trailing optional group left out by the fixture's CREATE step
    (a method requiring its argument fails, a default or no such parameter passes), and a table after such a step taken
    by an untyped second parameter. The other frameworks read N off the pattern."""
    from collections import namedtuple
    from v2tools.argcheck import placeholders
    D = namedtuple("D", "kind pattern")
    p, scorer = pair
    exact = [(r"^(a|b) and ((c)|d)$", "b and c", 3), (r"^(a|b) and ((c)|d)$", "a and d", 2),
             (r'^I add a shop group "(.+)" with name "(.+?)"(?: and color "(.+)")?$', 'I add a shop group "G1" with name "N1"', 2),
             (r'^I add a shop group "(.+)" with name "(.+?)"(?: and color "(.+)")?$',
              'I add a shop group "G2" with name "N2" and color "red"', 3),
             (r"^x(?: (\d+))? y (\d+)$", "x y 7", 2), (r"^the price is (?P<amount>\d+) (?<currency>[A-Z]+)$", "the price is 42 EUR", 2)]
    for pattern, step, n in exact:
        assert scorer._placeholders(D("regex_full", pattern), step) == placeholders("regex_full", pattern, "php", step) == n
    _, jscorer = jpair
    assert jscorer._placeholders(D("regex_full", r"^(a|b) and ((c)|d)$"), "a and d") == 2 == \
        placeholders("regex_full", r"^(a|b) and ((c)|d)$", "java", "b and c")
    head = "#[Then('/^I should see the total \"(\\d+)\"(?: in (\\w+))?$/')]\npublic function t("
    tail = "): void\n{\n    $this->checkout->assertTotal($total);\n}"
    got = [compare(p, scorer, block(head + sig + tail))["bv_arity"]
           for sig in ("string $total, string $currency", "string $total, string $currency = ''", "string $total")]
    assert got == [0, 1, 1]
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    p2 = prepare(dict(p.record, scenario=p.record["scenario"] + "\n    | a |\n    | 1 |"), p.repo, labels_row)
    got = [compare(p2, scorer, block(head + sig + tail))["bv_arity"] for sig in ("string $total, $rows", "string $total")]
    assert got == [1, 0]  # N = 1 + the table: taken by $rows; every group would ask for 3 and fail the first


def test_the_copied_hardcoded_keeps_the_pins_of_real_steps():
    """score_independent keeps its own copy of argcheck.hardcoded, so one edit of both would pass both scorers'
    agreement; the steps where a mutant flipped a scored gold (fineract WorkingCapitalLoanOrigination.feature:320) are
    pinned on the copy too."""
    from collections import namedtuple
    D = namedtuple("D", "kind pattern")
    hard = ind.Scorer._hardcoded
    assert not hard(D("cucumber", "Working Capital loan breach schedule has {int} period(s)"),
                    "Working Capital loan breach schedule has 1 period")
    assert not hard(D("regex_full", '^I change order "(.*)" note to "(.*)"$'), 'I change order "bo_order1" note to ""')
    assert not hard(D("turnip", "The default category of product :productName has no group reduction"),
                    'the default category of product "Mug" has no group reduction')

@pytest.fixture()
def jlpair(tmp_path):
    """The Java fixture read with the java8 lambda style too."""
    return build_pair(tmp_path, "Acme/Java", "java", ["java_annotation", "java_lambda"],
                      [(T2020, {"src/main/java/acme/Cart.java": CART, JBASE: BASE_2020, JPATH: JAVA_2020}),
                       (T2024, {JBASE: BASE_2024, JPATH: JAVA_2024})], JPATH, "I open the shop", "I should see the total {string}", 1)


def test_java8_step_lambdas_only_where_cucumber_runs_them(jlpair):
    """Through both scorers: a step lambda registers only as a top-level statement of the block or of a constructor
    body; in a `static {}` block or a method body it never binds."""
    p, scorer = jlpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    lam = 'Then("I should see the total {string}", (String total) -> {\n    cart.assertTotal(total);\n});'
    shapes = {"top": lam, "JB1_static": "static {\n" + lam + "\n}", "JB5_method": "public void probeHolder() {\n" + lam + "\n}",
              "constructor": "public ShopSteps() {\n    super();\n" + lam + "\n}"}
    got = {}
    for name, code in shapes.items():
        mine = compare(p, scorer, jblock(code))
        got[name] = (mine["bv_binding"], mine["bv"], scorer.ca(p.record, labels_row, mine)[1])
    assert (got["JB1_static"], got["JB5_method"]) == ((0, 0, 0), (0, 0, 0)) and got["top"][0] == got["constructor"][0] == 1, got


NOCTOR_PATH = "src/test/java/acme/steps/ShopSteps.java"
NOCTOR_2020 = ('package acme.steps;\n\nimport io.cucumber.java.en.Given;\nimport io.cucumber.java.en.Then;\n\n'
               'public class ShopSteps {\n\n    @Given("I open the shop")\n    public void iOpenTheShop() {\n'
               '    }\n}\n')
NOCTOR_2024 = NOCTOR_2020.replace("    }\n}\n", '    }\n\n    @Then("I should see the total {string}")\n'
                                  '    public void iShouldSeeTotal(String total) {\n    }\n}\n')


def test_java8_lambda_goes_into_the_outputs_own_constructor(tmp_path):
    """Fineract's standard cucumber-java8 form (fixed 26/09/2026): a java8 lambda step and an explicit no-argument
    constructor in the same output, on a class with no constructor at all: main puts the constructor first so the lambda
    joins IT, instead of a synthetic `X() {}` main would otherwise create for the lambda first and then clash with the
    output's own constructor (a `collision`, pinned false on both scorers here; a main without this order disagrees,
    independent never invents that synthetic constructor in the first place)."""
    p, scorer = build_pair(tmp_path, "Acme/JavaNoCtor", "java", ["java_annotation", "java_lambda"],
                           [(T2020, {NOCTOR_PATH: NOCTOR_2020}), (T2024, {NOCTOR_PATH: NOCTOR_2024})], NOCTOR_PATH,
                           "I open the shop", "I should see the total {string}", 1)
    code = ('public ShopSteps() {\n    int x = 1;\n}\n\n'
            'Then("I should see the total {string}", (String total) -> {\n    int y = x;\n});')
    text = f"{TICKS}java\n// FILE: {NOCTOR_PATH}\n{code}\n{TICKS}"
    mine = compare(p, scorer, text)
    assert mine["insert_status"] == "ok"


def test_a_shown_field_name_shows_a_method_of_that_name(tmp_path):
    """Visibility compares names as t_view gives them, so an old field `total` shows the name, and a method `total`
    written with the answer is not future: the gold's call (self, total) stays."""
    shop0 = JAVA_2020.replace("public class ShopSteps extends BaseSteps {\n", 'public class ShopSteps extends BaseSteps {\n\n    private String total = "0";\n')
    shop1 = shop0.replace("""        cart.open();
    }
""", """        cart.open();
    }

    @Then("I should see the total {string}")
    public void iShouldSeeTotal(String t) {
        total(t);
    }

    private void total(String t) {
        cart.assertTotal(t);
    }
""")
    p, scorer = build_pair(tmp_path, "Acme/J2", "java", ["java_annotation"],
                           [(T2020, {"src/main/java/acme/Cart.java": CART, JBASE: BASE_2020, JPATH: shop0}), (T2024, {JPATH: shop1})],
                           JPATH, "I open the shop", "I should see the total {string}", 1)
    tree, _ = ind.er_tree(p.record, scorer.fns, scorer.ghosts, scorer.git)
    assert "total" in scorer.visible(JPATH, p.record, tree) and not scorer.future(JPATH, p.record, tree)
    compare(p, scorer, jblock('@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String t) {\n    cart.assertTotal(t);\n}'))


# helper-call expansion: an abstract step class whose answer calls computeTotal, overloaded twice with arity 1
# and once with arity 2, and hook(), abstract; all written with the answer (future)
MCART = ("package acme;\n\npublic class Cart {\n    public void open() {\n    }\n\n    public void assertTotal(String t) {\n    }\n\n"
         "    public String total() {\n        return \"\";\n    }\n\n    public void reset() {\n    }\n\n    public void count() {\n    }\n}\n")
MSHOP0 = JAVA_2020.replace("public class ShopSteps", "public abstract class ShopSteps")
MSHOP1 = MSHOP0.replace("""        cart.open();
    }
""", """        cart.open();
    }

    @Then("I should see the total {string}")
    public void iShouldSeeTotal(String total) {
        cart.assertTotal(computeTotal(total));
        hook();
    }

    private String computeTotal(String t) {
        cart.total();
        return t;
    }

    private String computeTotal(int n) {
        cart.reset();
        return "";
    }

    private String computeTotal(int a, int b) {
        cart.count();
        return "";
    }

    protected abstract void hook();
""")


@pytest.fixture()
def mpair(tmp_path):
    return build_pair(tmp_path, "Acme/JM", "java", ["java_annotation"],
                      [(T2020, {"src/main/java/acme/Cart.java": MCART, JBASE: BASE_2020, JPATH: MSHOP0}), (T2024, {JPATH: MSHOP1})],
                      JPATH, "I open the shop", "I should see the total {string}", 1)


def test_overloads_bodiless_methods_and_replaced_keys(mpair):
    """Fixed 25/09/2026: overloads of the same arity are not collapsed: (self, computeTotal) opens into all three; a
    method without a body (abstract hook) never opens, (self, hook) stays; the output's computeTotal(String) replaces
    only the FIRST declaration of the key (computeTotal, 1) in the file, so the call opens into the output's version,
    the file's other (computeTotal, 1), computeTotal(int), and computeTotal(int, int). The expansion path holds bodies:
    the output's computeTotal calling its own name opens the other overloads and counts one cycle."""
    p, scorer = mpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    g = scorer.gold(p.record, labels_row)
    assert set().union(*g["calls"].values()) == {("cart", "assertTotal"), ("cart", "total"), ("cart", "reset"), ("cart", "count"), ("self", "hook")}
    step = '@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(computeTotal(total));\n}'
    out = compare(p, scorer, jblock(step + '\n\nprivate String computeTotal(String t) {\n    cart.open();\n    return t;\n}'))
    assert set().union(*out["calls"].values()) == {("cart", "assertTotal"), ("cart", "open"), ("cart", "reset"), ("cart", "count")}
    compare(p, scorer, jblock(step))
    out = compare(p, scorer, jblock(step + '\n\nprivate String computeTotal(String t) {\n    computeTotal(1);\n    return t;\n}'))
    assert set().union(*out["calls"].values()) == {("cart", "assertTotal"), ("cart", "reset"), ("cart", "count"), ("self", "computeTotal")}
    assert out["cuts"] == {"depth": 0, "cycle": 1}


def test_private_methods_of_a_parent_trait_and_csharp_defaults_are_not_future(tmp_path):
    """A private method of a trait a PARENT class uses is not future (it is the parent's, and an ancestor's private
    method is not shown); a private method of the class's own trait can be. A C# member without an access modifier is
    private."""
    base = "<?php\n\nnamespace App\\Behat;\n\nabstract class BaseContext\n{\n    use ParentTrait;\n}\n"
    ptrait0 = "<?php\n\nnamespace App\\Behat;\n\ntrait ParentTrait\n{\n}\n"
    ptrait1 = "<?php\n\nnamespace App\\Behat;\n\ntrait ParentTrait\n{\n    private function parentSecret(): void\n    {\n    }\n}\n"
    otrait0 = "<?php\n\nnamespace App\\Behat;\n\ntrait OwnTrait\n{\n}\n"
    otrait1 = "<?php\n\nnamespace App\\Behat;\n\ntrait OwnTrait\n{\n    private function ownSecret(): void\n    {\n    }\n}\n"
    shop0 = PSHOP_2020.replace("final class ShopContext extends BaseContext\n{\n", "final class ShopContext extends BaseContext\n{\n    use OwnTrait;\n\n")
    shop1 = PSHOP_2024.replace("final class ShopContext extends BaseContext\n{\n", "final class ShopContext extends BaseContext\n{\n    use OwnTrait;\n\n")
    composer = '{"autoload": {"psr-4": {"App\\\\": "src/"}}}\n'
    p, scorer = build_pair(tmp_path, "Acme/Traits", "php", ["php_attribute", "php_docblock"],
                           [(T2020, {"composer.json": composer, "src/Util/Language.php": PLANG, PBASE: base.replace("{\n    use", "{\n    protected function baseOld(): void\n    {\n    }\n\n    use"),
                                     "src/Behat/ParentTrait.php": ptrait0, "src/Behat/OwnTrait.php": otrait0, PPATH: shop0}),
                            (T2024, {"src/Behat/ParentTrait.php": ptrait1, "src/Behat/OwnTrait.php": otrait1, PPATH: shop1})],
                           PPATH, "I open the shop", "I should see the total :total", 2)
    tree, _ = ind.er_tree(p.record, scorer.fns, scorer.ghosts, scorer.git)
    names = {e[0] for found in scorer.future(PPATH, p.record, tree).values() for e in found}
    assert "ownSecret" in names and "parentSecret" not in names
    from v2tools.evaluate import future_methods
    assert names == {f.name for f in future_methods(p)[PPATH].values()}
    cs = ind.class_methods("csharp", "class A {\n    void M() { }\n    public void N() { }\n    internal void O() { }\n}")
    assert [(m.name, m.access) for m in cs] == [("M", False), ("N", True), ("O", True)]
    # no modifier means private, except in an interface: there a member is public (evaluate.csharp_interface)
    assert ind.csharp_private("csharp", "interface I {\n    void M();\n}", ind.class_methods("csharp", "interface I {\n    void M();\n}")[0]) is False
    assert ind.csharp_private("csharp", "class A {\n    void M() { }\n}", cs[0]) is True


def test_php_keywords_in_any_case_and_nullsafe_this(pair):
    """PHP keywords are case-insensitive (`SELF::`, `Static::`, `RETURN;`, `THROW`, `NEW`, `::CLASS`, `__CONSTRUCT`) and
    `$this?->` is a member access, as tree-sitter reads them; both scorers agree."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    shapes = {"SELF": (["SELF::computeTotal($total);", "$this->checkout->assertTotal($total);"], 1),
              "Static": (["Static::computeTotal($total);", "Parent::helperOld();", "$this->checkout->assertTotal($total);"], 1),
              "RETURN": (["RETURN;", "$this->checkout->assertTotal($total);"], 0),
              "THROW": (["THROW NEW \\RuntimeException('probe');", "$this->checkout->assertTotal($total);"], 0),
              "nullsafe": (["$this?->checkout->assertTotal($total);", "$this?->computeTotal($total);", "$x = $this?->cart;"], 1)}
    for name, (lines, want) in shapes.items():
        mine = compare(p, scorer, block(php_step(lines)))
        assert scorer.ca(p.record, labels_row, mine)[1] == want, name
    helper = php_step(["$this->probeHelper($total);"]) + "\n\n" + php_fn("probeHelper", ["RETURN;", "$this->checkout->assertTotal($total);"])
    assert scorer.ca(p.record, labels_row, compare(p, scorer, block(helper)))[1] == 0
    assert ind.units("php", "public function __CONSTRUCT()\n{\n}")[1][0]["ctor"] and ind.class_methods("php", "<?php class A { function __Construct() {} }")[0].ctor
    # `::class` in any case names the class, never a member (independent only: symbols.php_refs reads SELF::CLASS as a
    # member CLASS)
    assert ind.member_uses("php", php_step(["$x = SELF::CLASS;", "$y = Foo::Class;"]))[0] == set()


def test_a_new_target_file_keeps_its_ancestors_old_helpers(tmp_path):
    """Future by time (fixed 25/09/2026): for a target file new at t_vis, the shown names are those ④(a) shows for each
    in-repo ancestor, less an ancestor's constructor and its private members: BaseSteps.baseOld (2020) stays (self,
    baseOld), BaseSteps.baseNew (2024) and the new file's own computeTotal are future and open."""
    shop = JAVA_2024.replace("    @Given(\"I open the shop\")", "    @Given(\"I open the shop\")").replace(
        "cart.assertTotal(computeTotal(total));", "cart.assertTotal(computeTotal(total));\n        baseOld();")
    p, scorer = build_pair(tmp_path, "Acme/New", "java", ["java_annotation"],
                           [(T2020, {"src/main/java/acme/Cart.java": CART, JBASE: BASE_2020}), (T2024, {JBASE: BASE_2024, JPATH: shop})],
                           JPATH, "I open the shop", "I should see the total {string}", 1)
    from v2tools.evaluate import future_methods, shown_names
    tree, _ = ind.er_tree(p.record, scorer.fns, scorer.ghosts, scorer.git)
    assert scorer.visible(JPATH, p.record, tree) == shown_names(p, JPATH)
    assert {e[0] for found in scorer.future(JPATH, p.record, tree).values() for e in found} == {f.name for f in future_methods(p)[JPATH].values()}
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    assert set().union(*scorer.gold(p.record, labels_row)["calls"].values()) == {("cart", "assertTotal"), ("store", "total"), ("self", "baseOld")}
    compare(p, scorer, jblock(JGOLD))


def test_php_method_names_ignore_case(pair):
    """A PHP method call resolves (condition 6) and grounds (condition 5) in any case, and CA compares method names in
    any case, gold and output alike; property names stay case-sensitive."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    for lines, want in ((["$this->checkout->ASSERTTOTAL($total);"], (1, 1, 1)), (["$this->IADDTOCART($total);"], (1, 0, 1)),
                        (["$this->CHECKOUT->assertTotal($total);"], (0, 0, 1)),
                        (["$this->checkout->assertTotal($total);", "$this->checkout->ASSERTtotal($total);"], (1, 1, 1))):
        mine = compare(p, scorer, block(php_step(lines)))  # n_calls folds PHP method names too: two cases count once
        assert (mine["bv"], scorer.ca(p.record, labels_row, mine)[1], mine["n_calls"]) == want, lines


def test_dead_overload_probe_of_probe_outputs(jpair):
    """Pre-registered probe O1 (a documented limit): the step calls a new helper probeH() whose body
    only calls toString() on the gold's first receiver, and a never-called overload probeH(…, int neverCalled) holds
    the gold body; calls resolve by name to the union of overloads, so both scorers give CA as gold."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    p, scorer = jpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    gold = extract(scorer.gold_text(p.record), p.targets).by_path()
    out = po.probes(p.record, "java", "java_annotation", gold, controls(p)["stub-v3"][2])["O1_dead_overload"]
    assert "private void probeH() {\n    cart.toString();\n}" in out and "private void probeH(String total, int neverCalled) {" in out
    mine = compare(p, scorer, out)
    assert (mine["bv"], scorer.ca(p.record, labels_row, mine)[1]) == (1, 1)


def merged_record(p, steps: list[str], kinds: list[str], names: list[str]):
    """The fixture record of `p` re-prepared with `steps` (their kinds; `names` = the method binding each one): every
    CREATE step's method becomes a new CREATE function of the answer (probe SM needs two)."""
    from v2tools.compose import gold_answer
    fid = {n: next(f for f, row in p.repo.fns.items() if f"{n}(" in row["code"]) for n in set(names)}
    fns = [fid[n] for n in names]
    create = list(dict.fromkeys(f for f, k in zip(fns, kinds) if k == "create"))
    r = dict(p.record, steps=steps, step_kinds=kinds, step_fns=fns,
             scenario="Scenario: Merge\n" + "".join(f"  Given {s}\n" for s in steps).rstrip(),
             create=[{"fn_id": f, "path": p.repo.fns[f]["path"], "code": p.repo.fns[f]["code"], "preexisting": False,
                      "post_cutoff_code": True} for f in create],
             create_def_ids=sorted(p.repo.fns[f]["annotations"][0]["def_id"] for f in create))
    labels_row = {"step_def_ids": [[p.repo.fns[f]["annotations"][0]["def_id"]] for f in fns]}
    q = prepare(r, p.repo, labels_row)
    return q, labels_row, extract(gold_answer(r, p.repo.fns), q.targets).by_path()


def test_stub_merge_probe_of_probe_outputs(pair, jpair):
    """Pre-registered probe SM (a documented limit): for a record whose CREATE steps are bound by two or more gold
    functions, ONE method carries all their annotations, takes the first step's parameters and holds the union of their
    bodies. PHP (the cart and total steps, one parameter each): it binds both steps, BV and CA as gold (CA matches per
    gold function, through the union). Java (open, no parameter, and total, one): condition 4 fails for the total step,
    BV 0. Both scorers agree; a record with one CREATE function gets no SM."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    p, scorer = pair
    assert "SM_stub_merge" not in po.probes(p.record, "php", "php_attribute", controls(p)["gold"][2], controls(p)["stub-v3"][2])
    assert po.stub_merge("php", p.record, controls(p)["gold"][2]) == ("", "fewer than 2 gold functions bind CREATE steps")
    q, labels_row, gold = merged_record(p, ["I open the shop", 'I add "apple" to the cart', 'I should see the total "5"'],
                                        ["reuse", "create", "create"], ["iOpenTheShop", "iAddToCart", "iShouldSeeTotal"])
    out = po.probes(q.record, "php", "php_attribute", gold, controls(q)["stub-v3"][2])["SM_stub_merge"]
    assert ("#[When('I add :product to the cart')]\n#[Then('I should see the total :total')]\n"
            "public function iAddToCart(string $product): void\n{\n    $this->cart->add($product);\n"
            "    $this->checkout->assertTotal($this->computeTotal($total));\n}") in out and "iShouldSeeTotal" not in out
    mine = compare(q, scorer, out, labels_row)
    assert (mine["bv"], scorer.ca(q.record, labels_row, mine)) == (1, (True, 1))
    p, scorer = jpair
    q, labels_row, gold = merged_record(p, ["I open the shop", 'I should see the total "5"'], ["create", "create"],
                                        ["iOpenTheShop", "iShouldSeeTotal"])
    out = po.probes(q.record, "java", "java_annotation", gold, controls(q)["stub-v3"][2])["SM_stub_merge"]
    assert ('@Given("I open the shop")\n@Then("I should see the total {string}")\npublic void iOpenTheShop() {\n'
            "    cart.open();\n    cart.assertTotal(computeTotal(total));\n}") in out and "iShouldSeeTotal" not in out
    mine = compare(q, scorer, out, labels_row)
    assert (mine["bv"], mine["bv_binding"], mine["bv_arity"]) == (0, 1, 0)


def test_stub_merge_merges_docblocks_and_attributes():
    """Probe SM's merged head: PHP docblocks become ONE docblock (Behat reads the one right before the method), the
    attribute groups stay, each once; C# attribute lists (one spanning two lines) and Java annotations
    are kept as written; a lambda step or an expression-bodied method is not merged (the reason is given)."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    rec = lambda path: {"step_fns": ["a", "x", "b"], "step_kinds": ["create", "reuse", "create"],
                        "create": [{"fn_id": "a", "path": path}, {"fn_id": "b", "path": path}]}
    php = ("/**\n * @Given I have :n apples\n */\npublic function a(int $n): void\n{\n    $this->basket->fill($n);\n}\n\n"
           "/** @When I eat :k */\n#[Then('done')]\npublic function b(int $k): void\n{\n    $this->basket->eat($k);\n}")
    out, why = po.stub_merge("php", rec("F.php"), {"F.php": php})
    assert why == "" and ("/**\n * @Given I have :n apples\n * @When I eat :k\n */\n#[Then('done')]\n"
                          "public function a(int $n): void\n{\n    $this->basket->fill($n);\n    $this->basket->eat($k);\n}") in out
    assert "// FILE: F.php\nuse Behat\\Step\\Then;\n\n/**" in out  # the imports its attributes need
    cs = ('[Given(@"I have (\\d+) apples")]\npublic void A(int n)\n{\n    _basket.Fill(n);\n}\n\n'
          '[When(@"I eat (\\d+)"),\n Scope(Tag = "x")]\n[Then(@"done")]\npublic void B(int k)\n{\n    _basket.Eat(k);\n}')
    out, why = po.stub_merge("csharp", rec("F.cs"), {"F.cs": cs})
    assert why == "" and ('[Given(@"I have (\\d+) apples")]\n[When(@"I eat (\\d+)"),\n Scope(Tag = "x")]\n[Then(@"done")]\n'
                          "public void A(int n)\n{\n    _basket.Fill(n);\n    _basket.Eat(k);\n}") in out and "void B(" not in out
    java = ('@Given("I have {int} apples")\npublic void a(int n) {\n    basket.fill(n);\n}\n\n'
            '@When("I eat {int}")\n@Then("done")\npublic void b(int k) {\n    basket.eat(k);\n}')
    out, why = po.stub_merge("java", rec("F.java"), {"F.java": java})
    assert why == "" and ('@Given("I have {int} apples")\n@When("I eat {int}")\n@Then("done")\npublic void a(int n) {\n'
                          "    basket.fill(n);\n    basket.eat(k);\n}") in out
    assert ("// FILE: F.java\nimport io.cucumber.java.en.Given;\nimport io.cucumber.java.en.Then;\n"
            "import io.cucumber.java.en.When;\n\n@Given") in out
    expr = cs.replace("public void B(int k)\n{\n    _basket.Eat(k);\n}", "public void B(int k) => _basket.Eat(k);")
    assert po.stub_merge("csharp", rec("F.cs"), {"F.cs": expr}) == ("", "a gold function has no block body")
    assert po.stub_merge("csharp", rec("F.cs"), {"F.cs": cs.split("\n\n")[0]}) == ("", "the gold items do not match the CREATE functions")
    lam = 'Given("I have {int} apples", (Integer n) -> {\n    basket.fill(n);\n});\n\nWhen("I eat {int}", (Integer k) -> {\n    basket.eat(k);\n});'
    assert po.stub_merge("java", rec("F.java"), {"F.java": lam}) == ("", "a gold function is not a method")
    assert po.stub_merge("ruby", rec("f.rb"), {"f.rb": ""}) == ("", "not PHP, Java or C#")
    assert po.stub_merge("php", {"steps": ["t"], "step_kinds": ["create"]}, {"F.php": php}) == ("", "the record has no step_fns")


def test_preexisting_shells_never_open_from_the_er_tree(jpair):
    """D11 (fixed 25/09/2026): a preexisting CREATE shell (its HEAD body, the answer, stays
    in the ER tree) never opens from the ER tree: (self, f) stays; an output that redefines a
    shell V(r) does not show replaces it and opens its own version. The shown-name rule comes
    first (decision of 25/09/2026): a shown shell stays (self, f) even when rewritten."""
    _, scorer = jpair
    fut = {("computeTotal", 1): [("computeTotal", JPATH, "private String computeTotal(String t) {\n    cart.total();\n}", False, True)]}
    resolve = scorer.resolver(JPATH, set(), {}, set(), set(), fut, shells={("computeTotal", 1)})
    assert resolve("computeTotal") is None
    own = {"computeTotal": [(JPATH, "private String computeTotal(String t) {\n    cart.open();\n}")]}
    resolve = scorer.resolver(JPATH, set(), own, set(), {("computeTotal", 1)}, fut, shells={("computeTotal", 1)})
    assert resolve("computeTotal") == [(JPATH, "private String computeTotal(String t) {\n    cart.open();\n}")]
    resolve = scorer.resolver(JPATH, {"computeTotal"}, own, set(), {("computeTotal", 1)}, fut, shells={("computeTotal", 1)})
    assert resolve("computeTotal") is None


def test_the_er_tree_fails_closed(pair):
    """A catalog row with no annotation, or a catalog file absent from HEAD, stops the independent scorer as it stops
    ertree.phantoms: never a silent skip."""
    p, scorer = pair
    fns = catalog()
    bare = {**fns, "x:1": {**next(iter(fns.values())), "fn_id": "x:1", "annotations": []}}
    gone = {**fns, "y:1": {**next(iter(fns.values())), "fn_id": "y:1", "path": "src/Missing.php"}}
    for bad, msg in ((bare, "no annotation"), (gone, "not in HEAD")):
        with pytest.raises(ValueError, match=msg):
            ind.phantom_ids("php", bad, scorer.git)


def test_jsonl_rows_read_as_3a_reads_them(tmp_path):
    """Both CLIs read JSONL exactly as 3A's symcache.jsonl_rows: split on "\\n" only from the decoded
    bytes, a final empty line dropped, every other line a JSON value (a blank line inside the file raises; a lone "\\r"
    is not a separator). score_independent refuses such a file with exit 3, as score_run does."""
    from v2tools.symcache import jsonl_rows
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    row = json.dumps({"record_id": "r", "text": "a\u2028b\x85c\x0bd\x1ce"}, ensure_ascii=False)
    cases = {"separators": row + "\n" + '{"a": 2}\n', "trailing": '{"a": 1}\n', "no_trailing": '{"a": 1}\n{"a": 2}',
             "blank_inside": '{"a": 1}\n\n{"a": 2}\n', "lone_cr": '{"a": 1}\r{"a": 2}\r'}
    for name, text in cases.items():
        f = tmp_path / f"{name}.jsonl"
        f.write_bytes(text.encode("utf-8"))
        results = []
        for reader in (jsonl_rows, ind.load_jsonl, po.load):
            try:
                results.append(("rows", reader(f)))
            except ValueError:
                results.append(("raises", None))
        assert results[0] == results[1] == results[2], name
        assert (results[0][0] == "raises") == (name in ("blank_inside", "lone_cr")), name
    assert results[0] == ("raises", None)
    ph2 = tmp_path / "phase2"
    (ph2 / "records").mkdir(parents=True)
    for f in ["records/test.jsonl", "records/dev.jsonl", "records/aux_test.jsonl", "labels.jsonl", "catalog.jsonl", "pairs.jsonl"]:
        (ph2 / f).write_text("")
    (ph2 / "forbidden.json").write_text("{}")
    res = run("scripts/score_independent.py", "--phase2", str(ph2), "--pairs", str(ph2 / "pairs.jsonl"), "--scores",
              str(ph2 / "labels.jsonl"), "--outputs", str(tmp_path / "blank_inside.jsonl"))
    assert res.returncode == 3 and "not JSONL" in res.stderr, res.stderr


def test_hidden_step_holders_of_probe_outputs(pair, jpair):
    """The hidden step holders: the gold methods hidden where no framework registers them, PA2 (PHP, in a `new class`
    assigned by probeInit), JB4 (Java, in an anonymous class local to probeHolder) and CS1 (C#, as attributed local
    functions of ProbeHolder). Both scorers agree on them; since a step annotation counts only before the `{` that opens
    its item's body, the binding fails: BV 0, CA 0."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    cs = po.probes({"steps": ["t 1"], "step_kinds": ["create"]}, "csharp", "csharp_attribute",
                   {"S.cs": '[Then(@"t (.*)")]\npublic void T(string x)\n{\n    _c.F(x);\n}'}, {})["CS1_local_function"]
    assert 'public void ProbeHolder(string x)\n{\n    [Then(@"t (.*)")]\n    void T(string x)\n    {\n        _c.F(x);\n    }\n}' in cs
    got = {}
    for (p, scorer), lang, style, name in ((pair, "php", "php_attribute", "PA2_method_anon_class"), (jpair, "java", "java_annotation", "JB4_method_local_anon")):
        labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
        out = po.probes(p.record, lang, style, extract(scorer.gold_text(p.record), p.targets).by_path(), controls(p)["stub-v3"][2])
        mine = compare(p, scorer, out[name])
        got[name] = (mine["bv_binding"], mine["bv"], scorer.ca(p.record, labels_row, mine)[1])
    assert got == {"PA2_method_anon_class": (0, 0, 0), "JB4_method_local_anon": (0, 0, 0)}


def test_root_object_members_match_symbols():
    """The same root-object members on both sides (Java and C#; none for PHP and Ruby)."""
    from v2tools.symbols import ROOT_MEMBERS
    assert ind.ROOT == ROOT_MEMBERS


def test_helper_call_expansion_on_both_fixtures(pair, jpair):
    """Helper-call expansion on the independent side: the gold's calls open through the future helpers (PHP
    computeTotal; Java computeTotal, then BaseSteps.baseNew); an output's helper or the output's replacement of a future
    helper contributes exactly its own calls; visible methods stay (self, f); a cycle and a call at depth 4 stay (self,
    f), each counted as a cut; condition 5 reads the expanded bodies but never a helper's name alone."""
    for (p, scorer), wrap, cases in ((pair, block, GPROBES), (jpair, jblock, [(c, b, a, n, 0) for c, b, a, n in JGPROBES])):
        labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
        for code, bv, ca_, n, cuts in cases:
            mine = scorer.score(p.record, labels_row, wrap(code))
            got = (mine["bv"], scorer.ca(p.record, labels_row, mine)[1], scorer.overlap(p.record, labels_row, mine)["n_calls"],
                   (mine["cuts"]["depth"], mine["cuts"]["cycle"]))
            assert got == (bv, ca_, n, cuts or (0, 0)), code
    for (p, scorer), want in ((pair, {("checkout", "assertTotal")}), (jpair, {("cart", "assertTotal"), ("store", "total")})):
        g = scorer.gold(p.record, {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]})
        assert set().union(*g["calls"].values()) == want


def test_visible_names_follow_rule_3_as_t_view_shows_them(pair, jpair):
    """The names the independent scorer takes as shown in ④(a) (the pre-t versions' names of the class
    chain, from symtimes) are those of state.t_view on both fixtures; the future methods are the ER tree's others,
    constructors included, an ancestor's private method not (BaseContext::secret)."""
    from v2tools.evaluate import future_methods, shown_names
    from v2tools.state import t_view
    for (p, scorer), path, future in ((pair, SHOP, {"computeTotal"}), (jpair, JPATH, {"computeTotal", "baseNew", "ShopSteps"})):
        r = p.record
        gone = {c["fn_id"] for c in r["create"] if not c["preexisting"]}
        shown = {n for sh in t_view(p.repo.ctx, path, r["t_vis"], gone) for n in sh.member.names}
        assert scorer.visible(path, r, p.er_files) == shown
        assert {e[0] for found in scorer.future(path, r, p.er_files).values() for e in found} == future
        # agreement with the main scorer's shown and future names (evaluate.shown_names, evaluate.future_methods)
        assert scorer.visible(path, r, p.er_files) == shown_names(p, path)
        assert {k: (found[0][0], found[0][1]) for k, found in scorer.future(path, r, p.er_files).items()} == \
            {k: (f.name, f.origin) for k, f in future_methods(p)[path].items()}


def test_process_exit_ends_the_flow_on_the_fixture(pair):
    """Probe X7: the calls after `exit;` or `die(…);` are dead for CA; an exit inside an expression is not."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    for stmt, want in (("exit;", 0), ("die('stop');", 0), ("$x = $total ?: exit(1);", 1)):
        mine = scorer.score(p.record, labels_row, block(HEAD + f"    {stmt}\n    $this->checkout->assertTotal($total);\n}}"))
        assert mine["bv"] == 1 and scorer.ca(p.record, labels_row, mine) == (True, want), stmt


def test_process_exit_callee_ignores_a_comment(jpair):
    """A comment inside a process-exit statement's callee (`System./*c*/exit(1)`) never hides it, like X8 (JPROBES)
    without the comment; independent already strips comments before drop_dead (clean), so the comment was never part of
    the callee text it matches. Both scorers agree (compare)."""
    p, scorer = jpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    code = JHEAD + "    System./*c*/exit(0);\n    cart.assertTotal(total);\n}"
    mine = compare(p, scorer, jblock(code))
    assert (mine["bv"], scorer.ca(p.record, labels_row, mine)[1]) == (1, 0)


def test_independent_scorer_agrees_on_controls(pair):
    p, scorer = pair
    for name, (s, _, blocks) in controls(p).items():
        text = "\n\n".join(f"{TICKS}php\n// FILE: {path}\n{code}\n{TICKS}" for path, code in blocks.items())
        compare(p, scorer, text)


def test_gold_text_matches_compose(pair):
    from v2tools.compose import gold_answer
    p, scorer = pair
    assert scorer.gold_text(p.record) == gold_answer(p.record, p.repo.fns)


# Regex forms Python's `re` cannot read (a .NET/Ruby/PCRE named group, Unicode properties) or reads its own way (a
# comment, a `]` first in a class); checked since 26/09/2026. The dataset holds none of them.
REJECTED_FORMS = {"quote_named": r"^I have (?'n'\d+) apples$", "unicode_prop": r"^I have (\p{N}+) apples$",
                  "unicode_prop_negated": r"^I have (\P{L}+) apples$", "unicode_short": r"^I have (\pN+) apples$"}
ACCEPTED_FORMS = {"comment": (r"^I have (?#count)(\d+) apples$", "I have 3 apples"),
                  "bracket_first": (r"^I have ([]a]+) apples$", "I have ]a] apples"),
                  "bracket_negated_first": (r"^I have ([^]a]+) apples$", "I have 3 apples")}


def test_the_value_check_survives_a_pattern_python_cannot_read():
    """A latent bug: the hard-coded value check compiles a regex step pattern with patterns.compile_pattern, which
    returns None for a pattern `re` rejects; like argcheck.hardcoded, a None reports hard-coded (no TypeError).
    Unreachable in scoring (such a pattern never binds), pinned here."""
    from types import SimpleNamespace
    from v2tools.argcheck import hardcoded
    from v2tools.patterns import compile_pattern
    cases = [(pat, "I have 3 apples") for pat in REJECTED_FORMS.values()] + list(ACCEPTED_FORMS.values())
    for kind in ("regex", "regex_full"):
        for pat, step in cases:
            assert (compile_pattern(kind, pat) is None) == (pat in REJECTED_FORMS.values()), pat
            assert ind.Scorer._hardcoded(SimpleNamespace(kind=kind, pattern=pat), step) == hardcoded(kind, pat, step), (kind, pat)


FORMS = {  # the same regex forms, written for the fixtures' step 'I should see the total "5"'
    "quote_named": r'^I should see the total "(?\'total\'[^"]+)"$',
    "unicode_prop": r'^I should see the total "(\p{N}+)"$',
    "unicode_short": r'^I should see the total "(\pN+)"$',
    "comment": r'^I should see the total "(?#amount)([^"]+)"$',
    "bracket_first": r'^I should see the total "([^]"]+)"$',
}
FORM_PHP = "/**\n * @Then /{}/\n */\npublic function t(string $total): void\n{{\n    $this->checkout->assertTotal($total);\n}}"
FORM_JAVA = '@Then("{}")\npublic void t(String total) {{\n    cart.assertTotal(total);\n}}'


def test_regex_forms_python_cannot_read_on_both_fixtures(pair, jpair):
    """The five regex forms × PHP/Java, scored by both scorers (compare: BV, CA, the six conditions, the row's
    static columns and the cuts agree; no exception). A form `re` rejects never binds (bind.Library skips it): condition
    2 fails and N is 0; a comment or a `]` first in a class compiles and binds (Python reads them as PCRE does)."""
    from v2tools.patterns import compile_pattern
    from v2tools.stepdefs import extract as defs
    got = {}
    for form, pat in FORMS.items():
        for lang, (p, scorer), text in (("php", pair, block(FORM_PHP.format(pat))),
                                         ("java", jpair, jblock(FORM_JAVA.format(pat.replace("\\", "\\\\").replace('"', '\\"'))))):
            d = defs(p.targets[0], text.split("\n", 2)[2].rsplit("\n", 1)[0], "php_docblock" if lang == "php" else "java_annotation")
            assert d and d[0].pattern == pat, (form, lang, d)
            mine = compare(p, scorer, text)
            rejected = compile_pattern(d[0].kind, pat) is None
            got[(form, lang)] = (rejected, ind.regex_args(pat, lang), mine["bv_binding"], mine["bv"])
    assert got == {(f, lang): ((True, 0, 0, 0) if f in ("quote_named", "unicode_prop", "unicode_short") else (False, 1, 1, 1))
                   for f in FORMS for lang in ("php", "java")}


# Contrived shapes where main and independent could disagree (main is right): each shape, with its control,
# through compare() on the fixture of its language. (fixture, case) -> output text.
SHAPE_PHP = "#[Then('I should see the total :total')]\npublic function iShouldSeeTotal(string $total): void\n{{\n{}\n}}"
SHAPE_JAVA = '@Then("I should see the total {{string}}")\npublic void iShouldSeeTotal(String total) {{\n    cart.assertTotal(computeTotal(total));\n{}\n}}'
SHAPE_CASES = {
    # `$self::` / `$static::` is a variable's class (no member); `\self::` is no relative scope
    ("pair", "D3_dollar_self"): block(SHAPE_PHP.format("    $self = $this;\n    $self::helperOld();")),
    ("pair", "D3_dollar_static"): block(SHAPE_PHP.format("    $static = $this;\n    $static::helperOld();")),
    ("pair", "D3_backslash_self"): block(SHAPE_PHP.format("    \\self::helperOld();")),
    ("pair", "D3c_static_keyword"): block(SHAPE_PHP.format("    static::helperOld();")),
    # a class call right after `=>`, `?:`, `case 1:` or a label counts
    ("pair", "D4_arrow_no_space"): block(SHAPE_PHP.format("    $y = [1 =>BaseContext::helperOld()];")),
    ("pair", "D4_elvis_no_space"): block(SHAPE_PHP.format("    $y = $total ?:BaseContext::helperOld();")),
    ("pair", "D4_case_no_space"): block(SHAPE_PHP.format("    switch (1) { case 1:BaseContext::helperOld(); }")),
    ("pair", "D4_label_no_space"): block(SHAPE_PHP.format("    lbl:BaseContext::helperOld();")),
    ("pair", "D4c_arrow_space"): block(SHAPE_PHP.format("    $y = [1 => BaseContext::helperOld()];")),
    # inline HTML (`?>` ... `<?php`, `<?`, `<?PHP`) is no code; an echo tag `<?= … ?>` is; `?>` ends a // comment
    ("pair", "IC_inline_html"): block(SHAPE_PHP.format("    ?>\n    $this->checkout->assertTotal($this->computeTotal($total));\n    <?php")),
    ("pair", "IC_inline_html_short_tag"): block(SHAPE_PHP.format("    ?>\n    $this->checkout->assertTotal($total);\n    <?\n    $this->cart->add($total);")),
    ("pair", "IC_inline_html_upper_tag"): block(SHAPE_PHP.format("    ?>\n    $this->checkout->assertTotal($total);\n    <?PHP\n    $this->cart->add($total);")),
    ("pair", "IC_echo_tag"): block(SHAPE_PHP.format("    ?>\n    <?= $this->checkout->assertTotal($total) ?>\n    $this->cart->add($total);\n    <?php")),
    ("pair", "IC_comment_closes_php"): block(SHAPE_PHP.format("    // done ?>\n    $this->checkout->assertTotal($total);\n    <?php")),
    # a group use `use A\{B};` imports nothing, as 3A's outline reads it (main)
    ("ppair", "D2_group_use"): pblock("use App\\Util\\{Language};\n\n" + SHAPE_PHP.format("    Language::getIdByLocale($total);")),
    ("ppair", "D2c_plain_use"): pblock("use App\\Util\\Language;\n\n" + SHAPE_PHP.format("    Language::getIdByLocale($total);")),
    # Java pattern variables (a switch type pattern, a record pattern) are locals
    ("jpair", "D1_switch_type_pattern"): jblock(SHAPE_JAVA.format("    Object o = total;\n    switch (o) { case String s -> s.length(); default -> {} }")),
    ("jpair", "D1_record_pattern"): jblock(SHAPE_JAVA.format("    Object o = total;\n    if (o instanceof java.util.Map.Entry<?, ?>(var k, var v)) { k.hashCode(); }")),
    ("jpair", "D1u_instanceof_pattern"): jblock(SHAPE_JAVA.format("    Object o = total;\n    if (o instanceof String s) { s.length(); }")),
    ("jpair", "D1c_no_pattern"): jblock(SHAPE_JAVA.format("    Object o = total;\n    switch (o) { case String s -> {} default -> {} }")),
    # an import line with `*` anywhere but at the end is no import (main: a raw item, syntax 0)
    ("jpair", "D5_import_star_mid"): jblock("import java.*.util;\n\n" + JGOLD),
    ("jpair", "D5_import_star_only"): jblock("import *;\n\n" + JGOLD),
    ("jpair", "D5c_import_star_end"): jblock("import java.util.*;\n\n" + JGOLD),
}


@pytest.mark.parametrize("case", sorted(SHAPE_CASES), ids=lambda c: c[1])
def test_contrived_shapes_agree_in_both_scorers(case, request):
    """Both scorers agree on every contrived shape (BV, CA, the six conditions, the row's static columns, the cuts)."""
    p, scorer = request.getfixturevalue(case[0])
    compare(p, scorer, SHAPE_CASES[case])


def test_csharp_raw_strings_are_no_code():
    """A C# 11 raw string literal (three or more quotes, also interpolated with one or more `$`, on one line or several,
    quotes inside) is no code, as symbols' tree (raw_string_literal, interpolated_string_expression in DEAD_NODES) reads
    it: the references agree with symbols.jvm_refs. No C# fixture pair exists; the column it changes, open_resolution,
    is descriptive."""
    from v2tools.symbols import jvm_refs
    head, tail = "public void T(string x)\n{\n    var s = ", ";\n    _c.F(x);\n}"
    for lit in ('"""\n        Ghost.Call("q"); _ghost.X();\n        """', '$"""\n        {Ghost.Call()} "quoted" {_ghost.X()}\n        """',
                '$$""""\n        {{Ghost.Call()}} """ still inside\n        """"', '"""one line " Ghost.Call() """'):
        r = jvm_refs("csharp", head + lit + tail)
        assert ind.member_uses("csharp", head + lit + tail) == (set(r.members), set(r.calls), set(r.classes)) == ({"_c"}, set(), set()), lit


def test_condition_4_table_and_optional_rules_on_the_fixtures(pair, jlpair):
    """As main's argcheck: (1) a PHP parameter takes a table when an alternative of its type (split on | & ? ( ) and
    spaces) ends, after its last backslash, in TableNode or PyStringNode in any case; (2) a parameter is optional only
    when its tree gives a default, never for a `=` inside an attribute; (3) a java8 lambda counts the parameters of its
    tree (generic commas are no parameters). Both scorers agree; bv_arity pinned."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    head = "#[Then('I should see the total :total')]\npublic function t("
    tail = "): void\n{\n    $this->checkout->assertTotal($total);\n}"
    table = prepare(dict(p.record, scenario=p.record["scenario"] + "\n    | a |\n    | 1 |"), p.repo, labels_row)
    on_table = {"TableNode&\\Countable $total": 1, "(TableNode&\\Countable)|null $total": 1, "tablenode $total": 1,
                "\\behat\\gherkin\\node\\PYSTRINGNODE $total": 1, "\\Behat\\TableNode\\Other $total": 0,
                "\\PrestaShop\\TableNode $total": 1, "?TableNode $total": 1, "string $total": 0}
    got = {ps: compare(table, scorer, block(head + ps + tail))["bv_arity"] for ps in on_table}
    assert got == on_table
    plain = {"string $total, #[Foo(a: 'x=y')] string $other": 0, "string $total, string $other = 'x=y'": 1,
             "string $total, #[Foo(a: 'x=y')] string $other = 'z'": 1}
    got = {ps: compare(p, scorer, block(head + ps + tail))["bv_arity"] for ps in plain}
    assert got == plain
    p, scorer = jlpair
    lam = 'Then("I should see the total {{string}}", {} -> {{\n    cart.assertTotal(total.toString());\n}});'
    lambdas = {"(java.util.Map<String, String> total)": 1, "(String a, java.util.Map<String, java.util.List<Integer>> total)": 0,
               "total": 1, "(String total)": 1, "(a, total)": 0}
    got = {ps: compare(p, scorer, jblock(lam.format(ps)))["bv_arity"] for ps in lambdas}
    assert got == lambdas


def test_table_types_and_lambda_parameters_match_argcheck():
    """The two readings behind these condition 4 rules (tables, java8 lambdas), on the examples given with those rules,
    equal argcheck's takes_table and lambda_arity."""
    from v2tools.argcheck import lambda_arity, takes_table
    for typ in ("TableNode&\\Countable", "(TableNode&\\Countable)|null", "tablenode", "\\behat\\gherkin\\node\\PYSTRINGNODE",
                "\\Behat\\TableNode\\Other", "\\PrestaShop\\TableNode", "?TableNode", "string", "TableNode|string", "Table"):
        assert ind.php_takes_table(typ) == takes_table(typ), typ
    for params, n in (("(java.util.Map<String, String> m)", 1), ("(String a, Map<String, List<Integer>> b)", 2), ("(a, b)", 2),
                      ("x", 1), ("()", 0), ("(String... rest)", 1), ("(final int a, int[] b)", 2)):
        stmt = f'Given("x", {params} -> {{\n}});'
        assert ind.lambda_params(stmt) == lambda_arity(stmt) == n, params


# Mutants of this scorer that survived the suite, each killed by a shape where
# both scorers agree (compare) and a pinned value.
def test_two_overloads_on_the_expansion_path_count_one_cycle_each(jpair):
    """Mutants that count a cycle with len, not bool, or leave the root out of the path: probe() and probe(String)
    call each other; each cut call is ONE cycle."""
    p, scorer = jpair
    code = ('@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(total);\n'
            '    probe(total);\n}\n\nprivate void probe(String total) {\n    cart.open();\n    probe();\n}\n\n'
            'private void probe() {\n    cart.total();\n    probe("x");\n}')
    assert compare(p, scorer, jblock(code))["cuts"] == {"depth": 0, "cycle": 4}


def test_a_step_calling_itself_keeps_its_self_call(jpair):
    """Mutants that never give the root a body of its own, or leave it out of the path: a step that calls itself and a
    chain of four helpers: the self call is a cycle at the root, kept as (self, iShouldSeeTotal); h4 lies beyond depth 3
    and stays (self, h4)."""
    p, scorer = jpair
    code = ('@Then("I should see the total {string}")\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(total);\n'
            '    iShouldSeeTotal(total);\n    h1();\n}\n\nprivate void h1() {\n    cart.open();\n    h2();\n}\n\n'
            'private void h2() {\n    cart.total();\n    h3();\n}\n\nprivate void h3() {\n    h4();\n}\n\nprivate void h4() {\n    cart.open();\n}')
    mine = compare(p, scorer, jblock(code))
    assert (mine["n_calls"], mine["cuts"]) == (5, {"depth": 1, "cycle": 1}), (mine["n_calls"], mine["cuts"])


def test_a_nullsafe_this_self_call_counts(pair):
    """A mutant read `$this?->f(` as no self call: the output's call $this?->iAddToCart() is one of its calls."""
    p, scorer = pair
    mine = compare(p, scorer, block(php_step(["$this->checkout->assertTotal($total);", "$this?->iAddToCart($total);"])))
    assert mine["n_calls"] == 2


def test_a_new_file_shows_its_ancestors_as_main_does(tmp_path):
    """Mutants of the constructor, before-t private and HEAD private filters of a new file's view: a new file using an
    own trait, a parent with an old trait, members that changed visibility between t and HEAD; the visible names and
    the future methods equal main's shown_names and future_methods."""
    import tests.test_evaluate as te
    from tests.phase3_fixture import SHOP_2024
    from v2tools.evaluate import future_methods, shown_names
    own0 = te.trait_text("OwnTrait", te.member("private", "ownOld", "o0"))
    own1 = te.trait_text("OwnTrait", te.member("private", "ownOld", "o0"), te.member("private", "ownNew", "o1"))
    olds = (te.member("protected", "parOld", "p0"), te.member("private", "parOldPriv", "p3"))
    par0 = te.trait_text("ParentTrait", *olds)
    par1 = te.trait_text("ParentTrait", *olds, te.member("private", "parNew", "p1"), te.member("protected", "parNewProt", "p2"))
    flips = (te.member("private", "wasPrivate", "w"), te.member("protected", "nowPrivate", "n"))
    flipped = (te.member("protected", "wasPrivate", "w"), te.member("private", "nowPrivate", "n"))
    p = te.twin_prep(tmp_path, [(T2020, {te.BASE: te.base_text(*flips), te.OWN: own0, te.PAR: par0}),
                                (T2024, {te.BASE: te.base_text(te.member("protected", "baseNew", "b1"), *flipped), te.OWN: own1,
                                         te.PAR: par1, te.SHOP: te.with_own_trait(SHOP_2024)})])
    scorer = ind.Scorer("Acme/Shop", "php", ["php_attribute", "php_docblock"], p.repo.fns,
                        ind.Git(p.repo.ctx.git_dir, p.record["commit"]), p.repo.suite_steps)
    tree, _ = ind.er_tree(p.record, scorer.fns, scorer.ghosts, scorer.git)
    shown = shown_names(p, te.SHOP)
    # wasPrivate: private before t; nowPrivate: private at HEAD; parOldPriv: private in the parent's trait
    assert scorer.visible(te.SHOP, p.record, tree) == shown == {"helperOld", "ownOld", "parOld", "sharedStorage"}, shown
    assert {e[0].casefold() for found in scorer.future(te.SHOP, p.record, tree).values() for e in found} == \
        {f.name.casefold() for f in future_methods(p)[te.SHOP].values()}


def test_csharp_access_reads_the_first_class_like_type_as_main_does():
    """The type whose members the C# private default reads is the file's first class-like declaration
    (insert.first_class: class, interface, enum, record, trait; a struct is none), so an interface after a struct keeps
    its members public, as evaluate.csharp_interface says."""
    from v2tools.evaluate import csharp_interface
    text = "namespace N\n{\n    public struct S\n    {\n    }\n\n    public interface IBase\n    {\n        void Helper();\n    }\n}\n"
    m = next(x for x in ind.class_methods("csharp", text) if x.name == "Helper")
    assert csharp_interface(text) and not ind.csharp_private("csharp", text, m)
    alone = "namespace N\n{\n    public struct S\n    {\n        void Helper()\n        {\n        }\n    }\n}\n"
    assert not csharp_interface(alone) and ind.class_methods("csharp", alone) == []


def shell_pair(tmp_path, body: str = "$this->cart->add($product);"):
    """(Prepared, Scorer, labels row) of test_evaluate.shell_prep: the record's answer only adds `#[When('I put :product
    in the cart')]` to iAddToCart, a preexisting shell that ④(a) shows."""
    from tests.test_evaluate import shell_prep
    p = shell_prep(tmp_path, body)
    scorer = ind.Scorer("Acme/Shop", "php", ["php_attribute", "php_docblock"], p.repo.fns,
                        ind.Git(p.repo.ctx.git_dir, p.record["commit"]), p.repo.suite_steps)
    ids = dict(p.repo.suite_steps["s"])
    return p, scorer, {"step_def_ids": [list(ids[s]) for s in p.record["steps"]]}


def test_a_step_calling_a_visible_preexisting_shell_reaches_the_gold(tmp_path):
    """Delegation to a preexisting shell (D11 (e), fixed 26/09/2026; main's evaluate.delegated): a gold binder that is a
    preexisting shell of the record, whose name ④(a) shows (PHP in any case) and whose call set is not empty, also holds
    ("self", its name) in the CA and call_precision reference; an output calling the shell (in any case, or through
    self::) reaches CA 1 and precision 1. The gold answer scored as an output (the gold control) keeps its own call set.
    Both scorers agree (compare)."""
    from tests.test_evaluate import PUT_STEP
    p, scorer, labels_row = shell_pair(tmp_path)
    for call, want in (("$this->iAddToCart($product);", 1), ("$this->IADDTOCART($product);", 1), ("self::iAddToCart($product);", 1),
                       ("$this->iOpenTheShop();", 0)):
        mine = compare(p, scorer, block(PUT_STEP.format(call)), labels_row)
        assert (mine["bv"], scorer.ca(p.record, labels_row, mine)) == (1, (True, want)), call
        assert mine["call_precision"] == (1.0 if want else 0.0), call
    raw = compare(p, scorer, scorer.gold_text(p.record), labels_row)
    assert raw["calls"][(p.targets[0], "iaddtocart")] == {("cart", "add")}
    assert scorer.ca(p.record, labels_row, raw) == (True, 1) and (raw["n_calls"], raw["call_precision"]) == (1, 1.0)
    assert scorer.reference(p.record, labels_row)["calls"][(p.targets[0], "iaddtocart")] == {("cart", "add"), ("self", "iAddToCart")}


def test_a_visible_shell_with_no_project_call_gains_nothing(tmp_path):
    """The same delegation (D11 (e)) on an empty gold call set: nothing is added, so CA stays undefined (as main)."""
    from tests.test_evaluate import PUT_STEP
    p, scorer, labels_row = shell_pair(tmp_path, "$this->cart;")
    mine = compare(p, scorer, block(PUT_STEP.format("$this->iAddToCart($product);")), labels_row)
    assert scorer.ca(p.record, labels_row, mine) == (False, 0)


def test_php_open_and_close_tags_are_no_code(pair, ppair):
    """A PHP block's own leading `<?php` and trailing `?>` are stripped (fixed 26/09/2026) before anything reads it
    (units, items_of, members_src, the import lines scope_of joins), so a model that wraps its answer as a full `<?php
    ... ?>` file scores like one that does not. Both scorers agree (compare); an import line right after the tag still
    joins the file's imports (ppair: Language resolves through it). Aligned with main's `insert.php_open_tag` (`PHP_OPEN
    = \\A\\s*<?php…`), a blank line (or several) BEFORE the `<?php` tag is part of the stripped open tag too, not left
    as leading dead code (the fence's FILE-line splitter keeps a block's own leading blank lines; only trailing ones are
    ever rstripped away, so this direction is the one compare() can reach end to end)."""
    p, scorer = pair
    for code in ("<?php\n\n" + GOLD, GOLD + "\n?>", "<?PHP \n" + GOLD, "<?php\n\n" + GOLD + "\n\n?>\n",
                 "<?php\n\n" + php_step(["$this->iAddToCart($total);", "$this->checkout->assertTotal($total);"]),
                 "\n<?php\n\n" + GOLD,  # one blank line before the tag
                 "\n\n<?PHP\n\n" + GOLD,  # two blank lines before the tag, mixed case
                 "  \n<?php\n\n" + GOLD):  # a blank line that itself carries trailing spaces before the tag
        mine = compare(p, scorer, block(code))
        assert mine["bv_syntax"] == 1, code
    p, scorer = ppair
    imported = "<?php\n\nuse App\\Util\\Language;\n\n" + PGOLD + \
        "\n\nprotected function langId(string $locale): int\n{\n    return Language::getIdByLocale($locale);\n}"
    mine = compare(p, scorer, pblock(imported))
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    assert (mine["bv"], scorer.ca(p.record, labels_row, mine)[1]) == (1, 1)


def test_php_block_tags_matches_main_on_blank_lines_around_the_tags():
    """The tag rule of the design specification (e549ca5): "at the start" of a block is after ANY
    leading whitespace, blank lines included, and "at the end" is before ANY trailing whitespace, blank lines included;
    both scorers apply exactly this rule. `php_block_tags` (PHP_BLOCK_OPEN, PHP_BLOCK_CLOSE) must byte-for-byte match
    main's `insert.php_open_tag` on both directions and on both together. The fenced compare() pipeline can only ever
    reach the leading-blank-line direction (blocks_of rstrips a block's own trailing whitespace before php_block_tags
    ever runs, so a trailing blank line after `?>` never survives to reach it there); this direct parity check pins
    the close-tag direction, and it runs against v2tools.insert.php_open_tag itself, not a copy of it."""
    from v2tools.insert import php_open_tag
    cases = ["<?php\n" + GOLD, "<?php\n\n" + GOLD, "\n<?php\n\n" + GOLD, "\n\n<?PHP\n\n" + GOLD, "  \n<?php\n\n" + GOLD,
             GOLD + "\n?>", GOLD + "\n\n?>", GOLD + "\n\n?>\n\n", GOLD + "\n\n?>\n\n\n", GOLD + "\n?>  \n\n",
             "\n<?php\n\n" + GOLD + "\n\n?>\n\n",  # both directions together
             "  \n\n<?PHP  \n\n" + GOLD + "\n\n?>   \n\n\n",  # both, with extra whitespace variety
             GOLD, "not php at all"]
    for code in cases:
        assert ind.php_block_tags(code) == php_open_tag(code)[0], code



def test_two_fences_of_one_file_score_as_one_whatever_their_php_tags(pair):
    """The tag rule of §1.4 per block (design specification at c6999dc, fixed 27/09/2026): both scorers drop a PHP
    block's own tags before joining the blocks of one file (blocks_of, extract.by_path), so the gold and a helper
    written as two fences, each opened with `<?php`, each closed with `?>`, or one of each, score as the same code in
    one fence (before: BV 0, CA 0 in both scorers, which agreed)."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    helper = "private function computeTotal(string $total): string\n{\n    return $total;\n}"
    view = lambda r: (r["bv"], [r[f"bv_{c}"] for c in CONDITIONS], scorer.ca(p.record, labels_row, r))
    one = compare(p, scorer, block(GOLD + "\n\n" + helper))
    assert view(one)[0] == 1 and view(one)[2] == (True, 1)
    for a, b in (("<?php\n\n" + GOLD, "<?php\n\n" + helper), (GOLD + "\n?>", helper + "\n?>"),
                 ("<?php\n" + GOLD, helper), (GOLD, "<?php\n" + helper + "\n?>")):
        assert view(compare(p, scorer, block(a) + "\n" + block(b))) == view(one), (a, b)


RBPATH = "features/step_definitions/shop_steps.rb"
RB_2020 = "Given(/^I open the shop$/) do\n  open_shop\nend\n"
RB_2024 = RB_2020 + '\nThen(/^I should see the total "([^"]*)"$/) do |total|\n  check_total(total)\nend\n'
RB_HELPERS = "module ShopHelpers\n  def open_shop\n  end\n\n  def check_total(total)\n  end\nend\n\nWorld(ShopHelpers)\n"


@pytest.fixture()
def rbpair(tmp_path):
    """(Prepared, independent Scorer) of a Ruby twin (renalware's form): the REUSE step "I open the shop" (2020) and
    the CREATE answer `Then(/^I should see the total "([^"]*)"$/) do |total|` (2024), calling the helper module's
    check_total (both scorers are pinned on the two Ruby annotation forms)."""
    work = tmp_path / "rbwork"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in [(T2020, {"features/support/shop.rb": RB_HELPERS, RBPATH: RB_2020}), (T2024, {RBPATH: RB_2024})]:
        for fp, text in files.items():
            (work / fp).parent.mkdir(parents=True, exist_ok=True)
            (work / fp).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp_path / "rb.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    head = _git(bare, "rev-parse", "HEAD").strip()
    lines = RB_2024.split("\n")

    def fn(first, end, keyword, pattern, time):
        return {"repo": "Acme/Ruby", "fn_id": f"{RBPATH}:{first}", "path": RBPATH, "first_line": first, "body_end": end,
                "code": "\n".join(lines[first - 1:end]),
                "annotations": [{"def_id": f"{RBPATH}:{first}", "line": first, "keyword": keyword, "pattern": pattern,
                                 "time": time}],
                "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}
    opn, tot = fn(1, 3, "Given", "^I open the shop$", T2020), fn(5, 7, "Then", '^I should see the total "([^"]*)"$', T2024)
    cat = {f["fn_id"]: f for f in (opn, tot)}
    r = {"id": "Acme/Ruby|features/total.feature|3|s", "repo": "Acme/Ruby", "commit": head, "language": "ruby",
         "framework": "Cucumber", "scoping": None, "suite": "s", "tags": [], "partition": "test", "subsets": [],
         "exec_core": False, "neighbours": None, "feature_path": "features/total.feature", "scenario_line": 3,
         "scenario_name": "Total", "scenario_time": T2024, "t_vis": T2024, "background": "",
         "scenario": "Scenario: Total\n  Given I open the shop\n  Then I should see the total \"5\"",
         "steps": ["I open the shop", "I should see the total \"5\""], "step_kinds": ["reuse", "create"],
         "step_fns": [opn["fn_id"], tot["fn_id"]], "reuse_fns": [], "snapshot_size": 2,
         "create": [{"fn_id": tot["fn_id"], "path": RBPATH, "code": tot["code"], "preexisting": False,
                     "post_cutoff_code": True}], "create_def_ids": [tot["annotations"][0]["def_id"]]}
    d_open, d_tot = opn["annotations"][0]["def_id"], tot["annotations"][0]["def_id"]
    suite = {"s": [("I open the shop", (d_open,)), ('I should see the total "5"', (d_tot,))]}
    rp = Repo("Acme/Ruby", "ruby", ["ruby_block"], cat, RepoContext("Acme/Ruby", "ruby", bare, head, cat), frozenset(),
              suite)
    return (prepare(r, rp, {"step_def_ids": [[d_open], [d_tot]]}),
            ind.Scorer("Acme/Ruby", "ruby", ["ruby_block"], cat, ind.Git(bare, head), suite))


def test_either_ruby_annotation_form_gives_the_same_row(rbpair):
    """The independent scorer is pinned on the Ruby forms too (design specification at 081e985). `Then(/re/) do`, `Then
    /re/ do`, `Then(%r{re}) do` and `Then %r{re} do` of the same step give the same row in both scorers (compare()
    checks each against evaluate), BV 1 and CA 1, but for rouge_l and exact, which read the text; a wrong call
    (open_shop) gives CA 0 in both, so the pin is not blind."""
    p, scorer = rbpair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    ticks = "`" * 3
    fenced = lambda code: f"{ticks}ruby\n# FILE: {RBPATH}\n{code}\n{ticks}"
    rows_ = []
    for head in ('Then(/^I should see the total "([^"]*)"$/) do |total|', 'Then /^I should see the total "([^"]*)"$/ do |total|',
                 'Then(%r{^I should see the total "([^"]*)"$}) do |total|', 'Then %r{^I should see the total "([^"]*)"$} do |total|'):
        x = compare(p, scorer, fenced(head + "\n  check_total(total)\nend"), labels_row)
        x["ca_defined"], x["ca"] = scorer.ca(p.record, labels_row, x)
        assert x["bv"] == x["ca"] == 1, head
        rows_.append({k: v for k, v in x.items() if k not in ("rouge_l", "exact")})
    assert all(r == rows_[0] for r in rows_)
    wrong = compare(p, scorer, fenced('Then /^I should see the total "([^"]*)"$/ do |total|\n  open_shop\nend'),
                    labels_row)
    assert scorer.ca(p.record, labels_row, wrong) == (True, 0)

def test_kept_head_annotations_go_before_the_whole_method_head(pair):
    """Fixed 26/09/2026: a kept HEAD annotation line of a rewritten shell (iAddToCart, whose own `#[When(...)]` is
    dropped by the CREATE answer and must be kept) goes before the whole attribute list when the output's own attribute
    and signature share one line (never between the signature and the body), and before the signature line, never inside
    it, when the output's own attribute spans several lines. Pinned against with_old/with_annotations directly
    (test_items_and_annotations_follow_the_insertion_rule only exercises the common one-attribute-per-line shape); a
    full compare() of this shape is test_php_kept_annotations_compare_by_pattern below (iAddToCart is not a replaceable
    shell of the plain `pair` fixture, so compare() here would only agree on a collision, exercising nothing)."""
    from v2tools.insert import with_annotations
    old = ("#[When('I add :product to the cart')]\npublic function iAddToCart(string $product): void\n{\n"
           "    $this->cart->add($product);\n}")
    missing = ["#[When('I add :product to the cart')]"]
    same_line = "#[Then('I should see the total :total')] public function iAddToCart(string $total): void\n{\n}"
    multi_line = ("#[Then(\n    'I should see the total :total'\n)]\n"
                  "public function iAddToCart(string $total): void\n{\n}")
    for code in (same_line, multi_line):
        assert ind.with_old("php", code, old) == with_annotations("php", code, missing), code


def test_php_kept_annotations_compare_by_pattern(tmp_path):
    """Behat 4.0.0-alpha1 registers attributes only and refuses a pattern registered twice (rule fixed 26/09/2026):
    rewriting the shell iAddToCart (shell_pair: its own kept annotation is `#[When('I add :product to the cart')]`), the
    output's OWN duplicate of that pattern, in ANY form or keyword (docblock, a different keyword, unescaped quoting),
    is matched by PATTERN, not by normalised line text: the HEAD line is kept as HEAD wrote it and the output's
    duplicate is dropped, so no pattern is registered twice; the output's own CREATE pattern ('I put :product in the
    cart') is never touched. Both scorers agree (compare)."""
    p, scorer, labels_row = shell_pair(tmp_path)
    head = "public function iAddToCart(string $product): void\n{\n    $this->cart->add($product);\n}"
    dup_keyword = "#[When('I put :product in the cart')]\n#[Given('I add :product to the cart')]\n" + head
    dup_quote = '#[When(\'I put :product in the cart\')]\n#[When("I add :product to the cart")]\n' + head
    dup_docblock = ("#[When('I put :product in the cart')]\n/**\n * @Given I add :product to the cart\n */\n" + head)
    for code in (dup_keyword, dup_quote, dup_docblock):
        mine = compare(p, scorer, block(code), labels_row)
        assert mine["bv"] == 1, code


def test_php_scoped_calls_resolve_and_ground_in_any_case(pair):
    """Fixed 26/09/2026: a PHP scoped METHOD call (self::x(, static::x(, parent::x() also resolves
    (condition 6) and grounds (condition 5) case-insensitively on a method of the ER-tree chain,
    like `$this->x(` already does; a property or constant read through self::/static:: (no `(`)
    stays out of `calls` and so stays case-sensitive. Both scorers agree (compare); computeTotal
    is a future private helper of the fixture, called here through `self::` in the wrong case."""
    p, scorer = pair
    code = ("#[Then('I should see the total :total')]\npublic function t(string $total): void\n{\n"
            "    self::COMPUTETOTAL($total);\n    static::computetotal($total);\n"
            "    $this->checkout->assertTotal($total);\n}")
    mine = compare(p, scorer, block(code))
    assert (mine["bv_resolved"], mine["bv_grounded"], mine["bv"]) == (1, 1, 1)


def test_script_imports_no_scoring_module():
    src = (ROOT / "scripts" / "score_independent.py").read_text()
    for banned in ("compose", "evaluate", "insert", "ertree", "extract", "outline", "state", "inherit", "symbols", "calls",
                   "argcheck", "controls", "describe"):
        assert f"v2tools.{banned}" not in src and f"from v2tools import {banned}" not in src


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)


def common(w):
    return ["--phase2", str(w / "phase2"), "--pairs", str(w / "pairs.jsonl"), "--config", str(w / "sources.json"),
            "--full-root", str(w / "full")]


def test_cli_matches_score_run_on_controls_and_outputs(tmp_path):
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    gold_text = next(json.loads(l)["text"] for l in (ctl / "control_outputs.jsonl").read_text().splitlines()
                     if json.loads(l)["condition"] == "gold")
    gen = w / "gen.jsonl"
    gen.write_text(json.dumps({"record_id": r["id"], "model": "deepseek", "seed": "s42", "condition": "full_aware",
                               "variant": "main", "text": gold_text, "prompt_tokens": 100, "new_tokens": 50,
                               "max_new_tokens": 6144, "stopped_at_eos": True, "prompts_sha256": "a" * 64,
                               "prompt_sha256": "b" * 64, "fmt": "F2"}) + "\n")
    manifest, coverage = gen_plan([gen])  # the generation run's manifest: a model row needs it
    out = w / "scores"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--manifest", str(manifest), "--coverage", str(coverage),
               "--outputs", str(gen), "--controls-dir", str(ctl), "--self-contained", str(w / "self_contained.jsonl"),
               "--subsets", str(w / "subsets.json"), "--out", str(out),
               "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")).returncode == 0
    probes = w / "probes.jsonl"
    res = run("scripts/probe_outputs.py", "--phase2", str(w / "phase2"), "--config", str(w / "sources.json"), "--partition",
              "test", "--controls-dir", str(ctl), "--out", str(probes))
    assert res.returncode == 0, res.stderr
    names = {json.loads(l)["condition"] for l in probes.read_text().splitlines()}
    assert {"U1_whole_class", "U3_dup_fn", "U4_dup_reuse", "U12_empty_bodies", "A_comment", "C_after_return", "K_logger_only"} <= names
    pout = w / "probe_scores"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--outputs", str(probes), "--controls-dir", str(ctl), "--self-contained", str(w / "self_contained.jsonl"),
               "--subsets", str(w / "subsets.json"), "--out", str(pout),
               "--prompts-meta", str(w / "PROMPTS_META.json"), "--a3-repo", str(w / "a3")).returncode == 0
    control_cols = set(ind.ROW) & set(json.loads((ctl / "controls.jsonl").read_text().split("\n")[0]))  # D11: n_calls, call_precision
    assert {"open_resolution", "n_calls", "call_precision"} <= control_cols  # D11: the column contract is fixed
    assert "stub-v3names" in {json.loads(l)["condition"] for l in (ctl / "control_outputs.jsonl").read_text().splitlines()}
    for outputs, scores, cols in ((ctl / "control_outputs.jsonl", ctl / "controls.jsonl", control_cols),
                                  (gen, out / "scores.jsonl", set(ind.ROW)), (probes, pout / "scores.jsonl", set(ind.ROW))):
        res = run("scripts/score_independent.py", *common(w), "--outputs", str(outputs), "--scores", str(scores))
        assert res.returncode == 0, res.stdout + res.stderr
        last = json.loads(res.stdout.splitlines()[-1])
        assert last["differences"] == {} and set(ind.ROW) & set(last["compared"]) == cols
        assert len(set(last["compared"].values())) == 1  # every column on every row


def test_either_annotation_form_gives_the_same_row(pair):
    """④(b) shows step annotations as written at HEAD (fixed 26/09/2026; Sylius: PHP 8 attributes `#[Then('x')]` since
    02/2026, docblock `@Then x` before), so an output may copy either form. The independent scorer gives the SAME row
    for the same function in the two forms (compare() checks each against evaluate): the six conditions, binders, CA
    call sets after helper-call expansion, cuts, CA and every static column but conv_style, which measures the form
    itself (rouge_l and exact read the text); Turnip and regex patterns, a `\\'` escape included."""
    p, scorer = pair
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    attr = "#[Then('I should see the total :total')]"
    body, g = GOLD.split(attr, 1)[1], (SHOP, "ishouldseetotal")
    by_form = []
    for pattern in ("I should see the total :total", "I should see the (customer's) total :total",
                    "/^I should see the (?:grand )?total \"(\\d+)\"$/"):
        a = compare(p, scorer, block("#[Then('" + pattern.replace("'", "\\'") + "')]" + body))
        d = compare(p, scorer, block("/**\n * @Then " + pattern + "\n */" + body))
        for x in (a, d):
            x["ca_defined"], x["ca"] = scorer.ca(p.record, labels_row, x)
        assert a["bv"] == a["ca"] == 1 and a["calls"][g] == {("checkout", "assertTotal")}, pattern
        assert {k: v for k, v in a.items() if k not in ("conv_style", "rouge_l", "exact")} == \
            {k: v for k, v in d.items() if k not in ("conv_style", "rouge_l", "exact")}, pattern
        by_form.append((a["conv_style"], d["conv_style"]))
    assert by_form == [(1.0, 0.0)] * 3



# ----- Every annotation form the reader takes or refuses (design specification §2.2 at c6999dc, fixed 27/09/2026),
# pinned in both scorers on the fixtures of the four languages; the forms come from real framework runs (Behat
# 4.0.0-alpha1, Cucumber-JVM 7.34.3 and 7.8.1, SpecFlow 3.0.213, cucumber-ruby 11.1.1).

def forms_agree(p, scorer, texts: dict[str, str]) -> dict[str, tuple[int, int]]:
    """name -> (BV, CA) of each output text; compare() checks that both scorers give the same row for it."""
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    out = {}
    for name, text in texts.items():
        x = compare(p, scorer, text, labels_row)
        out[name] = (x["bv"], scorer.ca(p.record, labels_row, x)[1])
    return out


def test_php_attribute_forms_score_alike_in_both_scorers(pair, ppair):
    """The answer's attribute in each form: a step when its name resolves to Behat\\Step\\Then through the file's `use`
    (the fixture's ShopContext imports Given, Then and When) in any case, an alias the output's block imports, the
    fully qualified name as written, whitespace and `pattern:`, a concatenation, two attributes in one `#[...]`; BV 1,
    CA 1. Not a step (BV 0, CA 0): `\\behat\\step\\then`, another namespace's Then, a constant, a private method, and
    `When` in ppair's ShopContext, which imports Given and Then only."""
    p, scorer = pair
    body = GOLD.split("#[Then('I should see the total :total')]", 1)[1]
    read = {"plain": "#[Then('I should see the total :total')]", "lower": "#[then('I should see the total :total')]",
            "fqn": "#[\\Behat\\Step\\Then('I should see the total :total')]",
            "spaced": "#[ Then ( pattern: 'I should see the total :total' ) ]",
            "concat": "#[Then('I should see ' . \"the total :total\")]", "when": "#[When('I should see the total :total')]",
            "group": "#[Then('I should see the total :total'), Then('I see the sum :total')]",
            "alias": "use Behat\\Step\\Then as Thn;\n\n#[Thn('I should see the total :total')]"}
    refused = {"fqn_case": "#[\\behat\\step\\then('I should see the total :total')]",
               "other": "#[Other\\Then('I should see the total :total')]", "const": "#[Then(self::TOTAL)]"}
    texts = {k: block(a + body) for k, a in {**read, **refused}.items()}
    texts["private"] = block(read["plain"] + body.replace("public function", "private function"))
    assert forms_agree(p, scorer, texts) == {**{k: (1, 1) for k in read}, **{k: (0, 0) for k in refused},
                                             "private": (0, 0)}
    p2, s2 = ppair
    body2 = "\npublic function iShouldSeeTotal(string $total): void\n{\n    $this->langId($total);\n}"
    assert forms_agree(p2, s2, {k: pblock(f"#[{k}('I should see the total :total')]" + body2) for k in ("Then", "When")}) \
        == {"Then": (1, 1), "When": (0, 0)}


def test_java_annotation_forms_score_alike_in_both_scorers(jpair):
    """The answer's annotation in each form (the fixture's ShopSteps imports Given, Then and When): the simple name, the
    fully qualified name, whitespace, `value =`, a concatenation, a second annotation, `@And` through an on-demand
    import the output's block brings; BV 1, CA 1. Not a step (BV 0, CA 0): `@And` without an import,
    io.cucumber.java.Then, a constant, a package-private method."""
    p, scorer = jpair
    body = "\npublic void iShouldSeeTotal(String total) {\n    cart.assertTotal(computeTotal(total));\n}"
    pat = '"I should see the total {string}"'
    read = {"plain": f"@Then({pat})", "fqn": f"@io.cucumber.java.en.Then({pat})", "spaced": f"@ Then ( {pat} )",
            "value": f"@Then(value = {pat})", "concat": '@Then("I should see the total " + "{string}")',
            "two": f'@Then({pat})\n@Given("I see the sum {{string}}")',
            "star": f"import io.cucumber.java.en.*;\n\n@And({pat})"}
    refused = {"and": f"@And({pat})", "package": f"@io.cucumber.java.Then({pat})", "const": "@Then(TOTAL)"}
    texts = {k: jblock(a + body) for k, a in {**read, **refused}.items()}
    texts["package_private"] = jblock(read["plain"] + body.replace("public void", "void"))
    assert forms_agree(p, scorer, texts) == {**{k: (1, 1) for k in read}, **{k: (0, 0) for k in refused},
                                             "package_private": (0, 0)}


# A C# twin: a file-scoped namespace, `using TechTalk.SpecFlow;`, the REUSE step (2020) and the
# CREATE answer iShouldSeeTotal (2024) calling Cart.AssertTotal; the regexes hold no quote or backslash, so the source
# of each literal is its value (step_fn finds the annotation line by it).
CPATH = "Steps/ShopSteps.cs"
CCART = ("namespace Acme;\n\npublic class Cart\n{\n    public void Open()\n    {\n    }\n\n"
         "    public void AssertTotal(string total)\n    {\n    }\n}\n")
CS_2020 = ("using Acme;\nusing TechTalk.SpecFlow;\n\nnamespace Acme.Steps;\n\n[Binding]\npublic class ShopSteps\n{\n"
           "    private readonly Cart _cart = new Cart();\n\n    [Given(\"^I open the shop$\")]\n    public void iOpenTheShop()\n"
           "    {\n        _cart.Open();\n    }\n}\n")
CS_2024 = CS_2020.replace("        _cart.Open();\n    }\n", "        _cart.Open();\n    }\n\n"
                          "    [Then(\"^I should see the total .([0-9]+).$\")]\n    public void iShouldSeeTotal(string total)\n"
                          "    {\n        _cart.AssertTotal(total);\n    }\n")


@pytest.fixture()
def cpair(tmp_path):
    return build_pair(tmp_path, "Acme/CSharp", "csharp", ["csharp_attribute"],
                      [(T2020, {"Cart.cs": CCART, CPATH: CS_2020}), (T2024, {CPATH: CS_2024})], CPATH,
                      "^I open the shop$", "^I should see the total .([0-9]+).$", 2)


def test_csharp_attribute_forms_score_alike_in_both_scorers(cpair):
    """The answer's attribute in each form: the simple name, the Attribute suffix, the fully qualified name (with and
    without the suffix), `Regex =`, another named argument, a concatenation, a verbatim literal, StepDefinition,
    whitespace, a second attribute in the same `[...]`, a using alias the output's block brings, a private method
    (SpecFlow scans non-public methods); BV 1, CA 1. Not a step (BV 0, CA 0): another namespace's Then, a constant,
    `[Then]` without a regex, `[then(...)]` (C# names are case-sensitive)."""
    p, scorer = cpair
    body = "\npublic void iShouldSeeTotal(string total)\n{\n    _cart.AssertTotal(total);\n}"
    pat = '"^I should see the total .([0-9]+).$"'
    read = {"plain": f"[Then({pat})]", "suffix": f"[ThenAttribute({pat})]", "fqn": f"[TechTalk.SpecFlow.Then({pat})]",
            "fqn_suffix": f"[TechTalk.SpecFlow.ThenAttribute({pat})]", "regex": f"[Then(Regex = {pat})]",
            "named": f'[Then({pat}, Culture = "en-US")]', "concat": '[Then("^I should see the total " + ".([0-9]+).$")]',
            "verbatim": f"[Then(@{pat})]", "stepdef": f"[StepDefinition({pat})]", "spaced": f"[ Then ( {pat} ) ]",
            "list": f'[Then({pat}), When("^I see the sum$")]',
            "alias": f"using T = TechTalk.SpecFlow.ThenAttribute;\n\n[T({pat})]"}
    refused = {"other": f"[Other.Then({pat})]", "const": "[Then(TotalPattern)]", "bare": "[Then]", "lower": f"[then({pat})]"}
    fence = lambda code: f"{TICKS}csharp\n// FILE: {CPATH}\n{code}\n{TICKS}"
    texts = {k: fence(a + body) for k, a in {**read, **refused}.items()}
    texts["private"] = fence(read["plain"] + body.replace("public void", "private void"))
    assert forms_agree(p, scorer, texts) == {**{k: (1, 1) for k in read}, **{k: (0, 0) for k in refused},
                                             "private": (1, 1)}


def test_ruby_string_patterns_are_cucumber_expressions_in_both_scorers(rbpair):
    """cucumber-ruby 11.1.1: a String pattern is a Cucumber expression in either quote (BV 1, CA 1 with {string}); one
    written like a regex, `'^...$'`, is still an expression whose `^`, `$` and groups are literal text, so it binds no
    step (BV 0, CA 0); the Regexp literal binds (BV 1, CA 1)."""
    p, scorer = rbpair
    fence = lambda head: f"{TICKS}ruby\n# FILE: {RBPATH}\n{head}\n  check_total(total)\nend\n{TICKS}"
    texts = {"single": fence("Then('I should see the total {string}') do |total|"),
             "double": fence('Then("I should see the total {string}") do |total|'),
             "anchored": fence("""Then('^I should see the total "([^"]*)"$') do |total|"""),
             "regexp": fence("""Then(/^I should see the total "([^"]*)"$/) do |total|""")}
    assert forms_agree(p, scorer, texts) == {"single": (1, 1), "double": (1, 1), "anchored": (0, 0), "regexp": (1, 1)}



def test_u4_brings_the_import_of_its_given(pair, jpair):
    """Probe U4 (an extra function re-binding the first REUSE step; design specification §2.2 at 8e4405a, fixed
    27/09/2026) opens its block with the import of its `Given`, as a valid output would, so that it re-binds the step
    whatever the target file imports (without it, U4 passed BV in 82/186 Sylius and 80/96 fineract test records, whose
    target files do not import Given). On both fixtures both scorers give it BV 0 (the step is bound twice)."""
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    for (p, scorer), lang, style, head in ((pair, "php", "php_attribute", "use Behat\\Step\\Given;"),
                                           (jpair, "java", "java_annotation", "import io.cucumber.java.en.Given;")):
        gold = extract(scorer.gold_text(p.record), p.targets).by_path()
        u4 = po.probes(p.record, lang, style, gold, controls(p)["stub-v3"][2])["U4_dup_reuse"]
        assert u4.split("\n")[2] == head, u4[:120]
        assert compare(p, scorer, u4)["bv"] == 0, lang
    assert po.reuse_import("php", "php_docblock") is None


def test_the_mirrors_follow_the_main_reader_and_insertion_rule():
    """The independent scorer's own copies follow the same evidence as main: php_kept compares every pattern of a HEAD
    attribute group; a docblock tag is Behat 3's `@given|@when|@then` in any case at a doc line's start, not `@And`; a
    java8 `Step(...)` call is no lambda step and a Ruby `Step` block no step (java-forms.txt, ruby-forms.txt). Main and
    the mirror give the same text, lines and items."""
    from v2tools.insert import STEP_ANNOTATION, php_kept, split_code, with_annotations
    old = "#[Given('a :x'), When('b :x')]\n/**\n * @then c :x\n */\npublic function f($x): void\n{\n}"
    for new in ("#[When('b :x')]\npublic function f($x): void\n{\n}",
                "#[Given('a :x'), When('b :x')]\npublic function f($x): void\n{\n}",
                "#[When('b :x'), Then('d :x')]\npublic function f($x): void\n{\n}",
                "/**\n * @THEN c :x\n * @And e :x\n */\npublic function f($x): void\n{\n}"):
        assert ind.with_old("php", new, old) == with_annotations("php", *php_kept(new, old)), new
    for line in (" * @then c", " * @WHEN c", " * @And c", " * @Step c", " *  * @Then c", " * Use @Given c"):
        assert bool(ind.ANN_LINE["php"].match(line)) == bool(STEP_ANNOTATION["php"].match(line)), line
    java = 'Given("a", () -> {\n});\nStep("b", () -> {\n});'
    assert [(i["kind"], i["text"]) for i in ind.items_of("java", java)] == [(i.kind, i.text) for i in split_code("java", java)[1]]
    assert [i.kind for i in split_code("java", java)[1]] == ["lambda", "raw"]
    assert ind.ruby_step_params("Step(/^a$/) do |x|\nend") == [] and ind.ruby_step_params("And(/^a$/) do |x|\nend") == ["x"]


def test_member_uses_agree_with_symbols_on_receiver_and_local_forms():
    """The independent reader follows the same rules as symbols.refs (fixed 28/09/2026 and 04/10/2026), its own way:
    `this.f()`, `super.f()`, C# `this.F()`/`base.F()` are member calls; pattern, tuple and LINQ variables are locals; a
    bare or `this.` call inside an anonymous class body is no member."""
    from v2tools.symbols import refs
    cases = {("java", "public void a() {\n    this.helper();\n    super.other();\n    super.f = 1;\n}"): ({"helper", "other", "f"}, {"helper", "other"}),
             ("java", "public void a() {\n    Map m = new HashMap<>() {{ put(\"k\", value); this.put(\"x\", \"y\"); }};\n    api.go(m);\n}"): ({"value", "api"}, set()),
             ("java", "public void a(Object o) {\n    if (o instanceof String sx) {\n        api.go(sx);\n    }\n}"): ({"api"}, set()),
             ("csharp", "public void A()\n{\n    base.Helper();\n    this.Other();\n}"): ({"Helper", "Other"}, {"Helper", "Other"}),
             ("csharp", "public void A(object o)\n{\n    if (o is string sx) { _api.Go(sx); }\n    var (aa, bb) = _api.Pair();\n"
                        "    var q = from r in _api.Rows let t = r.V select t;\n    _api.Go(aa, bb, q);\n}"): ({"_api"}, set())}
    for (lang, code), (mem, calls) in cases.items():
        m, c, _ = ind.member_uses(lang, code)
        r = refs(lang, code)
        assert (m, c) == (mem, calls) == (set(r.members), set(r.calls)), (lang, code)


def test_n_keeps_a_trailing_group_that_matches_the_empty_string():
    """The independent mirror of argcheck.placeholders, its own way: a trailing group matching "" counts; one that
    takes no part in the match does not."""
    from types import SimpleNamespace as NS
    me = NS(lang="php")
    d = NS(kind="regex_full", pattern='^I change order "(.*)" note to "(.*)"$')
    assert ind.Scorer._placeholders(me, d, 'I change order "bo_order1" note to ""') == 2
    assert ind.Scorer._placeholders(me, NS(kind="regex_full", pattern=r"^x(?: (\d+))?$"), "x") == 0


# ----- Pins added after mutation testing of score_independent.py: 14 pins of rules both scorers mirror, the target
# values of three Java reading rules (`this.` calls, anonymous class bodies, pattern variables) and the CLI's
# fail-closed inputs. Each pin kills mutants that survived the earlier tests.

D = namedtuple("D", "kind pattern")


def _po():
    spec_po = importlib.util.spec_from_file_location("probe_outputs", ROOT / "scripts" / "probe_outputs.py")
    po = importlib.util.module_from_spec(spec_po)
    spec_po.loader.exec_module(po)
    return po


def test_pin_trailing_group_matching_empty_counts(pair):
    """Against main, as the test above: preg_match drops only UNMATCHED trailing groups; a trailing group matching ""
    counts."""
    from v2tools.argcheck import placeholders
    _, scorer = pair
    pat, step = '^I change order "(.*)" note to "(.*)"$', 'I change order "bo_order1" note to ""'
    assert scorer._placeholders(D("regex_full", pat), step) == placeholders("regex_full", pat, "php", step) == 2
    opt = r"^x(?: (\d+))?$"
    assert scorer._placeholders(D("regex_full", opt), "x") == placeholders("regex_full", opt, "php", "x") == 0


def test_pin_blocks_of_random_differential():
    """blocks_of equals extract on two minimal texts and on 3000 seeded random texts of fence, FILE and tag
    lines."""
    t = TICKS
    targets = ["a.php", "b.php"]
    fixed = [f"{t}\n{t}\n  {t}", f"{t}php\n// FILE: a.php\nf();\n// FILE: b.php\ng();"]
    pieces = [f"{t}php", t, f"{t}`", f"{t}`php", f"  {t}", f"{t}bash", "// FILE: a.php", "// FILE: b.php",
              "// FILE: ./a.php", "// FILE: c.php", "# FILE: a.php", "<?php", "?>", "<?php echo 1;", "", "f();",
              "/*", "*/", "Run:", "\r"]
    rng = random.Random(7)
    texts = fixed + ["\n".join(rng.choice(pieces) for _ in range(rng.randint(1, 8))) for _ in range(3000)]
    for text in texts:
        e = extract(text, targets)
        assert ind.blocks_of(text, targets) == (e.status, e.by_path(), e.truncated), repr(text)


def test_pin_php_and_doc_tag_is_not_matched_by_pattern():
    """An output's `@And c :x` docblock line is no Behat tag; a HEAD `@then c :x` is kept, the output line
    stays."""
    from v2tools.insert import php_kept, with_annotations
    old = "/**\n * @then c :x\n */\npublic function f($x): void\n{\n}"
    new = "/**\n * @And c :x\n */\npublic function f($x): void\n{\n}"
    assert ind.with_old("php", new, old) == with_annotations("php", *php_kept(new, old))


def test_pin_php_hash_comment_and_heredoc_are_no_code(pair):
    """The gold call in a `#` comment or inside a heredoc is no call: CA 0 in both scorers."""
    p, scorer = pair
    for body in ("    # $this->checkout->assertTotal($total);\n}",
                 "    $h = <<<EOT\n    {$this->checkout->assertTotal($total)}\n    EOT;\n}"):
        mine = compare(p, scorer, block(HEAD + body))
        labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
        assert scorer.ca(p.record, labels_row, mine) == (True, 0), body


def test_pin_java_char_literal_quote(jpair):
    """A Java char literal of a quote leaves one character, so the code after it is still read."""
    p, scorer = jpair
    out = compare(p, scorer, jblock(JHEAD + "    char c = ','; cart.assertTotal(total);\n}"))
    assert (out["bv"], out["n_calls"]) == (1, 2)


def test_pin_hardcoded_number(pair):
    """A number of the step written into the pattern is hard-coded (argcheck.hardcoded)."""
    from v2tools.argcheck import hardcoded
    _, scorer = pair
    for pat, step in ((r"^I have 5 apples$", "I have 5 apples"), (r"^I pay 12.50 now$", "I pay 12.50 now")):
        assert scorer._hardcoded(D("regex_full", pat), step) is True and hardcoded("regex_full", pat, step)


def test_pin_duplicate_reuse_only_for_the_records_reuse_steps(pair):
    """Re-binding a suite step that is NOT one of the record's steps breaks condition 3 but is no duplicate_reuse."""
    p, scorer = pair
    again = "\n\n#[When('I add :product to the cart')]\npublic function again2(string $product): void\n{\n}"
    out = compare(p, scorer, block(GOLD + again))
    assert (out["bv_no_ambiguity"], out["duplicate_reuse"]) == (0, False)


def test_pin_open_resolution_needs_condition_6(jpair):
    """An open path (a name of a static import the output adds) next to an unresolved name: open_resolution False."""
    p, scorer = jpair
    code = "import static org.other.Util.helpZ;\n\n" + JHEAD + "    helpZ();\n    ghostZ.x();\n}"
    out = compare(p, scorer, jblock(code))
    assert (out["bv_resolved"], out["open_resolution"]) == (0, False)


def test_pin_a_local_hides_a_field_receiver(jpair):
    """A local named like a field is no field receiver: its calls are not project calls of the field."""
    p, scorer = jpair
    out = compare(p, scorer, jblock(JHEAD.replace("    cart.open();\n", "    acme.Cart cart = new acme.Cart();\n")
                                    + "    cart.assertTotal(total);\n}"))
    assert out["n_calls"] == 0, out["calls"]


def test_pin_exact_ignores_whitespace_only(pair):
    """exact compares the token streams up to whitespace (re-indented gold is exact)."""
    p, scorer = pair
    out = compare(p, scorer, block(GOLD.replace("\n    $this", "\n        $this")))
    assert out["exact"] is True


def test_pin_cli_exits_1_on_a_difference(tmp_path):
    """A flipped bv, or a row missing from --scores, is a difference: exit 1."""
    w, r = make_phase2(tmp_path)
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    rows = [json.loads(l) for l in (ctl / "controls.jsonl").read_text().splitlines()]
    flipped = w / "flipped.jsonl"
    flipped.write_text("".join(json.dumps({**x, "bv": 1 - x["bv"]} if x["control"] == "gold" else x) + "\n"
                               for x in rows))
    outputs = str(ctl / "control_outputs.jsonl")
    res = run("scripts/score_independent.py", *common(w), "--outputs", outputs, "--scores", str(flipped))
    assert res.returncode == 1 and json.loads(res.stdout.splitlines()[-1])["differences"].get("bv") == 1
    short = w / "short.jsonl"
    short.write_text("".join(json.dumps(x) + "\n" for x in rows if x["control"] != "gold"))
    res = run("scripts/score_independent.py", *common(w), "--outputs", outputs, "--scores", str(short))
    assert res.returncode == 1 and json.loads(res.stdout.splitlines()[-1])["differences"] == {"missing_in_scores": 1}


def test_pin_probe_definitions(pair):
    """U4's C# using; U4 re-binds a REUSE step (duplicate_reuse); V10 is the gold cut at 60%; C puts `return;`
    before the gold lines."""
    po = _po()
    assert po.reuse_import("csharp", "csharp_attribute") == "using TechTalk.SpecFlow;"
    p, scorer = pair
    gold = extract(scorer.gold_text(p.record), p.targets).by_path()
    out = po.probes(p.record, "php", "php_attribute", gold, controls(p)["stub-v3"][2])
    assert compare(p, scorer, out["U4_dup_reuse"])["duplicate_reuse"] is True
    whole = po.fenced("php", gold)
    assert out["V10_cut_off"] == whole[:int(len(whole) * 0.6)] and len(out["V10_cut_off"]) < len(whole)
    c = out["C_after_return"]
    assert c.index("return;") < c.index("$this->checkout->assertTotal")


def test_pin_ca_unions_the_functions_bound_to_one_gold_functions_steps(pair):
    """Both sides (independent, and calls.agreement for main): one gold function binds two CREATE steps; the output
    binds them with two functions and only the second one's calls meet the gold's: CA 1 (the union per gold
    function)."""
    p, scorer = pair
    steps = ["I open the shop", 'I should see the total "5"', 'I should see the total "7"']
    q, labels_row, _ = merged_record(p, steps, ["reuse", "create", "create"],
                                     ["iOpenTheShop", "iShouldSeeTotal", "iShouldSeeTotal"])
    out = ("#[Then('/^I should see the total \"(5)\"$/')]\npublic function f5(string $t): void\n"
           "{\n    $this->cart->add($t);\n}\n\n"
           "#[Then('/^I should see the total \"(7)\"$/')]\npublic function f7(string $t): void\n{\n"
           "    $this->checkout->assertTotal($t);\n}")
    mine = compare(q, scorer, block(out), labels_row)
    assert (mine["bv"], scorer.ca(q.record, labels_row, mine)) == (1, (True, 1))


def test_pin_a_target_file_serves_the_suites_of_its_functions(tmp_path):
    """A step of another suite that a function of the target file serves is bound in the ER tree; an output function
    re-binding it breaks condition 3 in both scorers."""
    from tests.phase3_fixture import make_repo
    bare, head = make_repo(tmp_path)
    cat = catalog()
    opener = next(f for f, row in cat.items() if "iOpenTheShop(" in row["code"])
    cat[opener] = {**cat[opener], "suites": ["s", "t"]}
    ctx = RepoContext("Acme/Shop", "php", bare, head, cat)
    def_of = lambda text: next(a["def_id"] for f in cat.values() for a in f["annotations"] if text in a["pattern"])
    suite = {"s": [("I open the shop", (def_of("open the shop"),)), ('I should see the total "5"', (def_of("total"),)),
                   ('I add "apple" to the cart', (def_of("cart"),))],
             "t": [("I zap the shop", (def_of("open the shop"),))]}
    repo = Repo("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ctx, frozenset(), suite)
    r = record(cat, "test")
    r["commit"] = head
    p = prepare(r, repo, {"step_def_ids": [[def_of("open the shop")], [def_of("total")]]})
    scorer = ind.Scorer("Acme/Shop", "php", ["php_attribute", "php_docblock"], cat, ind.Git(ctx.git_dir, head), suite)
    zap = GOLD + "\n\n#[Given('I zap the shop')]\npublic function zap(): void\n{\n}"
    assert SHOP in p.targets and compare(p, scorer, block(zap))["bv_no_ambiguity"] == 0


# The pins of the three Java reading rules assert the TARGET VALUES on the Java fixture, not only that the two
# scorers agree (compare() checks that on every column of the row; the literals below are what each rule must give).
JOPEN = "    cart.open();\n"
JGOLD_CALLS = {("cart", "assertTotal"), ("store", "total")}  # the gold's calls after expansion, on the fixture


def jvalues(p, scorer, code: str) -> dict:
    """BV, the conditions grounded and resolved, `unresolved`, n_calls, the calls of the output's functions and CA of a
    Java output (the step class's members), after compare() has checked the whole row against evaluate."""
    mine = compare(p, scorer, jblock(code))
    labels_row = {"step_def_ids": [[ids[0]] for _, ids in p.repo.suite_steps["s"][:2]]}
    return {"bv": mine["bv"], "grounded": mine["bv_grounded"], "resolved": mine["bv_resolved"],
            "unresolved": mine["unresolved"], "n_calls": mine["n_calls"], "calls": set().union(*mine["calls"].values()),
            "ca": scorer.ca(p.record, labels_row, mine)}


def test_this_and_bare_calls_of_an_ancestors_method_have_one_value(jpair):
    """`this.baseOld();` alone and `baseOld();` alone are the same reference to the visible ancestor method
    (BaseSteps.baseOld): BV 1, nothing unresolved, the one call (self, baseOld). Before this rule `this.baseOld();` gave
    BV 0 (condition 5: `this.` calls were skipped by member_uses)."""
    p, scorer = jpair
    for call in ("this.baseOld();", "baseOld();"):
        got = jvalues(p, scorer, JHEAD.replace(JOPEN, f"    {call}\n") + "}")
        assert (got["bv"], got["unresolved"], got["n_calls"], got["calls"]) == (1, 0, 1, {("self", "baseOld")}), call
    assert ind.member_uses("java", "public void a() {\n    this.baseOld();\n}")[:2] == ({"baseOld"}, {"baseOld"})
    assert ind.member_uses("java", "public void a() {\n    baseOld();\n}")[:2] == ({"baseOld"}, {"baseOld"})


def test_an_invented_member_called_through_this_is_unresolved(jpair):
    """The same rule closes a known limit: `this.ghostZ();` names a member no class of the ER tree has, as `ghostZ();`
    does: BV 0 (grounded and resolved both fail), one unresolved name. It used to give BV 1, because member_uses skipped
    every `this.f()` and `super.f()` call; `super.ghostY` is a member access of the same kind."""
    p, scorer = jpair
    for call in ("this.ghostZ();", "ghostZ();", "super.ghostZ();", "super.ghostY.x();"):
        got = jvalues(p, scorer, JHEAD.replace(JOPEN, f"    {call}\n") + "}")
        assert (got["bv"], got["grounded"], got["resolved"], got["unresolved"]) == (0, 0, 0, 1), call
    assert ind.member_uses("java", "public void a() {\n    super.other();\n}")[:2] == ({"other"}, {"other"})


def test_a_double_brace_map_adds_no_member_of_the_step_class(jpair):
    """A bare or `this.` call inside a Java anonymous class body (`new HashMap<>() {{ put(...); }}`)
    is that class's own, never a member of the step class: the gold plus such a map keeps BV 1, CA 1, nothing
    unresolved, and the gold's two calls. Before this rule `put` read as an unresolved member (condition 6, BV 0)."""
    p, scorer = jpair
    gold = jvalues(p, scorer, JGOLD)
    assert (gold["bv"], gold["unresolved"], gold["n_calls"], gold["calls"], gold["ca"]) \
        == (1, 0, 2, JGOLD_CALLS, (True, 1))
    for body in ('put("a", "b");', 'put("a", "b"); this.put("c", "d");'):
        extra = "    java.util.Map<String, String> m = new java.util.HashMap<>() {{ " + body + " }};\n"
        got = jvalues(p, scorer, JGOLD.replace("computeTotal(total));\n", "computeTotal(total));\n" + extra))
        # the call sets of both scorers leave a bare or `this.` call of the anonymous class out too
        assert got == gold, body
    # `this.modCount` of an anonymous ArrayList is the anonymous class's own field, as `this.put(` is its own method
    own = "    java.util.List<String> l = new java.util.ArrayList<String>() {{ this.modCount = 0; }};\n"
    got = jvalues(p, scorer, JGOLD.replace("computeTotal(total));\n", "computeTotal(total));\n" + own))
    assert {k: got[k] for k in tuple(gold)} == {k: gold[k] for k in gold}
    anon = "public void a() {\n    Map m = new HashMap<>() {{ put(\"k\", 1); this.put(\"x\", 2); }};\n}"
    assert ind.member_uses("java", anon)[:2] == (set(), set())
    anon = "public void a() {\n    List l = new ArrayList<>() {{ this.modCount = 0; }};\n}"
    assert ind.member_uses("java", anon)[0] == set()


def test_an_instanceof_variable_is_a_local_in_both_copies_of_the_locals_list(jpair):
    """The pattern variable of `if (total instanceof String s2) { s2.length(); }`, inside the gold step body, is a
    local: BV 1 as the gold's, nothing unresolved, the call set and CA of the gold. This exercises the
    local-declaration list of member_uses (before this rule: `s2` read as an unresolved member, BV 0, unresolved 1).
    The second copy of that list, in Scorer._calls, is exercised by a pattern variable that has the name of a field of
    the class (`cart`): its call set must be main's (calls.jvm_calls); both read it as a local (the values are pinned
    by test_the_call_sets_of_both_scorers_read_pattern_locals_and_anonymous_classes_alike). A change of one copy alone
    shows as a different BV (member_uses) or as a disagreement with main (compare() inside jvalues, for _calls)."""
    p, scorer = jpair
    gold = jvalues(p, scorer, JGOLD)
    use = "    cart.assertTotal(computeTotal(total));\n"
    got = jvalues(p, scorer, JGOLD.replace(use, use + "    if (total instanceof String s2) { s2.length(); }\n"))
    assert (got["bv"], got["unresolved"], got["n_calls"], got["calls"], got["ca"]) == (1, 0, 2, JGOLD_CALLS, (True, 1))
    assert got == gold
    # a type pattern of a switch and a record pattern are locals as well: the same values
    for stmt in ("switch (total) { case String s3 -> s3.length(); default -> {} }",
                 "switch (total) { case Entry(var k3, var v3) -> k3.length(); default -> {} }"):
        got = jvalues(p, scorer, JGOLD.replace(use, use + f"    {stmt}\n"))
        assert got == gold, stmt
    # the name of a field: member_uses declares `cart` (no unresolved name), _calls follows main's call set
    shadow = jvalues(p, scorer, JGOLD.replace(use, use + "    if (total instanceof String cart) { cart.length(); }\n"))
    assert (shadow["bv"], shadow["unresolved"]) == (1, 0)
    code = "public void a(Object o) {\n    if (o instanceof String sx) { sx.length(); }\n}"
    assert ind.member_uses("java", code)[:2] == (set(), set())


# The CLI reports every differing cell and refuses, with exit 3 and a
# `refusing: ...` line on stderr (never a traceback), what it cannot compare; a disagreement stays exit 1.
@pytest.fixture(scope="module")
def cli_world(tmp_path_factory):
    """One Phase 2 + Phase 3 input tree and the controls score_run writes for it (never modified by the tests)."""
    w, r = make_phase2(tmp_path_factory.mktemp("cli"))
    ctl = w / "ctl"
    assert run("scripts/score_run.py", *common(w), "--symtimes", str(w / "symtimes.jsonl"), "--partition", "test",
               "--controls", "--out", str(ctl)).returncode == 0
    outs = [json.loads(l) for l in (ctl / "control_outputs.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (ctl / "controls.jsonl").read_text().splitlines()]
    return w, r, ctl, outs, rows


def jsonl(path, rows):
    """Write `rows` (JSON values, or raw strings kept as they are) as a JSONL file; returns the path."""
    path.write_text("".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in rows))
    return path


def indep(w, outputs, scores, *more, phase2=None):
    """score_independent on `outputs` against `scores` of the tree `w`; `more` replaces the value of an option."""
    args = common(w) + ["--outputs", str(outputs), "--scores", str(scores)]
    if phase2 is not None:
        args[args.index("--phase2") + 1] = str(phase2)
    for opt, value in zip(more[::2], more[1::2]):
        if opt in args:
            args[args.index(opt) + 1] = str(value)
        else:
            args += [opt, str(value)]
    return run("scripts/score_independent.py", *args)


def test_cli_reports_every_differing_cell(cli_world, tmp_path):
    """Every differing cell is printed (the old CLI stopped at 3 per column), each with the record, the run and both
    values; the last line is the summary, whose `differences` counts them; a row of the outputs with no row in --scores
    is printed too. Exit 1."""
    w, _, ctl, outs, rows = cli_world
    n = len(rows)
    assert n > 3
    flipped = jsonl(tmp_path / "flipped.jsonl", [{**x, "bv": 1 - x["bv"], "ca": 1 - x["ca"] if i < 5 else x["ca"]}
                                                  for i, x in enumerate(rows)])
    res = indep(w, ctl / "control_outputs.jsonl", flipped)
    lines = [json.loads(l) for l in res.stdout.splitlines()]
    assert res.returncode == 1, res.stderr
    assert lines[-1]["differences"] == {"bv": n, "ca": 5}, lines[-1]
    cells = lines[:-1]
    assert len(cells) == n + 5 and {c["differs"] for c in cells} == {"bv", "ca"}
    assert {(c["differs"], c["condition"]) for c in cells} >= {("bv", x["control"]) for x in rows}
    assert all({"record_id", "condition", "independent", "evaluate"} <= set(c) for c in cells)
    short = jsonl(tmp_path / "short.jsonl", rows[3:])
    res = indep(w, ctl / "control_outputs.jsonl", short)
    lines = [json.loads(l) for l in res.stdout.splitlines()]
    assert res.returncode == 1 and lines[-1]["differences"] == {"missing_in_scores": 3}
    assert [c["condition"] for c in lines[:-1]] == [x["control"] for x in rows[:3]]
    assert {c["differs"] for c in lines[:-1]} == {"missing_in_scores"}


def test_cli_refuses_an_unknown_record_id(cli_world, tmp_path):
    """An output of a record no records file holds is a refusal naming it: exit 3, no traceback (it
    was a KeyError, exit 1, the code of a disagreement)."""
    w, r, ctl, outs, rows = cli_world
    bad = jsonl(tmp_path / "bad.jsonl", [{**outs[0], "record_id": "Acme/Shop|features/none.feature|9|s"}] + outs[1:])
    res = indep(w, bad, ctl / "controls.jsonl")
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert "refusing:" in res.stderr and "features/none.feature" in res.stderr and res.stdout == ""


@pytest.mark.parametrize("what", ["output_no_text", "output_not_an_object", "output_text_not_a_string",
                                  "output_no_record_id", "scores_no_record_id", "scores_no_condition",
                                  "scores_not_an_object", "config_missing", "config_not_json", "forbidden_not_json",
                                  "label_missing"])
def test_cli_refuses_a_malformed_input(cli_world, tmp_path, what):
    """A row that is not a JSON object with the keys the comparison reads, a missing or unreadable
    config or forbidden.json, a record without a label: exit 3 and a `refusing:` line, never a traceback."""
    w, r, ctl, outs, rows = cli_world
    outputs, scores, more, phase2 = ctl / "control_outputs.jsonl", ctl / "controls.jsonl", [], None
    if what.startswith("output_"):
        bad = {"output_no_text": {k: v for k, v in outs[0].items() if k != "text"}, "output_not_an_object": [1, 2],
               "output_text_not_a_string": {**outs[0], "text": 7},
               "output_no_record_id": {k: v for k, v in outs[0].items() if k != "record_id"}}[what]
        outputs = jsonl(tmp_path / "o.jsonl", [bad] + outs[1:])
    elif what.startswith("scores_"):
        bad = {"scores_no_record_id": {k: v for k, v in rows[0].items() if k != "record_id"},
               "scores_not_an_object": '"x"',
               "scores_no_condition": {k: v for k, v in rows[0].items() if k != "control"}}[what]
        scores = jsonl(tmp_path / "s.jsonl", [bad] + rows[1:])
    elif what.startswith("config_"):
        cfg = tmp_path / "sources.json"
        if what == "config_not_json":
            cfg.write_text("[{")
        more = ["--config", cfg]
    else:
        phase2 = tmp_path / "phase2"
        shutil.copytree(w / "phase2", phase2)
        if what == "forbidden_not_json":
            (phase2 / "forbidden.json").write_text("{")
        else:
            jsonl(phase2 / "labels.jsonl", [])
    res = indep(w, outputs, scores, *more, phase2=phase2)
    assert res.returncode == 3 and "Traceback" not in res.stderr, (what, res.stderr)
    assert res.stderr.startswith("refusing:") and res.stdout == "", (what, res.stdout, res.stderr)


def test_cli_refuses_rows_of_scores_that_have_no_output(cli_world, tmp_path):
    """A row of --scores that no output row matches (more rows than were scored: the comparison would
    pass on the outputs given and say nothing about the rest) is a refusal naming it, before anything is scored."""
    w, r, ctl, outs, rows = cli_world
    extra = jsonl(tmp_path / "extra.jsonl", rows + [{**rows[0], "control": "ghost-control"}])
    res = indep(w, ctl / "control_outputs.jsonl", extra)
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert "refusing:" in res.stderr and "ghost-control" in res.stderr and "--scores" in res.stderr and res.stdout == ""
    fewer = jsonl(tmp_path / "fewer.jsonl", outs[:-1])
    res = indep(w, fewer, ctl / "controls.jsonl")
    assert res.returncode == 3 and rows[-1]["control"] in res.stderr, res.stderr


def test_cli_refuses_duplicate_keys_in_scores(cli_world, tmp_path):
    """Two rows of --scores with the same key (the old CLI kept the last one) are a refusal naming
    it."""
    w, r, ctl, outs, rows = cli_world
    dup = jsonl(tmp_path / "dup.jsonl", rows + [{**rows[2], "bv": 1 - rows[2]["bv"]}])
    res = indep(w, ctl / "control_outputs.jsonl", dup)
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert "refusing:" in res.stderr and "duplicate" in res.stderr and res.stdout == ""
    assert rows[2]["control"] in res.stderr


def test_cli_refuses_a_repository_whose_scored_records_are_at_two_commits(cli_world, tmp_path):
    """One Scorer is built per repository on the first record's commit, so outputs of two records of one repository at
    two commits are a refusal naming the repository (exit 3). A record at another commit that no output scores is no
    reason to refuse."""
    w, r, ctl, outs, rows = cli_world
    phase2 = tmp_path / "phase2"
    shutil.copytree(w / "phase2", phase2)
    other = {**r, "id": r["id"].replace("total.feature", "other.feature"), "commit": "0" * 40}
    jsonl(phase2 / "records" / "test.jsonl", [r, other])
    res = indep(w, ctl / "control_outputs.jsonl", ctl / "controls.jsonl", phase2=phase2)
    assert res.returncode == 0, res.stdout + res.stderr  # `other` is scored by no output
    both = jsonl(tmp_path / "both.jsonl", outs + [{**outs[0], "record_id": other["id"]}])
    res = indep(w, both, ctl / "controls.jsonl", phase2=phase2)
    assert res.returncode == 3 and "Traceback" not in res.stderr, res.stderr
    assert "refusing:" in res.stderr and "Acme/Shop" in res.stderr and "more than one commit" in res.stderr
    assert r["commit"] in res.stderr and "0" * 40 in res.stderr and res.stdout == ""


def test_cli_partition_skips_the_rows_of_the_test_aux_partner(cli_world, tmp_path):
    """The agreement also runs on the model outputs (the design specification asks for every output), and
    the generation run writes the test and aux_test rows of a run into one generation workdir. With --partition, a row
    of the partition's test/aux_test partner is skipped and counted (as score_run's skipped_other_partition), a row of a
    record of any other partition is a refusal (exit 3); without it every output row must still have its score row."""
    w, r, ctl, outs, rows = cli_world
    phase2 = tmp_path / "phase2"
    shutil.copytree(w / "phase2", phase2)
    aux = {**r, "id": r["id"].replace("total.feature", "aux.feature"), "feature_path": "features/aux.feature",
           "partition": "aux_test"}
    dev = {**r, "id": r["id"].replace("total.feature", "dev.feature"), "feature_path": "features/dev.feature",
           "partition": "dev"}
    jsonl(phase2 / "records" / "aux_test.jsonl", [aux])
    jsonl(phase2 / "records" / "dev.jsonl", [dev])
    mixed = jsonl(tmp_path / "mixed.jsonl", outs + [{**outs[0], "record_id": aux["id"]}])
    res = indep(w, mixed, ctl / "controls.jsonl", "--partition", "test", phase2=phase2)
    assert res.returncode == 0, res.stderr
    summary = json.loads(res.stdout.strip().splitlines()[-1])
    assert (summary["differences"], summary["skipped_other_partition"], summary["scored"]) == ({}, 1, len(outs))
    assert indep(w, mixed, ctl / "controls.jsonl", phase2=phase2).returncode != 0  # without --partition: not skipped
    wrong = jsonl(tmp_path / "wrong.jsonl", outs + [{**outs[0], "record_id": dev["id"]}])
    res = indep(w, wrong, ctl / "controls.jsonl", "--partition", "test", phase2=phase2)
    assert res.returncode == 3 and "is a dev record, not test" in res.stderr and "Traceback" not in res.stderr, res.stderr
    plain = indep(w, ctl / "control_outputs.jsonl", ctl / "controls.jsonl")
    assert plain.returncode == 0 and "skipped_other_partition" not in plain.stdout  # the summary is unchanged without it


def test_the_call_sets_of_both_scorers_read_pattern_locals_and_anonymous_classes_alike(jpair):
    """Fixed 04/10/2026: a pattern variable named like the field `store` (instanceof, a switch type pattern, a record
    pattern), which the step function does not use itself, is a local on the call side too, in main (calls.jvm_calls)
    and here (Scorer._calls): the gold plus a call on it keeps the gold's call set, n_calls and CA (before: main counted
    (store, length), here only the type and record patterns were locals); a bare or `this.` call inside a double-brace
    map body is the anonymous class's own, so the call set is the gold's, not the gold's plus (self, put). jvalues runs
    compare() first: the two scorers agree on every column. Locals are per function in both scorers, not per block: a
    pattern variable named like a field the function also uses hides that field there too (a limit both read alike)."""
    p, scorer = jpair
    gold = jvalues(p, scorer, JGOLD)
    use = "    cart.assertTotal(computeTotal(total));\n"
    for stmt in ("if (total instanceof String store) { store.length(); }",
                 "switch (total) { case String store -> store.length(); default -> {} }",
                 "switch (total) { case Entry(var store, var v) -> store.length(); default -> {} }"):
        got = jvalues(p, scorer, JGOLD.replace(use, use + f"    {stmt}\n"))
        assert got == gold, stmt
    extra = '    java.util.Map<String, String> m = new java.util.HashMap<>() {{ put("a", "b"); this.put("c", "d"); }};\n'
    got = jvalues(p, scorer, JGOLD.replace(use, use + extra))
    assert got == gold and ("self", "put") not in got["calls"]
    # the C# pattern-variable locals on the call side, here and in main (calls.calls), and `this.f` of an anonymous
    # Java class
    import types

    from v2tools.calls import calls
    own = lambda lang, code, avail: ind.Scorer._calls(types.SimpleNamespace(lang=lang), code, {}, avail)
    for stmt in ("if (o is string cart) { cart.Trim(); }", "var (cart, n) = (o, 1);\n    cart.Trim();",
                 "var q = from cart in items select cart.Trim();", "var q = from x in items let cart = x select cart.Trim();",
                 "var q = from x in items group x by x.K into cart select cart.Trim();"):
        code = f"public void A(object o)\n{{\n    {stmt}\n    store.Total();\n}}"
        avail = {"cart", "store", "items"}
        assert own("csharp", code, avail) == calls("csharp", code, members=avail) == {("store", "Total")}, stmt
    anon = "public void a() {\n    Map m = new HashMap<>() {{ this.store.total(); cart.add(); put(1, 2); }};\n}"
    avail = {"cart", "put", "store"}
    assert own("java", anon, avail) == calls("java", anon, members=avail) == {("cart", "add")}
