"""Small dated repos for tests/test_pre_t_pins.py (rule 5 shows the PRE-T FORM of a function changed at or after T),
built like tests/phase3_chain_fixture.py.

PHP, src/Ctx/CartContext.php (namespace App\\Ctx), a promoted `private Cart $cart` from the start:
  2020  iHaveACart #[Given('I have a cart')]; iSeeTheCart #[Then('I see the cart')] and #[Then('I check out')];
        iPay annotated by a docblock `@When I pay :amount`; iEmptyTheCart #[When('I empty the cart')]
  2021  iHaveACart gets a second annotation #[Given('there is a cart')]
  2022  #[Then('I check out')] moves from iSeeTheCart to the new iCheckOut (the queries' answer)
  2024  a cosmetic refactor adds `: void` to every step method; iEmptyTheCart also calls the new private fresh()
Java, src/test/java/org/x/: LambdaSteps (2020, a cucumber-java8 lambda step inside the constructor, its body edited in
2024), GoSteps (2020, two go() overloads, the first with a Javadoc, both edited in 2024), EatSteps (2022, the answer).
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from tests.phase3_chain_fixture import COMPOSER, T2021, build
from tests.phase3_fixture import T2020, T2022, T2024
from v2tools.records import render

REPO, JREPO = "Acme/Cart", "Acme/Apples"
CART = "src/Ctx/CartContext.php"
LAMBDA, GO, EAT = (f"src/test/java/org/x/{n}.java" for n in ("LambdaSteps", "GoSteps", "EatSteps"))
_ANN = re.compile(r"#\[(\w+)\('(.*)'\)\]|\* @(\w+) (.*)|@(\w+)\(\"(.*)\"\)")


def _cart(void: str, there: bool, moved: bool, fresh: bool) -> str:
    def method(anns: str, sig: str, body: str) -> str:
        return f"{anns}    public function {sig}{void}\n    {{\n        {body}\n    }}\n"
    body = "    public function __construct(private Cart $cart)\n    {\n    }\n\n" \
        + method("    #[Given('I have a cart')]\n" + ("    #[Given('there is a cart')]\n" if there else ""), "iHaveACart()",
                 "$this->cart->create();") + "\n" \
        + method("    #[Then('I see the cart')]\n" + ("" if moved else "    #[Then('I check out')]\n"), "iSeeTheCart()",
                 "$this->cart->show();") + "\n" \
        + method("    /**\n     * @When I pay :amount\n     */\n", "iPay(string $amount)", "$this->cart->pay($amount);") \
        + "\n" + method("    #[When('I empty the cart')]\n", "iEmptyTheCart()",
                        "$this->cart->clear($this->fresh());" if fresh else "$this->cart->clear();")
    if moved:
        body += "\n" + method("    #[Then('I check out')]\n", "iCheckOut()", "$this->cart->checkout();")
    if fresh:
        body += "\n    private function fresh(): bool\n    {\n        return true;\n    }\n"
    return f"<?php\n\nnamespace App\\Ctx;\n\nfinal class CartContext\n{{\n{body}}}\n"


CART_2020, CART_2021, CART_2022 = _cart("", False, False, False), _cart("", True, False, False), _cart("", True, True, False)
CART_2024 = _cart(": void", True, True, True)


def php_repo(tmp: Path) -> tuple[Path, str]:
    return build(tmp, [(T2020, T2020, {"composer.json": COMPOSER, CART: CART_2020}), (T2021, T2021, {CART: CART_2021}),
                       (T2022, T2022, {CART: CART_2022}), (T2024, T2024, {CART: CART_2024})])


LAMBDA_2020 = '''package org.x;

import io.cucumber.java8.En;

public class LambdaSteps implements En {
    private int count;

    public LambdaSteps() {
        Given("I have {int} apples", (Integer n) -> {
            this.count = n;
        });
    }
}
'''
LAMBDA_2024 = LAMBDA_2020.replace("this.count = n;", "this.count = n * 1;")
GO_2020 = '''package org.x;

public class GoSteps {
    /**
     * Goes once.
     */
    @When("I go")
    public void go() {
        run(0);
    }

    @When("I go {int} times")
    public void go(int n) {
        run(n);
    }

    private void run(int n) {
    }
}
'''
GO_2024 = GO_2020.replace("run(0);", "run(0, true);").replace("run(n);", "run(n, true);").replace(
    "run(int n)", "run(int n, boolean log)")
EAT_2022 = '''package org.x;

public class EatSteps {
    @Then("I eat {int} apples")
    public void iEatApples(int n) {
        System.out.println(n);
    }
}
'''


def java_repo(tmp: Path) -> tuple[Path, str]:
    return build(tmp, [(T2020, T2020, {LAMBDA: LAMBDA_2020, GO: GO_2020}), (T2022, T2022, {EAT: EAT_2022}),
                       (T2024, T2024, {LAMBDA: LAMBDA_2024, GO: GO_2024})])


def row(repo: str, path: str, text: str, decl: int, times: list[int]) -> dict:
    """The catalog row of the step whose declaration (or lambda call) is on line `decl` of `text` (HEAD), built as
    catalog.build_catalog does: code from its first annotation line, climbed to the opener of a docblock that holds it,
    to its closing brace, dedented; one annotation per entry of `times`, in order."""
    lines = text.splitlines()
    top = decl
    while lines[top - 2].strip().startswith(("#[", "@", "/**", "*")):
        top -= 1
    anns = [(i, [g for g in m.groups() if g is not None]) for i in range(top, decl + 1)
            if (m := _ANN.search(lines[i - 1])) or (m := re.search(r"(Given)\(\"(.*?)\"", lines[i - 1]))]
    first = anns[0][0]
    while lines[first - 2].strip().startswith("/**") or re.match(r"\s*\*(?!/)", lines[first - 2]):
        first -= 1
    body = next(i for i in range(decl, len(lines) + 1) if "{" in lines[i - 1])
    indent = re.match(r"\s*", lines[decl - 1]).group(0)
    end = next(i for i in range(body + 1, len(lines) + 1) if lines[i - 1] in (indent + "}", indent + "});"))
    return {"repo": repo, "fn_id": f"{path}:{body}", "path": path, "first_line": first, "body_end": end,
            "code": textwrap.dedent("\n".join(lines[first - 1:end])),
            "annotations": [{"def_id": f"{path}:{i}", "line": i, "keyword": kw, "pattern": p, "time": t}
                            for (i, (kw, p)), t in zip(anns, times)],
            "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}


def _decl(text: str, needle: str, nth: int = 0) -> int:
    return [i for i, l in enumerate(text.splitlines(), 1) if needle in l][nth]


def php_catalog() -> dict[str, dict]:
    rows = [row(REPO, CART, CART_2024, _decl(CART_2024, f"function {n}("), times) for n, times in (
        ("iHaveACart", [T2020, T2021]), ("iSeeTheCart", [T2020]), ("iPay", [T2020]), ("iEmptyTheCart", [T2020]),
        ("iCheckOut", [T2022]))]
    return {r["fn_id"]: r for r in rows}


def java_catalog() -> dict[str, dict]:
    rows = [row(JREPO, LAMBDA, LAMBDA_2024, _decl(LAMBDA_2024, 'Given("'), [T2020]),
            row(JREPO, GO, GO_2024, _decl(GO_2024, "public void go("), [T2020]),
            row(JREPO, GO, GO_2024, _decl(GO_2024, "public void go(", 1), [T2020]),
            row(JREPO, EAT, EAT_2022, _decl(EAT_2022, "public void iEatApples("), [T2022])]
    return {r["fn_id"]: r for r in rows}


def fid(cat: dict, needle: str) -> str:
    return next(f for f, r in cat.items() if needle in r["code"])


def record(cat: dict, name: str, items: list[tuple[str, bool, list[str]]], t_vis: int, scenario_time: int,
           partition: str, steps: list[str], language: str = "php") -> dict:
    """A record of features/<name>.feature; `items` = (fn_id, preexisting, its CREATE patterns)."""
    repo = next(iter(cat.values()))["repo"]
    cdefs = sorted(a["def_id"] for f, _, ps in items for a in cat[f]["annotations"] if a["pattern"] in ps)
    return {"id": f"{repo}|features/{name}.feature|1|s", "repo": repo, "commit": None, "language": language,
            "framework": "Behat" if language == "php" else "Cucumber-JVM", "scoping": None, "suite": "s", "tags": [],
            "partition": partition, "subsets": [], "exec_core": False,
            "neighbours": [] if partition == "train" else None, "feature_path": f"features/{name}.feature",
            "scenario_line": 1, "scenario_name": name, "scenario_time": scenario_time, "t_vis": t_vis,
            "background": "", "scenario": "Scenario: x\n" + "\n".join(f"  Given {s}" for s in steps), "steps": steps,
            "step_kinds": ["create"] * len(steps), "step_fns": [items[0][0]] * len(steps), "reuse_fns": [],
            "create": [{"fn_id": f, "path": cat[f]["path"], "preexisting": pre,
                        "code": render(cat[f], scenario_time, frozenset(cdefs))} for f, pre, _ in items],
            "create_def_ids": cdefs, "snapshot_size": 0}


def query(cat: dict) -> dict:
    """The test record whose answer is iCheckOut, at T2022."""
    return record(cat, "checkout", [(fid(cat, "iCheckOut"), False, ["I check out"])], T2022, T2022, "test",
                  ["I check out"])


def query_with_preexisting(cat: dict) -> dict:
    """A test record at T2022 that adds 'there is a cart' to the preexisting iHaveACart and creates iCheckOut."""
    return record(cat, "both", [(fid(cat, "iHaveACart"), True, ["there is a cart"]),
                                (fid(cat, "iCheckOut"), False, ["I check out"])], T2022, T2021 + 1, "test",
                  ["there is a cart", "I check out"])


def example(cat: dict, name: str, pattern: str, t: int = T2020) -> dict:
    """A train record at t (t_vis = scenario_time) that creates the function `name` with its pattern."""
    return record(cat, name, [(fid(cat, f"function {name}("), False, [pattern])], t, t, "train", [pattern])


def java_query(cat: dict) -> dict:
    return record(cat, "eat", [(fid(cat, "iEatApples"), False, ["I eat {int} apples"])], T2022, T2022, "test",
                  ["I eat 3 apples"], "java")
