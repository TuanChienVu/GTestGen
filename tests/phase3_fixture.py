"""A tiny PHP repo with dated commits, its catalog rows and records, shared by the Phase 3 tests.

History of src/Behat/Context/ShopContext.php (BaseContext.php exists from the first commit):
  2020-01-01  iOpenTheShop, promoted $indexPage
  2022-01-01  iAddToCart, promoted $cart
  2024-06-01  iShouldSeeTotal (the test answer), promoted $checkout, private helper computeTotal
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

T2020, T2022, T2024 = 1577836800, 1640995200, 1717200000

BASE = '''<?php

namespace App\\Behat\\Context;

abstract class BaseContext
{
    protected $sharedStorage;

    protected function helperOld(): void
    {
    }

    private function secret(): void
    {
    }
}
'''

SHOP_2020 = '''<?php

namespace App\\Behat\\Context;

use App\\Page\\IndexPage;
use Behat\\Step\\{Given, Then, When};
final class ShopContext extends BaseContext
{
    public function __construct(
        private IndexPage $indexPage,
    ) {
    }

    #[Given('I open the shop')]
    public function iOpenTheShop(): void
    {
        $this->indexPage->open();
    }
}
'''

SHOP_2022 = SHOP_2020.replace('''        private IndexPage $indexPage,
''', '''        private IndexPage $indexPage,
        private Cart $cart,
''').replace('''        $this->indexPage->open();
    }
''', '''        $this->indexPage->open();
    }

    #[When('I add :product to the cart')]
    public function iAddToCart(string $product): void
    {
        $this->cart->add($product);
    }
''')

SHOP_2024 = SHOP_2022.replace('''        private Cart $cart,
''', '''        private Cart $cart,
        private Checkout $checkout,
''').replace('''        $this->cart->add($product);
    }
''', '''        $this->cart->add($product);
    }

    #[Then('I should see the total :total')]
    public function iShouldSeeTotal(string $total): void
    {
        $this->checkout->assertTotal($this->computeTotal($total));
    }

    private function computeTotal(string $total): string
    {
        return $total;
    }
''')

COMPOSER = '{"autoload": {"psr-4": {"App\\\\Behat\\\\": "src/Behat/"}}}\n'
SHOP = "src/Behat/Context/ShopContext.php"


def _git(repo: Path, *args: str, date: int | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    if date is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{date} +0000"
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def make_repo(tmp: Path) -> tuple[Path, str]:
    """Commit the three versions and return (bare clone path, HEAD commit)."""
    work = tmp / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    for date, files in [(T2020, {"composer.json": COMPOSER, "src/Behat/Context/BaseContext.php": BASE, SHOP: SHOP_2020}),
                        (T2022, {SHOP: SHOP_2022}), (T2024, {SHOP: SHOP_2024})]:
        for p, text in files.items():
            (work / p).parent.mkdir(parents=True, exist_ok=True)
            (work / p).write_text(text)
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"c{date}", date=date)
    bare = tmp / "repo.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True)
    return bare, _git(bare, "rev-parse", "HEAD").strip()


def _fn(fn_id_line: int, name: str, keyword: str, pattern: str, time: int, text: str = SHOP_2024) -> dict:
    lines = text.splitlines()
    ann = next(i for i, l in enumerate(lines, 1) if f"'{pattern}'" in l)
    end = next(i for i in range(fn_id_line, len(lines) + 1) if lines[i - 1] == "    }")
    code = "\n".join(l[4:] for l in lines[ann - 1:end])
    return {"repo": "Acme/Shop", "fn_id": f"{SHOP}:{fn_id_line}", "path": SHOP, "first_line": ann, "body_end": end,
            "code": code, "annotations": [{"def_id": f"{SHOP}:{ann}", "line": ann, "keyword": keyword,
                                           "pattern": pattern, "time": time}],
            "suites": ["s"], "scope_tags": [], "catch_all": False, "dead": False, "contains_fn": False}


def catalog() -> dict[str, dict]:
    lines = SHOP_2024.splitlines()
    body = {n: next(i for i, l in enumerate(lines, 1) if f"function {n}(" in l) + 1
            for n in ("iOpenTheShop", "iAddToCart", "iShouldSeeTotal")}
    rows = [_fn(body["iOpenTheShop"], "iOpenTheShop", "Given", "I open the shop", T2020),
            _fn(body["iAddToCart"], "iAddToCart", "When", "I add :product to the cart", T2022),
            _fn(body["iShouldSeeTotal"], "iShouldSeeTotal", "Then", "I should see the total :total", T2024)]
    return {r["fn_id"]: r for r in rows}


def record(cat: dict[str, dict], partition: str = "test", rid: str = "Acme/Shop|features/total.feature|3|s") -> dict:
    answer = next(f for f in cat.values() if "iShouldSeeTotal" in f["code"])
    return {"id": rid, "repo": "Acme/Shop", "commit": None, "language": "php", "framework": "Behat",
            "scoping": None, "suite": "s", "tags": [], "partition": partition, "subsets": [], "exec_core": False,
            "neighbours": None, "feature_path": rid.split("|")[1], "scenario_line": 3,
            "scenario_name": "Total", "scenario_time": T2024, "t_vis": T2024,
            "background": "", "scenario": "Scenario: Total\n  Given I open the shop\n  Then I should see the total \"5\"",
            "steps": ["I open the shop", "I should see the total \"5\""], "step_kinds": ["reuse", "create"],
            "step_fns": [f"{SHOP}:{next(iter(cat)).rsplit(':', 1)[1]}", answer["fn_id"]], "reuse_fns": [],
            "create": [{"fn_id": answer["fn_id"], "path": SHOP, "code": answer["code"], "preexisting": False,
                        "post_cutoff_code": True}],
            "create_def_ids": [answer["annotations"][0]["def_id"]], "snapshot_size": 2}
