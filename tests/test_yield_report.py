import json
import subprocess
import sys
from pathlib import Path

import pytest

from v2tools import yield_report
from v2tools.yield_report import structure_key

def test_structure_key_ignores_values_only():
    a = structure_key(['there is a product "Mug" priced at 10', "I check out"])
    b = structure_key(["there is a product 'Cup' priced at 25", "I check out"])
    c = structure_key(['there is a product "Mug"', "I cancel"])
    assert a == b and a != c

def test_cli_end_to_end_on_a_tiny_repo(tmp_path: Path):
    repo = tmp_path / "hist" / "demo"
    (repo / "features").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "features" / "cart.feature").write_text(
        'Feature: Cart\n  Scenario: buy\n    Given there is a product "Mug"\n    When I check out\n'
        '  Scenario: fly\n    Given I fly\n')
    (repo / "src" / "CartContext.php").write_text(
        "<?php\nclass CartContext {\n    #[Given('there is a product :name')]\n    public function p($n) { }\n"
        "    #[When('I check out')]\n    public function c() { }\n}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    config = tmp_path / "sources.json"
    config.write_text(json.dumps([{"dir": "demo", "repo": "x/demo", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT"}]))
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "v2tools.yield_report", "--config", str(config), "--root", str(tmp_path / "hist"),
                    "--out", str(out), "--cutoff", "2000-01-01T00:00:00Z"], check=True, cwd=Path(__file__).resolve().parents[1])
    summary = json.loads((out / "yield.json").read_text())["by_repo"]["x/demo"]
    assert (summary["scenarios"], summary["bound"], summary["unbound"], summary["stepdefs"]) == (2, 1, 1, 2)
    assert summary["post_cutoff_bound_with_new_target"] == 1
    rows = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    assert rows[0]["target_ids"] == ["src/CartContext.php:3", "src/CartContext.php:5"]
    assert rows[0]["new_target_ids"] == rows[0]["target_ids"]

def test_cli_two_annotations_on_one_function_share_target_fns(tmp_path: Path):
    repo = tmp_path / "hist" / "twoattr"
    (repo / "features").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "features" / "f.feature").write_text(
        "Feature: F\n  Scenario: a\n    When I do a\n  Scenario: b\n    When I do b\n")
    # one PHP function carrying two #[When(...)] attributes -- 887 such functions were found across the real
    # repos (Sylius 853, PrestaShop 33, gpconnect 1); each attribute has its own annotation line (target_ids),
    # but both are the same answer function (target_fns).
    (repo / "src" / "C.php").write_text(
        "<?php\nclass C {\n    #[When('I do a')]\n    #[When('I do b')]\n    public function doIt() { }\n}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    config = tmp_path / "sources.json"
    config.write_text(json.dumps([{"dir": "twoattr", "repo": "x/twoattr", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT"}]))
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "v2tools.yield_report", "--config", str(config), "--root", str(tmp_path / "hist"),
                    "--out", str(out), "--cutoff", "2000-01-01T00:00:00Z"], check=True, cwd=Path(__file__).resolve().parents[1])
    rows = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    a = next(r for r in rows if r["scenario_name"] == "a")
    b = next(r for r in rows if r["scenario_name"] == "b")
    assert a["target_ids"] == ["src/C.php:3"] and b["target_ids"] == ["src/C.php:4"]
    assert a["target_fns"] == b["target_fns"] == ["src/C.php:5"]
    assert a["new_target_fns"] == b["new_target_fns"] == ["src/C.php:5"]

def test_cli_models_behat_suites_catch_alls_and_ambiguity(tmp_path: Path):
    repo = tmp_path / "hist" / "shop"
    files = {
        "features/countries.feature": ("@managing_countries\nFeature: C\n  @ui @api\n  Scenario: add\n    When I add it\n"
                                       "    Then it exists\n  @cli\n  Scenario: other\n    When I add it\n"),
        "src/Behat/Context/Ui/CountryContext.php": ("<?php\nclass CountryContext {\n    #[When('I add it')]\n    public function add() { }\n"
                                                    "    #[Then('it exists')]\n    public function exists() { }\n"
                                                    "    #[Given('/^.*$/')]\n    public function anything() { }\n}\n"),
        "src/Behat/Context/Api/CountryContext.php": ("<?php\nclass CountryContext {\n    #[When('I add it')]\n    public function add() { }\n"
                                                     "    #[Then('it exists')]\n    public function exists() { }\n"
                                                     "    #[Given('/.*[0-9]+.*/')]\n    public function broad() { }\n}\n"),
        "cfg/suites/ui.php": ("<?php (new Suite('ui_countries'))->withContexts('ctx.ui')->withContexts('ctx.missing')"
                             "->withFilter(new TagFilter('@managing_countries&&@ui'));\n"),
        "cfg/suites/api.php": ("<?php (new Suite('api_countries'))->withContexts('ctx.api')->withContexts('ctx.missing')"
                              "->withFilter(new TagFilter('@managing_countries&&@api'));\n"),
        "cfg/services.php": ("<?php\nuse Shop\\Behat\\Context\\Ui\\CountryContext as UiC;\nuse Shop\\Behat\\Context\\Api\\CountryContext as ApiC;\n"
                             "$s->set('ctx.ui', UiC::class)->set('ctx.api', ApiC::class);\n"),
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    config = tmp_path / "sources.json"
    config.write_text(json.dumps([{"dir": "shop", "repo": "x/shop", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT",
                                   "scoping": {"kind": "behat_suites", "suite_globs": ["cfg/suites/*.php"],
                                               "service_globs": ["cfg/services.php"], "psr4": {"Shop\\Behat\\": "src/Behat/"}}}]))
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "v2tools.yield_report", "--config", str(config), "--root", str(tmp_path / "hist"),
                    "--out", str(out), "--no-history"], check=True, cwd=Path(__file__).resolve().parents[1])
    s = json.loads((out / "yield.json").read_text())["by_repo"]["x/shop"]
    assert (s["scenarios"], s["pairs"], s["no_suite_scenarios"], s["bound"], s["bound_scenarios"]) == (2, 2, 1, 2, 1)
    assert (s["catch_all_defs"], s["bound_with_catch_all"], s["ambiguous_step_rate"], s["suites"]) == (1, 2, 0.0, 2)
    assert s["negative_control_bound"] == 20
    rows = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    assert sorted(r["suite"] for r in rows if r["suite"]) == ["api_countries", "ui_countries"]
    ui = next(r for r in rows if r["suite"] == "ui_countries")
    assert ui["target_ids"] == ["src/Behat/Context/Ui/CountryContext.php:3", "src/Behat/Context/Ui/CountryContext.php:5"]
    assert ui["tags"] == ["@managing_countries", "@ui", "@api"] and ui["ambiguous_steps"] == 0
    assert next(r for r in rows if r["suite"] is None)["status"] == "NO_SUITE"
    # 'ctx.missing' is unresolvable in BOTH suites -> a single distinct id, not two occurrences.
    assert s["missing_contexts"] == 1
    # a raw ->withContexts arg count independent of what the parser recognised: 1 resolved id + 1
    # unresolved id per suite, over 2 suites.
    assert (s["context_args_total"], s["context_args_resolved"]) == (4, 2)
    # the API context's extra regex_full def matches some but not all NEGATIVE_STEPS (kept, not catch-all,
    # just flagged for manual review) -- also listed in total.broad_patterns.
    assert s["broad_defs"] == 1
    assert s["broad_patterns"] == ["src/Behat/Context/Api/CountryContext.php:7 .*[0-9]+.*"]
    total = json.loads((out / "yield.json").read_text())["total"]
    assert total["broad_patterns"] == ["x/shop src/Behat/Context/Api/CountryContext.php:7 .*[0-9]+.*"]

def test_unknown_scoping_kind_raises(tmp_path: Path):
    config = tmp_path / "sources_typo.json"
    config.write_text(json.dumps([{"dir": "demo2", "repo": "x/demo2", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT",
                                   "scoping": {"kind": "behat_suite"}}]))  # typo: missing trailing 's'
    with pytest.raises(ValueError):
        yield_report.run(config, tmp_path / "hist", tmp_path / "out_typo", "2024-01-01T00:00:00Z", False)

def test_cli_excludes_ambiguous_step_targets_from_new_target_ids(tmp_path: Path):
    repo = tmp_path / "hist" / "amb"
    (repo / "features").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "features" / "f.feature").write_text("Feature: F\n  Scenario: s\n    Given I do it\n    Then I finish\n")
    (repo / "src" / "C.php").write_text(
        "<?php\nclass C {\n    #[Given('I do it')]\n    public function a() { }\n"
        "    #[Given('/^I do it$/')]\n    public function b() { }\n"
        "    #[Then('I finish')]\n    public function c() { }\n}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    config = tmp_path / "sources_amb.json"
    config.write_text(json.dumps([{"dir": "amb", "repo": "x/amb", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT"}]))
    out = tmp_path / "out_amb"
    subprocess.run([sys.executable, "-m", "v2tools.yield_report", "--config", str(config), "--root", str(tmp_path / "hist"),
                    "--out", str(out), "--cutoff", "2000-01-01T00:00:00Z"], check=True, cwd=Path(__file__).resolve().parents[1])
    rows = [json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines()]
    row = rows[0]
    assert row["status"] == "BOUND" and row["ambiguous_steps"] == 1
    # both "I do it" (ambiguous: 2 distinct functions) and "I finish" (unambiguous) targets are new by time,
    # but only the unambiguous one is reported as a new target.
    assert row["target_ids"] == ["src/C.php:3", "src/C.php:5", "src/C.php:7"]
    assert row["new_target_ids"] == ["src/C.php:7"]
    assert row["targets_new_count"] == 1

def test_cli_dates_split_content_by_first_appearance(tmp_path: Path):
    import os
    repo = tmp_path / "hist" / "demo"
    (repo / "features").mkdir(parents=True)
    (repo / "src").mkdir()

    def commit(date):
        env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", date], cwd=repo, check=True, env=env)

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "features" / "all.feature").write_text(
        "Feature: All\n  Scenario: keep\n    When I keep\n  Scenario: move\n    When I move\n")
    (repo / "src" / "AContext.php").write_text("<?php\nclass AContext {\n    #[When('I keep')]\n    public function k() { }\n"
                                               "    #[When('I move')]\n    public function m() { }\n}\n")
    commit("2022-01-01T00:00:00Z")
    (repo / "features" / "all.feature").write_text("Feature: All\n  Scenario: keep\n    When I keep\n")
    (repo / "features" / "moved.feature").write_text("Feature: Moved\n  Scenario: move\n    When I move\n")
    (repo / "src" / "AContext.php").write_text("<?php\nclass AContext {\n    #[When('I keep')]\n    public function k() { }\n}\n")
    (repo / "src" / "BContext.php").write_text("<?php\nclass BContext {\n    #[When('I move')]\n    public function m() { }\n}\n")
    commit("2025-01-01T00:00:00Z")
    config = tmp_path / "sources.json"
    config.write_text(json.dumps([{"dir": "demo", "repo": "x/demo", "branch": "main", "language": "php",
                                   "styles": ["php_attribute"], "step_globs": ["src/*.php"], "license_spdx": "MIT"}]))
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "v2tools.yield_report", "--config", str(config), "--root", str(tmp_path / "hist"),
                    "--out", str(out), "--cutoff", "2024-01-01T00:00:00Z"], check=True, cwd=Path(__file__).resolve().parents[1])
    s = json.loads((out / "yield.json").read_text())["by_repo"]["x/demo"]
    assert (s["bound"], s["post_cutoff_bound"], s["moved_content_reclassified"]) == (2, 0, 1)
    moved = next(json.loads(l) for l in (out / "pairs.jsonl").read_text().splitlines() if '"move"' in l)
    assert moved["scenario_min_time"] == 1640995200 and moved["new_target_ids"] == []
