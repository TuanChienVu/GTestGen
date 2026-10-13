#!/usr/bin/env python3
"""Wrong-suite negative control: for a behat_suites source, bind each scenario's steps against a suite it is
NOT active in (v2tools.scopes.pick_wrong_suite), to show that suite filtering actually discriminates -- a
scenario should mostly NOT bind against a suite it wasn't routed to. Reuses yield_report's own def-extraction,
catch-all filtering and binding (_git, _behat_suites, _classify_catch_all, _bind) rather than duplicating them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # running a script puts scripts/ on sys.path, not the repo root

from v2tools.bind import Library
from v2tools.gherkin_steps import parse_feature
from v2tools.scopes import pick_wrong_suite, tag_filter_matches
from v2tools.stepdefs import extract
from v2tools.yield_report import _behat_suites, _bind, _classify_catch_all, _git


def run(config: Path, root: Path, repo_name: str, out: Path) -> dict:
    sources = json.loads(config.read_text())
    source = next((s for s in sources if s["repo"] == repo_name), None)
    if source is None:
        raise SystemExit(f"no source with repo {repo_name!r} in {config}")
    scoping = source.get("scoping") or {}
    if scoping.get("kind") != "behat_suites":
        raise SystemExit(f"{repo_name} has no behat_suites scoping in {config}; nothing to negative-control")

    repo = root / source["dir"]
    behat_suites = _behat_suites(source, repo)
    if not behat_suites:
        raise SystemExit(f"{repo_name}: load_behat_suites found no suites under {repo}")
    all_names = [s.name for s in behat_suites]

    step_files = [p for p in _git(repo, "ls-files", *source["step_globs"]).splitlines() if p]
    defs = []
    for path in step_files:
        text = (repo / path).read_text(encoding="utf-8", errors="replace")
        for style in source["styles"]:
            try:
                defs.extend(extract(path, text, style))
            except ValueError:
                continue
    catch_all, _, _ = _classify_catch_all(defs)
    defs_no_catch_all = [d for d in defs if d not in catch_all]
    libraries = {s.name: Library([d for d in defs_no_catch_all if d.path in s.files]) for s in behat_suites}

    features = [p for p in _git(repo, "ls-files", "*.feature").splitlines() if p]
    tested = 0
    wrong_bound = 0
    by_prefix: Counter = Counter()
    for path in features:
        for scenario in parse_feature((repo / path).read_text(encoding="utf-8", errors="replace")):
            if not scenario.steps:
                continue
            active = [s.name for s in behat_suites if tag_filter_matches(s.tag_filter, scenario.tags)]
            if not active:
                continue
            wrong = pick_wrong_suite(active, all_names)
            if wrong is None:
                continue
            tested += 1
            if _bind(scenario, libraries[wrong])["status"] == "BOUND":
                wrong_bound += 1
                by_prefix[wrong.split("_", 1)[0]] += 1

    result = {"repo": repo_name, "scenarios_tested": tested, "wrong_suite_bound": wrong_bound,
              "wrong_suite_bound_rate": (wrong_bound / tested) if tested else 0.0,
              "by_wrong_suite_prefix": dict(sorted(by_prefix.items()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.config, args.root, args.repo, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
