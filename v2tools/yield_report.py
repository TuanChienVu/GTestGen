"""Measure how many (scenario -> step definitions) pairs each source repository yields, partitioned by which
Behat suite (TagFilter) or SpecFlow [Scope(Tag)] span is actually active for each scenario."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .bind import Library
from .gherkin_steps import parse_feature
from .history import earliest, first_seen, line_times, min_time
from .patterns import compile_pattern
from .scopes import (NEGATIVE_STEPS, Suite, active_in_scope, is_catch_all, load_behat_suites, negative_hits,
                     specflow_scope_tags, tag_filter_matches)
from .stepdefs import extract


def structure_key(steps: list[str]) -> str:
    norm = [re.sub(r"-?\d+(?:[.,]\d+)?", "<N>", re.sub(r"\"[^\"]*\"|'[^']*'", "<S>", s)).lower().strip() for s in steps]
    return hashlib.sha1("\n".join(norm).encode()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def _classify_catch_all(defs: list) -> tuple[set, int, list[str]]:
    """Repo-wide, suite-independent: which defs are catch-all (matched every NEGATIVE_STEPS sentence -> excluded
    from every binding library) versus BROAD (matched 1-19 -> kept, but listed for manual review)."""
    catch_all: set = set()
    broad_count = 0
    broad_patterns = []
    for d in defs:
        compiled = compile_pattern(d.kind, d.pattern)
        if compiled is None:
            continue
        if is_catch_all(*compiled):
            catch_all.add(d)
        else:
            hits = negative_hits(*compiled)
            if hits > 0:
                broad_count += 1
                broad_patterns.append(f"{d.path}:{d.line} {d.pattern}")
    return catch_all, broad_count, sorted(broad_patterns)


def _behat_suites(source: dict, repo: Path) -> list[Suite] | None:
    scoping = source.get("scoping")
    if not scoping or scoping["kind"] != "behat_suites":
        return None
    suite_files = [p for p in _git(repo, "ls-files", *scoping["suite_globs"]).splitlines() if p]
    service_files = [p for p in _git(repo, "ls-files", *scoping["service_globs"]).splitlines() if p]
    return load_behat_suites(repo, suite_files, service_files, scoping["psr4"])


def _bind(scenario, library: Library) -> dict:
    """Like bind.bind_scenario, but ambiguity is judged by distinct function (path, body_start) rather than by
    matching-pattern count (several patterns on one function, e.g. [Given(".*"), When(".*"), Then(".*")] on a
    single method, are not an ambiguous match), and the result also reports which targets came from steps that
    matched exactly one function, for targets_new_count / new_target_ids. target_fns
    collapses target_ids (annotation-line ids) down to unique functions (path:body_start): several annotations
    on one function are the same answer, so the answer-leak rules must compare functions, not annotation lines."""
    targets, unambiguous_targets, target_fns, unbound, ambiguous = set(), set(), set(), [], 0
    for step in scenario.steps:
        found = library.matches(step.text)
        if not found:
            unbound.append(step.text)
        ids = {f"{d.path}:{d.line}" for d in found}
        targets.update(ids)
        target_fns.update(f"{d.path}:{d.body_start}" for d in found)
        if len({(d.path, d.body_start) for d in found}) > 1:
            ambiguous += 1
        elif ids:
            unambiguous_targets.update(ids)
    status = "UNBOUND" if len(unbound) == len(scenario.steps) else ("PARTIAL" if unbound else "BOUND")
    return {"status": status, "ambiguous_steps": ambiguous, "target_ids": sorted(targets),
            "unbound_steps": unbound, "unambiguous_target_ids": sorted(unambiguous_targets),
            "target_fns": sorted(target_fns)}


def run(config: Path, root: Path, out: Path, cutoff: str, history: bool) -> dict:
    cut = int(dt.datetime.fromisoformat(cutoff.replace("Z", "+00:00")).timestamp())
    out.mkdir(parents=True, exist_ok=True)
    rows, by_repo = [], {}
    ambig_num_by_repo: dict[str, int] = {}
    ambig_den_by_repo: dict[str, int] = {}
    for source in json.loads(config.read_text()):
        scoping = source.get("scoping") or {}
        kind = scoping.get("kind")
        if scoping and kind not in ("behat_suites", "specflow_scope"):
            raise ValueError(f"unknown scoping kind {kind!r}")
        repo = root / source["dir"]
        commit = _git(repo, "rev-parse", "HEAD").strip()
        step_files = [p for p in _git(repo, "ls-files", *source["step_globs"]).splitlines() if p]
        texts = {p: (repo / p).read_text(encoding="utf-8", errors="replace") for p in step_files}
        defs = []
        for path in step_files:
            for style in source["styles"]:
                try:
                    defs.extend(extract(path, texts[path], style))
                except ValueError:
                    continue
        full_library = Library(defs)  # unfiltered: only for uncompilable_patterns and the negative control
        catch_all, broad_defs, broad_patterns = _classify_catch_all(defs)
        defs_no_catch_all = [d for d in defs if d not in catch_all]

        behat_suites = _behat_suites(source, repo)
        specflow_spans = {p: specflow_scope_tags(p, texts[p]) for p in step_files} if kind == "specflow_scope" else {}

        def_by_id = {f"{d.path}:{d.line}": d for d in defs}
        blame_cache: dict[str, dict[int, int]] = {}
        step_lines = {p: t.splitlines() for p, t in texts.items()}
        seen_feature = first_seen(repo, ["*.feature"]) if history else {}
        seen_step = first_seen(repo, source["step_globs"]) if history else {}

        def times(path: str) -> dict[int, int]:
            if path not in blame_cache:
                blame_cache[path] = line_times(repo, path) if history else {}
            return blame_cache[path]

        def def_time(target: str, blame_only: bool = False) -> int | None:
            d = def_by_id[target]
            blame = min_time(times(d.path), range(d.line, d.body_end + 1))
            return blame if blame_only else earliest(blame, seen_step.get(step_lines[d.path][d.line - 1].strip()))

        # Suite libraries whose membership doesn't depend on the scenario, precomputed once per repo.
        fixed_libraries: dict[str, tuple[Library, Library]] = {}
        if behat_suites is not None:
            for suite in behat_suites:
                fixed_libraries[suite.name] = (Library([d for d in defs_no_catch_all if d.path in suite.files]),
                                               Library([d for d in defs if d.path in suite.files]))
        elif kind != "specflow_scope":
            fixed_libraries["default"] = (Library(defs_no_catch_all), Library(defs))

        def active_suites(scenario) -> list[tuple[str, Library, Library]]:
            if behat_suites is not None:
                return [(s.name, *fixed_libraries[s.name]) for s in behat_suites if tag_filter_matches(s.tag_filter, scenario.tags)]
            if kind == "specflow_scope":
                tags = set(scenario.tags)
                no_ca = [d for d in defs_no_catch_all if active_in_scope(d, specflow_spans.get(d.path, []), tags)]
                with_ca = [d for d in defs if active_in_scope(d, specflow_spans.get(d.path, []), tags)]
                return [("default", Library(no_ca), Library(with_ca))]
            return [("default", *fixed_libraries["default"])]

        c: Counter = Counter()
        features = [p for p in _git(repo, "ls-files", "*.feature").splitlines() if p]
        c["feature_files"] = len(features)
        c["stepdefs"] = len(defs)
        c["uncompilable_patterns"] = full_library.uncompilable
        c["catch_all_defs"] = len(catch_all)
        c["broad_defs"] = broad_defs
        c["negative_control_bound"] = sum(1 for s in NEGATIVE_STEPS if full_library.matches(s))
        structures: set = set()
        bound_scenario_keys: set = set()
        for path in features:
            feature_text = (repo / path).read_text(encoding="utf-8", errors="replace")
            feature_lines = feature_text.splitlines()
            for scenario in parse_feature(feature_text):
                if not scenario.steps:
                    continue
                c["scenarios"] += 1
                if scenario.is_outline and len({frozenset(t) for t in scenario.examples_tags}) > 1:
                    c["outlines_with_differing_examples_tags"] += 1
                steps_text = [s.text for s in scenario.steps]
                key = structure_key(steps_text)
                suites_for_scenario = active_suites(scenario)
                if not suites_for_scenario:
                    c["no_suite_scenarios"] += 1
                    rows.append({"repo": source["repo"], "commit": commit, "feature_path": path,
                                 "scenario_line": scenario.line, "scenario_name": scenario.name, "steps": steps_text,
                                 "status": "NO_SUITE", "ambiguous_steps": 0, "target_ids": [], "target_fns": [],
                                 "unbound_steps": steps_text, "structure_key": key, "scenario_min_time": None,
                                 "targets_min_time": None, "targets_new_count": 0, "new_target_ids": [],
                                 "new_target_fns": [], "suite": None, "tags": scenario.tags,
                                 "status_with_catch_all": "NO_SUITE"})
                    continue
                own_lines = [scenario.line] + [s.line for s in scenario.steps if not s.from_background]
                s_blame = min_time(times(path), own_lines) if history else None
                s_time = earliest(s_blame, seen_feature.get(feature_lines[scenario.line - 1].strip())) if history else None
                scenario_bound = False
                for suite_name, lib, lib_with_catch_all in suites_for_scenario:
                    bound = _bind(scenario, lib)
                    bound_with_catch_all = _bind(scenario, lib_with_catch_all)
                    new_target_ids = sorted(t for t in bound["unambiguous_target_ids"]
                                            if history and (tm := def_time(t)) is not None and tm > cut)
                    new_target_fns = sorted({f"{def_by_id[t].path}:{def_by_id[t].body_start}" for t in new_target_ids})
                    all_times = [def_time(t) for t in bound["target_ids"]] if history else []
                    c["pairs"] += 1
                    c[bound["status"].lower()] += 1
                    c["ambiguous_scenarios"] += bound["ambiguous_steps"] > 0
                    c["bound_with_catch_all"] += bound_with_catch_all["status"] == "BOUND"
                    ambig_num_by_repo[source["repo"]] = ambig_num_by_repo.get(source["repo"], 0) + bound["ambiguous_steps"]
                    ambig_den_by_repo[source["repo"]] = ambig_den_by_repo.get(source["repo"], 0) + len(scenario.steps)
                    if bound["status"] == "BOUND":
                        structures.add(key)
                        scenario_bound = True
                        if s_time is not None and s_time > cut:
                            c["post_cutoff_bound"] += 1
                            c["post_cutoff_bound_with_new_target"] += len(new_target_ids) > 0
                        blame_says_new = (s_blame is not None and s_blame > cut and any(
                            (tm := def_time(t, blame_only=True)) is not None and tm > cut for t in bound["unambiguous_target_ids"]))
                        c["moved_content_reclassified"] += blame_says_new and not (
                            s_time is not None and s_time > cut and new_target_ids)
                    rows.append({"repo": source["repo"], "commit": commit, "feature_path": path,
                                 "scenario_line": scenario.line, "scenario_name": scenario.name, "steps": steps_text,
                                 "status": bound["status"], "ambiguous_steps": bound["ambiguous_steps"],
                                 "target_ids": bound["target_ids"], "target_fns": bound["target_fns"],
                                 "unbound_steps": bound["unbound_steps"],
                                 "structure_key": key, "scenario_min_time": s_time,
                                 "targets_min_time": min((t for t in all_times if t is not None), default=None),
                                 "targets_new_count": len(new_target_ids), "new_target_ids": new_target_ids,
                                 "new_target_fns": new_target_fns,
                                 "suite": suite_name, "tags": scenario.tags,
                                 "status_with_catch_all": bound_with_catch_all["status"]})
                if scenario_bound:
                    bound_scenario_keys.add((path, scenario.line))
        c["unique_bound_structures"] = len(structures)
        c["bound_scenarios"] = len(bound_scenario_keys)
        c["suites"] = len(behat_suites) if behat_suites is not None else 1
        c["missing_contexts"] = len({cid for s in behat_suites for cid in s.missing_contexts}) if behat_suites is not None else 0
        c["context_args_total"] = sum(s.context_args_total for s in behat_suites) if behat_suites is not None else 0
        c["context_args_resolved"] = sum(s.context_args_resolved for s in behat_suites) if behat_suites is not None else 0
        by_repo[source["repo"]] = {k: c.get(k, 0) for k in (
            "feature_files", "scenarios", "bound", "partial", "unbound", "ambiguous_scenarios", "stepdefs",
            "uncompilable_patterns", "unique_bound_structures", "post_cutoff_bound", "post_cutoff_bound_with_new_target",
            "pairs", "no_suite_scenarios", "bound_scenarios", "bound_with_catch_all", "catch_all_defs",
            "negative_control_bound", "suites", "missing_contexts", "broad_defs", "outlines_with_differing_examples_tags",
            "context_args_total", "context_args_resolved", "moved_content_reclassified")}
        den = ambig_den_by_repo.get(source["repo"], 0)
        by_repo[source["repo"]]["ambiguous_step_rate"] = (ambig_num_by_repo.get(source["repo"], 0) / den) if den else 0.0
        by_repo[source["repo"]]["broad_patterns"] = broad_patterns
    total: dict = defaultdict(int)
    for values in by_repo.values():
        for k, v in values.items():
            if k in ("ambiguous_step_rate", "broad_patterns"):
                continue
            total[k] += v
    total_den = sum(ambig_den_by_repo.values())
    total["ambiguous_step_rate"] = (sum(ambig_num_by_repo.values()) / total_den) if total_den else 0.0
    total["broad_patterns"] = sorted(f"{repo} {p}" for repo, values in by_repo.items() for p in values["broad_patterns"])
    summary = {"cutoff_utc": cutoff, "history": history, "by_repo": by_repo, "total": dict(total)}
    (out / "pairs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "yield.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", default="2024-01-01T00:00:00Z")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()
    summary = run(args.config, args.root, args.out, args.cutoff, not args.no_history)
    print(json.dumps(summary["total"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
