"""One JSONL record per (scenario, suite) pair of train/dev/test/aux_test: verbatim Gherkin, per-step reuse/create
labels, the code a model must generate, and hidden_for() + snapshot_fns(), the single definition of which existing
functions it may be shown (Phase 3 imports both)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .scenario_text import scenario_block

FRAMEWORK = {"php": "Behat", "java": "Cucumber-JVM", "ruby": "Cucumber (Ruby)", "csharp": "SpecFlow"}


def render(fn: dict, before: int | None, keep: frozenset[str] = frozenset()) -> str:
    """Function code without the annotation lines that did not exist strictly before `before`, except the def ids in
    `keep` (the CREATE patterns of the record, which belong to the answer). `before=None` keeps every annotation."""
    drop = {a["line"] for a in fn["annotations"]
            if before is not None and a["def_id"] not in keep and not (a["time"] is not None and a["time"] < before)}
    return "\n".join(l for i, l in enumerate(fn["code"].splitlines(), start=fn["first_line"]) if i not in drop)


def hidden_for(partition: str, forbidden_repo: dict) -> set[str]:
    """Answer functions a record of `partition` must not be shown, from its repo's forbidden.json entry: train
    hides test, dev and aux answers; dev hides test and aux answers; test and aux_test hide nothing, so their snapshot
    shows whatever existed before the scenario, even another test record's answer (each record is scored alone)."""
    keys = {"train": ("test", "dev", "aux"), "dev": ("test", "aux"), "test": (), "aux_test": ()}[partition]  # KeyError: typo
    return {f for k in keys for f in forbidden_repo.get(k, ())}


def snapshot_fns(record: dict, functions: dict[str, dict], hidden: set[str]) -> list[str]:
    """Functions a model may see for this record: active in the record's suite (Behat) or scope (SpecFlow), not a
    catch-all, not dead code and not a span containing another function (catalog flags), with at least one
    pattern that existed before the scenario, and not in `hidden`. It applies no partition
    rule of its own: callers pass hidden_for(record["partition"], forbidden.get(record["repo"], {}))."""
    tags = {t.lower() for t in record["tags"]}
    out = []
    for fn_id, fn in functions.items():
        if fn["catch_all"] or fn["dead"] or fn["contains_fn"]:   # dead code and span-swallowing entries
            continue
        if record["scoping"] == "specflow_scope":
            if fn["scope_tags"] and not any(set(t) & tags for t in fn["scope_tags"]):
                continue
        elif record["suite"] not in fn["suites"]:
            continue
        if not any(a["time"] is not None and a["time"] < record["scenario_time"] for a in fn["annotations"]):
            continue
        if fn_id in hidden:
            continue
        out.append(fn_id)
    return sorted(out)


def build(config: Path, root: Path, pairs: Path, phase2: Path) -> dict:
    sources = {s["repo"]: s for s in json.loads(config.read_text())}
    catalog: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in map(json.loads, (phase2 / "catalog.jsonl").read_text(encoding="utf-8").splitlines()):
        catalog[row["repo"]][row["fn_id"]] = row
    key = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
    labels = {key(l): l for l in map(json.loads, (phase2 / "labels.jsonl").read_text(encoding="utf-8").splitlines())}
    members = {key(m): m for m in map(json.loads, (phase2 / "split_members.jsonl").read_text(encoding="utf-8").splitlines())
               if m["partition"] in ("train", "dev", "test", "aux_test")}
    forbidden = {r: {k: set(v) for k, v in parts.items()} for r, parts in json.loads((phase2 / "forbidden.json").read_text()).items()}
    fn_of = {repo: {a["def_id"]: fid for fid, fn in fns.items() for a in fn["annotations"]} for repo, fns in catalog.items()}
    time_of = {repo: {a["def_id"]: a["time"] for fn in fns.values() for a in fn["annotations"]} for repo, fns in catalog.items()}
    texts: dict[tuple, str] = {}
    out: dict[str, list[dict]] = {"train": [], "dev": [], "test": [], "aux_test": []}
    for r in map(json.loads, pairs.read_text(encoding="utf-8").splitlines()):
        m = members.get(key(r))
        if m is None:
            continue
        src, lab = sources[r["repo"]], labels[key(r)]
        fkey = (r["repo"], r["feature_path"])
        if fkey not in texts:
            texts[fkey] = (root / src["dir"] / r["feature_path"]).read_text(encoding="utf-8", errors="replace")
        block = scenario_block(texts[fkey], r["scenario_line"])
        fns = catalog[r["repo"]]
        keep = frozenset(lab["create_def_ids"])
        step_fns = []
        for ids in lab["step_def_ids"]:
            owners = {fn_of[r["repo"]][i] for i in ids}
            if len(owners) != 1:
                raise ValueError(f"record {key(r)} has a step bound to {len(owners)} functions")
            step_fns.append(owners.pop())
        rec = {"id": "|".join(map(str, key(r))), "repo": r["repo"], "commit": r["commit"], "language": src["language"],
               "framework": FRAMEWORK[src["language"]], "scoping": (src.get("scoping") or {}).get("kind"),
               "suite": r["suite"], "tags": r["tags"], "partition": m["partition"], "subsets": m["subsets"],
               "exec_core": m["exec_core"], "neighbours": None, "feature_path": r["feature_path"], "scenario_line": r["scenario_line"],
               "scenario_name": r["scenario_name"], "scenario_time": r["scenario_min_time"],
               # t_vis: the latest time of the record's CREATE patterns, the single definition of "when the code the
               # model must write became visible"; scenario_time is the scenario's OLDEST line
               "t_vis": max((t for t in (time_of[r["repo"]][d] for d in lab["create_def_ids"]) if t is not None), default=None),
               "background": block["background"], "scenario": block["scenario"], "steps": r["steps"],
               "step_kinds": lab["step_kinds"], "step_fns": step_fns, "reuse_fns": lab["reuse_fns"],
               "create_def_ids": sorted(lab["create_def_ids"]),
               "create": [{"fn_id": f, "path": fns[f]["path"], "code": render(fns[f], r["scenario_min_time"], keep),
                           # True: the function already existed (older patterns) and this scenario only adds a pattern,
                           # so its earlier body may legitimately appear in a snapshot; False: a brand-new function
                           "preexisting": any(a["time"] is not None and a["time"] < r["scenario_min_time"]
                                              for a in fns[f]["annotations"]),
                           # True: Phase 1 dated this function's code after the cutoff (it is in the pair's new_target_fns)
                           "post_cutoff_code": f in set(r["new_target_fns"])}
                          for f in lab["create_fns"]]}
        rec["snapshot_size"] = len(snapshot_fns(rec, fns, hidden_for(m["partition"], forbidden.get(r["repo"], {}))))
        out[m["partition"]].append(rec)
    # neighbours (train only; null elsewhere): sorted ids of the OTHER train records in the same feature file or sharing at
    # least one CREATE function — direct neighbours, not transitive. Phase 3 keeps them out of a record's
    # example pool; a transitive co-creation group would swallow most of a repo's train records.
    by_file, by_fn = defaultdict(set), defaultdict(set)
    for rec in out["train"]:
        by_file[(rec["repo"], rec["feature_path"])].add(rec["id"])
        for x in rec["create"]:
            by_fn[(rec["repo"], x["fn_id"])].add(rec["id"])
    for rec in out["train"]:
        near = set(by_file[(rec["repo"], rec["feature_path"])])
        for x in rec["create"]:
            near |= by_fn[(rec["repo"], x["fn_id"])]
        rec["neighbours"] = sorted(near - {rec["id"]})
    (phase2 / "records").mkdir(parents=True, exist_ok=True)
    for part, recs in out.items():
        recs.sort(key=lambda x: x["id"])
        (phase2 / "records" / f"{part}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs),
                                                          encoding="utf-8")
    return {p: len(v) for p, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--root", type=Path, default=Path("repos-hist"))
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    args = ap.parse_args()
    print(json.dumps(build(args.config, args.root, args.pairs, args.phase2), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
