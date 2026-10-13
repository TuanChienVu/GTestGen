"""The final split (fixed 23/09/2026): stratified test and dev, train = everything clean that is left, nested
learning-curve subsets. Output files use the Phase 1 split format so scripts/check_split.py re-checks them."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .body import body_key, body_key_nosig, containments, near_duplicates, nosig_text

PARTS = ("train", "dev", "test")
REASONS = ("ambiguous", "excluded_tag", "no_create", "duplicate_structure", "test_sibling", "same_file_as_test",
           "uses_test_answer", "uses_aux_answer", "uses_dev_answer")
# Dev is drawn in whole CO-CREATION GROUPS: scenarios joined by the same feature file or by a shared CREATE function
# (scenarios added in one commit all CREATE its new steps). A group never straddles dev and train, so no train target
# contains a dev answer; train snapshots additionally hide dev answers (forbidden.json "dev").


def _h(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    s = sum(weights.values())
    if s == 0 or total <= 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(weights, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return {k: min(out[k], weights[k]) for k in weights}


def select(pairs: list[dict], labels: dict[tuple, dict], cfg: dict, catalog: dict[str, dict[str, dict]] | None = None,
           languages: dict[str, str] | None = None) -> dict:
    cut = int(dt.datetime.fromisoformat(cfg["cutoff_utc"].replace("Z", "+00:00")).timestamp())
    seed = cfg["seed"]
    excluded = {r: {t.lower() for t in ts} for r, ts in cfg["excluded_tags"].items()}
    # repos whose CI selects scenarios by INCLUSION tags: a scenario without any of them never runs in CI
    required = {r: {t.lower() for t in ts} for r, ts in cfg.get("required_tags_any", {}).items()}
    pid = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
    aside: dict[str, list[dict]] = {k: [] for k in REASONS}
    scenarios: dict[tuple, list[dict]] = {}
    for r in pairs:
        if r["status"] != "BOUND":
            continue
        lab = labels[pid(r)]
        tags = {t.lower() for t in r["tags"]}
        if r["ambiguous_steps"] > 0:
            aside["ambiguous"].append(r)
        elif tags & excluded.get(r["repo"], set()) or (r["repo"] in required and not tags & required[r["repo"]]):
            aside["excluded_tag"].append(r)  # not run by the repo's CI: an excluded tag, or none of the required tags
        elif not lab["create_fns"]:
            aside["no_create"].append(r)
        else:
            scenarios.setdefault(pid(r)[:3], []).append(r)
    kept, seen = {}, set()
    for sid, rs in scenarios.items():
        if rs[0]["structure_key"] in seen:
            aside["duplicate_structure"].extend(rs)
        else:
            seen.add(rs[0]["structure_key"])
            kept[sid] = rs

    def eligible(r: dict) -> bool:  # post-cutoff scenario whose CREATE part contains post-cutoff code
        return (r["scenario_min_time"] or 0) > cut and bool(set(labels[pid(r)]["create_def_ids"]) & set(r["new_target_ids"]))

    exec_prefix = cfg["exec_suite_prefix"]
    candidates = defaultdict(list)
    for sid, rs in kept.items():
        ok = [r for r in rs if eligible(r)]
        if not ok:
            continue
        if sid[0] in exec_prefix and not any(r["suite"].startswith(exec_prefix[sid[0]]) for r in ok):
            continue
        candidates[sid[0]].append(sid)
    test = {}
    for repo, quota in cfg["test_quota"].items():
        for sid in sorted(candidates.get(repo, []), key=lambda s: _h(seed, "test", *s))[:quota]:
            test[sid] = [r for r in kept[sid] if eligible(r)]
            aside["test_sibling"].extend(r for r in kept[sid] if not eligible(r))
    test_files = {sid[:2] for sid in test}
    answer = {(r["repo"], f) for rs in test.values() for r in rs
              for f in set(r["new_target_fns"]) | set(labels[pid(r)]["create_fns"])}
    # Twins: another function of the same repo with the body_key OR the body_key_nosig of a test answer is
    # a copy of it (the second key catches renamed copies). A twin whose earliest dated annotation is after the cutoff is
    # post-cutoff code of that answer and joins it, so uses_test_answer and forbidden["test"] cover it. Earlier twins (of
    # test, dev or aux answers) are legitimate past code: kept, and only described in DATASET.md. Without a catalog there
    # are no twins; `languages` (repo -> language) tells body_key_nosig which repos are Ruby.
    keys: dict[tuple, list[tuple]] = defaultdict(list)
    by_key = defaultdict(set)
    for repo, fns in (catalog or {}).items():
        for f, fn in fns.items():
            for k in (("sig", body_key(fn)), ("nosig", body_key_nosig(fn, (languages or {}).get(repo, "")))):
                if k[1] is not None:
                    keys[(repo, f)].append(k)
                    by_key[(repo, k)].add(f)

    def after_cut(repo: str, f: str) -> bool:
        times = [a["time"] for a in catalog[repo][f]["annotations"] if a["time"] is not None]
        return bool(times) and min(times) > cut

    twins = {(repo, g) for repo, f in answer for k in keys.get((repo, f), ()) for g in by_key[(repo, k)]} - answer
    joined = {x for x in twins if after_cut(*x)}
    # Near-duplicates: a post-cutoff function (not itself an answer) whose nosig_text reaches difflib ratio
    # NEAR_RATIO with a test answer's joins the answer too; exact twins are near-duplicates with ratio 1.0.
    near: dict[tuple, tuple[str, float]] = {}
    contain: dict[tuple, tuple[str, float]] = {}
    for repo in sorted({r for r, _ in answer}):
        fns, language = (catalog or {}).get(repo, {}), (languages or {}).get(repo, "")
        ans = {f: nosig_text(fns[f], language) for r, f in answer if r == repo and f in fns}
        cand = {g: nosig_text(fn, language) for g, fn in fns.items() if (repo, g) not in answer and after_cut(repo, g)}
        for g, (f, ratio) in near_duplicates(ans, cand).items():
            near[(repo, g)] = (f, round(ratio, 4))
        # Containment: a post-cutoff function holding >= 90% of a long answer (>= 200 characters) inside a
        # longer body joins the answer too; a = the answer, b = the function.
        for g, (f, share) in containments(ans, cand).items():
            contain[(repo, g)] = (f, round(share, 4))
    answer |= joined | set(near) | set(contain)
    rest = {}
    for sid, rs in kept.items():
        if sid in test:
            continue
        if sid[:2] in test_files:
            aside["same_file_as_test"].extend(rs)
        elif any((r["repo"], f) in answer for r in rs for f in r["target_fns"]):
            aside["uses_test_answer"].extend(rs)
        else:
            rest[sid] = rs
    # Auxiliary pre-cutoff test (fixed 24/09/2026): per configured repo, the K scenarios with the fewest dependants
    # (other scenarios in the same file, or using one of its CREATE functions), ties by sha1(seed|aux|...). Chosen BEFORE
    # dev so dev never loses to it; every other scenario in the same file or using an aux answer is set aside.
    aux, aux_answer = {}, set()
    for repo, k in sorted(cfg.get("aux_holdout", {}).items()):
        cands = [sid for sid in rest if sid[0] == repo]
        creates = {sid: {f for r in rest[sid] for f in labels[pid(r)]["create_fns"]} for sid in cands}
        uses = {sid: {f for r in rest[sid] for f in r["target_fns"]} for sid in cands}
        deps = {x: sum(1 for t in cands if t != x and (t[:2] == x[:2] or uses[t] & creates[x])) for x in cands}
        for x in sorted(cands, key=lambda x: (deps[x], _h(seed, "aux", *x)))[:k]:
            aux[x] = rest[x]
            aux_answer |= {(repo, f) for f in creates[x]}
    aux_files = {sid[:2] for sid in aux}
    for sid in list(rest):
        if sid in aux:
            del rest[sid]
        elif sid[:2] in aux_files or any((r["repo"], f) in aux_answer for r in rest[sid] for f in r["target_fns"]):
            aside["uses_aux_answer"].extend(rest.pop(sid))  # same feature file as an aux scenario, or uses its answer
    by_repo = defaultdict(list)
    for sid in rest:
        by_repo[sid[0]].append(sid)
    dev_quota = _largest_remainder(cfg["dev_total"], {r: len(s) for r, s in by_repo.items()})
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sid, rs in rest.items():
        find(sid)
        parent[find(sid)] = find(("file", *sid[:2]))
        for r in rs:
            for f in labels[pid(r)]["create_fns"]:
                parent[find(sid)] = find(("fn", sid[0], f))
    dev, train = {}, {}
    for repo, sids in by_repo.items():
        groups = defaultdict(list)
        for sid in sids:
            groups[find(sid)].append(sid)
        taken = 0
        for g in sorted(groups.values(), key=lambda g: _h(seed, "dev", *min(g))):  # first fit, whole groups only
            part = dev if taken + len(g) <= dev_quota[repo] else train
            for sid in g:
                part[sid] = rest[sid]
            taken += len(g) if part is dev else 0
    dev_answer = {(r["repo"], f) for rs in dev.values() for r in rs for f in labels[pid(r)]["create_fns"]}
    for sid in list(train):  # a later train scenario that REUSES a dev answer would call a function its snapshot hides
        if any((r["repo"], f) in dev_answer for r in train[sid] for f in r["target_fns"]):
            aside["uses_dev_answer"].extend(train.pop(sid))
    subsets: dict[tuple, list[str]] = defaultdict(list)
    train_by_repo = defaultdict(list)
    for sid in train:
        train_by_repo[sid[0]].append(sid)
    for size in cfg["learning_curve"]:
        quota = _largest_remainder(size, {r: len(s) for r, s in train_by_repo.items()})
        for repo, sids in train_by_repo.items():
            for sid in sorted(sids, key=lambda s: _h(seed, "lc", *s))[:quota[repo]]:
                subsets[sid].append(f"lc{size}")
    parts = {"train": train, "dev": dev, "test": test}
    return {"parts": {k: [r for rs in v.values() for r in rs] for k, v in parts.items()}, "aside": aside,
            "subsets": dict(subsets), "exec_prefix": exec_prefix, "answer_fns": sorted(answer),
            "dev_answer_fns": sorted(dev_answer), "aux": [r for rs in aux.values() for r in rs],
            "aux_answer_fns": sorted(aux_answer), "test_answer_twins": sorted(joined),
            "test_answer_near_duplicates": sorted((repo, g, f, ratio) for (repo, g), (f, ratio) in near.items()),
            "test_answer_containers": sorted((repo, g, f, share) for (repo, g), (f, share) in contain.items())}


def _counts(rows: list[dict]) -> dict:
    return {"unique_scenarios": len({(r["repo"], r["feature_path"], r["scenario_line"]) for r in rows}), "pairs": len(rows)}


def run(pairs_path: Path, labels_path: Path, cfg_path: Path, out: Path, catalog_path: Path, sources_path: Path) -> dict:
    cfg = json.loads(cfg_path.read_text())
    pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {(l["repo"], l["feature_path"], l["scenario_line"], l["suite"]): l
              for l in map(json.loads, labels_path.read_text(encoding="utf-8").splitlines())}
    catalog: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in map(json.loads, catalog_path.read_text(encoding="utf-8").splitlines()):
        catalog[row["repo"]][row["fn_id"]] = row
    languages = {s["repo"]: s["language"] for s in json.loads(sources_path.read_text())}
    res = select(pairs, labels, cfg, catalog, languages)
    counts = {k: _counts(res["parts"][k]) for k in PARTS}
    shortfall = {k: max(0, cfg["minimum_unique_scenarios"][k] - counts[k]["unique_scenarios"]) for k in PARTS}
    sub_rows = defaultdict(list)
    for r in res["parts"]["train"]:
        for s in res["subsets"].get((r["repo"], r["feature_path"], r["scenario_line"]), []):
            sub_rows[s].append(r)
    exec_rows = [r for r in res["parts"]["test"] if r["repo"] in res["exec_prefix"]
                 and r["suite"].startswith(res["exec_prefix"][r["repo"]])]
    by_repo = {k: {repo: _counts([r for r in res["parts"][k] if r["repo"] == repo])
                   for repo in sorted({r["repo"] for r in res["parts"][k]})} for k in PARTS}
    report = {"cutoff_utc": cfg["cutoff_utc"], "seed": cfg["seed"], "config": cfg,
              "input_status": dict(sorted(Counter(r["status"] for r in pairs).items())),
              "counts": counts, "verdict": "GO" if not any(shortfall.values()) else "SHORT", "shortfall": shortfall,
              "set_aside": {k: _counts(v) for k, v in res["aside"].items()}, "by_repo": by_repo,
              "subsets": {k: _counts(v) for k, v in sorted(sub_rows.items())}, "exec_core": _counts(exec_rows),
              "aux_test": {"counts": _counts(res["aux"]), "by_repo": {repo: _counts([r for r in res["aux"] if r["repo"] == repo])
                                                                      for repo in sorted({r["repo"] for r in res["aux"]})}},
              "test_answer_twins": [list(x) for x in res["test_answer_twins"]],
              "test_answer_near_duplicates": [list(x) for x in res["test_answer_near_duplicates"]],
              "test_answer_containers": [list(x) for x in res["test_answer_containers"]]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exec_ids = {(r["repo"], r["feature_path"], r["scenario_line"], r["suite"]) for r in exec_rows}
    members = []
    for name, rows in [*((p, res["parts"][p]) for p in PARTS), ("aux_test", res["aux"]),
                       *((f"set_aside:{k}", v) for k, v in res["aside"].items())]:
        for r in rows:
            key = (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
            members.append({"repo": r["repo"], "feature_path": r["feature_path"], "scenario_line": r["scenario_line"],
                            "suite": r["suite"], "partition": name, "subsets": res["subsets"].get(key[:3], []) if name == "train" else [],
                            "exec_core": key in exec_ids})
    members.sort(key=lambda m: (m["repo"], m["feature_path"], m["scenario_line"], m["suite"] or ""))
    (out / "split_members.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in members), encoding="utf-8")
    forbidden: dict = defaultdict(lambda: {"test": [], "dev": [], "aux": []})
    for key_name, pairs_ in (("test", res["answer_fns"]), ("dev", res["dev_answer_fns"]), ("aux", res["aux_answer_fns"])):
        for repo, f in pairs_:
            forbidden[repo][key_name].append(f)
    (out / "forbidden.json").write_text(json.dumps(dict(sorted(forbidden.items())), indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("out/phase2/labels.jsonl"))
    ap.add_argument("--catalog", type=Path, default=Path("out/phase2/catalog.jsonl"))
    ap.add_argument("--sources", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--config", type=Path, default=Path("config/phase2.json"))
    ap.add_argument("--out", type=Path, default=Path("out/phase2"))
    args = ap.parse_args()
    r = run(args.pairs, args.labels, args.config, args.out, args.catalog, args.sources)
    print(json.dumps({k: r[k] for k in ("counts", "verdict", "shortfall", "set_aside", "subsets", "exec_core")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
