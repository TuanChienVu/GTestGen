"""Time- and group-aware split of (scenario, suite) pairs into train / dev / test, with a GO/SHORT verdict."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = {"train": 600, "dev": 150, "test": 150}
PARTS = ("train", "dev", "test")
REASONS = ("ambiguous", "duplicate_structure", "test_sibling_old_targets", "same_file_as_test", "uses_test_new_target")


def _counts(pairs: list[dict]) -> dict:
    return {"unique_scenarios": len({(p["repo"], p["feature_path"], p["scenario_line"]) for p in pairs}), "pairs": len(pairs)}


def assign(rows: list[dict], cutoff_epoch: int, dev_target: int = 150) -> dict:
    scenarios: dict[tuple, list[dict]] = {}
    for r in rows:
        scenarios.setdefault((r["repo"], r["feature_path"], r["scenario_line"]), []).append(r)
    aside: dict[str, list[dict]] = {k: [] for k in REASONS}
    kept, seen = {}, set()
    for sid, pairs in scenarios.items():
        usable = []
        for p in pairs:
            if p["status"] != "BOUND":
                continue
            (aside["ambiguous"] if p["ambiguous_steps"] > 0 else usable).append(p)
        if not usable:
            continue
        if usable[0]["structure_key"] in seen:
            aside["duplicate_structure"].extend(usable)
            continue
        seen.add(usable[0]["structure_key"])
        kept[sid] = usable

    def is_test(p: dict) -> bool:
        return (p["scenario_min_time"] or 0) > cutoff_epoch and p["targets_new_count"] > 0

    test = {}
    for sid, pairs in kept.items():
        if any(is_test(p) for p in pairs):
            test[sid] = [p for p in pairs if is_test(p)]
            aside["test_sibling_old_targets"].extend(p for p in pairs if not is_test(p))
    test_files = {sid[:2] for sid in test}
    # Compared by FUNCTION (path:body_start), not by annotation line (path:line): several annotations on one
    # function are the same answer, so a train/dev pair that reuses a test scenario's new function through a
    # different annotation on it must still be excluded.
    test_new = {(p["repo"], f) for pairs in test.values() for p in pairs for f in p["new_target_fns"]}
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for sid, pairs in kept.items():
        if sid in test:
            continue
        if sid[:2] in test_files:
            aside["same_file_as_test"].extend(pairs)
        elif any((p["repo"], f) in test_new for p in pairs for f in p["target_fns"]):
            aside["uses_test_new_target"].extend(pairs)
        else:
            groups[sid[:2]].append(sid)
    dev, train = {}, {}
    for f in sorted(groups, key=lambda f: hashlib.sha1(f"{f[0]}/{f[1]}".encode()).hexdigest()):
        part = dev if len(dev) < dev_target else train
        for sid in groups[f]:
            part[sid] = kept[sid]
    result = {name: [p for pairs in part.values() for p in pairs] for name, part in (("train", train), ("dev", dev), ("test", test))}
    counts = {name: _counts(result[name]) for name in PARTS}
    shortfall = {k: max(0, TARGETS[k] - counts[k]["unique_scenarios"]) for k in TARGETS}
    return {**result, "set_aside": aside, "counts": counts,
            "verdict": "GO" if not any(shortfall.values()) else "SHORT", "shortfall": shortfall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--cutoff", default="2024-01-01T00:00:00Z")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cut = int(dt.datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).timestamp())
    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = assign(rows, cut)
    by_repo = {}
    for part in PARTS:
        per = defaultdict(list)
        for p in result[part]:
            per[p["repo"]].append(p)
        by_repo[part] = {repo: _counts(ps) for repo, ps in sorted(per.items())}
    report = {"cutoff_utc": args.cutoff, "input_status": dict(sorted(Counter(r["status"] for r in rows).items())),
              "counts": result["counts"], "verdict": result["verdict"], "shortfall": result["shortfall"],
              "set_aside": {k: _counts(v) for k, v in result["set_aside"].items()}, "by_repo": by_repo}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    members = [{"repo": p["repo"], "feature_path": p["feature_path"], "scenario_line": p["scenario_line"],
                "suite": p["suite"], "partition": name}
               for name, pairs in [*((part, result[part]) for part in PARTS),
                                   *((f"set_aside:{k}", v) for k, v in result["set_aside"].items())]
               for p in pairs]
    members.sort(key=lambda m: (m["repo"], m["feature_path"], m["scenario_line"], m["suite"] or ""))
    (args.out.parent / "split_members.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in members), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "verdict", "shortfall", "set_aside")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
