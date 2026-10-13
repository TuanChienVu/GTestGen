"""Per-step reuse/create labels for every BOUND pair, plus the serialized function catalog they refer to.

A step is REUSE when some step definition matching it already existed before the scenario was written (its pattern's
earliest evidence is strictly before `scenario_min_time`); otherwise it is CREATE: the developer had to write that
pattern for this scenario, so it is what a model must generate."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .catalog import Catalog, build_catalog


def label_pair(row: dict, cat: Catalog, ann_time: dict[str, int | None]) -> dict:
    if row["scenario_min_time"] is None:
        raise ValueError("labels need a history run (scenario_min_time is None)")
    per = cat.bind_steps(row["steps"], row["suite"], row["tags"])
    if sorted({i for ids in per for i in ids}) != row["target_ids"]:
        raise ValueError(f"binding differs from Phase 1 for {row['feature_path']}:{row['scenario_line']} {row['suite']}")
    kinds = []
    for ids in per:
        times = [ann_time[i] for i in ids if ann_time[i] is not None]
        kinds.append("reuse" if times and min(times) < row["scenario_min_time"] else "create")
    create_def_ids = sorted({i for ids, k in zip(per, kinds) if k == "create" for i in ids})
    create_fns = sorted({cat.fn_of(i) for i in create_def_ids})
    reuse_fns = sorted({cat.fn_of(i) for ids, k in zip(per, kinds) if k == "reuse" for i in ids} - set(create_fns))
    return {"repo": row["repo"], "feature_path": row["feature_path"], "scenario_line": row["scenario_line"],
            "suite": row["suite"], "step_def_ids": per, "step_kinds": kinds, "create_def_ids": create_def_ids,
            "create_fns": create_fns, "reuse_fns": reuse_fns}


def run(config: Path, root: Path, pairs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels, catalog_rows, counts = [], [], {}
    for source in json.loads(config.read_text()):
        cat = build_catalog(source, root / source["dir"])
        ann_time = {a.def_id: a.time for f in cat.fns.values() for a in f.annotations}
        catalog_rows += [{"repo": source["repo"], **asdict(f)} for f in sorted(cat.fns.values(), key=lambda f: f.fn_id)]
        mine = [label_pair(r, cat, ann_time) for r in rows if r["repo"] == source["repo"] and r["status"] == "BOUND"]
        labels += mine
        counts[source["repo"]] = {"functions": len(cat.fns), "labelled_pairs": len(mine),
                                  "pairs_with_create": sum(1 for l in mine if l["create_fns"])}
    (out / "catalog.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in catalog_rows), encoding="utf-8")
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--root", type=Path, default=Path("repos-hist"))
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("out/phase2"))
    args = ap.parse_args()
    print(json.dumps(run(args.config, args.root, args.pairs, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
