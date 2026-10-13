"""Re-check split output files independently of v2tools.split.assign; exit 1 if any check fails."""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
ap.add_argument("--split", type=Path, default=Path("out/split.json"))
ap.add_argument("--cutoff", default="2024-01-01T00:00:00Z")
args = ap.parse_args()
cut = int(dt.datetime.fromisoformat(args.cutoff.replace("Z", "+00:00")).timestamp())
rows = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
pid = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
pairs = {pid(r): r for r in rows}
members = [json.loads(l) for l in (args.split.parent / "split_members.jsonl").read_text(encoding="utf-8").splitlines()]
report = json.loads(args.split.read_text(encoding="utf-8"))
placed = [m for m in members if m["partition"] in ("train", "dev", "test")]
parts, keys = collections.defaultdict(set), collections.defaultdict(set)
for m in placed:
    parts[pid(m)[:3]].add(m["partition"])
    keys[pairs[pid(m)]["structure_key"]].add(pid(m)[:3])
test = [pairs[pid(m)] for m in placed if m["partition"] == "test"]
# Compared by FUNCTION (target_fns/new_target_fns), not by annotation line (target_ids/new_target_ids): several
# annotations on one function are the same answer, so a leak through a different annotation on the same
# function must still be caught.
test_new = {(p["repo"], f) for p in test for f in p["new_target_fns"]}
test_files = {(p["repo"], p["feature_path"]) for p in test}
train_dev = [m for m in placed if m["partition"] != "test"]
member_ids = {pid(m) for m in members}
bound_ids = {pid(r) for r in rows if r["status"] == "BOUND"}
checks = {
    "duplicate_pair_ids": len(rows) - len(pairs),
    "members_minus_bound": len(members) - report["input_status"].get("BOUND", 0),
    "members_not_bound": len(member_ids ^ bound_ids),
    "scenarios_in_2_parts": sum(len(v) > 1 for v in parts.values()),
    "structure_keys_on_2_scenarios": sum(len(v) > 1 for v in keys.values()),
    "bad_test_items": sum(not ((p["scenario_min_time"] or 0) > cut and p["targets_new_count"] > 0 and p["ambiguous_steps"] == 0)
                          for p in test),
    "train_dev_in_test_files": sum((m["repo"], m["feature_path"]) in test_files for m in train_dev),
    "train_dev_target_leaks": sum((pairs[pid(m)]["repo"], f) in test_new for m in train_dev for f in pairs[pid(m)]["target_fns"]),
    "ambiguous_in_train_dev": sum(pairs[pid(m)]["ambiguous_steps"] > 0 for m in train_dev),
}
print(json.dumps(checks, indent=2))
raise SystemExit(1 if any(checks.values()) else 0)
