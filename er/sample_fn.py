"""Check before the SAP lock: sample outputs that PASS plain ER but fail the action trace, for a manual check of false negatives
("different trace, same meaning").

    python -m er.sample_fn --root <manifest root> --run dev_pilot_v0 [--out DIR] [--k 30] [--seed 20260924] \\
        > er/FN_CHECK_dev_pilot.md
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path, PurePosixPath

from .batch import DEFAULT_OUT, load_refs, load_results
from .manifest import load_manifest
from .records import load_records
from .steps import step_plan
from .trace import action_trace

CONTROLS = {"neg", "gold", "stub"}


def event_diff(gold: dict, out: dict, actions: tuple[int, ...]) -> tuple[list, list]:
    g, o = action_trace(gold, actions), action_trace(out, actions)
    return sorted((g - o).elements()), sorted((o - g).elements())


def render_sample(manifest_dir: Path, run_dir: Path, records: dict[str, dict], k: int, seed: int) -> str:
    slug = {(e.record_id, e.condition): PurePosixPath(e.base).parts[-2]
            for e in load_manifest(manifest_dir) if e.skip is None}
    rows = load_results(run_dir / "er_results.jsonl")
    # a score_error row has trace_action = 0 by rule, not by a trace comparison (and its trace may be unreadable)
    pool = sorted((r for r in rows if r["condition"] not in CONTROLS and r["er"] == 1 and r["trace_action"] == 0
                   and not r.get("score_error")), key=lambda r: (r["record_id"], r["condition"]))
    pick = random.Random(seed).sample(pool, min(k, len(pool)))
    lines = [f"# Kiểm tay âm tính giả của ER-trace-action — {run_dir.name}\n",
             f"Pool: {len(pool)} output PASS ER nhưng trace_action = 0; lấy {len(pick)} (seed {seed}). "
             "Đánh dấu một ô cho mỗi ca.\n"]
    refs: dict[str, Path] = {}
    for ref_run in {r.get("ref_run") for r in pick}:
        ref_dir = run_dir.parent / ref_run if ref_run else run_dir  # frozen references of the controls run
        for rid, o in load_refs(ref_dir / "er_refs.jsonl").items():  # read-only; the later line of a record wins
            if o.get("ref_dir"):
                refs[rid] = ref_dir / o["ref_dir"] / "normal" / "trace.json"
    for n, r in enumerate(pick, 1):
        s = slug[(r["record_id"], r["condition"])]
        gold = json.loads(refs[r["record_id"]].read_text(encoding="utf-8"))
        out = json.loads((run_dir / "er_runs" / s / r["condition"] / "normal" / "trace.json").read_text(encoding="utf-8"))
        only_gold, only_out = event_diff(gold, out, step_plan(records[r["record_id"]]).actions)
        lines += [f"## {n}. `{r['record_id']}` — `{r['condition']}`\n",
                  "- chỉ có ở gold: " + ("; ".join(map(str, only_gold)) or "—"),
                  "- chỉ có ở output: " + ("; ".join(map(str, only_out)) or "—"),
                  "- [ ] tương đương về nghĩa (âm tính giả)   - [ ] khác thật   - ghi chú:\n"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    print(render_sample(a.root / a.run, a.out / a.run, load_records(), a.k, a.seed))


if __name__ == "__main__":
    main()
