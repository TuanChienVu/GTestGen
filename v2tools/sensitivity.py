"""Test subsets for the Phase-3 sensitivity analyses, declared before any model is scored:
out/phase2/sensitivity_subsets.json = {name: sorted test record ids}.
- post_cutoff_all: every CREATE entry of the record is post-cutoff code;
- time_coherent_30d: every CREATE pattern is dated within 30 days of the scenario (t_vis - scenario_time);
- residual_clean: no answer of the record (CREATE functions and the pair's new_target_fns) has a residual relation with
  a function that train/dev reach (target, bound step, or snapshot under hidden_for): a post-cutoff function at
  RESIDUAL_LOW <= ratio < NEAR_RATIO, or a function not after the cutoff at ratio >= NEAR_RATIO (difflib ratio of the
  nosig texts, a = the function, b = the answer, both >= MIN_CHARS). Post-cutoff containers and near-duplicates at
  >= NEAR_RATIO are already joined to the test answer and hidden, so they reach nothing."""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
from collections import defaultdict
from pathlib import Path

from .body import MIN_CHARS, NEAR_RATIO, nosig_text
from .records import hidden_for, snapshot_fns

RESIDUAL_LOW = 0.85
DAY = 86400


def touched_answers(answers: dict[str, str], reached: dict[str, tuple[str, bool]]) -> set[str]:
    """Answer ids with a residual relation to a reached function; `reached` maps a function id to (nosig text, whether
    its earliest dated annotation is after the cutoff). Texts under MIN_CHARS are ignored."""
    out: set[str] = set()
    sm = difflib.SequenceMatcher(None, autojunk=False)
    for f, b in sorted(answers.items()):
        if len(b) < MIN_CHARS:
            continue
        sm.set_seq2(b)
        for g, (a, post) in sorted(reached.items()):
            if g == f or len(a) < MIN_CHARS:
                continue
            low = RESIDUAL_LOW if post else NEAR_RATIO
            sm.set_seq1(a)
            if sm.real_quick_ratio() < low or sm.quick_ratio() < low:
                continue
            ratio = sm.ratio()
            if (low <= ratio < NEAR_RATIO) if post else ratio >= NEAR_RATIO:
                out.add(f)
                break
    return out


def build(phase2: Path, pairs: Path, sources: Path, config: Path) -> dict[str, list[str]]:
    cfg = json.loads(config.read_text())
    cut = int(dt.datetime.fromisoformat(cfg["cutoff_utc"].replace("Z", "+00:00")).timestamp())
    languages = {s["repo"]: s["language"] for s in json.loads(sources.read_text())}
    load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = lambda r: (r["repo"], r["feature_path"], r["scenario_line"], r["suite"])
    recs = {p: load(phase2 / "records" / f"{p}.jsonl") for p in ("train", "dev", "test")}
    catalog: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in load(phase2 / "catalog.jsonl"):
        catalog[row["repo"]][row["fn_id"]] = row
    forbidden = {r: {k: set(v) for k, v in parts.items()} for r, parts in json.loads((phase2 / "forbidden.json").read_text()).items()}
    new_fns = {key(r): r["new_target_fns"] for r in load(pairs)}
    reached: dict[str, set[str]] = defaultdict(set)
    for part in ("train", "dev"):
        for r in recs[part]:
            fns = catalog[r["repo"]]
            reached[r["repo"]] |= {x["fn_id"] for x in r["create"]} | set(r["step_fns"])
            reached[r["repo"]] |= set(snapshot_fns(r, fns, hidden_for(part, forbidden.get(r["repo"], {}))))

    def after_cut(fn: dict) -> bool:
        times = [a["time"] for a in fn["annotations"] if a["time"] is not None]
        return bool(times) and min(times) > cut

    answers_of = {r["id"]: {x["fn_id"] for x in r["create"]} | set(new_fns[key(r)]) for r in recs["test"]}
    touched: set[tuple[str, str]] = set()
    for repo in sorted({r["repo"] for r in recs["test"]}):
        fns, language = catalog[repo], languages[repo]
        answers = {f: nosig_text(fns[f], language) for r in recs["test"] if r["repo"] == repo for f in answers_of[r["id"]]
                   if f in fns}
        near = {g: (nosig_text(fns[g], language), after_cut(fns[g])) for g in reached[repo] if g in fns}
        touched |= {(repo, f) for f in touched_answers(answers, near)}
    test = recs["test"]
    return {"post_cutoff_all": sorted(r["id"] for r in test if all(x["post_cutoff_code"] for x in r["create"])),
            "residual_clean": sorted(r["id"] for r in test if not any((r["repo"], f) in touched for f in answers_of[r["id"]])),
            "time_coherent_30d": sorted(r["id"] for r in test if r["t_vis"] - r["scenario_time"] <= 30 * DAY)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--pairs", type=Path, default=Path("out/hist/pairs.jsonl"))
    ap.add_argument("--sources", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--config", type=Path, default=Path("config/phase2.json"))
    args = ap.parse_args()
    subsets = build(args.phase2, args.pairs, args.sources, args.config)
    (args.phase2 / "sensitivity_subsets.json").write_text(json.dumps(subsets, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in subsets.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
