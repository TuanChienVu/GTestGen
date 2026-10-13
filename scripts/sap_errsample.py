"""Đọc tay để phân tích lỗi (P25, declared 06/10/2026, before any test output; DESCRIPTIVE): draw the records a person
reads to show how Full-aware fails, by one fixed rule, and hash the sample before anyone reads it.

Population: the usable test records (P4: the controls run minus the mechanical exclusions minus compose_dropped of the
anchor model) whose Full-aware rows (anchor model, variant main; config/sap.json error_reading.condition) are real
output rows (not filled_missing) for EVERY seed of the anchor model, with ca_defined and CA = 0 at each of them. Scores
without any Full-aware main row of the anchor model (e.g. the dev scores) are refused.
Sample: error_reading.n_records records (all of them when fewer), stratified by repo in proportion to the population of
each repo, rounded by the largest remainder (ties: by repo name); within a repo the records are taken in the order of
sha256("<error_reading.seed>|<record_id>"). Each record is read at ONE seed, chosen by sha256(record_id) as the SemV
sheet chooses it: seeds[int(sha256(record_id), 16) % len(seeds)]. Each case carries the error-reading columns of that
row (COLUMNS), the output text when --outputs is given, and two empty fields for the reader (`reason`, `example`): the
reader files each case under those columns and picks illustrative examples. No number of it is a conclusion: no rate, no
p, nothing in the answer of an RQ. Run after sap_run on the test scores.
Writes <out>/sample.jsonl and <out>/ERRSAMPLE_META.json (the rule's parameters, the population and allocation per repo,
the sha256 of this script, of the config, of every input and of the sample), both into <out>.partial-<pid>, renamed to
<out> once whole (added 06/10/2026: a failure part-way leaves no <out> that would block the rerun).
Exit 3 on any refused input; nothing is written then."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.sap import SapInputError, code_digest, compose_dropped, jsonl_rows, usable_records  # noqa: E402

BV_CONDS = ("syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved")
COLUMNS = ("extract_status", "fence_truncated", "bv", *(f"bv_{c}" for c in BV_CONDS), "cap_hit", "open_resolution",
           "hallucinated", "members_used", "conv_style", "conv_pattern", "conv_naming", "n_calls", "call_precision",
           "ca_ref_v3m", "ca_ref_v3all", "ca_ref_v3names", "ca_ref_top4", "php_tag_stripped", "unread_step_annotations",
           "duplicate_reuse")  # the columns a case is filed under (copied as the row has them; None when absent)
KEY = ("record_id", "model", "seed", "condition", "variant")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def h(s: str) -> str:
    return sha256(s.encode("utf-8"))


def error_pool(rows: list[dict], cfg: dict, keep: set[str]) -> tuple[dict[str, dict[str, dict]], dict]:
    """record_id -> {seed: row} of the population, and the counts of the records left out, by reason."""
    spec, model = cfg["error_reading"], cfg["anchor_model"]
    seeds = cfg["seeds"][model]
    by: dict = defaultdict(dict)
    for r in rows:
        if r["model"] == model and r["condition"] == spec["condition"] and (r.get("variant") or "main") == "main":
            k = (r["record_id"], r["seed"])
            if r["seed"] in by[r["record_id"]]:
                raise SapInputError(f"score rows hold {k} twice")
            by[r["record_id"]][r["seed"]] = r
    if not by:  # e.g. the dev scores, whose variant is not main
        raise SapInputError(f"no score row of the {model} {spec['condition']} runs, variant main: these are not "
                            "the test scores of the design")
    pool, out = {}, Counter({"not_usable": 0, "incomplete": 0, "ca_undefined": 0, "ca_one": 0})
    for rid, got in sorted(by.items()):
        if rid not in keep:
            out["not_usable"] += 1
        elif any(sd not in got or got[sd].get("filled_missing") for sd in seeds):
            out["incomplete"] += 1
        elif not all(got[sd].get("ca_defined") for sd in seeds):
            out["ca_undefined"] += 1
        elif any(got[sd].get("ca") != 0 for sd in seeds):
            out["ca_one"] += 1
        else:
            pool[rid] = got
    return pool, dict(out)


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of n over the strata, largest remainder (ties by stratum name); all when total <= n. In
    integers (fixed 06/10/2026): the remainder of n * v / total is n * v % total, so equal remainders tie exactly
    and go by name, which float quotas broke (e.g. {A: 3, B: 31, C: 1}, n = 30)."""
    total = sum(sizes.values())
    if total <= n:
        return dict(sizes)
    out = {k: n * v // total for k, v in sizes.items()}
    for k in sorted(sizes, key=lambda k: (-(n * sizes[k] % total), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def error_sample(rows: list[dict], cfg: dict, keep: set[str]) -> tuple[list[dict], dict]:
    """(the sampled cases, the population and allocation per repo, the counts left out)."""
    spec, seeds = cfg["error_reading"], cfg["seeds"][cfg["anchor_model"]]
    pool, left = error_pool(rows, cfg, keep)
    repo_of = {rid: next(iter(got.values()))["repo"] for rid, got in pool.items()}
    sizes = dict(sorted(Counter(repo_of.values()).items()))
    alloc = allocate(sizes, spec["n_records"])
    sample = []
    for repo, k in alloc.items():
        ids = sorted((rid for rid in pool if repo_of[rid] == repo), key=lambda rid: h(f"{spec['seed']}|{rid}"))
        for rid in ids[:k]:
            sd = seeds[int(h(rid), 16) % len(seeds)]
            r = pool[rid][sd]
            sample.append({"record_id": rid, "repo": repo, "language": r.get("language"),
                           "scenario_key": r.get("scenario_key"), "condition": spec["condition"], "seed": sd,
                           **{c: r.get(c) for c in COLUMNS}, "text_sha256": r.get("text_sha256"), "reason": "",
                           "example": False})
    return sample, {"population": sizes, "allocation": alloc, "left_out": left}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--scores", type=Path, nargs="+", required=True, help="the official test scores.jsonl")
    ap.add_argument("--controls-dir", type=Path, required=True)
    ap.add_argument("--compose-excluded", type=Path, required=True)
    ap.add_argument("--outputs", type=Path, nargs="*", default=[], help="generation outputs, for the text of each case")
    ap.add_argument("--out", type=Path, required=True, help="a new directory")
    args = ap.parse_args()
    if args.out.exists():
        raise SapInputError(f"{args.out} exists (the sample is drawn once; sap_errsample never writes over a sample)")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg.get("error_reading"), dict):
        raise SapInputError(f"{args.config} has no error_reading (P25)")
    rows = [r for f in args.scores for r in jsonl_rows(f)]
    controls = jsonl_rows(args.controls_dir / "controls.jsonl")
    exclusions = jsonl_rows(args.controls_dir / "exclusions.jsonl")
    compose = jsonl_rows(args.compose_excluded)
    keep = usable_records(controls, exclusions, compose, cfg["anchor_model"])
    sample, info = error_sample(rows, cfg, keep)
    if args.outputs:
        texts = {tuple(o.get(k) if k != "variant" else (o.get(k) or "main") for k in KEY): o.get("text")
                 for f in args.outputs for o in jsonl_rows(f)}
        for s in sample:
            t = texts.get((s["record_id"], cfg["anchor_model"], s["seed"], s["condition"], "main"))
            if not isinstance(t, str):
                raise SapInputError(f"no output text for the sampled case {(s['record_id'], s['seed'])} in --outputs")
            if s["text_sha256"] is not None and sha256(t.encode("utf-8")) != s["text_sha256"]:  # the scored text
                raise SapInputError(f"the --outputs text of {(s['record_id'], s['seed'])} is not the one 3C scored "
                                    "(text_sha256 of its score row)")
            s["text"] = t
    data = "".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in sample).encode("utf-8")
    part = args.out.with_name(f"{args.out.name}.partial-{os.getpid()}")
    part.mkdir(parents=True)
    (part / "sample.jsonl").write_bytes(data)
    inputs = [*args.scores, args.controls_dir / "controls.jsonl", args.controls_dir / "exclusions.jsonl",
              args.compose_excluded, *args.outputs]
    meta = {"name": "đọc tay phân tích lỗi (P25)", "descriptive_only": True, **cfg["error_reading"],
            "anchor_model": cfg["anchor_model"], "seeds": cfg["seeds"][cfg["anchor_model"]], **info,
            "script_sha256": sha256(Path(__file__).read_bytes()), "config_sha256": sha256(args.config.read_bytes()),
            "code": code_digest(Path(__file__).resolve().parents[1]),  # as SAP_LOCK.json "code"
            "inputs": {str(p): sha256(p.read_bytes()) for p in inputs}, "sample_sha256": sha256(data),
            "n_sampled": len(sample)}
    (part / "ERRSAMPLE_META.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    part.rename(args.out)
    print(json.dumps({k: meta[k] for k in ("population", "allocation", "n_sampled", "sample_sha256")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SapInputError, OSError, ValueError, KeyError) as e:  # a refused or malformed input: nothing written
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(3)
