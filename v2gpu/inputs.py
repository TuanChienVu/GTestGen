"""Phase 2 inputs of a job, checked against the seal, and the catalog a training pod receives.

- `phase2_problems(paths, PHASE2_SHA256SUMS)`: every input must be listed in the sums (the Phase 2 build writes the
paths as
  out/phase2/<file>; a path matches by its location relative to the sums file) with the same sha256. No receipt route
  here: records, forbidden.json and any other Phase 2 file can only pass by being listed.
- `catalog_problems(path, sums)`: the ONE route for the catalog a training pod receives (the check
  is content-based, not filename-based). If a receipt `<file>.receipt.json` sits beside the path, that route is used
  ALWAYS, whatever the file is named (even the sealed name catalog.jsonl): the receipt must have source_sha256,
  forbidden_sha256 and output_sha256, source must be the sealed catalog.jsonl digest and forbidden the sealed
  forbidden.json digest (both read from the sums), and output must be the file's own sha256. With no receipt, the file
  must be listed in the sums with the same sha (the plain sealed route).
- `unstripped_problems(catalog_path, forbidden)`: a training pod's catalog must never carry the `code` of a forbidden
  test or aux answer function, whatever route it passed `catalog_problems` through; checked in every train_job mode.
- `strip_catalog`: the catalog shipped to a training pod has no `code` for the functions in forbidden test ∪ aux (ids,
  annotations, suites and the rest are kept; leakcheck only needs their ids there).
- `meta_problems(meta, sums)`: PROMPTS_META.json has the contract's keys (v2tools_commit, phase2_sums_sha256,
  symtimes_sha256, special_tokens_sha256, tokenizer_revisions, compose), names this seal (phase2_sums_sha256), the
  control tokens compose used (the sha256 of config/special_tokens.json) and the pinned tokenizer revision of every model.

python -m v2gpu.inputs strip-catalog --phase2-sums out/phase2/PHASE2_SHA256SUMS --out ship/catalog.train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from .models import MODELS

STRIPPED = ("test", "aux")
META_KEYS = ("v2tools_commit", "phase2_sums_sha256", "symtimes_sha256", "special_tokens_sha256", "tokenizer_revisions",
             "compose")
SPECIAL_TOKENS = Path(__file__).resolve().parents[1] / "config" / "special_tokens.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _listed(sums: Path) -> dict[str, str]:
    out = {}
    for line in Path(sums).read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest
    return out


def _digest_of(rel: str, listed: dict[str, str]) -> str | None:
    hits = [d for name, d in listed.items() if name == rel or name.endswith("/" + rel)]
    return hits[0] if len(hits) == 1 else None


def phase2_problems(paths, sums: Path) -> list[str]:
    sums = Path(sums)
    listed, base, problems = _listed(sums), sums.resolve().parent, []
    for path in map(Path, paths):
        rel = path.resolve().relative_to(base).as_posix() if path.resolve().is_relative_to(base) else path.name
        digest = _digest_of(rel, listed)
        if digest is None:
            problems.append(f"{rel}: not listed in {sums.name}")
        elif _sha(path) != digest:
            problems.append(f"{rel}: sha256 differs from {sums.name}")
    return problems


def catalog_problems(path: Path, sums: Path) -> list[str]:
    """The catalog a training pod receives: a receipt beside the file ALWAYS wins, so a
    correctly stripped catalog shipped under the sealed name catalog.jsonl is accepted, and a receipt cannot make an
    unlisted, unrelated file pass (its source and forbidden digests are checked against the seal)."""
    path, sums = Path(path), Path(sums)
    listed, base = _listed(sums), sums.resolve().parent
    rel = path.resolve().relative_to(base).as_posix() if path.resolve().is_relative_to(base) else path.name
    receipt_path = path.with_name(path.name + ".receipt.json")
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        missing = [k for k in ("source_sha256", "forbidden_sha256", "output_sha256") if k not in r]
        if missing:
            return [f"{rel}: receipt lacks key(s) {missing}"]
        problems = []
        if r["source_sha256"] != _digest_of("catalog.jsonl", listed) or \
                r["forbidden_sha256"] != _digest_of("forbidden.json", listed):
            problems.append(f"{rel}: not derived from the sealed catalog.jsonl and forbidden.json")
        if r["output_sha256"] != _sha(path):
            problems.append(f"{rel}: sha256 differs from its receipt")
        return problems
    digest = _digest_of(rel, listed)
    if digest is None:
        return [f"{rel}: not listed in {sums.name}"]
    if _sha(path) != digest:
        return [f"{rel}: sha256 differs from {sums.name}"]
    return []


def unstripped_problems(catalog_path: Path, forbidden: dict) -> list[str]:
    """A training pod's catalog must never carry the `code` of a forbidden test or aux answer function."""
    hidden = {(repo, f) for repo, parts in forbidden.items() for p in STRIPPED for f in parts.get(p, [])}
    from .examples import jsonl_lines          # examples imports this module: import at call time

    bad = []
    for line in jsonl_lines(catalog_path):
        row = json.loads(line)
        if (row["repo"], row["fn_id"]) in hidden and "code" in row:
            bad.append(row["fn_id"])
    if not bad:
        return []
    return [f"{Path(catalog_path).name}: holds the code of {len(bad)} test/aux answer function(s), e.g. {bad[0]}; "
            f"ship the output of python -m v2gpu.inputs strip-catalog"]


def strip_catalog(catalog: Path, forbidden: Path, out: Path) -> dict:
    hidden = {(repo, f) for repo, parts in json.loads(Path(forbidden).read_text()).items()
              for p in STRIPPED for f in parts.get(p, [])}
    from .examples import jsonl_lines

    rows, n = [], 0
    for line in jsonl_lines(catalog):
        row = json.loads(line)
        if (row["repo"], row["fn_id"]) in hidden:
            row.pop("code", None)
            n += 1
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    Path(out).write_text("".join(rows), encoding="utf-8")
    receipt = {"source_sha256": _sha(catalog), "forbidden_sha256": _sha(forbidden), "output_sha256": _sha(out),
               "stripped": n, "stripped_parts": list(STRIPPED)}
    Path(out).with_name(Path(out).name + ".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def meta_problems(meta: dict | None, sums: Path | None, synthetic: bool = False) -> list[str]:
    """META against the seal and the pins. v2tools_commit must be the 40-hex commit 3A built from;
    `synthetic` (train_job's canary mode: v2gpu.canary's prompts, which no 3A build made) accepts None there only."""
    if meta is None:
        return ["PROMPTS_META.json is missing"]
    missing = [k for k in META_KEYS if k not in meta]
    problems = [f"PROMPTS_META.json lacks key(s) {missing}"] if missing else []
    if "v2tools_commit" in meta and not (synthetic and meta["v2tools_commit"] is None) and \
            not re.fullmatch(r"[0-9a-f]{40}", str(meta["v2tools_commit"])):
        problems.append(f"PROMPTS_META.json v2tools_commit {meta['v2tools_commit']!r} is not a 40-hex commit sha")
    if sums is not None and meta.get("phase2_sums_sha256") != _sha(sums):
        problems.append(f"PROMPTS_META.json phase2_sums_sha256 is not the sha256 of {Path(sums).name}")
    if "special_tokens_sha256" in meta and meta["special_tokens_sha256"] != _sha(SPECIAL_TOKENS):
        problems.append("PROMPTS_META.json special_tokens_sha256 is not the sha256 of config/special_tokens.json")
    revs = meta.get("tokenizer_revisions")
    if not isinstance(revs, dict):
        return problems + ["PROMPTS_META.json has no tokenizer_revisions"]
    problems += [f"PROMPTS_META.json tokenizer revision of {k} is {revs.get(k)!r}, pinned {m.revision!r}"
                 for k, m in MODELS.items() if revs.get(k) != m.revision]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("strip-catalog")
    s.add_argument("--phase2-sums", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    base = a.phase2_sums.parent
    problems = phase2_problems([base / "catalog.jsonl", base / "forbidden.json"], a.phase2_sums)
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 2
    print(json.dumps(strip_catalog(base / "catalog.jsonl", base / "forbidden.json", a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
