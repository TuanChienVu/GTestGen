"""Precompute symtime.first_times for every file Phase 3 may outline: each catalog step file, each CREATE target file
and every in-repo ancestor of those, written as one JSONL row per (repo, path). Sealed with the other Phase 3 inputs,
so that compose and the scorer never read git history themselves."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .state import RepoContext


def jsonl_rows(path: Path) -> list[dict]:
    """The rows of a JSONL file, split on "\\n" only, from the decoded bytes, exactly as v2gpu's jsonl_lines reads.
    str.splitlines() also breaks at U+2028, U+2029 and U+0085, which json.dumps(ensure_ascii=False) writes raw inside
    strings, and read_text() turns a bare "\\r" into "\\n". A blank line inside the file still raises. phase3io and
    symcache share this reader."""
    lines = path.read_bytes().decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    return [json.loads(line) for line in lines]


def paths_needed(catalog_rows: list[dict], records: list[dict]) -> dict[str, set[str]]:
    need: dict[str, set[str]] = defaultdict(set)
    for f in catalog_rows:
        need[f["repo"]].add(f["path"])
    for r in records:
        for c in r["create"]:
            need[r["repo"]].add(c["path"])
    return need


def parent_gate(ctx: RepoContext, paths: set[str]) -> dict:
    """Every declared parent of every outlined file, by status (rule 4 of v2tools/state.py); the gate needs
    unresolved == 0."""
    counts = {"resolved": 0, "external": 0, "unresolved": 0}
    unresolved = []
    for p in sorted(paths):
        for par in ctx.parents(p):
            counts[par.status] += 1
            if par.status == "unresolved":
                unresolved.append({"path": p, "parent": par.written, "fqn": par.fqn})
    return {**counts, "unresolved_list": unresolved}


def build(sources: list[dict], full_root: Path, commits: dict[str, str], catalog_rows: list[dict],
          records: list[dict], gate: dict | None = None) -> list[dict]:
    rows = []
    by_repo = defaultdict(dict)
    for f in catalog_rows:
        by_repo[f["repo"]][f["fn_id"]] = f
    for repo, paths in sorted(paths_needed(catalog_rows, records).items()):
        src = next(s for s in sources if s["repo"] == repo)
        ctx = RepoContext(repo, src["language"], full_root / f"{src['dir']}.git", commits[repo], by_repo[repo])
        todo = set(paths)
        for p in sorted(paths):
            todo |= {par.path for par, _ in ctx.ancestors(p)}
        if gate is not None:
            gate[repo] = parent_gate(ctx, {p for p in todo if ctx.text(p) is not None})
        for p in sorted(todo):
            if ctx.text(p) is None:
                continue
            t = ctx.times(p)
            rows.append({"repo": repo, "path": p, "file": t["file"], "names": dict(sorted(t["names"].items())),
                         "params": dict(sorted(t["params"].items())), "versions": t["versions"], "vnames": t["vnames"],
                         "vprivate": t["vprivate"], "present": t["present"]})
    return rows


def load(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in jsonl_rows(path):
        out[row["repo"]][row["path"]] = {k: row[k] for k in ("file", "names", "params", "versions", "vnames", "vprivate",
                                                             "present")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--full-root", type=Path, default=Path("repos-full"))
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--out", type=Path, default=Path("out/phase3/symtimes.jsonl"))
    args = ap.parse_args()
    sources = json.loads(args.config.read_text())
    records = [r for p in ("train", "dev", "test", "aux_test")
               for r in jsonl_rows(args.phase2 / "records" / f"{p}.jsonl")]
    commits = {r["repo"]: r["commit"] for r in records}
    catalog_rows = jsonl_rows(args.phase2 / "catalog.jsonl")
    gate: dict = {}
    rows = build(sources, args.full_root, commits, catalog_rows, records, gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (args.out.parent / "PARENTS.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    bad = sum(g["unresolved"] for g in gate.values())
    print(json.dumps({"files": len(rows), "unresolved_parents": bad,
                      "parents": {r: {k: v for k, v in g.items() if k != "unresolved_list"} for r, g in gate.items()}}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
