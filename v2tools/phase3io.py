"""Load the sealed Phase 2 outputs and the Phase 3 symbol-time cache into a Composer (one RepoContext per repo)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .compose import Composer, Params
from .state import RepoContext
from .symcache import jsonl_rows  # noqa: F401  (re-exported: the one JSONL reader of 3A)
from .symcache import load as load_symtimes

PARTS = ("train", "dev", "test", "aux_test")


def read_records(phase2: Path) -> dict[str, dict]:
    out = {}
    for part in PARTS:
        for r in jsonl_rows(phase2 / "records" / f"{part}.jsonl"):
            out[r["id"]] = r
    return out


def read_catalog(phase2: Path) -> dict[str, dict[str, dict]]:
    catalog: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in jsonl_rows(phase2 / "catalog.jsonl"):
        catalog[row["repo"]][row["fn_id"]] = row
    return catalog


def parent_gate_ok(symtimes: Path, repos: set[str] | frozenset = frozenset()) -> None:
    """Rule 4 of V(r) (inherited members, state.py): compose stops unless PARENTS.json (written by symcache next to
    symtimes.jsonl) exists, lists every repo in `repos` (the repos being composed) and counts 0 unresolved parents in
    every repo."""
    gate = json.loads((symtimes.parent / "PARENTS.json").read_text())
    bad = {repo: g["unresolved"] for repo, g in gate.items() if g["unresolved"]}
    if bad:
        raise RuntimeError(f"unresolved parents: {bad}")
    missing = sorted(set(repos) - set(gate))
    if missing:
        raise RuntimeError(f"no parent gate for {missing}")


def load(config: Path, phase2: Path, symtimes: Path, full_root: Path, params: Params = Params()) -> Composer:
    parent_gate_ok(symtimes)
    sources = {s["repo"]: s for s in json.loads(config.read_text())}
    records = read_records(phase2)
    parent_gate_ok(symtimes, {r["repo"] for r in records.values()})
    catalog = read_catalog(phase2)
    forbidden = json.loads((phase2 / "forbidden.json").read_text())
    sym = load_symtimes(symtimes)
    commits = {r["repo"]: r["commit"] for r in records.values()}
    contexts = {repo: RepoContext(repo, s["language"], full_root / f"{s['dir']}.git", commits[repo], catalog[repo],
                                  sym.get(repo), strict=True) for repo, s in sources.items() if repo in commits}
    return Composer(contexts, dict(catalog), forbidden, records, params)


def pools(records: dict[str, dict]) -> dict[str, list[str]]:
    """Retrieval pools, named as the generation run's train_job subsets: "all" = every train record; one
    pool per learning-curve subset ("lc600", "lc1200")."""
    out: dict[str, list[str]] = {"all": sorted(i for i, r in records.items() if r["partition"] == "train")}
    for i, r in sorted(records.items()):
        for s in r["subsets"]:
            out.setdefault(s, []).append(i)
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dirty_code(root: Path) -> str:
    """Modified, staged or untracked files under v2tools/, v2gpu/ (the tokenizers and serialisation build_prompts
    uses), scripts/ and config/ ("" when clean): a script that records the v2tools commit must run exactly that commit's
    code."""
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--",
                           "v2tools", "v2gpu", "scripts", "config"], capture_output=True, text=True, check=True).stdout.strip()
