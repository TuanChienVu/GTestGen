"""Write data/provenance.jsonl and data/summary.json: one line per file of the six source repositories that the dataset draws on,
at the commit the dataset records pin (scripts/fetch_full.py:pins), read from the full clones (repos-full/<dir>.git).

A file is listed once, with every role it has (`kinds`):
- feature: every .feature file of the repository at the pin (the scenario corpus that Phase 1 mined); `records` counts
  the dataset records whose scenario comes from it;
- step: every file matched by the source's step_globs (config/sources.json); `catalog_functions` counts its rows in
  out/phase2/catalog.jsonl;
- context: any other file whose declarations a prompt may show, e.g. a parent class or a trait
  (out/phase3/symtimes.jsonl);
- scoping: the Behat suite and service files that decide which suite a scenario runs in (sources.json "scoping").
Every line carries the repository's licence: the SPDX id of config/sources.json and the licence and NOTICE files, whose
presence at the pin is checked. The output is deterministic (sorted, no timestamps). Stops on any inconsistency: a
record's feature file or a catalog path that the pinned tree does not hold, or a missing licence file.
usage: python tools/build_provenance.py [--config config/sources.json] [--phase2 out/phase2]
           [--symtimes out/phase3/symtimes.jsonl] [--full-root repos-full] [--out data]"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_full import pins  # noqa: E402  (the one definition of the pinned commits)

# Licence and NOTICE files at the pinned commits (names differ between repositories; fineract ships the source
# distribution's pair, LICENSE_SOURCE and NOTICE_SOURCE).
LICENSE_FILES = {"Sylius/Sylius": ("LICENSE", None), "PrestaShop/PrestaShop": ("LICENSE.md", None),
                 "apache/fineract": ("LICENSE_SOURCE", "NOTICE_SOURCE"), "airslie/renalware": ("MIT-LICENSE", None),
                 "nhsconnect/gpconnect-provider-testing": ("LICENSE", None),
                 "openMF/ph-ee-integration-test": ("LICENSE", None)}
SCENARIO = re.compile(r"^\s*(Scenario|Scenario Outline|Scenario Template|Example)\s*:", re.M)
FEATURE = re.compile(r"^\s*Feature:\s*(.+)$", re.M)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def tree(git_dir: Path, commit: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(git_dir), "ls-tree", "-r", "-z", "--name-only", commit], check=True,
                         capture_output=True).stdout.decode("utf-8")
    return [n for n in out.split("\0") if n]


def blobs(git_dir: Path, commit: str, paths: list[str]) -> dict[str, tuple[str, bytes]]:
    """{path: (git blob sha1, bytes)} through one git cat-file --batch."""
    proc = subprocess.Popen(["git", "-C", str(git_dir), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    out = {}
    for p in paths:
        proc.stdin.write(f"{commit}:{p}\n".encode("utf-8"))
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise SystemExit(f"{git_dir.name}: {p} is not a blob at {commit}")
        out[p] = (header[0].decode(), proc.stdout.read(int(header[2])))
        proc.stdout.read(1)
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"{git_dir.name}: git cat-file failed")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config/sources.json")
    ap.add_argument("--phase2", type=Path, default=ROOT / "out/phase2")
    ap.add_argument("--symtimes", type=Path, default=ROOT / "out/phase3/symtimes.jsonl")
    ap.add_argument("--full-root", type=Path, default=ROOT / "repos-full")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    sources = {s["repo"]: s for s in json.loads(args.config.read_text(encoding="utf-8"))}
    pin = pins(args.phase2)
    records = Counter()
    for part in ("train", "dev", "test", "aux_test"):
        for r in jsonl(args.phase2 / "records" / f"{part}.jsonl"):
            records[(r["repo"], r["feature_path"])] += 1
    functions = Counter((r["repo"], r["path"]) for r in jsonl(args.phase2 / "catalog.jsonl"))
    shown = defaultdict(set)
    for r in jsonl(args.symtimes):
        shown[r["repo"]].add(r["path"])
    lines, summary = [], {}
    for repo in sorted(sources):
        s, commit = sources[repo], pin[repo]
        git_dir = args.full_root / f"{s['dir']}.git"
        names = tree(git_dir, commit)
        kinds = defaultdict(list)
        sc = s.get("scoping", {})
        for n in names:
            if n.endswith(".feature"):
                kinds[n].append("feature")
            if any(fnmatch.fnmatchcase(n, g) for g in s["step_globs"]):
                kinds[n].append("step")
            if n in shown[repo] and "step" not in kinds[n]:
                kinds[n].append("context")
            if any(fnmatch.fnmatchcase(n, g) for g in sc.get("suite_globs", []) + sc.get("service_globs", [])):
                kinds[n].append("scoping")
        missing = sorted({p for (rp, p) in records if rp == repo} - set(kinds))
        missing += sorted({p for (rp, p) in functions if rp == repo} - {n for n, k in kinds.items() if "step" in k})
        missing += sorted(shown[repo] - set(kinds))
        license_file, notice_file = LICENSE_FILES[repo]
        for f in (license_file, notice_file):
            if f is not None and f not in names:
                missing.append(f)
        if missing:
            raise SystemExit(f"{repo} @ {commit}: not in the pinned tree: {missing[:5]}")
        data = blobs(git_dir, commit, sorted(kinds))
        count = Counter()
        for path in sorted(kinds):
            blob, raw = data[path]
            row = {"repository": repo, "commit": commit, "path": path, "kinds": kinds[path],
                   "url": f"https://github.com/{repo}/blob/{commit}/{path}", "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha1": blob,
                   "license_spdx": s["license_spdx"], "license_file": license_file, "notice_file": notice_file}
            if "feature" in kinds[path]:
                text = raw.decode("utf-8", "replace")
                m = FEATURE.search(text)
                row |= {"feature": m.group(1).strip() if m else None, "scenario_count": len(SCENARIO.findall(text)),
                        "records": records[(repo, path)]}
                count["scenarios"] += row["scenario_count"]
                count["records"] += row["records"]
            if "step" in kinds[path]:
                row["catalog_functions"] = functions[(repo, path)]
                count["catalog_functions"] += row["catalog_functions"]
            count.update(f"{k}_files" for k in kinds[path])
            lines.append(row)
        summary[repo] = {"commit": commit, "license_spdx": s["license_spdx"], "license_file": license_file,
                         "notice_file": notice_file, "files": len(kinds)} | dict(sorted(count.items()))
    total = Counter()
    for v in summary.values():
        total.update({k: n for k, n in v.items() if isinstance(n, int)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "provenance.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                       for r in lines), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({"by_repository": summary, "total": dict(sorted(total.items()))},
                                                      indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(dict(sorted(total.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
