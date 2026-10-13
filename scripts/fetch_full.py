"""Full bare clones (every blob, no partial-clone filter) of the source repos at their pinned commits, for Phase 3:
repos-full/<dir>.git. repos-hist/ is blob:none and must never be read by Phase 3. Re-running only
verifies; exit 1 if any repo is missing its pinned commit or is a partial clone."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def pins(phase2: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in ("train", "dev", "test", "aux_test"):
        for line in (phase2 / "records" / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if out.setdefault(r["repo"], r["commit"]) != r["commit"]:
                raise ValueError(f"{r['repo']}: two pinned commits in the records")
    return out


def problems(git_dir: Path, commit: str) -> list[str]:
    if not git_dir.exists():
        return ["missing"]
    run = lambda *a: subprocess.run(["git", "-C", str(git_dir), *a], capture_output=True, text=True)
    out = []
    if run("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        out.append(f"commit {commit} absent")
    for key in ("extensions.partialClone", "remote.origin.promisor", "remote.origin.partialclonefilter"):
        if run("config", "--get", key).stdout.strip():
            out.append(f"partial clone ({key})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sources.json"))
    ap.add_argument("--phase2", type=Path, default=Path("out/phase2"))
    ap.add_argument("--root", type=Path, default=Path("repos-full"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    report = {}
    for src in json.loads(args.config.read_text()):
        commit = pins(args.phase2)[src["repo"]]
        git_dir = args.root / f"{src['dir']}.git"
        if not git_dir.exists() and not args.verify_only:
            subprocess.run(["git", "clone", "--quiet", "--bare", f"https://github.com/{src['repo']}.git", str(git_dir)],
                           check=True)
        report[src["repo"]] = {"commit": commit, "problems": problems(git_dir, commit)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if any(v["problems"] for v in report.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
