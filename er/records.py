"""Read-only access to the sealed Phase 2 records (Sylius only)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PHASE2 = Path(os.environ.get("ER_PHASE2", "out/phase2"))  # relative to the working directory: the repository root


def load_records(phase2: Path = PHASE2) -> dict[str, dict]:
    out = {}
    for part in ("test", "dev"):
        with (phase2 / "records" / f"{part}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["repo"] == "Sylius/Sylius":
                    out[r["id"]] = r
    return out


def seal_sha256(phase2: Path = PHASE2) -> str:
    """sha256 of PHASE2_SHA256SUMS: identifies the sealed dataset a result was computed on."""
    return hashlib.sha256((phase2 / "PHASE2_SHA256SUMS").read_bytes()).hexdigest()
