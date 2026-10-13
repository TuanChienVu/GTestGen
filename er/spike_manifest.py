"""Convert overlays in the early per-condition layout (<slug>/{neg,pos,stub_f}, with a manifest.json) into a manifest
run of the er.manifest contract.

    python -m er.spike_manifest --spike <dir> --root <root> --run <run>

Used to exercise the runner before scripts/write_overlays.py existed, and for byte comparison with its output.
neg -> base (and condition neg), pos -> gold, stub_f -> stub. A condition keeps only the files that differ from base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import write_run
from .records import load_records

CONDITIONS = {"gold": "pos", "stub": "stub_f"}


def read_tree(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): p.read_text(encoding="utf-8") for p in sorted(d.rglob("*")) if p.is_file()}


def convert(spike: Path, root: Path, run: str, records: dict[str, dict]) -> Path:
    items = []
    for m in json.loads((spike / "manifest.json").read_text(encoding="utf-8")):
        rec, slug = records[m["record_id"]], Path(m["dir"]).name
        base = read_tree(spike / slug / "neg")
        items.append({"record": rec, "condition": "neg", "slug": slug, "base": base, "files": {}})
        for cond, src in CONDITIONS.items():
            tree = read_tree(spike / slug / src)
            items.append({"record": rec, "condition": cond, "slug": slug, "base": base,
                          "files": {k: v for k, v in tree.items() if base.get(k) != v}})
    return write_run(root, run, items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spike", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    print(convert(a.spike, a.root, a.run, load_records()))


if __name__ == "__main__":
    main()
