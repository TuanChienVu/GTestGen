"""Publish the MDE of every confirmatory comparison before any test output (§5 of the design specification), for both
tests: (1) from the dev pilot's score rows (anchor model, the locked dev variant, one seed), (2) as MDE(1) x r from
sap_calibrate's CALIBRATION.json. The target is the test's usable records (controls minus the mechanical exclusions minus
compose's excluded.jsonl), counted in clusters (a .feature file within its repo; primary) and in scenarios. Each dev
variant has its own score file; the rows of other variants are left out. MDE.json records the sha256 of the
config, of CALIBRATION.json and of every dev file read (dev_inputs), which sap_lock checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.sap import SapInputError, jsonl_rows, mde_from_dev, sha256_of, usable_records  # noqa: E402


def load(p: Path) -> list[dict]:
    return jsonl_rows(p)  # 3A's rule: "\n" only, a blank line inside the file refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--dev-scores", type=Path, nargs="+", required=True, help="dev score files (one per variant)")
    ap.add_argument("--dev-controls-dir", type=Path, required=True)
    ap.add_argument("--test-controls-dir", type=Path, required=True)
    ap.add_argument("--compose-excluded", type=Path, required=True, help="compose's excluded.jsonl of the test prompts")
    ap.add_argument("--calibration", type=Path, required=True, help="CALIBRATION.json of sap_calibrate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    rows = [r for f in args.dev_scores for r in load(f)]
    dev_exclusions = args.dev_controls_dir / "exclusions.jsonl"
    dev_excluded = {x["record_id"] for x in load(dev_exclusions)}
    test_files = [args.test_controls_dir / "controls.jsonl", args.test_controls_dir / "exclusions.jsonl", args.compose_excluded]
    keep = usable_records(*(load(p) for p in test_files), cfg["anchor_model"])
    scenarios = {rid.rsplit("|", 1)[0] for rid in keep}
    clusters = {s.rsplit("|", 1)[0] for s in scenarios}
    try:
        table = mde_from_dev(rows, cfg, dev_excluded, len(clusters), len(scenarios), json.loads(args.calibration.read_text()))
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    out = {"config_sha256": sha256_of(args.config), "calibration_sha256": sha256_of(args.calibration),
           "dev_variant": cfg["dev"]["variant"], "dev_inputs": {str(p): sha256_of(p) for p in [*args.dev_scores, dev_exclusions]},
           "test_inputs": {str(p): sha256_of(p) for p in test_files},
           "clusters_target": len(clusters), "n_target": len(scenarios), "mde": table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"target: {len(clusters)} clusters, {len(scenarios)} scenarios")
    for k, v in table.items():
        print(k, "–" if v is None else " ".join(f"{n}={v[n]:.4f}" for n in sorted(v) if n.startswith("mde")) + (" (degenerate)" if v["degenerate"] else ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SapInputError as e:  # an input that is not JSONL (sap.jsonl_rows), wherever it is read: bad input
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(3)
