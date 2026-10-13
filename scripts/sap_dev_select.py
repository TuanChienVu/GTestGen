"""Choose the compose configuration on dev by the pre-declared rule: keep the baseline unless another configuration
scores at least `margin_scenarios` more CA scenarios at Full-aware (anchor model). Reads the dev score files of every
configuration (one per variant; the rows' `variant` names it) and writes DEV_SELECTION.json, whose `inputs` (path as
given -> sha256 of every dev score file read and of the dev exclusions) sap_lock checks against the dev files it is
given."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2tools.sap import SapInputError, jsonl_rows, select_dev_config, sha256_of  # noqa: E402


def load(p: Path) -> list[dict]:
    return jsonl_rows(p)  # 3A's rule: "\n" only, a blank line inside the file refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--dev-scores", type=Path, nargs="+", required=True, help="dev score files (one per variant)")
    ap.add_argument("--dev-controls-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    dev_exclusions = args.dev_controls_dir / "exclusions.jsonl"
    excluded = {x["record_id"] for x in load(dev_exclusions)}
    try:
        res = select_dev_config([r for f in args.dev_scores for r in load(f)], cfg, excluded)
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 3
    res["inputs"] = {str(p): sha256_of(p) for p in [*args.dev_scores, dev_exclusions]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SapInputError as e:  # an input that is not JSONL (sap.jsonl_rows), wherever it is read: bad input
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(3)
