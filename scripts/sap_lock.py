"""Lock the SAP before any test output (§5 of the design specification and the analysis plan, fixed 24/09/2026): record
in SAP_LOCK.json the sha256 of the 3D analysis code (v2tools/stats.py, v2tools/sap.py and every scripts/sap_*.py
present, sap.code_files; no other code file), which must be a clean git checkout: an untracked or ignored
scripts/sap_*.py, a change not committed, a symlink among those files, or no git, refuse the lock (sap.code_tree_errors;
added 05/10/2026), and of every input: config/sap.json, MDE.json, CALIBRATION.json, DEV_SELECTION.json, the dev score
files and dev exclusions sap_mde read, the test controls, their mechanical exclusions and their future_methods.jsonl
(the future methods that decide CA), compose's excluded.jsonl, Phase 2's records/test.jsonl, sensitivity_subsets.json,
self_contained.jsonl, PHASE2_SHA256SUMS, the ER and ER-trace control results and the manifest.jsonl of their run
(--er-manifest, written by write_overlays), the aux_test controls and records (the descriptive-only run reads them), the
frozen pre-T list (--pre-t-flags, FLAGGED_PRE_T.tsv) and, when given, 3C's dose table (score_run --dose-table; --dose;
added 25/09/2026). The lock counts only once committed and pushed: sap_run takes the commit (--lock-commit). Refuses
(exit 2) when any input is missing (the ER control run included: no fallback), when the er_complete.json next to an ER
control file is missing, not complete, says running or has a manifest_sha256 that is not the --er-manifest's (the
er_results contract, 0c42865; it is checked, not locked), when an ER control file is empty, when the ER control results
do not cover exactly the exec_core records of records/test.jsonl with one gold and one stub row each (an exec_core
record without its gold row or its stub row, or with no row at all; a duplicated (record_id, condition) row; a row of a
test record that is not exec_core; the rows of records outside records/test.jsonl, the dev records of the same run, are
ignored: sap.er_scope), when an ER control row of a test record is not the result of its --er-manifest line (its
(record_id, condition) is not listed, or its overlay_sha256 is not that line's), when MDE.json or CALIBRATION.json was
not computed from the current config and calibration, when MDE.json's dev_inputs are not the dev files given here, when
the records file or sensitivity_subsets.json given here is not the one PHASE2_SHA256SUMS lists, when CALIBRATION.json's
inputs are not the test files and the records file given here, when DEV_SELECTION.json's inputs do not hold the chosen
variant's dev score files and the dev exclusions given here, or when the pre-T list is not "<partition>\\t<record_id>"
lines, its sha256 is not the one config/sap.json names (pre_t_flags), it lists a partition outside test, dev and
aux_test, or a test or aux_test record that is not one of the records given here (sap.pre_t_errors). sap_run refuses to
run when any sha256 differs. The git commit is recorded for information only. The lock is written once."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools.sap import (SapInputError, code_digest, code_tree_errors, digest_inputs, eligibility, er_complete_errors,  # noqa: E402
                         er_scope, git_head, listed_sha256, jsonl_rows, lock_inputs, manifest_overlays,
                         overlay_errors, pre_t_errors, pre_t_flags, provenance_errors)


def load(p: Path) -> list[dict]:
    return jsonl_rows(p)  # 3A's rule: "\n" only, a blank line inside the file refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/sap.json"))
    ap.add_argument("--mde", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--dev-selection", type=Path, required=True, help="DEV_SELECTION.json of sap_dev_select")
    ap.add_argument("--dev-scores", type=Path, nargs="+", required=True, help="the dev score files given to sap_mde")
    ap.add_argument("--dev-controls-dir", type=Path, required=True, help="dev controls run (exclusions.jsonl)")
    ap.add_argument("--controls-dir", type=Path, required=True,
                    help="test controls run (controls.jsonl, exclusions.jsonl, future_methods.jsonl)")
    ap.add_argument("--compose-excluded", type=Path, required=True, help="compose's excluded.jsonl of the test prompts")
    ap.add_argument("--records", type=Path, required=True, help="out/phase2/records/test.jsonl")
    ap.add_argument("--subsets", type=Path, required=True, help="out/phase2/sensitivity_subsets.json")
    ap.add_argument("--self-contained", type=Path, required=True, help="out/phase3/self_contained.jsonl")
    ap.add_argument("--phase2-sums", type=Path, required=True, help="out/phase2/PHASE2_SHA256SUMS")
    ap.add_argument("--er-controls", type=Path, nargs="+", required=True, help="ER and ER-trace control results")
    ap.add_argument("--er-manifest", type=Path, required=True,
                    help="manifest.jsonl of the ER controls run (write_overlays)")
    ap.add_argument("--aux-controls-dir", type=Path, required=True,
                    help="aux_test controls run (controls.jsonl, exclusions.jsonl, future_methods.jsonl)")
    ap.add_argument("--aux-records", type=Path, required=True, help="out/phase2/records/aux_test.jsonl")
    ap.add_argument("--pre-t-flags", type=Path, required=True,
                    help="the frozen pre-T list (evidence/pret_flags/FLAGGED_PRE_T.tsv)")
    ap.add_argument("--dose", type=Path, help="3C's dose table (score_run --dose-table; dose.jsonl: record_id, model, "
                                              "dose), when there is one")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print(f"{args.out} exists: the SAP is locked once", file=sys.stderr)
        return 2
    inputs = digest_inputs(lock_inputs(args))
    missing = sorted(k for k, v in inputs.items() if v is None)
    if missing:
        print(f"refusing: missing inputs {missing}", file=sys.stderr)
        return 2
    unfinished = er_complete_errors(args.er_controls, inputs["er_manifest"])
    if unfinished:
        print(f"refusing: {'; '.join(unfinished)}: the ER control results are read from a complete run only",
              file=sys.stderr)
        return 2
    if any(not p.read_text().strip() for p in args.er_controls):
        print("refusing: an ER control file is empty (no ER control run, no lock)", file=sys.stderr)
        return 2
    rows, expected = er_scope([x for p in args.er_controls for x in load(p)], load(args.records))
    try:
        eligibility(rows, {}, expected)
        stale = overlay_errors(rows, manifest_overlays(args.er_manifest))
    except SapInputError as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 2
    if stale:
        print(f"refusing: {'; '.join(stale)}", file=sys.stderr)
        return 2
    cfg, sel = json.loads(args.config.read_text()), json.loads(args.dev_selection.read_text())
    known = {"test": {r["id"] for r in load(args.records)}, "aux_test": {r["id"] for r in load(args.aux_records)}}
    bad = pre_t_errors(cfg, args.pre_t_flags, inputs["pre_t_flags"], pre_t_flags(args.pre_t_flags), known)
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 2
    chosen = [f"dev_scores[{i}]" for i, p in enumerate(args.dev_scores)
              if any(r["model"] == cfg["anchor_model"] and (r.get("variant") or "main") == sel.get("choice") for r in load(p))]
    bad = provenance_errors(json.loads(args.mde.read_text()), json.loads(args.calibration.read_text()), inputs, sel, cfg,
                            chosen, listed_sha256(args.phase2_sums))
    if bad:
        print(f"refusing: {'; '.join(bad)}: rerun sap_calibrate, sap_dev_select or sap_mde", file=sys.stderr)
        return 2
    tree = code_tree_errors(ROOT)  # the code of a clean git checkout only (added 05/10/2026)
    if tree:
        print(f"refusing: the 3D code is not a clean committed tree: {'; '.join(tree)}", file=sys.stderr)
        return 2
    code = code_digest(ROOT)
    if None in code.values():
        print(f"refusing: missing code files {sorted(k for k, v in code.items() if v is None)}", file=sys.stderr)
        return 2
    lock = {"code": code, "inputs": inputs, "tool_commit": git_head(ROOT),
            "locked_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=1) + "\n")
    print(json.dumps(lock))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SapInputError as e:  # an input that is not JSONL (sap.jsonl_rows), wherever it is read: bad input
        print(f"refusing: {e}", file=sys.stderr)
        raise SystemExit(2)
