"""The check, added to the 3D run on 28/09/2026, that the SAP's results rest on what origin holds and on what the
registry names, run from a fresh clone of origin after sap_run ran there.

sap_verify.py --clone DIR --part test|aux_test --results FILE --step-results FILE [--ledger-ref REF]

--results is the SAP_RESULTS.json of the sap_run run from the clone, --step-results the one of the earlier run (test or
aux_test) in the working checkout. Exit 0 when both hold:
- the results row of --part in PREREG_LEDGER.md at --ledger-ref (default origin/v2-phase3-spec, read in the clone)
equals the run in all six columns (sap.LEDGER_COLUMNS: the two commits, the sha256 of the lock and of the META as
committed, the scores_sha256 the META names, and META_COMMIT's commit date in +07);
- the two SAP_RESULTS.json are equal in every field except sap.RESULTS_IGNORED (the code's HEAD, the files' paths).
Exit 3 otherwise, naming each difference. This script is part of the locked 3D analysis code (scripts/sap_*.py) and runs
from the clone, so the check itself is origin's code."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2tools.sap import git_run, ledger_errors, ledger_rows, ledger_values, results_differences  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clone", type=Path, required=True, help="the fresh clone of origin that sap_run ran from")
    ap.add_argument("--part", choices=("test", "aux_test"), required=True)
    ap.add_argument("--results", type=Path, required=True, help="SAP_RESULTS.json of the run from the clone")
    ap.add_argument("--step-results", type=Path, required=True,
                    help="SAP_RESULTS.json of the earlier run (test or aux_test) in the working checkout")
    ap.add_argument("--ledger-ref", default="origin/v2-phase3-spec", help="where PREREG_LEDGER.md is read in the clone")
    args = ap.parse_args()
    got = json.loads(args.results.read_text(encoding="utf-8"))
    ref = json.loads(args.step_results.read_text(encoding="utf-8"))
    ledger = git_run(args.clone, "show", f"{args.ledger_ref}:PREREG_LEDGER.md")
    if ledger.returncode != 0:
        print(f"refusing: no PREREG_LEDGER.md at {args.ledger_ref} in {args.clone}", file=sys.stderr)
        return 3
    bad = ledger_errors(ledger_rows(ledger.stdout.decode("utf-8")), args.part, ledger_values(args.clone, got))
    bad += [f"SAP_RESULTS.json of the fresh clone differs from the earlier step at {k}"
            for k in results_differences(got, ref)]
    if bad:
        print(f"refusing: {'; '.join(bad)}", file=sys.stderr)
        return 3
    print(f"{args.part}: khớp sổ §6 và khớp SAP_RESULTS của bước trước")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
