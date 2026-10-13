#!/usr/bin/env bash
# Pass only when the pytest summary is "<EXPECTED> passed", plus warnings and the bootstrap's own declared deselects:
# on the pod every collected test must run and pass. A skip means a missing tokenizer or package, i.e. not the
# environment the plan was tested with. Refused: skipped, xfailed, xpassed, failed, errors, and any deselect count
# other than DESELECTED. Warnings are accepted: the pod venvs hold packages the
# laptop never loads (bitsandbytes, vLLM), and a third-party DeprecationWarning must not fail the gate.
#   usage: no_skips.sh PYTEST_LOG EXPECTED_COUNT [DESELECTED]   (EXPECTED_COUNT from pytest --collect-only -q, with
#          the same --deselect options; DESELECTED is how many tests those options leave out, default 0)
set -euo pipefail
LOG=$1
EXPECTED=${2:?expected test count}
DESELECTED=${3:-0}
SUMMARY=$(tail -n 1 "$LOG")
DESEL=""
if [ "$DESELECTED" != 0 ]; then DESEL=", ${DESELECTED} deselected"; fi
PATTERN="^${EXPECTED} passed${DESEL}(, [0-9]+ warnings?)? in "
if [[ ! "$SUMMARY" =~ $PATTERN ]]; then
  echo "FAILED: expected exactly '$EXPECTED passed${DESEL}' (warnings allowed) and nothing else, got: $SUMMARY"
  grep -E 'SKIPPED|FAILED|ERROR|XPASS|XFAIL' "$LOG" || true
  exit 1
fi
echo "NO_SKIPS $EXPECTED passed${DESEL}"
