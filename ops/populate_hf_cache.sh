#!/usr/bin/env bash
# Populate the laptop's V2-only HF cache with the tokenizer-only files of the 3 pinned
# models, at their pinned revisions, so the laptop stops reading the user's default Hugging Face cache.
#
#   bash ops/populate_hf_cache.sh [HF_HOME]     (default: hf-cache/ at the repository root, see DEFAULT_HOME below)
#
# LAPTOP ONLY. Uses the network, against the Hugging Face Hub, to fetch v2gpu.fetch_models.TOKENIZER_ONLY patterns
# ("*.json", "tokenizer.model") for every model in v2gpu.models.MODELS, then checks every downloaded LFS file's
# sha256 against the Hub's own metadata (files_metadata=True), exactly as v2gpu.fetch_models.check() does for a real
# pod fetch -- the same expected_lfs()/check() functions, just forced to TOKENIZER_ONLY for every model (fetch_models
# itself always fetches weights for at least one model when given no MODEL arguments; this script never does).
# It only ever downloads and verifies: no file already in HF_HOME is ever deleted, moved or overwritten with a
# smaller/older one (huggingface_hub's own cache layout is content-addressed and additive).
set -euo pipefail
DEFAULT_HOME=$(cd "$(dirname "$0")/.." && pwd)/hf-cache
HF_HOME=${1:-$DEFAULT_HOME}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$HF_HOME"
echo "populate_hf_cache: HF_HOME=$HF_HOME (network: Hugging Face Hub)"
cd "$REPO"
set +e
HF_HOME="$HF_HOME" "$REPO/.venv-gpu/bin/python" - <<'PY'
import json

from huggingface_hub import HfApi, snapshot_download

from v2gpu.fetch_models import TOKENIZER_ONLY, check, expected_lfs
from v2gpu.models import MODELS

api = HfApi()
out = {}
ok = True
for key, spec in sorted(MODELS.items()):
    path = snapshot_download(spec.hf_id, revision=spec.revision, allow_patterns=TOKENIZER_ONLY)
    info = api.model_info(spec.hf_id, revision=spec.revision, files_metadata=True)
    exp = expected_lfs(info.siblings, TOKENIZER_ONLY)
    problems = check(path, exp, weights=False)
    ok &= not problems
    out[key] = {"hf_id": spec.hf_id, "revision": spec.revision, "path": path, "lfs_files_checked": sorted(exp),
                "problems": problems}
    print(f"{key}: {spec.hf_id}@{spec.revision[:12]} -> {path} "
          f"({len(exp)} LFS file(s) sha256-checked, {len(problems)} problem(s))")
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
RC=$?
set -e
if [ "$RC" = 0 ]; then
  echo "POPULATE_HF_CACHE_OK"
else
  echo "POPULATE_HF_CACHE_FAILED (see the problems above)" >&2
fi
exit "$RC"
