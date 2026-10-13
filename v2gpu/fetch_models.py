"""Download the three pinned models into HF_HOME and check every file against the Hub's LFS sha256.

python -m v2gpu.fetch_models REPORT.json [MODEL ...]   -> exit 0 only if all weights are present and every LFS digest
matches. Without MODEL arguments all three models are fetched with weights. With MODEL arguments only those get weights;
the others still get their tokenizer files, because the test suite reads all three tokenizers (the H100 comparison pod
trains only deepseek).
An HF_TOKEN, if the pod has one (RunPod secret), is read from the environment by huggingface_hub; it is never passed
on a command line or written to a report.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import sys
from pathlib import Path

from .models import MODELS
from .receipt import sha256_file

ALLOW = ["*.json", "model-*.safetensors", "tokenizer.model"]
TOKENIZER_ONLY = ["*.json", "tokenizer.model"]


def expected_lfs(siblings, patterns=ALLOW) -> dict[str, str]:
    """{filename: sha256} for the LFS files to download (model_info(..., files_metadata=True).siblings)."""
    return {s.rfilename: s.lfs.sha256 for s in siblings
            if s.lfs and any(fnmatch.fnmatch(s.rfilename, pat) for pat in patterns)}


def check(local_dir: Path, expected: dict[str, str], weights: bool = True) -> list[str]:
    problems = []
    local_dir = Path(local_dir)
    index = local_dir / "model.safetensors.index.json"
    if not weights:
        pass
    elif not index.exists():
        problems.append("model.safetensors.index.json missing")
    else:
        for name in sorted(set(json.loads(index.read_text())["weight_map"].values())):
            if name not in expected:
                problems.append(f"{name}: in the index but not an LFS file on the Hub")
    for name, sha in sorted(expected.items()):
        path = local_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256_file(path) != sha:
            problems.append(f"{name}: sha256 mismatch")
    return problems


def main(report: Path, models=None, api=None, download=None) -> int:
    if api is None or download is None:
        from huggingface_hub import HfApi, snapshot_download

        api, download = api or HfApi(), download or snapshot_download
    out, ok = {}, True
    for key, spec in MODELS.items():
        weights = not models or key in models
        patterns = ALLOW if weights else TOKENIZER_ONLY
        path = Path(download(spec.hf_id, revision=spec.revision, allow_patterns=patterns))
        exp = expected_lfs(api.model_info(spec.hf_id, revision=spec.revision, files_metadata=True).siblings, patterns)
        problems = check(path, exp, weights)
        ok &= not problems
        out[key] = {"spec": dataclasses.asdict(spec), "path": str(path), "weights": weights, "lfs_files": exp,
                    "problems": problems}
    Path(report).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2:] or None))
