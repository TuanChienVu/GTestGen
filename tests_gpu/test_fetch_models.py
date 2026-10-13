import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

from v2gpu import fetch_models


def test_effective_hf_home_on_the_laptop_is_never_under_the_users_cache():
    """The laptop must not read the user's default Hugging Face cache. conftest.py sets HF_HOME to
    the project's own cache at collection, before any test runs here, whenever HF_HOME was not already set by the
    caller (a pod's bootstrap, and the pod-parity gate container, always set it themselves to /workspace/hf, which
    this guard also accepts: it only refuses the home .cache directory itself or anything nested under it)."""
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).resolve()
    cache = (Path.home() / ".cache").resolve()
    assert home != cache and cache not in home.parents, \
        f"HF_HOME={home} is under {cache}: the laptop must read the V2-only hf-cache instead"


def repo(tmp_path, files: dict[str, bytes]):
    d = tmp_path / "snap"
    d.mkdir(exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-1.safetensors",
                                                                               "b": "model-2.safetensors"}}))
    return d


def sibs(files):
    out = [NS(rfilename=n, lfs=NS(sha256=hashlib.sha256(b).hexdigest())) for n, b in files.items()]
    return out + [NS(rfilename="pytorch_model.bin", lfs=NS(sha256="x")), NS(rfilename="config.json", lfs=None)]


FILES = {"model-1.safetensors": b"w1", "model-2.safetensors": b"w2", "tokenizer.model": b"t"}


def test_expected_lfs_keeps_only_downloaded_patterns():
    assert sorted(fetch_models.expected_lfs(sibs(FILES))) == sorted(FILES)


def test_clean_download_passes(tmp_path):
    assert fetch_models.check(repo(tmp_path, FILES), fetch_models.expected_lfs(sibs(FILES))) == []


def test_corrupt_and_missing_files_are_reported(tmp_path):
    d = repo(tmp_path, {**FILES, "model-1.safetensors": b"flipped"})
    (d / "tokenizer.model").unlink()
    problems = fetch_models.check(d, fetch_models.expected_lfs(sibs(FILES)))
    assert problems == ["model-1.safetensors: sha256 mismatch", "tokenizer.model: missing"]


def test_main_reports_per_model_and_exit_code(tmp_path):
    d = repo(tmp_path, FILES)
    api = NS(model_info=lambda *a, **k: NS(siblings=sibs(FILES)))
    assert fetch_models.main(tmp_path / "r.json", api=api, download=lambda *a, **k: str(d)) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert sorted(r) == ["codellama", "deepseek", "mistral"] and r["mistral"]["problems"] == []
    (d / "model-2.safetensors").write_bytes(b"bad")
    assert fetch_models.main(tmp_path / "r.json", api=api, download=lambda *a, **k: str(d)) == 1


def test_listed_models_get_weights_and_the_others_only_their_tokenizers(tmp_path):
    """The H100 comparison pod trains only deepseek, but the test suite needs all three tokenizers."""
    d = repo(tmp_path, FILES)
    tok_only = tmp_path / "tok"
    tok_only.mkdir()
    (tok_only / "tokenizer.model").write_bytes(b"t")
    calls = []

    def download(repo_id, revision, allow_patterns):
        calls.append((repo_id, tuple(allow_patterns)))
        return str(d if "deepseek" in repo_id else tok_only)

    api = NS(model_info=lambda *a, **k: NS(siblings=sibs(FILES)))
    assert fetch_models.main(tmp_path / "r.json", ["deepseek"], api=api, download=download) == 0
    r = json.loads((tmp_path / "r.json").read_text())
    assert sorted(r) == ["codellama", "deepseek", "mistral"]
    assert r["deepseek"]["weights"] and not r["mistral"]["weights"] and r["mistral"]["problems"] == []
    assert dict(calls)["mistralai/Mistral-7B-Instruct-v0.2"] == tuple(fetch_models.TOKENIZER_ONLY)
