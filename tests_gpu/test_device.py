import json

import torch

from tests_gpu.tiny import make_tiny_llama
from v2gpu.modeling import LoadCfg, flash_probe, load_for_training


def pretend_cuda(monkeypatch):
    """torch.cuda.is_available() says True, as on a GPU host, while the card stays hidden (CUDA_VISIBLE_DEVICES="" in the
    CPU suite). torch's own accelerator probe must keep seeing no accelerator: Adam.step's graph-capture check (torch
    2.14) asks torch.accelerator.current_accelerator(check_available=True), which trusts the patched is_available, and
    then initialises CUDA: "No CUDA GPUs are available" on a pod (pre-canary A6000, 27/09), never on a CPU-only torch."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available=False: None)


def test_pretend_cuda_keeps_torchs_own_accelerator_probe_off(monkeypatch):
    pretend_cuda(monkeypatch)
    assert torch.cuda.is_available() and torch.accelerator.current_accelerator(check_available=True) is None


def test_cpu_tests_hide_the_gpu():
    """On a GPU pod the CPU suite must still run on CPU (the tiny model is never placed on the card)."""
    import os

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()


def test_flash_probe_is_none_for_a_cpu_model_even_when_cuda_exists(tmp_path, tokenizers, monkeypatch):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    model, _ = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32"), seed=42)
    pretend_cuda(monkeypatch)
    assert flash_probe(model, "cpu") is None


def test_train_job_follows_the_model_device_when_cuda_exists(tmp_path, tokenizers, monkeypatch):
    from tests_gpu.test_train_job import files, job
    from v2gpu import train_job

    args = files(tmp_path, tokenizers)
    pretend_cuda(monkeypatch)
    monkeypatch.setattr(train_job, "environment", lambda: {"stub": True})
    assert job(tmp_path, args) == 0
    r = json.loads((tmp_path / "run" / "receipt-train.json").read_text())
    assert r["status"] == "DONE" and r["model_load"]["flash_kernel_eligible"] is None


def test_smoke_follows_the_model_device_when_cuda_exists(tmp_path, tokenizers, monkeypatch):
    from tests_gpu import test_smoke

    pretend_cuda(monkeypatch)
    test_smoke.test_smoke_end_to_end_on_cpu(tmp_path, tokenizers)
