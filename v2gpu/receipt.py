"""Digests and environment facts written into every job receipt.

environment() also records what else decides the numbers: the sha256 of every
requirements-*.lock of the checkout (`locks`, {} before the first lock is committed), the nvidia-smi driver version
(`driver`, None without nvidia-smi), and the environment variables that steer CUDA, cuBLAS, cuDNN, PyTorch, TF32 and
vLLM (`numerics_env`: CUDA_*, CUBLAS_*, CUDNN_*, PYTORCH_*, TORCH_*, NVIDIA_TF32_OVERRIDE, VLLM_*) as set; and
`first_run_lock`, the {lock name: sha256} a first-run venv froze (FIRST_RUN_LOCK.json in the venv, written by
ops/pod_bootstrap.sh with FIRST_RUN=1; None otherwise).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
from pathlib import Path

PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "safetensors", "tokenizers", "huggingface_hub", "vllm")
REPO = Path(__file__).resolve().parents[1]
NUMERICS_ENV = ("CUDA_", "CUBLAS_", "CUDNN_", "PYTORCH_", "TORCH_", "NVIDIA_TF32_OVERRIDE", "VLLM_")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(safetensors_path: Path) -> str:
    """Digest of the tensors themselves (name, dtype, shape, bytes), independent of safetensors header metadata."""
    import torch
    from safetensors.torch import load_file

    h = hashlib.sha256()
    tensors = load_file(str(safetensors_path))
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {}}
    for p in PACKAGES:
        try:
            env["packages"][p] = md.version(p)
        except md.PackageNotFoundError:
            env["packages"][p] = None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["device"] = {"name": props.name, "capability": f"{props.major}.{props.minor}",
                             "total_memory_gb": round(props.total_memory / 1e9, 2)}
    except ImportError:
        env["cuda_available"] = False
    env["locks"] = {p.name: sha256_file(p) for p in sorted(Path(REPO).glob("requirements-*.lock"))}
    first = Path(sys.prefix) / "FIRST_RUN_LOCK.json"         # a venv the first pod built without a lock
    try:
        env["first_run_lock"] = json.loads(first.read_text()) if first.is_file() else None
    except ValueError:
        env["first_run_lock"] = {"unreadable": str(first)}
    env["numerics_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith(NUMERICS_ENV)}
    try:
        from .engines import nvidia_driver           # lazily: engines imports the training stack

        env["driver"] = nvidia_driver()
    except ImportError:
        env["driver"] = None
    return env


if __name__ == "__main__":
    import json

    print(json.dumps(environment(), indent=2, sort_keys=True))
