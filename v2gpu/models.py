"""The three base models (chosen 23/09/2026), pinned to the revisions already in the HF cache."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    window: int  # max_position_embeddings of the pinned config


MODELS = {
    "codellama": ModelSpec("codellama", "codellama/CodeLlama-7b-Instruct-hf",
                           "22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed", 16384),
    "deepseek": ModelSpec("deepseek", "deepseek-ai/deepseek-coder-6.7b-instruct",
                          "e5d64addd26a6a1db0f9b863abf6ee3141936807", 16384),
    "mistral": ModelSpec("mistral", "mistralai/Mistral-7B-Instruct-v0.2",
                         "63a8b081895390a26e140280378bc85ec8bce07a", 32768),
}
SEEDS = (42, 11, 23)
DEEPSEEK_EXTRA_SEEDS = (7, 99)  # the author's decision, 24/09: DeepSeek on the full train set also runs seeds 7 and 99


def allowed_seeds(model_key: str, subset: str) -> tuple[int, ...]:
    """Job-mode seed lock: DeepSeek on the full train set runs 5 seeds; every other (model, subset) keeps 3."""
    return SEEDS + DEEPSEEK_EXTRA_SEEDS if (model_key, subset) == ("deepseek", "all") else SEEDS


def load_tokenizer(spec: ModelSpec, path: str | None = None):
    """Tokenizer at the pinned revision; `path` overrides the hub id (a local snapshot on the pod)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or spec.hf_id, revision=None if path else spec.revision)
