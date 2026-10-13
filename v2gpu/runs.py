"""The generation run matrix of V2 and the output line (§0b).

A run is (model, seed, condition). The five conditions pair model weights with a prompt format: zero_shot = base + F0,
rag_only = base + F1, ft_only = adapter + F0, full_neutral = adapter + F1, full_aware = adapter + F2.
- Base conditions decode greedily on the same base weights, so every seed would give the same output: ONE run per model,
  seed "base".
- Adapter conditions run once per trained seed: the training job's seed lock on the full train set
  (models.allowed_seeds(model, "all"): DeepSeek 42/11/23/7/99, CodeLlama and Mistral 42/11/23; one source).
- variant "main": test = every model (6 base runs + 33 adapter runs = 39); aux_test = DeepSeek only (2 + 15 = 17).
- variants "dev_pilot_v0" and "dev_pilot_v1": dev, DeepSeek base + seed 42 (5 runs).
- dev grid variants (labels fixed 24/09, shared with the scoring code): "dev_grid_k{k}_n{n}_r{r}" and
  "dev_grid_v1_k{k}_n{n}_r{r}", k in {4, 8, 12} (top-k), n in {1, 2, 3} (examples), r in {033, 050, 067} (skeleton share
  of block 4: 1/3, 1/2, 2/3); each is the run matrix of dev_pilot_v0 (dev, DeepSeek base + seed 42).
- variant "gen_canary": the 16 GB generation canary; any model and trained seed, never mixed with real outputs.
Runs are counted per record here; record counts are never constants: gencoverage derives them from the prompt files
and excluded.jsonl (e.g. on 24/09 the sealed dev set 8473872 had 206 records, 2 of them uses_test_infra, so 204 x 3
formats; excluded.jsonl may still change at the dev prompt build of 3A, so the counts are
recounted on every run).
`variant_spec` is the ONE resolver of a variant label (unknown or malformed labels raise ValueError); `variant_arg` is
its argparse face. A job is one engine load (one model, one weights source: "base" or one seed's adapter) and runs every
planned condition of those weights on the expected partitions (`job_runs`).

Output line (fixed 24/09): one JSON object per (record, run), exactly OUTPUT_FIELDS.
prompts_sha256 is the sha256 of the prompt FILE the prompt came from; prompt_sha256 the sha256 of
json.dumps(messages, ensure_ascii=False, sort_keys=True); adapter_tensor_sha256 is null for base runs;
leakcheck_receipt_sha256 is the sha256 of the laptop leak-audit receipt the job verified. The engine (vLLM version,
quantisation, max_num_seqs, card) belongs to the job receipt, not to the line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass

from .models import MODELS, allowed_seeds
from .serialize import GEN_CAP

CONDITIONS = {"zero_shot": ("base", "F0"), "rag_only": ("base", "F1"), "ft_only": ("adapter", "F0"),
              "full_neutral": ("adapter", "F1"), "full_aware": ("adapter", "F2")}
ALL_MODELS = ("codellama", "deepseek", "mistral")
RUN_SEEDS = {m: allowed_seeds(m, "all") for m in ALL_MODELS}
DEV_PILOT = {"dev": (("deepseek",), {"deepseek": (42,)})}
VARIANTS = {
    "main": {"test": (ALL_MODELS, RUN_SEEDS), "aux_test": (("deepseek",), RUN_SEEDS)},
    "dev_pilot_v0": DEV_PILOT,
    "dev_pilot_v1": DEV_PILOT,
    "gen_canary": {p: (ALL_MODELS, RUN_SEEDS) for p in ("train", "dev", "test", "aux_test")},
}
DEV_GRID = re.compile(r"dev_grid_(?:v1_)?k(4|8|12)_n([123])_r(033|050|067)")   # groups: k, n_examples, 100 x ratio
OUTPUT_FIELDS = ("record_id", "model", "seed", "condition", "variant", "fmt", "text", "prompt_tokens", "new_tokens",
                 "max_new_tokens", "stopped_at_eos", "prompts_sha256", "prompt_sha256", "adapter_tensor_sha256",
                 "leakcheck_receipt_sha256", "finish_reason")
FINISH_REASONS = ("stop", "length")


@dataclass(frozen=True, order=True)
class Run:
    model: str
    seed: str          # "s42", "s11", "s23", "s7", "s99" or "base"
    condition: str

    @property
    def weights(self) -> str:
        return CONDITIONS[self.condition][0]

    @property
    def fmt(self) -> str:
        return CONDITIONS[self.condition][1]


def seed_label(seed: int | None) -> str:
    return "base" if seed is None else f"s{seed}"


def variant_spec(variant: str) -> dict:
    """{partition: (models, {model: seeds})} of a variant label: a fixed variant or a dev grid label."""
    if isinstance(variant, str) and variant in VARIANTS:
        return VARIANTS[variant]
    if isinstance(variant, str) and DEV_GRID.fullmatch(variant):
        return DEV_PILOT
    raise ValueError(f"unknown variant {variant!r} (known: {sorted(VARIANTS)} and dev_grid_[v1_]k{{4|8|12}}_n{{1|2|3}}"
                     f"_r{{033|050|067}})")


GRID_CONDITIONS = ("full_aware",)


def variant_conditions(variant: str) -> tuple[str, ...]:
    """The conditions a variant runs, in CONDITIONS order. Since 04/10 a dev grid label runs Full-aware only --
    the selection rule reads only CA at Full-aware, and a grid label changes only the examples and
    the knowledge block, so its F0 prompts equal v0's and its F1 prompts come in three sets; every other variant runs
    all five conditions."""
    variant_spec(variant)                          # an unknown label raises here, as everywhere
    return GRID_CONDITIONS if DEV_GRID.fullmatch(variant) else tuple(CONDITIONS)


def variant_arg(value: str) -> str:
    """argparse type of --variant: the resolver decides, not a fixed choices list."""
    try:
        variant_spec(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return value


def planned_runs(variant: str, partition: str) -> list[Run]:
    spec = variant_spec(variant)
    if partition not in spec:
        raise ValueError(f"variant {variant!r} has no runs on partition {partition!r}")
    models, seeds = spec[partition]
    out = []
    for model in models:
        for cond in variant_conditions(variant):
            weights = CONDITIONS[cond][0]
            labels = ["base"] if weights == "base" else [seed_label(s) for s in seeds[model]]
            out += [Run(model, label, cond) for label in labels]
    return out


def job_runs(variant: str, partitions, model: str, weights: str) -> list[Run]:
    """The runs of one engine load, in CONDITIONS order; each must be planned on every expected partition."""
    wanted = [Run(model, weights, c) for c in variant_conditions(variant)
              if (CONDITIONS[c][0] == "base") == (weights == "base")]
    planned = {part: planned_runs(variant, part) for part in partitions}    # an unplanned partition raises first
    if not wanted:
        raise ValueError(f"{model} {weights}: {variant} plans no run on {weights!r} weights (conditions: "
                         f"{list(variant_conditions(variant))})")
    for part in partitions:
        missing = [r for r in wanted if r not in planned[part]]
        if missing:
            raise ValueError(f"{model} {weights}: runs {[r.condition for r in missing]} are not planned for {variant} "
                             f"{part} (seeds: {RUN_SEEDS.get(model)})")
    return wanted


def follows_budget_rule(prompt_tokens: int, max_new_tokens: int, window: int) -> bool:
    """True when max_new_tokens follows serialize.max_new_tokens: min(6,144, window - prompt_tokens)."""
    return max_new_tokens == min(GEN_CAP, window - prompt_tokens)


def prompt_sha256(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def row_key(row: dict) -> tuple[str, str, str, str, str]:
    return row["record_id"], row["model"], row["seed"], row["condition"], row["variant"]


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def row_problems(row, window_rule: bool = True) -> list[str]:
    """Why an output line is not a valid line of the output schema (empty list if it is). "length" means
    new_tokens == max_new_tokens and "stop" at least one token (the EOS id); for a job variant (any but gen_canary)
    max_new_tokens must be min(6144, window - prompt_tokens) of the model's pinned window, unless the writer switches
    that rule off (window_rule=False: test mode on a tiny model, where the job pins each budget itself)."""
    if not isinstance(row, dict):
        return ["output line is not a JSON object"]
    if set(row) != set(OUTPUT_FIELDS):
        return [f"output keys {sorted(row)} are not {sorted(OUTPUT_FIELDS)}"]
    out = []
    try:
        variant_spec(row["variant"])
    except ValueError:
        out.append(f"variant {row['variant']!r} is unknown")
    model = row["model"] if isinstance(row["model"], str) else None
    if model not in RUN_SEEDS:
        out.append(f"model {row['model']!r} is unknown")
    if row["condition"] not in CONDITIONS:
        return out + [f"condition {row['condition']!r} is unknown"]
    if isinstance(row["variant"], str) and DEV_GRID.fullmatch(row["variant"]) and \
            row["condition"] not in GRID_CONDITIONS:
        out.append(f"condition {row['condition']!r}: a dev_grid variant runs only {list(GRID_CONDITIONS)} by design")
    weights, fmt = CONDITIONS[row["condition"]]
    if row["fmt"] != fmt:
        out.append(f"fmt {row['fmt']!r} is not {fmt} for {row['condition']}")
    seeds = [seed_label(s) for s in RUN_SEEDS.get(model, ())]
    if weights == "base" and row["seed"] != "base":
        out.append(f"seed {row['seed']!r}: {row['condition']} runs on base weights (seed 'base')")
    if weights == "adapter" and row["seed"] not in seeds:
        out.append(f"seed {row['seed']!r} is not a trained seed of {row['model']} {seeds} "
                   f"(and 'base' has no {row['condition']})")
    if (row["adapter_tensor_sha256"] is None) != (weights == "base") or \
            (row["adapter_tensor_sha256"] is not None and not _is_sha(row["adapter_tensor_sha256"])):
        out.append("adapter_tensor_sha256 must be null for base runs and a sha256 for adapter runs")
    for k in ("prompts_sha256", "prompt_sha256", "leakcheck_receipt_sha256"):
        if not _is_sha(row[k]):
            out.append(f"{k} is not a sha256")
    if not isinstance(row["record_id"], str) or not isinstance(row["text"], str):
        out.append("record_id and text must be strings")
    if row["finish_reason"] not in FINISH_REASONS:
        out.append(f"finish_reason {row['finish_reason']!r} is not one of {FINISH_REASONS}")
    ints = [row[k] for k in ("prompt_tokens", "new_tokens", "max_new_tokens")]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in ints):
        out.append("prompt_tokens, new_tokens and max_new_tokens must be non-negative integers")
    elif row["new_tokens"] > row["max_new_tokens"]:
        out.append(f"new_tokens {row['new_tokens']} > max_new_tokens {row['max_new_tokens']}")
    else:
        if row["finish_reason"] == "length" and row["new_tokens"] != row["max_new_tokens"]:
            out.append(f"finish_reason 'length' with new_tokens {row['new_tokens']} below max_new_tokens "
                       f"{row['max_new_tokens']}")
        if row["finish_reason"] == "stop" and row["new_tokens"] < 1:
            out.append("finish_reason 'stop' with no new token (the EOS id is kept)")
        spec = MODELS.get(model)
        if window_rule and spec is not None and row["variant"] != "gen_canary" and not follows_budget_rule(
                row["prompt_tokens"], row["max_new_tokens"], spec.window):
            out.append(f"max_new_tokens {row['max_new_tokens']} is not min(6144, {spec.window} - prompt_tokens "
                       f"{row['prompt_tokens']})")
    if not isinstance(row["stopped_at_eos"], bool) or row["stopped_at_eos"] != (row["finish_reason"] == "stop"):
        out.append("stopped_at_eos must be true exactly when finish_reason is 'stop'")
    return out
