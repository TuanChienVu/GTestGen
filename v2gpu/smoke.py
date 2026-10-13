"""Greedy generation smoke test with HF generate, batch 1 (vLLM runs on the 16 GB generation card).

python -m v2gpu.smoke --model deepseek --adapter RUN/epoch-3 --dev-prompts DEV.F2.jsonl --prompts-sums SUMS --out smoke.json
    --check gate
The dev prompt files are checked against PROMPTS_SHA256SUMS first (the gate decides the other 10 jobs); a
mismatch writes a failed verdict with the problems and exits 2 before any model loads.
- `--check codepath` (canary): the adapter loads, generates, and two runs are byte-identical; EOS is not required
  (a 12-step canary adapter has not learned to stop).
- `--check gate` (after job 1, before the other 10 jobs): additionally at least 4 of the 5 prompts stop at EOS.
Budget per prompt: serialize.max_new_tokens (min(6,144, window - prompt)). The n prompts are spread over the repos (round
robin in record-id order), so a PHP-heavy dev set does not decide the gate alone. Exit 0 only if the check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .examples import prompt_files, read_jsonl
from .models import MODELS, load_tokenizer
from .serialize import decode_answer, max_new_tokens, prompt_ids

MIN_EOS = 4


def generate(model, tok, messages: list[dict], window: int, device, cap: int | None = None) -> dict:
    import torch

    ids = prompt_ids(tok, messages)
    budget = max_new_tokens(len(ids), window) if cap is None else min(cap, max_new_tokens(len(ids), window))
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=budget, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    new = out[0][len(ids):].tolist()
    return {"prompt_tokens": len(ids), "budget": budget, "new_ids": new, "new_tokens": len(new),
            "stopped_at_eos": tok.eos_token_id in new, "text": decode_answer(tok, new)}


def sample(examples: list, n: int) -> list:
    """Round robin over repos (the record id starts with "<repo>|"), each repo in record-id order."""
    by_repo: dict[str, list] = {}
    for e in sorted(examples, key=lambda e: e.record_id):
        by_repo.setdefault(e.record_id.split("|")[0], []).append(e)
    queues, out = [by_repo[r] for r in sorted(by_repo)], []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def verdict(first: list[dict], second: list[dict], check: str) -> dict:
    identical = [a["new_ids"] == b["new_ids"] for a, b in zip(first, second)]
    eos = sum(r["stopped_at_eos"] for r in first)
    ok = all(identical) and (check == "codepath" or eos >= MIN_EOS)
    return {"check": check, "pass": ok, "identical": identical, "stopped_at_eos": eos, "n": len(first),
            "min_eos_for_gate": MIN_EOS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--dev-prompts", type=Path, nargs="+", required=True)
    ap.add_argument("--prompts-sums", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", required=True, choices=["codepath", "gate"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--fmt", default="F2")
    ap.add_argument("--cap", type=int, help="tests only: lower the generation cap")
    ap.add_argument("--model-path")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args(argv)
    problems, prompts = prompt_files(a.dev_prompts, a.prompts_sums)
    if problems:
        a.out.write_text(json.dumps({"verdict": {"check": a.check, "pass": False}, "problems": problems,
                                     "prompts": prompts}, indent=2, sort_keys=True))
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 2

    import torch
    from peft import PeftModel

    from .modeling import LoadCfg, load_base, model_device

    spec = MODELS[a.model]
    tok = load_tokenizer(spec)
    cfg = LoadCfg(a.model_path or spec.hf_id, None if a.model_path else spec.revision, a.quant, "sdpa", a.dtype)
    base = load_base(cfg)
    model = PeftModel.from_pretrained(base, str(a.adapter), torch_device=str(model_device(base))).eval()
    device = model_device(model)
    exs = sample([e for (_, fmt), e in read_jsonl(a.dev_prompts).items() if fmt == a.fmt], a.n)
    window = spec.window if not a.model_path else model.config.max_position_embeddings
    runs = [[generate(model, tok, e.messages, window, device, a.cap) for e in exs] for _ in range(2)]
    v = verdict(runs[0], runs[1], a.check)
    a.out.write_text(json.dumps({"verdict": v, "examples": [e.record_id for e in exs], "prompts": prompts,
                                 "outputs": [{k: r[k] for k in r if k != "new_ids"} for r in runs[0]]},
                                indent=2, sort_keys=True))
    print(json.dumps(v))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
