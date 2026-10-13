"""Paid canary on the rented training card, independent of v2tools.compose (not yet specified).

Per model, synthetic examples are built from real Phase 2 records: the real Gherkin and the real answer, padded with
real code from OTHER records until prompt + answer is just under each target length (default 1,024 / 2,048 / 4,096 /
8,192: the real length histogram spans all four). A candidate whose prompt fails leakcheck (its own code, its method
name, its signature line) is skipped, so preflight passes.
Every run goes through train_job in canary mode, so preflight, receipts and exit codes are the real ones.

Pre-declared criteria per model:
- every length run reaches --steps (train_job exit 4, MAX_STEPS);
- the loaded attention is the requested one, and a forward with SDPA restricted to FLASH_ATTENTION runs;
- peak RESERVED memory at the longest length <= 90% of the card;
- GPU resume: at 2,048 tokens, N steps then --resume to 2N reaches 2N (functional: CUDA is not bit-deterministic);
- the saved adapter reloads onto a fresh base with an equal tensor digest;
- smoke.py --check codepath: generation runs and two runs are byte-identical (EOS is NOT expected after N steps).
Reported, not gated: tokens/s per length, the projection of scenario A on the measured histogram (3 epochs expected;
x1.25 for run-to-run noise; x5/3 for the 5-epoch ceiling), and for --compare-upcast MODEL three 50-step runs on the same
inputs and seed (decision of 24/09): bf16, bf16 again, and the original QLoRA fp32 upcast, run after every model's own
canary. Rule (decision of 24/09; the 0.5% equivalence margin is part of it): over
the last 10 steps, noise = mean |bf16_i - bf16again_i| and difference = mean |bf16_i - fp32_i|, paired by step; flag when
difference > max(2 x noise, 0.5% of the fp32 mean). Step 0 (identical weights: LoRA B starts at zero) isolates the purely
numerical difference and is flagged by the same rule on relative differences. A failed comparison is a flag too.
One model crashing never hides the others: canary.json is always written.
The inputs are written in the Phase 3 prompt-file format (records, catalog, forbidden, prompts with SUMS and META), so
the real train_job preflight, leak checks included, runs on every canary job.

python -m v2gpu.canary --records out/phase2/records/train.jsonl --out /workspace/runs/canary \
    [--models deepseek codellama mistral] [--plan-tokens config/plan_tokens_scenarioA.json --price 1.61]
Exit 0 only if every model passes every criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .examples import Example, jsonl_lines, pool_digest, read_records, write_jsonl
from .inputs import SPECIAL_TOKENS
from .leakcheck import Context, check
from .models import MODELS, load_tokenizer
from .schedule import FORMATS
from .serialize import answer_ids, prompt_ids

OVERHEAD, NOISE = 0.15, 1.25
LENGTHS = [1024, 2048, 4096, 8192]
RESUME_LENGTH = 2048
UPCAST_STEPS, UPCAST_NOISE_FACTOR, UPCAST_MIN_REL, UPCAST_TAIL = 50, 2.0, 0.005, 10
NO_PROVENANCE = {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}


def _messages(record: dict, context: str) -> list[dict]:
    return [{"role": "user", "content": f"Write the step definitions for this scenario.\n{record.get('background', '')}\n"
                                        f"{record['scenario']}\n### Context\n{context}"}]


def _answer(record: dict) -> str:
    return "\n\n".join(c["code"] for c in record["create"])


def _canary_record(record: dict, rid: str, partition: str) -> dict:
    """Phase 2 fields train_job needs; every CREATE function counts as new (the strictest leak rule); a feature path and
    fn ids of its own, so canary records have no neighbours; one suite, no annotation times, no patterns."""
    r = {"id": rid, "repo": record.get("repo", "canary/repo"), "language": record.get("language", "php"),
         "partition": partition, "subsets": [], "feature_path": f"canary/{rid}.feature", "suite": "canary", "tags": [],
         "scoping": None, "scenario_time": 0, "t_vis": None, "create_def_ids": [],
         "background": record.get("background", ""), "scenario": record["scenario"], "steps": record.get("steps", []),
         "create": [{"fn_id": f"{rid}#{i}", "path": c.get("path", "canary.php"), "code": c["code"],
                     "preexisting": False} for i, c in enumerate(record["create"])]}
    return {**r, "neighbours": []} if partition == "train" else r


def _catalog_rows(record: dict) -> list[dict]:
    return [{"repo": record["repo"], "fn_id": c["fn_id"], "code": c["code"], "first_line": 1, "suites": ["canary"],
             "catch_all": False, "scope_tags": [], "annotations": []} for c in record["create"]]


def _leaks(record: dict, messages: list[dict]) -> bool:
    """The real leakcheck on the candidate alone: its own code, method name or signature line in the prompt."""
    ctx = Context(pool={}, catalog={record["repo"]: {f["fn_id"]: f for f in _catalog_rows(record)}}, hidden={},
                  language={record["repo"]: record["language"]}, neighbours={})
    return bool(check(record, messages, NO_PROVENANCE, ctx))


def synthetic(records: list[dict], tok, target: int, n: int, tag: str = "t",
              offset: int = 0) -> list[tuple[dict, list[dict], str]]:
    """n (canary record, messages, answer) whose serialised length is in (target - 64, target] and whose prompt holds
    no slice of the record's own CREATE code; the first `offset` qualifying records are skipped (dev after train)."""
    pool = sorted((r for r in records if r["create"]), key=lambda r: r["id"])
    out, skipped = [], 0
    for r in pool:
        answer = _answer(r)
        own = {c["code"] for c in r["create"]}
        filler = "\n\n".join(_answer(o) for o in pool if o is not r and not own & {c["code"] for c in o["create"]})
        n_answer = len(answer_ids(tok, _messages(r, ""), answer))
        if len(prompt_ids(tok, _messages(r, ""))) + n_answer > target - 64:
            continue
        lo, hi = 0, min(len(filler), 8 * target)  # chars; ~3 chars per token leaves ample room
        while lo < hi:  # longest filler prefix that still fits
            mid = (lo + hi + 1) // 2
            if len(prompt_ids(tok, _messages(r, filler[:mid]))) + n_answer <= target:
                lo = mid
            else:
                hi = mid - 1
        cr = _canary_record(r, f"canary-{tag}{target}-{len(out)}", "train" if tag == "t" else "dev")
        msgs = _messages(r, filler[:lo])
        if _leaks(cr, msgs):
            continue
        if skipped < offset:
            skipped += 1
            continue
        out.append((cr, msgs, answer))
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} records fit under {target} tokens")
    return out


def project(rates: dict[int, float], buckets_m: dict[str, float], price: float, epochs_expected: int = 3,
            epochs_max: int = 5) -> dict:
    """Hours and dollars for the planned tokens, each length bucket at its own measured rate."""
    secs = sum(buckets_m[str(L)] * 1e6 / rates[L] for L in sorted(rates) if str(L) in buckets_m)
    h = secs / 3600 * (1 + OVERHEAD)
    h_cons, h_ceiling = h * NOISE, h * NOISE * epochs_max / epochs_expected
    return {"tokens_m": round(sum(buckets_m.values()), 2), "hours_expected": round(h, 2),
            "usd_expected": round(h * price, 2), "hours_conservative": round(h_cons, 2),
            "usd_conservative": round(h_cons * price, 2), "hours_ceiling_5_epochs": round(h_ceiling, 2),
            "usd_ceiling_5_epochs": round(h_ceiling * price, 2)}


def verdict(runs: dict, resume: dict | None, reload: tuple[str, str], smoke: dict | None, requested_attn: str) -> dict:
    """runs: {length: receipt}. 'n/a' only where the CPU test path cannot measure."""
    longest = runs[max(runs)]
    total = (longest.get("environment", {}).get("device") or {}).get("total_memory_gb")
    peak = longest.get("result", {}).get("peak_reserved_gb")
    flash = [r.get("model_load", {}).get("flash_kernel_eligible") for r in runs.values()]
    crit = {
        "all_runs_reached_max_steps": all(r.get("status") == "MAX_STEPS" for r in runs.values()),
        "attention_as_requested": all(r.get("model_load", {}).get("attn_implementation_loaded") == requested_attn
                                      for r in runs.values()),
        "flash_kernel_eligible": "n/a" if all(f is None for f in flash) else all(f is True for f in flash),
        "peak_reserved_at_longest_within_90pct": "n/a" if peak is None or total is None else peak <= 0.9 * total,
        "gpu_resume_reaches_2n": resume is not None and resume.get("status") == "MAX_STEPS"
        and resume.get("result", {}).get("global_step") == 2 * resume.get("_n", -1),
        "adapter_reload_digest_equal": reload[0] == reload[1],
        "smoke_codepath": bool(smoke and smoke.get("pass")),
    }
    return {"criteria": crit, "pass": all(v is True or v == "n/a" for v in crit.values()),
            "tokens_per_second": {L: r.get("result", {}).get("tokens_per_second") for L, r in runs.items()},
            "peak_reserved_gb_at_longest": peak, "device_total_gb": total}


def _write_inputs(d: Path, key: str, train_items, dev_items) -> None:
    pool = pool_digest(cr["id"] for cr, _, _ in train_items)
    (d / "records.jsonl").write_text("".join(json.dumps(cr) + "\n" for cr, _, _ in train_items))
    (d / "dev_records.jsonl").write_text("".join(json.dumps(cr) + "\n" for cr, _, _ in dev_items))
    (d / "catalog.jsonl").write_text("".join(json.dumps(row) + "\n" for cr, _, _ in train_items + dev_items
                                             for row in _catalog_rows(cr)))
    (d / "forbidden.json").write_text("{}")
    P = d / "prompts"  # the prompt directory holds nothing its sums do not list
    P.mkdir(exist_ok=True)
    sums = []
    for name, items, part in (("train.prompts.jsonl", train_items, "train"), ("dev.prompts.jsonl", dev_items, "dev")):
        digest = write_jsonl(P / name, [Example(cr["id"], f, key, part, pool, m, ans, NO_PROVENANCE)
                                        for cr, m, ans in items for f in FORMATS])
        sums.append(f"{digest}  {name}\n")
    (P / "excluded.jsonl").write_text("")
    (P / "infra.json").write_text("{}")
    ids = sorted(cr["id"] for cr, _, _ in train_items)
    (P / "pools.json").write_text(json.dumps({"all": {"sha256": pool_digest(ids), "ids": ids}}))
    (P / "PROMPTS_META.json").write_text(json.dumps({
        "v2tools_commit": None, "phase2_sums_sha256": None, "symtimes_sha256": None, "compose": None,
        "special_tokens_sha256": hashlib.sha256(SPECIAL_TOKENS.read_bytes()).hexdigest(),
        "note": "canary synthetic prompts", "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}}))
    sums += [f"{hashlib.sha256((P / n).read_bytes()).hexdigest()}  {n}\n"
             for n in ("excluded.jsonl", "infra.json", "pools.json", "PROMPTS_META.json")]
    (P / "PROMPTS_SHA256SUMS").write_text("".join(sums))


def _train(d: Path, key: str, L: int, steps: int, a, workdir: str = "run", extra=()) -> dict:
    cmd = [sys.executable, "-m", "v2gpu.train_job", "--mode", "canary", "--model", key, "--seed", "42",
           "--subset", "all", "--records", str(d / "records.jsonl"), "--dev-records", str(d / "dev_records.jsonl"),
           "--catalog", str(d / "catalog.jsonl"), "--forbidden", str(d / "forbidden.json"),
           "--train-prompts", str(d / "prompts" / "train.prompts.jsonl"),
           "--dev-prompts", str(d / "prompts" / "dev.prompts.jsonl"), "--prompts-sums", str(d / "prompts" / "PROMPTS_SHA256SUMS"),
           "--workdir", str(d / workdir), "--quant", a.quant, "--attn", a.attn, "--dtype", a.dtype,
           "--grad-accum", "1", "--max-seq", str(L), "--max-steps", str(steps), *extra]
    cmd += ["--model-path", a.model_path] if a.model_path else []
    with (d / f"{workdir}.log").open("a") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    rp = d / workdir / "receipt-train.json"
    return json.loads(rp.read_text()) if rp.exists() else {"status": "NO_RECEIPT"}


def _losses(workdir: Path) -> list[float]:
    path = workdir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in jsonl_lines(path)]
    return [r["loss"] for r in rows if "step" in r]


def _model_canary(key: str, a, records: list[dict], plan: dict) -> dict:
    from .modeling import LoadCfg, reload_roundtrip

    spec, tok = MODELS[key], load_tokenizer(MODELS[key])
    runs, resume = {}, None
    for L in a.lengths:
        d = a.out / f"{key}-{L}"
        d.mkdir(parents=True, exist_ok=False)
        _write_inputs(d, key, synthetic(records, tok, L, a.steps), synthetic(records, tok, L, 2, "d", a.steps))
        runs[L] = _train(d, key, L, a.steps, a)
        if L == a.resume_length:
            _train(d, key, L, a.steps, a, workdir="resume")
            resume = dict(_train(d, key, L, 2 * a.steps, a, workdir="resume", extra=["--resume"]), _n=a.steps)
    d = a.out / f"{key}-{max(a.lengths)}"
    adapter = d / "run" / "state" / "adapter"
    cfg = LoadCfg(a.model_path or spec.hf_id, None if a.model_path else spec.revision, a.quant, a.attn, a.dtype)
    reload = reload_roundtrip(cfg, str(adapter), str(a.out / f"{key}-reloaded")) if adapter.exists() else ("", "x")
    smoke_cmd = [sys.executable, "-m", "v2gpu.smoke", "--model", key, "--adapter", str(adapter), "--dev-prompts",
                 str(d / "prompts" / "dev.prompts.jsonl"), "--prompts-sums", str(d / "prompts" / "PROMPTS_SHA256SUMS"),
                 "--out", str(a.out / f"{key}-smoke.json"), "--check", "codepath",
                 "--n", "2", "--cap", str(a.smoke_cap), "--quant", a.quant, "--dtype", a.dtype]
    smoke_cmd += ["--model-path", a.model_path] if a.model_path else []
    with (a.out / f"{key}-smoke.log").open("w") as log:
        subprocess.run(smoke_cmd, stdout=log, stderr=subprocess.STDOUT)
    sp = a.out / f"{key}-smoke.json"
    smoke = json.loads(sp.read_text())["verdict"] if sp.exists() else None
    v = verdict(runs, resume, reload, smoke, a.attn)
    rates = {L: r for L, r in v["tokens_per_second"].items() if r}
    if key in plan.get("models", {}) and len(rates) == len(a.lengths):
        v["projection"] = project(rates, plan["models"][key], a.price, plan["epochs_expected"], plan["epochs_max"])
    return v


def _paired(a: list[float], b: list[float]) -> float:
    """Mean absolute difference, step by step, over the last UPCAST_TAIL steps of the common prefix both curves share
    (pairing the raw tails of curves of different lengths compares different steps)."""
    m = min(len(a), len(b))
    n = min(m, UPCAST_TAIL)
    return sum(abs(x - y) for x, y in zip(a[:m][-n:], b[:m][-n:])) / n


def upcast_verdict(bf16: list[float], bf16_again: list[float], fp32: list[float], steps: int | None = None) -> dict:
    """decision of 24/09: flag when the paired difference bf16 vs fp32 exceeds max(2 x the paired noise bf16 vs bf16 again,
    0.5% of the fp32 mean) over the last 10 steps, or when the relative step-0 difference exceeds max(2 x the relative
    step-0 noise, 0.5%). `steps`, when given, is the number of steps every run was meant to reach; a shorter curve is a
    flag of its own, not just three-way agreement on an accidentally truncated comparison."""
    if not (bf16 and bf16_again and fp32):
        return {"error": f"missing loss curve ({len(bf16)}, {len(bf16_again)}, {len(fp32)} steps)", "flag_for_review": True}
    lengths = [len(bf16), len(bf16_again), len(fp32)]
    if steps is not None and any(n != steps for n in lengths):
        return {"error": f"incomplete loss curve ({lengths} of {steps} steps)", "flag_for_review": True}
    noise, delta = _paired(bf16, bf16_again), _paired(bf16, fp32)
    tail_fp32 = sum(fp32[-UPCAST_TAIL:]) / len(fp32[-UPCAST_TAIL:])
    threshold = max(UPCAST_NOISE_FACTOR * noise, UPCAST_MIN_REL * abs(tail_fp32))
    rel0, noise0 = abs(bf16[0] - fp32[0]) / abs(fp32[0]), abs(bf16[0] - bf16_again[0]) / abs(fp32[0])
    step0 = {"bf16": bf16[0], "bf16_again": bf16_again[0], "fp32": fp32[0], "numeric_difference": abs(bf16[0] - fp32[0]),
             "relative_difference": rel0, "relative_noise": noise0,
             "flag": rel0 > max(UPCAST_NOISE_FACTOR * noise0, UPCAST_MIN_REL)}
    return {"steps": [len(bf16), len(bf16_again), len(fp32)], "step0": step0, "noise_floor": noise, "difference": delta,
            "threshold": threshold, "flag_for_review": delta > threshold or step0["flag"]}


def _upcast(a, records: list[dict]) -> dict:
    """Reported, never gated; any failure is itself a flag."""
    key, L = a.compare_upcast, a.resume_length
    try:
        tok = load_tokenizer(MODELS[key])
        d = a.out / f"{key}-upcast"
        d.mkdir(exist_ok=False)
        _write_inputs(d, key, synthetic(records, tok, L, a.upcast_steps), synthetic(records, tok, L, 2, "d", a.upcast_steps))
        results = {w: _train(d, key, L, a.upcast_steps, a, workdir=w, extra=extra)
                   for w, extra in (("bf16", ()), ("bf16_again", ()), ("fp32", ("--upcast-fp32",)))}
        curves = {w: _losses(d / w) for w in results}
        statuses = {w: r.get("status") for w, r in results.items()}
        if any(s != "MAX_STEPS" for s in statuses.values()):
            return {"model": key, "error": f"runs did not finish: {statuses}", "flag_for_review": True, "curves": curves}
        return {"model": key, **upcast_verdict(curves["bf16"], curves["bf16_again"], curves["fp32"], steps=a.upcast_steps),
                "curves": curves}
    except Exception as e:  # noqa: BLE001 - the comparison must never take the canary report down with it
        return {"model": key, "error": repr(e), "flag_for_review": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["deepseek", "codellama", "mistral"])
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    ap.add_argument("--resume-length", type=int, default=RESUME_LENGTH)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--compare-upcast", help="model key: 50 steps with and without the fp32 upcast")
    ap.add_argument("--upcast-steps", type=int, default=UPCAST_STEPS)
    ap.add_argument("--smoke-cap", type=int, default=64)
    ap.add_argument("--plan-tokens", type=Path)
    ap.add_argument("--price", type=float, default=1.61)
    ap.add_argument("--model-path")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    records = read_records(a.records)
    plan = json.loads(a.plan_tokens.read_text()) if a.plan_tokens else {}
    report = {"lengths": a.lengths, "steps": a.steps, "price_per_hour": a.price, "models": {}}
    try:
        for key in a.models:
            try:
                report["models"][key] = _model_canary(key, a, records, plan)
            except Exception as e:  # noqa: BLE001 - one model failing must not hide the others' numbers
                report["models"][key] = {"pass": False, "error": repr(e)}
        if a.compare_upcast:
            report["upcast_comparison"] = _upcast(a, records)
    finally:
        report["pass"] = bool(report["models"]) and all(m["pass"] for m in report["models"].values())
        if report["models"] and all("projection" in m for m in report["models"].values()):
            keys = next(iter(report["models"].values()))["projection"].keys()
            report["total"] = {k: round(sum(m["projection"][k] for m in report["models"].values()), 2) for k in keys}
        (a.out / "canary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
