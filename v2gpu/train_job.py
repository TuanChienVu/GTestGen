"""One adapter job: python -m v2gpu.train_job --model deepseek --seed 42 --subset all ... --workdir RUN.

Inputs are the prebuilt prompt files of Phase 3 (§1.5), checked against PROMPTS_SHA256SUMS (META and excluded.jsonl
included), plus the Phase 2 records, catalog and forbidden.json, checked against PHASE2_SHA256SUMS (--phase2-sums,
required in job mode; the catalog may be the one inputs.strip_catalog derived, through its receipt, content-checked by
inputs.catalog_problems whatever it is named). In every mode, inputs.unstripped_problems refuses a catalog that still
carries the `code` of a forbidden test or aux answer function. META must name that seal and the
pinned tokenizer revisions. Preflight re-checks every anti-leak rule independently (leakcheck, through
examples.audit_prompt) on every train and dev prompt, and checks that each answer holds the code of every CREATE item.
Modes: `job` (default) locks every training parameter to the pre-declared values; `canary` allows --max-steps and the
upcast comparison; `test` (CPU tests) allows everything. `--preflight-only` runs the input checks and exits, on the
laptop, before any pod exists.
Rule D: a (record, format) whose prompt + answer exceeds --max-seq is excluded from that format and counted per format;
it never refuses the job. A (record, format) with no prompt refuses the job unless excluded.jsonl (beside the sums,
listed in them) names it; such pairs are counted by reason, and a listed pair that still has a prompt is refused.
Neighbours: the train records' `neighbours` field must equal the direct links computed here, and no
dev record may be a direct neighbour of a train record; either is a refusal, never a silent drop.
Exit codes: 0 DONE (or preflight OK), 4 MAX_STEPS (canary), 2 refused input (nothing trained), 3 attention or flash
kernel not as required (nothing trained), 1 crash. Every exit after the workdir check leaves receipt-train.json; a prompt
line the reader refuses (keys, an answer in a test line) exits 2 before it, with the reason on stderr.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from .examples import (audit_prompt, control_strings, coverage, pool_digest, pool_problems, prompt_files, read_excluded,
                       jsonl_lines, read_infra, read_jsonl, read_pools, read_records, retrieval_pool, subset_ids)
from .inputs import catalog_problems, meta_problems, phase2_problems, unstripped_problems
from .leakcheck import Context, neighbour_problems, neighbours, normalize_ws, render_answer
from .models import MODELS, allowed_seeds, load_tokenizer
from .receipt import environment, sha256_file, tensor_digest
from .schedule import FORMATS, epoch_items
from .serialize import AnswerTooLong, encode
from .train_loop import IdentityMismatch, LoopCfg, run, state_dir

SCHEMA = "gtestgen.v2.train_receipt.v3"
# Decision 27/09, after the real A6000 canary: fp32 upcast is now mandatory for every real
# training job (fp32 loses 28% throughput but is lower-loss on 41/50 steps; the pre-declared flag fired). A real job
# is one given --phase2-sums (job mode always requires it, see the flag check in main()); a job without
# --upcast-fp32 is REFUSED here, exit 2, with the same clear "job mode requires True" message every other locked
# parameter gets. Canary jobs (mode "canary") never go through JOB_LOCK at all, so the canary's own bf16/fp32
# upcast comparison (--compare-upcast) is unaffected.
JOB_LOCK = {"lr": 2e-4, "grad_accum": 16, "max_seq": 8192, "quant": "nf4", "attn": "sdpa", "dtype": "bfloat16",
            "max_steps": None, "model_path": None, "no_gradient_checkpointing": False, "upcast_fp32": True}
# Every v2gpu module train_job, train_loop and modeling load between them
# (tested: test_every_v2gpu_module_the_training_code_loads_is_in_its_code_digest), in this checkout.
# Since 27/09: v2gpu/engines.py is loaded too (receipt.environment() calls
# engines.nvidia_driver), but it is deliberately EXCLUDED here: it only fills receipt.environment.driver (a report
# field, never a training decision), and it is the generation engine's own module. Including it would freeze the
# generation engine code for the whole span of a variant's training (every job would need the SAME train_code_sha256),
# for zero effect on how training actually runs. The module probe below runs a real tiny job and checks the
# loaded modules are exactly TRAIN_CODE plus {engines} -- not TRAIN_CODE alone -- so this exclusion is provable, not
# just asserted.
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CODE = ("v2gpu/train_job.py", "v2gpu/train_loop.py", "v2gpu/modeling.py", "v2gpu/examples.py",
             "v2gpu/inputs.py", "v2gpu/leakcheck.py", "v2gpu/models.py", "v2gpu/receipt.py", "v2gpu/schedule.py",
             "v2gpu/serialize.py")


def train_code_sha256(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative path: sha256} of every module whose code decides how a job trains, in this checkout."""
    return {rel: sha256_file(Path(root) / rel) for rel in TRAIN_CODE}


def train_cfg_sha256(train_config: dict) -> str:
    """The canonical (sorted-key) JSON digest of the normalised training config."""
    return hashlib.sha256(json.dumps(train_config, sort_keys=True).encode()).hexdigest()


def check_attention(info: dict) -> str | None:
    """Refusal reason when the loaded attention is not the requested one (e.g. Mistral falling back to eager)."""
    if info["attn_implementation_loaded"] != info["attn_implementation_requested"]:
        return (f"requested {info['attn_implementation_requested']!r} but loaded "
                f"{info['attn_implementation_loaded']!r}")
    return None


def locked_violations(args: argparse.Namespace) -> list[str]:
    out = [f"--{k.replace('_', '-')}={getattr(args, k)!r} (job mode requires {v!r})"
           for k, v in JOB_LOCK.items() if getattr(args, k) != v]
    allowed = allowed_seeds(args.model, args.subset)
    if args.seed not in allowed:
        out.append(f"--seed={args.seed} (job mode requires one of {allowed})")
    return out


def schedule_digest(record_ids: list[str], seed: int, cfg: LoopCfg) -> str:
    h = hashlib.sha256()
    for e in range(1, cfg.epochs + cfg.max_extra_epochs + 1):
        for it in epoch_items(record_ids, seed, e):
            h.update(f"{it.epoch}|{it.record_id}|{it.fmt}\n".encode())
    return h.hexdigest()


def read_catalog(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for line in jsonl_lines(path):
        fn = json.loads(line)
        out.setdefault(fn["repo"], {})[fn["fn_id"]] = fn
    return out


def preflight(tok, model_key: str, records: list[dict], dev_records: list[dict], ids: list[str], train_ex: dict,
              dev_ex: dict, catalog: dict, forbidden: dict, max_seq: int,
              compose_excluded: dict | None = None, infra: dict | None = None, pools: dict | None = None,
              pool_name: str = "all") -> tuple[list[str], dict, set]:
    """Every check that needs no GPU, over every prompt the job can touch; returns (problems, stats, excluded).
    compose_excluded is read_excluded's {(record_id, fmt): reason} for this model; infra is read_infra's pairs; pools is
    read_pools' pools.json, whose pool `pool_name` must equal the pool recomputed here."""
    listed = compose_excluded or {}
    problems = neighbour_problems(records, dev_records)
    by_id = {r["id"]: r for r in records}
    dev_by_id = {r["id"]: r for r in dev_records}
    idset = set(ids)
    problems += [f"record {i} is partition {by_id[i].get('partition')!r}, not 'train'" for i in ids
                 if by_id[i].get("partition") != "train"]
    problems += [f"dev record {i} is partition {r.get('partition')!r}, not 'dev'" for i, r in sorted(dev_by_id.items())
                 if r.get("partition") != "dev"]
    problems += [f"record {i} is in both train and dev" for i in sorted(set(by_id) & set(dev_by_id))]
    by_reason = {}
    for part, wanted, have in (("train", [(i, f) for i in ids for f in FORMATS], train_ex),
                               ("dev", [(i, f) for i in sorted(dev_by_id) for f in FORMATS], dev_ex)):
        found, by_reason[part] = coverage(part, wanted, have, listed)
        problems += found
    problems += [f"train prompt {k[0]} {k[1]} is outside the subset" for k in sorted(train_ex) if k[0] not in idset]
    problems += [f"dev prompt {k[0]} {k[1]} is not a dev record" for k in sorted(dev_ex) if k[0] not in dev_by_id]
    pool_ids = retrieval_pool(records, ids, listed, control_strings(), catalog)
    pool = pool_digest(pool_ids)
    problems += pool_problems(pools or {}, pool_name, pool_ids)
    ctx = Context(pool={i: by_id[i] for i in pool_ids}, catalog=catalog, hidden={},
                  language={r["repo"]: r["language"] for r in records + dev_records},
                  neighbours=neighbours(records + dev_records), infra=infra or {})
    excluded, longest, n = set(), 0, 0
    counts = {"train": Counter(), "dev": Counter()}
    dropped = {"train": Counter(), "dev": Counter()}
    report = Counter()
    for part, exs, table in (("train", train_ex, by_id), ("dev", dev_ex, dev_by_id)):
        for (rid, fmt), ex in sorted(exs.items()):
            record = table.get(rid)
            if record is None or (part == "train" and rid not in idset):
                continue
            n += 1
            label = f"{rid} {fmt}"
            try:
                problems += audit_prompt(record, ex, part, model_key, pool, ctx, forbidden, report, dropped[part])
                if ex.answer is None:
                    problems.append(f"{label}: no answer in a {part} prompt")
                    continue
                answer = normalize_ws(ex.answer)
                fns = catalog.get(record["repo"], {})
                golds = {c["fn_id"]: render_answer(fns[c["fn_id"]], record.get("t_vis"), set(record.get("create_def_ids") or []))
                         if c["preexisting"] and "code" in fns.get(c["fn_id"], {}) else c["code"] for c in record["create"]}
                missing = [f for f, code in golds.items() if normalize_ws(code) not in answer]
                if missing:
                    problems.append(f"{label}: answer lacks the code of CREATE function(s) {missing}")
                longest = max(longest, len(encode(tok, ex.messages, ex.answer, max_seq).input_ids))
            except AnswerTooLong:
                excluded.add((rid, fmt))
                counts[part][fmt] += 1
            except Exception as e:  # noqa: BLE001 - a missing field, control tokens, or messages the template rejects
                problems.append(f"{label}: {type(e).__name__}: {e}")
    stats = {"longest_tokens": longest, "examples_checked": n, "pool_sha256": pool,
             "excluded_rule_d": {p: dict(sorted(c.items())) for p, c in counts.items()},
             "excluded_by_compose": {p: dict(sorted(c.items())) for p, c in by_reason.items()},
             "dropped_by_compose": {p: dict(sorted(c.items())) for p, c in dropped.items()},
             "report_only": {k: v for k, v in sorted(report.items()) if v},
             "excluded_sha256": hashlib.sha256("\n".join(f"{a}|{b}" for a, b in sorted(excluded)).encode()).hexdigest()}
    return problems, stats, excluded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--subset", required=True, choices=["all", "lc600", "lc1200"])
    ap.add_argument("--records", type=Path, required=True, help="Phase 2 records/train.jsonl")
    ap.add_argument("--dev-records", type=Path, required=True, help="Phase 2 records/dev.jsonl")
    ap.add_argument("--catalog", type=Path, required=True, help="Phase 2 catalog.jsonl")
    ap.add_argument("--forbidden", type=Path, required=True, help="Phase 2 forbidden.json")
    ap.add_argument("--train-prompts", type=Path, nargs="+", required=True, help="prebuilt, this model, this subset")
    ap.add_argument("--dev-prompts", type=Path, nargs="+", required=True, help="prebuilt, this model, this subset's pool")
    ap.add_argument("--prompts-sums", type=Path, required=True, help="PROMPTS_SHA256SUMS of the prompt files")
    ap.add_argument("--phase2-sums", type=Path, help="PHASE2_SHA256SUMS of the seal (required in job mode)")
    ap.add_argument("--catalog-sha256", help="sha256 of --catalog, written into the job's command on the laptop "
                                             "(QUEUE.json is the job's input manifest); required in job mode")
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--mode", default="job", choices=["job", "canary", "test"])
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--model-path", help="local snapshot or tiny test model (default: hub id at the pinned revision)")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--upcast-fp32", action="store_true",
                    help="the original QLoRA fp32 upcast of the non-quantised weights; required for every real job "
                         "(job mode, JOB_LOCK), optional in canary mode")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    violations = locked_violations(a) + [f"{flag} is required in job mode" for flag, v in (
        ("--phase2-sums", a.phase2_sums), ("--catalog-sha256", a.catalog_sha256)) if v is None]
    if a.mode == "job" and violations:
        print("REFUSED: " + "; ".join(violations), file=sys.stderr)
        return 2
    if a.mode == "canary" and a.max_steps is None:
        print("REFUSED: canary mode needs --max-steps", file=sys.stderr)
        return 2
    spec = MODELS[a.model]
    tok = load_tokenizer(spec)
    records, dev_records = read_records(a.records), read_records(a.dev_records)
    ids = subset_ids(records, a.subset)
    file_problems, prompts_meta = prompt_files(list(a.train_prompts) + list(a.dev_prompts), a.prompts_sums)
    excluded_problems, compose_excluded = read_excluded(a.prompts_sums, a.model)
    infra_problems, infra = read_infra(a.prompts_sums)
    pools_problems, pools = read_pools(a.prompts_sums)
    file_problems += excluded_problems + infra_problems + pools_problems + meta_problems(
        prompts_meta["meta"], a.phase2_sums, synthetic=a.mode == "canary")     # the canary's synthetic prompts
    forbidden = json.loads(a.forbidden.read_text())
    if a.catalog_sha256 is not None and sha256_file(a.catalog) != a.catalog_sha256:
        file_problems.append(f"{a.catalog.name}: sha256 is not the --catalog-sha256 of the job")
    file_problems += unstripped_problems(a.catalog, forbidden)  # every mode: never train on an unstripped catalog
    if a.phase2_sums is not None:
        file_problems += phase2_problems([a.records, a.dev_records, a.forbidden], a.phase2_sums)
        file_problems += catalog_problems(a.catalog, a.phase2_sums)
    try:
        train_ex, dev_ex = read_jsonl(a.train_prompts), read_jsonl(a.dev_prompts)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    problems, stats, excluded = preflight(tok, a.model, records, dev_records, ids, train_ex, dev_ex,
                                          read_catalog(a.catalog), forbidden, a.max_seq,
                                          compose_excluded, infra, pools, a.subset)
    problems = file_problems + problems
    if a.preflight_only:
        print(json.dumps({"problems": problems[:50], "n_problems": len(problems), **stats}, indent=2))
        return 2 if problems else 0
    if a.workdir is None:
        print("REFUSED: --workdir is required", file=sys.stderr)
        return 2

    a.workdir.mkdir(parents=True, exist_ok=True)
    receipt_path = a.workdir / "receipt-train.json"
    if not a.resume and any(a.workdir.iterdir()):
        print(f"REFUSED: {a.workdir} is not empty; use a fresh workdir or --resume", file=sys.stderr)
        return 2  # write nothing into someone else's run directory
    if a.resume:
        if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "DONE":
            print(f"REFUSED: {a.workdir} already finished (receipt status DONE)", file=sys.stderr)
            return 2
        if receipt_path.exists():
            # The previous attempt's receipt is archived for evidence, but it is no
            # longer the reference for the identity check below -- a killed attempt (SIGTERM/SIGKILL, a pod loss)
            # leaves state/ with no receipt at all, so a receipt-only compare could never see it.
            k = len(list(a.workdir.glob("receipt-train.attempt-*.json"))) + 1
            receipt_path.rename(a.workdir / f"receipt-train.attempt-{k}.json")
    receipt = {"schema": SCHEMA, "mode": a.mode, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "argv": sys.argv, "model": dataclasses.asdict(spec), "seed": a.seed, "subset": a.subset,
               "resume": a.resume}

    def finish(status: str, code: int, **extra) -> int:
        receipt.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return code

    if a.resume and state_dir(a.workdir) is None:
        return finish("REFUSED_NO_STATE", 2)
    cfg = LoopCfg(lr=a.lr, grad_accum=a.grad_accum, max_seq=a.max_seq, max_steps=a.max_steps)
    # the normalised training config: every field a resume must match, but never --max-steps
    # (canary's one allowed knob: how a canary resume extends training further; pinned by
    # test_canary_max_steps_exits_4_and_resume_continues)
    receipt["train_config"] = {"lr": a.lr, "grad_accum": a.grad_accum, "max_seq": a.max_seq, "quant": a.quant,
                               "attn": a.attn, "dtype": a.dtype,
                               "gradient_checkpointing": not a.no_gradient_checkpointing,
                               "upcast_fp32": a.upcast_fp32}
    # Every new receipt carries the digest of the code that decided how it
    # trained, and of its own normalised config, so genqueue and gencoverage can require one value of each per variant.
    receipt["train_code_sha256"] = train_code_sha256()
    receipt["train_cfg_sha256"] = train_cfg_sha256(receipt["train_config"])
    receipt["data"] = {"records_sha256": sha256_file(a.records), "dev_records_sha256": sha256_file(a.dev_records),
                       "catalog_sha256": sha256_file(a.catalog), "forbidden_sha256": sha256_file(a.forbidden),
                       "prompt_files": {str(p): sha256_file(p) for p in list(a.train_prompts) + list(a.dev_prompts)},
                       "prompts": prompts_meta, "n_records": len(ids),
                       "phase2_sums_sha256": sha256_file(a.phase2_sums) if a.phase2_sums else None,
                       "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                       "pool_sha256": stats["pool_sha256"], "schedule_sha256": schedule_digest(ids, a.seed, cfg)}
    receipt["preflight"] = stats
    try:
        receipt["environment"] = environment()
        # Since 27/09, `identity` is what a resume must never silently change (model, seed, subset, every data and
        # prompt
        # digest, schedule_sha256, the normalised training config, and the training-code digest). It is
        # saved into every snapshot (train_loop._save_state) and, on --resume, train_loop._load_state compares it
        # against the snapshot it actually picks BEFORE touching anything else in it -- the state is the primary
        # reference (never a receipt: a killed attempt leaves state/ but no receipt, which is exactly the resume
        # path that went unchecked). gpu_name (the training card, receipt.environment()
        # ["device"]["name"]) joins the identity too, computed here (after environment(), which fills it) rather
        # than before, so a resume on another card class is refused with REFUSED_RESUME_IDENTITY like any other
        # identity field -- card class is chosen per seed and must never silently change mid-training.
        identity = {"model": receipt["model"], "seed": receipt["seed"], "subset": receipt["subset"],
                   "data": receipt["data"], "train_config": receipt["train_config"],
                   "train_code_sha256": receipt["train_code_sha256"],
                   "gpu_name": (receipt["environment"].get("device") or {}).get("name")}
        if problems:
            return finish("REFUSED_PREFLIGHT", 2, problems=problems[:50], n_problems=len(problems))

        from .modeling import LoadCfg, flash_probe, load_for_training, model_device

        model, info = load_for_training(LoadCfg(a.model_path or spec.hf_id, None if a.model_path else spec.revision,
                                                a.quant, a.attn, a.dtype, not a.no_gradient_checkpointing,
                                                a.upcast_fp32), a.seed)
        device = model_device(model)
        info["flash_kernel_eligible"] = flash_probe(model, device)
        receipt["model_load"] = info
        reason = check_attention(info)
        if reason:
            return finish("REFUSED_ATTENTION_FALLBACK", 3, reason=reason)
        if info["flash_kernel_eligible"] is False and not a.upcast_fp32:
            return finish("REFUSED_FLASH_INELIGIBLE", 3, reason="a forward with SDPA restricted to FLASH_ATTENTION failed")
        train_use = {k: e for k, e in train_ex.items() if k not in excluded}
        dev_encs = {}
        for (rid, fmt), e in sorted(dev_ex.items()):
            if (rid, fmt) not in excluded:
                dev_encs.setdefault(fmt, []).append(encode(tok, e.messages, e.answer, a.max_seq))
        res = run(model, tok, cfg, ids, a.seed, train_use, dev_encs, a.workdir, device, identity, resume=a.resume)
    except IdentityMismatch as e:
        # Raised by train_loop._load_state before any state file is touched, so
        # state/ stays exactly as it was; mismatches names every differing field, or says the identity is missing.
        return finish("REFUSED_RESUME_IDENTITY", 2,
                     mismatches=e.fields if e.fields is not None else ["no identity.json in the state"])
    except Exception as e:  # noqa: BLE001 - the receipt must record any crash, before or during training
        finish("CRASHED", 1, error=repr(e))
        raise
    receipt["result"] = res
    if res["status"] == "DONE":
        sel = a.workdir / f"epoch-{res['selected_epoch']}" / "adapter_model.safetensors"
        receipt["selected_adapter"] = {"path": str(sel.relative_to(a.workdir)), "tensor_sha256": tensor_digest(sel),
                                       "file_sha256": sha256_file(sel)}
        return finish("DONE", 0)
    return finish(res["status"], 4)


if __name__ == "__main__":
    raise SystemExit(main())
