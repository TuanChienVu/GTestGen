"""Laptop audit of prebuilt prompts before they go to a pod; prompt files have 3A's names,
{model}.{pool}.{part}.{fmt}.jsonl:

python -m v2gpu.leakcheck --model deepseek --subset all --prompts P/deepseek.all.test.F*.jsonl
    P/deepseek.all.aux_test.F*.jsonl --prompts-sums P/PROMPTS_SHA256SUMS
    --records out/phase2/records/test.jsonl out/phase2/records/aux_test.jsonl
    --train-records out/phase2/records/train.jsonl --catalog out/phase2/catalog.jsonl
    --forbidden out/phase2/forbidden.json --phase2-sums out/phase2/PHASE2_SHA256SUMS
    --out receipts/leakcheck-receipt.deepseek.json
    (--out outside P: every file of P must be listed in its sums; a dev directory: --prompts P/deepseek.all.dev.F*.jsonl
    and its train files P/deepseek.all.train.F*.jsonl, --records out/phase2/records/dev.jsonl)

It runs every rule train_job's preflight runs on train and dev prompts, with the query's own gold from the records:
SUMS (every file of the directory, META, excluded.jsonl, infra.json, pools.json), META against the seal, the control
tokens and the pinned tokenizers, the recomputed pool against pools.json, the Phase 2 inputs against the seal, the
strict reader (a test or aux_test line with an answer is refused), coverage (a missing pair must be in excluded.jsonl),
no train neighbour for any held-out query, structure, model, partition and retrieval pool, leakcheck with the
partition's hidden set, and control tokens with the model's tokenizer. It writes a receipt (status PASS or FAIL, the
sha256 of every input, the counts); the generation job checks the sha256 of this receipt. Exit 0 on PASS, 2 on
FAIL.
Scope per check (fixed 25/09), written in the receipt (`check_scope`, and per file in `files`):
- --records holds the HELD-OUT query records only (dev, test, aux_test); a train record in --records is refused at the
  start (exit 2, no receipt). A train prompt line takes its query record from --train-records.
- CONTENT checks run on EVERY prompt line of every partition, train included: `content` (structure, model, partition,
  retrieval pool, leakcheck.check with the partition's hidden set), `forbidden_code` (leakcheck.hidden_code: no catalog
  code of a function hidden for the partition, i.e. a forbidden.json answer of a later partition, in the text) and
  `control_tokens` (the model's tokenizer).
- HELD-OUT query checks run only for dev, test and aux_test queries: `held_out_query` (no train neighbour, every prompt
  line is a query record's, and coverage against excluded.jsonl). Train queries are linked to train records by
  construction; their `neighbours` field must still equal the direct links among train records.
- `files`: per prompt file its sha256, its partitions, the number of prompt lines, the problems found on its lines and
  the number of lines each check ran on; a train file that passed shows content = forbidden_code = control_tokens =
  prompts and 0 problems.
--out is deleted before any other work, so an audit that crashes never leaves an older PASS behind;
the receipt names the code that decided it (`code_sha256`: {repo-relative path: sha256} of AUDIT_CODE, every v2gpu
module the audit loads), and a generation job refuses a receipt made by other code than its own checkout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from .examples import (audit_prompt, control_strings, coverage, pool_digest, pool_problems, prompt_files, read_excluded,
                       jsonl_lines, read_infra, read_jsonl, read_pools, read_records, retrieval_pool, subset_ids)
from .inputs import meta_problems, phase2_problems
from .leakcheck import Context, hidden_code, hidden_for, neighbour_problems, neighbours
from .models import MODELS, load_tokenizer
from .receipt import sha256_file
from .schedule import FORMATS
from .serialize import prompt_ids

SCHEMA = "gtestgen.v2.leakcheck_receipt.v1"
CHECKS = ("content", "forbidden_code", "control_tokens", "held_out_query")
CHECK_SCOPE = {
    "content": "every prompt line: structure, model, partition, retrieval pool, leakcheck.check with the partition's "
               "hidden set",
    "forbidden_code": "every prompt line: no catalog code of a function hidden for its partition (forbidden.json) in "
                      "the text",
    "control_tokens": "every prompt line: the model's tokenizer accepts the messages (no control token)",
    "held_out_query": "dev, test and aux_test query lines only: no train neighbour, a query record of --records, "
                      "coverage against excluded.jsonl"}
AUDIT_CODE = ("v2gpu/audit.py", "v2gpu/examples.py", "v2gpu/inputs.py", "v2gpu/leakcheck.py", "v2gpu/models.py",
              "v2gpu/receipt.py", "v2gpu/schedule.py", "v2gpu/serialize.py")
ROOT = Path(__file__).resolve().parents[1]


def code_sha256(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative path: sha256} of every module whose code decides the audit's PASS, in this checkout."""
    return {rel: sha256_file(Path(root) / rel) for rel in AUDIT_CODE}


def _catalog(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for line in jsonl_lines(path):
        fn = json.loads(line)
        out.setdefault(fn["repo"], {})[fn["fn_id"]] = fn
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m v2gpu.leakcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--subset", default="all", choices=["all", "lc600", "lc1200"], help="the retrieval pool")
    ap.add_argument("--prompts", type=Path, nargs="+", required=True)
    ap.add_argument("--prompts-sums", type=Path, required=True)
    ap.add_argument("--records", type=Path, nargs="+", required=True,
                    help="the held-out query records (dev, test, aux_test), never train")
    ap.add_argument("--train-records", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True, help="the sealed catalog, with every function's code")
    ap.add_argument("--forbidden", type=Path, required=True)
    ap.add_argument("--phase2-sums", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    inputs = [*a.prompts, a.prompts_sums, *a.records, a.train_records, a.catalog, a.forbidden, a.phase2_sums]
    out = a.out.resolve()
    if out in {p.resolve() for p in inputs} or out.is_relative_to(a.prompts_sums.resolve().parent):
        ap.error(f"--out {a.out} is an input or lies in the prompt directory of --prompts-sums: the audit deletes "
                 f"--out first")
    a.out.unlink(missing_ok=True)       # before any other work: a crash must never leave an older PASS behind
    queries = [r for path in a.records for r in read_records(path)]
    in_records = sorted(r["id"] for r in queries if r.get("partition") == "train")
    if in_records:                      # refused at the start, no receipt
        print(f"REFUSED: --records holds {len(in_records)} train record(s) ({', '.join(in_records[:5])}): --records "
              f"takes the held-out query records only (dev, test, aux_test); a train prompt line takes its query "
              f"record from --train-records", file=sys.stderr)
        return 2

    problems, prompts_meta = prompt_files(a.prompts, a.prompts_sums)
    excluded_problems, listed = read_excluded(a.prompts_sums, a.model)
    infra_problems, infra = read_infra(a.prompts_sums)
    pools_problems, pools = read_pools(a.prompts_sums)
    problems += excluded_problems + infra_problems + pools_problems + meta_problems(prompts_meta["meta"], a.phase2_sums)
    problems += phase2_problems([*a.records, a.train_records, a.catalog, a.forbidden], a.phase2_sums)
    train = read_records(a.train_records)
    problems += neighbour_problems(train, queries)
    catalog = _catalog(a.catalog)
    ids = retrieval_pool(train, subset_ids(train, a.subset), listed, control_strings(), catalog)
    problems += pool_problems(pools, a.subset, ids)
    ctx = Context(pool={r["id"]: r for r in train if r["id"] in set(ids)}, catalog=catalog, hidden={},
                  language={r["repo"]: r["language"] for r in train + queries}, neighbours=neighbours(train + queries),
                  infra=infra)
    forbidden = json.loads(a.forbidden.read_text())
    report, n = Counter(), 0
    by_part, excluded, dropped = Counter(), {}, {}
    prompts, origin = {}, {}
    files = {Path(p).name: {"sha256": sha256_file(p), "partitions": set(), "prompts": 0, "problems": 0,
                            "checks": dict.fromkeys(CHECKS, 0)} for p in a.prompts}
    for path in map(Path, a.prompts):
        try:
            rows = read_jsonl(path)
        except ValueError as e:
            problems.append(str(e))
            files[path.name]["problems"] += 1
            continue
        for k, ex in rows.items():
            if k in prompts:
                problems.append(f"duplicate example {k} (in {origin[k]} and {path.name})")
                continue
            prompts[k], origin[k] = ex, path.name
    by_id, train_by_id = {r["id"]: r for r in queries}, {r["id"]: r for r in train}
    tok = load_tokenizer(MODELS[a.model]) if prompts else None
    for part in sorted({r["partition"] for r in queries}):
        wanted = [(r["id"], f) for r in queries if r["partition"] == part for f in FORMATS]
        found, excluded[part] = coverage(part, wanted, prompts, listed)
        problems += found
        dropped[part] = Counter()
    for (rid, fmt), ex in sorted(prompts.items()):
        file = files[origin[(rid, fmt)]]
        file["prompts"] += 1
        file["partitions"].add(ex.partition)
        held_out = ex.partition != "train"
        record = (by_id if held_out else train_by_id).get(rid)
        if record is None:
            problems.append(f"prompt {rid} {fmt} is not a query record" if held_out else
                            f"prompt {rid} {fmt} ({origin[(rid, fmt)]}) is not a train record of --train-records")
            file["problems"] += 1
            continue
        n += 1
        part, found, checks = record["partition"], [], file["checks"]
        by_part[part] += 1
        try:
            found += audit_prompt(record, ex, part, a.model, pool_digest(ids), ctx, forbidden, report,
                                  dropped.setdefault(part, Counter()))
            checks["content"] += 1
            hidden = hidden_for(part, forbidden.get(record["repo"], {}))
            found += [f"{rid} {fmt}: code of {fid}, hidden for partition {part} (forbidden.json), appears in the prompt"
                      for fid in hidden_code(record, ex.messages, ctx, hidden)]
            checks["forbidden_code"] += 1
            prompt_ids(tok, ex.messages)
            checks["control_tokens"] += 1
        except Exception as e:  # noqa: BLE001 - control tokens, a missing field, messages the template rejects
            found.append(f"{rid} {fmt}: {type(e).__name__}: {e}")
        checks["held_out_query"] += held_out
        problems += found
        file["problems"] += len(found)
    receipt = {"schema": SCHEMA, "status": "FAIL" if problems else "PASS", "model": a.model, "subset": a.subset,
               "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "inputs": {str(p): sha256_file(p) for p in inputs}, "code_sha256": code_sha256(),
               "prompts": prompts_meta, "n_checked": n, "check_scope": CHECK_SCOPE,
               "files": {k: {**v, "partitions": sorted(v["partitions"])} for k, v in sorted(files.items())},
               "by_partition": dict(sorted(by_part.items())),
               "excluded_by_compose": {p: dict(sorted(c.items())) for p, c in excluded.items() if c},
               "dropped_by_compose": {p: dict(sorted(c.items())) for p, c in dropped.items() if c},
               "report_only": {k: v for k, v in sorted(report.items()) if v}, "n_problems": len(problems),
               "problems": problems[:200]}
    a.out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("status", "n_checked", "n_problems")}))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
