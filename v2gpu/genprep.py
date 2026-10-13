"""Preflight of a generation job: every check made before an engine loads. It needs no GPU, and no records,
catalog or forbidden.json: a generation pod never receives them.

- Prompt files: each file and PROMPTS_META.json against PROMPTS_SHA256SUMS (examples.prompt_files: a file of the
  directory the sums do not list is refused too), META keys and pinned tokenizer revisions (inputs.meta_problems; the
  Phase 2 seal itself was checked by the laptop audit), excluded.jsonl and pools.json (listed in the sums).
- The reader gets the EXPECTED partitions (examples.read_jsonl(partitions=...)): a line of any other partition is
  refused, so a test job cannot be handed a line mislabelled "train", which could carry an answer; a test or aux_test
  line with an answer is refused by the reader in any case. The job never reads `answer`.
- Dev gold: dev prompt files carry `answer` (training needs it for the dev loss), so a dev generation pod
  holds dev gold; the job never reads it. Test and aux_test files never carry gold. Dev is tuning only and already sits
  on training pods.
- Every line: one user turn and the four provenance keys (examples.structure_problems), built for the job's model, from
  the job's retrieval pool (the sha256 of pools.json[subset]).
- Prompt files: every expected partition has at least one prompt line, and every prompt file the sums
  list that is present beside them and holds lines of this model, an expected partition and a needed format is among
  --prompts (so "exactly the files the sums list for (model, partition, formats)" holds on the laptop, where every
  file is present; a pod receives only the job's files).
- Pairs: the records of a partition are the record ids of its prompt lines. For every run of the job
  (runs.job_runs: the planned conditions of these weights) each (record, format) has a prompt unless excluded.jsonl
  lists it (examples.coverage; excluded pairs are counted by reason, and an excluded pair must not have a prompt).
- Leak-audit receipt (`python -m v2gpu.leakcheck`, laptop): its sha256 is the one in the job command (QUEUE.json, the
  job's input manifest); schema gtestgen.v2.leakcheck_receipt.v1, status PASS with 0 problems, the job's model and pool,
  the same PROMPTS_SHA256SUMS, every partition the job reads, and every prompt file the job reads (same file name, same
  sha256) among its inputs; and `code_sha256` equal to audit.code_sha256() of THIS checkout, so a receipt
  made by other leak rules (e.g. one made before 24/09/2026, when short answer slices became a refusal) never passes.
- Adapter runs: engines.adapter_problems (the tensor digest the train receipt selected; a config vLLM serves as
  trained).
- Compose: the dev labels all read dev prompts with one run matrix, so a
  dev grid label (either round) is bound to its prompt directory: PROMPTS_META.json compose must have k == k,
  n_examples == n and round(100 * ratio) == r (compose_problems). For every variant the compose block goes into the
  job identity, and gencoverage requires one per variant. Following the 3A --variant
  convention of GEN_RUNSHEET.md, every real variant needs META compose; dev_pilot_vN reads the directory 3A built with
  --variant vN, a dev grid label the one built with --variant <that label> (so a round-1 label never reads a round-0 directory),
  both in mode full; adapter_compose_problems: for dev_pilot_vN and main the adapter's train-receipt compose
  (data.prompts.meta.compose) equals META compose, every key, and a dev grid label runs its round's adapter
  (dev_grid_*: v0, dev_grid_v1_*: round 1) — genqueue and gencoverage check it; META files is exactly the list of prompt
  files PROMPTS_SHA256SUMS holds (meta_files_problems).
  Since 27/09/2026: a dev grid label's own directory legitimately varies k, n_examples and ratio away from its
  round's pilot, so comparing the adapter's train compose against the GRID's own META compose let a stale adapter
  through (only compose["variant"] was checked). Now the caller gives the round's OWN dev_pilot_vN META compose
  separately (dev_pilot_compose, --dev-pilot-meta in genqueue/gencoverage), and the adapter's train compose must
  equal it on EVERY key; no --dev-pilot-meta is a refusal, not a silent pass.
- A prompt file the reader cannot decode as UTF-8 is a problem naming it.
- tokenize: serialize.prompt_ids and serialize.max_new_tokens (min(6,144, window - prompt)) for every pair; a prompt
  that leaves no room is a problem; `ids_sha256` digests every prompt's ids, so a pod's tokenisation can be compared
  with the training venv's.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .examples import (Example, _sums, coverage, jsonl_lines, prompt_files, read_excluded, read_jsonl, read_pools,
                       structure_problems)
from .inputs import meta_problems
from .receipt import sha256_file
from .runs import DEV_GRID, Run, job_runs
from .serialize import AnswerTooLong, max_new_tokens, prompt_ids

LEAKCHECK_SCHEMA = "gtestgen.v2.leakcheck_receipt.v1"


@dataclass(frozen=True)
class Pair:
    run: Run
    record_id: str
    partition: str
    example: Example
    prompts_sha256: str            # sha256 of the prompt FILE the line came from


@dataclass
class Prepared:
    pairs: list[Pair] = field(default_factory=list)            # run order (CONDITIONS), then record id
    excluded: dict = field(default_factory=dict)               # {partition: {reason: pairs}} for this job's runs
    prompts_meta: dict = field(default_factory=dict)           # prompt_files: {"sums_sha256", "meta"}
    files: dict = field(default_factory=dict)                  # {prompt file name: sha256}
    receipt_sha256: str | None = None
    pool_sha256: str | None = None


def receipt_problems(path, expected_sha256: str, model: str, subset: str, files: dict[str, str], sums_sha256: str,
                     partitions) -> list[str]:
    path = Path(path)
    if not path.exists():
        return [f"leak-audit receipt {path} is missing"]
    got = sha256_file(path)
    if got != expected_sha256:
        return [f"leak-audit receipt sha256 {got} is not the job's {expected_sha256}"]
    try:
        r = json.loads(path.read_text())
    except ValueError as e:
        return [f"leak-audit receipt is not JSON ({e})"]
    if not isinstance(r, dict):
        return [f"leak-audit receipt is not a JSON object ({type(r).__name__})"]
    out = [f"leak-audit receipt field {k!r} is not an object" for k in ("prompts", "inputs")
           if not isinstance(r.get(k) or {}, dict)]
    if out:
        return out
    if r.get("schema") != LEAKCHECK_SCHEMA:
        out.append(f"leak-audit receipt schema {r.get('schema')!r} is not {LEAKCHECK_SCHEMA}")
    if r.get("status") != "PASS" or r.get("n_problems") != 0:
        out.append(f"leak-audit receipt status {r.get('status')!r} with {r.get('n_problems')} problem(s): PASS needed")
    if r.get("model") != model:
        out.append(f"leak-audit receipt is for model {r.get('model')!r}, the job's is {model!r}")
    if r.get("subset") != subset:
        out.append(f"leak-audit receipt checked pool {r.get('subset')!r}, the job reads pool {subset!r}")
    if (r.get("prompts") or {}).get("sums_sha256") != sums_sha256:
        out.append("leak-audit receipt audited another PROMPTS_SHA256SUMS")
    audited = {(Path(k).name, v) for k, v in (r.get("inputs") or {}).items()}
    missing = sorted(name for name, digest in files.items() if (name, digest) not in audited)
    if missing:
        out.append(f"prompt files not audited by the receipt: {missing}")
    parts = sorted(r.get("by_partition") or {})
    if set(partitions) - set(parts):
        out.append(f"leak-audit receipt audited partitions {parts}, the job reads {sorted(partitions)}")
    from .audit import code_sha256

    code, here = r.get("code_sha256"), code_sha256()
    if not isinstance(code, dict):
        out.append("leak-audit receipt has no code_sha256 (an audit made before receipts named their code): "
                   "rerun the laptop audit")
    else:
        differ = sorted((set(code) ^ set(here)) | {k for k in set(code) & set(here) if code[k] != here[k]})
        if differ:
            out.append(f"leak-audit receipt was made by other code than this checkout: {differ}")
    return out


DEV_PILOT_LABEL = re.compile(r"dev_pilot_(v\d+)")


def compose_problems(variant: str, meta) -> list[str]:
    """The run label is bound to the 3A directory it reads: every
    real variant needs PROMPTS_META.json compose; dev_pilot_vN reads the directory built with --variant vN, a dev grid
    label the one built with --variant <that label> and its k, n_examples and round(100 * ratio); both in mode full."""
    if variant == "gen_canary":
        return []
    c = meta.get("compose") if isinstance(meta, dict) else None
    if not isinstance(c, dict) or not c:
        return [f"variant {variant} needs PROMPTS_META.json compose (the 3A build parameters); there is none"]
    out = []
    pilot = DEV_PILOT_LABEL.fullmatch(variant) if isinstance(variant, str) else None
    grid = DEV_GRID.fullmatch(variant) if isinstance(variant, str) else None
    if pilot and c.get("variant") != pilot.group(1):
        out.append(f"variant {variant} reads the 3A directory built with --variant {pilot.group(1)}; PROMPTS_META.json "
                   f"compose variant is {c.get('variant')!r}")
    if grid:
        k, n, r = map(int, grid.groups())
        ok = type(c.get("k")) is int and type(c.get("n_examples")) is int and \
            type(c.get("ratio")) in (int, float) and (c["k"], c["n_examples"], round(100 * c["ratio"])) == (k, n, r)
        if not ok:
            out.append(f"variant {variant} needs PROMPTS_META.json compose k {k}, n_examples {n}, ratio {r / 100}; "
                       f"PROMPTS_META.json compose has k {c.get('k')!r}, n_examples {c.get('n_examples')!r}, ratio "
                       f"{c.get('ratio')!r}")
        if c.get("variant") != variant:
            out.append(f"variant {variant} reads the 3A directory built with --variant {variant}; PROMPTS_META.json "
                       f"compose variant is {c.get('variant')!r}")
    if (pilot or grid) and c.get("mode") != "full":
        out.append(f"variant {variant} runs compose mode full; PROMPTS_META.json compose mode is {c.get('mode')!r}")
    return out


# A dev grid's own PROMPTS_META.json compose legitimately varies these keys away
# from its round's pilot compose (k/n_examples/ratio per grid point; the two ratio-derived token budgets, since
# ratio splits their shared source budget between them, F1's Params.ratio). Every other key must agree with the
# pilot's, keeping the grid's own prompts bound to the same round as its adapter.
GRID_META_EXEMPT_KEYS = frozenset({"variant", "k", "n_examples", "ratio", "budget_examples", "budget_knowledge"})


def adapter_compose_problems(variant: str, meta, train_compose, pilot_compose: dict | None = None) -> list[str]:
    """The adapter was trained on the compose the run label needs (rule fixed 25/09/2026, its dev grid part
    27/09/2026): dev_pilot_vN and main run an adapter whose train receipt's data.prompts.meta.compose equals the
    generation META compose, every key.

    A dev grid label's own directory legitimately varies k, n_examples and ratio away from its round's pilot
    (compose_problems already checks those three against the label), so a grid label's ADAPTER cannot be checked
    against its own meta here: `pilot_compose`, the round's OWN dev_pilot_vN META compose, must be given separately
    by the caller (genqueue/gencoverage read it from --dev-pilot-meta, via dev_pilot_compose). For a DEV_GRID variant
    three checks all apply (an earlier version dropped the round-binding ones, so a v0
    adapter given the v0 pilot META passed under a dev_grid_v1_* label -- pilot_compose == train_compose there, so
    plain equality alone finds nothing wrong):
      - pilot_compose's own variant is `want` (round 1 for dev_grid_v1_*, else v0): the CALLER gave the right round's
        pilot META, not some other round's;
      - train_compose's own variant is `want`: the adapter itself is a dev_pilot_{want}-trained one;
      - train_compose equals pilot_compose on EVERY key, no exemptions (the old rule only checked
        compose["variant"], so an adapter trained at the round's stale ratio, before a dev_pilot_vN rebuild, kept
        passing every dev_grid_vN_* label of that round).
    A missing pilot_compose is a refusal, never a silent pass. Also: the grid's OWN PROMPTS_META.json compose must
    equal pilot_compose outside GRID_META_EXEMPT_KEYS -- a
    grid built in a different compose mode than its round's pilot is a defect the adapter-vs-pilot checks above
    cannot see on their own, since the adapter can match pilot perfectly while the grid's own prompts do not."""
    if variant == "gen_canary":
        return []
    if not isinstance(train_compose, dict) or not train_compose:
        return ["the adapter's train receipt names no compose (data.prompts.meta.compose)"]
    if DEV_GRID.fullmatch(variant):
        want = "v1" if variant.startswith("dev_grid_v1_") else "v0"
        if not isinstance(pilot_compose, dict) or not pilot_compose:
            return [f"variant {variant} needs the round's dev_pilot_{want} META compose to check the adapter; "
                    f"none was given"]
        out = []
        if pilot_compose.get("variant") != want:
            out.append(f"the given --dev-pilot-meta compose variant is {pilot_compose.get('variant')!r}, not "
                       f"{want!r}: the wrong round's dev_pilot META was given for {variant}")
        if train_compose.get("variant") != want:
            out.append(f"the adapter's train compose variant is {train_compose.get('variant')!r}, not {want!r}: "
                       f"this is not a dev_pilot_{want}-trained adapter")
        differ = sorted(k for k in set(pilot_compose) | set(train_compose)
                        if pilot_compose.get(k) != train_compose.get(k))
        if differ:
            out.append(f"the adapter was trained on another compose than the round's dev_pilot_{want} (differs in "
                       f"{', '.join(differ)})")
        c = meta.get("compose") if isinstance(meta, dict) and isinstance(meta.get("compose"), dict) else {}
        grid_differ = sorted(k for k in (set(pilot_compose) | set(c)) - GRID_META_EXEMPT_KEYS
                             if pilot_compose.get(k) != c.get(k))
        if grid_differ:
            out.append(f"variant {variant}'s own PROMPTS_META.json compose differs from the round's "
                       f"dev_pilot_{want} outside {{{', '.join(sorted(GRID_META_EXEMPT_KEYS))}}} (differs in "
                       f"{', '.join(grid_differ)})")
        return out
    c = meta.get("compose") if isinstance(meta, dict) and isinstance(meta.get("compose"), dict) else {}
    differ = sorted(k for k in set(c) | set(train_compose) if c.get(k) != train_compose.get(k))
    return [] if not differ else [f"the adapter was trained on another compose than PROMPTS_META.json's (differs in "
                                  f"{', '.join(differ)})"]


def dev_pilot_compose(path) -> dict | None:
    """The round's OWN dev_pilot_vN META compose, read straight from its PROMPTS_META.json
    (None if unreadable, not a JSON object, or without a compose block). adapter_compose_problems needs this as a
    separate reference for a dev_grid_* variant, since the grid's own directory legitimately varies k, n_examples
    and ratio away from it."""
    try:
        meta = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    compose = meta.get("compose") if isinstance(meta, dict) else None
    return compose if isinstance(compose, dict) and compose else None


def meta_files_problems(meta, sums) -> list[str]:
    """PROMPTS_META.json files is the list of prompt files PROMPTS_SHA256SUMS holds."""
    listed = sorted(n for n in _sums(sums) if n.endswith(".jsonl") and n != "excluded.jsonl")
    files = meta.get("files") if isinstance(meta, dict) else None
    if isinstance(files, list) and sorted(files) == listed:
        return []
    return [f"PROMPTS_META.json files {files!r} are not the prompt files {Path(sums).name} lists {listed}"]


def listed_prompt_problems(prompts, sums, model: str, partitions, formats) -> list[str]:
    """A prompt file the sums list, present beside them, holding lines of this job, must be among --prompts."""
    sums = Path(sums)
    given, base, out = {Path(p).resolve() for p in prompts}, sums.resolve().parent, []
    for name in sorted(_sums(sums)):
        path = base / name
        if not name.endswith(".jsonl") or name == "excluded.jsonl" or path.resolve() in given or not path.is_file():
            continue
        try:
            lines = jsonl_lines(path)
        except ValueError as e:                  # not UTF-8, so its lines cannot be told apart
            out.append(f"{name}: listed in {sums.name}, {e}")
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                break
            if isinstance(row, dict) and row.get("model") == model and row.get("partition") in partitions and \
                    row.get("fmt") in formats:
                out.append(f"{name}: listed in {sums.name} with {model} {row['partition']} {row['fmt']} prompts, and "
                           f"not among --prompts")
                break
    return out


def preflight(model: str, weights: str, variant: str, partitions, prompts, sums, subset: str, receipt,
              receipt_sha256: str, adapter=None, adapter_sha256: str | None = None,
              adapter_config_sha256: str | None = None) -> tuple[list[str], Prepared]:
    from .engines import adapter_problems

    prep = Prepared()
    problems, prep.prompts_meta = prompt_files(prompts, sums)
    if not partitions:
        problems.append("no partitions: a job reads at least one partition")
    problems += meta_problems(prep.prompts_meta["meta"], None)
    problems += compose_problems(variant, prep.prompts_meta["meta"]) + meta_files_problems(prep.prompts_meta["meta"],
                                                                                              sums)
    excluded_problems, listed = read_excluded(sums, model)
    pools_problems, pools = read_pools(sums)
    problems += excluded_problems + pools_problems
    prep.pool_sha256 = pools.get(subset, {}).get("sha256")
    if prep.pool_sha256 is None:
        problems.append(f"pools.json has no pool {subset!r}")
    try:
        runs = job_runs(variant, partitions, model, weights)
    except ValueError as e:
        problems.append(str(e))
        runs = []
    if weights == "base" and (adapter is not None or adapter_sha256 is not None):
        problems.append("base weights take no adapter (--adapter/--adapter-sha256 given)")
    if weights != "base":
        if adapter is None or adapter_sha256 is None:
            problems.append(f"weights {weights}: --adapter and --adapter-sha256 (train receipt) are required")
        else:
            problems += adapter_problems(adapter, adapter_sha256, adapter_config_sha256)
    examples, source = {}, {}
    for path in map(Path, prompts):
        prep.files[path.name] = sha256_file(path)
        try:
            rows = read_jsonl(path, partitions=tuple(partitions))
        except ValueError as e:
            problems.append(str(e))
            continue
        for key, ex in rows.items():
            if key in examples:
                problems.append(f"{key[0]} {key[1]}: in {source[key]} and {path.name}")
            examples[key], source[key] = ex, path.name
    for (rid, fmt), ex in sorted(examples.items()):
        problems += [f"{rid} {fmt}: {p}" for p in structure_problems(ex)]
        if ex.model != model:
            problems.append(f"{rid} {fmt}: built for model {ex.model!r}, job model {model!r}")
        if ex.pool_sha256 != prep.pool_sha256:
            problems.append(f"{rid} {fmt}: retrieval pool is not pools.json[{subset!r}]")
    problems += receipt_problems(receipt, receipt_sha256, model, subset, prep.files,
                                 prep.prompts_meta["sums_sha256"], partitions)
    prep.receipt_sha256 = receipt_sha256
    records = {}
    for (rid, _), ex in examples.items():
        records[rid] = ex.partition
    problems += [f"no {part} prompt among --prompts (an expected partition)" for part in partitions
                 if part not in set(records.values())]
    problems += listed_prompt_problems(prompts, sums, model, partitions, {run.fmt for run in runs})
    for part in sorted(set(records.values())):
        ids = sorted(r for r, p in records.items() if p == part)
        have = {k: e for k, e in examples.items() if e.partition == part}
        counts = Counter()
        for run in runs:
            found, c = coverage(part, [(r, run.fmt) for r in ids], have, listed)
            problems += [p for p in found if p not in problems]
            counts += c
        if counts:
            prep.excluded[part] = dict(sorted(counts.items()))
    for run in runs:
        prep.pairs += [Pair(run, rid, records[rid], examples[(rid, run.fmt)], prep.files[source[(rid, run.fmt)]])
                       for rid in sorted(records) if (rid, run.fmt) in examples and (rid, run.fmt) not in listed]
    return problems, prep


def tokenize(tok, pairs: list[Pair], window: int) -> tuple[list[str], dict, str]:
    """{(record_id, fmt): (prompt ids, max_new_tokens)} for every pair, problems, and a digest of every prompt's ids."""
    problems, out = [], {}
    for p in pairs:
        key = (p.record_id, p.run.fmt)
        if key in out:
            continue
        try:
            ids = prompt_ids(tok, p.example.messages)
            out[key] = (ids, max_new_tokens(len(ids), window))
        except (AnswerTooLong, ValueError) as e:
            problems.append(f"{p.record_id} {p.run.fmt}: {e}")
    h = hashlib.sha256()
    for (rid, fmt), (ids, _) in sorted(out.items()):
        h.update(f"{rid}|{fmt}|{' '.join(map(str, ids))}\n".encode())
    return problems, out, h.hexdigest()
