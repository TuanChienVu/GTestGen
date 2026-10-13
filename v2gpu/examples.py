"""Prompt files: one (messages, answer, provenance) per record and prompt format, for ONE model and ONE retrieval pool.

Contract with the prompt builder (v2tools):
- `compose_prompt(record, fmt, tok, pool) -> (messages, provenance)` and `gold_answer(record, catalog=None) -> str`
  (compose raises without a catalog when the record has a preexisting item), two separate
  functions; test and aux_test prompts carry answer = None, because gold_answer is never called for them and gold never
  goes to a generation pod; the reader refuses a test or aux_test line whose answer is not null;
- messages: exactly one turn, role "user"; provenance: exactly four keys, each a list: "examples" (record ids of ③),
  "knowledge" (fn ids of ④(b)), "skeleton" (fn ids whose signature ④(a) shows, inherited ones included), "dropped"
  ({kind, id, rule} per item compose removed); F0 has empty examples, knowledge and skeleton, F1 empty knowledge and
  skeleton;
- one JSONL file per model x pool x partition x format, named {model}.{pool}.{part}.{fmt}.jsonl by 3A (for example
  deepseek.all.dev.F0.jsonl; no reader parses the name), prebuilt on the laptop with a pinned v2tools commit; the
  directory holds
  PROMPTS_SHA256SUMS (shasum format), PROMPTS_META.json ({"v2tools_commit", "phase2_sums_sha256",
  "tokenizer_revisions": {model: revision}, "compose": {...}}) and excluded.jsonl ({record_id, fmt|null, model|null,
  reason} per line: pairs compose left out, e.g. control-token records, the test/aux 16,384 rule, or "uses_test_infra")
  and infra.json ({repo: [[path, name], ...]}: members of test-answer infrastructure, counted by preflight) and
  pools.json ({pool name: {"sha256", "ids" (sorted)}}); all are listed in the sums, and so is every other file of the
  directory (a file the sums do not list is refused). Every directory's excluded.jsonl lists ALL uses_test_infra
  records, whatever partitions it holds. GPU code only reads these files and checks them against the sums; a missing
  (record, format) is accepted only when excluded.jsonl lists it; a blank or broken line is a refusal, not a crash;
- the retrieval pool (decision of 24/09; compose's v2tools.prompts.clean_pool) is ONE pool for every model: the subset
  (all, lc600, lc1200) minus the records compose drops as "uses_test_infra" minus the records holding a control token
  of ANY model (config/special_tokens.json, leading ▁ removed) in their Background, Scenario, steps or gold;
  pool_sha256 = sha256 of the sorted ids joined by newlines. Preflight recomputes it and compares with pools.json.
Every example carries the model it was built for (budgets are counted with that model's tokenizer) and the sha256 of its
retrieval pool; for lc600/lc1200 the pool is the subset itself, at training and at inference.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

from .inputs import SPECIAL_TOKENS
from .schedule import FORMATS

ComposePrompt = Callable[[dict, str, object, list], tuple]
GoldAnswer = Callable[[dict], str]
LABELLED = ("train", "dev")
UNLABELLED = ("test", "aux_test")
PROVENANCE_KEYS = ("examples", "knowledge", "skeleton", "dropped")
EMPTY_BY_FORMAT = {"F0": ("examples", "knowledge", "skeleton"), "F1": ("knowledge", "skeleton"), "F2": ()}


@dataclass(frozen=True)
class Example:
    record_id: str
    fmt: str
    model: str
    partition: str
    pool_sha256: str
    messages: list[dict]
    answer: str | None
    provenance: dict


def pool_digest(record_ids) -> str:
    return hashlib.sha256("\n".join(sorted(record_ids)).encode()).hexdigest()


def compose_prompt_pending(record, fmt, tok, pool):
    raise NotImplementedError("v2tools.compose.compose_prompt chưa có: chờ kế hoạch 3A")


def gold_answer_pending(record):
    raise NotImplementedError("v2tools.compose.gold_answer chưa có: chờ kế hoạch 3A")


def build(records: list[dict], model: str, tok, pool: list[dict], compose_prompt: ComposePrompt,
          gold_answer: GoldAnswer, formats: tuple[str, ...] = FORMATS) -> list[Example]:
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate record ids")
    digest = pool_digest(r["id"] for r in pool)
    out = []
    for r in records:
        answer = gold_answer(r) if r["partition"] in LABELLED else None
        for fmt in formats:
            messages, provenance = compose_prompt(r, fmt, tok, pool)
            out.append(Example(r["id"], fmt, model, r["partition"], digest, messages, answer, provenance))
    return sorted(out, key=lambda e: (e.record_id, e.fmt))


def write_jsonl(path: Path, examples: list[Example]) -> str:
    examples = sorted(examples, key=lambda e: (e.record_id, e.fmt))
    data = "".join(json.dumps(asdict(e), ensure_ascii=False, sort_keys=True) + "\n" for e in examples).encode()
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


FIELDS = frozenset(f.name for f in fields(Example))


def jsonl_lines(path) -> list[str]:
    """The lines of a JSONL file, split on "\\n" only: json.dumps(ensure_ascii=False) writes U+2028, U+2029 and U+0085
    raw, and str.splitlines would cut a line there. A final newline ends the last line. The file is
    decoded as strict UTF-8 first: json.loads(bytes) would accept a BOM, UTF-16/32 and lone surrogates,
    which the A2 readers (read_text(encoding="utf-8")) refused; a BOM stays in the first line, so its parse fails.
    A decode error names the file (a torn multibyte tail of a crashed run is a problem to report)."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path}: not UTF-8 ({e})") from e
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def read_jsonl(paths, partitions=None) -> dict[tuple[str, str], Example]:
    """Every line must have exactly the eight keys; a test or aux_test line must have answer = null. With `partitions`
    (the partitions a job expects), a line of any other partition is refused, so a generation pod reading test
    prompts cannot be handed a line mislabelled "train", which may carry an answer."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    out = {}
    for path in paths:
        for n, line in enumerate(jsonl_lines(path), start=1):
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            if set(row) != FIELDS:
                raise ValueError(f"{path}:{n}: keys {sorted(row)} are not {sorted(FIELDS)}")
            if partitions is not None and row["partition"] not in partitions:
                raise ValueError(f"{path}:{n}: partition {row['partition']!r}, expected one of {sorted(partitions)}")
            if row["partition"] in UNLABELLED and row["answer"] is not None:
                raise ValueError(f"{path}:{n}: a {row['partition']} line carries an answer")
            e = Example(**row)
            if (e.record_id, e.fmt) in out:
                raise ValueError(f"duplicate example {(e.record_id, e.fmt)} (in {path})")
            out[(e.record_id, e.fmt)] = e
    return out


def structure_problems(ex: Example) -> list[str]:
    """One user turn; exactly the four provenance keys, each a list, empty where the format shows nothing."""
    roles = [m.get("role") for m in ex.messages] if isinstance(ex.messages, list) else None
    out = [] if roles == ["user"] else [f"messages must be exactly one user turn, got roles {roles}"]
    p = ex.provenance
    if not isinstance(p, dict) or set(p) != set(PROVENANCE_KEYS):
        return out + [f"provenance keys {sorted(p) if isinstance(p, dict) else type(p).__name__} are not "
                      f"{list(PROVENANCE_KEYS)}"]
    out += [f"provenance.{k} is not a list" for k in PROVENANCE_KEYS if not isinstance(p[k], list)]
    if ex.fmt not in EMPTY_BY_FORMAT:
        return out + [f"unknown format {ex.fmt!r}"]
    return out + [f"{ex.fmt} must have empty provenance.{k}" for k in EMPTY_BY_FORMAT[ex.fmt] if p[k]]


def coverage(part: str, wanted: list[tuple[str, str]], have, listed: dict) -> tuple[list[str], Counter]:
    """Every wanted (record, format) has a prompt unless excluded.jsonl lists it (counted by reason); a listed pair must
    not have a prompt."""
    problems, counts = [], Counter()
    for rid, fmt in wanted:
        if (rid, fmt) in listed:
            if (rid, fmt) in have:
                problems.append(f"{part} {rid} {fmt}: listed in excluded.jsonl but has a prompt")
            else:
                counts[listed[(rid, fmt)]] += 1
        elif (rid, fmt) not in have:
            problems.append(f"missing {part} prompt {rid} {fmt}")
    return problems, counts


def _sums(sums_path: Path) -> dict[str, str]:
    listed = {}
    for line in Path(sums_path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip().lstrip("*")] = digest
    return listed


def _check_listed(path: Path, sums_path: Path, listed: dict[str, str]) -> list[str]:
    base = Path(sums_path).resolve().parent
    rel = path.resolve().relative_to(base).as_posix() if path.resolve().is_relative_to(base) else path.name
    if rel not in listed:
        return [f"{rel}: not listed in {Path(sums_path).name}"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != listed[rel]:
        return [f"{rel}: sha256 differs from {Path(sums_path).name}"]
    return []


def prompt_files(paths, sums_path: Path) -> tuple[list[str], dict]:
    """Check each prompt file and PROMPTS_META.json against PROMPTS_SHA256SUMS; return (problems, {"sums_sha256",
    "meta"[, "ignored"]}) for the receipt. A file of the directory the sums do not list is refused, except macOS Finder
    litter (.DS_Store, ._*), which is ignored and named in "ignored"."""
    sums_path = Path(sums_path)
    listed = _sums(sums_path)
    problems = [p for path in map(Path, paths) for p in _check_listed(path, sums_path, listed)]
    base, given = sums_path.resolve().parent, {Path(p).resolve() for p in paths}
    stray = [f for f in sorted(base.rglob("*")) if f.is_file() and f != sums_path.resolve() and f not in given
             and f.relative_to(base).as_posix() not in listed]
    ignored = [f.relative_to(base).as_posix() for f in stray if f.name == ".DS_Store" or f.name.startswith("._")]
    problems += [f"{f.relative_to(base).as_posix()}: in the prompt directory but not listed in {sums_path.name}"
                 for f in stray if f.relative_to(base).as_posix() not in ignored]
    meta_path = sums_path.parent / "PROMPTS_META.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is None:
        problems.append(f"PROMPTS_META.json missing beside {sums_path.name}")
    else:
        problems += _check_listed(meta_path, sums_path, listed)
    return problems, {"sums_sha256": hashlib.sha256(sums_path.read_bytes()).hexdigest(), "meta": meta,
                      **({"ignored": ignored} if ignored else {})}


def read_excluded(sums_path: Path, model: str) -> tuple[list[str], dict[tuple[str, str], str]]:
    """excluded.jsonl beside the sums (listed in them): {(record_id, fmt): reason} for this model. A line with fmt or
    model null applies to every format or every model (e.g. the test/aux 16,384 rule, control-token records)."""
    sums_path = Path(sums_path)
    path = sums_path.parent / "excluded.jsonl"
    if not path.exists():
        return [f"excluded.jsonl missing beside {sums_path.name}"], {}
    problems, pairs = _check_listed(path, sums_path, _sums(sums_path)), {}
    for n, line in enumerate(jsonl_lines(path), start=1):
        if not line.strip():
            problems.append(f"excluded.jsonl:{n}: blank line")
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("not a JSON object")
            if not isinstance(row.get("record_id"), str) or not isinstance(row.get("reason"), str) or \
                    row.get("fmt") not in (None, *FORMATS):
                raise ValueError(f"needs record_id, reason and fmt in {list(FORMATS)} or null")
        except ValueError as e:
            problems.append(f"excluded.jsonl:{n}: {e}")
            continue
        if row.get("model") not in (None, model):
            continue
        for fmt in ([row["fmt"]] if row.get("fmt") else FORMATS):
            pairs[(row["record_id"], fmt)] = row["reason"]
    return problems, pairs


def _read_listed_json(sums_path: Path, name: str):
    sums_path = Path(sums_path)
    path = sums_path.parent / name
    if not path.exists():
        return [f"{name} missing beside {sums_path.name}"], None
    problems = _check_listed(path, sums_path, _sums(sums_path))
    try:
        return problems, json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        return problems + [f"{name}: {e}"], None


def read_infra(sums_path: Path) -> tuple[list[str], dict[str, set[tuple[str, str]]]]:
    """infra.json beside the sums (listed in them): {repo: {(path, name)} of test-answer infrastructure members}."""
    problems, rows = _read_listed_json(sums_path, "infra.json")
    try:
        return problems, {repo: {(path, name) for path, name in pairs} for repo, pairs in (rows or {}).items()}
    except (AttributeError, TypeError, ValueError) as e:
        return problems + [f"infra.json: not {{repo: [[path, name], ...]}} ({e})"], {}


def read_pools(sums_path: Path) -> tuple[list[str], dict]:
    """pools.json beside the sums (listed in them): {pool name: {"sha256", "ids"}}."""
    problems, pools = _read_listed_json(sums_path, "pools.json")
    if pools is not None and not (isinstance(pools, dict) and all(
            isinstance(v, dict) and isinstance(v.get("ids"), list) and "sha256" in v for v in pools.values())):
        return problems + ["pools.json: not {name: {\"sha256\": ..., \"ids\": [...]}}"], {}
    return problems, pools or {}


def pool_problems(pools: dict, name: str, pool_ids: list[str]) -> list[str]:
    """The pool recomputed here must be the pool compose wrote, under the same name and digest."""
    if name not in pools:
        return [f"pools.json has no pool {name!r}"]
    ids = pools[name]["ids"]
    if not all(isinstance(i, str) for i in ids):
        return [f"pools.json pool {name!r}: ids are not all strings"]
    problems = []
    if sorted(ids) != sorted(pool_ids):
        problems.append(f"pools.json pool {name!r} differs from the recomputed pool (missing "
                        f"{sorted(set(pool_ids) - set(ids))[:5]}, extra {sorted(set(ids) - set(pool_ids))[:5]})")
    if pools[name]["sha256"] != pool_digest(ids):
        problems.append(f"pools.json pool {name!r}: sha256 is not the digest of its ids")
    return problems


def control_strings(path: Path = SPECIAL_TOKENS) -> set[str]:
    """The control tokens of every model, as text (a sentencepiece ▁ in front means a preceding space)."""
    return {t["content"].lstrip("\u2581") for m in json.loads(Path(path).read_text(encoding="utf-8"))["models"].values()
            for t in m["control_tokens"]}


def retrieval_pool(records: list[dict], ids: list[str], listed: dict, control: set[str], catalog: dict) -> list[str]:
    from .leakcheck import render_answer

    infra = {rid for (rid, _), reason in listed.items() if reason == "uses_test_infra"}
    by_id = {r["id"]: r for r in records}

    def texts(r):
        fns = catalog.get(r.get("repo"), {})
        gold = [render_answer(fns[c["fn_id"]], r.get("t_vis"), set(r.get("create_def_ids") or []))
                if c["preexisting"] and "code" in fns.get(c["fn_id"], {}) else c["code"] for c in r["create"]]
        return [r.get("background") or "", r.get("scenario") or "", *r.get("steps", []), *gold]

    return [i for i in ids if i not in infra and not any(c in t for t in texts(by_id[i]) for c in control)]


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl_lines(path)]


def subset_ids(records: list[dict], subset: str) -> list[str]:
    """Record ids of a Phase 2 train records list for subset 'all', 'lc600' or 'lc1200'."""
    ids = sorted(r["id"] for r in records if subset == "all" or subset in r["subsets"])
    if not ids:
        raise ValueError(f"subset {subset!r} is empty")
    return ids


def audit_prompt(record: dict, ex: Example, part: str, model_key: str, pool_sha: str, ctx, forbidden: dict,
                 report: Counter, dropped: Counter) -> list[str]:
    """Every rule on one prebuilt prompt that needs neither a tokenizer nor a GPU (train_job preflight and the laptop
    audit of test/aux prompts): structure, model, partition, retrieval pool, then leakcheck with the partition's hidden
    set. provenance.dropped is only counted by rule; the final prompt is what is judged."""
    from .leakcheck import check, hidden_for

    label = f"{ex.record_id} {ex.fmt}"
    shape = structure_problems(ex)
    problems = [f"{label}: {p}" for p in shape]
    if ex.model != model_key:
        problems.append(f"{label}: built for model {ex.model!r}, job model {model_key!r}")
    if ex.partition != part:
        problems.append(f"{label}: prompt file says partition {ex.partition!r}")
    if ex.pool_sha256 != pool_sha:
        problems.append(f"{label}: retrieval pool is not the job's subset")
    if any("provenance" in p or "roles" in p for p in shape):
        return problems  # the leak rules need the four provenance lists and plain user text
    ctx.hidden = {record["repo"]: hidden_for(part, forbidden.get(record["repo"], {}))}
    problems += [f"[{ex.fmt}] {p}" for p in check(record, ex.messages, ex.provenance, ctx, report)]
    for d in ex.provenance["dropped"]:
        dropped[d.get("rule", "?") if isinstance(d, dict) else "?"] += 1
    return problems
