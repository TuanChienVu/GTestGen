"""Pre-rendered prompt files (the interface with the GPU pipeline v2gpu, fixed 24/09/2026): one JSONL per (model, pool,
partition, format) with rows {record_id, partition, fmt, model, pool_sha256, messages, answer, provenance}; `answer` is
null for test and aux_test. PROMPTS_META.json, excluded.jsonl, infra.json, pools.json and PROMPTS_SHA256SUMS sit next to
them; the sums list every other file of the directory, so a job can refuse any file that was changed, added or removed.
META keys (read by v2gpu): v2tools_commit, phase2_sums_sha256, symtimes_sha256, special_tokens_sha256,
tokenizer_revisions, compose, files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .compose import FORMATS, Composer

HELD_OUT = ("test", "aux_test")
LIMIT_HELD_OUT = 16384


def pool_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def control_strings(tokens: list[str]) -> list[str]:
    """Control tokens as they appear in text: the SentencePiece word markers "▁" are stripped from the front
    ("▁<PRE>" -> "<PRE>"), exactly as v2gpu's examples.control_strings() does (lstrip). A token that is nothing but
    markers is dropped: the empty string would match every text."""
    return [s for s in (t.lstrip("\u2581") for t in tokens) if s]


def has_control_token(text: str, control: list[str]) -> bool:
    return any(c in text for c in control_strings(control))


def clean_pool(comp: Composer, pool: list[str], control: dict[str, list[str]]) -> list[str]:
    """The retrieval pool actually used (as the GPU pipeline v2gpu expects it since 24/09/2026): the pool minus the
    users of test-answer infrastructure and minus records whose Gherkin or gold answer holds a control token of ANY
    model, so that the pool, and its pool_sha256, is the same for every model."""
    every = [c for toks in control.values() for c in toks]
    users = comp.infra()["users"]
    return sorted(p for p in pool if p not in users and not has_control_token(
        "\n".join([comp.records[p]["background"], comp.records[p]["scenario"], *comp.records[p]["steps"], comp.gold(comp.records[p])]),
        every))


def held_out_exclusions(comp: Composer, records: list[dict], pools: dict[str, list[str]],
                        lengths: dict[str, Callable[[list[dict], str], int]], counters: dict[str, Callable[[str], int]],
                        control: dict[str, list[str]]) -> list[dict]:
    """Held-out records dropped from EVERY model, pool and condition: (F2 prompt + gold answer) longer than
    LIMIT_HELD_OUT tokens under ANY of the tokenizers, or a control token of ANY model in the prompt or the gold answer
    (the union keeps the held-out set identical across models, so that models are compared on the same records). The
    only function that reads the gold answer of a held-out record; compose_prompt never does. Each F2 prompt is
    composed with that model's counter and each (clean) pool that build() writes, i.e. exactly the prompts shipped."""
    out = []
    every = [c for toks in control.values() for c in toks]
    for r in records:
        gold, reason = comp.gold(r), None
        for model, count in sorted(counters.items()):
            for _, pool in sorted(pools.items()):
                msgs, _ = comp.compose_prompt(r, "F2", count, pool)
                if has_control_token(msgs[0]["content"] + gold, every):
                    reason = "control_token"
                elif lengths[model](msgs, gold) > LIMIT_HELD_OUT:
                    reason = "too_long_union"
                if reason:
                    break
            if reason:
                break
        if reason:
            out.append({"record_id": r["id"], "model": None, "fmt": None, "reason": reason})
    return out


def build(comp: Composer, out: Path, variant: str, models: list[str], parts: list[str], pools: dict[str, list[str]],
          counters: dict[str, Callable[[str], int]], lengths: dict[str, Callable[[list[dict], str], int]],
          control: dict[str, list[str]], meta: dict) -> dict:
    """Write the prompt files of one compose variant into a new or empty directory. `pools` maps a pool name ("all",
    "lc600", "lc1200": v2gpu's subset names) to train ids; for a pool other than "all", only train records of that
    subset are written for the train partition. When a held-out partition is built, held_out_exclusions() runs here, on
    the same pools and counters, so the held-out length rule can be neither skipped nor fed other prompts."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: a stale file would sit next to the new PROMPTS_SHA256SUMS")
    out.mkdir(parents=True, exist_ok=True)
    used_pools = {name: clean_pool(comp, raw, control) for name, raw in sorted(pools.items())}
    held = sorted((r for r in comp.records.values() if r["partition"] in HELD_OUT and r["partition"] in parts),
                  key=lambda r: r["id"])
    excluded = held_out_exclusions(comp, held, used_pools, lengths, counters, control) if held else []
    # EVERY user of test-answer infrastructure, whatever partitions this directory holds, so that a directory of
    # test prompts alone still tells the pool; pools.json states the pools outright
    excluded += [{"record_id": i, "model": None, "fmt": None, "reason": "uses_test_infra"} for i in sorted(comp.infra()["users"])]
    dropped_ids = {e["record_id"] for e in excluded}
    files, seen = [], set()
    for model in models:
        count = counters[model]
        for pool_name, raw_pool in sorted(pools.items()):
            pool = used_pools[pool_name]
            psha = pool_sha256(pool)
            in_pool = set(raw_pool)
            for part in parts:
                recs = sorted((r for r in comp.records.values() if r["partition"] == part
                               and (part != "train" or pool_name == "all" or r["id"] in in_pool)), key=lambda r: r["id"])
                for fmt in FORMATS:
                    rows = []
                    for r in recs:
                        if r["id"] in dropped_ids:
                            continue
                        msgs, prov = comp.compose_prompt(r, fmt, count, pool)
                        answer = None if part in HELD_OUT else comp.gold(r)
                        if has_control_token(msgs[0]["content"] + (answer or ""), control.get(model, [])):
                            key = (r["id"], model, fmt)
                            if key not in seen:  # the same (record, model, format) under another pool
                                seen.add(key)
                                excluded.append({"record_id": r["id"], "model": model, "fmt": fmt, "reason": "control_token"})
                            continue
                        rows.append({"record_id": r["id"], "partition": part, "fmt": fmt, "model": model,
                                     "pool_sha256": psha, "messages": msgs, "answer": answer, "provenance": prov})
                    name = f"{model}.{pool_name}.{part}.{fmt}.jsonl"
                    (out / name).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
                                            encoding="utf-8")
                    files.append(name)
    (out / "excluded.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in excluded), encoding="utf-8")
    infra = {repo: sorted([p, n] for p, n in v) for repo, v in sorted(comp.infra()["members"].items())}
    (out / "pools.json").write_text(json.dumps({k: {"sha256": pool_sha256(v), "ids": v} for k, v in used_pools.items()},
                                               indent=1, sort_keys=True) + "\n")
    (out / "infra.json").write_text(json.dumps(infra, indent=2, sort_keys=True) + "\n")
    (out / "PROMPTS_META.json").write_text(json.dumps({**meta, "compose": {"variant": variant, **asdict(comp.params)},
                                                       "files": sorted(files)}, indent=2, sort_keys=True) + "\n")
    names = sorted(files) + ["PROMPTS_META.json", "excluded.jsonl", "infra.json", "pools.json"]
    sums = "".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in names)
    (out / "PROMPTS_SHA256SUMS").write_text(sums)
    return {"files": len(files), "excluded": len(excluded)}
