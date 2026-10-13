"""Verdict of the 16 GB generation canary of one model, from the canary runs of v2gpu.generate:

python -m v2gpu.gencanary --a RUN_A --b RUN_B --base RUN_BASE --base-b RUN_BASE_B --hf RUN_HF --probe PROBE ...
    --model-prompts <every prompt file of the model that exists: dev, and test/aux once built> [--trained]
    --out verdict.json
python -m v2gpu.gencanary decide --variant main --verdicts codellama=VERDICT_C deepseek=VERDICT_D mistral=VERDICT_M
    --prompts-sums PT/PROMPTS_SHA256SUMS --out decision.json

The runs (mode canary, variant gen_canary, one card, dev prompts; see ops/GEN_RUNSHEET.md):
- A: the adapter's three conditions on the canary sample (--sample 5: 5 round-robin records plus the records of the
  3 longest prompts), --extend-cap 40, --repeat-longest 10;
- B: the same in a FRESH process, pairs in reverse order (--reverse), --extend-cap 40;
- BASE and BASE_B: the two base conditions, --sample 5, forward and reversed in two processes;
- HF: --teacher-force-of A with the HF reference engine (NF4 + peft);
- PROBE(s): --probe-longest, one per prompt directory (dev; test/aux once built): no text written.
Refusals (exit 2; the verdict used to pass on repeated or mismatched runs): the directories must be
distinct; every receipt canary mode, variant gen_canary, one gpu_name, and DONE (a probe may be PROBE_SHORT); A, B and
HF share the identity (generate.SHARED) with weights != base; A and B share engine, sample, cap and extend_cap; BASE
and BASE_B are base runs of the same prompts and card; reverse is False for A and BASE, True for B and BASE_B; A and B
(and BASE and BASE_B) ran in different processes (pid or started_utc); HF is an hf run of A (teacher_force_of) with an
hf_agreement; every probe is a --probe-longest run without --cap of this model, with A's engine settings (memory
steps included) and pin set; BASE and BASE_B have A's pin set.
The verdict records the canary's decision (decision 24/09: the memory steps and the pin set are chosen ONLY here, then
fixed for the whole run): {"gpu_name", "pin_set", "gpu_memory_utilization", "enforce_eager"} of A. genqueue writes it
into every job command.
Gates, pre-declared (decision 24/09):
- probe: DONE, new_tokens == max_new_tokens, prompt_tokens + new_tokens == window, the probes read every file of
  --model-prompts, and the longest probed prompt is >= the longest prompt of those files (tokenized here);
- identical: A and B hold the same pairs and every line is byte-identical (two empty runs are not identical);
  identical_base likewise;
- sample: A holds >= 15 adapter pairs, the 3 longest-prompt records are in A and B, A was run with extend_cap 40 and
  repeat_longest 10; and either A holds a cap-hit full_aware pair or it holds min(40, dev records) records;
- repeat: the full_aware pair with the longest output of A, 10 times in one process, all identical;
- eos (a gate with --trained, reported otherwise): of the full_aware pairs of the 5 round-robin records, n == 5 and
  at least 4 stop at EOS;
- adapter_effect (--trained gate): ft_only (A) differs from zero_shot (BASE) on at least 4 of the 5 round-robin
  records, so a LoRA adapter vLLM ignored cannot pass;
- hf_agreement (always a gate): the HF receipt scored exactly generate.select_hf_pairs(A) (5 full_aware pairs: the
  longest prompt, a cap-hit pair or, with none, the longest output, the rest round robin) of this A (outputs sha256),
  the first of them the model's longest F2 dev prompt (tokenized here from the dev F2 lines of --model-prompts), with
  teacher-forced top-1 agreement >= 0.95 on EVERY pair and a macro (per-pair) mean >= 0.98 (decision of
  24/09: a long cap-hit pair, usually a loop that teacher forcing predicts almost perfectly, must not carry weak short
  pairs). The pooled rate over all positions and the first divergence per pair are report-only. A missing cap-hit pair
  is written in the verdict ("no cap-hit pair found"), never silent. The scored pairs must be exactly the selected
  pairs, in order, and each must have integer counts with 0 <= agree <= positions and positions >= 1.
The verdict also records A's adapter digest and the PROMPTS_SHA256SUMS digest of every directory a probe read
(probed_sums): genqueue checks both; and the longest probed prompt (criteria.probe.probed_prompt), which
bounds the prompts of every queue built on this verdict (24/09).
Re-decode (24/09): every generated id list of A (canary_ids.jsonl) is decoded with THIS venv's tokenizer
(serialize.decode_answer) and must equal the line's text byte for byte. Run on the laptop, in the training venv, this
proves that output text does not depend on the generation venv's tokenizers. The record (lines, differing keys, A's
outputs sha256, transformers and tokenizers versions) is always written; it is a gate when A's pin set tokenizes with
other versions than training (fallback-tf, fallback-tf-bnb), and genqueue refuses such a set without a passing record
made with the training versions.
Further checks:
- refused also: a run whose outputs.jsonl is not the file its receipt names (a, b, base, base_b: outputs_sha256
  present and equal to the file's sha256), an unreadable receipt or line, and a second line of one pair;
- the 3 longest-prompt records of the sample gate are computed here from --model-prompts (dev records, the longest of
  their formats), not read from A's receipt; the repeat gate needs repeat_in_process (the repeated pair generated by
  A's own process);
- hf_agreement: the HF run scored A's FILE (of_outputs_sha256 = the sha256 of A's outputs.jsonl), each scored pair
  covers every generated id of that pair (positions = its new_tokens in A), rates and the macro mean are exact
  fractions, and "why" names every failing cause;
- --out is deleted first, so a crash never leaves an older verdict behind; the redecode record cannot set its own gate.
Refused also, a canary run that is not the jobs' engine: --cap on A, B,
BASE, BASE_B or HF, --model-path on any run, a line of A, B, BASE or BASE_B that breaks the budget rule of the model's
window, a B or BASE_B that did not generate every line itself (result.resumed 0, generated = its lines), an HF run
other than the NF4 bf16 hub model; and an --out inside a run directory or equal to a prompt file.
Exit 0 if every gate passes, 1 if one fails, 2 on a refusal.

decide (decision 24/09): ONE decision per variant, taken on the laptop before any QUEUE
of the variant. It needs a passing canary verdict (trained or not) of EVERY model the variant plans (main: all three;
the dev variants: DeepSeek), each of the model it is given for, with a decision engines.decision_problems accepts; with
--prompts-sums (required for main: the test/aux directory), a probe of each verdict must have read that directory. The
decisions must be identical (card, pin set, memory step). If they differ, decide refuses, and the refusal names the
common target, engines.conservative_decision: the most conservative decision covering every model's own (the highest
memory step on MEMORY_LADDER, the pin set holding every fallback any model needed, so fallback-tf and fallback-bnb give
fallback-tf-bnb), and lists the models whose canary must be re-run at that target (fresh workdirs), before decide runs
again. A tf fallback target holds for the whole variant (every command keeps its own --ids-sha256). The prefill chunk
(max_num_batched_tokens 4,096) is never chosen here. The decision file (schema gtestgen.v2.gen_decision.v2) records
the decision and every verdict's path, sha256 and trained flag; on a refusal it holds pass false, the problems, the
target and the models to re-run. genqueue requires it (--decision) and a TRAINED verdict of its model with the same
decision. Exit 0 on a decision, 2 on a refusal.

Code and engine binding: the canary's runs must share one gen_code_sha256; the verdict
records it, A's declared engine (engines.declared_engine of identity.engine_settings: the settings without what names
the model) and A's measured engine fingerprint (engines.engine_fingerprint of the receipt's engine). decide requires
every verdict of one decision to share all three and copies them into the decision file (schema v2). genqueue refuses a
decision whose gen_code_sha256 or declared engine is not this checkout's, and writes the decision file's sha256 into
every job command (--decision-sha256, part of the job identity); gencoverage --decision requires every workdir to carry
that sha256, the same code, the same declared engine, card, pin set and engine fingerprint. --probe may repeat
(`--probe DEV --probe TEST` used to keep only the last).
A, B and HF share adapter_config_sha256 (generate.SHARED; required of adapter runs) and
the verdict records it; A, B, BASE and BASE_B must show one measured engine fingerprint. genqueue compares the TRAINED
verdict with the decision on every key of BINDING (equal and non-null), since main's decision comes from untrained
canaries and the trained canary runs later.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from .examples import jsonl_lines, read_jsonl
from .engines import (PIN_SETS, conservative_decision, decision_problems, declared_engine, engine_fingerprint,
                      names_engine, tokenizes_like_training)
from .generate import SHARED, select_hf_pairs
from .models import MODELS, load_tokenizer
from .receipt import sha256_file
from .runs import follows_budget_rule, row_key, variant_arg, variant_spec
from .serialize import decode_answer, prompt_ids

MIN_OF_FIVE, SAMPLE_N = 4, 5
HF_MIN_PAIR, HF_MIN_MACRO = 0.95, 0.98      # every scored pair, and the mean of the per-pair rates
EXACT_MIN_PAIR, EXACT_MIN_MACRO = Fraction("0.95"), Fraction("0.98")   # compared exactly
MIN_ADAPTER_PAIRS, LONGEST_RECORDS = 15, 3
HF_PAIRS_NEEDED = 5
EXTEND_LIMIT, REPEAT = 40, 10
NAMES = ("a", "b", "base", "base_b", "hf")


def _read_run(d: Path) -> tuple[dict, dict, list[str]]:
    """(lines {key: line}, receipt, problems) of a canary workdir. An unreadable receipt, an unreadable line or a second
    line of one pair is a problem, not a traceback or a silently dropped line."""
    d, lines, problems, n = Path(d), {}, [], 0
    try:
        rc = json.loads((d / "receipt-generate.json").read_text())
        if not isinstance(rc, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        rc = {}
        problems.append(f"{d}: no readable receipt-generate.json ({e})")
    path = d / "outputs.jsonl"
    if path.exists():
        try:
            for line in jsonl_lines(path):
                if line.strip():
                    n += 1
                    lines[row_key(json.loads(line))] = line
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"{path}: unreadable ({e!r})")
        if n != len(lines) and not problems:
            problems.append(f"{path}: {n} lines for {len(lines)} pairs (a duplicate line)")
    return lines, rc, problems


def _load(d: Path) -> tuple[dict, dict]:
    lines, rc, _ = _read_run(d)
    return lines, rc


def _identical(x: dict, y: dict) -> dict:
    differ = sorted([list(k) for k in x if k in y and x[k] != y[k]])
    same_keys = set(x) == set(y) and bool(x)
    return {"gate": True, "pass": same_keys and not differ, "n": len(x), "same_pairs": same_keys, "differ": differ}


def refusals(dirs: dict, runs: dict, probes: list) -> list[str]:
    """Why these runs are not the canary of one model; empty when they are."""
    out = []
    paths = [Path(d).resolve() for d in [*dirs.values(), *(p[0] for p in probes)]]
    if len(set(paths)) != len(paths):
        out.append("the canary directories are not distinct")
    for n in ("a", "b", "base", "base_b"):          # each run file is its receipt's
        f, want = Path(dirs[n]) / "outputs.jsonl", runs[n][1].get("outputs_sha256")
        if want is None or not f.exists() or sha256_file(f) != want:
            out.append(f"{n}: outputs.jsonl is not the file its receipt names (outputs_sha256 {want!r})")
    out += _job_engine_problems(runs, probes)
    ident = {n: runs[n][1].get("identity") or {} for n in NAMES}
    everything = [(n, runs[n][1]) for n in NAMES] + [(f"probe {p[0]}", p[1]) for p in probes]
    for n, rc in everything:
        i = rc.get("identity") or {}
        if rc.get("mode") != "canary" or i.get("mode") != "canary":
            out.append(f"{n}: not a canary-mode run")
        if i.get("variant") != "gen_canary":
            out.append(f"{n}: variant {i.get('variant')!r}, not gen_canary")
        if rc.get("status") != "DONE" and not (n.startswith("probe") and rc.get("status") == "PROBE_SHORT"):
            out.append(f"{n}: receipt status {rc.get('status')!r}, not DONE")
    if len({(rc.get("identity") or {}).get("gpu_name") for _, rc in everything}) != 1:
        out.append("the runs name several cards (gpu_name)")
    codes = {json.dumps((rc.get("identity") or {}).get("gen_code_sha256"), sort_keys=True) for _, rc in everything}
    if len(codes) != 1 or "null" in codes:           # one generation code for the whole canary
        out.append("the runs name several generation codes (gen_code_sha256), or none")
    for n in ("b", "hf"):
        differ = sorted(k for k in SHARED if ident[n].get(k) != ident["a"].get(k))
        if differ:
            out.append(f"{n} differs from a in {differ}")
    for n in ("a", "b", "hf"):
        if ident[n].get("weights") in (None, "base"):
            out.append(f"{n}: weights {ident[n].get('weights')!r}; an adapter run is needed")
        if not ident[n].get("adapter_config_sha256"):         # the adapter config is bound too
            out.append(f"{n}: no adapter_config_sha256 (--adapter-config-sha256 of the adapter it ran)")
    prints = {n: json.dumps(engine_fingerprint(runs[n][1].get("engine") or {}), sort_keys=True)
              for n in ("a", "b", "base", "base_b")}
    if len(set(prints.values())) != 1:                         # one measured vLLM engine
        differ = [n for n in prints if prints[n] != prints["a"]]
        out.append(f"the vLLM runs differ in the engine fingerprint ({', '.join(differ)} differ from a)")
    for k in ("engine", "engine_settings", "sample", "cap", "extend_cap"):
        if ident["a"].get(k) != ident["b"].get(k):
            out.append(f"b differs from a in {k}")
    for n in ("base", "base_b"):
        if ident[n].get("weights") != "base":
            out.append(f"{n}: weights {ident[n].get('weights')!r}, a base run is needed")
    for k in ("model", "prompts", "prompts_sums_sha256", "ids_sha256", "window", "engine", "engine_settings", "cap",
              "sample", "pin_set"):
        if ident["base"].get(k) != ident["base_b"].get(k):
            out.append(f"base_b differs from base in {k}")
    for k in ("model", "prompts", "window", "engine", "engine_settings", "pin_set"):
        if ident["base"].get(k) != ident["a"].get(k):
            out.append(f"base differs from a in {k}")
    for n, want in (("a", False), ("base", False), ("b", True), ("base_b", True)):
        if ident[n].get("reverse") is not want:
            out.append(f"{n}: reverse is {ident[n].get('reverse')!r}, the canary needs {want}")
    for x, y in (("a", "b"), ("base", "base_b")):
        rx, ry = runs[x][1], runs[y][1]
        if rx.get("pid") == ry.get("pid") and rx.get("started_utc") == ry.get("started_utc"):
            out.append(f"{x} and {y} ran in one process (same pid and start): a fresh process is needed")
    for n in ("a", "b", "base", "base_b"):
        if ident[n].get("engine") != "vllm":
            out.append(f"{n}: engine {ident[n].get('engine')!r}, not vllm")
    if ident["hf"].get("engine") != "hf" or not ident["hf"].get("teacher_force_of") or \
            "hf_agreement" not in runs["hf"][1]:
        out.append("hf: not an hf teacher-forced run (--engine hf --teacher-force-of A)")
    for d, rc in probes:
        i = rc.get("identity") or {}
        if not i.get("probe_longest"):
            out.append(f"probe {d}: not a --probe-longest run")
        if i.get("cap") is not None:
            out.append(f"probe {d}: run with --cap {i.get('cap')}; the probe fills the window")
        if i.get("model") != ident["a"].get("model"):
            out.append(f"probe {d}: model {i.get('model')!r}, not {ident['a'].get('model')!r}")
        differ = [k for k in ("engine_settings", "pin_set") if i.get(k) != ident["a"].get(k)]
        if differ:
            out.append(f"probe {d}: differs from a in {differ} (the probe must run what the jobs will run)")
    return out


def _job_engine_problems(runs: dict, probes: list) -> list[str]:
    """The canary runs what the jobs will run. No --cap and no --model-path (the hub model
    at its pinned revision); every line of A, B, BASE and BASE_B follows the budget rule of the model's window; B and
    BASE_B generated every line in their own process; the HF reference is the NF4 bf16 hub model."""
    out = []
    model = (runs["a"][1].get("identity") or {}).get("model")
    spec = MODELS.get(model)
    if spec is None:
        return [f"a: model {model!r} is not a pinned model"]
    for n, rc in [(n, runs[n][1]) for n in NAMES] + [(f"probe {d}", rc) for d, rc in probes]:
        i = rc.get("identity") or {}
        if not n.startswith("probe") and i.get("cap") is not None:        # the probes: refusals() already
            out.append(f"{n}: run with --cap {i.get('cap')}; the canary runs the jobs' budget")
        if i.get("model_path") is not None:
            out.append(f"{n}: run with --model-path {i.get('model_path')!r}; the canary runs the hub model "
                       f"{spec.hf_id} at its pinned revision")
    for n in ("a", "b", "base", "base_b"):
        rows = [json.loads(line) for line in runs[n][0].values()]
        bad = sorted({r["record_id"] for r in rows
                      if not follows_budget_rule(r["prompt_tokens"], r["max_new_tokens"], spec.window)})
        if bad:
            out.append(f"{n}: lines break the budget rule min(6144, {spec.window} - prompt_tokens): {bad[:5]}")
    for n in ("b", "base_b"):
        result = runs[n][1].get("result") or {}
        if result.get("resumed") != 0 or result.get("generated") != len(runs[n][0]):
            out.append(f"{n}: generated {result.get('generated')!r} and resumed {result.get('resumed')!r} of "
                       f"{len(runs[n][0])} lines; the rerun must generate every line itself")
    want = {"model": spec.hf_id, "quant": "nf4", "dtype": "bfloat16"}
    got = (runs["hf"][1].get("identity") or {}).get("engine_settings")
    if got != want:
        out.append(f"hf: engine settings {got!r}; the reference is {want!r}")
    return out


def longest_prompt(files, model: str) -> tuple[int, int, str | None, list[str]]:
    """(tokens of the longest prompt, dev records, the record of the longest F2 dev prompt, the records of the
    LONGEST_RECORDS longest dev prompts over every format) over the model's prompt files (serialize.prompt_ids; ties to
    the smaller record id, as generate.select_hf_pairs and generate.select_pairs)."""
    tok = load_tokenizer(MODELS[model])
    longest, dev, f2 = 0, {}, []
    for path in files:
        for (rid, _), ex in read_jsonl(path).items():
            n = len(prompt_ids(tok, ex.messages))
            longest = max(longest, n)
            if ex.partition == "dev":
                dev[rid] = max(dev.get(rid, 0), n)
                if ex.fmt == "F2":
                    f2.append((-n, rid))
    top = sorted(dev, key=lambda r: (-dev[r], r))[:LONGEST_RECORDS]
    return longest, len(dev), min(f2)[1] if f2 else None, top


def redecode(a_dir: Path, tok) -> dict:
    """Decode every generated id list of canary run A with `tok` and compare with each line's text."""
    import importlib.metadata as md

    lines, _ = _load(a_dir)
    ids, path = {}, Path(a_dir) / "canary_ids.jsonl"
    for line in jsonl_lines(path) if path.exists() else []:
        if line.strip():
            row = json.loads(line)
            ids[tuple(row["key"])] = row["new_ids"]
    missing = sorted(list(k) for k in lines if k not in ids)
    differ = sorted(list(k) for k in lines if k in ids and decode_answer(tok, ids[k]) != json.loads(lines[k])["text"])
    out = Path(a_dir) / "outputs.jsonl"
    return {"pass": bool(lines) and not missing and not differ, "lines": len(lines), "missing": missing[:20],
            "differ": differ[:20], "of_outputs_sha256": sha256_file(out) if out.exists() else None,
            "versions": {k: md.version(k) for k in ("transformers", "tokenizers")}}


def verdict(runs: dict, probes: list, longest: int, prompt_shas: set, dev_records: int, trained: bool,
            longest_f2: str | None, redecoded: dict, longest_records: list[str], a_file_sha256: str | None) -> dict:
    """runs: {name: (lines {key: line}, receipt)}; probes: [(dir, receipt)]; longest_records: the 3 longest dev records
    of --model-prompts; a_file_sha256: the sha256 of A's outputs.jsonl."""
    a, a_rc = runs["a"]
    crit = {}
    why, done = [], True
    probed = max(((rc.get("probe") or {}).get("prompt_tokens") or 0 for _, rc in probes), default=0)
    read = {sha for _, rc in probes for sha in ((rc.get("identity") or {}).get("prompts") or {}).values()}
    for d, rc in probes:
        p = rc.get("probe") or {}
        if rc.get("status") != "DONE" or p.get("new_tokens") != p.get("max_new_tokens"):
            why.append(f"{d}: new_tokens {p.get('new_tokens')} is not max_new_tokens {p.get('max_new_tokens')}")
        if (p.get("prompt_tokens") or 0) + (p.get("new_tokens") or 0) != (rc.get("identity") or {}).get("window"):
            why.append(f"{d}: prompt + new tokens do not fill the window {(rc.get('identity') or {}).get('window')}")
    if probed < longest:
        why.append(f"the longest probed prompt has {probed} tokens; the longest prompt of the model has {longest}")
    why += [f"a prompt file of the model was not read by any probe: {name}" for name, sha in sorted(prompt_shas.items())
            if sha not in read]
    done = not why
    crit["probe"] = {"gate": True, "pass": done, "why": why, "longest_prompt": longest, "probed_prompt": probed,
                     "cuda": [(rc.get("probe") or {}).get("cuda") for _, rc in probes]}
    crit["identical"] = _identical(a, runs["b"][0])
    crit["identical_base"] = _identical(runs["base"][0], runs["base_b"][0])
    rows = {k: json.loads(v) for k, v in a.items()}
    sample = a_rc.get("sample") or {}
    records = {k[0] for k in a}
    f2 = [r for r in rows.values() if r["condition"] == "full_aware"]
    cap_hit = any(r["finish_reason"] == "length" for r in f2)
    ident = a_rc.get("identity") or {}
    swhy = []
    if len(a) < MIN_ADAPTER_PAIRS:
        swhy.append(f"a holds {len(a)} adapter pairs, fewer than {MIN_ADAPTER_PAIRS}")
    if len(longest_records) != LONGEST_RECORDS or not set(longest_records) <= records or \
            not set(longest_records) <= {k[0] for k in runs["b"][0]}:       # from the prompt files
        swhy.append(f"the {LONGEST_RECORDS} longest-prompt records {longest_records} are not all in a and b")
    if ident.get("extend_cap") != EXTEND_LIMIT or ident.get("repeat_longest") != REPEAT or sample.get("n") != SAMPLE_N:
        swhy.append(f"a was not run with --sample {SAMPLE_N} --extend-cap {EXTEND_LIMIT} --repeat-longest {REPEAT}")
    if not cap_hit and len(records) < min(EXTEND_LIMIT, dev_records):
        swhy.append(f"no cap hit and a holds {len(records)} records, fewer than min({EXTEND_LIMIT}, {dev_records})")
    crit["sample"] = {"gate": True, "pass": not swhy, "why": swhy, "adapter_pairs": len(a), "records": len(records),
                      "longest_records": longest_records}
    rep = a_rc.get("repeat") or {}
    top = min(f2, key=lambda r: (-r["new_tokens"], r["record_id"])) if f2 else None
    crit["repeat"] = {"gate": True, "pass": bool(top) and rep.get("n") == REPEAT and rep.get("identical") is True and
                      rep.get("key") == list(row_key(top)) and a_rc.get("repeat_in_process") is True,
                      "n": rep.get("n"), "identical": rep.get("identical"), "new_tokens": rep.get("new_tokens"),
                      "in_process": a_rc.get("repeat_in_process")}
    rr = list(sample.get("records") or [])
    aware = [r for r in f2 if r["record_id"] in rr]
    stopped = sum(r["stopped_at_eos"] for r in aware)
    crit["eos"] = {"gate": trained, "pass": len(aware) == SAMPLE_N and stopped >= MIN_OF_FIVE, "stopped": stopped,
                   "n": len(aware), "min": MIN_OF_FIVE}
    base = {k[0]: json.loads(v)["text"] for k, v in runs["base"][0].items() if k[3] == "zero_shot"}
    ft = {r["record_id"]: r["text"] for r in rows.values() if r["condition"] == "ft_only" and r["record_id"] in rr}
    both = [r for r in ft if r in base]
    differ = sum(ft[r] != base[r] for r in both)
    crit["adapter_effect"] = {"gate": trained, "pass": len(both) == SAMPLE_N and differ >= MIN_OF_FIVE,
                              "differ": differ, "n": len(both), "min": MIN_OF_FIVE}
    hf = runs["hf"][1].get("hf_agreement") or {}
    want = select_hf_pairs(list(rows.values()))
    hwhy = []
    if longest_f2 is None or not want["keys"] or want["keys"][0][0] != longest_f2:
        hwhy.append(f"the HF pairs do not start at the model's longest F2 dev prompt ({longest_f2}): a lacks it")
    pairs = hf.get("pairs") or []

    def counted(p) -> bool:                   # a corrupt pair must not lift the macro mean
        n, a = p.get("positions"), p.get("agree")
        row = rows.get(tuple(p.get("key") or ()))
        return all(type(x) is int for x in (n, a)) and n >= 1 and 0 <= a <= n and row is not None and \
            n == row["new_tokens"]             # every generated id of that pair scored

    rates = [Fraction(p["agree"], p["positions"]) if counted(p) else Fraction(0) for p in pairs]
    macro = sum(rates, Fraction(0)) / len(rates) if rates else Fraction(0)
    if hf.get("keys") != want["keys"]:
        hwhy.append("the HF receipt's keys are not generate.select_hf_pairs of this a")
    if len(want["keys"]) != HF_PAIRS_NEEDED or len(pairs) != HF_PAIRS_NEEDED:
        hwhy.append(f"{len(want['keys'])} pairs selected and {len(pairs)} pairs scored; {HF_PAIRS_NEEDED} are needed")
    if [p.get("key") for p in pairs] != want["keys"]:
        hwhy.append("the scored pairs are not the selected pairs, in order")
    if hf.get("of_outputs_sha256") is None or hf.get("of_outputs_sha256") != a_file_sha256:
        hwhy.append(f"the HF run scored outputs {hf.get('of_outputs_sha256')!r} (of_outputs_sha256), not a's file "
                    f"{a_file_sha256!r}")
    hwhy += [f"pair {p.get('key')}: agree {p.get('agree')!r} / positions {p.get('positions')!r} are not integers "
             f"with 0 <= agree <= positions = the pair's new_tokens in a, positions >= 1" for p in pairs
             if not counted(p)]
    hwhy += [f"pair {p.get('key')}: rate {float(r):.4f} < {HF_MIN_PAIR}" for p, r in zip(pairs, rates, strict=True)
             if counted(p) and r < EXACT_MIN_PAIR]
    if macro < EXACT_MIN_MACRO:
        hwhy.append(f"macro (per-pair) mean {float(macro):.4f} < {HF_MIN_MACRO}")
    crit["hf_agreement"] = {
        "gate": True, "pass": not hwhy,
        "pair_rates": [float(r) for r in rates], "macro_rate": float(macro), "min_pair": HF_MIN_PAIR,
        "min_macro": HF_MIN_MACRO, "pooled_rate": hf.get("rate"), "positions": hf.get("positions"),
        "cap_hit_found": want["cap_hit_found"], "longest_f2": longest_f2, "why": hwhy,
        "first_divergence": [p.get("first_divergence") for p in pairs]}
    if not want["cap_hit_found"]:
        crit["hf_agreement"]["note"] = "no cap-hit pair found: the longest output is scored instead"
    pin_set = ident.get("pin_set")
    crit["redecode"] = {**redecoded, "gate": pin_set not in PIN_SETS or not tokenizes_like_training(pin_set)}
    ok = all(c["pass"] for c in crit.values() if c["gate"])
    settings = ident.get("engine_settings") or {}
    decision = {"gpu_name": ident.get("gpu_name"), "pin_set": ident.get("pin_set"),
                "gpu_memory_utilization": settings.get("gpu_memory_utilization"),
                "enforce_eager": settings.get("enforce_eager")}
    probed_sums = sorted({(rc.get("identity") or {}).get("prompts_sums_sha256") for _, rc in probes} - {None})
    return {"pass": ok, "trained": trained, "criteria": crit, "decision": decision,
            "adapter_tensor_sha256": ident.get("adapter_tensor_sha256"),
            "adapter_config_sha256": ident.get("adapter_config_sha256"), "probed_sums": probed_sums,
            "a_outputs_sha256": a_file_sha256, "gen_code_sha256": ident.get("gen_code_sha256"),
            "engine_settings": declared_engine(settings),
            "engine_fingerprint": engine_fingerprint(a_rc.get("engine") or {})}


DECISION_SCHEMA = "gtestgen.v2.gen_decision.v2"      # v2: bound to code and engine


BINDING = ("gen_code_sha256", "engine_settings", "engine_fingerprint")   # what a decision binds


def planned_models(variant: str) -> list[str]:
    return sorted({m for models, _ in variant_spec(variant).values() for m in models})


def decide(verdicts: dict, variant: str, sums_sha256: str | None) -> dict:
    """The decision record of a variant from its models' canary verdicts; see the docstring (decide)."""
    planned, problems = planned_models(variant), []
    if variant == "gen_canary":
        problems.append("gen_canary is the canary's own variant: decide for a real variant")
    problems += [f"{m}: no verdict ({variant} plans {', '.join(planned)})" for m in planned if m not in verdicts]
    problems += [f"{m}: {variant} does not plan {m} (it plans {', '.join(planned)})" for m in sorted(verdicts)
                 if m not in planned]
    if variant == "main" and sums_sha256 is None:
        problems.append("--prompts-sums is required for main: every model's canary probes the test/aux directory")
    decisions, record, bound = {}, {}, {}
    for m, path in sorted(verdicts.items()):
        try:
            v = json.loads(Path(path).read_text())
            if not isinstance(v, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as e:
            problems.append(f"{m}: {path}: no readable canary verdict ({e})")
            continue
        if v.get("model") != m:
            problems.append(f"{m}: {path} is a verdict of {v.get('model')!r}")
        if v.get("pass") is not True:
            problems.append(f"{m}: {path} is not a passing canary verdict")
        if sums_sha256 is not None and sums_sha256 not in (v.get("probed_sums") or []):
            problems.append(f"{m}: no probe of {path} read the prompt directory (PROMPTS_SHA256SUMS "
                            f"{sums_sha256[:12]})")
        found = decision_problems(v.get("decision"))
        problems += [f"{m}: {path}: {p}" for p in found]
        if not found:
            decisions[m] = v["decision"]
        record[m] = {"verdict": str(path), "sha256": sha256_file(path), "trained": v.get("trained") is True}
        bound[m] = {k: v.get(k) for k in BINDING}
    target, rerun = None, []
    if len({json.dumps(d, sort_keys=True) for d in decisions.values()}) > 1:
        target, rerun, found = conservative_decision(decisions)
        shown = "; ".join(f"{m} {json.dumps(d, sort_keys=True)}" for m, d in sorted(decisions.items()))
        problems += found or [f"the canary decisions differ ({shown}): common target "
                              f"{json.dumps(target, sort_keys=True)}; re-run the canary of {', '.join(rerun)} at that "
                              f"target (fresh workdirs), then decide again"]
    elif bound:                       # one decision: then one code and one engine
        for k in BINDING:
            values = {json.dumps(b[k], sort_keys=True) for b in bound.values()}
            nameless = k == "engine_fingerprint" and not all(names_engine(b[k]) for b in bound.values())
            if len(values) != 1 or "null" in values or nameless:
                problems.append(f"the verdicts name several {k}, or none: {', '.join(sorted(bound))} must have run "
                                f"one checkout and one engine; re-run the canaries that differ")
    if problems:
        return {"schema": DECISION_SCHEMA, "pass": False, "variant": variant, "problems": problems, "target": target,
                "rerun": rerun}
    return {"schema": DECISION_SCHEMA, "pass": True, "variant": variant, "decision": next(iter(decisions.values())),
            "verdicts": record, "prompts_sums_sha256": sums_sha256, **next(iter(bound.values()))}


def _model_path(value: str) -> tuple[str, Path]:
    model, sep, path = value.partition("=")
    if not sep or model not in MODELS or not path:
        raise argparse.ArgumentTypeError(f"{value!r} is not MODEL=VERDICT with MODEL one of {sorted(MODELS)}")
    return model, Path(path)


def decide_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="python -m v2gpu.gencanary decide", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, type=variant_arg)
    ap.add_argument("--verdicts", required=True, nargs="+", type=_model_path, help="MODEL=verdict.json, one per model")
    ap.add_argument("--prompts-sums", type=Path, help="the directory every probe must have read (main: test/aux)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    verdicts = dict(a.verdicts)
    if len(verdicts) != len(a.verdicts):
        ap.error("--verdicts names a model twice")
    if a.out.resolve() in {Path(p).resolve() for p in [*verdicts.values(), a.prompts_sums] if p is not None}:
        ap.error(f"--out {a.out} is one of the inputs: decide deletes --out first")
    a.out.unlink(missing_ok=True)
    out = decide(verdicts, a.variant, None if a.prompts_sums is None else sha256_file(a.prompts_sums))
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    if not out["pass"]:
        print("REFUSED: " + "; ".join(out["problems"][:20]), file=sys.stderr)
        return 2
    print(json.dumps({"pass": True, "variant": a.variant, "decision": out["decision"]}, sort_keys=True))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["decide"]:
        return decide_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m v2gpu.gencanary", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--a", "--b", "--base", "--base-b", "--hf"):
        ap.add_argument(flag, type=Path, required=True)
    ap.add_argument("--probe", type=Path, nargs="+", action="extend", required=True,
                    help="one or more probe runs; the flag may repeat")
    ap.add_argument("--model-prompts", type=Path, nargs="+", required=True)
    ap.add_argument("--trained", action="store_true", help="the adapter is trained: EOS and adapter effect are gates")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if any(out.is_relative_to(Path(d).resolve()) for d in [*(getattr(args, n) for n in NAMES), *args.probe]) or \
            out in {p.resolve() for p in args.model_prompts}:
        ap.error(f"--out {args.out} lies in a run directory or is a prompt file: the verdict deletes --out first")
    args.out.unlink(missing_ok=True)       # a crash never leaves an older verdict behind
    dirs = {n: getattr(args, n) for n in NAMES}
    read = {n: _read_run(d) for n, d in dirs.items()}
    probes_read = [(str(d), _read_run(d)) for d in args.probe]
    problems = [p for _, _, found in [*read.values(), *(r for _, r in probes_read)] for p in found]
    runs = {n: (lines, rc) for n, (lines, rc, _) in read.items()}
    probes = [(d, rc) for d, (_, rc, _) in probes_read]
    problems += refusals(dirs, runs, probes) if not problems else []
    if problems:
        args.out.write_text(json.dumps({"pass": False, "problems": problems}, indent=2, sort_keys=True))
        print("REFUSED: " + "; ".join(problems[:5]))
        return 2
    model = runs["a"][1]["identity"]["model"]
    longest, dev, longest_f2, longest_records = longest_prompt(args.model_prompts, model)
    v = verdict(runs, probes, longest, {p.name: sha256_file(p) for p in args.model_prompts}, dev, args.trained,
                longest_f2, redecode(dirs["a"], load_tokenizer(MODELS[model])), longest_records,
                sha256_file(Path(dirs["a"]) / "outputs.jsonl"))
    v["model"] = model
    v["runs"] = {**{n: str(d) for n, d in dirs.items()}, "probe": [str(d) for d in args.probe]}
    args.out.write_text(json.dumps(v, indent=2, sort_keys=True))
    print(json.dumps({"pass": v["pass"], "model": v["model"]}))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
