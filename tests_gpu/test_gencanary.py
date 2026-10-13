# tests_gpu/test_gencanary.py
import hashlib
import json
from dataclasses import asdict

import pytest

from v2gpu import gencanary, generate
from v2gpu.examples import Example
from v2gpu.gencanary import main
from v2gpu.runs import row_key

IDS = [f"R{i % 3}|f{i}.feature|{i}|s" for i in range(8)]       # 8 dev records over 3 repos
RR = generate.round_robin_records(IDS[:5], 5)                     # the 5 round-robin records of the sample
LONGEST = IDS[5:]                                                 # the 3 longest prompts
COND_FMT = {"zero_shot": "F0", "rag_only": "F1", "ft_only": "F0", "full_neutral": "F1", "full_aware": "F2"}
PROMPT = {r: 100 + 10 * i for i, r in enumerate(IDS)}           # IDS[7] has the longest prompt
PROV = {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line(rid, cond, text, eos=True, seed="s42"):
    base = cond in ("zero_shot", "rag_only")
    new = 5 if eos else 6144
    return {"record_id": rid, "model": "mistral", "seed": "base" if base else seed, "condition": cond,
            "variant": "gen_canary", "fmt": COND_FMT[cond], "text": text, "prompt_tokens": PROMPT[rid],
            "new_tokens": new, "max_new_tokens": 6144, "stopped_at_eos": eos, "prompts_sha256": "a" * 64,
            "prompt_sha256": "b" * 64, "adapter_tensor_sha256": None if base else "c" * 64,
            "leakcheck_receipt_sha256": "d" * 64, "finish_reason": "stop" if eos else "length"}


def adapter_rows(text=lambda r: f"adapted {r}", capped=(LONGEST[0],), no_eos=()):
    return [line(r, c, text(r), eos=not (c == "full_aware" and (r in capped or r in no_eos)))
            for c in ("ft_only", "full_neutral", "full_aware") for r in IDS]


def base_rows(text=lambda r: f"base {r}"):
    return [line(r, c, text(r)) for c in ("zero_shot", "rag_only") for r in IDS]


IDENTITY = {"model": "mistral", "variant": "gen_canary", "partitions": ["dev"], "subset": "all", "mode": "canary",
            "prompts": {"dev.mistral.jsonl": "e" * 64}, "prompts_sums_sha256": "f" * 64,
            "leakcheck_receipt_sha256": "d" * 64, "ids_sha256": "1" * 64, "window": 32768,
            "gpu_name": "NVIDIA RTX A4000", "engine": "vllm", "pin_set": "main", "cap": None, "sample": 5,
            "engine_settings": {"k": 1, "gpu_memory_utilization": 0.92, "enforce_eager": False},
            "reverse": False, "probe_longest": False, "extend_cap": None, "repeat_longest": None,
            "teacher_force_of": None, "recheck_of": None, "model_path": None,
            "gen_code_sha256": {"v2gpu/generate.py": "7" * 64}, "adapter_config_sha256": None}
HF_SETTINGS = {"model": "mistralai/Mistral-7B-Instruct-v0.2", "quant": "nf4", "dtype": "bfloat16"}   # the HF engine
ADAPTER = {"weights": "s42", "adapter_tensor_sha256": "c" * 64, "adapter_config_sha256": "5" * 64}
BASE = {"weights": "base", "adapter_tensor_sha256": None}


def write_run(d, rows, identity, receipt=None, status="DONE", reverse=False):
    d.mkdir(parents=True)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in (reversed(rows) if reverse else rows))
    (d / "outputs.jsonl").write_text(body)
    rc = {"status": status, "mode": identity["mode"], "identity": identity,
          "pid": int(hashlib.sha256(str(d).encode()).hexdigest()[:6], 16),
          "started_utc": f"2026-09-24T0{len(str(d)) % 10}:00:00+00:00", "outputs_sha256": sha(d / "outputs.jsonl"),
          "result": {"generated": len(rows), "resumed": 0}, **(receipt or {})}
    (d / "receipt-generate.json").write_text(json.dumps(rc))
    return d


def repeat_of(rows, n=10, identical=True):
    f2 = [r for r in rows if r["condition"] == "full_aware"]
    top = min(f2, key=lambda r: (-r["new_tokens"], r["record_id"]))
    return {"key": [top["record_id"], "mistral", "s42", "full_aware", "gen_canary"], "n": n, "identical": identical,
            "new_tokens": top["new_tokens"]}


def hf_of(rows, rate=1.0, a_sha=None, pairs=None):
    """What generate --teacher-force-of writes; `pairs` = [(positions, agree)] per selected pair (default: positions =
    the pair's new_tokens in A, as teacher forcing scores every generated id, and agree = int(positions * rate))."""
    sel = generate.select_hf_pairs(rows)
    new = {row_key(r): r["new_tokens"] for r in rows}
    pairs = pairs or [(new[tuple(k)], int(new[tuple(k)] * rate)) for k in sel["keys"]]
    positions, agree = sum(n for n, _ in pairs), sum(a for _, a in pairs)
    return {**sel, "pairs": [{"key": k, "positions": n, "agree": a, "first_divergence": None}
                             for k, (n, a) in zip(sel["keys"], pairs, strict=True)], "positions": positions,
            "agree": agree, "rate": agree / positions if positions else 0.0, "of_outputs_sha256": a_sha}


def write_ids(workdir, rows, text=None):
    """canary_ids.jsonl as generate writes it in canary mode: the generated ids of every pair (here: the training
    tokenizer's answer ids of each line's text, or of `text(row)`), so a re-decode gives the text back."""
    from v2gpu.models import MODELS, load_tokenizer
    from v2gpu.runs import row_key
    from v2gpu.serialize import answer_ids

    tok = load_tokenizer(MODELS["mistral"])
    (workdir / "canary_ids.jsonl").write_text("".join(json.dumps({"key": list(row_key(r)), "new_ids": answer_ids(
        tok, [{"role": "user", "content": "q"}], (text or (lambda r: r["text"]))(r))}) + "\n" for r in rows))


def model_prompts(d, longest=IDS[7], extra=0):
    """One mistral dev F2 prompt file (the longest prompt, about 170 tokens, is IDS[7]'s unless `longest` says
    otherwise; `extra` short dev records that the canary runs do not hold); the verdict tokenizes it."""
    d.mkdir(parents=True, exist_ok=True)
    order = [r for r in IDS if r != longest] + [longest]
    text = {r: f"Given step {r} " + "then " * 20 * i for i, r in enumerate(order)}
    more = [f"X{j}|x{j}.feature|{j}|s" for j in range(extra)]
    rows = [Example(r, "F2", "mistral", "dev", "p", [{"role": "user", "content": text.get(r, f"Given {r}")}], "A",
                    PROV) for r in IDS + more]
    path = d / "dev.mistral.jsonl"
    path.write_text("".join(json.dumps(asdict(e), sort_keys=True) + "\n" for e in rows))
    return path


class Canary:
    """The six runs and a probe of one model's canary, as generate writes them (identity and receipt fields)."""

    def __init__(self, tmp_path, a_rows=None, b_rows=None, base=None, base_b=None, hf_rate=1.0, repeat=None,
                 probe=None, a_sample=None, longest_f2=IDS[7], settings=None, pin_set="main", hf_pairs=None,
                 extra_dev=0):
        self.root = tmp_path
        self.prompts = model_prompts(tmp_path / "prompts", longest_f2, extra_dev)
        a_rows = adapter_rows() if a_rows is None else a_rows
        if hf_pairs:                     # the scored pairs' positions are their new_tokens in A (and B)
            keys = [tuple(k) for k in generate.select_hf_pairs(a_rows)["keys"]]
            new = dict(zip(keys, (n for n, _ in hf_pairs), strict=True))
            a_rows = [{**r, "new_tokens": new.get(row_key(r), r["new_tokens"])} for r in a_rows]
            b_rows = a_rows if b_rows is None else b_rows
        sample = a_sample or {"rule": "canary", "n": 5, "records": RR, "longest": LONGEST[::-1],
                              "longest_f2": IDS[7], "extension": []}
        pid = {"prompts": {"dev.mistral.jsonl": sha(self.prompts)}, "pin_set": pin_set,
               "engine_settings": {**IDENTITY["engine_settings"], **(settings or {})}}
        ai = {**IDENTITY, **ADAPTER, **pid, "extend_cap": 40}
        self.a = write_run(tmp_path / "a", a_rows, {**ai, "repeat_longest": 10},
                           {"sample": sample, "repeat": repeat or repeat_of(a_rows), "repeat_in_process": True})
        write_ids(self.a, a_rows)
        self.b = write_run(tmp_path / "b", adapter_rows() if b_rows is None else b_rows, {**ai, "reverse": True},
                           {"sample": sample}, reverse=True)
        bi = {**IDENTITY, **BASE, **pid}
        self.base = write_run(tmp_path / "base", base_rows() if base is None else base, bi, {"sample": sample})
        self.base_b = write_run(tmp_path / "base_b", base_rows() if base_b is None else base_b,
                                {**bi, "reverse": True}, {"sample": sample}, reverse=True)
        hi = {**IDENTITY, **ADAPTER, **pid, "engine": "hf", "engine_settings": dict(HF_SETTINGS), "sample": None,
              "teacher_force_of": str(self.a)}
        self.hf = write_run(tmp_path / "hf", [], hi, {"hf_agreement": hf_of(a_rows, hf_rate, sha(
            self.a / "outputs.jsonl"), hf_pairs)})
        (self.hf / "outputs.jsonl").unlink()
        tokens = 2000
        p = {"prompt_tokens": tokens, "max_new_tokens": 32768 - tokens, "new_tokens": 32768 - tokens,
             "finish_reason": "length", "window": 32768, **(probe or {})}
        pi = {**IDENTITY, **ADAPTER, **pid, "sample": None, "probe_longest": True}
        self.probe = write_run(tmp_path / "probe", [], pi, {"probe": p},
                               status="DONE" if p["new_tokens"] == p["max_new_tokens"] else "PROBE_SHORT")
        (self.probe / "outputs.jsonl").unlink()

    def argv(self, *extra, **swap):
        dirs = {"a": self.a, "b": self.b, "base": self.base, "base_b": self.base_b, "hf": self.hf,
                "probe": self.probe, **swap}
        return ["--a", str(dirs["a"]), "--b", str(dirs["b"]), "--base", str(dirs["base"]), "--base-b",
                str(dirs["base_b"]), "--hf", str(dirs["hf"]), "--probe", str(dirs["probe"]),
                "--model-prompts", str(self.prompts), "--out", str(self.root / "v.json"), *extra]

    def verdict(self):
        return json.loads((self.root / "v.json").read_text())


def edit(workdir, change):
    rc = json.loads((workdir / "receipt-generate.json").read_text())
    change(rc)
    (workdir / "receipt-generate.json").write_text(json.dumps(rc))


def test_a_healthy_trained_adapter_passes_every_gate(tmp_path, tokenizers):
    c = Canary(tmp_path)
    assert main(c.argv("--trained")) == 0
    v = c.verdict()
    assert v["pass"] is True and v["model"] == "mistral"
    crit = v["criteria"]
    assert crit["identical"]["pass"] and crit["identical"]["n"] == 24 and crit["identical_base"]["pass"]
    assert crit["eos"] == {"gate": True, "pass": True, "stopped": 5, "n": 5, "min": 4}
    assert crit["adapter_effect"]["differ"] == 5 and crit["probe"]["pass"] and crit["repeat"]["pass"]
    assert crit["hf_agreement"]["gate"] is True and crit["hf_agreement"]["macro_rate"] == 1.0
    assert crit["hf_agreement"]["cap_hit_found"] is True and crit["sample"]["pass"]


@pytest.mark.parametrize("pin_set", ["main", "fallback-tf"])
def test_the_training_tokenizer_decodes_run_a_back_to_its_text(tmp_path, tokenizers, pin_set):
    """24/09: on the laptop, the training venv's tokenizer re-decodes every generated id list of A
    (serialize.decode_answer) and must give each line's text byte for byte; a gate for a pin set that tokenizes with
    other versions than training (fallback-tf), recorded (with the versions and A's outputs sha256) for every set."""
    import importlib.metadata as md

    c = Canary(tmp_path, pin_set=pin_set)
    assert main(c.argv("--trained")) == 0
    v = c.verdict()
    r = v["criteria"]["redecode"]
    assert r["pass"] is True and r["gate"] is (pin_set != "main") and r["lines"] == 24 and r["differ"] == []
    assert r["of_outputs_sha256"] == v["a_outputs_sha256"] == sha(c.a / "outputs.jsonl")
    assert r["versions"] == {"transformers": md.version("transformers"), "tokenizers": md.version("tokenizers")}


@pytest.mark.parametrize("pin_set, code", [("main", 0), ("fallback-tf", 1)])
def test_a_one_character_difference_fails_the_re_decode(tmp_path, tokenizers, pin_set, code):
    rows = adapter_rows()
    c = Canary(tmp_path, pin_set=pin_set)
    write_ids(c.a, rows, text=lambda r: r["text"] + ("!" if (r["record_id"], r["condition"]) == (IDS[2], "ft_only")
                                                       else ""))
    assert main(c.argv("--trained")) == code                         # a gate only for the tf fallback sets
    r = c.verdict()["criteria"]["redecode"]
    assert r["pass"] is False and r["differ"] == [[IDS[2], "mistral", "s42", "ft_only", "gen_canary"]]


def test_a_rerun_that_changes_one_byte_fails(tmp_path, tokenizers):
    c = Canary(tmp_path, b_rows=adapter_rows(text=lambda r: f"adapted {r}" + ("!" if r == IDS[3] else "")))
    assert main(c.argv("--trained")) == 1
    v = c.verdict()["criteria"]["identical"]
    assert v["pass"] is False and len(v["differ"]) == 3                    # ft_only, full_neutral, full_aware of IDS[3]


def test_the_base_path_must_rerun_identically_too(tmp_path, tokenizers):
    c = Canary(tmp_path, base_b=base_rows(text=lambda r: "other" if r == IDS[0] else f"base {r}"))
    assert main(c.argv()) == 1
    assert c.verdict()["criteria"]["identical_base"]["pass"] is False


def test_the_repeat_in_one_process_must_be_identical(tmp_path, tokenizers):
    """The longest-output adapter pair, 10 times in one process, byte-identical."""
    c = Canary(tmp_path, repeat=repeat_of(adapter_rows(), identical=False))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["repeat"]["pass"] is False


def test_eos_gate_needs_four_of_the_five_round_robin_records(tmp_path, tokenizers):
    c = Canary(tmp_path, a_rows=adapter_rows(no_eos=RR[:2]), b_rows=adapter_rows(no_eos=RR[:2]))
    assert main(c.argv("--trained")) == 1
    assert c.verdict()["criteria"]["eos"] == {"gate": True, "pass": False, "stopped": 3, "n": 5, "min": 4}


def test_eos_and_adapter_effect_are_reported_not_gated_for_an_untrained_adapter(tmp_path, tokenizers):
    rows = adapter_rows(no_eos=IDS, text=lambda r: f"base {r}")
    c = Canary(tmp_path, a_rows=rows, b_rows=rows)
    assert main(c.argv()) == 0
    crit = c.verdict()["criteria"]
    assert crit["eos"]["gate"] is False and crit["eos"]["stopped"] == 0 and crit["adapter_effect"]["gate"] is False


def test_an_adapter_that_changes_nothing_fails_when_trained(tmp_path, tokenizers):
    same = adapter_rows(text=lambda r: f"base {r}")
    c = Canary(tmp_path, a_rows=same, b_rows=same)
    assert main(c.argv("--trained")) == 1
    assert c.verdict()["criteria"]["adapter_effect"] == {"gate": True, "pass": False, "differ": 0, "n": 5, "min": 4}


def test_hf_agreement_below_098_fails(tmp_path, tokenizers):
    """decision 24/09: a pre-declared gate, teacher-forced top-1 >= 0.98 over the 5 selected full_aware pairs."""
    c = Canary(tmp_path, hf_rate=0.97)
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["hf_agreement"]["pass"] is False


@pytest.mark.parametrize("pairs, ok", [
    ([(6144, 6113)] + [(600, 570)] * 4, False),   # pooled 98.2 % would pass; the macro mean is 95.9 %
    ([(100, 100), (100, 100), (100, 99), (100, 98), (100, 95)], True),       # every pair >= 0.95, macro 98.4 %
    ([(100, 100)] * 4 + [(100, 94)], False),                                 # macro 98.8 %, one pair below 0.95
])
def test_hf_agreement_gates_every_pair_and_the_macro_mean(tmp_path, tokenizers, pairs, ok):
    """decision 24/09: per model, every one of the 5 pairs >= 0.95 AND the macro (per-pair) mean >= 0.98;
    the pooled rate is report-only, so the long cap-hit pair cannot carry weak short pairs."""
    c = Canary(tmp_path, hf_pairs=pairs)
    assert main(c.argv()) == (0 if ok else 1)
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["pass"] is ok and h["min_pair"] == 0.95 and h["min_macro"] == 0.98
    assert h["pair_rates"] == [a / n for n, a in pairs] and h["pooled_rate"] == sum(a for _, a in pairs) / sum(
        n for n, _ in pairs)


@pytest.mark.parametrize("change", [
    lambda pairs: pairs[:4],                                          # 4 of the 5 selected pairs scored
    lambda pairs: pairs[::-1],                                        # the selected pairs, not in their order
])
def test_hf_agreement_needs_every_selected_pair_scored_in_order(tmp_path, tokenizers, change):
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["hf_agreement"].update(pairs=change(rc["hf_agreement"]["pairs"])))
    assert main(c.argv()) == 1
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["pass"] is False and any("selected pairs" in w for w in h["why"])


@pytest.mark.parametrize("bad", [(100, 130), (100, -1), (0, 0), (100, 95.5), (100.0, 95)])
def test_a_scored_pair_outside_0_to_positions_is_refused(tmp_path, tokenizers, bad):
    """Agree > positions (1.3 here) would lift the macro mean over four pairs at 0.95 to 1.02."""
    c = Canary(tmp_path, hf_pairs=[bad] + [(100, 95)] * 4)
    assert main(c.argv()) == 1
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["pass"] is False and any("0 <= agree <= positions" in w for w in h["why"])


def test_hf_agreement_must_be_of_this_a_run(tmp_path, tokenizers):
    """The HF run scored THIS A (its outputs sha256), not an earlier A with the same pairs."""
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["hf_agreement"].update(of_outputs_sha256="0" * 64))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["hf_agreement"]["pass"] is False


def test_the_count_gates_need_all_five_round_robin_records(tmp_path, tokenizers):
    """4 of 4 is not 4 of 5; a missing round-robin record fails the EOS and adapter-effect gates."""
    rows = [r for r in adapter_rows() if (r["record_id"], r["condition"]) != (RR[0], "full_aware")]
    base = [r for r in base_rows() if (r["record_id"], r["condition"]) != (RR[1], "zero_shot")]
    c = Canary(tmp_path, a_rows=rows, b_rows=rows, base=base, base_b=base)
    assert main(c.argv("--trained")) == 1
    crit = c.verdict()["criteria"]
    assert crit["eos"] == {"gate": True, "pass": False, "stopped": 4, "n": 4, "min": 4}
    assert crit["adapter_effect"] == {"gate": True, "pass": False, "differ": 4, "n": 4, "min": 4}


def test_hf_agreement_must_score_the_selected_pairs_of_this_a_run(tmp_path, tokenizers):
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["hf_agreement"].update(keys=rc["hf_agreement"]["keys"][::-1][:4]))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["hf_agreement"]["pass"] is False


def test_without_a_cap_hit_the_verdict_says_so(tmp_path, tokenizers):
    rows = adapter_rows(capped=())
    c = Canary(tmp_path, a_rows=rows, b_rows=rows)
    assert main(c.argv()) == 0
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["cap_hit_found"] is False and h["note"] == "no cap-hit pair found: the longest output is scored instead"


def test_the_hf_pairs_must_start_at_the_longest_full_aware_dev_prompt(tmp_path, tokenizers):
    """decision 24/09: the first HF pair is the model's longest F2 dev prompt, tokenized here from --model-prompts
    (here IDS[0]'s, which run A does not rank first)."""
    c = Canary(tmp_path, longest_f2=IDS[0])
    assert main(c.argv()) == 1
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["pass"] is False and h["longest_f2"] == IDS[0]
    assert any("longest F2 dev prompt" in w for w in h["why"])


def test_the_verdict_records_the_decision_the_queue_reads(tmp_path, tokenizers):
    """decision 24/09: the memory steps and the pin set are chosen at the canary only; genqueue writes this
    decision into every job command, so the whole run uses what the canary proved."""
    c = Canary(tmp_path, settings={"gpu_memory_utilization": 0.95, "enforce_eager": True}, pin_set="fallback-bnb")
    assert main(c.argv("--trained")) == 0
    v = c.verdict()
    assert v["decision"] == {"gpu_name": "NVIDIA RTX A4000", "pin_set": "fallback-bnb",
                             "gpu_memory_utilization": 0.95, "enforce_eager": True}
    assert v["adapter_tensor_sha256"] == "c" * 64 and v["probed_sums"] == ["f" * 64]


@pytest.mark.parametrize("probe, words", [
    ({"new_tokens": 100}, "new_tokens"),                              # DONE but short
    ({"prompt_tokens": 50, "max_new_tokens": 32718, "new_tokens": 32718}, "longest"),
    ({"max_new_tokens": 4, "new_tokens": 4}, "window"),
])
def test_the_probe_must_fill_the_window_from_the_longest_prompt(tmp_path, tokenizers, probe, words):
    c = Canary(tmp_path, probe=probe)
    assert main(c.argv()) == 1
    p = c.verdict()["criteria"]["probe"]
    assert p["pass"] is False and any(words in x for x in p["why"])


def test_the_probe_must_have_read_every_prompt_file_of_the_model(tmp_path, tokenizers):
    c = Canary(tmp_path)
    other = c.prompts.with_name("test.mistral.jsonl")
    other.write_text(c.prompts.read_text().replace("Given step", "When step"))      # another file, not probed
    argv = c.argv()
    argv[argv.index("--model-prompts") + 1:argv.index("--model-prompts") + 2] = [str(c.prompts), str(other)]
    assert main(argv) == 1 and any("test.mistral.jsonl" in x for x in c.verdict()["criteria"]["probe"]["why"])


def test_the_sample_must_be_the_declared_one(tmp_path, tokenizers):
    """At least 15 adapter pairs, the 3 longest prompts in A and B, extend 40, repeat 10."""
    rows = [r for r in adapter_rows() if r["record_id"] in RR[:4]]
    c = Canary(tmp_path, a_rows=rows, b_rows=rows)
    assert main(c.argv()) == 1
    assert c.verdict()["criteria"]["sample"]["pass"] is False


def test_verdict_refuses_empty_runs_as_identical():
    """Two empty runs are not "identical"."""
    from v2gpu.gencanary import _identical

    assert _identical({}, {})["pass"] is False


REFUSALS = [
    ("the same directory twice", lambda c: c.argv(b=c.a), "distinct"),
    ("a job-mode run", lambda c: (edit(c.b, lambda rc: rc.update(mode="job")), c.argv())[1], "canary"),
    ("another variant", lambda c: (edit(c.base, lambda rc: rc["identity"].update(variant="main")), c.argv())[1],
     "gen_canary"),
    ("another card", lambda c: (edit(c.b, lambda rc: rc["identity"].update(gpu_name="X")), c.argv())[1], "gpu_name"),
    ("another adapter in B", lambda c: (edit(c.b, lambda rc: rc["identity"].update(adapter_tensor_sha256="9" * 64)),
                                         c.argv())[1], "adapter_tensor_sha256"),
    ("a base run as A", lambda c: c.argv(a=c.base), "weights"),
    ("an adapter run as BASE_B", lambda c: (edit(c.base_b, lambda rc: rc["identity"].update(weights="s42")),
                                            c.argv())[1], "base"),
    ("B not reversed", lambda c: (edit(c.b, lambda rc: rc["identity"].update(reverse=False)), c.argv())[1], "reverse"),
    ("A reversed", lambda c: (edit(c.a, lambda rc: rc["identity"].update(reverse=True)), c.argv())[1], "reverse"),
    ("A and B one process", lambda c: (edit(c.b, lambda rc: rc.update(
        pid=json.loads((c.a / "receipt-generate.json").read_text())["pid"],
        started_utc=json.loads((c.a / "receipt-generate.json").read_text())["started_utc"])), c.argv())[1], "process"),
    ("an HF run that is vLLM", lambda c: (edit(c.hf, lambda rc: rc["identity"].update(engine="vllm")), c.argv())[1],
     "hf"),
    ("an unfinished run", lambda c: (edit(c.base, lambda rc: rc.update(status="CRASHED")), c.argv())[1], "DONE"),
    ("a probe that is not a probe", lambda c: (edit(c.probe, lambda rc: rc["identity"].update(probe_longest=False)),
                                               c.argv())[1], "probe"),
    ("a probe with a cap", lambda c: (edit(c.probe, lambda rc: rc["identity"].update(cap=4)), c.argv())[1], "cap"),
    ("A, B and HF all on the base weights", lambda c: ([edit(d, lambda rc: rc["identity"].update(
        weights="base", adapter_tensor_sha256=None)) for d in (c.a, c.b, c.hf)], c.argv())[1], "an adapter run"),
    ("a probe with another memory step", lambda c: (edit(c.probe, lambda rc: rc["identity"]["engine_settings"].update(
        enforce_eager=True)), c.argv())[1], "engine_settings"),
    ("B in another pin set", lambda c: (edit(c.b, lambda rc: rc["identity"].update(pin_set="fallback-bnb")),
                                        c.argv())[1], "pin_set"),
    ("BASE in another pin set", lambda c: (edit(c.base, lambda rc: rc["identity"].update(pin_set="fallback-bnb")),
                                           c.argv())[1], "pin_set"),
]


@pytest.mark.parametrize("what, make, words", REFUSALS, ids=[r[0] for r in REFUSALS])
def test_runs_that_are_not_this_canary_are_refused(tmp_path, tokenizers, what, make, words):
    """The verdict passed vacuously on repeated or mismatched runs; now each is a refusal (exit 2)."""
    c = Canary(tmp_path)
    assert main(make(c)) == 2
    assert any(words in p for p in c.verdict()["problems"]), (what, c.verdict()["problems"])


# Each gate clause has a test that fails without it


def test_b_must_hold_every_pair_of_a(tmp_path, tokenizers):
    rows = adapter_rows()
    c = Canary(tmp_path, b_rows=rows[:-1])
    assert main(c.argv()) == 1
    v = c.verdict()["criteria"]["identical"]
    assert v["pass"] is False and v["same_pairs"] is False and v["differ"] == []


def full_neutral_dropped():
    """13 adapter pairs: every record's full_aware, ft_only of 5 records, no full_neutral."""
    keep = set(IDS[:5])
    return [r for r in adapter_rows() if r["condition"] == "full_aware" or
            (r["condition"] == "ft_only" and r["record_id"] in keep)]


@pytest.mark.parametrize("make, words", [
    (lambda tmp: Canary(tmp, a_rows=full_neutral_dropped(), b_rows=full_neutral_dropped()),
     "a holds 13 adapter pairs, fewer than 15"),
    (lambda tmp: (c := Canary(tmp), edit(c.a, lambda rc: rc["identity"].update(repeat_longest=9)))[0],
     "--repeat-longest 10"),
    (lambda tmp: Canary(tmp, a_sample={"rule": "canary", "n": 4, "records": RR, "longest": LONGEST[::-1],
                                       "longest_f2": IDS[7], "extension": []}), "--sample 5"),
    (lambda tmp: Canary(tmp, a_rows=adapter_rows(capped=()), b_rows=adapter_rows(capped=()), extra_dev=2),
     "no cap hit and a holds 8 records, fewer than min(40, 10)"),
])
def test_each_sample_clause_fails_on_its_own(tmp_path, tokenizers, make, words):
    c = make(tmp_path)
    assert main(c.argv()) == 1
    s = c.verdict()["criteria"]["sample"]
    assert s["pass"] is False and len(s["why"]) == 1 and words in s["why"][0], s["why"]


@pytest.mark.parametrize("change", [
    lambda rep: rep.update(n=9),
    lambda rep: rep.update(key=[IDS[0], "mistral", "s42", "full_aware", "gen_canary"]),
])
def test_the_repeat_must_be_ten_runs_of_the_longest_output(tmp_path, tokenizers, change):
    c = Canary(tmp_path)
    edit(c.a, lambda rc: change(rc["repeat"]))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["repeat"]["pass"] is False


def test_the_repeated_pair_must_come_from_this_process(tmp_path, tokenizers):
    """After a resume the first of the repeated outputs came from another process."""
    c = Canary(tmp_path)
    edit(c.a, lambda rc: rc.update(repeat_in_process=False))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["repeat"]["pass"] is False


def test_hf_agreement_needs_five_pairs(tmp_path, tokenizers):
    """Four full_aware pairs in A: the selection has 4 keys and the HF receipt scored exactly those 4."""
    rows = [r for r in adapter_rows() if r["condition"] != "full_aware" or r["record_id"] in (IDS[7], IDS[5], *IDS[:2])]
    c = Canary(tmp_path, a_rows=rows, b_rows=rows)
    assert main(c.argv()) == 1
    h = c.verdict()["criteria"]["hf_agreement"]
    assert h["pass"] is False and any("4 pairs" in w for w in h["why"]), h["why"]


def test_a_run_file_changed_after_its_receipt_is_refused(tmp_path, tokenizers):
    """A's outputs.jsonl is tied to its receipt; an A tampered after the run (receipt and HF still name the
    old sha) is refused, as is a receipt without outputs_sha256."""
    c = Canary(tmp_path)
    out = c.a / "outputs.jsonl"
    out.write_text(out.read_text().replace("adapted", "Adapted", 1))
    assert main(c.argv()) == 2 and any("a:" in p and "outputs_sha256" in p for p in c.verdict()["problems"])
    c2 = Canary(tmp_path / "2")
    edit(c2.base_b, lambda rc: rc.pop("outputs_sha256"))
    assert main(c2.argv()) == 2 and any("base_b:" in p for p in c2.verdict()["problems"])


def test_hf_must_have_scored_the_file_of_a(tmp_path, tokenizers):
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["hf_agreement"].update(of_outputs_sha256=None))
    edit(c.a, lambda rc: rc.update(outputs_sha256=sha(c.a / "outputs.jsonl")))
    assert main(c.argv()) == 1
    assert any("of_outputs_sha256" in w for w in c.verdict()["criteria"]["hf_agreement"]["why"])


def test_an_unreadable_receipt_is_a_refusal_and_no_older_verdict_survives(tmp_path, tokenizers, monkeypatch):
    """--out is deleted first; a missing receipt is a refusal (exit 2), a crash leaves no verdict."""
    c = Canary(tmp_path)
    (tmp_path / "v.json").write_text(json.dumps({"pass": True}))
    (c.base / "receipt-generate.json").unlink()
    assert main(c.argv()) == 2 and any("receipt" in p for p in c.verdict()["problems"])
    c2 = Canary(tmp_path / "2")
    (tmp_path / "2" / "v.json").write_text(json.dumps({"pass": True}))
    monkeypatch.setattr(gencanary, "longest_prompt", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        main(c2.argv())
    assert not (tmp_path / "2" / "v.json").exists()


def test_a_duplicate_line_in_a_run_is_refused(tmp_path, tokenizers):
    """A second copy of a line would vanish in the {key: line} map."""
    c = Canary(tmp_path)
    out = c.b / "outputs.jsonl"
    out.write_text(out.read_text() + out.read_text().splitlines(keepends=True)[0])
    edit(c.b, lambda rc: rc.update(outputs_sha256=sha(out)))
    assert main(c.argv()) == 2 and any("duplicate" in p for p in c.verdict()["problems"])


def test_the_macro_mean_is_exact(tmp_path, tokenizers):
    """These five rates average exactly 0.98 (the sum of floats gives 0.9799999999999999)."""
    pairs = [(300, 295), (300, 298), (60, 59), (200, 198), (140, 133)]
    c = Canary(tmp_path, hf_pairs=pairs)
    assert main(c.argv()) == 0 and c.verdict()["criteria"]["hf_agreement"]["pass"] is True


def test_a_scored_pair_must_cover_every_generated_token(tmp_path, tokenizers):
    """Positions is the pair's new_tokens in A (teacher forcing scores every generated id)."""
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["hf_agreement"]["pairs"][1].update(positions=3, agree=3))
    assert main(c.argv()) == 1
    assert any("new_tokens" in w for w in c.verdict()["criteria"]["hf_agreement"]["why"])


@pytest.mark.parametrize("change, words", [
    (lambda h: h.update(keys=h["keys"][::-1]), "keys"),
    (lambda h: h.update(of_outputs_sha256="0" * 64), "of_outputs_sha256"),
    (lambda h: h["pairs"][2].update(agree=h["pairs"][2]["positions"] - 1), "0.95"),
])
def test_hf_agreement_names_every_cause(tmp_path, tokenizers, change, words):
    """The verdict says why the gate failed."""
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: change(rc["hf_agreement"]))
    assert main(c.argv()) == 1
    assert any(words in w for w in c.verdict()["criteria"]["hf_agreement"]["why"])


def test_the_macro_floor_is_named_too(tmp_path, tokenizers):
    c = Canary(tmp_path, hf_pairs=[(100, 96)] * 5)                 # every pair >= 0.95, macro 0.96 < 0.98
    assert main(c.argv()) == 1
    assert any("0.98" in w for w in c.verdict()["criteria"]["hf_agreement"]["why"])


def test_the_redecode_record_cannot_switch_its_own_gate_off(tmp_path, tokenizers, monkeypatch):
    """The gate is set after the record's own keys."""
    monkeypatch.setattr(gencanary, "redecode", lambda *a: {"pass": False, "gate": False, "lines": 0})
    c = Canary(tmp_path, pin_set="fallback-tf")
    assert main(c.argv("--trained")) == 1 and c.verdict()["criteria"]["redecode"]["gate"] is True


def test_the_three_longest_records_come_from_the_prompt_files(tmp_path, tokenizers):
    """A's receipt may name any three records; the verdict computes the three longest from --model-prompts."""
    rows = [r for r in adapter_rows() if r["record_id"] != IDS[6]]
    c = Canary(tmp_path, a_rows=rows, b_rows=rows, a_sample={"rule": "canary", "n": 5, "records": RR,
                                                            "longest": IDS[:3], "longest_f2": IDS[7], "extension": []})
    assert main(c.argv()) == 1
    s = c.verdict()["criteria"]["sample"]
    assert s["pass"] is False and any("longest-prompt records" in w for w in s["why"])
    assert s["longest_records"] == [IDS[7], IDS[6], IDS[5]]


def test_a_probe_that_is_not_done_fails_even_at_full_length(tmp_path, tokenizers):
    """The probe clause status != DONE (a PROBE_SHORT receipt is allowed through the refusals)."""
    c = Canary(tmp_path)
    edit(c.probe, lambda rc: rc.update(status="PROBE_SHORT"))
    assert main(c.argv()) == 1 and c.verdict()["criteria"]["probe"]["pass"] is False



@pytest.mark.parametrize("change, words", [
    (lambda c: [edit(d, lambda rc: rc["identity"].update(cap=512)) for d in (c.a, c.b, c.base, c.base_b)], "cap"),
    (lambda c: edit(c.hf, lambda rc: rc["identity"].update(cap=512)), "hf: run with --cap"),
    (lambda c: [edit(d, lambda rc: rc["identity"].update(model_path="/tmp/snapshot")) for d in (c.a, c.b)],
     "--model-path"),
])
def test_a_canary_run_must_be_the_job_engine(tmp_path, tokenizers, change, words):
    """A canary with every run capped at 512 used to pass with --trained; a capped or local-snapshot run
    is not the job's engine, so its A/B, cap-hit and HF evidence says nothing about the real run."""
    c = Canary(tmp_path)
    change(c)
    assert main(c.argv("--trained")) == 2
    assert any(words in p for p in c.verdict()["problems"]), c.verdict()["problems"]


def test_every_line_follows_the_budget_rule_of_the_model_window(tmp_path, tokenizers):
    """Generate checks the budget rule in job mode only; the canary lines are checked here."""
    rows = [{**r, "max_new_tokens": 512} if r["record_id"] == IDS[0] else r for r in adapter_rows()]
    c = Canary(tmp_path, a_rows=rows, b_rows=rows)
    assert main(c.argv()) == 2
    assert any("budget" in p and IDS[0] in p for p in c.verdict()["problems"]), c.verdict()["problems"]


@pytest.mark.parametrize("run", ["b", "base_b"])
@pytest.mark.parametrize("resumed, missing", [(None, None), (1, 0), (0, 1)])
def test_the_rerun_must_have_generated_every_line_itself(tmp_path, tokenizers, run, resumed, missing):
    """A B seeded with A's outputs resumes every pair and looks identical; it must have generated them all. Each
    clause alone (resumed 0, and generated equal to its lines) refuses."""
    c = Canary(tmp_path)
    n = len((getattr(c, run) / "outputs.jsonl").read_text().splitlines())
    result = {"generated": 0, "resumed": n} if resumed is None else {"generated": n - missing, "resumed": resumed}
    edit(getattr(c, run), lambda rc: rc.update(result=result))
    assert main(c.argv("--trained")) == 2
    assert any(f"{run}:" in p and "generated" in p for p in c.verdict()["problems"])


@pytest.mark.parametrize("settings", [{**HF_SETTINGS, "quant": "none"}, {**HF_SETTINGS, "dtype": "float32"},
                                      {**HF_SETTINGS, "model": "/tmp/other"}])
def test_the_hf_reference_is_the_nf4_bf16_hub_model(tmp_path, tokenizers, settings):
    """An HF run with quant none, float32 and another model path used to pass."""
    c = Canary(tmp_path)
    edit(c.hf, lambda rc: rc["identity"].update(engine_settings=settings))
    assert main(c.argv("--trained")) == 2 and any("hf:" in p and "nf4" in p for p in c.verdict()["problems"])


def test_an_out_inside_a_run_directory_is_refused_before_anything_is_deleted(tmp_path, tokenizers):
    """--out is deleted first; an --out that names a run's file (or a prompt file) would destroy evidence."""
    c = Canary(tmp_path)
    for target in (c.a / "receipt-generate.json", c.probe / "verdict.json", c.prompts):
        kept = {p: p.read_bytes() for p in (c.a / "receipt-generate.json", c.prompts)}
        argv = c.argv()
        argv[argv.index("--out") + 1] = str(target)
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2 and {p: p.read_bytes() for p in kept} == kept


def test_the_extend_cap_clause_fails_on_its_own(tmp_path, tokenizers):
    """A and B run with --extend-cap 39 (equal, so no refusal): the sample gate names the flags."""
    c = Canary(tmp_path)
    for d in (c.a, c.b):
        edit(d, lambda rc: rc["identity"].update(extend_cap=39))
    assert main(c.argv()) == 1
    s = c.verdict()["criteria"]["sample"]
    assert s["pass"] is False and len(s["why"]) == 1 and "--extend-cap 40" in s["why"][0]


def test_a_receipt_that_is_not_an_object_is_a_refusal(tmp_path, tokenizers):
    """A receipt that is not a JSON object is a refusal, not a crash."""
    c = Canary(tmp_path)
    (c.base / "receipt-generate.json").write_text("[]")
    assert main(c.argv()) == 2 and any("not a JSON object" in p for p in c.verdict()["problems"])


# `gencanary decide`, one decision for the variant (added 24/09)

DECISION = {"gpu_name": "NVIDIA RTX A4000", "pin_set": "main", "gpu_memory_utilization": 0.92, "enforce_eager": False}
GEN = {"v2gpu/generate.py": "7" * 64}
ENGINE_SETTINGS = {"batch_invariant": False, "gpu_memory_utilization": 0.92, "enforce_eager": False}
FINGERPRINT = {"name": "vllm", "version": "0.27.1"}


class Decide:
    """Passing canary verdicts (untrained by default) of every model main plans, and their test/aux probe digest."""

    def __init__(self, tmp_path, models=("codellama", "deepseek", "mistral")):
        self.root = tmp_path
        self.sums = tmp_path / "pt" / "PROMPTS_SHA256SUMS"
        self.sums.parent.mkdir()
        self.sums.write_text("0" * 64 + "  test.deepseek.jsonl\n")
        self.paths = {m: self.verdict(m) for m in models}

    def verdict(self, model, decision=None, **change):
        path = self.root / f"verdict.{model}.json"
        path.write_text(json.dumps({"pass": True, "trained": False, "model": model, "probed_sums": [sha(self.sums)],
                                    "decision": {**DECISION, **(decision or {})},
                                    "criteria": {"hf_agreement": {"pass": True}}, "gen_code_sha256": GEN,
                                    "engine_settings": {**ENGINE_SETTINGS, **{k: v for k, v in (decision or {}).items()
                                                                               if k in ENGINE_SETTINGS}},
                                    "engine_fingerprint": FINGERPRINT, **change}))
        return path

    def argv(self, variant="main", sums=True, paths=None):
        paths = self.paths if paths is None else paths
        return ["decide", "--variant", variant, "--verdicts", *(f"{m}={p}" for m, p in paths.items()),
                *(["--prompts-sums", str(self.sums)] if sums else []), "--out", str(self.root / "decision.json")]

    def out(self):
        return json.loads((self.root / "decision.json").read_text())


def test_identical_decisions_of_every_planned_model_make_the_variants_decision(tmp_path):
    d = Decide(tmp_path)
    assert main(d.argv()) == 0
    out = d.out()
    assert out["pass"] is True and out["variant"] == "main" and out["decision"] == DECISION
    assert out["schema"] == "gtestgen.v2.gen_decision.v2"                  # bound to code and engine
    assert out["verdicts"] == {m: {"verdict": str(p), "sha256": sha(p), "trained": False} for m, p in d.paths.items()}


@pytest.mark.parametrize("decisions, target, rerun", [
    ({"codellama": {"gpu_memory_utilization": 0.95}}, {"gpu_memory_utilization": 0.95}, ["deepseek", "mistral"]),
    ({"codellama": {"pin_set": "fallback-tf"}, "mistral": {"pin_set": "fallback-bnb"}},
     {"pin_set": "fallback-tf-bnb"}, ["codellama", "deepseek", "mistral"]),
    ({"deepseek": {"gpu_memory_utilization": 0.95, "enforce_eager": True}, "mistral": {"gpu_memory_utilization": 0.95}},
     {"gpu_memory_utilization": 0.95, "enforce_eager": True}, ["codellama", "mistral"]),
])
def test_a_mismatch_is_refused_naming_the_common_target_and_the_models_to_rerun(tmp_path, capsys, decisions, target,
                                                                                rerun):
    """decision 24/09: the target is the most conservative decision covering every model's own; the listed models rerun
    their canary at it (fresh workdirs), then decide runs again. The tool never picks max_num_batched_tokens."""
    d = Decide(tmp_path)
    for m, change in decisions.items():
        d.verdict(m, change)
    assert main(d.argv()) == 2
    out, err = d.out(), capsys.readouterr().err
    assert out["pass"] is False and "decision" not in out
    assert out["target"] == {**DECISION, **target} and out["rerun"] == rerun
    assert all(m in err for m in rerun) and "target" in err and "max_num_batched_tokens" not in json.dumps(out)
    for m in rerun:                                       # the listed models rerun at the target: now identical
        d.verdict(m, target)
    for m in set(d.paths) - set(rerun):
        d.verdict(m, target)                              # (a model already at the target reruns nothing)
    assert main(d.argv()) == 0 and d.out()["decision"] == {**DECISION, **target}


@pytest.mark.parametrize("case", ["missing", "failing", "other model", "unplanned", "no probe", "no sums"])
def test_every_model_the_variant_plans_needs_a_passing_verdict_that_probed_its_prompts(tmp_path, capsys, case):
    d = Decide(tmp_path)
    paths, sums = dict(d.paths), True
    if case == "missing":
        del paths["mistral"]
    elif case == "failing":
        d.verdict("mistral", **{"pass": False})
    elif case == "other model":
        paths["mistral"] = d.paths["deepseek"]
    elif case == "unplanned":
        paths = {"deepseek": d.paths["deepseek"], "codellama": d.paths["codellama"]}
    elif case == "no probe":
        d.verdict("mistral", probed_sums=[])
    else:
        sums = False
    variant = "dev_pilot_v0" if case == "unplanned" else "main"
    assert main(d.argv(variant, sums, paths)) == 2
    assert "codellama" in capsys.readouterr().err if case == "unplanned" else d.out()["pass"] is False


def test_a_dev_variant_decides_on_deepseek_alone(tmp_path):
    d = Decide(tmp_path, models=("deepseek",))
    assert main(d.argv("dev_pilot_v0", sums=False)) == 0 and d.out()["variant"] == "dev_pilot_v0"
    (tmp_path / "g").mkdir()
    every = Decide(tmp_path / "g")
    assert main(every.argv("gen_canary", sums=False)) == 2 and "gen_canary" in every.out()["problems"][0]


def test_decide_refuses_an_out_that_is_one_of_its_verdicts(tmp_path):
    d = Decide(tmp_path)
    argv = d.argv()
    argv[argv.index("--out") + 1] = str(d.paths["deepseek"])
    kept = d.paths["deepseek"].read_bytes()
    with pytest.raises(SystemExit):
        main(argv)
    assert d.paths["deepseek"].read_bytes() == kept


# The canary is bound to the code and the engine it ran


def test_the_verdict_records_the_generation_code_and_the_engine_it_ran(tmp_path, tokenizers):
    from v2gpu.engines import declared_engine, engine_fingerprint

    c = Canary(tmp_path)
    engine = {"name": "vllm", "version": "0.27.1", "kwargs": {"gpu_memory_utilization": 0.92}, "env": {"E": "0"}}
    for d in (c.a, c.b, c.base, c.base_b):                   # one measured engine
        edit(d, lambda rc: rc.update(engine=engine))
    for d in (c.a, c.b, c.base, c.base_b, c.probe):          # what names the model is left out of the binding
        edit(d, lambda rc: rc["identity"]["engine_settings"].update(revision="r", max_model_len=32768))
    assert main(c.argv("--trained")) == 0
    v = c.verdict()
    assert v["gen_code_sha256"] == IDENTITY["gen_code_sha256"]
    assert v["engine_settings"] == declared_engine(IDENTITY["engine_settings"])
    assert v["engine_fingerprint"] == engine_fingerprint(engine)


def test_runs_of_other_generation_code_are_refused(tmp_path, tokenizers):
    c = Canary(tmp_path)
    edit(c.b, lambda rc: rc["identity"].update(gen_code_sha256={"v2gpu/generate.py": "8" * 64}))
    assert main(c.argv()) == 2 and any("gen_code_sha256" in p for p in c.verdict()["problems"])


def test_a_repeated_probe_flag_keeps_every_probe(tmp_path, tokenizers):
    """`--probe DEV --probe TEST` (the form used for main) reads both directories."""
    import shutil

    c = Canary(tmp_path)
    second = tmp_path / "probe_test"
    shutil.copytree(c.probe, second)
    edit(second, lambda rc: rc["identity"].update(prompts_sums_sha256="9" * 64))
    assert main(c.argv("--trained", "--probe", str(second))) == 0
    assert c.verdict()["probed_sums"] == sorted(["f" * 64, "9" * 64])


def test_decide_binds_one_code_and_one_engine_and_copies_them(tmp_path):
    d = Decide(tmp_path)
    assert main(d.argv()) == 0
    out = d.out()
    assert out["schema"] == "gtestgen.v2.gen_decision.v2"
    assert out["gen_code_sha256"] == GEN and out["engine_settings"] == ENGINE_SETTINGS
    assert out["engine_fingerprint"] == FINGERPRINT
    for m in d.paths:                                            # no verdict names its code: no binding
        d.verdict(m, gen_code_sha256=None)
    assert main(d.argv()) == 2 and d.out()["pass"] is False
    for m in d.paths:
        d.verdict(m)
    for change in ({"gen_code_sha256": {"v2gpu/generate.py": "8" * 64}}, {"gen_code_sha256": None},
                   {"engine_settings": {**ENGINE_SETTINGS, "batch_invariant": True}},
                   {"engine_fingerprint": {**FINGERPRINT, "version": "0.28.0"}}):
        d.verdict("mistral", **change)
        assert main(d.argv()) == 2 and d.out()["pass"] is False, change
        d.verdict("mistral")


# A verdict whose engine record names no engine binds nothing

EMPTY_FINGERPRINT = {"name": None, "version": None, "quantization": None, "batch_invariant": None, "env": {},
                     "tuned_config_files": None, "sampling": {}, "kwargs": {}, "effective": {}, "lora_shrink": []}


def test_runs_without_an_engine_record_give_no_fingerprint(tmp_path, tokenizers):
    """A receipt without its engine record (none of these runs names one) gave the dict of None values; now the
    verdict's engine_fingerprint is null, which decide refuses."""
    c = Canary(tmp_path)
    assert main(c.argv("--trained")) == 0 and c.verdict()["engine_fingerprint"] is None


@pytest.mark.parametrize("fingerprint", [EMPTY_FINGERPRINT, {"name": "vllm"}, None])
def test_decide_refuses_verdicts_whose_engine_is_not_named(tmp_path, fingerprint):
    """Equal is not enough: the same nameless fingerprint in every verdict binds no engine (non-null check)."""
    d = Decide(tmp_path)
    for m in d.paths:
        d.verdict(m, engine_fingerprint=fingerprint)
    assert main(d.argv()) == 2 and d.out()["pass"] is False
    assert any("engine_fingerprint" in p for p in d.out()["problems"])


# Mutants that once survived: the cap clause per run and the budget rule


@pytest.mark.parametrize("run", ["base", "base_b"])
def test_a_capped_base_run_is_refused_on_its_own(tmp_path, tokenizers, run):
    """The cap clause per run, the lines unchanged (so the budget rule cannot catch it)."""
    c = Canary(tmp_path)
    edit(getattr(c, run), lambda rc: rc["identity"].update(cap=4))
    assert main(c.argv()) == 2 and any(f"{run}: run with --cap" in p for p in c.verdict()["problems"])


@pytest.mark.parametrize("runs", [("b",), ("base", "base_b")])
def test_the_budget_rule_is_checked_on_every_run_not_only_a(tmp_path, tokenizers, runs):
    """A untouched; B (or BASE and BASE_B together, so their byte gate still agrees) break the budget
    rule: a refusal, not a gate failure."""
    rows = {"b": adapter_rows(), "base": base_rows(), "base_b": base_rows()}
    for n in runs:
        rows[n] = [{**r, "max_new_tokens": 512} if r["record_id"] == IDS[0] else r for r in rows[n]]
    c = Canary(tmp_path, b_rows=rows["b"], base=rows["base"], base_b=rows["base_b"])
    assert main(c.argv()) == 2
    assert all(any(p.startswith(f"{n}: lines break the budget rule") for p in c.verdict()["problems"]) for n in runs)


def test_decide_refuses_an_off_ladder_decision_even_when_every_model_shares_it(tmp_path):
    """0.92 with enforce_eager is not a step of the ladder, whatever the models agree on."""
    d = Decide(tmp_path)
    for m in d.paths:
        d.verdict(m, {"enforce_eager": True})
    assert main(d.argv()) == 2 and any("memory step" in p for p in d.out()["problems"])


# The adapter config and the binding of the verdict


@pytest.mark.parametrize("run", ["b", "hf"])
def test_the_adapter_runs_share_the_adapter_config(tmp_path, tokenizers, run):
    c = Canary(tmp_path)
    edit(getattr(c, run), lambda rc: rc["identity"].update(adapter_config_sha256="6" * 64))
    assert main(c.argv()) == 2 and any("adapter_config_sha256" in p for p in c.verdict()["problems"])


def test_an_adapter_run_without_its_config_digest_is_refused(tmp_path, tokenizers):
    c = Canary(tmp_path)
    for d in (c.a, c.b, c.hf, c.probe):
        edit(d, lambda rc: rc["identity"].update(adapter_config_sha256=None))
    assert main(c.argv()) == 2 and any("adapter_config_sha256" in p for p in c.verdict()["problems"])


def test_the_verdict_records_the_adapter_config(tmp_path, tokenizers):
    c = Canary(tmp_path)
    assert main(c.argv("--trained")) == 0 and c.verdict()["adapter_config_sha256"] == "5" * 64


@pytest.mark.parametrize("run", ["b", "base", "base_b"])
def test_the_vllm_runs_share_one_measured_engine(tmp_path, tokenizers, run):
    """The engine fingerprint (vLLM version, effective config, shrink configs...) is one for A, B, BASE and BASE_B."""
    c = Canary(tmp_path)
    engine = {"name": "vllm", "version": "0.27.1", "kwargs": {"gpu_memory_utilization": 0.92}}
    for d in (c.a, c.b, c.base, c.base_b):
        edit(d, lambda rc: rc.update(engine=engine))
    assert main(c.argv("--trained")) == 0
    edit(getattr(c, run), lambda rc: rc.update(engine={**engine, "version": "0.28.0"}))
    assert main(c.argv("--trained")) == 2 and any("engine fingerprint" in p for p in c.verdict()["problems"])
