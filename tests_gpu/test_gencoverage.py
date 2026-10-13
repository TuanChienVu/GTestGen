# tests_gpu/test_gencoverage.py
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from v2gpu.audit import code_sha256
from v2gpu.engines import GEN_PINS, PIN_SETS, engine_fingerprint
from v2gpu.examples import Example, jsonl_lines
from v2gpu.gencoverage import main
from v2gpu.runs import planned_runs, prompt_sha256

PROV = {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}
V0 = {"variant": "v0", "k": 8, "n_examples": 3, "ratio": 0.5, "budget_examples": 1536, "budget_knowledge": 1536,
      "mode": "full"}                                   # the compose of the prompt directory and of every adapter
RECORDS = {"test": ["Sylius/Sylius|a.feature|1|s", "apache/fineract|b.feature|2|s"],
           "aux_test": ["renalware/renalware-core|c.feature|3|s"]}
EXCLUDED = [{"record_id": "apache/fineract|b.feature|2|s", "fmt": "F2", "model": None, "reason": "too_long_test_rule"}]
ENGINE = {"name": "vllm", "version": "0.27.1", "quantization": "bitsandbytes", "batch_invariant": False,
          "env": {"VLLM_BATCH_INVARIANT": "0"}, "sampling": {"temperature": 0.0, "stop_token_ids": [2]},
          "kwargs": {"model": "m", "revision": "r", "max_model_len": 16384, "max_num_seqs": 1,
                     "gpu_memory_utilization": 0.92, "enforce_eager": False},
          "effective": {"attention_backend": "FLASH_ATTN", "cudagraph_mode": "FULL_AND_PIECEWISE",
                        "vllm_config_sha256": "x"},
          "lora_shrink": {"checked": [{"num_slices": 1, "m": 1, "k": 4096, "config": {"split_k": 1}}]}}
GEN_CODE = {"v2gpu/generate.py": "1" * 64, "v2gpu/engines.py": "2" * 64}     # identity.gen_code_sha256
SETTINGS = {"quantization": "bitsandbytes", "gpu_memory_utilization": 0.92, "enforce_eager": False,
            "batch_invariant": False, "env": {"VLLM_BATCH_INVARIANT": "0"}}      # identity.engine_settings, declared
DECISION = {"gpu_name": "NVIDIA RTX A4000", "pin_set": "main", "gpu_memory_utilization": 0.92, "enforce_eager": False}
MODELS3 = ("codellama", "deepseek", "mistral")


SUMS_SHA = "5" * 64                                     # the PROMPTS_SHA256SUMS digest main's canaries probed


def verdict_text(model, decision=DECISION, **change):
    """A passing canary verdict of the model, as gencanary writes one (the parts coverage reads,
    with the code and engine it binds and the prompt directories it probed)."""
    return json.dumps({"model": model, "pass": True, "trained": False, "decision": decision,
                       "gen_code_sha256": GEN_CODE, "engine_settings": SETTINGS,
                       "engine_fingerprint": engine_fingerprint(ENGINE), "probed_sums": [SUMS_SHA], **change},
                      sort_keys=True)


def verdict_name(model, decision=DECISION):
    """Its path, relative to the directory decide ran in (each test runs in its tmp_path: _in_tmp_path)."""
    return f"verdict.{model}.{hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()[:8]}.json"


def write_verdicts(root, decision=DECISION):
    for m in MODELS3:
        (root / verdict_name(m, decision)).write_text(verdict_text(m, decision))


def decision_doc(decision=DECISION, **extra):
    """The decision file decide writes from the three verdicts (bound to code and engine)."""
    return json.dumps({"schema": "gtestgen.v2.gen_decision.v2", "pass": True, "variant": "main", "decision": decision,
                       "gen_code_sha256": GEN_CODE, "engine_settings": SETTINGS,
                       "verdicts": {m: {"verdict": verdict_name(m, decision), "trained": False,
                                        "sha256": hashlib.sha256(verdict_text(m, decision).encode()).hexdigest()}
                                    for m in MODELS3},
                       "engine_fingerprint": engine_fingerprint(ENGINE), "prompts_sums_sha256": SUMS_SHA, **extra})


DECISION_DOC = decision_doc()
DECISION_SHA = hashlib.sha256(DECISION_DOC.encode()).hexdigest()


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    """decide records each verdict's path as given, relative to where it ran; coverage runs there too."""
    monkeypatch.chdir(tmp_path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prompts(d, excluded=EXCLUDED):
    """One prompt file per model x partition, all formats, answer null; excluded pairs have no prompt."""
    d.mkdir()
    files = []
    for part, ids in RECORDS.items():
        for model in ("codellama", "deepseek", "mistral"):
            skip = {(e["record_id"], e["fmt"]) for e in excluded if e["model"] in (None, model)}
            rows = [Example(r, f, model, part, "p", [{"role": "user", "content": f"{model} {f} {r}"}], None, PROV)
                    for r in ids for f in ("F0", "F1", "F2") if (r, f) not in skip]
            path = d / f"{part}.{model}.jsonl"
            path.write_text("".join(json.dumps(asdict(e), sort_keys=True) + "\n" for e in rows))
            files.append(path)
    (d / "excluded.jsonl").write_text("".join(json.dumps(e) + "\n" for e in excluded))
    (d / "PROMPTS_META.json").write_text(json.dumps({"compose": V0, "files": sorted(p.name for p in files)}))
    names = sorted(p.name for p in d.iterdir())
    (d / "PROMPTS_SHA256SUMS").write_text("".join(f"{sha(d / n)}  {n}\n" for n in names))
    return files


def adapter_sha(model, seed):
    return hashlib.sha256(f"{model}/{seed}".encode()).hexdigest()


def leak_receipts(d):
    """One PASS laptop audit receipt per model, made by this checkout's code."""
    d.mkdir()
    out = {}
    for model in ("codellama", "deepseek", "mistral"):
        path = d / f"leakcheck-receipt.{model}.json"
        path.write_text(json.dumps({"schema": "gtestgen.v2.leakcheck_receipt.v1", "status": "PASS", "n_problems": 0,
                                    "model": model, "subset": "all", "code_sha256": code_sha256(),
                                    "by_partition": {"test": 1, "aux_test": 1}}))
        out[model] = path
    return out


TRAIN_CODE_SHA256 = {"v2gpu/train_job.py": "a" * 64}         # one value shared by every adapter
TRAIN_CFG_SHA256 = "b" * 64


TRAIN_CONFIG = {"upcast_fp32": True}   # mandatory for every real adapter
TRAIN_GPU = "NVIDIA RTX A6000"          # the training card, for coverage.json


def train_receipts(d, gpu=TRAIN_GPU):
    d.mkdir()
    paths = []
    for model, seeds in (("codellama", (42, 11, 23)), ("deepseek", (42, 11, 23, 7, 99)), ("mistral", (42, 11, 23))):
        for seed in seeds:
            path = d / f"{model}-s{seed}.json"
            path.write_text(json.dumps({"status": "DONE", "mode": "job", "seed": seed, "subset": "all",
                                        "model": {"key": model},
                                        "selected_adapter": {"tensor_sha256": adapter_sha(model, f"s{seed}")},
                                        "data": {"prompts": {"meta": {"compose": V0}}},    # its compose
                                        "train_code_sha256": TRAIN_CODE_SHA256,
                                        "train_cfg_sha256": TRAIN_CFG_SHA256,
                                        "train_config": TRAIN_CONFIG,
                                        "environment": {"device": {"name": gpu}} if gpu else {}}))
            paths.append(path)
    return paths


def output_rows(d, leaks, drop=(), excluded=EXCLUDED):
    """A valid output line for every planned pair of variant main (the generation itself is not under test here)."""
    out = []
    for part, ids in RECORDS.items():
        for run in planned_runs("main", part):
            skip = {(e["record_id"], e["fmt"]) for e in excluded if e["model"] in (None, run.model)}
            for r in ids:
                if (r, run.fmt) in skip or (r, run.condition, run.model, run.seed) in drop:
                    continue
                capped = not r.startswith("Sylius")
                out.append({"record_id": r, "model": run.model, "seed": run.seed, "condition": run.condition,
                            "variant": "main", "fmt": run.fmt, "text": "// FILE: x", "prompt_tokens": 10,
                            "new_tokens": 6144 if capped else 3, "max_new_tokens": 6144, "stopped_at_eos": not capped,
                            "prompts_sha256": sha(d / f"{part}.{run.model}.jsonl"),
                            "prompt_sha256": prompt_sha256([{"role": "user", "content": f"{run.model} {run.fmt} {r}"}]),
                            "adapter_tensor_sha256": None if run.seed == "base" else adapter_sha(run.model, run.seed),
                            "leakcheck_receipt_sha256": sha(leaks[run.model]),
                            "finish_reason": "length" if capped else "stop"})
    return out


def write_run(d, name, rows, identity=None, receipt=None, pod="pod-1"):
    """A copied-home workdir: outputs.jsonl and a DONE job-mode vLLM receipt. identity.weights (for
    the run manifest: gencoverage.gens_by_pair keys on (model, weights), the "seed" of the job's own output rows)
    defaults
    to the rows' own seed, exactly what genqueue names the workdir with."""
    w = d / name
    w.mkdir(parents=True)
    (w / "outputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    rc = {"status": "DONE", "mode": "job", "pod_id": pod, "outputs_sha256": sha(w / "outputs.jsonl"),
          "identity": {"variant": "main", "engine": "vllm", "gpu_name": "NVIDIA RTX A4000", "recheck_of": None,
                       "mode": "job", "pin_set": "main", "model": rows[0]["model"] if rows else None,
                       "weights": rows[0]["seed"] if rows else None,
                       "gen_code_sha256": GEN_CODE, "engine_settings": SETTINGS, "decision_sha256": DECISION_SHA,
                       "compose": V0,
                       **(identity or {})},
          "environment": {"packages": dict(GEN_PINS), "locks": {"requirements-gen.lock": "4" * 64}},
          "engine": ENGINE, "code_sha256": code_sha256(),
          **(receipt or {})}
    (w / "receipt-generate.json").write_text(json.dumps(rc))
    return w


def write_recheck(d, gen, name=None, pod="pod-2", identical=True, rows=(), n=5):
    rc = json.loads((gen / "receipt-generate.json").read_text())
    return write_run(d, name or "recheck-" + gen.name, list(rows), pod=pod,
                     identity={"recheck_of": f"/workspace/runs/{gen.name}", "model": rc["identity"]["model"],
                               "sample": n},
                     receipt={"recheck": {"identical": identical, "of_outputs_sha256": rc["outputs_sha256"],
                                          "of_pod_id": rc["pod_id"]}, "sample": {"rule": "recheck", "n": n}})


class Fixture:
    def __init__(self, tmp_path, drop=(), excluded=EXCLUDED):
        self.root = tmp_path
        self.files = write_prompts(tmp_path / "p", excluded)
        self.leaks = leak_receipts(tmp_path / "leaks")
        self.trains = train_receipts(tmp_path / "trains")
        self.rows = output_rows(tmp_path / "p", self.leaks, drop, excluded)
        write_verdicts(tmp_path)

    def runs(self, rows=None, split=True):
        """One gen workdir per (model, seed), each rechecked on another pod."""
        rows = self.rows if rows is None else rows
        groups = {}
        for r in rows:
            groups.setdefault(f"gen-main-{r['model']}-{r['seed']}", []).append(r)
        if not split:
            groups = {"gen-main-all": rows}
        dirs = []
        for name, rs in sorted(groups.items()):
            gen = write_run(self.root / "w", name, rs)
            dirs += [gen, write_recheck(self.root / "w", gen)]
        return dirs

    def argv(self, workdirs, *extra, records=True, decision=DECISION_DOC):
        recs = []
        if records:
            for part, ids in RECORDS.items():
                path = self.root / f"records.{part}.jsonl"
                path.write_text("".join(json.dumps({"id": r, "partition": part}) + "\n" for r in ids))
                recs.append(str(path))
        (self.root / "decision.json").write_text(decision)
        return ["--variant", "main", "--decision", str(self.root / "decision.json"), "--prompts", *map(str, self.files),
                "--prompts-sums",
                str(self.root / "p" / "PROMPTS_SHA256SUMS"), "--workdirs", *map(str, workdirs),
                "--train-receipts", *map(str, self.trains), "--leakcheck-receipts", *map(str, self.leaks.values()),
                *(["--records", *recs] if recs else []), "--out", str(self.root / "coverage.json"), *extra]

    def report(self):
        return json.loads((self.root / "coverage.json").read_text())


def test_every_planned_pair_once_passes_and_is_counted(tmp_path):
    f = Fixture(tmp_path)
    f.rows[0]["prompt_tokens"] = 9000                                # one long prompt: reported by length group
    f.rows[0]["max_new_tokens"] = f.rows[0]["new_tokens"] = min(6144, 16384 - 9000)
    assert main(f.argv(f.runs())) == 0
    r = f.report()
    assert r["status"] == "PASS" and r["expected"] == {"aux_test": 17, "test": 39 * 2 - 11} and r["found"] == 17 + 67
    assert r["excluded"] == {"test": {"too_long_test_rule": 11}}                    # 11 runs use F2 on test
    assert r["by_run"]["deepseek/s7/full_aware"] == {"expected": 2, "found": 2}   # a (test) and c (aux); b excluded
    assert r["length_capped"]["test"]["apache/fineract"]["ft_only"] == [11, 11]
    assert r["length_capped"]["test"]["Sylius/Sylius"]["ft_only"] == [0, 11]
    assert r["prompt_length_groups"] == {"aux_test": {"<=8192": 17}, "test": {"<=8192": 66, ">8192": 1}}
    assert r["workdirs"] == {"gen": 4 + 6 + 4, "recheck": 14} and r["cross_pod"]["deepseek"] is True


def test_report_records_the_training_gpu_of_each_adapter(tmp_path):
    """coverage.json's training_gpu names the training card of every adapter, for the
    report; train_receipts() (this file's own fixture) trains everything on TRAIN_GPU."""
    f = Fixture(tmp_path)
    assert main(f.argv(f.runs())) == 0
    r = f.report()
    assert r["training_gpu"]["deepseek s42"] == TRAIN_GPU and r["training_gpu"]["mistral s11"] == TRAIN_GPU
    assert len(r["training_gpu"]) == len(f.trains)


def test_an_exclusion_of_one_model_skips_that_model_only(tmp_path):
    """{"model": "mistral"} excludes the pair for Mistral's runs only."""
    one = EXCLUDED + [{"record_id": "Sylius/Sylius|a.feature|1|s", "fmt": "F0", "model": "mistral",
                       "reason": "control_token"}]
    f = Fixture(tmp_path, excluded=one)
    assert main(f.argv(f.runs())) == 0
    r = f.report()
    assert r["excluded"]["test"] == {"too_long_test_rule": 11, "control_token": 4}      # base + 3 seeds, F0, Mistral
    assert r["by_run"]["mistral/s42/ft_only"] == {"expected": 1, "found": 1}
    assert r["by_run"]["deepseek/s42/ft_only"] == {"expected": 3, "found": 3}               # a, b and aux c


def test_a_missing_line_fails(tmp_path):
    f = Fixture(tmp_path, drop={("Sylius/Sylius|a.feature|1|s", "full_aware", "mistral", "s23")})
    assert main(f.argv(f.runs())) == 2
    r = f.report()
    assert r["status"] == "FAIL" and r["missing"]["n"] == 1
    assert r["missing"]["first"] == [["Sylius/Sylius|a.feature|1|s", "mistral", "s23", "full_aware", "main"]]


def test_a_pair_written_twice_fails(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    extra = write_run(tmp_path / "w", "gen-main-extra", f.rows[:1])
    assert main(f.argv(dirs + [extra, write_recheck(tmp_path / "w", extra)])) == 2
    assert f.report()["duplicates"]["n"] == 1


@pytest.mark.parametrize("change, field", [
    ({"record_id": "apache/fineract|b.feature|2|s", "condition": "full_aware", "fmt": "F2"}, "unexpected"),  # excluded
    ({"prompt_sha256": "c" * 64}, "mismatched"),
    ({"prompts_sha256": "c" * 64}, "mismatched"),              # the prompt FILE digest
])
def test_lines_outside_the_plan_or_the_prompt_files_fail(tmp_path, change, field):
    f = Fixture(tmp_path)
    i = next(i for i, r in enumerate(f.rows)
             if (r["model"], r["seed"], r["condition"]) == ("deepseek", "s42", "ft_only"))
    bad = dict(f.rows[i], **change)
    rows = f.rows[:i] + [bad] + f.rows[i + 1:] if field == "mismatched" else f.rows + [bad]
    assert main(f.argv(f.runs(rows))) == 2
    assert f.report()[field]["n"] >= 1


def test_an_invalid_line_is_a_problem_not_a_crash(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    out = dirs[0] / "outputs.jsonl"
    with out.open("a") as fh:
        fh.write('{"record_id": "x"}\nnot json\n')
    rc = json.loads((dirs[0] / "receipt-generate.json").read_text())
    (dirs[0] / "receipt-generate.json").write_text(json.dumps({**rc, "outputs_sha256": sha(out)}))
    assert main(f.argv(dirs)) == 2
    assert sum(":" in p and ("not JSON" in p or "keys" in p) for p in f.report()["problems"]) == 2


def test_a_text_with_unicode_line_separators_is_one_line(tmp_path):
    """Lines are split on b"\\n" only."""
    f = Fixture(tmp_path)
    f.rows[0]["text"] = "a b\x85c"
    assert main(f.argv(f.runs())) == 0


def test_records_check_the_universe_of_the_prompt_files(tmp_path):
    f = Fixture(tmp_path)
    argv = f.argv(f.runs())
    assert main(argv) == 0
    recs = tmp_path / "records.test.jsonl"
    recs.write_text(recs.read_text() + json.dumps({"id": "Sylius/Sylius|z.feature|9|s", "partition": "test"}) + "\n")
    assert main(argv) == 2
    assert any("Sylius/Sylius|z.feature|9|s" in p for p in f.report()["problems"])


def test_records_are_required_for_real_variants(tmp_path, capsys):
    """Without --records a record dropped from every prompt file would go unnoticed."""
    f = Fixture(tmp_path)
    assert main(f.argv(f.runs(), records=False)) == 2 and "--records" in capsys.readouterr().err


def test_prompt_files_that_differ_from_their_sums_fail(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    with f.files[0].open("a") as fh:
        fh.write("\n")
    assert main(f.argv(dirs)) == 2
    assert any("sha256 differs" in p for p in f.report()["problems"])


def rewrite(workdir, change):
    rc = json.loads((workdir / "receipt-generate.json").read_text())
    change(rc)
    (workdir / "receipt-generate.json").write_text(json.dumps(rc))


@pytest.mark.parametrize("change, words", [
    (lambda rc: rc.update(status="INCOMPLETE"), "DONE"),
    (lambda rc: rc.update(mode="test"), "job mode"),
    (lambda rc: rc["identity"].update(engine="hf"), "vllm"),
    (lambda rc: rc["identity"].update(variant="dev_pilot_v0"), "variant"),
    (lambda rc: rc.update(outputs_sha256="0" * 64), "outputs_sha256"),
    (lambda rc: rc["environment"]["packages"].update(vllm="0.26.0"), "0.27.1"),
    (lambda rc: rc["identity"].update(gpu_name="NVIDIA RTX A5000"), "gpu_name"),      # never mixed
    (lambda rc: rc["identity"].update(pin_set="custom"), "pin set 'custom'"),
    (lambda rc: rc.update(engine={**ENGINE, "batch_invariant": True}), "batch_invariant"),
    (lambda rc: rc.update(engine={**ENGINE, "effective": {**ENGINE["effective"], "cudagraph_mode": "NONE"}}),
     "engine"),
])
def test_a_workdir_that_is_not_a_finished_pinned_job_fails(tmp_path, change, words):
    """Coverage opens the receipts (decision 24/09: batch_invariant true/false are never mixed)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[2], change)
    assert main(f.argv(dirs)) == 2
    assert any(words in p for p in f.report()["problems"]), f.report()["problems"][:5]


def test_gen_card_by_pair_problems_names_the_seed_with_mixed_generation_cards(tmp_path):
    """Within one (model, seed) adapter, its recheck running on a different gpu_name than
    its own job is named specifically ("mistral s23"), not just folded into the whole-variant "workdirs name
    several cards" list (already covered by a test above)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    recheck = next(d for d in dirs if d.name == "recheck-gen-main-mistral-s23")
    rewrite(recheck, lambda rc: rc["identity"].update(gpu_name="NVIDIA RTX 2000 Ada Generation"))
    assert main(f.argv(dirs)) == 2
    problems = f.report()["problems"]
    assert any("mistral s23: generation workdirs record different gpu_name across conditions" in p and
              "NVIDIA RTX A4000" in p and "NVIDIA RTX 2000 Ada Generation" in p for p in problems), problems


def test_workdirs_in_two_pin_sets_fail(tmp_path):
    """decision 24/09: a fallback pin set is chosen at the canary for the whole run; a workdir in another
    declared set is valid on its own, but the variant never mixes sets."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    for w in dirs[2:4]:                                               # a gen workdir and its recheck
        rewrite(w, lambda rc: (rc["identity"].update(pin_set="fallback-bnb"),
                               rc["environment"].update(packages=dict(PIN_SETS["fallback-bnb"]))))
    assert main(f.argv(dirs)) == 2
    assert any("pin sets" in p and "fallback-bnb" in p for p in f.report()["problems"])


@pytest.mark.parametrize("setting, value", [("gpu_memory_utilization", 0.95), ("enforce_eager", True)])
def test_workdirs_that_differ_in_a_memory_step_fail(tmp_path, setting, value):
    """decision 24/09: the memory steps are decided at the canary and fixed for the whole run."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[2], lambda rc: rc.update(engine={**ENGINE, "kwargs": {**ENGINE["kwargs"], setting: value}}))
    assert main(f.argv(dirs)) == 2
    assert any(f"kwargs.{setting}" in p for p in f.report()["problems"]), f.report()["problems"][:5]


def test_an_adapter_digest_that_is_not_the_train_receipts_fails(tmp_path):
    """s42 outputs labelled s11 carry the s42 digest."""
    f = Fixture(tmp_path)
    for r in f.rows:
        if (r["model"], r["seed"]) == ("mistral", "s11"):
            r["adapter_tensor_sha256"] = adapter_sha("mistral", "s42")
    assert main(f.argv(f.runs())) == 2
    assert any("mistral s11" in p and "train receipt" in p for p in f.report()["problems"])


def test_a_line_whose_leak_receipt_is_not_a_pass_of_this_code_fails(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    path = f.leaks["codellama"]
    rc = json.loads(path.read_text())
    path.write_text(json.dumps({**rc, "code_sha256": {**rc["code_sha256"], "v2gpu/leakcheck.py": "0" * 64}}))
    assert main(f.argv(dirs)) == 2
    problems = f.report()["problems"]
    assert any("codellama" in p and "code_sha256" in p for p in problems)
    assert any("not among --leakcheck-receipts" in p for p in problems)     # the lines named the old receipt's sha


def test_coverage_binds_the_leak_receipts_to_the_code_that_generated(tmp_path):
    """Coverage may run at any later commit. The leak receipts must match the code_sha256 that the
    generation receipts record (the pod's checkout), not the laptop's checkout at coverage time."""
    f = Fixture(tmp_path)
    old = {**code_sha256(), "v2gpu/leakcheck.py": "0" * 64}              # the code at generation time
    for path in f.leaks.values():
        path.write_text(json.dumps({**json.loads(path.read_text()), "code_sha256": old}))
    f.rows = output_rows(tmp_path / "p", f.leaks)                          # lines name the re-written receipts
    dirs = f.runs()
    for w in dirs:
        rewrite(w, lambda rc: rc.update(code_sha256=old))
    assert main(f.argv(dirs)) == 0, f.report()["problems"][:5]
    rewrite(dirs[2], lambda rc: rc.update(code_sha256=code_sha256()))
    assert main(f.argv(dirs)) == 2
    assert any("code_sha256" in p and "generated by different code" in p for p in f.report()["problems"])


def test_a_second_train_receipt_of_one_adapter_fails(tmp_path):
    """A glob that catches both the dev-pilot and the main deepseek s42 receipts must not let the
    order of the files decide which digest the lines are checked against."""
    f = Fixture(tmp_path)
    twin = tmp_path / "trains" / "deepseek-s42-pilot.json"
    twin.write_text(json.dumps({**json.loads((tmp_path / "trains" / "deepseek-s42.json").read_text()),
                                "selected_adapter": {"tensor_sha256": "9" * 64}}))
    argv = f.argv(f.runs())
    i = argv.index("--train-receipts")
    argv[i + 1:i + 1] = [str(twin)]
    assert main(argv) == 2
    assert any("deepseek s42" in p and "second train receipt" in p for p in f.report()["problems"])


# Every train receipt needs train_code_sha256 and train_cfg_sha256, and
# every adapter of the variant must share one value of each


@pytest.mark.parametrize("field", ["train_code_sha256", "train_cfg_sha256"])
def test_a_train_receipt_without_the_code_or_config_digest_is_refused(tmp_path, field):
    f = Fixture(tmp_path)
    target = tmp_path / "trains" / "deepseek-s42.json"
    rc = json.loads(target.read_text())
    del rc[field]
    target.write_text(json.dumps(rc))
    assert main(f.argv(f.runs())) == 2
    assert any(field in p for p in f.report()["problems"])


@pytest.mark.parametrize("field", ["train_code_sha256", "train_cfg_sha256"])
def test_adapters_disagreeing_on_the_code_or_config_digest_are_refused(tmp_path, field):
    f = Fixture(tmp_path)
    target = tmp_path / "trains" / "deepseek-s42.json"
    rc = json.loads(target.read_text())
    rc[field] = {"v2gpu/train_job.py": "d" * 64} if field == "train_code_sha256" else "d" * 64
    target.write_text(json.dumps(rc))
    assert main(f.argv(f.runs())) == 2
    assert any(field in p and "differs across adapters" in p for p in f.report()["problems"])


# The real pod layout (pinned 27/09/2026) is
# results/<model>-s<seed>-all/receipt-train.json -- every adapter's receipt shares the SAME basename, which hid the
# old path.name-keyed dict silently collapsing every adapter's train_code_sha256/train_cfg_sha256 into one entry


def production_receipt(root, model, seed, code=None, cfg=None, gpu=TRAIN_GPU):
    d = root / f"{model}-s{seed}-all"
    d.mkdir(parents=True)
    path = d / "receipt-train.json"
    path.write_text(json.dumps({
        "status": "DONE", "mode": "job", "seed": seed, "subset": "all", "model": {"key": model},
        "selected_adapter": {"tensor_sha256": adapter_sha(model, f"s{seed}")},
        "data": {"prompts": {"meta": {"compose": V0}}},
        "train_code_sha256": code or TRAIN_CODE_SHA256, "train_cfg_sha256": cfg or TRAIN_CFG_SHA256,
        "train_config": TRAIN_CONFIG, "environment": {"device": {"name": gpu}} if gpu else {}}))
    return path


def test_train_code_sha256_differing_across_models_in_the_real_layout_is_caught(tmp_path):
    """Every receipt below is literally named receipt-train.json (only its parent directory differs, exactly as
    genqueue's own usage line ships them): the OLD path.name key would keep only the last one read and never notice."""
    from v2gpu.gencoverage import train_digests

    paths = [production_receipt(tmp_path / "results", "codellama", 42),
             production_receipt(tmp_path / "results", "deepseek", 42, code={"v2gpu/train_job.py": "z" * 64}),
             production_receipt(tmp_path / "results", "mistral", 42)]
    problems, out = train_digests(paths)
    assert any("train_code_sha256" in p and "differs across adapters" in p and "codellama s42" in p and
              "deepseek s42" in p for p in problems), problems
    assert set(out) == {("codellama", "s42"), ("deepseek", "s42"), ("mistral", "s42")}


def test_train_cfg_sha256_differing_across_models_in_the_real_layout_is_caught(tmp_path):
    from v2gpu.gencoverage import train_digests

    paths = [production_receipt(tmp_path / "results", "codellama", 42),
             production_receipt(tmp_path / "results", "deepseek", 42, cfg="z" * 64)]
    problems, _ = train_digests(paths)
    assert any("train_cfg_sha256" in p and "differs across adapters" in p for p in problems), problems


def test_agreeing_adapters_in_the_real_layout_pass(tmp_path):
    """The control: same basename everywhere, every adapter agrees, no problem."""
    from v2gpu.gencoverage import train_digests

    paths = [production_receipt(tmp_path / "results", m, 42) for m in ("codellama", "deepseek", "mistral")]
    problems, out = train_digests(paths)
    assert problems == [] and set(out) == {("codellama", "s42"), ("deepseek", "s42"), ("mistral", "s42")}


def test_train_digests_refuses_an_adapter_trained_without_upcast_fp32(tmp_path):
    """Decision 27/09: fp32 upcast is mandatory for every real training job; gencoverage
    refuses an adapter whose train receipt says otherwise, for every variant, and names it."""
    from v2gpu.gencoverage import train_digests

    off = production_receipt(tmp_path / "off", "deepseek", 42)
    rc = json.loads(off.read_text())
    rc["train_config"] = {"upcast_fp32": False}
    off.write_text(json.dumps(rc))
    problems, _ = train_digests([off])
    assert any("upcast_fp32 is not true" in p and "deepseek s42" in p for p in problems), problems
    # omitting train_config entirely is the same refusal, never a silent pass
    missing = production_receipt(tmp_path / "missing", "codellama", 42)
    rc = json.loads(missing.read_text())
    del rc["train_config"]
    missing.write_text(json.dumps(rc))
    problems, _ = train_digests([missing])
    assert any("upcast_fp32 is not true" in p and "codellama s42" in p for p in problems), problems


def test_train_gpu_names_reads_the_training_card_per_adapter(tmp_path):
    """{(model, seed): training GPU name}, read from each train receipt's
    environment.device.name, for coverage.json (the report)."""
    from v2gpu.gencoverage import train_gpu_names

    paths = [production_receipt(tmp_path / "results", "deepseek", 42, gpu="NVIDIA RTX A6000"),
             production_receipt(tmp_path / "results", "codellama", 42, gpu="NVIDIA A100-SXM4-80GB"),
             production_receipt(tmp_path / "results", "mistral", 11, gpu=None)]
    assert train_gpu_names(paths) == {("deepseek", "s42"): "NVIDIA RTX A6000",
                                      ("codellama", "s42"): "NVIDIA A100-SXM4-80GB", ("mistral", "s11"): None}


def test_train_card_problems_flags_a_seed_trained_on_mixed_cards(tmp_path):
    """Since 27/09, card assignment is per SEED -- deepseek-s42 and codellama-s42 on
    different card classes would confound the seed-paired comparison across models. Adapters with no recorded GPU
    (None) are never flagged on their own, and a seed trained on one card class throughout passes."""
    from v2gpu.gencoverage import train_card_problems

    mixed = {("deepseek", "s42"): "NVIDIA RTX A6000", ("codellama", "s42"): "NVIDIA A100-SXM4-80GB",
             ("mistral", "s42"): "NVIDIA RTX A6000", ("deepseek", "s11"): None}
    problems = train_card_problems(mixed)
    assert len(problems) == 1
    assert "seed s42" in problems[0] and "card assignment is per seed" in problems[0]
    assert "NVIDIA RTX A6000" in problems[0] and "NVIDIA A100-SXM4-80GB" in problems[0]
    same = {("deepseek", "s42"): "NVIDIA RTX A6000", ("codellama", "s42"): "NVIDIA RTX A6000",
           ("mistral", "s11"): None}
    assert train_card_problems(same) == []


def test_problems_name_the_adapter_not_just_the_bare_filename(tmp_path):
    """Every receipt shares the basename receipt-train.json; a problem must still say which adapter it is about."""
    from v2gpu.gencoverage import train_digests

    path = production_receipt(tmp_path / "results", "deepseek", 42)
    rc = json.loads(path.read_text())
    del rc["train_code_sha256"]
    path.write_text(json.dumps(rc))
    problems, _ = train_digests([path])
    assert any(str(path) in p and "deepseek s42" in p and "no train_code_sha256" in p for p in problems), problems


def test_the_budget_rule_is_checked_per_line(tmp_path):
    """max_new_tokens = min(6144, window - prompt_tokens), no tokenizer needed."""
    f = Fixture(tmp_path)
    f.rows[3]["max_new_tokens"] = 64
    f.rows[3]["new_tokens"] = min(f.rows[3]["new_tokens"], 64)
    assert main(f.argv(f.runs())) == 2
    assert any("min(6144" in p for p in f.report()["problems"])


def test_every_gen_workdir_needs_a_done_identical_recheck(tmp_path):
    """The recheck receipts are the evidence; coverage requires them before the pod may go."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    assert main(f.argv([d for d in dirs if d.name != "recheck-gen-main-codellama-base"])) == 2
    assert any("gen-main-codellama-base" in p and "recheck" in p for p in f.report()["problems"])
    rewrite(tmp_path / "w" / "recheck-gen-main-codellama-base", lambda rc: rc["recheck"].update(identical=False))
    assert main(f.argv(dirs)) == 2


def test_main_needs_a_cross_pod_recheck_per_model(tmp_path):
    """decision 24/09: for every model of variant main, one recheck runs on another pod than its job."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    for d in dirs:
        if d.name.startswith("recheck-gen-main-mistral"):
            rewrite(d, lambda rc: rc.update(pod_id=rc["recheck"]["of_pod_id"]))
    assert main(f.argv(dirs)) == 2
    r = f.report()
    assert r["cross_pod"] == {"codellama": True, "deepseek": True, "mistral": False}
    assert any("cross-pod" in p and "mistral" in p for p in r["problems"])


def test_dev_grid_variants_resolve_and_malformed_labels_are_argparse_errors(tmp_path, capsys):
    """--variant goes through runs.variant_spec. A grid label is checked against dev (so these test prompt
    files are the wrong partition); a malformed label is refused by argparse."""
    f = Fixture(tmp_path)
    a = f.argv(f.runs())
    a[a.index("--variant") + 1] = "dev_grid_k4_n2_r033"
    assert main(a) == 2
    r = f.report()
    assert r["variant"] == "dev_grid_k4_n2_r033" and r["expected"] == {"dev": 0}
    assert any("partition 'test', expected one of ['dev']" in p for p in r["problems"])
    a[a.index("--variant") + 1] = "dev_grid_k4_n2_r033_x"
    with pytest.raises(SystemExit) as e:
        main(a)
    assert e.value.code == 2 and "dev_grid_k4_n2_r033_x" in capsys.readouterr().err


def test_a_report_says_when_its_problem_list_was_cut(tmp_path):
    """The report lists 200 problems; problems_truncated says there were more."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    assert main(f.argv(dirs)) == 0 and f.report()["problems_truncated"] is False
    out = dirs[0] / "outputs.jsonl"
    out.write_text(out.read_text() + "not json\n" * 205)
    rewrite(dirs[0], lambda rc: rc.update(outputs_sha256=sha(out)))
    assert main(f.argv(dirs)) == 2
    r = f.report()
    assert r["problems_truncated"] is True and len(r["problems"]) == 200 and r["n_problems"] > 200




def test_every_workdir_of_the_variant_ran_one_generation_code(tmp_path):
    """identity.gen_code_sha256, one per variant (code_sha256 names the leak rules only)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[2], lambda rc: rc["identity"].update(gen_code_sha256={**GEN_CODE, "v2gpu/runs.py": "3" * 64}))
    assert main(f.argv(dirs)) == 2
    assert any("gen_code_sha256" in p for p in f.report()["problems"])
    for w in dirs:                                       # none names it: no digest is no evidence either
        rewrite(w, lambda rc: rc["identity"].pop("gen_code_sha256"))
    assert main(f.argv(dirs)) == 2 and any("gen_code_sha256" in p for p in f.report()["problems"])


def test_every_workdir_names_the_compose_of_the_prompt_directory(tmp_path):
    """One compose per variant, the one PROMPTS_META.json of --prompts-sums holds."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[0], lambda rc: rc["identity"].update(compose={"k": 4, "n_examples": 1, "ratio": 0.33}))
    assert main(f.argv(dirs)) == 2
    assert any("compose" in p for p in f.report()["problems"])


def with_compose(d, compose):
    meta = json.loads((d / "PROMPTS_META.json").read_text())
    (d / "PROMPTS_META.json").write_text(json.dumps({**meta, "compose": compose}))
    names = sorted(p.name for p in d.iterdir() if p.name != "PROMPTS_SHA256SUMS")
    (d / "PROMPTS_SHA256SUMS").write_text("".join(f"{sha(d / n)}  {n}\n" for n in names))


@pytest.mark.parametrize("label, bound", [("dev_grid_k4_n1_r033", False), ("dev_grid_k8_n3_r050", True),
                                          ("dev_grid_v1_k8_n3_r050", True), ("dev_grid_k8_n3_r067", False)])
def test_a_dev_grid_label_must_name_the_compose_of_its_prompt_directory(tmp_path, label, bound):
    """A k8 directory under a k4 label is refused (the lines would carry the k4 label on k8 evidence)."""
    f = Fixture(tmp_path)
    compose = {"variant": label, "k": 8, "n_examples": 3, "ratio": 0.5, "mode": "full"}    # 3A --variant = the label
    with_compose(tmp_path / "p", compose)
    dirs = f.runs()
    for w in dirs:
        rewrite(w, lambda rc: rc["identity"].update(variant=label, compose=compose))
    a = f.argv(dirs)
    a[a.index("--variant") + 1] = label
    assert main(a) == 2                             # test prompts under a dev label: refused in any case
    assert any("PROMPTS_META.json compose" in p for p in f.report()["problems"]) != bound


def test_every_recheck_regenerated_the_declared_sample(tmp_path):
    """identity.sample and the receipt's sample.n equal --recheck-sample (default 5)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(tmp_path / "w" / "recheck-gen-main-codellama-base", lambda rc: rc["sample"].update(n=0))
    assert main(f.argv(dirs)) == 2
    assert any("recheck-gen-main-codellama-base" in p and "sample" in p for p in f.report()["problems"])
    with pytest.raises(SystemExit):                                   # 0 is refused as a declaration too
        main(f.argv(dirs, "--recheck-sample", "0"))


def test_a_workdir_whose_outputs_are_not_utf8_is_a_problem_not_a_crash(tmp_path):
    """A CRASHED workdir torn inside a multibyte character names its file in the report."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    out = dirs[0] / "outputs.jsonl"
    out.write_bytes(out.read_bytes() + "{\"text\": \"\u00e9".encode()[:-1])
    rewrite(dirs[0], lambda rc: rc.update(outputs_sha256=sha(out)))
    assert main(f.argv(dirs)) == 2
    assert any(dirs[0].name in p and "not UTF-8" in p for p in f.report()["problems"])


def test_queue_folders_beside_their_workdirs_are_ignored(tmp_path):
    """copy_out.sh leaves results/<job>.queue/ beside each workdir; a glob that catches them does
    not FAIL a clean run. A .queue folder whose workdir is not given is still a problem."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    extra = []
    for d in dirs[:2]:
        q = d.with_name(d.name + ".queue")
        q.mkdir()
        (q / f"{d.name}.status").write_text("DONE\n")
        extra.append(q)
    assert main(f.argv(dirs + extra)) == 0
    assert f.report()["ignored_queue_dirs"] == sorted(q.name for q in extra)
    lone = tmp_path / "w" / "gen-main-nothing.queue"
    lone.mkdir()
    assert main(f.argv(dirs + [lone])) == 2
    assert any("gen-main-nothing.queue" in p for p in f.report()["problems"])


# decision 24/09: one generation card class per variant, from the allow-list


def test_one_variant_never_mixes_the_two_card_classes(tmp_path):
    """A4000 and RTX 2000 Ada are both allowed, never in one variant (main of all three models on ONE class)."""
    ada = "NVIDIA RTX 2000 Ada Generation"
    doc = decision_doc({**DECISION, "gpu_name": ada})                               # the Ada class's own decision
    f = Fixture(tmp_path)
    write_verdicts(tmp_path, {**DECISION, "gpu_name": ada})                        # and its canaries' verdicts
    dirs = f.runs()
    for w in dirs:
        rewrite(w, lambda rc: rc["identity"].update(gpu_name=ada, decision_sha256=hashlib.sha256(
            doc.encode()).hexdigest()))
    assert main(f.argv(dirs, decision=doc)) == 0, f.report()["problems"][:3]
    for w in dirs[:4]:
        rewrite(w, lambda rc: rc["identity"].update(gpu_name="NVIDIA RTX A4000"))
    assert main(f.argv(dirs, decision=doc)) == 2 and any("several cards" in p for p in f.report()["problems"])
    for w in dirs:
        rewrite(w, lambda rc: rc["identity"].update(gpu_name="NVIDIA RTX 4000 Ada Generation"))
    assert main(f.argv(dirs, decision=doc)) == 2 and any("card class" in p for p in f.report()["problems"])


# Every workdir ran the variant's decision, code and engine


def test_workdirs_of_another_memory_step_and_pin_set_than_the_decision_fail(tmp_path):
    """As reproduced once: a variant run at 0.95 + eager + fallback-bnb, consistent in itself, against a decision of
    0.92/main gave PASS."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    assert main(f.argv(dirs)) == 0
    engine = {**ENGINE, "kwargs": {**ENGINE["kwargs"], "gpu_memory_utilization": 0.95, "enforce_eager": True}}
    for w in dirs:
        rewrite(w, lambda rc: (rc["identity"].update(pin_set="fallback-bnb", engine_settings={
            **SETTINGS, "gpu_memory_utilization": 0.95, "enforce_eager": True}),
            rc.update(engine=engine, environment={"packages": dict(PIN_SETS["fallback-bnb"])})))
    assert main(f.argv(dirs)) == 2
    problems = f.report()["problems"]
    assert any("decision" in p and "pin_set" in p for p in problems)
    assert any("decision" in p and "engine" in p for p in problems)


@pytest.mark.parametrize("change, words", [
    (lambda rc: rc["identity"].update(decision_sha256="0" * 64), "decision_sha256"),
    (lambda rc: rc["identity"].update(gen_code_sha256={**GEN_CODE, "v2gpu/runs.py": "3" * 64}), "gen_code_sha256"),
    (lambda rc: rc["identity"].update(engine_settings={**SETTINGS, "batch_invariant": True}), "engine_settings"),
    (lambda rc: rc.update(engine={**ENGINE, "version": "0.28.0"}), "engine_fingerprint"),
])
def test_every_workdir_carries_the_decision_and_its_code_and_engine(tmp_path, change, words):
    f = Fixture(tmp_path)
    dirs = f.runs()
    for w in dirs:
        rewrite(w, change)
    assert main(f.argv(dirs)) == 2 and any(words in p for p in f.report()["problems"]), words


def test_coverage_needs_the_variants_decision(tmp_path, capsys):
    """Every workdir carries the sha256 of the decision given, and that decision must be a passing one of this
    variant (the workdirs are re-pointed at each bad file, so only the file's own check can refuse it)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    a = f.argv(dirs)
    i = a.index("--decision")
    assert main(a[:i] + a[i + 2:]) == 2 and "--decision" in capsys.readouterr().err
    for bad, words in (({"variant": "dev_pilot_v0"}, "variant"), ({"pass": False}, "not a passing"),
                       ({"schema": "gtestgen.v2.gen_decision.v1"}, "not a passing")):
        doc = json.dumps({**json.loads(DECISION_DOC), **bad})
        digest = hashlib.sha256(doc.encode()).hexdigest()
        for w in dirs:
            rewrite(w, lambda rc, digest=digest: rc["identity"].update(decision_sha256=digest))
        assert main(f.argv(dirs, decision=doc)) == 2, bad
        assert any(words in p for p in f.report()["problems"]), bad


# The records file, and the cross-pod recheck (decision 25/09)


@pytest.mark.parametrize("body", [b"not json\n", None])
def test_an_unreadable_or_missing_records_file_is_a_problem(tmp_path, body):
    """An unreadable records file, and a missing one, is a problem, not a traceback."""
    f = Fixture(tmp_path)
    a = f.argv(f.runs())
    bad = tmp_path / "records.bad.jsonl"
    if body is not None:
        bad.write_bytes(body)
    i = a.index("--records")
    assert main(a[:i + 1] + [str(bad)] + a[i + 1:]) == 2
    assert any("records.bad.jsonl" in p for p in f.report()["problems"])


@pytest.mark.parametrize("variant", ["dev_pilot_v0", "dev_grid_k8_n3_r050", "dev_grid_v1_k4_n1_r033"])
def test_every_dev_variant_needs_its_cross_pod_recheck_as_main_does(tmp_path, variant):
    """decision 25/09, extended to the dev grid labels: coverage requires one
    cross-pod recheck per model for every real variant, as for main; gen_canary has none."""
    from v2gpu.gencoverage import workdir_problems

    f = Fixture(tmp_path)
    dirs = f.runs()
    for w in dirs:
        rewrite(w, lambda rc: (rc["identity"].update(variant=variant),
                               rc.update(pod_id=(rc.get("recheck") or {}).get("of_pod_id") or rc["pod_id"])))
    assert any("cross-pod" in p for p in workdir_problems(dirs, variant)[0])
    assert not any("cross-pod" in p for p in workdir_problems(dirs, "gen_canary")[0])


# The run label, the prompt directory and the adapters share one compose


def test_an_adapter_trained_on_another_compose_fails(tmp_path):
    """(c): main runs adapters whose train-receipt compose equals the generation META compose, every key."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    path = tmp_path / "trains" / "mistral-s23.json"
    rc = json.loads(path.read_text())
    path.write_text(json.dumps({**rc, "data": {"prompts": {"meta": {"compose": {**V0, "k": 12}}}}}))
    assert main(f.argv(dirs)) == 2
    assert any("mistral-s23.json" in p and "compose" in p for p in f.report()["problems"])


# Since 27/09/2026 a dev grid label's adapter is checked
# against the round's OWN dev_pilot_vN META compose (train_digests' pilot_compose), never the grid's own directory


def test_train_digests_checks_a_dev_grid_adapter_against_the_rounds_pilot_compose(tmp_path):
    """A dev grid variant cannot be checked against `meta` (the grid's own PROMPTS_META.json compose): it legitimately
    varies k, n_examples and ratio away from its round's pilot. No pilot_compose given is a refusal, never a silent
    pass; the stale-ratio probe (a round-1 adapter trained at a stale ratio, after its round's pilot was rebuilt) is refused too."""
    from v2gpu.gencoverage import train_digests

    receipt = tmp_path / "receipt-train.json"
    receipt.write_text(json.dumps({
        "status": "DONE", "mode": "job", "model": {"key": "deepseek"}, "seed": 42, "subset": "all",
        "selected_adapter": {"tensor_sha256": "a" * 64}, "data": {"prompts": {"meta": {"compose": V0}}},
        "train_code_sha256": {"v2gpu/train_job.py": "b" * 64}, "train_cfg_sha256": "c" * 64,
        "train_config": TRAIN_CONFIG, "environment": {"device": {"name": TRAIN_GPU}}}))
    # Mode must still agree with the pilot (not exempt); every
    # other key here is irrelevant to a grid variant's OWN adapter-vs-pilot checks (GRID_META_EXEMPT_KEYS covers it)
    meta = {"compose": {"variant": "not the round's pilot, irrelevant to a grid variant", "mode": "full"}}
    label = f"{receipt} (deepseek s42)"       # every problem names path AND adapter
    assert train_digests([receipt], "dev_grid_k4_n1_r033", meta)[0] == [
        f"{label}: variant dev_grid_k4_n1_r033 needs the round's dev_pilot_v0 META compose to check the "
        "adapter; none was given"]
    assert train_digests([receipt], "dev_grid_k4_n1_r033", meta, V0)[0] == []
    drifted = {**V0, "ratio": 0.6666666666666666}                          # the round's pilot was rebuilt at r067
    assert train_digests([receipt], "dev_grid_k4_n1_r033", meta, drifted)[0] == [
        f"{label}: the adapter was trained on another compose than the round's dev_pilot_v0 (differs in "
        "ratio)"]


def test_a_round1_label_on_the_v0_directory_fails_coverage(tmp_path):
    """dev_pilot_v1 labels on the v0 directory are refused by coverage as by preflight."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    a = f.argv(dirs)
    a[a.index("--variant") + 1] = "dev_pilot_v1"
    assert main(a) == 2 and any("PROMPTS_META.json compose" in p for p in f.report()["problems"])


def test_meta_files_must_be_the_prompt_files_the_sums_list(tmp_path):
    """META files must equal the prompt files PROMPTS_SHA256SUMS lists."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    meta = json.loads((tmp_path / "p" / "PROMPTS_META.json").read_text())
    (tmp_path / "p" / "PROMPTS_META.json").write_text(json.dumps({**meta, "files": meta["files"][1:]}))
    names = sorted(x.name for x in (tmp_path / "p").iterdir() if x.name != "PROMPTS_SHA256SUMS")
    (tmp_path / "p" / "PROMPTS_SHA256SUMS").write_text("".join(f"{sha(tmp_path / 'p' / n)}  {n}\n" for n in names))
    assert main(f.argv(dirs)) == 2 and any("PROMPTS_META.json files" in p for p in f.report()["problems"])


def test_every_train_receipt_is_of_the_pool_all(tmp_path):
    """The scored adapters are trained on pool all; an lc600 receipt of a seed fails."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    path = tmp_path / "trains" / "codellama-s11.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "subset": "lc600"}))
    assert main(f.argv(dirs)) == 2
    assert any("codellama-s11.json" in p and "all" in p for p in f.report()["problems"])


# One requirements lock for the variant


LOCK = {"requirements-gen.lock": "4" * 64}


def with_locks(dirs, value, first=None):
    for w in dirs:
        rewrite(w, lambda rc: rc["environment"].update(locks=value, first_run_lock=first))


def test_every_workdir_ran_one_requirements_lock(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    with_locks(dirs, LOCK)
    assert main(f.argv(dirs)) == 0, f.report()["problems"][:3]
    assert f.report()["locks"] == {"requirements-gen.lock": LOCK["requirements-gen.lock"]}
    with_locks(dirs[:2], {"requirements-gen.lock": "5" * 64})
    assert main(f.argv(dirs)) == 2 and any("lock" in p for p in f.report()["problems"])


def test_a_workdir_without_a_lock_needs_the_recorded_first_run_exception(tmp_path):
    """The first pod of a KIND runs without the committed lock: its receipt records the lock it froze
    (first_run_lock), which later pods install; coverage accepts it as that lock and shows the exception."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    with_locks(dirs, {})
    assert main(f.argv(dirs)) == 2 and any("lock" in p for p in f.report()["problems"])
    with_locks(dirs[:3], {}, first=LOCK)
    with_locks(dirs[3:], LOCK)
    assert main(f.argv(dirs)) == 0, f.report()["problems"][:3]
    assert f.report()["first_run_workdirs"] == sorted(w.name for w in dirs[:3])


@pytest.mark.parametrize("where", ["locks", "first_run_lock"])
@pytest.mark.parametrize("value", ["", "x", "4" * 63, "4" * 63 + "G", {"sha256": "4" * 64}, None, 7,
                                   "4" * 63 + "A", "4" * 65, "4" * 64 + "\n", "4" * 32 + "\n" + "4" * 31],
                         ids=["empty", "x", "63-hex", "G", "dict", "null", "int", "uppercase", "65-hex", "newline-end",
                              "newline-inside"])
def test_a_lock_that_is_not_a_sha256_is_a_problem_not_a_pass_or_a_crash(tmp_path, where, value):
    """"" or "x" passed as the lock (an empty first-run lock was even left out of
    first_run_workdirs), a dict crashed; both sources must be a 64-hex sha256, else a problem naming the workdir."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    with_locks(dirs, {"requirements-gen.lock": value} if where == "locks" else {},
               first={"requirements-gen.lock": value} if where == "first_run_lock" else None)
    assert main(f.argv(dirs)) == 2
    problems = f.report()["problems"]
    assert any(where in p and "requirements-gen.lock" in p and "sha256" in p and dirs[0].name in p
               for p in problems), problems[:5]


def test_the_committed_lock_of_the_workdirs_pin_set_comes_first():
    """Mutants that once survived: the lock name ignored the pin set, and the first-run lock was preferred to
    the committed one."""
    from v2gpu.gencoverage import lock_of, lock_problems

    def rc(pin_set, locks, first=None):
        return {"identity": {"pin_set": pin_set}, "environment": {"locks": locks, "first_run_lock": first}}

    main_lock, other = {"requirements-gen.lock": "4" * 64}, {"requirements-gen.lock": "5" * 64}
    assert lock_of(rc("main", main_lock, other))[:3] == ("requirements-gen.lock", "4" * 64, False)
    assert lock_of(rc("main", {}, other))[:3] == ("requirements-gen.lock", "5" * 64, True)
    bnb = {"requirements-gen-fallback-bnb.lock": "6" * 64}
    assert lock_of(rc("fallback-bnb", {**main_lock, **bnb}))[:3] == ("requirements-gen-fallback-bnb.lock", "6" * 64,
                                                                     False)
    problems, locks, _ = lock_problems([(Path("gen-x"), rc("fallback-bnb", main_lock))])
    assert locks == {} and any("gen-x" in p for p in problems)


@pytest.mark.parametrize("engine, why", [
    ({}, "it names no vLLM engine (name and version)"),
    ({k: v for k, v in ENGINE.items() if k != "version"}, "it names no vLLM engine (name and version)"),
    ("vllm", "the engine record is not an object"),
    ({**ENGINE, "kwargs": None}, "engine.kwargs is NoneType, not an object"),
    ({**ENGINE, "env": ["VLLM_BATCH_INVARIANT=0"]}, "engine.env is list, not an object"),
    ({**ENGINE, "sampling": [0.0]}, "engine.sampling is list, not an object"),
    ({**ENGINE, "lora_shrink": {"checked": [{"num_slices": 1}]}},
     "engine.lora_shrink.checked is not a list of records with a config"),
])
def test_the_engine_problem_line_names_the_workdir_and_why(tmp_path, engine, why):
    """The odd types of engines.engine_fingerprint: the exact problem line, never an
    exception."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[2], lambda rc: rc.update(engine=engine))
    assert main(f.argv(dirs)) == 2
    assert f"{dirs[2].name}: the receipt's engine record gives no engine fingerprint: {why}" in f.report()["problems"]


@pytest.mark.parametrize("engine", [{}, {k: v for k, v in ENGINE.items() if k != "version"}, "vllm"])
def test_a_workdir_whose_engine_record_names_no_engine_is_a_problem_not_a_crash(tmp_path, engine):
    """A regression once: an engine record without a name and version
    raised TypeError in _differing; it is a problem naming the workdir."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    rewrite(dirs[2], lambda rc: rc.update(engine=engine))
    assert main(f.argv(dirs)) == 2
    assert any(dirs[2].name in p and "engine" in p for p in f.report()["problems"]), f.report()["problems"][:5]


# The decision file itself, and what the report records


def with_decision(f, dirs, text):
    """The workdirs re-pointed at this decision file (so only the file's own checks can refuse it); coverage's exit."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    for w in dirs:
        rewrite(w, lambda rc: rc["identity"].update(decision_sha256=digest))
    return main(f.argv(dirs, decision=text))


def test_the_decision_covers_every_model_the_variant_plans(tmp_path):
    """Verdicts named without their files, {model: {}}, no longer pass."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    doc = json.loads(DECISION_DOC)
    for verdicts, ok in ((doc["verdicts"], True), ({m: {} for m in MODELS3}, False),
                         ({"deepseek": doc["verdicts"]["deepseek"]}, False)):
        assert (with_decision(f, dirs, json.dumps({**doc, "verdicts": verdicts})) == 0) == ok, \
            f.report()["problems"][:3]
        assert ok or any("verdict" in p and "codellama" in p for p in f.report()["problems"])


def other_verdict(tmp_path, of, **change):
    """A verdict file of its own for the model `of` (another name), and its decision record."""
    path = tmp_path / f"other.{of}.json"
    path.write_text(json.dumps({**json.loads(verdict_text(of)), **change}, sort_keys=True))
    return {"verdict": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trained": False}


@pytest.mark.parametrize("case, words", [
    ("fake path, zero sha", "not a file"),
    ("wrong sha", "sha256"),
    ("not passing", "not a passing"),
    ("another model", "verdict of"),
    ("another decision", "canary decision"),
    ("no path", "names no file"),
    ("not a record", "names no file"),
])
def test_coverage_reads_every_verdict_the_decision_names(tmp_path, case, words):
    """Each verdict is read from its path; its sha256 is the recorded one, it passes,
    it is of its model, and its decision is the decision file's; a hand-written decision naming the three models
     or a fake path with sha 0...0 no longer passes."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    doc = json.loads(DECISION_DOC)
    bad = {"fake path, zero sha": lambda: {"verdict": "results/no-such/verdict.json", "sha256": "0" * 64,
                                           "trained": False},
           "wrong sha": lambda: {**doc["verdicts"]["mistral"], "sha256": "0" * 64},
           "not passing": lambda: other_verdict(tmp_path, "mistral", **{"pass": False}),
           "another model": lambda: other_verdict(tmp_path, "mistral", model="deepseek"),
           "another decision": lambda: other_verdict(tmp_path, "mistral", decision={**DECISION, "enforce_eager": True}),
           "no path": lambda: {"sha256": doc["verdicts"]["mistral"]["sha256"]},
           "not a record": lambda: "verdict.mistral.json"}[case]()
    assert with_decision(f, dirs, json.dumps({**doc, "verdicts": {**doc["verdicts"], "mistral": bad}})) == 2
    problems = f.report()["problems"]
    assert any("mistral" in p and words in p for p in problems), problems[:5]
    assert with_decision(f, dirs, DECISION_DOC) == 0, f.report()["problems"][:3]        # the real three verdicts pass


@pytest.mark.parametrize("bad", [{"gpu_memory_utilization": 0.9}, {"pin_set": "custom"},
                                 {"gpu_name": "NVIDIA RTX 4000 Ada Generation"}])
def test_coverage_revalidates_the_decision_itself(tmp_path, bad):
    """The ladder, the pin set and the card of the decision are checked again, not only matched."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    doc = json.loads(DECISION_DOC)
    text = json.dumps({**doc, "decision": {**doc["decision"], **bad}})
    digest = hashlib.sha256(text.encode()).hexdigest()
    for w in dirs:
        rewrite(w, lambda rc: rc["identity"].update(decision_sha256=digest, **{
            k: v for k, v in bad.items() if k in ("pin_set", "gpu_name")}))
    assert main(f.argv(dirs, decision=text)) == 2
    assert any("decision" in p and list(bad)[0].split("_")[0] in p for p in f.report()["problems"]), bad


def test_the_report_records_the_card_the_pin_set_and_the_decision(tmp_path):
    f = Fixture(tmp_path)
    assert main(f.argv(f.runs())) == 0
    r = f.report()
    assert r["gpu_name"] == "NVIDIA RTX A4000" and r["pin_set"] == "main" and r["decision_sha256"] == DECISION_SHA


# Each verdict binds the decision's code, engine and prompt
# directory; the decision's own values are checked even when its verdicts agree with it


NEW_ENGINE = {**ENGINE, "version": "0.28.0"}
HAND_EDITS = {                                  # a new binding in the decision, the workdirs made to agree with it
    "gen_code_sha256": ({**GEN_CODE, "v2gpu/runs.py": "3" * 64},
                        lambda rc: rc["identity"].update(gen_code_sha256={**GEN_CODE, "v2gpu/runs.py": "3" * 64})),
    "engine_settings": ({**SETTINGS, "quantization": "fp8"},
                        lambda rc: rc["identity"].update(engine_settings={**SETTINGS, "quantization": "fp8"})),
    "engine_fingerprint": (engine_fingerprint(NEW_ENGINE), lambda rc: rc.update(engine=NEW_ENGINE)),
    "prompts_sums_sha256": ("6" * 64, lambda rc: None),
}


@pytest.mark.parametrize("key", sorted(HAND_EDITS))
def test_a_hand_edited_decision_binding_is_refused_by_its_verdicts(tmp_path, key):
    """The decision file says a new code, engine or prompt directory while its
    verdicts (sha256 unchanged) ran the old ones; the workdirs agree with the edited decision. Each verdict must carry
    the decision's value of every BINDING key, and main's prompt directory in its probed_sums."""
    value, change = HAND_EDITS[key]
    f = Fixture(tmp_path)
    dirs = f.runs()
    for w in dirs:
        rewrite(w, change)
    doc = json.loads(DECISION_DOC)
    assert with_decision(f, dirs, json.dumps({**doc, key: value})) == 2
    word = "probed_sums" if key == "prompts_sums_sha256" else key
    assert sum(word in p and "verdict of" in p for p in f.report()["problems"]) == 3, f.report()["problems"][:5]
    theirs = {"prompts_sums_sha256": {"probed_sums": [value]}}.get(key, {key: value})
    for m in MODELS3:                                  # the canaries re-run with the new binding: then it passes
        (tmp_path / verdict_name(m)).write_text(verdict_text(m, **theirs))
    doc["verdicts"] = {m: {**doc["verdicts"][m], "sha256": sha(tmp_path / verdict_name(m))} for m in MODELS3}
    assert with_decision(f, dirs, json.dumps({**doc, key: value})) == 0, f.report()["problems"][:5]


def test_main_needs_the_prompt_directory_its_canaries_probed(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    doc = json.loads(DECISION_DOC)
    assert with_decision(f, dirs, json.dumps({**doc, "prompts_sums_sha256": None})) == 2
    assert any("prompts_sums_sha256" in p for p in f.report()["problems"])


def test_a_decision_off_the_ladder_fails_even_when_its_verdicts_agree(tmp_path):
    """The decision's own values (engines.decision_problems) are checked, not only matched with its verdicts."""
    bad = {**DECISION, "gpu_memory_utilization": 0.9}
    f = Fixture(tmp_path)
    dirs = f.runs()
    write_verdicts(tmp_path, bad)
    assert with_decision(f, dirs, decision_doc(bad)) == 2
    problems = f.report()["problems"]
    assert any("pre-declared orders" in p for p in problems) and not any("verdict of" in p for p in problems), \
        problems[:5]


@pytest.mark.parametrize("key", ["gen_code_sha256", "engine_settings", "engine_fingerprint"])
def test_a_decision_that_binds_nothing_fails_even_with_agreeing_verdicts(tmp_path, key):
    """A decision (and verdicts) with a null BINDING value binds nothing: decide never writes one."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    doc = json.loads(DECISION_DOC)
    for m in MODELS3:
        (tmp_path / verdict_name(m)).write_text(verdict_text(m, **{key: None}))
    doc["verdicts"] = {m: {**doc["verdicts"][m], "sha256": sha(tmp_path / verdict_name(m))} for m in MODELS3}
    assert with_decision(f, dirs, json.dumps({**doc, key: None})) == 2
    assert any(f"binds no {key}" in p for p in f.report()["problems"]), f.report()["problems"][:5]


def test_an_engine_record_of_odd_types_never_raises():
    from v2gpu.engines import engine_fingerprint, engine_record_problems

    for odd in (None, "vllm", [], {"name": "vllm"}, {"name": "vllm", "version": "1", "kwargs": None},
                {"name": "vllm", "version": "1", "env": [1]}, {"name": "vllm", "version": "1", "sampling": "x"},
                {"name": "vllm", "version": "1", "lora_shrink": {"checked": None}},
                {"name": "vllm", "version": "1", "lora_shrink": {"checked": [{"m": 1}]}},
                {"name": "vllm", "version": "1", "lora_shrink": [1]}, {"name": "vllm", "version": "1", "effective": 3}):
        assert engine_record_problems(odd) and engine_fingerprint(odd) is None, odd
    assert engine_record_problems(ENGINE) == [] and engine_fingerprint(ENGINE)["version"] == "0.27.1"


# The run manifest for 3C/3D: gencoverage writes manifest.<variant>.jsonl next to coverage.<variant>.json


def manifest_rows(report, root):
    """root is coverage.json's own directory: manifest_path is a basename relative to it."""
    return [json.loads(l) for l in (Path(root) / report["manifest_path"]).read_text().splitlines()]


def test_manifest_ok_rows_carry_line_and_receipt_evidence(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    assert main(f.argv(dirs)) == 0
    report = f.report()
    # manifest_path is the basename, relative to coverage.json's own directory,
    # never a full/absolute path (it was str(a.out.parent / ...), not portable to another
    # checkout, though nothing downstream read it -- the scoring code reads --manifest and the hash).
    assert report["manifest_path"] == "manifest.main.jsonl"
    manifest_path = tmp_path / report["manifest_path"]
    assert report["manifest_sha256"] == sha(manifest_path)
    rows = manifest_rows(report, tmp_path)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert len(ok_rows) == report["found"]
    sample = next(r for r in ok_rows if r["seed"] != "base")
    assert sample["workdir"] == f"gen-main-{sample['model']}-{sample['seed']}"
    assert sample["receipt_status"] is None and sample["reason"] is None and sample["attempts"] is None
    assert sample["evidence_eligible"] is False and sample["line_sha256"] is not None
    receipt = tmp_path / "w" / sample["workdir"] / "receipt-generate.json"
    assert sample["receipt_sha256"] == sha(receipt) and sample["outputs_sha256"] == json.loads(
        receipt.read_text())["outputs_sha256"]
    assert sample["decision_sha256"] == DECISION_SHA and sample["gen_code_sha256"] == GEN_CODE
    assert sample["adapter_tensor_sha256"] == adapter_sha(sample["model"], sample["seed"])
    base = next(r for r in ok_rows if r["seed"] == "base")
    assert base["adapter_tensor_sha256"] is None


def test_manifest_excluded_row_has_its_reason_and_null_hashes(tmp_path):
    f = Fixture(tmp_path)
    assert main(f.argv(f.runs())) == 0
    rows = manifest_rows(f.report(), tmp_path)
    excluded = [r for r in rows if r["status"] == "excluded"]
    assert excluded
    one = next(r for r in excluded if r["record_id"] == "apache/fineract|b.feature|2|s" and r["fmt"] == "F2")
    assert one["reason"] == "too_long_test_rule"
    assert all(one[k] is None for k in ("prompts_sha256", "prompt_sha256", "adapter_tensor_sha256", "decision_sha256",
                                        "gen_code_sha256", "leakcheck_receipt_sha256", "workdir", "receipt_sha256",
                                        "outputs_sha256", "line_sha256", "receipt_status", "attempts"))
    assert one["evidence_eligible"] is False


def test_manifest_missing_row_for_a_done_job_with_one_dropped_line(tmp_path):
    """A plain, otherwise-DONE job missing just one line: receipt_status is DONE (not a failure status at all), so
    evidence_eligible is false; the prompt hashes are still known (the prompt exists, only the line does not)."""
    f = Fixture(tmp_path, drop={("Sylius/Sylius|a.feature|1|s", "full_aware", "mistral", "s23")})
    assert main(f.argv(f.runs())) == 2
    rows = manifest_rows(f.report(), tmp_path)
    missing = [r for r in rows if r["status"] == "missing"]
    assert len(missing) == 1
    row = missing[0]
    assert (row["record_id"], row["model"], row["seed"], row["condition"]) == (
        "Sylius/Sylius|a.feature|1|s", "mistral", "s23", "full_aware")
    assert row["receipt_status"] == "DONE" and row["workdir"] == "gen-main-mistral-s23"
    assert row["line_sha256"] is None and row["evidence_eligible"] is False
    assert row["prompts_sha256"] and row["prompt_sha256"]
    assert len(row["attempts"]) == 1 and row["attempts"][0]["status"] == "DONE"


def test_manifest_missing_row_for_an_absent_job(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    absent = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23"}
    kept = [d for d in dirs if d.name not in absent]
    assert main(f.argv(kept)) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(r["receipt_status"] == "absent" and r["attempts"] == [] and r["workdir"] is None and
              r["evidence_eligible"] is False for r in missing)


def test_manifest_missing_row_crashed_with_two_attempts_is_evidence_eligible(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    target = tmp_path / "w" / "gen-main-mistral-s23"
    (target / "outputs.jsonl").write_text("")
    rewrite(target, lambda rc: rc.update(status="CRASHED", outputs_sha256=sha(target / "outputs.jsonl")))
    (target / "receipt-generate.attempt-1.json").write_text(json.dumps({"status": "INCOMPLETE"}))
    assert main(f.argv(dirs)) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(r["receipt_status"] == "CRASHED" and len(r["attempts"]) == 2 for r in missing)
    assert all(r["evidence_eligible"] is True for r in missing)
    assert all(r["receipt_sha256"] == sha(target / "receipt-generate.json") for r in missing)


def test_manifest_missing_row_refused_is_never_evidence_eligible_even_with_attempts(tmp_path):
    """REFUSED_* never qualifies for the evidence rule, regardless of attempt count."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    target = tmp_path / "w" / "gen-main-mistral-s23"
    (target / "outputs.jsonl").write_text("")
    rewrite(target, lambda rc: rc.update(status="REFUSED_PREFLIGHT", outputs_sha256=sha(target / "outputs.jsonl")))
    (target / "receipt-generate.attempt-1.json").write_text(json.dumps({"status": "REFUSED_PREFLIGHT"}))
    (target / "receipt-generate.attempt-2.json").write_text(json.dumps({"status": "CRASHED"}))
    assert main(f.argv(dirs)) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(r["receipt_status"] == "REFUSED_PREFLIGHT" and len(r["attempts"]) == 3 for r in missing)
    assert all(r["evidence_eligible"] is False for r in missing)


def test_the_same_workdir_in_both_workdirs_and_lost_workdirs_counts_its_attempts_once(tmp_path):
    """--lost-workdirs and --workdirs read the SAME files when a job is given to
    both (an operator mistake, or a job moved but the old path also kept around) -- attempts must be de-duplicated
    by sha256, so "'the same receipt counted twice' never inflates a job past the >= 2 DISTINCT attempts the
    evidence rule requires. Before the fix: attempts == [receipt-generate.json CRASHED, receipt-generate.json
    CRASHED] (1 distinct sha, only 1 real attempt) still gave evidence_eligible: true."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    target = tmp_path / "w" / "gen-main-mistral-s23"
    (target / "outputs.jsonl").write_text("")
    rewrite(target, lambda rc: rc.update(status="CRASHED", outputs_sha256=sha(target / "outputs.jsonl")))
    assert main(f.argv(dirs) + ["--lost-workdirs", str(target)]) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(len(r["attempts"]) == 1 for r in missing), [r["attempts"] for r in missing]
    assert all(r["evidence_eligible"] is False for r in missing)   # 1 distinct attempt, not >= 2


@pytest.mark.parametrize("order", [0, 1])
def test_two_lost_copies_of_the_same_job_merge_regardless_of_argument_order(tmp_path, order):
    """After a "Job crash" retry, GEN §7's copy loop now copies EACH failed
    attempt to a FRESH name (results-lost/<job>.a<timestamp>/<job>) instead of skipping the second one because the
    first copy already exists. lost_evidence must merge every directory of the SAME (model, weights) job -- a union
    of their attempts by sha256 -- not keep only the last one --lost-workdirs happened to list; before this fix, the
    dict assignment `out[key] = {...}` simply overwrote the first copy's entry with the second one's, so which
    directory came LAST in --lost-workdirs (an argument-order accident) silently decided whether 1 or 2 attempts, and
    therefore evidence_eligible, were recorded."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    absent = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23"}
    kept = [d for d in dirs if d.name not in absent]
    rows_for_this = [r for r in f.rows if r["model"] == "mistral" and r["seed"] == "s23"]
    lost_a = write_run(tmp_path / "results-lost", "gen-main-mistral-s23.a1000000000", rows_for_this,
                       identity={"weights": "s23"})
    rewrite(lost_a, lambda rc: rc.update(status="INCOMPLETE"))
    lost_b = write_run(tmp_path / "results-lost", "gen-main-mistral-s23.a2000000000", rows_for_this,
                       identity={"weights": "s23"})
    rewrite(lost_b, lambda rc: rc.update(status="CRASHED"))
    both = [lost_a, lost_b] if order == 0 else [lost_b, lost_a]
    assert main(f.argv(kept) + ["--lost-workdirs", *map(str, both)]) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(len(r["attempts"]) == 2 for r in missing), [r["attempts"] for r in missing]
    assert all(r["evidence_eligible"] is True for r in missing)
    assert all(r["receipt_status"] == "CRASHED" for r in missing)   # the chronologically later copy's own status


@pytest.mark.parametrize("order", [0, 1])
def test_lost_copies_in_the_nested_run_sheet_layout_are_ordered_by_receipt_time(tmp_path, order):
    """GEN §7's REAL layout copies each failed attempt to
    results-lost/<job>.a<timestamp>/<job> (nested one level) -- the leaf folder name is always the bare job name, so
    the OLD sort-by-leaf-name rule tied on every copy and silently fell back to argument order (a wrong "last"
    status, and evidence_eligible true when it should be false, exactly as reproduced). lost_evidence must
    order by each copy's own receipt time instead, so the chronologically later copy always wins, in EITHER
    argument order (unlike the flat-layout test above, whose leaf names happen to embed the timestamp already)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    absent = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23"}
    kept = [d for d in dirs if d.name not in absent]
    rows_for_this = [r for r in f.rows if r["model"] == "mistral" and r["seed"] == "s23"]
    lost_a = write_run(tmp_path / "results-lost" / "gen-main-mistral-s23.a1000000000", "gen-main-mistral-s23",
                       rows_for_this, identity={"weights": "s23"},
                       receipt={"status": "INCOMPLETE", "started_utc": "2026-09-27T10:00:00+00:00",
                                "finished_utc": "2026-09-27T10:05:00+00:00"})
    lost_b = write_run(tmp_path / "results-lost" / "gen-main-mistral-s23.a2000000000", "gen-main-mistral-s23",
                       rows_for_this, identity={"weights": "s23"},
                       receipt={"status": "CRASHED", "started_utc": "2026-09-27T11:00:00+00:00",
                                "finished_utc": "2026-09-27T11:05:00+00:00"})
    assert lost_a.name == lost_b.name == "gen-main-mistral-s23"          # the nested layout's shared leaf name
    both = [lost_a, lost_b] if order == 0 else [lost_b, lost_a]
    assert main(f.argv(kept) + ["--lost-workdirs", *map(str, both)]) == 2
    missing = [r for r in manifest_rows(f.report(), tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(len(r["attempts"]) == 2 for r in missing), [r["attempts"] for r in missing]
    assert all(r["evidence_eligible"] is True for r in missing)
    assert all(r["receipt_status"] == "CRASHED" for r in missing)   # lost_b: the chronologically later copy, always


def test_a_failed_jobs_pairs_via_lost_workdirs_are_missing_status_fail_except_missing_pass(tmp_path):
    """The intended flow -- a failed job's workdir goes ONLY into
    --lost-workdirs, never --workdirs (GEN §6, "chưa DONE" rule) -- gives exactly this shape: the pair is `missing`,
    overall `status` is FAIL (a pair is still missing), but `status_except_missing` is PASS (nothing else is wrong:
    the failed workdir is not in --workdirs, so workdir_problems never sees it) and `problems` is empty."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    absent = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23"}
    kept = [d for d in dirs if d.name not in absent]
    rows_for_this = [r for r in f.rows if r["model"] == "mistral" and r["seed"] == "s23"]
    lost = write_run(tmp_path / "results-lost", "gen-main-mistral-s23", rows_for_this, identity={"weights": "s23"})
    rewrite(lost, lambda rc: rc.update(status="CRASHED"))
    (lost / "receipt-generate.attempt-1.json").write_text(json.dumps({"status": "INCOMPLETE"}))
    assert main(f.argv(kept) + ["--lost-workdirs", str(lost)]) == 2
    report = f.report()
    assert report["status"] == "FAIL" and report["status_except_missing"] == "PASS" and report["problems"] == []
    missing = [r for r in manifest_rows(report, tmp_path)
              if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing and all(r["receipt_status"] == "CRASHED" and r["evidence_eligible"] is True for r in missing)


def test_a_non_done_gen_workdir_in_workdirs_gives_missing_rows_never_ok(tmp_path):
    """The opposite of the intended flow above -- a non-DONE gen workdir given
    directly in --workdirs (an operator mistake, or a partial workdir left in results/ by hand) must never let its
    outputs.jsonl lines become `ok` rows, even when those lines are perfectly well-formed. Before the fix: a CRASHED
    workdir's valid-looking lines in --workdirs became `ok` manifest rows from a job with no recheck -- the exact
    defect a sample run accidentally demonstrated."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    target = tmp_path / "w" / "gen-main-mistral-s23"
    rewrite(target, lambda rc: rc.update(status="CRASHED"))          # outputs.jsonl itself is untouched: still valid
    assert main(f.argv(dirs)) == 2
    rows = manifest_rows(f.report(), tmp_path)
    this = [r for r in rows if r["model"] == "mistral" and r["seed"] == "s23"]
    assert this and any(r["status"] == "missing" for r in this)   # excluded rows (unrelated to DONE) may coexist
    assert not any(r["status"] == "ok" for r in this)


def test_manifest_lost_workdirs_are_evidence_only_never_a_source_of_output_lines(tmp_path):
    """Evidence from failed jobs: --lost-workdirs is read for its receipts and attempts only; its
    outputs.jsonl (even with perfectly valid lines) is never scored."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    absent = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23"}
    kept = [d for d in dirs if d.name not in absent]
    rows_for_this = [r for r in f.rows if r["model"] == "mistral" and r["seed"] == "s23"]
    lost = write_run(tmp_path / "results-lost", "gen-main-mistral-s23", rows_for_this, identity={"weights": "s23"})
    rewrite(lost, lambda rc: rc.update(status="CRASHED"))
    (lost / "receipt-generate.attempt-1.json").write_text(json.dumps({"status": "INCOMPLETE"}))
    assert main(f.argv(kept) + ["--lost-workdirs", str(lost)]) == 2
    rows = manifest_rows(f.report(), tmp_path)
    missing = [r for r in rows if r["status"] == "missing" and r["model"] == "mistral" and r["seed"] == "s23"]
    assert missing
    assert all(r["receipt_status"] == "CRASHED" and r["workdir"] == "gen-main-mistral-s23" and
              len(r["attempts"]) == 2 and r["evidence_eligible"] is True for r in missing)
    assert not any(r["status"] == "ok" for r in rows if r["model"] == "mistral" and r["seed"] == "s23")


def test_manifest_is_byte_identical_across_two_runs(tmp_path):
    f = Fixture(tmp_path)
    argv = f.argv(f.runs())
    assert main(argv) == 0
    manifest_path = tmp_path / f.report()["manifest_path"]
    first = manifest_path.read_bytes()
    assert main(argv) == 0
    second = manifest_path.read_bytes()
    assert first == second and len(first) > 0


def test_status_except_missing_pass_when_the_only_problem_is_missing_pairs(tmp_path):
    f = Fixture(tmp_path, drop={("Sylius/Sylius|a.feature|1|s", "full_aware", "mistral", "s23")})
    assert main(f.argv(f.runs())) == 2
    report = f.report()
    assert report["status"] == "FAIL" and report["status_except_missing"] == "PASS"


def test_the_by_design_note_is_in_the_module_docstring_too():
    """The "a non-DONE workdir in --workdirs makes status_except_missing FAIL
    by design" note used to live only in GEN §6, GEN §7's copy loop context and a code comment on
    status_except_missing itself -- never in gencoverage's own module docstring, where a reader of the module (not
    the operating notes) would look first."""
    import v2gpu.gencoverage as gc
    doc = gc.__doc__
    assert "status_except_missing" in doc and "FAIL BY DESIGN" in doc
    assert "--workdirs" in doc and "--lost-workdirs" in doc


def test_status_except_missing_fail_when_another_problem_exists(tmp_path):
    f = Fixture(tmp_path)
    dirs = f.runs()
    extra = write_run(tmp_path / "w", "gen-main-extra", f.rows[:1])
    assert main(f.argv(dirs + [extra, write_recheck(tmp_path / "w", extra)])) == 2
    report = f.report()
    assert report["status"] == "FAIL" and report["status_except_missing"] == "FAIL"


def test_recheck_mismatch_fails_status_except_missing_and_is_not_a_manifest_row(tmp_path):
    """RECHECK_MISMATCH: the whole coverage FAILs, status_except_missing FAILs too, and it is not
    a manifest row (recheck workdirs are never rows)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    recheck = tmp_path / "w" / "recheck-gen-main-mistral-s23"
    rewrite(recheck, lambda rc: rc.update(status="RECHECK_MISMATCH"))
    assert main(f.argv(dirs)) == 2
    report = f.report()
    assert report["status"] == "FAIL" and report["status_except_missing"] == "FAIL"
    rows = manifest_rows(report, tmp_path)
    assert not any(r["workdir"] == "recheck-gen-main-mistral-s23" for r in rows)


def test_manifest_sample_for_the_scoring_step(tmp_path):
    """The run manifest for 3C/3D, built through the intended
    flow: a sample manifest+coverage pair, generated from this file's own test fixture, for the scoring code.
    Contains, by construction: one "ok" row (plus a matching outputs.jsonl sample, so line_sha256 can be checked on
    raw bytes); one "excluded" row; one "missing" row with evidence_eligible true (CRASHED, 2 attempts, its workdir
    given ONLY via --lost-workdirs -- the intended flow); one "missing" REFUSED_* row (also --lost-workdirs only)
    and one "missing" absent row, both evidence_eligible false. Because the two failed workdirs are never in
    --workdirs, overall status is FAIL (pairs are still missing) but status_except_missing is PASS and problems is
    empty (this is what the intended flow gives; putting a failed workdir in --workdirs instead is the
    mistake status_except_missing is meant to catch, and the old sample, before this fix, accidentally demonstrated
    exactly that mistake)."""
    f = Fixture(tmp_path)
    dirs = f.runs()
    lost_names = {"gen-main-mistral-s23", "recheck-gen-main-mistral-s23",
                 "gen-main-deepseek-s11", "recheck-gen-main-deepseek-s11"}
    absent = {"gen-main-codellama-s11", "recheck-gen-main-codellama-s11"} | lost_names
    crashed = tmp_path / "w" / "gen-main-mistral-s23"
    refused = tmp_path / "w" / "gen-main-deepseek-s11"
    (crashed / "outputs.jsonl").write_text("")
    rewrite(crashed, lambda rc: rc.update(status="CRASHED", outputs_sha256=sha(crashed / "outputs.jsonl")))
    (crashed / "receipt-generate.attempt-1.json").write_text(json.dumps({"status": "INCOMPLETE"}))
    (refused / "outputs.jsonl").write_text("")
    rewrite(refused, lambda rc: rc.update(status="REFUSED_PREFLIGHT", outputs_sha256=sha(refused / "outputs.jsonl")))
    kept = [d for d in dirs if d.name not in absent]
    assert main(f.argv(kept) + ["--lost-workdirs", str(crashed), str(refused)]) == 2
    report = f.report()
    assert report["status"] == "FAIL" and report["status_except_missing"] == "PASS" and report["problems"] == []
    rows = manifest_rows(report, tmp_path)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert ok_rows
    ok_sample = ok_rows[0]
    assert any(r["status"] == "excluded" for r in rows)
    assert any(r["status"] == "missing" and r["evidence_eligible"] is True for r in rows)
    assert any(r["status"] == "missing" and r["receipt_status"] == "REFUSED_PREFLIGHT" and
              r["evidence_eligible"] is False for r in rows)
    assert any(r["status"] == "missing" and r["receipt_status"] == "absent" and r["evidence_eligible"] is False
              for r in rows)
    # Pin the actual shape the README below describes in words -- 3 missing
    # jobs (not two: mistral s23, deepseek s11 and codellama s11), 18 missing rows in total.
    missing_rows = [r for r in rows if r["status"] == "missing"]
    assert len(missing_rows) == 18
    assert {(r["model"], r["seed"]) for r in missing_rows} == {
        ("mistral", "s23"), ("deepseek", "s11"), ("codellama", "s11")}
    # Since 27/09/2026 the sample copy for the scoring code is opt-in via an env var, never a hardcoded path
    # of one machine's own temporary directory -- baked in, it made this test fail closed (PermissionError) on any
    # other machine, including the gate's own container, since that temporary directory never
    # exists there. The functional assertions above (row shapes) run unconditionally on every machine.
    sample_dir_env = os.environ.get("V2GPU_MANIFEST_SAMPLE_DIR")
    if not sample_dir_env:
        return
    sample_dir = Path(sample_dir_env)
    sample_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(f.root / report["manifest_path"], sample_dir / "manifest.main.jsonl")
    shutil.copy(f.root / "coverage.json", sample_dir / "coverage.main.json")
    # An outputs.jsonl sample matching the "ok" row's own workdir, so the scoring code can check line_sha256 against
    # the exact raw bytes of a real line (only the line the manifest row itself points to, not the whole file, to
    # avoid the sample growing every model's full output).
    (sample_dir / "outputs.sample.jsonl").write_text(
        next(l for l in jsonl_lines(tmp_path / "w" / ok_sample["workdir"] / "outputs.jsonl")
             if hashlib.sha256(l.encode("utf-8")).hexdigest() == ok_sample["line_sha256"]) + "\n")
    (sample_dir / "README.md").write_text(
        "# manifest.main.jsonl / coverage.main.json / outputs.sample.jsonl\n\n"
        "The run manifest for 3C/3D, built through the "
        "intended flow. Generated from tests_gpu/test_gencoverage.py's own Fixture (variant \"main\", branch "
        "v2-runpod-f4) by test_manifest_sample_for_the_scoring_step.\n\n"
        "Rows included, by construction:\n"
        f"- one \"ok\" row ({ok_sample['workdir']}, record {ok_sample['record_id']}, fmt {ok_sample['fmt']}): its "
        "exact output line is outputs.sample.jsonl (one line, raw bytes -- check line_sha256 against it directly).\n"
        "- one \"excluded\" row: apache/fineract|b.feature|2|s, fmt F2, reason too_long_test_rule.\n"
        "- one \"missing\" row with evidence_eligible true: gen-main-mistral-s23, forced CRASHED with an "
        "attempt-1.json (INCOMPLETE), 2 attempts total, given ONLY via --lost-workdirs.\n"
        "- one \"missing\" REFUSED_* row, evidence_eligible false: gen-main-deepseek-s11, REFUSED_PREFLIGHT, also "
        "given ONLY via --lost-workdirs.\n"
        "- one \"missing\" absent row, evidence_eligible false: gen-main-codellama-s11, left out of --workdirs and "
        "--lost-workdirs entirely.\n\n"
        f"Overall: status {report['status']!r}, status_except_missing {report['status_except_missing']!r}, "
        "problems []. Neither failed workdir is in --workdirs (the intended rule); status is FAIL because of "
        "3 missing jobs (18 rows): mistral s23, deepseek s11 and codellama s11 -- status_except_missing is PASS.\n")
    assert (sample_dir / "manifest.main.jsonl").exists() and (sample_dir / "coverage.main.json").exists()
    assert (sample_dir / "outputs.sample.jsonl").exists()


def test_lost_evidence_docstring_matches_its_receipt_time_order():
    """lost_evidence orders a pair's copies by each copy's receipt time,
    name only as the tiebreak; its docstring once said "by NAME"."""
    from v2gpu.gencoverage import lost_evidence

    doc = " ".join(lost_evidence.__doc__.split())
    assert "ordered by sorting the directories by NAME" not in doc and "receipt time" in doc


# decision 04/10: a dev grid label plans Full-aware only, so coverage expects only Full-aware rows for it


@pytest.mark.parametrize("variant, conditions", [
    ("dev_grid_k4_n1_r033", ["full_aware"]), ("dev_grid_v1_k12_n2_r067", ["full_aware"]),
    ("dev_pilot_v0", sorted(["zero_shot", "rag_only", "ft_only", "full_neutral", "full_aware"]))])
def test_the_manifest_of_a_dev_grid_label_has_full_aware_rows_only(tmp_path, monkeypatch, variant, conditions):
    from v2gpu import gencoverage

    def fake_row(run, rid, fmt, variant, *rest):
        return {"model": run.model, "condition": run.condition, "seed": run.seed, "record_id": rid, "fmt": fmt}
    monkeypatch.setattr(gencoverage, "_missing_row", fake_row)
    index = {("deepseek", rid, fmt): ("x", "p1", "p2") for rid in ("r1", "r2") for fmt in ("F0", "F1", "F2")}
    gencoverage.write_manifest(tmp_path / "m.jsonl", variant, ["dev"], {"dev": ["r1", "r2"]}, {"deepseek": {}}, index,
                               {}, {}, {}, {}, {}, None)
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert sorted({r["condition"] for r in rows}) == conditions
    assert len(rows) == 2 * len(conditions)
    if variant.startswith("dev_grid"):
        assert {r["fmt"] for r in rows} == {"F2"} and {r["seed"] for r in rows} == {"s42"}
