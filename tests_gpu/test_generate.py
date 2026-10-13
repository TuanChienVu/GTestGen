# tests_gpu/test_generate.py
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests_gpu.test_audit import V0, main as audit, setup
from tests_gpu.tiny import make_tiny_llama
from v2gpu import generate
from v2gpu.examples import jsonl_lines
from v2gpu.modeling import LoadCfg, load_for_training
from v2gpu.receipt import tensor_digest
from v2gpu.runs import prompt_sha256, row_problems


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, tokenizers):
    """A tiny random Llama with DeepSeek's vocabulary (the audit fixture builds DeepSeek prompts), one peft adapter."""
    d = tmp_path_factory.mktemp("tiny")
    path = make_tiny_llama(d / "m", len(tokenizers["deepseek"]))
    model, _ = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32",
                                         gradient_checkpointing=False), seed=42)
    model.save_pretrained(d / "ad")
    return path, d / "ad", tensor_digest(d / "ad" / "adapter_model.safetensors")


@pytest.fixture
def job(tmp_path, tiny):
    """Audited prompts + receipt, and a function giving the argv of a CPU test job (tiny model, HF engine)."""
    assert audit(setup(tmp_path)) == 0
    d, r = tmp_path / "prompts", tmp_path / "leakcheck-receipt.json"

    def argv(workdir="run", weights="base", *extra, mode="test", engine="hf", cap="4", cpu=True):
        a = ["--model", "deepseek", "--weights", weights, "--variant", "main", "--partitions", "test", "aux_test",
             "--prompts", str(d / "test.deepseek.jsonl"), "--prompts-sums", str(d / "PROMPTS_SHA256SUMS"),
             "--leakcheck-receipt", str(r), "--leakcheck-receipt-sha256", sha(r), "--workdir", str(tmp_path / workdir),
             "--mode", mode]
        if engine:
            a += ["--engine", engine]
        if cpu:
            a += ["--model-path", tiny[0], "--quant", "none", "--dtype", "float32"]
        if cap:
            a += ["--cap", cap]
        if weights != "base":
            a += ["--adapter", str(tiny[1]), "--adapter-sha256", tiny[2],
                  "--adapter-config-sha256", sha(Path(tiny[1]) / "adapter_config.json")]
        if mode == "job":                                    # genqueue writes the decision's sha256
            a += ["--decision-sha256", "d" * 64]
        return a + list(extra)

    argv.dir, argv.receipt, argv.root = d, r, tmp_path
    return argv


def rows(path):
    return [json.loads(x) for x in jsonl_lines(path)]                  # b"\n" only, as the job writes


def receipt(workdir):
    return json.loads((workdir / "receipt-generate.json").read_text())


def test_a_base_job_writes_one_valid_line_per_record_and_condition(job):
    assert generate.main(job()) == 0
    out = rows(job.root / "run" / "outputs.jsonl")
    assert sorted((r["condition"], r["record_id"]) for r in out) == sorted(
        (c, q) for c in ("zero_shot", "rag_only") for q in ("a1", "x1", "x2"))
    assert all(row_problems(r, window_rule=False) == [] for r in out)       # tiny model, cap 4: no budget rule
    prompts = {(json.loads(x)["record_id"], json.loads(x)["fmt"]): json.loads(x)["messages"]
               for x in jsonl_lines(job.dir / "test.deepseek.jsonl")}
    for r in out:
        assert r["seed"] == "base" and r["adapter_tensor_sha256"] is None and r["variant"] == "main"
        assert r["leakcheck_receipt_sha256"] == sha(job.receipt)
        assert r["prompts_sha256"] == sha(job.dir / "test.deepseek.jsonl")
        assert r["prompt_sha256"] == prompt_sha256(prompts[(r["record_id"], r["fmt"])])
        assert r["max_new_tokens"] == 4 and r["new_tokens"] <= 4
    rc = receipt(job.root / "run")
    assert rc["status"] == "DONE" and rc["schema"] == "gtestgen.v2.generate_receipt.v1"
    assert rc["result"]["generated"] == 6 and rc["result"]["resumed"] == 0 and rc["engine"]["name"] == "hf"
    assert rc["outputs_sha256"] == sha(job.root / "run" / "outputs.jsonl") and len(rc["preflight"]["ids_sha256"]) == 64
    assert rc["identity"]["leakcheck_receipt_sha256"] == sha(job.receipt)


def test_an_adapter_job_carries_the_adapter_digest(job, tiny):
    assert generate.main(job("ad", "s42")) == 0
    out = rows(job.root / "ad" / "outputs.jsonl")
    assert len(out) == 9 and {r["adapter_tensor_sha256"] for r in out} == {tiny[2]}
    assert {r["condition"] for r in out} == {"ft_only", "full_neutral", "full_aware"}
    assert {r["seed"] for r in out} == {"s42"}


def test_a_killed_job_resumes_without_duplicates_and_gives_the_same_bytes(job):
    assert generate.main(job("full")) == 0
    full = (job.root / "full" / "outputs.jsonl").read_bytes()
    assert generate.main(job("cut")) == 0
    lines = full.split(b"\n")
    (job.root / "cut" / "outputs.jsonl").write_bytes(b"\n".join(lines[:4]) + b"\n" + lines[4][:17])   # torn 5th line
    killed = dict(receipt(job.root / "cut"), status="RUNNING")          # what a SIGTERM from the queue leaves
    del killed["outputs_sha256"]
    (job.root / "cut" / "receipt-generate.json").write_text(json.dumps(killed))
    assert generate.main(job("cut")) == 0
    assert (job.root / "cut" / "outputs.jsonl").read_bytes() == full
    rc = receipt(job.root / "cut")
    assert rc["result"]["resumed"] == 4 and rc["result"]["generated"] == 2 and rc["result"]["truncated_tail"] == 17
    assert json.loads((job.root / "cut" / "receipt-generate.attempt-1.json").read_text())["status"] == "RUNNING"


def test_the_order_of_the_pairs_does_not_change_any_line(job):
    assert generate.main(job("fwd")) == 0
    assert generate.main(job("rev", "base", "--reverse")) == 0
    fwd, rev = (job.root / w / "outputs.jsonl" for w in ("fwd", "rev"))
    assert fwd.read_bytes() != rev.read_bytes()                                  # written in another order
    assert sorted(fwd.read_bytes().split(b"\n")) == sorted(rev.read_bytes().split(b"\n"))


def test_a_rerun_of_a_done_job_loads_no_engine(job, monkeypatch):
    assert generate.main(job()) == 0
    before = (job.root / "run" / "outputs.jsonl").read_bytes()
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    assert generate.main(job()) == 0
    assert (job.root / "run" / "outputs.jsonl").read_bytes() == before


def test_a_workdir_of_another_job_is_refused(job):
    assert generate.main(job()) == 0
    before = (job.root / "run" / "outputs.jsonl").read_bytes()
    assert generate.main(job("run", "base", cap="3")) == 2                        # same workdir, other settings
    assert (job.root / "run" / "outputs.jsonl").read_bytes() == before
    (job.root / "other").mkdir()
    (job.root / "other" / "notes.txt").write_text("x")
    assert generate.main(job("other")) == 2


def test_the_same_job_started_twice_is_refused_without_touching_the_running_one(job):
    """A second process on a workdir whose job is running (it holds the workdir's lock) must not rename
    or overwrite the running job's receipt, nor open its outputs."""
    import fcntl
    import os

    assert generate.main(job()) == 0
    before = {p.name: p.read_bytes() for p in (job.root / "run").iterdir()}
    held = os.open(job.root / "run", os.O_RDONLY)                    # the running job
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert generate.main(job()) == 2
    finally:
        os.close(held)
    assert {p.name: p.read_bytes() for p in (job.root / "run").iterdir()} == before
    assert generate.main(job()) == 0                                  # lock released: the DONE job is recognised


@pytest.mark.parametrize("field, value", [("prompts_sha256", "f" * 64), ("prompt_tokens", None)])
def test_a_resumed_line_that_differs_from_the_job_inputs_is_refused(job, field, value):
    """Same job.json, a line whose file digest or prompt length is not this job's."""
    assert generate.main(job()) == 0
    out = job.root / "run" / "outputs.jsonl"
    lines = jsonl_lines(out)
    first = json.loads(lines[0])
    first[field] = value if value is not None else first[field] + 1
    out.write_text("\n".join([json.dumps(first, ensure_ascii=False, sort_keys=True), *lines[1:]]) + "\n")
    (job.root / "run" / "receipt-generate.json").unlink()
    assert generate.main(job()) == 2
    rc = receipt(job.root / "run")
    assert rc["status"] == "REFUSED_OUTPUTS" and field in rc["problems"][0]


def test_outputs_written_on_another_pod_are_refused(job, monkeypatch, capsys):
    """One pod per volume path; job.json names the pod."""
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-a")
    assert generate.main(job()) == 0
    assert json.loads((job.root / "run" / "job.json").read_text())["pod_id"] == "pod-a"
    (job.root / "run" / "receipt-generate.json").unlink()
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-b")
    assert generate.main(job()) == 2 and "one pod per volume path" in capsys.readouterr().err


def test_outputs_the_job_cannot_account_for_are_refused(job):
    assert generate.main(job()) == 0
    out = job.root / "run" / "outputs.jsonl"
    first = json.loads(jsonl_lines(out)[0])
    out.write_text(out.read_text() + json.dumps({**first, "record_id": "zz"}, sort_keys=True) + "\n")
    (job.root / "run" / "receipt-generate.json").unlink()
    assert generate.main(job()) == 2
    assert receipt(job.root / "run")["status"] == "REFUSED_OUTPUTS"


def test_a_preflight_problem_refuses_before_any_engine_or_output(job, monkeypatch):
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    argv = job()
    argv[argv.index("--leakcheck-receipt-sha256") + 1] = "0" * 64
    assert generate.main(argv) == 2
    rc = receipt(job.root / "run")
    assert rc["status"] == "REFUSED_PREFLIGHT" and any("leak-audit receipt sha256" in p for p in rc["problems"])
    assert not (job.root / "run" / "outputs.jsonl").exists()
    monkeypatch.undo()
    assert generate.main(job()) == 0                  # corrected command, same workdir: nothing was generated before
    assert receipt(job.root / "run")["argv"] == job()


@pytest.mark.parametrize("extra, words", [
    (["--cap", "4"], "--cap"), (["--sample", "2"], "--sample"), (["--reverse"], "--reverse"),
    (["--engine", "hf"], "--engine"), (["--model-path", "x"], "--model-path"),
    (["--probe-longest"], "--probe-longest")])
def test_job_mode_locks_the_generation_settings(job, capsys, extra, words):
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    assert generate.main(argv + extra) == 2
    assert words in capsys.readouterr().err and not (job.root / "j").exists()


def test_job_mode_needs_the_card_name_and_a_real_variant(job, capsys):
    assert generate.main(job("j", "base", mode="job", engine=None, cap=None, cpu=False)) == 2
    assert "--gpu-name" in capsys.readouterr().err
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    argv[argv.index("--variant") + 1] = "gen_canary"
    assert generate.main(argv) == 2 and "gen_canary" in capsys.readouterr().err


def test_dev_grid_labels_are_job_variants_and_malformed_labels_argparse_errors(job, capsys):
    """--variant goes through runs.variant_spec; a grid label passes the mode check (it is refused later,
    only because this fixture's partitions are test and aux_test), a malformed label never reaches the job."""
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    argv[argv.index("--variant") + 1] = "dev_grid_v1_k12_n1_r067"
    assert generate.main(argv) == 2
    rc = receipt(job.root / "j")
    assert rc["status"] == "REFUSED_PREFLIGHT" and rc["identity"]["variant"] == "dev_grid_v1_k12_n1_r067"
    assert any("no runs on partition 'test'" in p for p in rc["problems"])
    argv[argv.index("--variant") + 1] = "dev_grid_k8_n3_r50"
    with pytest.raises(SystemExit) as e:
        generate.main(argv)
    assert e.value.code == 2 and "dev_grid_k8_n3_r50" in capsys.readouterr().err


def fake_environment(packages, device=None):
    return lambda: {"python": "3.12", "platform": "test", "packages": packages, "cuda_available": device is not None,
                    **({"device": device} if device else {})}


A4000 = {"name": "NVIDIA RTX A4000", "capability": "8.6", "total_memory_gb": 16.86}


def test_job_mode_off_the_generation_venv_is_refused_before_the_engine(job, monkeypatch):
    """Independent of the venv the suite runs in (the gen venv has the pins; the laptop has not)."""
    from v2gpu.engines import GEN_PINS

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    monkeypatch.setattr(generate, "environment", fake_environment({**GEN_PINS, "vllm": "0.26.0"}))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    assert generate.main(argv) == 2
    rc = receipt(job.root / "j")
    assert rc["status"] == "REFUSED_ENVIRONMENT"
    assert rc["problems"] == ["vllm '0.26.0' installed; pin set 'main' pins 0.27.1", "no CUDA device"]


def test_job_mode_with_the_pins_and_no_card_is_refused_for_the_card_only(job, monkeypatch):
    from v2gpu.engines import GEN_PINS

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    monkeypatch.setattr(generate, "environment", fake_environment(dict(GEN_PINS)))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    assert generate.main(argv) == 2
    assert receipt(job.root / "j")["problems"] == ["no CUDA device"]


def test_memory_steps_and_the_pin_set_are_recorded_job_settings(job, monkeypatch):
    """decision 24/09: the canary's choices travel in the job command and land in the job identity (so a
    resumed job with other choices is refused, and gencoverage compares them across workdirs)."""
    from v2gpu.engines import PIN_SETS

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    monkeypatch.setattr(generate, "environment", fake_environment(dict(PIN_SETS["fallback-bnb"])))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + [
        "--gpu-name", "NVIDIA RTX A4000", "--gpu-memory-utilization", "0.95", "--enforce-eager", "--pin-set",
        "fallback-bnb"]
    assert generate.main(argv) == 2
    rc = receipt(job.root / "j")
    assert rc["problems"] == ["no CUDA device"] and rc["identity"]["pin_set"] == "fallback-bnb"
    settings = rc["identity"]["engine_settings"]
    assert settings["gpu_memory_utilization"] == 0.95 and settings["enforce_eager"] is True
    assert settings["max_num_batched_tokens"] == 8192
    monkeypatch.setattr(generate, "environment", fake_environment(dict(PIN_SETS["main"])))
    assert generate.main(argv) == 2
    assert any("bitsandbytes" in p and "fallback-bnb" in p for p in receipt(job.root / "j")["problems"])


@pytest.mark.parametrize("extra", [["--gpu-memory-utilization", "0.9"], ["--max-num-batched-tokens", "4096"],
                                   ["--pin-set", "custom"]])
def test_only_the_pre_declared_memory_steps_and_pin_sets_exist(job, capsys, extra):
    """decision 24/09: 0.95 and enforce_eager are pre-declared; the prefill chunk 4,096 is not (no option sets it)."""
    with pytest.raises(SystemExit) as e:
        generate.main(job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "X", *extra])
    assert e.value.code == 2 and extra[0] in capsys.readouterr().err


def test_decision_0510_a_command_without_memory_flags_runs_at_the_determinism_step():
    """decision 05/10: (0.95, enforce_eager) is the default of every command; genqueue still writes the canary's
    decision into every job, and a real decision can only be this step (the vLLM engine refuses any other)."""
    from v2gpu.engines import DETERMINISM_STEP

    ap = generate.build_parser()
    assert (ap.get_default("gpu_memory_utilization"), ap.get_default("enforce_eager")) == DETERMINISM_STEP


def test_a_set_that_tokenizes_differently_needs_the_laptops_prompt_ids(job, capsys):
    """decision 24/09: transformers 5.15.0 (with tokenizers 0.22.2) is accepted only if the prompt ids equal the
    training venv's: the job command then carries the laptop's ids_sha256, and other ids refuse the job."""
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000",
                                                                            "--pin-set", "fallback-tf"]
    assert generate.main(argv) == 2 and "--ids-sha256" in capsys.readouterr().err
    i = job().index("--workdir")
    assert generate.main(job()[:i] + job()[i + 2:] + ["--preflight-only"]) == 0
    ids = json.loads(capsys.readouterr().out)["ids_sha256"]
    assert generate.main(job("bad", "base", "--ids-sha256", "0" * 64)) == 2
    rc = receipt(job.root / "bad")
    assert rc["status"] == "REFUSED_PREFLIGHT" and any("ids_sha256" in p for p in rc["problems"])
    assert generate.main(job("good", "base", "--ids-sha256", ids)) == 0


def test_a_card_without_a_tuned_lora_shrink_file_is_refused(job, monkeypatch, tmp_path):
    """vLLM falls back to split-K silently when <card>_SHRINK.json is missing (here: a folder holding the
    A4000 file only, on an RTX 2000 Ada)."""
    from v2gpu import engines
    from v2gpu.engines import GEN_PINS

    folder = tmp_path / "lora_ops"
    folder.mkdir()
    a4000 = "NVIDIA_RTX_A4000_SHRINK.json"
    (folder / a4000).write_bytes((engines.LORA_OPS / a4000).read_bytes())
    monkeypatch.setattr(engines, "LORA_OPS", folder)
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    ada = {"name": "NVIDIA RTX 2000 Ada Generation", "capability": "8.9", "total_memory_gb": 16.7}
    monkeypatch.setattr(generate, "environment", fake_environment(dict(GEN_PINS), ada))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", ada["name"]]
    assert generate.main(argv) == 2
    assert any("NVIDIA_RTX_2000_Ada_Generation_SHRINK.json" in p for p in receipt(job.root / "j")["problems"])


@pytest.mark.parametrize("mode", ["job", "canary"])
def test_the_gpu_name_is_one_of_the_generation_card_classes(job, capsys, mode):
    """decision 24/09: A4000 or RTX 2000 Ada, named before anything runs."""
    argv = job("j", "base", mode=mode, engine=None if mode == "job" else "hf", cap=None, cpu=False)
    if mode == "canary":
        argv[argv.index("--variant") + 1] = "gen_canary"
        argv[argv.index("--partitions") + 1:argv.index("--partitions") + 3] = ["dev"]
    assert generate.main(argv + ["--gpu-name", "NVIDIA RTX A5000"]) == 2
    assert "card class" in capsys.readouterr().err and not (job.root / "j").exists()


def test_test_mode_writes_job_variants_only_for_the_test_suite(job, monkeypatch, capsys):
    """An operator reusing a --mode test command cannot write "main" lines with another engine or cap."""
    monkeypatch.delenv("V2GPU_TEST_JOB_VARIANTS", raising=False)
    assert generate.main(job()) == 2 and "--mode job" in capsys.readouterr().err
    assert not (job.root / "run").exists()
    argv = job("c")
    argv[argv.index("--variant") + 1] = "gen_canary"
    assert generate.main(argv) == 2 and "test/aux_test" in capsys.readouterr().err     # test text
    assert not (job.root / "c").exists()


def test_canary_mode_reads_test_prompts_only_for_the_probe(job, capsys):
    """A canary run on test prompts would produce test outputs before the compose lock."""
    argv = job("c", "base", mode="canary") + ["--gpu-name", "NVIDIA RTX A4000"]
    argv[argv.index("--variant") + 1] = "gen_canary"
    assert generate.main(argv) == 2 and "--probe-longest" in capsys.readouterr().err


def test_hf_hub_offline_is_set_before_huggingface_hub_is_imported():
    """The huggingface_hub constant is frozen at import, so setting it later has no effect."""
    code = ("import os, sys\n"
            "from v2gpu import generate\n"
            "assert 'huggingface_hub' not in sys.modules\n"
            "try:\n    generate.main(['--help'])\nexcept SystemExit:\n    pass\n"
            "import huggingface_hub.constants as c\nprint(c.HF_HUB_OFFLINE)\n")
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "True"


def test_preflight_only_needs_no_workdir(job, capsys):
    argv = job()
    i = argv.index("--workdir")
    assert generate.main(argv[:i] + argv[i + 2:] + ["--preflight-only"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_problems"] == 0 and out["pairs"] == 6


def test_recheck_regenerates_a_sample_in_a_fresh_job_and_compares_bytes(job, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-a")
    assert generate.main(job("first")) == 0
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-b")                      # a copy of the workdir on another pod
    assert generate.main(job("again", "base", "--recheck-of", str(job.root / "first"), "--sample", "1")) == 0
    rc = receipt(job.root / "again")
    assert rc["status"] == "DONE" and rc["recheck"]["identical"] is True
    assert rc["recheck"]["of_pod_id"] == "pod-a" and rc["pod_id"] == "pod-b"
    assert rc["recheck"]["of_outputs_sha256"] == sha(job.root / "first" / "outputs.jsonl")
    out = job.root / "first" / "outputs.jsonl"
    lines = jsonl_lines(out)
    picked = {tuple(x) for x in rc["recheck"]["keys"]}
    for i, x in enumerate(lines):
        r = json.loads(x)
        if (r["record_id"], r["model"], r["seed"], r["condition"], r["variant"]) in picked:
            lines[i] = json.dumps({**r, "text": r["text"] + " changed"}, ensure_ascii=False, sort_keys=True)
            break
    out.write_text("\n".join(lines) + "\n")
    assert generate.main(job("third", "base", "--recheck-of", str(job.root / "first"), "--sample", "1")) == 2
    assert receipt(job.root / "third")["status"] == "REFUSED_RECHECK"      # outputs changed after DONE
    rc0 = json.loads((job.root / "first" / "receipt-generate.json").read_text())
    rc0["outputs_sha256"] = sha(out)
    (job.root / "first" / "receipt-generate.json").write_text(json.dumps(rc0))
    assert generate.main(job("third", "base", "--recheck-of", str(job.root / "first"), "--sample", "1")) == 1
    rc = receipt(job.root / "third")
    assert rc["status"] == "RECHECK_MISMATCH" and rc["recheck"]["identical"] is False
    assert len(rc["recheck"]["differ"]) == 1


def test_the_recheck_sample_adds_the_longest_prompt_and_two_capped_pairs(job):
    """N round-robin records, PLUS the job's longest prompt pair, PLUS up to two pairs whose original line
    finished "length" (here every line hits the cap of 4)."""
    assert generate.main(job("first")) == 0
    assert generate.main(job("again", "base", "--recheck-of", str(job.root / "first"), "--sample", "1")) == 0
    rc = receipt(job.root / "again")
    lines = sorted((json.loads(x) for x in jsonl_lines(job.root / "first" / "outputs.jsonl")),
                   key=lambda r: (r["record_id"], r["condition"]))
    rr = rc["sample"]["records"]
    longest = min(lines, key=lambda r: (-r["prompt_tokens"], r["record_id"], r["condition"]))
    expected = {(r["record_id"], r["condition"]) for r in lines if r["record_id"] in rr}
    expected.add((longest["record_id"], longest["condition"]))
    capped = [(r["record_id"], r["condition"]) for r in lines if (r["record_id"], r["condition"]) not in expected]
    expected |= set(capped[:2])                                         # every line hit the cap of 4
    assert rc["sample"]["rule"] == "recheck" and len(rr) == 1 and len(rc["sample"]["length"]) == 2
    assert rc["sample"]["length"] == [list(k) for k in capped[:2]]
    assert rc["sample"]["longest"] == [longest["record_id"], longest["condition"]]
    assert {(k[0], k[3]) for k in rc["recheck"]["keys"]} == expected


def test_the_recheck_sample_holds_the_longest_prompt_even_when_it_stopped_at_eos():
    """The longest-prompt pair is added by its own rule, not because its line hit the cap."""
    from v2gpu.genprep import Pair
    from v2gpu.runs import Run

    records, conditions = ["R|a", "R|b", "R|c", "R|d"], ("zero_shot", "rag_only")
    pairs = [Pair(Run("deepseek", "base", c), r, "test", None, "p") for r in records for c in conditions]
    budget = {(r, f): ([0] * (900 if (r, f) == ("R|d", "F1") else 10), 4) for r in records for f in ("F0", "F1")}
    original = {(r, "deepseek", "base", c, "main"): b'{"finish_reason": "stop"}\n' for r in records for c in conditions}
    kept, info = generate.select_pairs(pairs, budget, 1, "recheck", original)
    assert info["records"] == ["R|a"] and info["longest"] == ["R|d", "rag_only"] and info["length"] == []
    assert {(p.record_id, p.run.condition) for p in kept} == {("R|a", "zero_shot"), ("R|a", "rag_only"),
                                                            ("R|d", "rag_only")}


def test_the_receipt_names_the_code_that_generated_the_outputs(job):
    """Coverage binds the leak receipts to the code that generated the outputs (the pod's checkout,
    the audit's own code_sha256 function), not to the laptop's checkout at coverage time."""
    from v2gpu.audit import code_sha256

    assert generate.main(job()) == 0
    assert receipt(job.root / "run")["code_sha256"] == code_sha256()


def test_a_recheck_of_an_unfinished_job_is_refused(job):
    """The original's receipt must be DONE."""
    assert generate.main(job("first")) == 0
    rc0 = json.loads((job.root / "first" / "receipt-generate.json").read_text())
    (job.root / "first" / "receipt-generate.json").write_text(json.dumps({**rc0, "status": "RUNNING"}))
    assert generate.main(job("again", "base", "--recheck-of", str(job.root / "first"), "--sample", "1")) == 2
    assert receipt(job.root / "again")["status"] == "REFUSED_RECHECK"


def test_a_recheck_of_a_job_with_other_settings_is_refused(job):
    assert generate.main(job("first")) == 0
    assert generate.main(job("again", "base", "--recheck-of", str(job.root / "first"), "--sample", "2", cap="3")) == 2
    assert receipt(job.root / "again")["status"] == "REFUSED_RECHECK"


def test_the_memory_probe_runs_the_longest_prompt_to_its_full_budget(job):
    assert generate.main(job("probe", "base", "--probe-longest")) == 0
    p = receipt(job.root / "probe")["probe"]
    assert p["new_tokens"] == p["max_new_tokens"] == 4 and p["finish_reason"] == "length"
    assert p["prompt_tokens"] == max(receipt(job.root / "probe")["preflight"]["prompt_tokens_max"], p["prompt_tokens"])
    assert not (job.root / "probe" / "outputs.jsonl").exists()


def test_without_a_cap_the_probe_fills_the_whole_window(job):
    """The worst case any real pair can reach: prompt + budget = max_model_len (the tiny model's window is 512)."""
    assert generate.main(job("fill", "base", "--probe-longest", cap=None)) == 0
    p = receipt(job.root / "fill")["probe"]
    assert p["prompt_tokens"] + p["new_tokens"] == 512 and p["max_new_tokens"] == 512 - p["prompt_tokens"]


def test_the_probe_budget_is_the_rest_of_the_window_not_the_6144_cap():
    assert generate.probe_budget(5000, 16384, None) == 11384 and generate.probe_budget(5000, 16384, 4) == 4
    assert generate.probe_budget(9000, 32768, None) == 23768


def canary_argv(job, workdir, *extra, weights="s42"):
    argv = job(workdir, weights, *extra)
    argv[argv.index("--variant") + 1] = "gen_canary"
    return argv


def test_the_canary_sample_adds_the_longest_prompts_and_records_the_ids(job):
    """N round-robin records plus the records of the 3 longest prompts; every pair's token ids kept."""
    assert generate.main(canary_argv(job, "a", "--sample", "1")) == 0
    rc = receipt(job.root / "a")
    assert rc["sample"]["rule"] == "canary" and len(rc["sample"]["longest"]) == 3
    rows = [json.loads(x) for x in jsonl_lines(job.root / "a" / "outputs.jsonl")]
    assert {r["record_id"] for r in rows} == {"a1", "x1", "x2"} and len(rows) == 9       # 3 records x 3 conditions
    ids = [json.loads(x) for x in jsonl_lines(job.root / "a" / "canary_ids.jsonl")]
    assert len(ids) == 9 and all(len(i["new_ids"]) == 4 for i in ids)


def test_the_canary_sample_holds_the_longest_full_aware_prompt():
    """decision 24/09: the HF pairs start at the model's longest F2 dev prompt, so the canary sample holds that record
    even when three other records have a longer F0/F1 prompt."""
    from v2gpu.genprep import Pair
    from v2gpu.runs import Run

    records = ["R|a", "R|b", "R|c", "R|d", "R|e", "R|f"]
    fmts = {"ft_only": "F0", "full_neutral": "F1", "full_aware": "F2"}
    pairs = [Pair(Run("deepseek", "s42", c), r, "dev", None, "p") for r in records for c in fmts]
    length = {("R|a", "F1"): 900, ("R|b", "F1"): 800, ("R|c", "F1"): 700, ("R|f", "F2"): 600}
    budget = {(r, f): ([0] * length.get((r, f), 10), 4) for r in records for f in fmts.values()}
    kept, info = generate.select_pairs(pairs, budget, 1, "canary")
    assert info["longest"] == ["R|a", "R|b", "R|c"] and info["longest_f2"] == "R|f"
    assert {p.record_id for p in kept} == set(info["records"]) | {"R|a", "R|b", "R|c", "R|f"}


def test_repeat_longest_regenerates_the_longest_output_in_one_process(job):
    assert generate.main(canary_argv(job, "a", "--sample", "1", "--repeat-longest", "3")) == 0
    rep = receipt(job.root / "a")["repeat"]
    assert rep["n"] == 3 and rep["identical"] is True and len(set(rep["sha256"])) == 1
    assert rep["sha256"][0] == rep["first_sha256"] and rep["key"][3] == "full_aware"


def test_extend_cap_adds_full_aware_pairs_until_one_hits_the_cap():
    """--extend-cap: while no full_aware pair of the run finished "length", add further records' full_aware pairs."""
    from v2gpu.genprep import Pair
    from v2gpu.runs import Run

    pairs = [Pair(Run("deepseek", "s42", c), r, "dev", None, "p") for r in ("R|a", "R|b", "R|c", "R|d")
             for c in ("ft_only", "full_aware")]

    class Log:
        def __init__(self):
            self.done = {}

        def __contains__(self, key):
            return key in self.done

    log, ran = Log(), []

    def run_pair(p):
        ran.append(p.record_id)
        reason = "length" if p.record_id == "R|c" else "stop"
        log.done[(p.record_id, "deepseek", "s42", p.run.condition, "gen_canary")] = json.dumps(
            {"finish_reason": reason}).encode()

    a = type("A", (), {"extend_cap": 10, "model": "deepseek", "weights": "s42", "variant": "gen_canary"})()
    for p in pairs[:2]:
        run_pair(p)
    added = generate._extend_for_cap(a, pairs, pairs[:2], log, run_pair)
    assert added == ["R|b", "R|c"] and ran == ["R|a", "R|a", "R|b", "R|c"]
    a.extend_cap = 2
    log2 = Log()
    assert generate._extend_for_cap(a, pairs, pairs[:2], log2, lambda p: None) == ["R|b"]


def rows7(cap=("E|g",)):
    return [{"record_id": r, "model": "deepseek", "seed": "s42", "condition": c, "variant": "gen_canary",
             "prompt_tokens": 100 + i, "new_tokens": 50 if r not in cap else 6144,
             "finish_reason": "length" if r in cap else "stop"}
            for i, r in enumerate(["A|a", "A|b", "B|c", "C|d", "D|e", "D|f", "E|g"]) for c in ("ft_only", "full_aware")]


def test_hf_pairs_are_the_longest_prompt_a_cap_hit_and_round_robin():
    """decision 24/09: 5 full_aware pairs per model; the longest prompt, a cap-hit pair, the rest round robin."""
    sel = generate.select_hf_pairs(rows7())
    keys = [k[0] for k in sel["keys"]]
    assert sel["cap_hit_found"] is True and keys[0] == "E|g" and len(keys) == 5    # E|g: longest AND cap hit
    assert keys[1:] == ["A|a", "B|c", "C|d", "D|e"] and all(k[3] == "full_aware" for k in sel["keys"])
    rows = rows7(cap=("B|c",))
    sel = generate.select_hf_pairs(rows)
    assert [k[0] for k in sel["keys"]][:2] == ["E|g", "B|c"] and sel["cap_hit_found"] is True


def test_without_a_cap_hit_the_hf_pairs_take_the_longest_output_and_say_so():
    rows = rows7(cap=())
    for r in rows:
        if r["record_id"] == "C|d":
            r["new_tokens"] = 900
    sel = generate.select_hf_pairs(rows)
    assert sel["cap_hit_found"] is False and [k[0] for k in sel["keys"]][:2] == ["E|g", "C|d"]


def test_teacher_forcing_scores_the_canary_run_under_the_hf_model(job, monkeypatch):
    """The HF side scores the vLLM run's own token ids (here both are the tiny model, so they agree)."""
    monkeypatch.setattr(generate, "HF_PAIRS", 3)                      # the fixture has 3 records
    assert generate.main(canary_argv(job, "a", "--sample", "3")) == 0
    assert generate.main(canary_argv(job, "hf", "--teacher-force-of", str(job.root / "a"))) == 0
    hf = receipt(job.root / "hf")["hf_agreement"]
    assert hf["rate"] == 1.0 and hf["positions"] == 12 and len(hf["pairs"]) == 3
    assert not (job.root / "hf" / "outputs.jsonl").exists()
    monkeypatch.setattr(generate, "HF_PAIRS", 5)
    assert generate.main(canary_argv(job, "hf5", "--teacher-force-of", str(job.root / "a"))) == 2
    assert receipt(job.root / "hf5")["status"] == "REFUSED_TEACHER_FORCE"


def test_python_m_v2gpu_generate_has_a_help():
    out = subprocess.run([sys.executable, "-m", "v2gpu.generate", "--help"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0 and "--leakcheck-receipt-sha256" in out.stdout


def set_status(workdir, status):
    rc = receipt(workdir)
    (workdir / "receipt-generate.json").write_text(json.dumps({**rc, "status": status}))


def test_a_finished_run_without_outputs_is_not_another_jobs_done(job, capsys):
    """A probe (DONE, no outputs.jsonl) left in `w` made the next, different job print
    ALREADY_DONE and exit 0 without running. That job is refused now; and ALREADY_DONE needs the receipt's identity."""
    assert generate.main(job("w", "base", "--probe-longest")) == 0
    assert generate.main(job("w")) == 2
    assert "another job" in capsys.readouterr().err and not (job.root / "w" / "outputs.jsonl").exists()
    assert receipt(job.root / "w")["identity"]["probe_longest"] is True
    assert generate.main(job("x")) == 0
    rc = receipt(job.root / "x")
    (job.root / "x" / "receipt-generate.json").write_text(json.dumps({**rc, "identity": {**rc["identity"], "cap": 9}}))
    assert generate.main(job("x")) == 0 and "ALREADY_DONE" not in capsys.readouterr().out
    assert receipt(job.root / "x")["identity"] == json.loads((job.root / "x" / "job.json").read_text())


def test_the_engine_route_is_part_of_the_job_identity(job, monkeypatch, capsys):
    """Batch invariance, the vLLM environment and the tuned shrink files decide the
    numbers, so a resume across a route flip is refused (since decision 05/10 the route is (a): the flip goes back to
    (b)); no absolute path, so a cross-pod recheck matches."""
    from v2gpu import engines

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    monkeypatch.setattr(generate, "environment", fake_environment(dict(engines.GEN_PINS)))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    assert generate.main(argv) == 2                                   # no card: refused after job.json is written
    ident = json.loads((job.root / "j" / "job.json").read_text())
    s = ident["engine_settings"]
    assert s["batch_invariant"] is True and s["tuned_config_files"] == engines.tuned_config_files()
    assert s["env"] == {k: v for k, v in engines.VLLM_ENV.items() if k != "VLLM_TUNED_CONFIG_FOLDER"}
    assert str(engines.LORA_OPS) not in json.dumps(ident)
    (job.root / "j" / "outputs.jsonl").write_text("{}\n")             # a line of the earlier attempt
    monkeypatch.setattr(engines, "BATCH_INVARIANT", False)            # the route flipped back to (b)
    assert generate.main(argv) == 2 and "another job's outputs" in capsys.readouterr().err


def test_an_interrupted_attempt_records_its_engine_and_environment(job, monkeypatch):
    """A SIGTERM leaves no finish(); the receipt is written right after environment() and after
    the engine starts, so the attempt still names its engine."""
    class Killed(BaseException):
        pass

    class Engine:
        def describe(self):
            return {"name": "fake", "route": "b"}

        def generate(self, *a, **k):
            raise Killed()

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: Engine())
    with pytest.raises(Killed):
        generate.main(job())
    rc = receipt(job.root / "run")
    assert rc["status"] == "RUNNING" and rc["engine"] == {"name": "fake", "route": "b"}
    assert "packages" in rc["environment"]


def test_a_canary_killed_while_writing_its_ids_resumes(job):
    """A torn last line of canary_ids.jsonl is cut, as OutputLog cuts outputs.jsonl, and the ids of
    a pair are appended after its output line, so every line ends with exactly one ids entry."""
    assert generate.main(canary_argv(job, "a", "--sample", "1")) == 0
    w = job.root / "a"
    lines = (w / "outputs.jsonl").read_bytes().split(b"\n")
    (w / "outputs.jsonl").write_bytes(b"\n".join(lines[:4]) + b"\n")
    ids = (w / "canary_ids.jsonl").read_bytes().split(b"\n")
    (w / "canary_ids.jsonl").write_bytes(b"\n".join(ids[:4]) + b"\n" + ids[4][:10])   # killed mid-write
    set_status(w, "RUNNING")
    assert generate.main(canary_argv(job, "a", "--sample", "1")) == 0
    keys = [tuple(json.loads(x)["key"]) for x in jsonl_lines(w / "canary_ids.jsonl")]
    assert len(keys) == len(set(keys)) == len(jsonl_lines(w / "outputs.jsonl")) == 9
    assert receipt(w)["result"]["ids_truncated_tail"] == 1


def test_job_json_and_the_receipt_are_replaced_atomically(job, monkeypatch):
    """A temporary file and os.replace, so a kill never leaves half a job.json or receipt."""
    real, replaced = os.replace, []
    monkeypatch.setattr(generate.os, "replace", lambda src, dst: (replaced.append(Path(dst).name), real(src, dst)))
    assert generate.main(job()) == 0
    assert "job.json" in replaced and replaced.count("receipt-generate.json") >= 2
    assert not list((job.root / "run").glob("*.tmp"))


def test_teacher_forcing_a_run_without_outputs_is_refused(job):
    """A DONE probe has no outputs.jsonl: refused, not a crash."""
    assert generate.main(canary_argv(job, "p", "--probe-longest")) == 0
    assert generate.main(canary_argv(job, "hf", "--teacher-force-of", str(job.root / "p"))) == 2
    assert receipt(job.root / "hf")["status"] == "REFUSED_TEACHER_FORCE"


def test_the_receipt_says_whether_the_repeated_pair_came_from_this_process(job):
    """After a resume the first of the repeated outputs may come from an earlier process."""
    argv = canary_argv(job, "a", "--sample", "1", "--repeat-longest", "2")
    assert generate.main(argv) == 0 and receipt(job.root / "a")["repeat_in_process"] is True
    set_status(job.root / "a", "RUNNING")
    assert generate.main(argv) == 0 and receipt(job.root / "a")["repeat_in_process"] is False


def test_a_resume_with_nothing_left_loads_no_engine(job, monkeypatch):
    """Every line is there (killed before finish): no engine; the receipt keeps the engine of the
    attempt that generated the lines."""
    assert generate.main(job()) == 0
    engine = receipt(job.root / "run")["engine"]
    set_status(job.root / "run", "RUNNING")
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    assert generate.main(job()) == 0
    rc = receipt(job.root / "run")
    assert rc["status"] == "DONE" and rc["engine"] == engine and rc["engine_from"] == "receipt-generate.attempt-1.json"




def test_a_stale_tmp_file_of_a_killed_first_write_is_not_content(job):
    """A kill during the very first job.json write leaves only job.json.tmp; the workdir is still empty."""
    w = job.root / "run"
    w.mkdir()
    (w / "job.json.tmp").write_text('{"half')
    assert generate.main(job()) == 0


def test_hf_runs_do_not_need_the_tuned_config_folder(job, monkeypatch):
    """config/lora_ops is read for the vLLM engine only."""
    from v2gpu import engines

    monkeypatch.setattr(engines, "LORA_OPS", job.root / "no-lora-ops")
    assert generate.main(job()) == 0
    assert "tuned_config_files" not in json.loads((job.root / "run" / "job.json").read_text())["engine_settings"]


def test_a_vllm_job_without_the_tuned_config_folder_is_refused_with_a_receipt(job, monkeypatch):
    """A missing folder is REFUSED_ENVIRONMENT with a receipt, not a FileNotFoundError before the workdir."""
    from v2gpu import engines

    monkeypatch.setattr(engines, "LORA_OPS", job.root / "no-lora-ops")
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    monkeypatch.setattr(generate, "environment", fake_environment(dict(engines.GEN_PINS), A4000))
    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    assert generate.main(argv) == 2
    rc = receipt(job.root / "j")
    assert rc["status"] == "REFUSED_ENVIRONMENT" and any("NVIDIA_RTX_A4000_SHRINK.json" in p for p in rc["problems"])


@pytest.mark.parametrize("case, code", [("done", 2), ("probe", 2), ("teacher_force", 2), ("ids", 2), ("nothing", 0)])
def test_each_kind_of_finished_run_keeps_its_workdir(job, capsys, case, code):
    """Each evidence of a finished run of another job (a DONE receipt, probe results, teacher-force results,
    canary_ids.jsonl) refuses the new job on its own; a refused attempt with nothing produced does not."""
    assert generate.main(job("w", "base", "--probe-longest")) == 0          # job.json names the probe job
    w = job.root / "w"
    rc = {k: v for k, v in receipt(w).items() if k != "probe"}
    body = {"done": {**rc, "status": "DONE"}, "probe": {**rc, "status": "PROBE_SHORT", "probe": {"new_tokens": 1}},
            "teacher_force": {**rc, "status": "CRASHED", "hf_agreement": {}}, "ids": {**rc, "status": "CRASHED"},
            "nothing": {**rc, "status": "CRASHED"}}[case]
    (w / "receipt-generate.json").write_text(json.dumps(body))
    if case == "ids":
        (w / "canary_ids.jsonl").write_text("")
    assert generate.main(job("w")) == code
    assert (case == "nothing") != ("another job" in capsys.readouterr().err)


def test_a_pair_whose_line_was_not_written_has_no_ids(job, monkeypatch):
    """A pair's ids are appended after its output line, so a failed line leaves no ids behind."""
    from v2gpu.outlog import OutputLog

    real, calls = OutputLog.append, []

    def append(self, row):
        calls.append(row)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(self, row)

    monkeypatch.setattr(OutputLog, "append", append)
    with pytest.raises(OSError):
        generate.main(canary_argv(job, "a", "--sample", "1"))
    w = job.root / "a"
    assert len(jsonl_lines(w / "canary_ids.jsonl")) == len(jsonl_lines(w / "outputs.jsonl")) == 2


def test_the_environment_is_saved_before_the_engine_loads(job, monkeypatch):
    """Killed while the engine loads, the attempt's receipt already holds its environment."""
    class Killed(BaseException):
        pass

    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(Killed()))
    with pytest.raises(Killed):
        generate.main(job())
    rc = receipt(job.root / "run")
    assert rc["status"] == "RUNNING" and "packages" in rc["environment"] and "engine" not in rc


def test_the_job_identity_names_the_generation_code_and_the_model_path(job):
    """gen_code_sha256 covers every v2gpu module generate loads (a probe in a fresh process, as for the audit);
    A run from a local snapshot says so (model_path), so it never shares an identity with a hub run."""
    probe = "import sys, v2gpu.generate; print(' '.join(sorted(m for m in sys.modules if m.startswith('v2gpu.'))))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                         text=True, check=True)
    loaded = {"v2gpu/" + m.split(".", 1)[1].replace(".", "/") + ".py" for m in out.stdout.split()}
    assert "v2gpu/generate.py" in loaded and "v2gpu/engines.py" in loaded and loaded <= set(generate.GEN_CODE)
    assert generate.main(job()) == 0
    ident = json.loads((job.root / "run" / "job.json").read_text())
    assert ident["gen_code_sha256"] == generate.gen_code_sha256() and set(ident["gen_code_sha256"]) == set(
        generate.GEN_CODE)
    assert ident["model_path"] == str(job(cpu=True)[job(cpu=True).index("--model-path") + 1])


def test_the_job_identity_holds_the_compose_block_of_its_prompt_directory(job):
    """Every variant's META compose goes into the job identity; coverage requires one per variant."""
    from types import SimpleNamespace

    assert generate.main(job()) == 0
    assert json.loads((job.root / "run" / "job.json").read_text())["compose"] == V0      # the fixture's META
    ns = generate.build_parser().parse_args(job())
    prep = SimpleNamespace(pairs=[], files={}, prompts_meta={"sums_sha256": "s", "meta": {"compose": {"k": 8}}})
    assert generate._identity(ns, "hf", prep, "i", 16384)["compose"] == {"k": 8}


def test_a_job_names_the_variants_decision_file_in_its_identity(job, capsys, monkeypatch):
    """The decision's sha256 is part of what a job is; job mode requires it."""
    from v2gpu import engines

    argv = job("j", "base", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    i = argv.index("--decision-sha256")
    assert generate.main(argv[:i] + argv[i + 2:]) == 2 and "--decision-sha256" in capsys.readouterr().err
    monkeypatch.setattr(generate, "environment", fake_environment(dict(engines.GEN_PINS)))
    assert generate.main(argv) == 2
    assert json.loads((job.root / "j" / "job.json").read_text())["decision_sha256"] == "d" * 64
    assert "decision_sha256" not in generate.NOT_IDENTITY


def test_the_tuned_shrink_file_is_required_of_the_vllm_engine_only(job, monkeypatch, tmp_path):
    """An HF canary run (the teacher-force reference) on a card whose shrink file is
    missing is not refused for it; a vLLM run on that card is."""
    from v2gpu import engines

    folder = tmp_path / "lora_ops"
    folder.mkdir()
    monkeypatch.setattr(engines, "LORA_OPS", folder)
    monkeypatch.setattr(generate, "open_engine", lambda *a, **k: (_ for _ in ()).throw(AssertionError("engine")))
    ada = {"name": "NVIDIA RTX 2000 Ada Generation", "capability": "8.9", "total_memory_gb": 16.7}
    monkeypatch.setattr(generate, "environment", fake_environment(dict(engines.GEN_PINS), ada))
    argv = job("c", "base", "--probe-longest", mode="canary", engine="hf", cap=None, cpu=True)
    argv[argv.index("--variant") + 1] = "gen_canary"
    with pytest.raises(AssertionError, match="engine"):                # past every environment check
        generate.main(argv + ["--gpu-name", ada["name"]])
    assert receipt(job.root / "c")["status"] == "CRASHED"


def test_the_adapter_config_is_part_of_the_job(job, capsys, monkeypatch):
    """adapter_config.json (rank, alpha, targets) decides the numbers with the tensors: its
    sha256 is in the job command and the identity, checked against the file, and required in job mode."""
    from v2gpu import engines

    assert generate.main(job("t", "s42")) == 0
    ident = json.loads((job.root / "t" / "job.json").read_text())
    i = job("t", "s42").index("--adapter-config-sha256")
    assert ident["adapter_config_sha256"] == job("t", "s42")[i + 1] and len(ident["adapter_config_sha256"]) == 64
    bad = job("b", "s42")
    bad[i + 1] = "0" * 64
    assert generate.main(bad) == 2
    assert any("adapter_config.json" in p for p in receipt(job.root / "b")["problems"])
    argv = job("j", "s42", mode="job", engine=None, cap=None, cpu=False) + ["--gpu-name", "NVIDIA RTX A4000"]
    k = argv.index("--adapter-config-sha256")
    monkeypatch.setattr(generate, "environment", fake_environment(dict(engines.GEN_PINS)))
    assert generate.main(argv[:k] + argv[k + 2:]) == 2 and "--adapter-config-sha256" in capsys.readouterr().err


def test_the_receipt_records_the_locks_the_driver_and_the_numerics_environment(monkeypatch, tmp_path):
    """What else decides the numbers is written down: the requirements locks present in the
    checkout (sha256), the nvidia-smi driver version, and env vars beyond VLLM_ENV (CUDA_*, CUBLAS_*, PYTORCH_*...)."""
    from v2gpu import receipt as rc_mod

    monkeypatch.setattr(rc_mod, "REPO", tmp_path)
    (tmp_path / "requirements-gen.lock").write_text("vllm==0.27.1\n")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("UNRELATED_SETTING", "x")
    env = rc_mod.environment()
    assert env["locks"] == {"requirements-gen.lock": hashlib.sha256(b"vllm==0.27.1\n").hexdigest()}
    assert {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0"}.items() <= env["numerics_env"].items()
    assert "UNRELATED_SETTING" not in env["numerics_env"] and "driver" in env


def test_a_canary_adapter_run_names_its_adapter_config(job, capsys):
    """The canary binds the adapter config as the job does (gencanary SHARED)."""
    argv = job("c", "s42", "--probe-longest", mode="canary", engine="hf", cap=None, cpu=True)
    argv[argv.index("--variant") + 1] = "gen_canary"
    k = argv.index("--adapter-config-sha256")
    assert generate.main(argv[:k] + argv[k + 2:] + ["--gpu-name", "NVIDIA RTX A4000"]) == 2
    assert "--adapter-config-sha256" in capsys.readouterr().err
    assert "adapter_config_sha256" in generate.SHARED


def test_the_receipt_records_a_first_run_venv(monkeypatch, tmp_path):
    """A venv built without a lock (FIRST_RUN=1) carries FIRST_RUN_LOCK.json, the frozen lock it made;
    the receipt records it, so coverage can show the first-run exception."""
    from v2gpu import receipt as rc_mod

    monkeypatch.setattr(rc_mod, "REPO", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(rc_mod.sys, "prefix", str(tmp_path / "venv"))
    (tmp_path / "venv").mkdir()
    assert rc_mod.environment()["first_run_lock"] is None
    (tmp_path / "venv" / "FIRST_RUN_LOCK.json").write_text(json.dumps({"requirements-gen.lock": "a" * 64}))
    assert rc_mod.environment()["first_run_lock"] == {"requirements-gen.lock": "a" * 64}
