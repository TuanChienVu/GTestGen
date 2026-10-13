# tests_gpu/test_genqueue.py
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests_gpu.lockfree import dry_run
from tests_gpu.test_audit import V0, main as audit, setup
from tests_gpu.tiny import make_tiny_llama
from v2gpu.engines import GEN_PINS, PIN_SETS
from v2gpu.genqueue import main
from v2gpu.modeling import LoadCfg, load_for_training
from v2gpu.receipt import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
POD = "/workspace"
VERDICT = {"pass": True, "trained": True, "model": "deepseek", "a_outputs_sha256": "a" * 64,
           "criteria": {"hf_agreement": {"pass": True}, "probe": {"pass": True, "probed_prompt": 16384}},
           "decision": {"gpu_name": "NVIDIA RTX A4000", "pin_set": "main", "gpu_memory_utilization": 0.95,
                        "enforce_eager": True}}       # decision 05/10: the only step a real canary can decide now
REDECODE = {"gate": True, "pass": True, "lines": 24, "missing": [], "differ": [], "of_outputs_sha256": "a" * 64,
            "versions": {"transformers": GEN_PINS["transformers"], "tokenizers": GEN_PINS["tokenizers"]}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A selected adapter directory and its train receipt, as train_job leaves them."""
    from v2gpu.train_job import train_cfg_sha256, train_code_sha256

    d = tmp_path_factory.mktemp("run")
    model, _ = load_for_training(LoadCfg(make_tiny_llama(d / "tiny", 64), None, quant="none", attn="sdpa",
                                         dtype="float32", gradient_checkpointing=False), seed=42)
    model.save_pretrained(d / "deepseek-s42-all" / "epoch-3")
    digest = tensor_digest(d / "deepseek-s42-all" / "epoch-3" / "adapter_model.safetensors")
    # Decision 27/09: every real adapter is trained with --upcast-fp32; this fixture is
    # the "trained" adapter every genqueue test builds a queue from, so it must be true here too.
    train_config = {"lr": 2e-4, "grad_accum": 16, "max_seq": 8192, "quant": "nf4", "attn": "sdpa",
                    "dtype": "bfloat16", "gradient_checkpointing": True, "upcast_fp32": True}
    rc = {"status": "DONE", "mode": "job", "seed": 42, "subset": "all", "model": {"key": "deepseek"},
          "data": {"prompts": {"meta": {"compose": V0}}},                       # the compose it was trained on
          "selected_adapter": {"path": "epoch-3/adapter_model.safetensors", "tensor_sha256": digest},
          "train_config": train_config, "train_code_sha256": train_code_sha256(),
          "train_cfg_sha256": train_cfg_sha256(train_config)}
    (d / "deepseek-s42-all" / "receipt-train.json").write_text(json.dumps(rc))
    return d / "deepseek-s42-all", digest


def argv(tmp_path, trained, *extra, weights=("base", "s42"), receipts=None, **setup_kw):
    assert audit(setup(tmp_path, **setup_kw)) == 0
    run, digest = trained
    # the model's passing canary verdict: trained seed-42 adapter (this digest), its probes read these prompts
    probed = [sha(tmp_path / "prompts" / "PROMPTS_SHA256SUMS")]
    (tmp_path / "verdict.json").write_text(json.dumps({**VERDICT, "adapter_tensor_sha256": digest,
                                                       "probed_sums": probed, **verdict_binding(VERDICT["decision"]),
                                                       "adapter_config_sha256": sha(run / "epoch-3" /
                                                                                    "adapter_config.json")}))
    write_decision(tmp_path)
    adapters = [] if weights == ("base",) else ["--train-receipts", *(receipts or [str(run / "receipt-train.json")]),
                                                "--adapters", str(run / "epoch-3")]
    return ["--variant", "main", "--model", "deepseek", "--weights", *weights, "--partitions", "test", "aux_test",
            "--prompts", str(tmp_path / "prompts" / "test.deepseek.jsonl"),
            "--prompts-sums", str(tmp_path / "prompts" / "PROMPTS_SHA256SUMS"),
            "--leakcheck-receipt", str(tmp_path / "leakcheck-receipt.json"), *adapters,
            "--pod-root", POD, "--gpu-name", "NVIDIA RTX A4000", "--deadline-utc", "2026-10-08T12:00:00+00:00",
            "--max-hours", "9.5", "--recheck-max-hours", "1.5", "--recheck-sample", "5",
            "--canary-verdict", str(tmp_path / "verdict.json"), "--decision", str(tmp_path / "decision.json"),
            "--canary-train-receipt", str(run / "receipt-train.json"), "--out", str(tmp_path / "QUEUE.json"), *extra]


def checkout_engine(decision):
    """The declared engine of this checkout at a decision's memory step (what the canary ran)."""
    from v2gpu import generate
    from v2gpu.engines import declared_engine

    try:
        return declared_engine(generate.job_engine_settings("deepseek", None, 16384,
                                                            decision.get("gpu_memory_utilization"),
                                                            decision.get("enforce_eager"), "vllm"))
    except ValueError:                                   # an off-ladder step: the verdict check refuses it anyway
        return {}


FINGERPRINT = {"name": "vllm", "version": "0.27.1"}          # a measured engine names itself


def verdict_binding(decision):
    """What a canary verdict binds (gencanary.BINDING), as this checkout runs it at the decision's memory step."""
    from v2gpu.generate import gen_code_sha256

    return {"gen_code_sha256": gen_code_sha256(), "engine_settings": checkout_engine(decision),
            "engine_fingerprint": FINGERPRINT}


def write_decision(tmp_path, decision=None, variant="main", models=("codellama", "deepseek", "mistral"), **change):
    """The variant's decision file, as `gencanary decide` writes it from every planned model's passing verdict: bound
    to the generation code and the engine the canaries ran."""
    from v2gpu.generate import gen_code_sha256

    decision = decision or VERDICT["decision"]
    (tmp_path / "decision.json").write_text(json.dumps({
        "schema": "gtestgen.v2.gen_decision.v2", "pass": True, "variant": variant, "decision": decision,
        "verdicts": {m: {"verdict": f"{m}.json", "sha256": "e" * 64, "trained": False} for m in models},
        "gen_code_sha256": gen_code_sha256(), "engine_settings": checkout_engine(decision),
        "engine_fingerprint": FINGERPRINT, **change}))


def edit_verdict(tmp_path, **change):
    """Change the verdict; a changed decision is the variant's decision too (decide ran on these verdicts)."""
    path = tmp_path / "verdict.json"
    old = json.loads(path.read_text())
    bound = verdict_binding(change["decision"]) if isinstance(change.get("decision"), dict) else {}
    path.write_text(json.dumps({**old, **bound, **change}))
    if isinstance(change.get("decision"), dict):
        write_decision(tmp_path, change["decision"])


def test_the_generation_venv_can_live_off_the_pod_volume(tmp_path, trained):
    """Importing torch from a MooseFS venv took 58.9 s on a smoke pod (28/09), so the venvs live on the
    container's own disk (/opt/v2); runs and queue stay under --pod-root. --venv-root moves the python and (since
    05/10: the bootstrap's HF_HOME is $V2GPU_LOCAL/hf) the HF cache; without it the queue is unchanged."""
    assert main([*argv(tmp_path, trained), "--venv-root", "/opt/v2"]) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    gens = [j for j in q["jobs"] if j["name"].startswith(("gen-", "recheck-"))]
    assert gens and all(j["cmd"][:4] == ["env", "HF_HUB_OFFLINE=1", "HF_HOME=/opt/v2/hf", "/opt/v2/venv-gen/bin/python"]
                        for j in gens)
    assert q["queue_dir"] == f"{POD}/queue" and all(j["workdir"].startswith(f"{POD}/runs/") for j in q["jobs"])


def test_every_job_carries_the_receipt_sha_and_adapter_digest_and_is_rechecked(tmp_path, trained):
    assert main(argv(tmp_path, trained)) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    assert q["lock_path"] == "/root/v2gpu-queue.lock" and q["deadline_utc"] == "2026-10-08T12:00:00+00:00"
    assert q["self_stop_cmd"] == ["runpodctl", "stop", "pod", "$RUNPOD_POD_ID"] and q["queue_dir"] == f"{POD}/queue"
    names = [j["name"] for j in q["jobs"]]
    assert names == ["gen-main-deepseek-base", "recheck-main-deepseek-base", "gen-main-deepseek-s42",
                     "recheck-main-deepseek-s42", "copy-window-main-deepseek"]
    gen, recheck = q["jobs"][2], q["jobs"][3]
    cmd = gen["cmd"]
    assert cmd[:6] == ["env", "HF_HUB_OFFLINE=1", f"HF_HOME={POD}/hf", f"{POD}/venv-gen/bin/python", "-m",
                       "v2gpu.generate"]
    assert gen["max_hours"] == 9.5
    assert cmd[cmd.index("--leakcheck-receipt-sha256") + 1] == sha(tmp_path / "leakcheck-receipt.json")
    assert cmd[cmd.index("--adapter-sha256") + 1] == trained[1]
    prompts = f"{POD}/prompts/{sha(tmp_path / 'prompts' / 'PROMPTS_SHA256SUMS')[:16]}"
    assert cmd[cmd.index("--adapter") + 1] == f"{POD}/adapters/deepseek-s42-{trained[1][:12]}"
    assert cmd[cmd.index("--prompts") + 1] == f"{prompts}/test.deepseek.jsonl"
    assert cmd[cmd.index("--prompts-sums") + 1] == f"{prompts}/PROMPTS_SHA256SUMS"
    assert cmd[cmd.index("--workdir") + 1] == gen["workdir"] == f"{POD}/runs/gen-main-deepseek-s42"
    assert "--mode" not in cmd and "--cap" not in cmd                      # job mode, locked settings
    assert recheck["cmd"][recheck["cmd"].index("--recheck-of") + 1] == gen["workdir"]
    assert recheck["cmd"][recheck["cmd"].index("--sample") + 1] == "5" and recheck["max_hours"] == 1.5
    assert "--adapter" not in q["jobs"][0]["cmd"]
    assert q["ship"]["adapters"] == {f"{POD}/adapters/deepseek-s42-{trained[1][:12]}": str(trained[0] / "epoch-3")}
    assert all(t.startswith(prompts + "/") for t in q["ship"]["prompts"])
    receipt = f"{POD}/receipts/{sha(tmp_path / 'leakcheck-receipt.json')[:16]}/leakcheck-receipt.json"
    assert cmd[cmd.index("--leakcheck-receipt") + 1] == receipt
    assert q["ship"]["receipts"] == {receipt: str(tmp_path / "leakcheck-receipt.json")}
    shipped = {Path(k).name for k in q["ship"]["prompts"]}             # what the pod's preflight reads, nothing more
    assert shipped == {"test.deepseek.jsonl", "PROMPTS_SHA256SUMS", "PROMPTS_META.json", "excluded.jsonl",
                       "pools.json"}


def test_the_recheck_budget_is_required(tmp_path, trained, capsys):
    """A recheck TIMEOUT blocks every later job of the pod; its max_hours comes from the canary speed."""
    a = argv(tmp_path, trained)
    i = a.index("--recheck-max-hours")
    with pytest.raises(SystemExit):
        main(a[:i] + a[i + 2:])
    assert "--recheck-max-hours" in capsys.readouterr().err


def test_prompt_files_must_sit_beside_their_sums(tmp_path, trained, capsys):
    """The pod checks files by their path relative to PROMPTS_SHA256SUMS; genqueue ships them flat."""
    a = argv(tmp_path, trained, weights=("base",))
    sub = tmp_path / "prompts" / "sub"
    sub.mkdir()
    moved = sub / "test.deepseek.jsonl"
    moved.write_bytes((tmp_path / "prompts" / "test.deepseek.jsonl").read_bytes())
    a[a.index("--prompts") + 1] = str(moved)
    assert main(a) == 2 and "beside" in capsys.readouterr().err


LEAKED = {"KIND": "gen", "GEN_SET": "fallback-tf"}      # what `KIND=gen GEN_SET=... bash ops/pod_bootstrap.sh` exports
# Bootstrap dry runs: tests_gpu.lockfree.dry_run, on a lock-free copy of this checkout's bootstrap, without the
# selectors of a bootstrap that runs this suite


def real_gen_workdir(root, monkeypatch, weights="base", pod="pod-1", change=None):
    """A gen workdir finished on another pod, whose job.json generate itself wrote (job mode, a fake environment
    without a card stops it right after job.json) for the same inputs as this fixture's queue."""
    from v2gpu import generate

    w, r = root / "results" / f"gen-main-deepseek-{weights}", root / "leakcheck-receipt.json"
    monkeypatch.setattr(generate, "environment", lambda: {"python": "3.12", "platform": "t",
                                                          "packages": dict(GEN_PINS), "cuda_available": False})
    monkeypatch.setenv("RUNPOD_POD_ID", pod)
    args = ["--model", "deepseek", "--weights", weights, "--variant", "main", "--partitions", "test", "aux_test",
            "--prompts", str(root / "prompts" / "test.deepseek.jsonl"),
            "--prompts-sums", str(root / "prompts" / "PROMPTS_SHA256SUMS"), "--leakcheck-receipt", str(r),
            "--leakcheck-receipt-sha256", sha(r), "--gpu-name", "NVIDIA RTX A4000", "--workdir", str(w),
            "--decision-sha256", sha(root / "decision.json")]
    assert generate.main(args) == 2 and json.loads((w / "receipt-generate.json").read_text())["problems"] == [
        "no CUDA device"]
    monkeypatch.delenv("RUNPOD_POD_ID")
    ident = json.loads((w / "job.json").read_text())
    if change:
        change(ident)
        (w / "job.json").write_text(json.dumps(ident))
    (w / "outputs.jsonl").write_text('{"x": 1}\n')
    (w / "receipt-generate.json").write_text(json.dumps({"status": "DONE", "pod_id": pod, "identity": ident,
                                                         "outputs_sha256": sha(w / "outputs.jsonl")}))
    return w


def finished_gen_workdir(root, name="gen-main-deepseek-base", pod="pod-1", status="DONE"):
    w = root / "results" / name
    w.mkdir(parents=True)
    (w / "outputs.jsonl").write_text('{"x": 1}\n')
    (w / "job.json").write_text(json.dumps({"model": "deepseek", "weights": "base", "variant": "main"}))
    (w / "receipt-generate.json").write_text(json.dumps({
        "status": status, "pod_id": pod, "outputs_sha256": sha(w / "outputs.jsonl"),
        "identity": {"model": "deepseek", "weights": "base", "variant": "main", "recheck_of": None}}))
    return w


def test_recheck_only_queues_rechecks_of_another_pods_workdirs(tmp_path, trained, monkeypatch):
    """decision 24/09: one recheck per (model, variant main) on another pod than its job; the originals are
    copied in first, and the queue holds recheck jobs only."""
    a = argv(tmp_path, trained, "--recheck-only", weights=("base",))
    orig = real_gen_workdir(tmp_path, monkeypatch)
    assert main(a + ["--originals", str(orig)]) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    assert [j["name"] for j in q["jobs"]] == ["recheck-main-deepseek-base-xpod", "copy-window-main-deepseek-xpod"]
    cmd = q["jobs"][0]["cmd"]
    assert cmd[cmd.index("--recheck-of") + 1] == f"{POD}/runs/gen-main-deepseek-base"
    assert q["ship"]["originals"] == {f"{POD}/runs/gen-main-deepseek-base": str(orig)}


def test_recheck_only_refuses_an_original_that_is_not_done(tmp_path, trained, capsys):
    orig = finished_gen_workdir(tmp_path, status="INCOMPLETE")
    assert main(argv(tmp_path, trained, "--recheck-only", "--originals", str(orig), weights=("base",))) == 2
    assert "DONE" in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


def test_the_generation_test_list_passes_in_a_simulated_generation_venv(tmp_path):
    """The KIND=gen bootstrap runs its test list in a venv that HAS the generation pins (vllm 0.27.1 ...);
    a test that assumed the laptop's packages failed there. Simulate that venv: package versions report GEN_PINS."""
    out = dry_run(tmp_path, ws=tmp_path / "ws", KIND="gen").stdout
    files = next(l for l in out.splitlines() if "pytest --collect-only -q " in l).split(
        "pytest --collect-only -q ", 1)[1].split()
    assert "tests_gpu/test_generate.py" in files and "tests_gpu/test_genqueue.py" in files
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "sitecustomize.py").write_text(
        "import importlib.metadata as md, json, os\n"
        "_real = md.version\n"
        "def version(name):\n"
        "    pins = json.loads(os.environ['V2GPU_SIM_PINS'])\n"
        "    return pins[name] if name in pins else _real(name)\n"
        "md.version = version\n")
    me = "tests_gpu/test_genqueue.py::test_the_generation_test_list_passes_in_a_simulated_generation_venv"
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp",
                          str(tmp_path / "bt"), *files, "--deselect", me], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(tmp_path / "site"),
                              "V2GPU_SIM_PINS": json.dumps(GEN_PINS), **LEAKED})    # as a pod leaks them
    assert run.returncode == 0, run.stdout[-3000:]


def test_the_queue_carries_the_canary_decision_and_the_laptops_prompt_ids(tmp_path, trained):
    """decision 24/09: the memory steps and the pin set come from the canary verdict into every command; every
    command carries the laptop's (training venv's) digest of its prompt ids; a final copy window keeps the pod up
    while the operator copies the last workdirs home (the queue's self-stop stays)."""
    from v2gpu.genprep import preflight, tokenize
    from v2gpu.models import MODELS, load_tokenizer

    decision = {**VERDICT["decision"], "pin_set": "fallback-tf", "gpu_memory_utilization": 0.95, "enforce_eager": True}
    a = argv(tmp_path, trained, "--copy-window-minutes", "20", weights=("base",))
    edit_verdict(tmp_path, decision=decision, criteria={**VERDICT["criteria"], "redecode": REDECODE})
    assert main(a) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    gen, recheck, window = q["jobs"]
    for cmd in (gen["cmd"], recheck["cmd"]):
        assert cmd[cmd.index("--pin-set") + 1] == "fallback-tf" and "--enforce-eager" in cmd
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.95"
    d, r = tmp_path / "prompts", tmp_path / "leakcheck-receipt.json"
    _, prep = preflight("deepseek", "base", "main", ["test", "aux_test"], [d / "test.deepseek.jsonl"],
                        d / "PROMPTS_SHA256SUMS", "all", r, sha(r))
    ids = tokenize(load_tokenizer(MODELS["deepseek"]), prep.pairs, MODELS["deepseek"].window)[2]
    assert gen["cmd"][gen["cmd"].index("--ids-sha256") + 1] == ids == recheck["cmd"][recheck["cmd"].index(
        "--ids-sha256") + 1]
    assert window["name"] == "copy-window-main-deepseek" and window["cmd"][:2] == ["bash", "-c"]
    assert "sleep 1200" in window["cmd"][2] and window["workdir"] in window["cmd"][2]
    assert window["max_hours"] > 1200 / 3600


@pytest.mark.parametrize("change, words", [
    ({"pass": False}, "passing"), ({"trained": False}, "--trained"), ({"model": "mistral"}, "mistral"),
    ({"decision": {**VERDICT["decision"], "gpu_name": "NVIDIA RTX A5000"}}, "gpu_name"),
    ({"decision": {**VERDICT["decision"], "pin_set": "custom"}}, "pin set"),
    ({"decision": {**VERDICT["decision"], "gpu_memory_utilization": 0.9}}, "gpu_memory_utilization"),
    ({"decision": None}, "decision"),
    ({"criteria": {"hf_agreement": {"pass": False}}}, "hf_agreement"),
    ({"adapter_tensor_sha256": "0" * 64}, "adapter"),                       # not the seed-42 adapter of this run
    ({"probed_sums": []}, "probe"),                                         # main: test/aux prompts never probed
])
def test_a_queue_needs_the_passing_trained_canary_verdict_of_this_model_and_card(tmp_path, trained, capsys, change,
                                                                                  words):
    a = argv(tmp_path, trained, weights=("base",))
    edit_verdict(tmp_path, **change)
    assert main(a) == 2
    assert words in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


@pytest.mark.parametrize("step", [(0.92, False), (0.95, False)])
def test_decision_0510_a_verdict_off_the_determinism_step_is_refused(tmp_path, trained, capsys, step):
    """decision 05/10 (defence in depth): with batch invariance on, a verdict at another pre-declared
    step of the ladder is refused here too (the vLLM engine would refuse that step on the pod anyway)."""
    a = argv(tmp_path, trained, weights=("base",))
    edit_verdict(tmp_path, decision={**VERDICT["decision"], "gpu_memory_utilization": step[0],
                                     "enforce_eager": step[1]})
    assert main(a) == 2
    assert "decision 05/10" in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


@pytest.mark.parametrize("pin_set", ["fallback-tf", "fallback-tf-bnb"])
@pytest.mark.parametrize("redecode, ok", [
    (REDECODE, True),
    (None, False),                                                                   # no re-decode record
    ({**REDECODE, "pass": False, "differ": [["x"]]}, False),                         # a line decodes differently
    ({**REDECODE, "of_outputs_sha256": "b" * 64}, False),                            # a re-decode of another run A
    ({**REDECODE, "versions": {"transformers": "5.15.0", "tokenizers": "0.22.2"}}, False),   # not the training venv
])
def test_a_tf_fallback_set_needs_a_passing_re_decode_in_the_training_venv(tmp_path, trained, capsys, pin_set,
                                                                           redecode, ok):
    """24/09: transformers 5.15.0 brings tokenizers 0.22.2; the prompt ids are checked per job
    (--ids-sha256), the output text by re-decoding canary run A with the training venv's tokenizer."""
    a = argv(tmp_path, trained, weights=("base",))
    criteria = {**VERDICT["criteria"], **({"redecode": redecode} if redecode else {})}
    edit_verdict(tmp_path, decision={**VERDICT["decision"], "pin_set": pin_set}, criteria=criteria)
    assert main(a) == (0 if ok else 2)
    assert ok or "re-decode" in capsys.readouterr().err


def test_the_main_set_needs_no_re_decode(tmp_path, trained):
    a = argv(tmp_path, trained, weights=("base",))
    assert "redecode" not in json.loads((tmp_path / "verdict.json").read_text())["criteria"]
    assert main(a) == 0


@pytest.mark.parametrize("margin, ok", [(-1, False), (0, True), (500, True)])
def test_a_prompt_longer_than_the_canary_probed_needs_a_new_probe(tmp_path, trained, capsys, margin, ok):
    """24/09: the probe filled the window from its longest prompt, so every shorter prompt fits in
    memory; a queue (e.g. a dev grid directory) whose longest prompt is longer needs a new probe first."""
    from v2gpu.gencanary import longest_prompt

    a = argv(tmp_path, trained, weights=("base",))
    longest = longest_prompt([tmp_path / "prompts" / "test.deepseek.jsonl"], "deepseek")[0]
    edit_verdict(tmp_path, criteria={**VERDICT["criteria"], "probe": {"pass": True, "probed_prompt": longest + margin}})
    assert main(a) == (0 if ok else 2)
    assert ok or (f"{longest} tokens" in (err := capsys.readouterr().err) and "--probe-longest" in err)


def test_a_queue_without_the_seed_42_adapter_names_the_canarys_train_receipt(tmp_path, trained, capsys):
    """The canary ran the model's seed-42 adapter; a queue of other weights proves which one."""
    a = argv(tmp_path, trained, weights=("base",))
    i = a.index("--canary-train-receipt")
    assert main(a[:i] + a[i + 2:]) == 2 and "--canary-train-receipt" in capsys.readouterr().err
    assert main(argv(tmp_path / "s42", trained)) == 0                    # s42 in the queue: its own receipt is used


def test_ship_targets_are_content_addressed_so_two_queues_never_collide(tmp_path, trained):
    """A dev canary or the dev-pilot cross-pod recheck and the main queue meet on one pod. Prompt
    directories ship to prompts/<sha256(PROMPTS_SHA256SUMS)[:16]>/ and adapters to <model>-<weights>-<digest[:12]>, so
    both sets of files coexist, and each job's preflight reads exactly its own directory."""
    from v2gpu.examples import prompt_files

    run, _ = trained
    pilot = tmp_path / "pilot"
    shutil.copytree(run, pilot)
    tensors = pilot / "epoch-3" / "adapter_model.safetensors"
    data = bytearray(tensors.read_bytes())
    data[-1] ^= 1
    tensors.write_bytes(bytes(data))
    rc = json.loads((pilot / "receipt-train.json").read_text())
    rc["selected_adapter"]["tensor_sha256"] = tensor_digest(tensors)
    (pilot / "receipt-train.json").write_text(json.dumps(rc))
    assert main(argv(tmp_path / "one", trained)) == 0
    other = argv(tmp_path / "two", (pilot, tensor_digest(tensors)), text=lambda r, fmt: f"<{fmt}> {r['scenario']}")
    assert main(other) == 0
    q1, q2 = (json.loads((tmp_path / d / "QUEUE.json").read_text()) for d in ("one", "two"))
    assert not set(q1["ship"]["prompts"]) & set(q2["ship"]["prompts"])
    assert not set(q1["ship"]["adapters"]) & set(q2["ship"]["adapters"])
    pod = tmp_path / "pod"                                          # both ship maps copied onto one pod, in turn
    for q in (q1, q2):
        for target, source in q["ship"]["prompts"].items():
            dest = pod / Path(target).relative_to(POD)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, dest)
    for q in (q1, q2):
        cmd = q["jobs"][0]["cmd"]
        files = [pod / Path(p).relative_to(POD) for p in cmd[cmd.index("--prompts") + 1:cmd.index("--prompts-sums")]]
        assert prompt_files(files, pod / Path(cmd[cmd.index("--prompts-sums") + 1]).relative_to(POD))[0] == []


@pytest.mark.parametrize("section", ["prompts", "receipts", "adapters"])
def test_one_ship_target_never_takes_two_files(tmp_path, section):
    """genqueue.ship_put refuses a second, different source for a target; the same file again (another
    spelling of its path) is no conflict."""
    from v2gpu.genqueue import ship_put

    (tmp_path / "d").mkdir()
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text("1")
    two.write_text("2")
    ship = {section: {}}
    assert ship_put(ship, section, "/workspace/x/f.json", one) == []
    assert ship_put(ship, section, "/workspace/x/f.json", tmp_path / "d" / ".." / "one.json") == []
    assert ship_put(ship, section, "/workspace/x/f.json", two) == [
        f"ship target /workspace/x/f.json would receive two sources: {one} and {two}"]
    assert ship[section] == {"/workspace/x/f.json": str(one)}


def test_a_ship_map_that_sends_two_sources_to_one_target_is_refused(tmp_path, trained, capsys):
    one, two = finished_gen_workdir(tmp_path / "a"), finished_gen_workdir(tmp_path / "b")
    assert main(argv(tmp_path, trained, "--recheck-only", "--originals", str(one), str(two), weights=("base",))) == 2
    assert "two sources" in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


def test_a_train_receipt_that_does_not_match_the_weights_is_refused(tmp_path, trained):
    run, _ = trained
    bad = tmp_path / "receipt-train.json"
    for change in ({"seed": 11}, {"status": "MAX_STEPS"}, {"model": {"key": "mistral"}}, {"mode": "canary"},
                   {"subset": "lc600"}):
        bad.write_text(json.dumps({**json.loads((run / "receipt-train.json").read_text()), **change}))
        a = argv(tmp_path / str(len(list(tmp_path.iterdir()))), trained, receipts=[str(bad)])
        assert main(a) == 2, change
        assert not Path(a[a.index("--out") + 1]).exists()


def test_an_adapter_trained_without_upcast_fp32_is_refused(tmp_path, trained, capsys):
    """Decision 27/09: fp32 upcast is mandatory for every real training job; genqueue
    refuses an adapter whose train receipt says otherwise, and one that omits train_config entirely."""
    run, _ = trained
    base = json.loads((run / "receipt-train.json").read_text())
    for change in ({"train_config": {**base["train_config"], "upcast_fp32": False}}, {"train_config": {}}):
        bad = tmp_path / f"receipt-{change == {'train_config': {}}}.json"
        bad.write_text(json.dumps({**base, **change}))
        a = argv(tmp_path / str(len(list(tmp_path.iterdir()))), trained, receipts=[str(bad)])
        assert main(a) == 2, change
    assert "upcast_fp32 is not true" in capsys.readouterr().err


def test_an_adapter_directory_that_is_not_the_selected_one_is_refused(tmp_path, trained):
    run, _ = trained
    other = tmp_path / "other"
    other.mkdir()
    for f in (run / "epoch-3").iterdir():
        other.joinpath(f.name).write_bytes(f.read_bytes())
    data = bytearray((other / "adapter_model.safetensors").read_bytes())
    data[-1] ^= 1
    (other / "adapter_model.safetensors").write_bytes(bytes(data))
    a = argv(tmp_path / "x", trained)
    a[a.index("--adapters") + 1] = str(other)
    assert main(a) == 2


def test_a_failed_audit_is_refused_on_the_laptop(tmp_path, trained):
    a = argv(tmp_path, trained, weights=("base",))
    r = json.loads((tmp_path / "leakcheck-receipt.json").read_text())
    (tmp_path / "leakcheck-receipt.json").write_text(json.dumps(dict(r, status="FAIL", n_problems=3)))
    assert main(a) == 2 and not (tmp_path / "QUEUE.json").exists()


def test_a_receipt_of_other_leak_rules_is_refused_on_the_laptop(tmp_path, trained, capsys):
    """The code_sha256 rule reaches genqueue through genprep.preflight: a receipt without code_sha256 builds no
    queue."""
    a = argv(tmp_path, trained, weights=("base",))
    r = json.loads((tmp_path / "leakcheck-receipt.json").read_text())
    del r["code_sha256"]
    (tmp_path / "leakcheck-receipt.json").write_text(json.dumps(r))
    assert main(a) == 2 and not (tmp_path / "QUEUE.json").exists() and "code_sha256" in capsys.readouterr().err


def test_variants_go_through_the_resolver_and_gen_canary_is_refused(tmp_path, trained, capsys):
    """A grid label is accepted by --variant (refused later: this fixture's prompts are test/aux, not dev);
    gen_canary never gets a queue; a malformed label is an argparse error."""
    a = argv(tmp_path, trained, weights=("base",))
    a[a.index("--variant") + 1] = "dev_grid_k8_n3_r050"
    assert main(a) == 2 and "partitions" in capsys.readouterr().err        # dev grid labels read dev
    a[a.index("--variant") + 1] = "gen_canary"
    assert main(a) == 2 and "gen_canary" in capsys.readouterr().err
    a[a.index("--variant") + 1] = "dev_grid_k8_n3"
    with pytest.raises(SystemExit) as e:
        main(a)
    assert e.value.code == 2 and "dev_grid_k8_n3" in capsys.readouterr().err
    assert not (tmp_path / "QUEUE.json").exists()


def test_the_command_line_must_be_consistent(tmp_path, trained):
    a = argv(tmp_path, trained, weights=("base", "s42", "s11"))            # two adapter weights, one train receipt
    assert main(a) == 2
    a = argv(tmp_path / "y", trained, weights=("base",))
    a[a.index("--deadline-utc") + 1] = "2026-10-08T12:00:00"             # no timezone
    assert main(a) == 2


def test_generation_pins_match_requirements_gen():
    """Every declared pin set has its requirements file (the fallbacks: requirements-gen-<set>.in), same extras."""
    def read(name):
        lines = (ROOT / name).read_text().splitlines()
        pin = r"^([A-Za-z0-9_.-]+)==(\S+)"
        pins = dict(re.match(pin, x).groups() for x in lines if re.match(pin, x))
        return pins, [x for x in lines if x and not x.startswith("#") and "==" not in x]

    main_pins, extras = read("requirements-gen.in")
    assert main_pins == GEN_PINS == PIN_SETS["main"]
    for name, pins in PIN_SETS.items():
        if name != "main":
            assert read(f"requirements-gen-{name}.in") == (pins, extras), name


def test_bootstrap_installs_the_pin_set_chosen_at_the_canary(tmp_path):
    """decision 24/09: KIND=gen GEN_SET=<set> builds the generation venv of a declared fallback set."""
    def boot(**env):
        return dry_run(tmp_path, **env)

    out = boot(KIND="gen", GEN_SET="fallback-tf")
    assert out.returncode == 0 and "requirements-gen-fallback-tf.in" in out.stdout and "BOOTSTRAP_OK" in out.stdout
    assert boot(KIND="gen", GEN_SET="custom").returncode == 2
    assert boot(KIND="gpu", GEN_SET="fallback-tf").returncode == 2


def test_bootstrap_builds_the_generation_venv_on_request(tmp_path):
    out = dry_run(tmp_path, default_local=True, KIND="gen").stdout      # the default root itself, dry
    assert "/opt/v2/venv-gen" in out and "requirements-gen.in" in out and "requirements-gpu" not in out
    steps = ["nvidia-smi", "venv-gen", "v2gpu.fetch_models", "tests_gpu/test_generate.py", "no_skips.sh",
             "import vllm", "v2gpu.receipt", "BOOTSTRAP_OK"]
    pos = [out.index(s) for s in steps]
    assert pos == sorted(pos) and "test_train_loop" not in out
    assert not (tmp_path / "venv-gen").exists()
    default = dry_run(tmp_path).stdout
    assert "requirements-gpu.in" in default and "venv-gen" not in default and "import vllm" not in default


def test_the_generation_run_sheet_names_every_gate():
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    for needle in ("v2gpu.gencanary", "--probe-longest", "--reverse", "--trained", "v2gpu.genqueue",
                   "v2gpu.gencoverage", "--recheck-of", "KIND=gen", "tác giả", "--teacher-force-of",
                   "--extend-cap 40", "--repeat-longest 10", "--recheck-only", "--recheck-max-hours",
                   "--leakcheck-receipts", "--train-receipts", "--workdirs", "VLLM_BATCH_INVARIANT=1",
                   "pod cho mỗi đường dẫn volume", "BLOCKED_BY_", "--canary-verdict", "GEN_SET=", "--pin-set",
                   "--ids-sha256", "--copy-window-minutes", "--canary-train-receipt", "probed_sums", "results-lost",
                   "/workspace/prompts/$D", "adapters/$MODEL-s42-${ASHA:0:12}", "bất kỳ commit",
                   "≥ 0,95 trên MỖI cặp VÀ trung bình theo cặp ≥ 0,98", "probed_prompt",
                   "venv huấn luyện (`.venv-gpu`)", "cửa sổ chép 30 phút",
                   "P=/workspace/prompts/$D; PT=/workspace/prompts/$T", "receipt-train.$MODEL-s42-$A12.json",
                   "W=/workspace/runs/gencanary-$MODEL-$A12"):         # one driver per model ($MODEL)
        assert needle in sheet, needle


def test_the_run_sheet_quotes_its_commands_and_copies_every_workdir():
    """A card name with spaces survives only in a bash array; the <sha> is computed. Every
    gen-* and recheck-* workdir of the QUEUE comes home before the pod may go."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    assert 'C=(' in sheet and '"${C[@]}"' in sheet and "<sha>" not in sheet
    assert "for job in $(" in sheet and "QUEUE" in section(sheet, 7) and "COPY_OK" in section(sheet, 7)


def section(sheet, number):
    start = sheet.index(f"\n## {number}.")
    end = sheet.find("\n## ", start + 1)
    return sheet[start:end if end >= 0 else len(sheet)]


def test_the_run_sheet_reads_both_clouds_and_creates_where_the_card_is_in_stock():
    """RTX A4000 was out on SECURE and Low on COMMUNITY (24/09); the card name pin does not change."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    prices, create = section(sheet, 2), section(sheet, 3)
    assert "SECURE" in prices and "COMMUNITY" in prices and "NVIDIA RTX A4000" in prices and "13.0" in prices
    assert "cloud" in create and "NVIDIA RTX A4000" in create


def test_the_run_sheet_memory_steps_are_the_two_pre_declared_ones_chosen_at_the_canary():
    """decision 24/09: 0.95, then enforce_eager, pre-declared, chosen ONLY at the canary and fixed for the whole
    run; max_num_batched_tokens 4,096 is NOT pre-declared."""
    canary = section((ROOT / "ops" / "GEN_RUNSHEET.md").read_text(), 5)
    steps = [canary.index(s) for s in ("--gpu-memory-utilization 0.95", "--enforce-eager", "max_num_batched_tokens")]
    assert steps == sorted(steps) and "CHỈ ở canary" in canary and "cả lượt" in canary
    assert "4.096 KHÔNG" in canary


def test_decision_0510_the_canary_block_runs_at_the_eager_step_with_batch_invariance():
    """decision 05/10: fallback (a) is on and every job runs at (0.95, enforce_eager), so the canary block's default
    step is m095e (the engine refuses any other step) and the fallback paragraph says why and since when."""
    canary = section((ROOT / "ops" / "GEN_RUNSHEET.md").read_text(), 5)
    assert "PIN=main; STEP=m095e; M=(--gpu-memory-utilization 0.95 --enforce-eager); TRY=1" in canary
    assert "STEP=m092" not in canary
    assert "decision 05/10" in canary and "evidence-xpod-v0-mismatch" in canary and "batch_invariant: true" in canary
    assert "8d7ab5a" in canary and "e0f346b" in canary                     # the ledger entries of the supplement
    assert "GPU UUID" in canary and "lặp lại cùng mẫu" in canary            # what the 2-GPU gate records


def test_the_run_sheet_waits_for_the_card_and_copies_each_job_as_soon_as_it_is_done():
    """decision 24/09: stock-out means WAIT (report after 24 h; a class change redoes the WHOLE main run
    after a new canary); each DONE job is copied home at once; delete only after COPY_OK and coverage PASS."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    create, copy, end = section(sheet, 3), section(sheet, 7), section(sheet, 8)
    assert "CHỜ" in create and "24 giờ" in create and "TOÀN BỘ lượt chính" in create and "_SHRINK.json" in create
    assert ".status" in copy and "DONE" in copy and "copy_out.sh" in copy and "ngay khi" in copy
    assert "tự dừng" in end and "COPY_OK" in end and "gencoverage" in end and "xoá" in end


@pytest.mark.parametrize("what", ["--train-receipts", "--adapters"])
def test_a_missing_train_receipt_or_adapter_is_a_refusal_not_a_crash(tmp_path, trained, capsys, what):
    """train_receipt_problems reads the receipt and the adapter's tensors like
    canary_digest does: a missing file is a REFUSED problem (exit 2), not a traceback."""
    a = argv(tmp_path, trained)
    a[a.index(what) + 1] = str(tmp_path / "nowhere")
    assert main(a) == 2
    assert "nowhere" in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


# genqueue's train_receipt_problems requires train_code_sha256 and
# train_cfg_sha256, and every adapter of a variant must share one of each


@pytest.mark.parametrize("field", ["train_code_sha256", "train_cfg_sha256"])
def test_a_train_receipt_without_the_code_or_config_digest_is_refused(tmp_path, trained, capsys, field):
    run, _ = trained
    other = tmp_path / "no-digest-receipt.json"
    rc = json.loads((run / "receipt-train.json").read_text())
    del rc[field]
    other.write_text(json.dumps(rc))
    a = argv(tmp_path, trained, receipts=[str(other)])
    assert main(a) == 2 and field in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


def test_train_code_consistency_problems_flags_a_disagreement_across_adapters():
    from v2gpu.genqueue import train_code_consistency_problems

    agree_code, agree_cfg = {"s42": "x", "s43": "x"}, {"s42": "y", "s43": "y"}
    assert train_code_consistency_problems(agree_code, agree_cfg) == []
    disagree_code = {"s42": "x", "s43": "z"}
    problems = train_code_consistency_problems(disagree_code, agree_cfg)
    assert len(problems) == 1 and "train_code_sha256" in problems[0] and "s42=x" in problems[0] and \
        "s43=z" in problems[0]
    disagree_cfg = {"s42": "y", "s43": "w"}
    problems = train_code_consistency_problems(agree_code, disagree_cfg)
    assert len(problems) == 1 and "train_cfg_sha256" in problems[0]
    # a None (already refused elsewhere by train_receipt_problems) is never itself a disagreement
    assert train_code_consistency_problems({"s42": "x", "s43": None}, agree_cfg) == []


def test_main_refuses_two_real_adapters_disagreeing_on_train_code_sha256(tmp_path, trained, capsys):
    """End to end (not just the pure function above): main() must actually call train_code_consistency_problems and
    surface its problems, for a second adapter under another seed that reused the same real tensor file."""
    run, digest = trained
    other_dir = tmp_path / "other-adapter"
    other_dir.mkdir()
    shutil.copy2(run / "epoch-3" / "adapter_model.safetensors", other_dir / "adapter_model.safetensors")
    shutil.copy2(run / "epoch-3" / "adapter_config.json", other_dir / "adapter_config.json")
    other_rc = {**json.loads((run / "receipt-train.json").read_text()), "seed": 99,
               "train_code_sha256": {"v2gpu/train_job.py": "9" * 64}}
    other_receipt = tmp_path / "other-receipt.json"
    other_receipt.write_text(json.dumps(other_rc))
    a = argv(tmp_path, trained, weights=("base", "s42", "s99"),
            receipts=[str(run / "receipt-train.json"), str(other_receipt)])
    i = a.index("--adapters")
    a[i + 1:i + 2] = [str(run / "epoch-3"), str(other_dir)]
    assert main(a) == 2
    err = capsys.readouterr().err
    assert "train_code_sha256" in err and "differs across adapters" in err, err


def test_the_refusal_shows_the_count_and_at_most_twenty_problems():
    """A long refusal stays readable and hides nothing."""
    from v2gpu.genqueue import refusal_message

    assert refusal_message([f"p{i}" for i in range(25)]) == \
        "REFUSED (25 problems): " + "; ".join(f"p{i}" for i in range(20)) + " (+5 more)"
    assert refusal_message(["only"]) == "REFUSED (1 problem): only"



def test_a_queue_needs_the_variants_decision_and_a_trained_verdict_that_agrees(tmp_path, trained, capsys):
    """The decision is taken once for the variant, before any main QUEUE, from every model's canary; a model's
    trained verdict must carry that same decision."""
    a = argv(tmp_path, trained, weights=("base",))
    write_decision(tmp_path, {**VERDICT["decision"], "gpu_memory_utilization": 0.92, "enforce_eager": False})
    assert main(a) == 2 and "is not this model's trained canary decision" in capsys.readouterr().err
    write_decision(tmp_path, models=("deepseek",))                     # not every model main plans
    assert main(a) == 2 and "codellama" in capsys.readouterr().err
    write_decision(tmp_path, variant="dev_pilot_v0", models=("deepseek",))
    assert main(a) == 2 and "dev_pilot_v0" in capsys.readouterr().err
    write_decision(tmp_path)
    good = json.loads((tmp_path / "decision.json").read_text())
    for bad in ({**good, "pass": False}, {**good, "schema": "gtestgen.v2.other.v1"},
                {k: v for k, v in good.items() if k != "decision"}):   # a refusal record of decide has no decision
        (tmp_path / "decision.json").write_text(json.dumps(bad))
        assert main(a) == 2 and "not a passing gtestgen.v2.gen_decision.v2" in capsys.readouterr().err, bad
    write_decision(tmp_path)
    assert main(a) == 0


def test_a_recheck_whose_original_the_pod_would_refuse_is_refused_on_the_laptop(tmp_path, trained, monkeypatch,
                                                                                 capsys):
    """The recheck command's identity core is computed here, as the pod computes it, and compared with the
    original's job.json (generate._core): a regenerated leak receipt or another memory step fails here, not on a
    paid pod."""
    a = argv(tmp_path, trained, "--recheck-only", weights=("base",))
    orig = real_gen_workdir(tmp_path, monkeypatch, change=lambda i: i.update(leakcheck_receipt_sha256="0" * 64))
    assert main(a + ["--originals", str(orig)]) == 2
    err = capsys.readouterr().err
    assert "leakcheck_receipt_sha256" in err and "identity" in err and not (tmp_path / "QUEUE.json").exists()


def test_a_normal_queue_runs_prepended_cross_pod_rechecks_first_with_one_window_and_one_stop(tmp_path, trained,
                                                                                            monkeypatch, capsys):
    """The dev pilot's (or another model's) cross-pod rechecks ride on this pod's QUEUE: first, then this pod's
    jobs, one copy window, one self-stop, one merged ship map. The original shipped for a recheck is never also a
    workdir of this pod's own jobs."""
    run, _ = trained
    xa = argv(tmp_path, trained, "--recheck-only", weights=("base",))
    xa[xa.index("--out") + 1] = str(tmp_path / "XQUEUE.json")
    orig = real_gen_workdir(tmp_path, monkeypatch)
    assert main(xa + ["--originals", str(orig)]) == 0
    a = [x for x in xa if x != "--recheck-only"]                 # the same inputs, this pod's own weights
    a[a.index("--out") + 1] = str(tmp_path / "QUEUE.json")
    a[a.index("--weights") + 1:a.index("--weights") + 2] = ["base", "s42"]
    a += ["--train-receipts", str(run / "receipt-train.json"), "--adapters", str(run / "epoch-3"),
          "--prepend-queue", str(tmp_path / "XQUEUE.json")]
    assert main(a) == 2                                        # its gen-main-deepseek-base would meet the original
    assert f"{POD}/runs/gen-main-deepseek-base" in capsys.readouterr().err
    a.remove("base")                                           # this pod runs s42; base ran on the other pod
    assert main(a) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    names = [j["name"] for j in q["jobs"]]
    assert names == ["recheck-main-deepseek-base-xpod", "gen-main-deepseek-s42", "recheck-main-deepseek-s42",
                     "copy-window-main-deepseek"]
    assert q["self_stop_cmd"] == ["runpodctl", "stop", "pod", "$RUNPOD_POD_ID"]
    assert sum(n.startswith("copy-window") for n in names) == 1
    x = json.loads((tmp_path / "XQUEUE.json").read_text())
    assert x["ship"]["originals"].items() <= q["ship"]["originals"].items() and q["ship"]["originals"]
    assert q["jobs"][0] == x["jobs"][0] and q["queue_dir"] == x["queue_dir"]


def test_originals_without_recheck_only_are_refused_with_the_way_to_prepend_them(tmp_path, trained, monkeypatch,
                                                                                capsys):
    """Another model's or variant's originals have other prompts, adapters and verdicts; they ride on this queue
    as a --recheck-only QUEUE.json given to --prepend-queue."""
    a = argv(tmp_path, trained, weights=("base",))
    orig = real_gen_workdir(tmp_path, monkeypatch)
    assert main(a + ["--originals", str(orig)]) == 2 and "--prepend-queue" in capsys.readouterr().err


def test_only_a_recheck_only_queue_can_be_prepended(tmp_path, trained, capsys):
    other = argv(tmp_path / "x", trained, weights=("base",))
    assert main(other) == 0
    a = argv(tmp_path, trained, "--prepend-queue", str(tmp_path / "x" / "QUEUE.json"))
    assert main(a) == 2 and "recheck-only" in capsys.readouterr().err
    q = json.loads((tmp_path / "x" / "QUEUE.json").read_text())
    for names in (["gen-main-deepseek-base-xpod", "copy-window-main-deepseek-xpod"],     # a job that is no recheck
                  ["recheck-main-deepseek-base-xpod", "recheck-main-deepseek-s42-xpod"]):  # no copy window last
        (tmp_path / "odd.json").write_text(json.dumps({**q, "jobs": [{**q["jobs"][0], "name": n} for n in names]}))
        a[a.index("--prepend-queue") + 1] = str(tmp_path / "odd.json")
        assert main(a) == 2 and "recheck-only" in capsys.readouterr().err, names


def test_the_canary_train_receipt_must_be_the_queues_own_s42(tmp_path, trained, capsys):
    """A queue holding the official s42 cannot be bound to a canary of another s42 adapter (the pilot's)."""
    run, _ = trained
    other = tmp_path / "pilot-receipt.json"
    other.write_text(json.dumps({**json.loads((run / "receipt-train.json").read_text()),
                                 "selected_adapter": {"tensor_sha256": "9" * 64}}))
    a = argv(tmp_path, trained)
    a[a.index("--canary-train-receipt") + 1] = str(other)
    assert main(a) == 2 and "--canary-train-receipt" in capsys.readouterr().err


def test_the_recheck_sample_is_at_least_one(tmp_path, trained, capsys):
    """A recheck sample of 0 is refused."""
    a = argv(tmp_path, trained, weights=("base",))
    a[a.index("--recheck-sample") + 1] = "0"
    assert main(a) == 2 and "--recheck-sample" in capsys.readouterr().err


def test_the_partitions_are_exactly_those_the_variant_plans_for_the_model(tmp_path, trained, capsys):
    """DeepSeek main plans test and aux_test; a queue of test alone would finish DONE without aux lines."""
    a = argv(tmp_path, trained, weights=("base",))
    i = a.index("--partitions")
    assert main(a[:i + 1] + ["test"] + a[i + 3:]) == 2 and "aux_test" in capsys.readouterr().err


def test_a_train_receipt_that_is_not_an_object_is_refused(tmp_path, trained, capsys):
    """A train receipt that is not a JSON object is refused, not a crash."""
    bad = tmp_path / "list-receipt.json"
    bad.write_text("[]")
    assert main(argv(tmp_path, trained, receipts=[str(bad)])) == 2 and "not a JSON object" in capsys.readouterr().err


# The pod's model cache


def test_every_command_names_the_pods_model_cache(tmp_path, trained):
    """The bootstrap downloads the models into {root}/hf; the jobs run offline and must look there."""
    assert main(argv(tmp_path, trained, weights=("base",))) == 0
    for job in json.loads((tmp_path / "QUEUE.json").read_text())["jobs"][:-1]:
        assert job["cmd"][:4] == ["env", "HF_HUB_OFFLINE=1", f"HF_HOME={POD}/hf", f"{POD}/venv-gen/bin/python"]


def test_a_run_tag_gives_a_lost_pods_jobs_new_names_and_workdirs(tmp_path, trained, capsys):
    """On a re-attached volume the old workdir (other pod) and the old .status files would block the same names;
    --run-tag names the new jobs and workdirs apart."""
    a = argv(tmp_path, trained, "--run-tag", "r2", "--results", str(tmp_path / "results"), weights=("base",))
    assert main(a) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    assert [j["name"] for j in q["jobs"]] == ["gen-main-deepseek-base-r2", "recheck-main-deepseek-base-r2",
                                              "copy-window-main-deepseek-r2"]
    rc = q["jobs"][1]["cmd"]
    assert q["jobs"][0]["workdir"] == f"{POD}/runs/gen-main-deepseek-base-r2" == rc[rc.index("--recheck-of") + 1]
    assert q["jobs"][1]["workdir"] == f"{POD}/runs/recheck-main-deepseek-base-r2" == rc[rc.index("--workdir") + 1]
    for bad in ("R2", "../x", "r 2", ""):
        a[a.index("--run-tag") + 1] = bad
        assert main(a) == 2 and "--run-tag" in capsys.readouterr().err


def test_ship_copies_every_target_and_the_queue_then_checks_every_sha256_on_the_pod(tmp_path, trained):
    """ops/ship.sh QUEUE.json IP PORT KEY, dry run: one scp per shipped file to its target, QUEUE.json to the pod
    root, then sha256sum -c of the digests computed on the laptop."""
    assert main(argv(tmp_path, trained)) == 0
    q = json.loads((tmp_path / "QUEUE.json").read_text())
    out = subprocess.run(["bash", str(ROOT / "ops" / "ship.sh"), str(tmp_path / "QUEUE.json"), "1.2.3.4", "2222",
                          "/k/id"], capture_output=True, text=True,
                         env={**os.environ, "DRY_RUN": "1", "KNOWN_HOSTS": str(tmp_path / "known_hosts.pod-1")})
    assert out.returncode == 0 and out.stdout.rstrip().endswith("SHIP_OK"), out.stderr
    lines = out.stdout.splitlines()
    copies = {l.split()[-1].split(":", 1)[1]: l.split()[-2] for l in lines if l.startswith("+ scp")}
    files = {t: s for sec in q["ship"].values() for t, s in sec.items() if Path(s).is_file()}
    adapter_target, adapter_dir = next(iter(q["ship"]["adapters"].items()))
    files |= {f"{adapter_target}/{p.name}": str(p) for p in Path(adapter_dir).iterdir() if p.is_file()}
    files[f"{POD}/QUEUE.json"] = str(tmp_path / "QUEUE.json")
    sums = [l for l in copies if "SHIP_SHA256SUMS" in l]
    assert {t: s for t, s in copies.items() if t not in sums} == files and len(sums) == 1
    expect = {l.split()[2]: l.split()[1] for l in lines if l.startswith("  expect ")}
    assert expect == {t: sha(s) for t, s in files.items()}
    check = next(l for l in lines if "sha256sum -c" in l)
    assert lines.index(check) > max(i for i, l in enumerate(lines) if l.startswith("+ scp")) and sums[0] in check
    assert "-p 2222" in check and "root@1.2.3.4" in check and "-i /k/id" in check
    bad = subprocess.run(["bash", str(ROOT / "ops" / "ship.sh"), str(tmp_path / "QUEUE.json")], capture_output=True,
                         text=True, env={**os.environ, "DRY_RUN": "1", "KNOWN_HOSTS": str(tmp_path / "known_hosts")})
    assert bad.returncode == 2 and "usage" in bad.stderr


def test_the_run_sheet_runs_generation_with_the_pods_model_cache():
    """HF_HOME=/opt/v2/hf (since 05/10: the bootstrap's cache on the container disk) with HF_HUB_OFFLINE=1
    in the canary array G and on the queue launch, which names
    the generation venv's python itself (inside the proven `setsid nohup bash -c "..."` shape, so no PY
    variable any more)."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    g = next(l for l in section(sheet, 5).splitlines() if l.startswith("G=("))
    assert "HF_HUB_OFFLINE=1" in g and "HF_HOME=/opt/v2/hf" in g
    run = section(sheet, 6)
    launch = next(l for l in run.splitlines() if "v2gpu.jobqueue" in l)
    assert "HF_HOME=/opt/v2/hf" in launch and "HF_HUB_OFFLINE=1" in launch
    assert "/opt/v2/venv-gen/bin/python -m v2gpu.jobqueue" in launch


def test_the_run_sheet_uses_the_3a_prompt_names_and_never_the_train_files():
    """3A writes {model}.{pool}.{part}.{fmt}.jsonl; --model-prompts names dev (+ test/aux) files only."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    assert not re.search(r"(?:dev|test|aux_test|train)\.deepseek\.\*\.jsonl", sheet)
    for glob in ("deepseek.all.dev.*.jsonl", "deepseek.all.test.*.jsonl", "deepseek.all.aux_test.*.jsonl"):
        assert glob in sheet, glob
    verdict = sheet[sheet.index("-m v2gpu.gencanary --a"):].split("--out")[0]         # the gencanary command
    mp = [l for l in section(sheet, 5).splitlines() if "MP=(" in l]                    # its --model-prompts
    assert '--model-prompts "${MP[@]}"' in verdict and mp and all(".all.dev.*.jsonl" in l for l in mp)
    assert not any("all.train" in l for l in mp)


def test_the_run_sheet_coverage_leaves_out_the_queue_folders():
    """copy_out.sh leaves results/<job>.queue/ beside each workdir; the coverage glob drops them."""
    copy = section((ROOT / "ops" / "GEN_RUNSHEET.md").read_text(), 7)
    cmd = copy[copy.index("v2gpu.gencoverage"):]
    assert "--workdirs $(ls -d results/gen-dev_pilot_v0-deepseek-* results/recheck-dev_pilot_v0-deepseek-* | grep -v " \
           "'\\.queue$')" in cmd and "*.queue" in copy


def test_the_run_sheet_decides_once_per_variant_before_the_first_main_queue():
    """decision 24/09: every model's canary with its test/aux probe, then gencanary decide; a mismatch names the
    target and the models to re-run there in fresh workdirs, then decide again; then the trained canary per model
    before its main QUEUE; genqueue takes --decision."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    canary = section(sheet, 5)
    assert "TRƯỚC QUEUE `main` đầu tiên của bất kỳ mô hình nào" in canary
    assert "v2gpu.gencanary decide --variant main" in canary and "--prompts-sums \"$PT/PROMPTS_SHA256SUMS\"" in canary
    assert "`target`" in canary and "`rerun`" in canary and "rồi chạy `decide` lại" in canary
    assert "fallback-tf-bnb" in canary and "--ids-sha256" in canary
    assert "W=/workspace/runs/gencanary-$MODEL-$A12-$PIN-$STEP" in canary and "workdir MỚI" in canary
    assert "STEP=m095" in canary and "STEP=m095e" in canary and '--pin-set "$PIN"' in canary
    assert "--decision results/decision.dev_pilot_v0.json" in section(sheet, 1)
    assert "gencanary decide --variant dev_pilot_v0" in canary


def test_the_run_sheet_ships_with_ship_sh_commits_the_lock_and_recovers_a_lost_pod_with_a_run_tag():
    """GEN_RUNSHEET.md §6: cross-pod rechecks ride at the head of another pod's QUEUE (the dev
    pilot's on the second pod of the dev phase or a short pod, see test_runsheets.py)."""
    sheet = (ROOT / "ops" / "GEN_RUNSHEET.md").read_text()
    setup, run = section(sheet, 4), section(sheet, 6)
    assert "bash ops/ship.sh" in setup and "SHIP_OK" in setup
    assert "requirements-gen.lock" in setup and "commit" in setup and "bundle" in setup and "pod thứ hai" in setup
    assert "--run-tag" in run and "--prepend-queue" in run and "--recheck-only" in run
    assert "MỘT cửa sổ chép" in run and "MỘT lần tự dừng" in run


# Card classes (decision 24/09)


def test_the_queue_names_a_generation_card_class(tmp_path, trained, capsys):
    a = argv(tmp_path, trained, weights=("base",))
    a[a.index("--gpu-name") + 1] = "NVIDIA RTX 4000 Ada Generation"
    assert main(a) == 2 and "card class" in capsys.readouterr().err


# The queue runs the code and the engine the canary ran


def test_every_command_carries_the_decisions_sha256(tmp_path, trained):
    assert main(argv(tmp_path, trained)) == 0
    for job in json.loads((tmp_path / "QUEUE.json").read_text())["jobs"][:-1]:
        assert job["cmd"][job["cmd"].index("--decision-sha256") + 1] == sha(tmp_path / "decision.json")


def test_a_route_flipped_after_the_canary_is_refused(tmp_path, trained, monkeypatch, capsys):
    """As reproduced once: BATCH_INVARIANT flipped after the canary gave exit 0 (since decision 05/10 the canary runs
    with it on, so the flip turns it off)."""
    from v2gpu import engines

    a = argv(tmp_path, trained, weights=("base",))
    monkeypatch.setattr(engines, "BATCH_INVARIANT", False)
    monkeypatch.setitem(engines.VLLM_ENV, "VLLM_BATCH_INVARIANT", "0")
    assert main(a) == 2 and "engine" in capsys.readouterr().err and not (tmp_path / "QUEUE.json").exists()


def test_a_decision_made_by_other_generation_code_is_refused(tmp_path, trained, capsys):
    a = argv(tmp_path, trained, weights=("base",))
    write_decision(tmp_path, gen_code_sha256={"v2gpu/generate.py": "0" * 64})
    assert main(a) == 2 and "gen_code_sha256" in capsys.readouterr().err


def test_the_bootstrap_tests_pass_with_the_bootstrap_selectors_exported(tmp_path):
    """On a pod the suite runs inside `KIND=gen [GEN_SET=...] bash ops/pod_bootstrap.sh`; the bootstrap's own
    tests must not read those selectors (the gen bootstrap gate failed 2 of 490 tests)."""
    tests = ["tests_gpu/test_genqueue.py::test_bootstrap_installs_the_pin_set_chosen_at_the_canary",
             "tests_gpu/test_genqueue.py::test_bootstrap_builds_the_generation_venv_on_request",
             "tests_gpu/test_ops_scripts.py"]
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                          str(tmp_path / "bt"), *tests], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, **LEAKED})
    assert run.returncode == 0, run.stdout[-3000:]


def test_the_bootstrap_runs_the_suite_without_its_own_selectors(tmp_path):
    """Belt and braces: the pod's test step equals the laptop's, whatever KIND/GEN_SET the operator exported.
    A plain root recollection (the v2test gate) runs too, with no runuser wrapper at all."""
    for kind, env in (("gen", LEAKED), ("gpu", {"KIND": "gpu"})):
        out = dry_run(tmp_path, ws=tmp_path / kind, **env).stdout
        lines = [l for l in out.splitlines() if "pytest" in l and ("--collect-only" in l or " -q -rs " in l)]
        assert len(lines) == 3, (kind, lines)
        switched = [l for l in lines if "runuser -u v2test -- env -i " in l]
        assert len(switched) == 2, (kind, lines)
        assert not any(re.search(r"\b(KIND|GEN_SET|FIRST_RUN)=", l) for l in lines), (kind, lines)


SIM = "tests_gpu/test_genqueue.py::test_the_generation_test_list_passes_in_a_simulated_generation_venv"


def test_the_training_bootstrap_leaves_out_the_nested_generation_simulation(tmp_path):
    """The gpu bootstrap deselects the nested gen-list simulation in BOTH the count and the run (the gate counts
    the collected tests), and tells the gate one test is deselected; the gen bootstrap runs everything."""
    gpu = dry_run(tmp_path, ws=tmp_path / "gpu").stdout
    lines = [l for l in gpu.splitlines() if "pytest" in l and ("--collect-only" in l or " -q -rs " in l)]
    assert len(lines) == 3 and all(f"--deselect {SIM}" in l for l in lines), lines   # +1 root recollection line
    gate = next(l for l in gpu.splitlines() if "no_skips.sh" in l)
    assert gate.split()[-1] == "1"                                   # one deselected test, exactly
    gen = dry_run(tmp_path, ws=tmp_path / "gen", KIND="gen").stdout
    assert "--deselect" not in gen and next(l for l in gen.splitlines() if "no_skips.sh" in l).split()[-1] == "0"


def test_the_bootstrap_reads_the_count_when_a_test_is_deselected():
    """`pytest --collect-only -q` prints "723/724 tests collected (1 deselected)"; N is 723."""
    script = (ROOT / "ops" / "pod_bootstrap.sh").read_text()
    prog = re.search(r"tail -n 1 \|\s*awk '([^']+)'", script).group(1)
    for line, n in (("723/724 tests collected (1 deselected) in 1.09s", "723"), ("724 tests collected in 1.09s",
                                                                                "724")):
        assert subprocess.run(["awk", prog], input=line + "\n", capture_output=True, text=True).stdout.strip() == n


# Mutants that once survived


def xpod_queue(tmp_path, trained, monkeypatch):
    """A --recheck-only QUEUE of this fixture's inputs, and the argv of a normal queue for s42 that could prepend it."""
    run, _ = trained
    xa = argv(tmp_path, trained, "--recheck-only", weights=("base",))
    xa[xa.index("--out") + 1] = str(tmp_path / "XQUEUE.json")
    orig = real_gen_workdir(tmp_path, monkeypatch)
    assert main(xa + ["--originals", str(orig)]) == 0
    a = [x for x in xa if x != "--recheck-only"]
    a[a.index("--out") + 1] = str(tmp_path / "QUEUE.json")
    a[a.index("--weights") + 1] = "s42"
    return a + ["--train-receipts", str(run / "receipt-train.json"), "--adapters", str(run / "epoch-3")]


def test_a_prepended_recheck_queue_of_another_pin_set_or_card_is_refused(tmp_path, trained, monkeypatch, capsys):
    """A pod has one venv-gen (one GEN_SET) and one card: rechecks built under another pin set would fail
    REFUSED_ENVIRONMENT at the head of the queue and block every job after them."""
    a = xpod_queue(tmp_path, trained, monkeypatch)
    assert main(a + ["--prepend-queue", str(tmp_path / "XQUEUE.json")]) == 0
    q = json.loads((tmp_path / "XQUEUE.json").read_text())
    for flag, value in (("--pin-set", "fallback-bnb"), ("--gpu-name", "NVIDIA RTX 2000 Ada Generation")):
        odd = [{**j, "cmd": j["cmd"][:j["cmd"].index(flag) + 1] + [value] + j["cmd"][j["cmd"].index(flag) + 2:]}
               if "cmd" in j and flag in j["cmd"] else j for j in q["jobs"]]
        (tmp_path / "odd.json").write_text(json.dumps({**q, "jobs": odd}))
        assert main(a + ["--prepend-queue", str(tmp_path / "odd.json")]) == 2, flag
        assert value in capsys.readouterr().err


def test_a_prepended_queue_of_another_pod_root_or_on_a_recheck_only_queue_is_refused(tmp_path, trained,
                                                                                     monkeypatch, capsys):
    """A prepended queue of another pod root, or a prepend on a recheck-only queue, is refused."""
    a = xpod_queue(tmp_path, trained, monkeypatch)
    q = json.loads((tmp_path / "XQUEUE.json").read_text())
    for change in ({"queue_dir": "/other/queue"}, {"lock_path": "/root/other.lock"}, {"self_stop_cmd": ["true"]}):
        (tmp_path / "odd.json").write_text(json.dumps({**q, **change}))
        assert main(a + ["--prepend-queue", str(tmp_path / "odd.json")]) == 2, change
        assert "another pod root" in capsys.readouterr().err
    assert main(a + ["--recheck-only", "--prepend-queue", str(tmp_path / "XQUEUE.json")]) == 2
    assert "not on a --recheck-only one" in capsys.readouterr().err


def test_a_recovery_queue_never_lists_weights_already_done_at_home(tmp_path, trained, monkeypatch, capsys):
    """--run-tag renames every job, so a recovery queue listing weights whose gen workdir is DONE and copied
    home would regenerate them (and coverage would then count every pair twice): --results is required with
    --run-tag, and such weights are refused (a DONE gen whose recheck was lost goes through --recheck-only)."""
    a = argv(tmp_path, trained, "--run-tag", "r2", weights=("base", "s42"))
    assert main(a) == 2 and "--results" in capsys.readouterr().err
    home = tmp_path / "results"
    real_gen_workdir(tmp_path, monkeypatch)                              # results/gen-main-deepseek-base: DONE
    tagged = home / "gen-main-deepseek-s42-r1"                          # an earlier recovery, not DONE
    tagged.mkdir()
    (tagged / "receipt-generate.json").write_text(json.dumps({"status": "FAILED"}))
    (home / "gen-main-deepseek-base.queue").mkdir()
    assert main(a + ["--results", str(home)]) == 2
    err = capsys.readouterr().err
    assert "gen-main-deepseek-base" in err and "--recheck-only" in err and "gen-main-deepseek-s42" not in err
    a[a.index("--weights") + 1:a.index("--weights") + 3] = ["s42"]
    assert main(a + ["--results", str(home)]) == 0
    (tagged / "receipt-generate.json").write_text(json.dumps({"status": "DONE"}))   # an earlier recovery, DONE
    assert main(a + ["--results", str(home)]) == 2 and "gen-main-deepseek-s42-r1" in capsys.readouterr().err


# The adapter was trained on the compose of the queue's prompts


def test_an_adapter_trained_on_another_compose_is_refused(tmp_path, trained, capsys):
    run, _ = trained
    other = tmp_path / "round1-receipt.json"
    other.write_text(json.dumps({**json.loads((run / "receipt-train.json").read_text()),
                                 "data": {"prompts": {"meta": {"compose": {**V0, "variant": "v1", "k": 12}}}}}))
    a = argv(tmp_path, trained, receipts=[str(other)])
    a[a.index("--canary-train-receipt") + 1] = str(other)
    assert main(a) == 2 and "compose" in capsys.readouterr().err


# Since 27/09/2026 a dev grid label's adapter is checked
# against the round's OWN dev_pilot_vN META compose (--dev-pilot-meta), never the grid's own directory. The rule
# itself (genprep.adapter_compose_problems, including the stale-ratio probe) is pinned in tests_gpu/test_genprep.py; here
# only genqueue's own wiring of --dev-pilot-meta (a new, optional flag) is checked: an unreadable file is a refusal,
# and a readable one changes nothing for a variant that does not use it (main, dev_pilot_vN).


def test_dev_pilot_meta_is_read_and_an_unreadable_one_refuses_the_queue(tmp_path, trained, capsys):
    a = argv(tmp_path, trained)
    assert main(a) == 0                                            # baseline: unaffected without the flag
    missing = tmp_path / "no-such-pilot-meta.json"
    assert main(a + ["--dev-pilot-meta", str(missing)]) == 2
    assert "no readable PROMPTS_META.json compose" in capsys.readouterr().err
    pilot = tmp_path / "pilot-meta.json"
    pilot.write_text(json.dumps({"compose": V0}))
    assert main(a + ["--dev-pilot-meta", str(pilot)]) == 0          # a valid one is accepted, unused by "main"


# The adapter's config digest travels with the job


def test_every_adapter_command_carries_the_adapter_config_digest(tmp_path, trained):
    run, _ = trained
    assert main(argv(tmp_path, trained)) == 0
    jobs = json.loads((tmp_path / "QUEUE.json").read_text())["jobs"]
    s42 = [j["cmd"] for j in jobs if "-s42" in j["name"]]
    assert s42 and all(c[c.index("--adapter-config-sha256") + 1] == sha(run / "epoch-3" / "adapter_config.json")
                       for c in s42)
    assert all("--adapter-config-sha256" not in j["cmd"] for j in jobs if "-base" in j["name"])


# The TRAINED verdict is bound to the decision and to this checkout


@pytest.mark.parametrize("key, value", [("gen_code_sha256", {"v2gpu/generate.py": "0" * 64}),
                                        ("engine_fingerprint", {"name": "vllm", "version": "0.28.0"}),
                                        ("engine_settings", {"batch_invariant": True}),
                                        ("gen_code_sha256", None), ("engine_fingerprint", None)])
def test_a_trained_verdict_of_other_code_or_another_engine_is_refused(tmp_path, trained, capsys, key, value):
    """The decision comes from untrained canaries; the trained canary runs later and must have run the same code
    and engine (until 25/09/2026 a trained verdict of other code was accepted with exit 0)."""
    a = argv(tmp_path, trained)
    edit_verdict(tmp_path, **{key: value})
    assert main(a) == 2 and key in capsys.readouterr().err


EMPTY_FINGERPRINT = {"name": None, "version": None, "quantization": None, "batch_invariant": None, "env": {},
                     "tuned_config_files": None, "sampling": {}, "kwargs": {}, "effective": {}, "lora_shrink": []}


@pytest.mark.parametrize("fingerprint", [None, EMPTY_FINGERPRINT, {"name": "vllm"}])
def test_a_binding_neither_the_verdict_nor_the_decision_names_is_refused(tmp_path, trained, capsys, fingerprint):
    """Equal is not enough: a fingerprint that neither names binds nothing; nor does the fingerprint of an empty
    engine record (the old dict of None values) or one without a version."""
    a = argv(tmp_path, trained)
    edit_verdict(tmp_path, engine_fingerprint=fingerprint)
    write_decision(tmp_path, engine_fingerprint=fingerprint)
    assert main(a) == 2 and "engine_fingerprint" in capsys.readouterr().err


def test_a_trained_verdict_of_another_adapter_config_is_refused(tmp_path, trained, capsys):
    """The verdict records the adapter_config.json sha256 of the adapter it ran."""
    a = argv(tmp_path, trained)
    edit_verdict(tmp_path, adapter_config_sha256="0" * 64)
    assert main(a) == 2 and "adapter_config" in capsys.readouterr().err


def test_a_recheck_only_queue_never_takes_a_run_tag(tmp_path, trained, monkeypatch, capsys):
    """--recheck-only --run-tag is always refused: the recheck workdirs keep their -xpod names, and a lost
    pod's half-done recheck workdir is deleted before the volume is re-attached."""
    a = argv(tmp_path, trained, "--recheck-only", "--run-tag", "r2", "--results", str(tmp_path / "empty"),
             weights=("base",))
    (tmp_path / "empty").mkdir()
    orig = real_gen_workdir(tmp_path, monkeypatch)
    assert main(a + ["--originals", str(orig)]) == 2 and "--run-tag" in capsys.readouterr().err
