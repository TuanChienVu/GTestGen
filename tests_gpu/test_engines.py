# tests_gpu/test_engines.py
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests_gpu.tiny import make_tiny_llama
from v2gpu.engines import (BATCH_INVARIANT, GEN_PINS, LORA_INPUT_SIZES, LORA_OPS, SHRINK_DEFAULT_KEYS, VLLM_ENV,
                           HFEngine, VLLMCfg, VLLMEngine, adapter_problems, device_problems, engine_fingerprint,
                           load_hf, nvidia_driver, pin_problems, sampling_kwargs, shrink_table,
                           teacher_forced_agreement, tuned_config_name, tuned_config_problems, unpack, vllm_kwargs)
from v2gpu.modeling import LoadCfg, load_for_training
from v2gpu.receipt import tensor_digest

REPO = Path(__file__).resolve().parents[1]
CFG = VLLMCfg(model="deepseek-ai/deepseek-coder-6.7b-instruct", revision="e5d64addd26a6a1db0f9b863abf6ee3141936807",
              max_model_len=16384, eos_id=32021, lora_input_sizes=(4096, 11008))

FAKE_VLLM = '''
import os
from types import SimpleNamespace
__version__ = "0.27.1"
ENV_AT_IMPORT = dict(os.environ)
CALLS = []
NEXT = ([5, 6, 32021], "stop", 32021)


class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw


BACKEND = "FLASH_ATTN"
VLLM_CONFIG = SimpleNamespace(
    attention_config=SimpleNamespace(backend=SimpleNamespace(name=BACKEND), flash_attn_version=2),
    compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"),
                                       mode=SimpleNamespace(name="VLLM_COMPILE"), enabled_custom_ops={},
                                       disabled_custom_ops={"rms_norm": 65, "silu_and_mul": 32}),
    scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=1, max_num_batched_tokens=8192),
    cache_config=SimpleNamespace(enable_prefix_caching=False))
EAGER_TAKES_EFFECT = True     # vLLM 0.27.1 config/vllm.py L1193-1199: enforce_eager sets mode and cudagraph_mode NONE


class LLM:
    def __init__(self, **kw):
        CALLS.append(("LLM", kw))
        cc = VLLM_CONFIG.compilation_config
        if kw.get("enforce_eager") and EAGER_TAKES_EFFECT:     # mode NONE: custom_ops "all" (config/vllm.py L1285)
            cc = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"), mode=SimpleNamespace(name="NONE"),
                                 enabled_custom_ops={"rms_norm": 65, "silu_and_mul": 32}, disabled_custom_ops={})
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(**{**vars(VLLM_CONFIG), "compilation_config": cc}))

    def generate(self, prompts, sampling_params, use_tqdm=True, lora_request=None):
        CALLS.append(("generate", prompts, sampling_params.kw, lora_request, use_tqdm))
        ids, reason, stop = NEXT
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(ids), finish_reason=reason, stop_reason=stop)])]
'''
FAKE_REQUEST = '''
class LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.args = (lora_name, lora_int_id, lora_path)
'''
FAKE_PLATFORMS = '''
FAMILY = 80


class _Platform:
    def is_device_capability_family(self, family):
        return family == FAMILY


current_platform = _Platform()
'''
FAKE_BATCH_INVARIANT = '''
from types import SimpleNamespace
_batch_invariant_MODE = True
_batch_invariant_LIB = SimpleNamespace(_op_impls={"aten/mm/CUDA", "aten/addmm/CUDA", "aten/matmul/CUDA",
                                                  "aten/linear/CUDA", "aten/_log_softmax/CUDA", "aten/softmax/CUDA",
                                                  "aten/_softmax/CUDA", "aten/mean.dim/CUDA", "aten/bmm/CUDA"})
'''
FAKE_OP_UTILS = '''
SPLIT_K = 1
CALLS = []


def get_lora_op_configs(op_type, max_loras, batch, hidden_size, rank, num_slices, add_inputs=None,
                        moe_intermediate_size=None):
    CALLS.append((op_type, max_loras, batch, hidden_size, rank, num_slices))
    return {"block_m": 32, "block_n": 16, "block_k": 256 if batch < 128 else 32, "split_k": SPLIT_K, "num_warps": 4,
            "num_ctas": 1, "group_size_m": 8, "num_stages": 2, "max_nreg": None}
'''


@pytest.fixture
def vllm_env(monkeypatch):
    """The engine sets VLLM_ENV in os.environ; give every variable back its value after the test."""
    for k in VLLM_ENV:
        if k in os.environ:
            monkeypatch.setenv(k, os.environ[k])
        else:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch, vllm_env):
    """A stand-in `vllm` package (vLLM does not install on the laptop): records what the engine passes to it. The
    fixture returns a getter: the package is imported by the engine, not here."""
    pkg = tmp_path / "fake" / "vllm"
    (pkg / "lora" / "ops" / "triton_ops").mkdir(parents=True)
    (pkg / "model_executor" / "layers").mkdir(parents=True)
    (pkg / "platforms").mkdir()
    (pkg / "__init__.py").write_text(FAKE_VLLM)
    for d in ("lora", "lora/ops", "lora/ops/triton_ops", "model_executor", "model_executor/layers"):
        (pkg / d / "__init__.py").write_text("")
    (pkg / "lora" / "request.py").write_text(FAKE_REQUEST)
    (pkg / "lora" / "ops" / "triton_ops" / "utils.py").write_text(FAKE_OP_UTILS)
    (pkg / "platforms" / "__init__.py").write_text(FAKE_PLATFORMS)
    (pkg / "model_executor" / "layers" / "batch_invariant.py").write_text(FAKE_BATCH_INVARIANT)
    monkeypatch.syspath_prepend(str(tmp_path / "fake"))
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    yield lambda: sys.modules["vllm"]
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]


def test_the_engines_module_imports_without_vllm():
    out = subprocess.run([sys.executable, "-c", "import sys, v2gpu.engines, v2gpu.runs; print('vllm' in sys.modules)"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_vllm_kwargs_fix_every_setting_that_decides_the_numbers():
    kw = vllm_kwargs(CFG)
    assert kw["max_num_seqs"] == 1 and kw["enable_prefix_caching"] is False
    assert kw["quantization"] == "bitsandbytes" and kw["dtype"] == "bfloat16" and kw["kv_cache_dtype"] == "auto"
    assert kw["enable_lora"] is True and kw["max_lora_rank"] == 16 and kw["max_loras"] == 1
    assert kw["lora_dtype"] == "bfloat16" and kw["max_model_len"] == 16384 and kw["max_num_batched_tokens"] == 8192
    assert kw["skip_tokenizer_init"] is True and kw["generation_config"] == "vllm"
    assert kw["attention_config"] == {"backend": "FLASH_ATTN", "flash_attn_version": 2}
    assert kw["seed"] == 0 and kw["tensor_parallel_size"] == 1
    assert kw["revision"] == CFG.revision and kw["model"] == CFG.model
    assert "enable_chunked_prefill" not in kw and "cpu_offload_gb" not in kw
    assert kw["async_scheduling"] is False and kw["compilation_config"] == {"cudagraph_mode": "FULL_AND_PIECEWISE"}


def test_vllm_env_pins_the_lora_shrink_config_folder_and_the_scheduling_knobs():
    """The tuned shrink config folder (split_k 1) is still named; batch invariance is ON since the decision of
    05/10 (fallback (a), after the v0 cross-pod recheck mismatch): vLLM then ignores the tuned folder and sets split_k 1
    itself (lora/ops/triton_ops/utils.py L221-223 at v0.27.1), which the engine still checks after start. The V2 model
    runner, the LoRA dual stream and AOT compile are off. HF_HUB_OFFLINE is not an engine variable any more:
    generate.main sets it before any import."""
    assert VLLM_ENV["VLLM_TUNED_CONFIG_FOLDER"] == str(LORA_OPS) and Path(LORA_OPS).is_absolute() and LORA_OPS.is_dir()
    assert LORA_OPS == REPO / "config" / "lora_ops"
    assert BATCH_INVARIANT is True and VLLM_ENV["VLLM_BATCH_INVARIANT"] == "1"
    assert {k: VLLM_ENV[k] for k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER",
                                     "VLLM_LORA_ENABLE_DUAL_STREAM", "VLLM_USE_AOT_COMPILE")} == dict.fromkeys(
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_LORA_ENABLE_DUAL_STREAM",
         "VLLM_USE_AOT_COMPILE"), "0")
    assert "HF_HUB_OFFLINE" not in VLLM_ENV


def test_the_vllm_env_turns_the_flashinfer_sampler_off():
    """First generation pod, 04/10: vLLM 0.27.1 samples through FlashInfer by default (envs.py: unset means on), even
    in the dummy sampler run of the engine's memory profile, and FlashInfer JIT-builds that module with ninja. The venv's
    ninja is not on PATH, so every engine start failed (FileNotFoundError 'ninja'). "0" forces the PyTorch-native
    sampler: no JIT, no ninja or nvcc, and greedy decoding is unchanged."""
    assert VLLM_ENV["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_the_tuned_shrink_file_is_named_the_way_vllm_0_27_1_names_it():
    """vllm/lora/ops/triton_ops/utils.py load_lora_op_config: torch.cuda.get_device_name(), spaces and dashes to "_",
    then "_SHRINK.json"."""
    assert tuned_config_name("NVIDIA RTX A4000") == "NVIDIA_RTX_A4000_SHRINK.json"
    assert tuned_config_name("NVIDIA A100-SXM4-80GB") == "NVIDIA_A100_SXM4_80GB_SHRINK.json"
    assert (LORA_OPS / "NVIDIA_RTX_A4000_SHRINK.json").is_file() and tuned_config_problems("NVIDIA RTX A4000") == []
    assert any("NVIDIA_RTX_A5000_SHRINK.json" in p for p in tuned_config_problems("NVIDIA RTX A5000"))


def test_the_tuned_shrink_file_is_the_default_config_with_split_k_1():
    """config_data[max_loras][num_slices][m][k][n]: only num_slices ("1", "2", "3") is an exact-key lookup; max_loras
    (the kernel passes lora_ids.size(0)), m, k and n fall back to the nearest key (utils.py:284-305). Every leaf is
    v0.27.1's default shrink dict (block_k 256 below m = 128, 32 from 128) with split_k 1."""
    data = json.loads((LORA_OPS / "NVIDIA_RTX_A4000_SHRINK.json").read_text())
    assert data == shrink_table() and set(data) == {"1"} and set(data["1"]) == {"1", "2", "3"}
    ks = {str(k) for sizes in LORA_INPUT_SIZES.values() for k in sizes}
    assert ks == {"4096", "11008", "14336"}
    for by_m in data["1"].values():
        assert {"1", "127", "128", "8192"} <= set(by_m)
        for m, by_k in by_m.items():
            assert ks <= set(by_k)
            for by_n in by_k.values():
                assert set(by_n) == {"16"}
                leaf = by_n["16"]
                assert set(leaf) == SHRINK_DEFAULT_KEYS and leaf["split_k"] == 1
                assert leaf == {"block_m": 32, "block_n": 16, "block_k": 256 if int(m) < 128 else 32, "split_k": 1,
                                "num_warps": 4, "num_ctas": 1, "group_size_m": 8, "num_stages": 2, "max_nreg": None}


def test_lora_input_sizes_are_the_hidden_and_intermediate_sizes_of_the_pinned_models():
    from transformers import AutoConfig

    from v2gpu.models import MODELS

    for key, spec in MODELS.items():
        try:
            c = AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision)
        except OSError as e:
            pytest.skip(f"config of {spec.hf_id} not cached: {e}")
        assert LORA_INPUT_SIZES[key] == (c.hidden_size, c.intermediate_size), key


def test_sampling_is_greedy_on_token_ids_and_stops_only_at_the_tokenizer_eos():
    assert sampling_kwargs(300, 32021) == {"n": 1, "temperature": 0.0, "max_tokens": 300, "stop_token_ids": [32021],
                                           "ignore_eos": False, "detokenize": False}
    probe = sampling_kwargs(300, 32021, ignore_eos=True)
    assert probe["ignore_eos"] is True and probe["stop_token_ids"] == []


def test_vllm_engine_passes_token_ids_and_the_adapter(fake_vllm, tmp_path):
    engine = VLLMEngine(CFG, adapter=tmp_path / "adapter")
    vllm = fake_vllm()
    assert vllm.ENV_AT_IMPORT["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"        # set before vllm is imported
    assert vllm.CALLS[0] == ("LLM", vllm_kwargs(CFG))
    ids, reason = engine.generate([32013, 10, 11], 300)
    assert (ids, reason) == ([5, 6, 32021], "stop")
    _, prompts, sp, lora, tqdm = vllm.CALLS[1]
    assert prompts == [{"prompt_token_ids": [32013, 10, 11]}] and sp == sampling_kwargs(300, 32021)
    assert lora.args == ("adapter", 1, str(tmp_path / "adapter")) and tqdm is False
    d = engine.describe()
    assert d["name"] == "vllm" and d["version"] == "0.27.1" and d["kwargs"]["max_num_seqs"] == 1
    assert d["quantization"].startswith("bitsandbytes") and d["adapter"] == str(tmp_path / "adapter")
    assert d["batch_invariant"] is True and d["effective"]["attention_backend"] == "FLASH_ATTN"
    assert d["effective"]["cudagraph_mode"] == "NONE" and d["effective"]["compilation_mode"] == "NONE"
    assert len(d["effective"]["vllm_config_sha256"]) == 64
    assert len(d["lora_shrink"]["checked"]) == 3 * 4 * 2 and len(d["lora_shrink"]["sha256"]) == 64
    assert {c["config"]["split_k"] for c in d["lora_shrink"]["checked"]} == {1}
    assert {(c["num_slices"], c["m"], c["k"]) for c in d["lora_shrink"]["checked"]} == {
        (s, m, k) for s in (1, 2, 3) for m in (1, 127, 128, 8192) for k in (4096, 11008)}


def test_an_effective_shrink_config_with_split_k_is_refused_before_any_generation(fake_vllm, monkeypatch):
    """split-K reduces fp32 partial sums with atomics in completion order (non-deterministic). A card-name
    mismatch or a missing tuned file falls back to split_k 64 / 8 silently; the engine refuses to start instead."""
    import importlib

    utils = importlib.import_module("vllm.lora.ops.triton_ops.utils")
    monkeypatch.setattr(utils, "SPLIT_K", 64)
    with pytest.raises(RuntimeError, match="split_k"):
        VLLMEngine(CFG, adapter=None)
    assert [c[0] for c in fake_vllm().CALLS] == ["LLM"]                      # started, then refused: no generate


def test_an_effective_attention_backend_other_than_the_pinned_one_is_refused(fake_vllm, monkeypatch):
    """LLM._make_config drops unknown dict keys silently, so the effective config is read back."""
    import importlib

    vllm = importlib.import_module("vllm")
    monkeypatch.setattr(vllm.VLLM_CONFIG.attention_config, "backend", type("B", (), {"name": "TRITON_ATTN"})())
    with pytest.raises(RuntimeError, match="FLASH_ATTN"):
        VLLMEngine(CFG, adapter=None)


def test_the_engine_fingerprint_ignores_the_model_and_not_the_batch_invariance(fake_vllm):
    a = VLLMEngine(CFG, adapter=None).describe()
    mistral = VLLMCfg(model="mistralai/Mistral-7B-Instruct-v0.2", revision="r", max_model_len=32768, eos_id=2,
                      lora_input_sizes=(4096, 14336))
    b = VLLMEngine(mistral, adapter="x").describe()
    assert engine_fingerprint(a) == engine_fingerprint(b)
    assert engine_fingerprint(a) != engine_fingerprint({**a, "batch_invariant": False})
    assert engine_fingerprint(a) != engine_fingerprint({**a, "effective": {**a["effective"],
                                                                           "cudagraph_mode": "FULL_AND_PIECEWISE"}})
    assert engine_fingerprint(a) != engine_fingerprint({**a, "effective": {**a["effective"],
                                                                           "compilation_mode": "VLLM_COMPILE"}})


def test_the_engine_fingerprint_holds_the_tuned_file_digests_not_the_checkout_path(fake_vllm):
    """Pods may check the repo out at different paths, so the fingerprint drops the folder path
    (report-only in describe()) and holds the sha256 of every file of the tuned config folder instead."""
    import hashlib

    from v2gpu.engines import tuned_config_files

    d = VLLMEngine(CFG, adapter=None).describe()
    name = "NVIDIA_RTX_A4000_SHRINK.json"
    names = (name, "NVIDIA_RTX_2000_Ada_Generation_SHRINK.json")                     # both classes
    assert tuned_config_files() == {n: hashlib.sha256((LORA_OPS / n).read_bytes()).hexdigest() for n in names}
    assert d["tuned_config"] == {"folder": str(LORA_OPS), "files": tuned_config_files()}
    moved = {**d, "env": {**d["env"], "VLLM_TUNED_CONFIG_FOLDER": "/root/other/config/lora_ops"},
             "tuned_config": {**d["tuned_config"], "folder": "/root/other/config/lora_ops"}}
    assert engine_fingerprint(moved) == engine_fingerprint(d)
    edited = {**d, "tuned_config": {**d["tuned_config"], "files": {name: "0" * 64}}}
    assert engine_fingerprint(edited) != engine_fingerprint(d)


# An engine record that names no engine has no fingerprint

EMPTY_FINGERPRINT = {"name": None, "version": None, "quantization": None, "batch_invariant": None, "env": {},
                     "tuned_config_files": None, "sampling": {}, "kwargs": {}, "effective": {}, "lora_shrink": []}


@pytest.mark.parametrize("record", [{}, {"name": "vllm"}, {"version": "0.27.1"}, {"name": "hf", "version": "4.57.1"},
                                    {"name": "vllm", "version": ""}, None])
def test_an_engine_record_that_names_no_engine_has_no_fingerprint(record):
    """engine_fingerprint({}) was a dict of None values (EMPTY_FINGERPRINT), which every non-null check accepted;
    now a record without name "vllm" and a version has no fingerprint (None), and names_engine says why."""
    from v2gpu.engines import names_engine

    assert engine_fingerprint(record or {}) is None and not names_engine(record)
    assert not names_engine(EMPTY_FINGERPRINT)


def test_a_real_engine_record_keeps_its_fingerprint(fake_vllm):
    from v2gpu.engines import names_engine

    d = VLLMEngine(CFG, adapter=None).describe()
    assert names_engine(d) and names_engine(engine_fingerprint(d)) and engine_fingerprint(d)["version"] == d["version"]


def test_the_driver_version_is_none_without_nvidia_smi(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert nvidia_driver() is None


def test_every_receipt_names_its_physical_gpu_by_uuid(monkeypatch):
    """Since 05/10 the 2-GPU gate needs two DIFFERENT physical GPUs, so the engine's gpu record (in every
    receipt) holds the GPU's UUID: torch's device properties first, nvidia-smi as the fallback, an error string (never a
    crash) when neither answers. Both read as nvidia-smi prints it, "GPU-..."."""
    import shutil
    from types import SimpleNamespace

    import torch

    from v2gpu import engines

    uuid = "5b9fe33a-5fc3-2096-fe4d-23b9b664024c"
    props = {"name": "NVIDIA RTX A4000", "major": 8, "minor": 6, "total_memory": 16_760_000_000}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engines, "nvidia_driver", lambda: "580.119.02")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(**props, uuid=uuid))
    gpu = engines._gpu()
    assert gpu["uuid"] == f"GPU-{uuid}" and gpu["uuid_source"] == "torch" and gpu["capability"] == "8.6"

    calls = []

    def smi(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="GPU-11111111-2222-3333-4444-555555555555\n")

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(**props))      # no uuid
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(engines.subprocess, "run", smi)
    gpu = engines._gpu()
    assert gpu["uuid"] == "GPU-11111111-2222-3333-4444-555555555555" and gpu["uuid_source"] == "nvidia-smi"
    assert calls == [["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"]]

    monkeypatch.setattr(engines.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no smi")))
    gpu = engines._gpu()
    assert gpu["uuid"].startswith("error: ") and "no smi" in gpu["uuid"] and gpu["uuid_source"] is None


def test_base_runs_use_the_same_engine_settings_without_a_lora_request(fake_vllm):
    engine = VLLMEngine(CFG, adapter=None)
    engine.generate([1, 2], 10)
    assert fake_vllm().CALLS[0] == ("LLM", vllm_kwargs(CFG)) and fake_vllm().CALLS[1][3] is None


def test_without_vllm_the_engine_names_the_generation_venv(monkeypatch, vllm_env):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(RuntimeError, match="requirements-gen"):
        VLLMEngine(CFG)


def out(ids, reason, stop=None):
    from types import SimpleNamespace
    return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason=reason, stop_reason=stop)])]


def test_unpack_accepts_only_eos_stops_and_length_caps():
    assert unpack(out([4, 2], "stop", 2), 10, 2) == ([4, 2], "stop")
    assert unpack(out([4, 4, 4], "length"), 3, 2) == ([4, 4, 4], "length")
    assert unpack(out([4, 2], "stop", 2), 2, 2) == ([4, 2], "stop")        # EOS as the last allowed token: a stop
    with pytest.raises(RuntimeError, match="abort"):
        unpack(out([4], "abort"), 10, 2)
    with pytest.raises(RuntimeError, match="EOS"):
        unpack(out([4, 5], "stop", 5), 10, 2)                      # stopped on something that is not the EOS
    with pytest.raises(RuntimeError, match="max_new_tokens"):
        unpack(out([4, 4, 4, 4], "length"), 3, 2)
    with pytest.raises(RuntimeError, match="one completion"):
        unpack(out([4], "stop") * 2, 3, 2)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_llama(tmp_path_factory.mktemp("tiny") / "m", 128)


def test_hf_engine_is_greedy_and_capped(tiny):
    model = load_hf(tiny, None, "none", "float32")
    engine = HFEngine(model, eos_id=127)
    a, b = engine.generate([1, 5, 9], 6), engine.generate([1, 5, 9], 6)
    assert a == b and len(a[0]) <= 6


def test_hf_engine_stops_at_eos_and_can_ignore_it(tiny):
    model = load_hf(tiny, None, "none", "float32")
    first = HFEngine(model, eos_id=127).generate([1, 5, 9], 1)[0][0]
    stopped = HFEngine(model, eos_id=first)
    assert stopped.generate([1, 5, 9], 6) == ([first], "stop")
    ids, reason = stopped.generate([1, 5, 9], 6, ignore_eos=True)
    assert len(ids) == 6 and reason == "length"
    assert HFEngine(model, eos_id=first).describe()["name"] == "hf"


def test_teacher_forced_agreement_of_a_model_with_its_own_greedy_output_is_total(tiny):
    """hf_agreement is a gate: the HF side scores another engine's output token by token; on its
    own greedy output every position agrees, and a changed token is the first divergence."""
    model = load_hf(tiny, None, "none", "float32")
    prompt = [1, 5, 9, 11]
    new, _ = HFEngine(model, eos_id=127).generate(prompt, 12, ignore_eos=True)
    r = teacher_forced_agreement(model, prompt, new, chunk=5)
    assert r == {"positions": 12, "agree": 12, "first_divergence": None}
    bad = list(new)
    bad[7] = (bad[7] + 1) % 128
    r = teacher_forced_agreement(model, prompt, bad, chunk=5)
    assert r["first_divergence"] == 7 and r["agree"] < 12 and r["positions"] == 12


def save_adapter(tiny, out):
    model, _ = load_for_training(LoadCfg(tiny, None, quant="none", attn="sdpa", dtype="float32",
                                         gradient_checkpointing=False), seed=42)
    model.save_pretrained(out)
    return tensor_digest(out / "adapter_model.safetensors")


def test_hf_engine_loads_a_peft_adapter(tiny, tmp_path):
    save_adapter(tiny, tmp_path / "ad")
    model = load_hf(tiny, None, "none", "float32", adapter=tmp_path / "ad")
    assert type(model).__name__ == "PeftModelForCausalLM"


def test_adapter_checks_digest_and_what_vllm_can_serve(tiny, tmp_path):
    digest = save_adapter(tiny, tmp_path / "ad")
    assert adapter_problems(tmp_path / "ad", digest) == []
    assert any("tensor digest" in p for p in adapter_problems(tmp_path / "ad", "0" * 64))
    cfg_path = tmp_path / "ad" / "adapter_config.json"
    good = json.loads(cfg_path.read_text())
    for change, words in (({"r": 32}, "max_lora_rank"), ({"use_dora": True}, "DoRA"), ({"bias": "all"}, "bias"),
                          ({"modules_to_save": ["lm_head"]}, "modules_to_save"), ({"use_rslora": True}, "rsLoRA"),
                          ({"target_modules": "all-linear"}, "target_modules"),
                          ({"target_modules": ["model.embed_tokens", "model.layers.0.mlp.up_proj"]}, "embed_tokens"),
                          ({"lora_alpha": 16}, "lora_alpha")):
        cfg_path.write_text(json.dumps({**good, **change}))
        assert any(words in p for p in adapter_problems(tmp_path / "ad", digest)), change
    assert any("missing" in p for p in adapter_problems(tmp_path / "nothing", digest))


def test_memory_steps_are_the_two_pre_declared_settings_and_nothing_else():
    """decision 24/09: gpu_memory_utilization 0.95 and enforce_eager are the only pre-declared memory steps, chosen
    at the canary and recorded as engine settings. The prefill chunk (max_num_batched_tokens) is not a setting at all:
    4,096 needs a new declaration, i.e. a change of PREFILL_CHUNK in the code."""
    import dataclasses

    from v2gpu.engines import GPU_MEMORY_UTILIZATION, PREFILL_CHUNK

    assert GPU_MEMORY_UTILIZATION == (0.92, 0.95) and PREFILL_CHUNK == 8192
    kw = vllm_kwargs(dataclasses.replace(CFG, gpu_memory_utilization=0.92, enforce_eager=False))
    assert kw["gpu_memory_utilization"] == 0.92 and kw["enforce_eager"] is False and kw["max_num_batched_tokens"] == 8192
    assert vllm_kwargs(CFG)["gpu_memory_utilization"] == 0.95 and vllm_kwargs(CFG)["enforce_eager"] is True
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        dataclasses.replace(CFG, gpu_memory_utilization=0.9)
    assert "max_num_batched_tokens" not in {f.name for f in dataclasses.fields(VLLMCfg)}


def test_decision_0510_every_job_runs_at_the_eager_step_of_the_ladder():
    """decision 05/10 (the v0 cross-pod recheck mismatch, evidence-xpod-v0-mismatch/): fallback (a) plus the pre-declared
    memory step (0.95, enforce_eager) for EVERY job. Batch invariance alone leaves RMSNorm on Inductor kernels whose
    block size is autotuned per GPU (vLLM 0.27.1 sets custom_ops "none" under inductor, so RMSNorm runs forward_native);
    eager turns torch.compile off, custom_ops become "all" and RMSNorm runs forward_cuda (rms_norm_batch_invariant).
    The step is the top of the ladder and the default of every engine config."""
    import dataclasses

    from v2gpu.engines import DETERMINISM_STEP, MEMORY_LADDER

    assert DETERMINISM_STEP == MEMORY_LADDER[-1] == (0.95, True)
    defaults = {f.name: f.default for f in dataclasses.fields(VLLMCfg)}
    assert (defaults["gpu_memory_utilization"], defaults["enforce_eager"]) == DETERMINISM_STEP


@pytest.mark.parametrize("step", [(0.92, False), (0.95, False)])
def test_the_engine_refuses_any_other_memory_step_before_vllm_starts(fake_vllm, step):
    """decision 05/10: no output can be made at another step while batch invariance is on: the vLLM engine refuses it
    before vllm.LLM is built (the laptop tools keep the pre-declared ladder; a real verdict can only carry this step)."""
    import dataclasses

    cfg = dataclasses.replace(CFG, gpu_memory_utilization=step[0], enforce_eager=step[1])
    with pytest.raises(RuntimeError, match="decision 05/10"):
        VLLMEngine(cfg)
    assert "vllm" not in sys.modules                       # refused before vLLM is even imported
    VLLMEngine(CFG)
    assert [c[1]["enforce_eager"] for c in fake_vllm().CALLS if c[0] == "LLM"] == [True]


def test_decision_0510_the_receipt_says_what_batch_invariance_installed_on_this_card(fake_vllm, monkeypatch):
    """decision 05/10 (read in the vLLM code): fallback (a) installs its aten matmul overrides only where vLLM's
    is_device_capability_family(80) holds (batch_invariant.py L915), and RMSNorm takes forward_cuda only as an enabled
    custom op (no residual: rms_norm_batch_invariant; with a residual: ops.fused_add_rms_norm). The engine reads back,
    after start and in its own process, the card's capability, that test, the batch-invariant mode, the aten ops it
    overrides, and which custom ops vLLM enabled; the receipt carries them. The 2-GPU byte gate decides; these explain."""
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    d = VLLMEngine(CFG).describe()
    assert d["batch_invariance"] == {
        "device_capability": [8, 6], "capability_family_80": True, "batch_invariant_mode": True,
        "aten_overrides": ["aten/_log_softmax/CUDA", "aten/_softmax/CUDA", "aten/addmm/CUDA", "aten/bmm/CUDA",
                           "aten/linear/CUDA", "aten/matmul/CUDA", "aten/mean.dim/CUDA", "aten/mm/CUDA",
                           "aten/softmax/CUDA"]}
    assert d["effective"]["custom_ops_enabled"] == ["rms_norm", "silu_and_mul"]
    assert d["effective"]["custom_ops_disabled"] == []


def test_a_batch_invariance_fact_that_cannot_be_read_is_recorded_not_fatal(fake_vllm, monkeypatch):
    """No CUDA on this laptop and a vLLM without the module: each unreadable fact names its error; the engine runs."""
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (_ for _ in ()).throw(RuntimeError("no CUDA")))
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    probe = VLLMEngine(CFG).describe()["batch_invariance"]
    assert probe["device_capability"] == {"error": "RuntimeError: no CUDA"}
    assert probe["capability_family_80"]["error"].startswith("ModuleNotFoundError: import of vllm.platforms")
    assert probe["batch_invariant_mode"] is True


def test_an_engine_that_still_compiles_after_start_is_refused(fake_vllm, monkeypatch):
    """decision 05/10: eager is checked where it acts. If vLLM kept torch.compile or CUDA graphs on despite
    enforce_eager (read back from the effective config), Inductor kernels could run: the engine refuses to start."""
    import importlib

    monkeypatch.setattr(importlib.import_module("vllm"), "EAGER_TAKES_EFFECT", False)
    with pytest.raises(RuntimeError, match="compilation_mode"):
        VLLMEngine(CFG)
    assert [c[0] for c in fake_vllm().CALLS] == ["LLM"]                      # started, then refused: no generate


def test_pin_sets_are_complete_and_differ_only_in_the_declared_fallbacks():
    """decision 24/09: transformers 5.15.0 and bitsandbytes 0.49.2 are pre-declared alternatives. transformers 5.15.0
    requires tokenizers <= 0.23.0 (0.23.0 itself was never released), so its sets pin tokenizers 0.22.2."""
    from v2gpu.engines import PIN_SETS

    assert PIN_SETS["main"] == GEN_PINS
    assert all(set(pins) == set(GEN_PINS) for pins in PIN_SETS.values())
    changed = {name: {k: v for k, v in pins.items() if GEN_PINS[k] != v} for name, pins in PIN_SETS.items()}
    assert changed == {"main": {}, "fallback-tf": {"transformers": "5.15.0", "tokenizers": "0.22.2"},
                       "fallback-bnb": {"bitsandbytes": "0.49.2"},
                       "fallback-tf-bnb": {"transformers": "5.15.0", "tokenizers": "0.22.2", "bitsandbytes": "0.49.2"}}


def test_pin_problems_check_the_named_set():
    from v2gpu.engines import PIN_SETS, tokenizes_like_training

    assert pin_problems(dict(PIN_SETS["fallback-tf"]), "fallback-tf") == []
    assert any("transformers" in p for p in pin_problems(dict(PIN_SETS["fallback-tf"])))       # main by default
    assert any("bitsandbytes" in p for p in pin_problems(dict(GEN_PINS), "fallback-bnb"))
    assert pin_problems(dict(GEN_PINS), "custom") == ["pin set 'custom' is not one of ['fallback-bnb', 'fallback-tf', "
                                                      "'fallback-tf-bnb', 'main']"]
    assert [n for n in sorted(PIN_SETS) if tokenizes_like_training(n)] == ["fallback-bnb", "main"]


def test_pins_must_match_the_generation_lock():
    assert pin_problems(dict(GEN_PINS)) == []
    assert GEN_PINS["vllm"] == "0.27.1" and GEN_PINS["torch"] == "2.13.0" and GEN_PINS["bitsandbytes"] == "0.50.2"
    problems = pin_problems({**GEN_PINS, "torch": "2.14.0", "vllm": None})
    assert len(problems) == 2 and any("torch" in p for p in problems) and any("vllm" in p for p in problems)


def test_device_must_be_the_one_16gb_sm86_card_class():
    a4000 = {"name": "NVIDIA RTX A4000", "capability": "8.6", "total_memory_gb": 16.86}
    assert device_problems(a4000, "NVIDIA RTX A4000") == []
    assert device_problems(None, "NVIDIA RTX A4000") == ["no CUDA device"]
    assert any("capability" in p for p in device_problems({**a4000, "capability": "7.5"}, "NVIDIA RTX A4000"))
    assert any("name" in p for p in device_problems({**a4000, "name": "NVIDIA A16"}, "NVIDIA RTX A4000"))
    assert any("memory" in p for p in device_problems({**a4000, "total_memory_gb": 24.0}, "NVIDIA RTX A4000"))


def test_engines_module_documents_its_pins():
    doc = textwrap.dedent(__import__("v2gpu.engines", fromlist=["x"]).__doc__)
    assert "0.27.1" in doc and "requirements-gen" in doc


# One decision per variant (added 24/09); the common target of a mismatch

D = {"gpu_name": "NVIDIA RTX A4000", "pin_set": "main", "gpu_memory_utilization": 0.92, "enforce_eager": False}
M95, EAGER = {"gpu_memory_utilization": 0.95}, {"gpu_memory_utilization": 0.95, "enforce_eager": True}


def test_the_memory_ladder_and_the_pin_set_parts_are_the_two_orders_in_one_place():
    """0.92 with CUDA graphs < 0.95 < 0.95 + enforce_eager; main < one fallback < fallback-tf-bnb. The prefill chunk
    (max_num_batched_tokens 4,096) is not a step of the ladder: the tool never chooses it."""
    from v2gpu.engines import GPU_MEMORY_UTILIZATION, MEMORY_LADDER, PIN_SET_PARTS, PIN_SETS

    assert MEMORY_LADDER == ((0.92, False), (0.95, False), (0.95, True))
    assert {m for m, _ in MEMORY_LADDER} == set(GPU_MEMORY_UTILIZATION)
    assert set(PIN_SET_PARTS) == set(PIN_SETS) and PIN_SET_PARTS["main"] == frozenset()
    assert PIN_SET_PARTS["fallback-tf-bnb"] == PIN_SET_PARTS["fallback-tf"] | PIN_SET_PARTS["fallback-bnb"]


@pytest.mark.parametrize("decisions, target, rerun", [
    ({"codellama": D, "deepseek": D, "mistral": D}, D, []),
    ({"codellama": {**D, **M95}, "deepseek": D, "mistral": D}, {**D, **M95}, ["deepseek", "mistral"]),
    ({"codellama": {**D, **M95}, "deepseek": {**D, **EAGER}, "mistral": D}, {**D, **EAGER}, ["codellama", "mistral"]),
    ({"codellama": {**D, "pin_set": "fallback-tf"}, "deepseek": {**D, **M95}, "mistral": D},
     {**D, **M95, "pin_set": "fallback-tf"}, ["codellama", "deepseek", "mistral"]),
    ({"codellama": {**D, "pin_set": "fallback-tf"}, "deepseek": {**D, "pin_set": "fallback-bnb"}, "mistral": D},
     {**D, "pin_set": "fallback-tf-bnb"}, ["codellama", "deepseek", "mistral"]),
    ({"codellama": {**D, "pin_set": "fallback-tf-bnb"}, "deepseek": {**D, "pin_set": "fallback-bnb"},
      "mistral": {**D, "pin_set": "fallback-tf-bnb"}}, {**D, "pin_set": "fallback-tf-bnb"}, ["deepseek"]),
])
def test_the_common_target_is_the_most_conservative_decision_that_covers_every_model(decisions, target, rerun):
    from v2gpu.engines import conservative_decision

    assert conservative_decision(decisions) == (target, rerun, [])


@pytest.mark.parametrize("odd, words", [({"gpu_name": "NVIDIA A16"}, "gpu_name"),
                                        ({"enforce_eager": True}, "memory step"),
                                        ({"gpu_memory_utilization": 0.9}, "memory step"),
                                        ({"enforce_eager": 0}, "memory step"),
                                        ({"pin_set": "custom"}, "pin set"),
                                        ({"extra": 1}, "keys")])
def test_a_decision_off_the_ladder_or_of_another_card_has_no_target(odd, words):
    from v2gpu.engines import conservative_decision

    target, rerun, problems = conservative_decision({"codellama": {**D, **odd}, "deepseek": D})
    assert target is None and rerun == [] and any(words in p for p in problems), problems


# decision 24/09: the generation card class is one of two 16 GB classes

ADA = {"name": "NVIDIA RTX 2000 Ada Generation", "capability": "8.9", "total_memory_gb": 16.7}


def test_the_generation_cards_are_an_allow_list_of_16gb_classes():
    """A4000 (sm86, preferred) or RTX 2000 Ada 16 GB (sm89, pre-declared fallback); nothing else, not the RTX 4000 Ada
    20 GB; a listed name with another capability is refused; the memory window stays 16 GB."""
    from v2gpu.engines import CARD, CARD_MEMORY_GB

    assert CARD == {"NVIDIA RTX A4000": "8.6", "NVIDIA RTX 2000 Ada Generation": "8.9"}
    assert CARD_MEMORY_GB == (15.0, 17.5)
    a4000 = {"name": "NVIDIA RTX A4000", "capability": "8.6", "total_memory_gb": 16.86}
    assert device_problems(ADA, ADA["name"]) == [] and device_problems(a4000, a4000["name"]) == []
    assert any("capability" in p for p in device_problems({**ADA, "capability": "8.6"}, ADA["name"]))
    assert any("capability" in p for p in device_problems({**a4000, "capability": "8.9"}, a4000["name"]))
    ada20 = {"name": "NVIDIA RTX 4000 Ada Generation", "capability": "8.9", "total_memory_gb": 21.2}
    assert any("card class" in p for p in device_problems(ada20, ada20["name"]))
    assert any("card class" in p for p in device_problems({**ada20, "total_memory_gb": 16.7}, ada20["name"]))
    assert any("memory" in p for p in device_problems({**ADA, "total_memory_gb": 20.0}, ADA["name"]))


def test_every_generation_card_has_its_tuned_shrink_file_named_as_vllm_names_it():
    """torch.cuda.get_device_name() with spaces and dashes turned into "_" (vLLM 0.27.1 load_lora_op_config); the RTX
    2000 Ada file is the same default-with-split_k-1 table as the A4000 file, unmeasured until its generation canary."""
    from v2gpu.engines import CARD

    assert tuned_config_name("NVIDIA RTX 2000 Ada Generation") == "NVIDIA_RTX_2000_Ada_Generation_SHRINK.json"
    assert tuned_config_name("NVIDIA RTX 2000-Ada") == "NVIDIA_RTX_2000_Ada_SHRINK.json"
    for name in CARD:
        assert tuned_config_problems(name) == [], name
        assert json.loads((LORA_OPS / tuned_config_name(name)).read_text()) == shrink_table(), name


def test_a_decision_names_a_generation_card_class():
    from v2gpu.engines import decision_problems

    assert decision_problems({**D, "gpu_name": ADA["name"]}) == []
    assert any("card class" in p for p in decision_problems({**D, "gpu_name": "NVIDIA A100-SXM4-80GB"}))


# The declared engine a canary binds, the same for every model of a variant


def test_the_declared_engine_leaves_out_only_what_names_the_model():
    from v2gpu.engines import declared_engine

    settings = {**vllm_kwargs(CFG), "batch_invariant": False, "tuned_config_files": {"f": "1"}, "env": {"E": "0"}}
    settings.pop("model")
    other = {**settings, "revision": "other", "max_model_len": 32768}
    assert declared_engine(settings) == declared_engine(other) and "revision" not in declared_engine(settings)
    assert declared_engine({**settings, "batch_invariant": True}) != declared_engine(settings)
    assert declared_engine({**settings, "gpu_memory_utilization": 0.92}) != declared_engine(settings)
