"""Greedy generation engines behind one interface: generate(prompt_ids, max_new_tokens, ignore_eos=False) ->
(new_ids, finish_reason), finish_reason "stop" (the tokenizer's EOS, kept as the last id) or "length".

- VLLMEngine (on the generation pods): vLLM 0.27.1, the last release with bitsandbytes in tree (0.28.0 moved it to the
  vllm-bnb-plugin). The base is quantised to NF4 while loading (vLLM calls bitsandbytes quantize_4bit(nf4,
  compress_statistics=True), blocksize 64: the NF4 + double quant of the training load, from the same bf16 checkpoint)
  and the LoRA adapter is applied at runtime (enable_lora; one adapter per engine, rank 16). Greedy (temperature 0),
  max_num_seqs = 1, no prefix caching (a cached prefix would make an output depend on the prompts run before it), one
  fixed chunked-prefill size (PREFILL_CHUNK: max_num_batched_tokens 8,192, not a setting; it is the prefill chunk, not
  max_new_tokens, but it changes prefill numerics, so another value such as 4,096 must be declared before any
  output), the FLASH_ATTN backend pinned (no automatic choice that could differ between pods), max_model_len = the
  model window, prompt token ids from serialize.prompt_ids (skip_tokenizer_init: vLLM never tokenises or detokenises;
  text comes from serialize.decode_answer). Base conditions use the same LoRA-enabled engine settings with no
  LoRARequest, so the five conditions differ only in weights and prompt.
  Memory steps (decision 24/09): gpu_memory_utilization 0.92, or the pre-declared 0.95, and enforce_eager off, or the
  pre-declared on; chosen ONLY at the canary, then fixed for the whole run (the job command carries them, the job
  identity and the engine fingerprint record them, gencoverage refuses workdirs that differ).
  VLLM_ENABLE_V1_MULTIPROCESSING=0 (the engine core runs in-process: deterministic scheduling, one process group for
  the queue to stop, and the effective configs can be read back after start).
  LoRA shrink determinism (route (b), chosen 24/09): with batch invariance off, vLLM
  0.27.1 picks split_k = 64 (batch < 128) or 8 for the LoRA shrink kernel (lora/ops/triton_ops/utils.py:219-223), and
  the split-K partial sums are reduced with tl.atomic_add(..., sem="relaxed") into a float32 buffer
  (kernel_utils.py:337-340, punica_gpu.py:243-247): completion order, not deterministic. Route (b) keeps
  VLLM_BATCH_INVARIANT=0 and points VLLM_TUNED_CONFIG_FOLDER at config/lora_ops, whose <card>_SHRINK.json is the
  v0.27.1 default shrink config with split_k = 1 everywhere (the expand kernel has no split-K). After start the engine
  reads the EFFECTIVE shrink config back through vllm.lora.ops.triton_ops.utils.get_lora_op_configs and refuses to run
  unless every checked (num_slices, m, k) has split_k == 1 (this also catches a card name without a tuned file, which
  falls back to the split-K default silently). Route (a), the pre-declared fallback used only if the canary
  determinism gate fails under (b), and only after a new declaration: BATCH_INVARIANT = True (VLLM_BATCH_INVARIANT=1)
  for EVERY job, base and adapter alike, then a new canary; the receipt says batch_invariant true and coverage refuses
  to mix the two.
  Route (a) IS ON since decision 05/10, with DETERMINISM_STEP = the pre-declared memory step (0.95, enforce_eager) for
  every job. Why: the v0 cross-pod recheck on a second A4000 differed from v0 in 9 of 13 pairs while each pod
  reproduced itself byte for byte (evidence kept with the run records). Read in
  the v0.27.1 sources: under the inductor backend vLLM sets custom_ops "none" (config/vllm.py L1285-1292), so RMSNorm
  runs forward_native and Inductor generates its reductions with a block size autotuned on the GPU; batch invariance
  replaces RMSNorm only in forward_cuda (layernorm.py L96-113). enforce_eager sets the compilation mode and the
  cudagraph mode to NONE (config/vllm.py L1193-1199), custom_ops become "all" and RMSNorm runs forward_cuda:
  rms_norm_batch_invariant (Triton, BLOCK_SIZE 1024) without a residual, ops.fused_add_rms_norm (CUDA) with one
  (batch_invariant.py L848-855). With batch invariance on, vLLM ignores the tuned folder and sets the shrink split_k to
  1 itself (lora/ops/triton_ops/utils.py L221-223), which the engine still checks. The engine refuses any other step
  before vLLM starts, and refuses to run when the effective compilation or cudagraph mode is not NONE. The receipt
  records what (a) installed on the card (batch_invariance_probe: capability, the sm80-family test that decides the
  aten matmul overrides, the overridden ops) and the custom ops vLLM enabled; whether bnb's gemm_4bit and the LoRA
  kernels are deterministic across GPUs is not knowable from the sources: the gate is byte equality on two GPUs, whose
  UUIDs every receipt records (gpu.uuid). Pre-declaration supplement, decision 05/10 (ledger 8d7ab5a and
  e0f346b): (a) was declared for a failed SAME-pod determinism gate and (0.95, enforce_eager) as a MEMORY step; both
  are now used together for determinism ACROSS GPUs, because of the RECHECK_MISMATCH of 05/10 and its diagnosis,
  before any new output.
  Pinned knobs (names checked in the v0.27.1 sources): async_scheduling False (config/scheduler.py),
  VLLM_USE_V2_MODEL_RUNNER=0, VLLM_LORA_ENABLE_DUAL_STREAM=0, VLLM_USE_AOT_COMPILE=0 (envs.py),
  VLLM_USE_FLASHINFER_SAMPLER=0 (pinned 05/10 after the first generation pod, 04/10: on by default, FlashInfer JIT-builds its sampling
  module with ninja during the engine's memory profile, and the venv's ninja is not on PATH; "0" forces the
  PyTorch-native sampler, no JIT, greedy unchanged), flash_attn_version 2
  (config/attention.py; FA2 is the sm86 and sm89 default), cudagraph_mode FULL_AND_PIECEWISE (config/compilation.py,
  the vLLM default, written explicitly). The effective attention backend, flash-attn version, cudagraph mode, async
  scheduling, prefix caching and max_num_seqs are read back from llm.llm_engine.vllm_config after start and checked (an
  explicitly selected backend that is invalid already raises in platforms/cuda.py:407-419; LLM._make_config drops
  unknown dict keys silently, so a typo would otherwise pass). HF_HUB_OFFLINE is set by generate.main before any
  import. vLLM is imported lazily, inside the constructor: this module imports on a laptop without vLLM.
- HFEngine: transformers generate(do_sample=False), batch 1. The CPU tests run it on a tiny Llama; on the pod it is the
  canary's reference: the training loader (NF4) and peft, in the generation venv (torch 2.13.0, not training's
  2.14.0). `teacher_forced_agreement` scores another engine's output under this model: one forward over prompt +
  output, logits only at the output positions and in chunks (fits 16 GB at 6,144 new tokens plus the prompt).

vLLM pins torch 2.13.0 while training uses torch 2.14.0, so generation has its own venv: requirements-gen.in, with
GEN_PINS (transformers 5.17.0 and tokenizers 0.23.2 as in training, so prompt ids are the training ids). PIN_SETS adds
the pre-declared fallbacks (decision 24/09), each a complete set with its own requirements-gen-<name>.in, chosen ONLY at
the canary and recorded in the job identity: transformers 5.15.0 (which needs tokenizers <= 0.23.0, so tokenizers
0.22.2; a job in such a set needs the laptop's ids_sha256 in its command and refuses other prompt ids) and
bitsandbytes 0.49.2.

One decision per variant (decision 24/09): every model's canary decides for itself, and
the variant takes one decision for all of them (gencanary decide). When the models' decisions differ, the common
target is the most conservative decision that covers every model's own, from the two orders encoded here and nowhere
else: MEMORY_LADDER (0.92 with CUDA graphs < 0.95 < 0.95 + enforce_eager; the highest step any model needed) and
PIN_SET_PARTS (main < one fallback < fallback-tf-bnb: the union of the fallbacks the models needed). The prefill chunk
(max_num_batched_tokens 4,096) is not a step of the ladder and is never chosen by a tool (that needs a new declaration
and a
change of PREFILL_CHUNK).

Card classes (decision 24/09): CARD is the allow-list of 16 GB generation cards, NVIDIA RTX A4000
(sm86, preferred) and NVIDIA RTX 2000 Ada Generation (sm89, the pre-declared fallback), each with its tuned shrink
file config/lora_ops/<name>_SHRINK.json (the same default table with split_k 1; the RTX 2000 Ada file is unmeasured
until that class's own generation canary passes). Any other name, or another capability for a listed name, is
refused. One class per variant: gencoverage refuses two gpu_names, and the dev pilot and main may use different
classes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .modeling import LORA

GEN_PINS = {"vllm": "0.27.1", "torch": "2.13.0", "transformers": "5.17.0", "tokenizers": "0.23.2",
            "bitsandbytes": "0.50.2", "peft": "0.21.0", "safetensors": "0.8.0"}
_TF, _BNB = {"transformers": "5.15.0", "tokenizers": "0.22.2"}, {"bitsandbytes": "0.49.2"}
PIN_SETS = {"main": GEN_PINS, "fallback-tf": {**GEN_PINS, **_TF}, "fallback-bnb": {**GEN_PINS, **_BNB},
            "fallback-tf-bnb": {**GEN_PINS, **_TF, **_BNB}}
GPU_MEMORY_UTILIZATION = (0.92, 0.95)   # the default, and the pre-declared memory step (decision 24/09)
MEMORY_LADDER = ((0.92, False), (0.95, False), (0.95, True))    # (gpu_memory_utilization, enforce_eager), in order
PIN_SET_PARTS = {"main": frozenset(), "fallback-tf": frozenset({"tf"}), "fallback-bnb": frozenset({"bnb"}),
                 "fallback-tf-bnb": frozenset({"tf", "bnb"})}
DECISION_KEYS = ("gpu_name", "pin_set", "gpu_memory_utilization", "enforce_eager")    # a canary verdict's decision
PREFILL_CHUNK = 8192                    # max_num_batched_tokens: another value must be declared first
REPO = Path(__file__).resolve().parents[1]
LORA_OPS = REPO / "config" / "lora_ops"
BATCH_INVARIANT = True        # fallback (a), decision 05/10: EVERY job, at DETERMINISM_STEP, after a new canary
DETERMINISM_STEP = (0.95, True)          # decision 05/10: the only memory step while BATCH_INVARIANT (eager: no Inductor)
VLLM_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_BATCH_INVARIANT": "1" if BATCH_INVARIANT else "0",
            "VLLM_TUNED_CONFIG_FOLDER": str(LORA_OPS), "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_LORA_ENABLE_DUAL_STREAM": "0", "VLLM_USE_AOT_COMPILE": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
MAX_LORA_RANK = 16
LORA_INPUT_SIZES = {"codellama": (4096, 11008), "deepseek": (4096, 11008), "mistral": (4096, 14336)}  # hidden, inter.
SHRINK_M = (1, 127, 128, 8192)          # decode, both sides of the m = 128 default switch, a full prefill chunk
SHRINK_SLICES = (1, 2, 3)               # o/down, gate_up, qkv
SHRINK_DEFAULT_KEYS = frozenset({"block_m", "block_n", "block_k", "split_k", "num_warps", "num_ctas", "group_size_m",
                                 "num_stages", "max_nreg"})
EFFECTIVE = {"attention_backend": "FLASH_ATTN", "flash_attn_version": 2, "async_scheduling": False,
             "enable_prefix_caching": False, "max_num_seqs": 1}
LORA_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
QUANTIZATION = "bitsandbytes in-flight NF4 (quantize_4bit nf4, compress_statistics=True, blocksize 64), bf16 compute"
CARD = {"NVIDIA RTX A4000": "8.6", "NVIDIA RTX 2000 Ada Generation": "8.9"}   # 16 GB classes: name -> capability
CARD_MEMORY_GB = (15.0, 17.5)      # torch total_memory / 1e9 of a 16 GB card (decision 24/09: 16 GB only)


@dataclass(frozen=True)
class VLLMCfg:
    model: str                    # hub id (weights from the HF cache) or a local snapshot
    revision: str | None
    max_model_len: int            # the model window: max_new_tokens = min(6,144, window - prompt)
    eos_id: int
    gpu_memory_utilization: float = DETERMINISM_STEP[0]     # a memory step: decision 05/10, DETERMINISM_STEP
    enforce_eager: bool = DETERMINISM_STEP[1]               # (0.92 and CUDA graphs only with BATCH_INVARIANT off)
    lora_input_sizes: tuple[int, ...] = field(default=())   # k of the LoRA shrink: hidden and intermediate sizes

    def __post_init__(self):
        if self.gpu_memory_utilization not in GPU_MEMORY_UTILIZATION:
            raise ValueError(f"gpu_memory_utilization {self.gpu_memory_utilization} is not one of "
                             f"{GPU_MEMORY_UTILIZATION} (other values must be declared first)")


def vllm_kwargs(cfg: VLLMCfg) -> dict:
    return {"model": cfg.model, "revision": cfg.revision, "quantization": "bitsandbytes", "dtype": "bfloat16",
            "kv_cache_dtype": "auto", "max_model_len": cfg.max_model_len, "max_num_seqs": 1,
            "max_num_batched_tokens": PREFILL_CHUNK, "enable_prefix_caching": False,
            "gpu_memory_utilization": cfg.gpu_memory_utilization, "enable_lora": True, "max_lora_rank": MAX_LORA_RANK,
            "max_loras": 1, "max_cpu_loras": 1, "lora_dtype": "bfloat16", "skip_tokenizer_init": True,
            "generation_config": "vllm", "seed": 0, "enforce_eager": cfg.enforce_eager, "tensor_parallel_size": 1,
            "attention_config": {"backend": "FLASH_ATTN", "flash_attn_version": 2}, "async_scheduling": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}, "disable_log_stats": True}


def tuned_config_name(gpu_name: str, op: str = "shrink") -> str:
    """The file vLLM 0.27.1 looks for (lora/ops/triton_ops/utils.py load_lora_op_config)."""
    return f"{gpu_name.replace(' ', '_').replace('-', '_')}_{op.upper()}.json"


def tuned_config_problems(gpu_name: str) -> list[str]:
    name = tuned_config_name(gpu_name)
    return [] if (LORA_OPS / name).is_file() else [f"no tuned LoRA shrink config {name} in {LORA_OPS} for {gpu_name!r}"]


def default_shrink(m: int) -> dict:
    """vLLM 0.27.1's default shrink config (utils.py get_lora_op_configs) with split_k 1 instead of 64 / 8."""
    return {"block_m": 32, "block_n": 16, "block_k": 256 if m < 128 else 32, "split_k": 1, "num_warps": 4,
            "num_ctas": 1, "group_size_m": 8, "num_stages": 2, "max_nreg": None}


def shrink_table() -> dict:
    """config_data[max_loras][num_slices][m][k][n] of config/lora_ops/<card>_SHRINK.json."""
    ks = sorted({k for sizes in LORA_INPUT_SIZES.values() for k in sizes})
    return {"1": {str(s): {str(m): {str(k): {str(MAX_LORA_RANK): default_shrink(m)} for k in ks} for m in SHRINK_M}
                  for s in SHRINK_SLICES}}


def shrink_config_problems(get_configs, input_sizes) -> tuple[list[str], list[dict]]:
    """Read the EFFECTIVE shrink config back for every checked (num_slices, m, k); split_k must be 1 everywhere."""
    problems, checked = [], []
    for s in SHRINK_SLICES:
        for m in SHRINK_M:
            for k in input_sizes:
                cfg = dict(get_configs("shrink", max_loras=1, batch=m, hidden_size=k, rank=MAX_LORA_RANK,
                                       num_slices=s))
                checked.append({"num_slices": s, "m": m, "k": k, "config": cfg})
                if cfg.get("split_k") != 1:
                    problems.append(f"effective LoRA shrink config for num_slices={s} m={m} k={k} has split_k="
                                    f"{cfg.get('split_k')!r}, not 1 (non-deterministic split-K)")
    if not checked:
        problems.append("no LoRA input sizes: the shrink config was not checked")
    return problems, checked


def _name(value):
    return getattr(value, "name", value)


def effective_config(llm) -> dict:
    vc = llm.llm_engine.vllm_config
    return {"attention_backend": _name(vc.attention_config.backend),
            "flash_attn_version": vc.attention_config.flash_attn_version,
            "cudagraph_mode": _name(vc.compilation_config.cudagraph_mode),
            "compilation_mode": _name(vc.compilation_config.mode),
            "custom_ops_enabled": sorted(vc.compilation_config.enabled_custom_ops),     # e.g. rms_norm: forward_cuda
            "custom_ops_disabled": sorted(vc.compilation_config.disabled_custom_ops),
            "async_scheduling": vc.scheduler_config.async_scheduling,
            "enable_prefix_caching": vc.cache_config.enable_prefix_caching,
            "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "vllm_config_sha256": hashlib.sha256(str(vc).encode()).hexdigest()}


def batch_invariance_probe() -> dict:
    """decision 05/10: what fallback (a) installed in THIS process (vLLM runs in-process), read after the engine started:
    the card's capability, vLLM's is_device_capability_family(80) that decides its aten matmul overrides
    (batch_invariant.py L915), the batch-invariant mode, and the aten ops it overrides. A fact that cannot be read
    records its error instead: the 2-GPU byte gate decides, these explain the result."""
    import torch

    def capability():
        return list(torch.cuda.get_device_capability())

    def family():
        from vllm.platforms import current_platform
        return bool(current_platform.is_device_capability_family(80))

    def mode():
        from vllm.model_executor.layers import batch_invariant
        return bool(batch_invariant._batch_invariant_MODE)

    def overrides():
        from vllm.model_executor.layers import batch_invariant
        lib = batch_invariant._batch_invariant_LIB
        return None if lib is None else sorted(lib._op_impls)

    out = {}
    for key, read in (("device_capability", capability), ("capability_family_80", family),
                      ("batch_invariant_mode", mode), ("aten_overrides", overrides)):
        try:
            out[key] = read()
        except Exception as e:                 # recorded, not fatal: the receipt says what could not be read
            out[key] = {"error": f"{type(e).__name__}: {e}"}
    return out


def effective_problems(effective: dict) -> list[str]:
    return [f"effective {k} is {effective.get(k)!r}, pinned {v!r}" for k, v in EFFECTIVE.items()
            if effective.get(k) != v]


def tuned_config_files() -> dict[str, str]:
    """{file name: sha256} of the tuned LoRA op config folder (what vLLM reads, not where the checkout lives); {} when
    the folder is missing, which tuned_config_problems then refuses."""
    if not LORA_OPS.is_dir():
        return {}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(LORA_OPS.iterdir()) if p.is_file()}


def declared_engine(engine_settings: dict) -> dict:
    """The declared engine of a vLLM job identity (generate.job_engine_settings) without what names the model
    (revision, max_model_len): what a canary binds for every model of the variant and every job."""
    return {k: v for k, v in (engine_settings or {}).items() if k not in ("revision", "max_model_len")}


def names_engine(record) -> bool:
    """A vLLM engine record (VLLMEngine.describe) or its fingerprint names its engine: name "vllm" and a version (an
    empty record gave a fingerprint of None values, which passed every non-null
    check)."""
    return isinstance(record, dict) and record.get("name") == "vllm" and bool(record.get("version"))


def engine_record_problems(describe) -> list[str]:
    """Why an engine record gives no fingerprint ([] for a usable one): not
    an object, no vLLM name and version, or a part of an odd type (kwargs not an object; env, sampling, effective,
    lora_shrink or tuned_config neither absent, null nor an object; lora_shrink.checked not a list of records with a
    config). Coverage reports these as problem lines; nothing raises."""
    if not isinstance(describe, dict):
        return ["the engine record is not an object"]
    if not names_engine(describe):
        return ["it names no vLLM engine (name and version)"]
    out = []
    if not isinstance(describe.get("kwargs", {}), dict):
        out.append(f"engine.kwargs is {type(describe['kwargs']).__name__}, not an object")
    for part in ("env", "sampling", "effective", "lora_shrink", "tuned_config"):
        value = describe.get(part)
        if value is not None and not isinstance(value, dict):
            out.append(f"engine.{part} is {type(value).__name__}, not an object")
    shrink = describe.get("lora_shrink")
    checked = (shrink or {}).get("checked", []) if isinstance(shrink or {}, dict) else []
    if not (isinstance(checked, list) and all(isinstance(c, dict) and "config" in c for c in checked)):
        out.append("engine.lora_shrink.checked is not a list of records with a config")
    return out


def engine_fingerprint(describe: dict) -> dict | None:
    """What must be identical across every workdir of a variant: the engine without what names the model
    (model, revision, max_model_len, the EOS id in stop_token_ids, the vLLM config digest, the LoRA input sizes) and
    without the checkout path of the tuned config folder: its files' sha256 stand for it. None for a
    record that names no vLLM engine (names_engine), so a non-null check refuses it, and for a
    record with a part of an odd type (engine_record_problems)."""
    if engine_record_problems(describe):
        return None
    kwargs = {k: v for k, v in describe.get("kwargs", {}).items() if k not in ("model", "revision", "max_model_len")}
    env = {k: v for k, v in (describe.get("env") or {}).items() if k != "VLLM_TUNED_CONFIG_FOLDER"}
    effective = {k: v for k, v in (describe.get("effective") or {}).items() if k != "vllm_config_sha256"}
    shrink = sorted({json.dumps(c["config"], sort_keys=True) for c in (describe.get("lora_shrink") or {}).get(
        "checked", [])})
    return {"name": describe.get("name"), "version": describe.get("version"),
            "quantization": describe.get("quantization"),
            "batch_invariant": describe.get("batch_invariant"), "env": env,
            "tuned_config_files": (describe.get("tuned_config") or {}).get("files"),
            "sampling": {k: v for k, v in (describe.get("sampling") or {}).items() if k != "stop_token_ids"},
            "kwargs": kwargs, "effective": effective, "lora_shrink": shrink}


def sampling_kwargs(max_new_tokens: int, eos_id: int, ignore_eos: bool = False) -> dict:
    """Greedy on token ids. ignore_eos is for the canary's memory probe only (it runs to the full budget)."""
    return {"n": 1, "temperature": 0.0, "max_tokens": max_new_tokens, "stop_token_ids": [] if ignore_eos else [eos_id],
            "ignore_eos": ignore_eos, "detokenize": False}


def unpack(outputs, max_new_tokens: int, eos_id: int, ignore_eos: bool = False) -> tuple[list[int], str]:
    """vLLM RequestOutputs of ONE prompt -> (new ids, finish reason); anything unexpected raises."""
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("vLLM must return one completion for one prompt")
    c = outputs[0].outputs[0]
    ids, reason = list(c.token_ids), c.finish_reason
    if reason not in ("stop", "length"):
        raise RuntimeError(f"vLLM finish_reason {reason!r}")
    if reason == "stop" and (ignore_eos or not ids or ids[-1] != eos_id):
        raise RuntimeError(f"vLLM stopped on {ids[-1:]} (stop_reason {c.stop_reason!r}), not on the EOS {eos_id}")
    if len(ids) > max_new_tokens:
        raise RuntimeError(f"vLLM returned {len(ids)} tokens, more than max_new_tokens {max_new_tokens}")
    return ids, reason


def nvidia_driver() -> str | None:
    """The driver version nvidia-smi reports (recorded per job to compare pods), None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None


def _gpu() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_memory_gb": round(p.total_memory / 1e9, 2),
            "driver": nvidia_driver(), **_gpu_uuid(p)}


def _gpu_uuid(props) -> dict:
    """Recorded since 05/10: the physical GPU's UUID, as nvidia-smi prints it ("GPU-..."): torch's device properties
    first, nvidia-smi as the fallback, an error string (never a crash) when neither answers. The 2-GPU gate needs two
    different UUIDs."""
    uuid = getattr(props, "uuid", None)
    if uuid:
        text = str(uuid)
        return {"uuid": text if text.startswith("GPU-") else f"GPU-{text}", "uuid_source": "torch"}
    try:
        if shutil.which("nvidia-smi") is None:
            raise OSError("nvidia-smi not found")
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True, timeout=30)
        line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
        if not line.startswith("GPU-"):
            raise OSError(f"nvidia-smi gave {out.stdout.strip()!r} (exit {out.returncode})")
        return {"uuid": line, "uuid_source": "nvidia-smi"}
    except (OSError, subprocess.SubprocessError) as e:
        return {"uuid": f"error: {type(e).__name__}: {e}", "uuid_source": None}


class VLLMEngine:
    name = "vllm"

    def __init__(self, cfg: VLLMCfg, adapter: str | Path | None = None):
        step = (cfg.gpu_memory_utilization, cfg.enforce_eager)
        if BATCH_INVARIANT and step != DETERMINISM_STEP:
            raise RuntimeError(f"memory step {step} refused: decision 05/10, every job runs at {DETERMINISM_STEP} "
                               "(gpu_memory_utilization, enforce_eager) while batch invariance is on")
        os.environ.update(VLLM_ENV)            # before vllm is imported: it reads them at import and engine start
        try:
            import vllm
            from vllm.lora.request import LoRARequest
        except ImportError as e:
            raise RuntimeError("vLLM is not installed here: generation runs in the generation venv built from "
                               "requirements-gen.in, on the pod") from e
        self.cfg, self.version = cfg, vllm.__version__
        self.adapter = None if adapter is None else str(adapter)
        self._sampling = vllm.SamplingParams
        self._llm = vllm.LLM(**vllm_kwargs(cfg))
        self._lora = None if adapter is None else LoRARequest(Path(adapter).name, 1, str(adapter))
        from vllm.lora.ops.triton_ops.utils import get_lora_op_configs     # same process: the worker's configs

        problems, self.lora_shrink = shrink_config_problems(get_lora_op_configs, cfg.lora_input_sizes)
        self.effective = effective_config(self._llm)
        self.batch_invariance = batch_invariance_probe()
        problems += effective_problems(self.effective)
        if BATCH_INVARIANT:                    # decision 05/10: eager where it acts, no Inductor kernel, no CUDA graph
            problems += [f"effective {k} is {self.effective.get(k)!r}, eager needs 'NONE' (decision 05/10)"
                         for k in ("compilation_mode", "cudagraph_mode") if self.effective.get(k) != "NONE"]
        if problems:
            raise RuntimeError("vLLM engine is not the pinned configuration: " + "; ".join(problems[:5]))

    def generate(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False) -> tuple[list[int], str]:
        params = self._sampling(**sampling_kwargs(max_new_tokens, self.cfg.eos_id, ignore_eos))
        outputs = self._llm.generate([{"prompt_token_ids": list(prompt_ids)}], params, use_tqdm=False,
                                     lora_request=self._lora)
        return unpack(outputs, max_new_tokens, self.cfg.eos_id, ignore_eos)

    def describe(self) -> dict:
        sampling = {**sampling_kwargs(0, self.cfg.eos_id), "max_tokens": "min(6144, window - prompt_tokens)"}
        shrink = json.dumps(self.lora_shrink, sort_keys=True).encode()
        return {"name": self.name, "version": self.version, "quantization": QUANTIZATION, "max_num_seqs": 1,
                "kwargs": vllm_kwargs(self.cfg), "sampling": sampling, "env": dict(VLLM_ENV), "adapter": self.adapter,
                "batch_invariant": BATCH_INVARIANT, "batch_invariance": self.batch_invariance,
                "effective": self.effective,
                "lora_shrink": {"checked": self.lora_shrink, "sha256": hashlib.sha256(shrink).hexdigest()},
                "tuned_config": {"folder": str(LORA_OPS), "files": tuned_config_files()}, "gpu": _gpu()}


def load_hf(path: str, revision: str | None, quant: str, dtype: str, adapter: str | Path | None = None):
    """The training load (modeling.load_base: NF4 on the pod, full precision on CPU) plus the peft adapter; eval."""
    from .modeling import LoadCfg, load_base, model_device

    model = load_base(LoadCfg(path, revision, quant, "sdpa", dtype, gradient_checkpointing=False))
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), torch_device=str(model_device(model)))
    return model.eval()


class HFEngine:
    name = "hf"

    def __init__(self, model, eos_id: int, info: dict | None = None):
        self.model, self.eos_id, self.info = model, eos_id, info or {}

    def generate(self, prompt_ids: list[int], max_new_tokens: int, ignore_eos: bool = False) -> tuple[list[int], str]:
        import torch

        from .modeling import model_device

        ids = torch.tensor([prompt_ids], device=model_device(self.model))
        extra = {"min_new_tokens": max_new_tokens} if ignore_eos else {}
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, eos_token_id=self.eos_id, pad_token_id=self.eos_id, **extra)
        new = out[0][len(prompt_ids):].tolist()
        return new, "stop" if new and new[-1] == self.eos_id and not ignore_eos else "length"

    def describe(self) -> dict:
        return {"name": self.name, "max_num_seqs": 1, "sampling": "transformers generate(do_sample=False)",
                "gpu": _gpu(), **self.info}


def teacher_forced_agreement(model, prompt_ids: list[int], new_ids: list[int], chunk: int = 1024) -> dict:
    """Top-1 agreement of `model` with an output it did not generate: one forward over prompt + output (no cache), the
    LM head applied only at the output positions, `chunk` positions at a time, so logits never exceed chunk x vocab.
    Returns {"positions", "agree", "first_divergence"} (index into new_ids, None if every position agrees)."""
    import torch

    from .modeling import model_device

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids = torch.tensor([list(prompt_ids) + list(new_ids)], device=model_device(model))
    agree, first, start = 0, None, len(prompt_ids) - 1
    with torch.no_grad():
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
        for lo in range(0, len(new_ids), chunk):
            hi = min(lo + chunk, len(new_ids))
            pred = base.lm_head(hidden[:, start + lo:start + hi]).argmax(-1)[0].tolist()
            for i, (p, t) in enumerate(zip(pred, new_ids[lo:hi], strict=True)):
                if p == t:
                    agree += 1
                elif first is None:
                    first = lo + i
    return {"positions": len(new_ids), "agree": agree, "first_divergence": first}


def adapter_problems(adapter_dir: str | Path, expected_tensor_sha256: str,
                     expected_config_sha256: str | None = None) -> list[str]:
    """The adapter is the one the train receipt selected (tensor digest; and, when the job command names it, the
    sha256 of adapter_config.json) and one vLLM serves as trained."""
    from .receipt import tensor_digest

    d = Path(adapter_dir)
    missing = [n for n in ("adapter_config.json", "adapter_model.safetensors") if not (d / n).exists()]
    if missing:
        return [f"adapter {d}: missing {missing}"]
    cfg = json.loads((d / "adapter_config.json").read_text())
    out = []
    if cfg.get("r") != LORA["r"] or not isinstance(cfg.get("r"), int) or cfg["r"] > MAX_LORA_RANK:
        out.append(f"adapter r={cfg.get('r')!r}: trained rank is {LORA['r']}, engine max_lora_rank {MAX_LORA_RANK}")
    if cfg.get("lora_alpha") != LORA["lora_alpha"]:
        out.append(f"adapter lora_alpha={cfg.get('lora_alpha')!r}, trained with {LORA['lora_alpha']}")
    if cfg.get("use_dora"):
        out.append("adapter uses DoRA: vLLM does not serve DoRA")
    if cfg.get("use_rslora"):
        out.append("adapter uses rsLoRA scaling: not the trained alpha/r scaling")
    if cfg.get("bias", "none") != "none":
        out.append(f"adapter bias={cfg.get('bias')!r}: vLLM serves bias 'none' only")
    if cfg.get("modules_to_save"):
        out.append(f"adapter modules_to_save={cfg['modules_to_save']!r}: vLLM needs none")
    targets = cfg.get("target_modules")
    if not isinstance(targets, list):
        out.append(f"adapter target_modules={targets!r}: expected the resolved list peft saves")
    else:
        bad = sorted({t for t in targets if str(t).rsplit(".", 1)[-1] not in LORA_TARGETS})
        if bad:
            out.append(f"adapter target_modules outside {sorted(LORA_TARGETS)}: {bad[:5]}")
    digest = tensor_digest(d / "adapter_model.safetensors")
    if digest != expected_tensor_sha256:
        out.append(f"adapter tensor digest {digest} differs from the train receipt's {expected_tensor_sha256}")
    config = hashlib.sha256((d / "adapter_config.json").read_bytes()).hexdigest()
    if expected_config_sha256 is not None and config != expected_config_sha256:
        out.append(f"adapter_config.json sha256 {config} differs from the job command's {expected_config_sha256}")
    return out


def pin_problems(packages: dict, pin_set: str = "main") -> list[str]:
    """The installed packages are exactly the named pin set (every set is complete; any other set is refused)."""
    if pin_set not in PIN_SETS:
        return [f"pin set {pin_set!r} is not one of {sorted(PIN_SETS)}"]
    return [f"{k} {packages.get(k)!r} installed; pin set {pin_set!r} pins {v}" for k, v in PIN_SETS[pin_set].items()
            if packages.get(k) != v]


def tokenizes_like_training(pin_set: str) -> bool:
    """The set has training's transformers and tokenizers; otherwise a job must prove its prompt ids (ids_sha256)."""
    return all(PIN_SETS[pin_set][k] == GEN_PINS[k] for k in ("transformers", "tokenizers"))


def memory_step(decision: dict) -> int | None:
    """The decision's place on MEMORY_LADDER, None off the ladder (0.92 with enforce_eager is not a step)."""
    step = (decision.get("gpu_memory_utilization"), decision.get("enforce_eager"))
    found = [i for i, s in enumerate(MEMORY_LADDER)
             if s == step and all(type(x) is type(y) for x, y in zip(s, step, strict=True))]
    return found[0] if found else None


def decision_problems(decision) -> list[str]:
    """A canary decision the pre-declared orders know: exactly DECISION_KEYS, a pin set, a step of the ladder."""
    if not isinstance(decision, dict):
        return ["no decision"]
    out = []
    if sorted(decision) != sorted(DECISION_KEYS):
        out.append(f"decision keys {sorted(decision)} are not {sorted(DECISION_KEYS)}")
    if decision.get("pin_set") not in PIN_SET_PARTS:
        out.append(f"pin set {decision.get('pin_set')!r} is not one of {sorted(PIN_SET_PARTS)}")
    if memory_step(decision) is None:
        out.append(f"gpu_memory_utilization {decision.get('gpu_memory_utilization')!r} with enforce_eager "
                   f"{decision.get('enforce_eager')!r} is not a memory step of {MEMORY_LADDER}")
    out += card_problems(decision.get("gpu_name"))
    return out


def conservative_decision(decisions: dict[str, dict]) -> tuple[dict | None, list[str], list[str]]:
    """(target, the models whose canary must rerun at it, problems) of the decisions of one variant's models: the
    target is the decision they all share, else the most conservative one covering each: the highest memory step any
    model needed and the pin set holding every fallback any model needed, on the one card they all ran on. A decision
    off these orders, or two cards, has no target (problems say why)."""
    problems = [f"{m}: {p}" for m, d in sorted(decisions.items()) for p in decision_problems(d)]
    cards = sorted({str(d.get("gpu_name")) for d in decisions.values() if isinstance(d, dict)})
    if len(cards) > 1:
        problems.append(f"the canaries ran on several cards (gpu_name {cards}): one card class for the variant")
    if problems or not decisions:
        return None, [], problems or ["no decisions"]
    step = MEMORY_LADDER[max(memory_step(d) for d in decisions.values())]
    parts = frozenset().union(*(PIN_SET_PARTS[d["pin_set"]] for d in decisions.values()))
    target = {"gpu_name": cards[0], "pin_set": next(n for n, p in PIN_SET_PARTS.items() if p == parts),
              "gpu_memory_utilization": step[0], "enforce_eager": step[1]}
    return target, sorted(m for m, d in decisions.items() if d != target), []


def card_problems(gpu_name) -> list[str]:
    """The card class of a job, a canary or a decision is one of CARD (decision 24/09)."""
    return [] if gpu_name in CARD else [f"{gpu_name!r} is not a generation card class (one of {sorted(CARD)})"]


def device_problems(device: dict | None, gpu_name: str) -> list[str]:
    """Every output of a variant on ONE 16 GB card class of CARD: the name recorded in the job command, with that
    class's compute capability (A4000 sm86, RTX 2000 Ada sm89) and 16 GB of memory."""
    if not device:
        return ["no CUDA device"]
    out = card_problems(gpu_name)
    if gpu_name in CARD and device.get("capability") != CARD[gpu_name]:
        out.append(f"compute capability {device.get('capability')!r}; {gpu_name} has {CARD[gpu_name]}")
    if device.get("name") != gpu_name:
        out.append(f"device name {device.get('name')!r} is not the job's {gpu_name!r}")
    if not CARD_MEMORY_GB[0] <= float(device.get("total_memory_gb") or 0) <= CARD_MEMORY_GB[1]:
        out.append(f"total memory {device.get('total_memory_gb')} GB is not a 16 GB card")
    return out
