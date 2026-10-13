"""Load a base model for QLoRA training (NF4 on the pod, full precision in CPU tests) and attach the LoRA adapter.

peft's prepare_model_for_kbit_training is deliberately NOT used: it upcasts layer norms and embeddings to fp32, so the
residual stream and q/k/v reach SDPA in fp32 and the flash kernel can never run (it needs fp16/bf16), which silently
turns 8,192-token training into the slow memory-efficient/math path. bf16 has fp32's exponent range, so norms stay in
bf16; gradient checkpointing (non-reentrant) and input grads are enabled by hand instead. `flash_probe` proves on the
card that one forward runs with the flash kernel ONLY. `upcast_fp32=True` reproduces the original QLoRA upcast; it
exists only for the canary's side-by-side loss comparison and is refused in job mode.
"""
from __future__ import annotations

from dataclasses import dataclass

QLORA_DEVIATION = (
    "Norm and embedding weights stay bf16 (peft prepare_model_for_kbit_training, which upcasts them to fp32, is not "
    "used) so q/k/v reach SDPA in bf16 and the flash kernel can run. LlamaRMSNorm and MistralRMSNorm in transformers "
    "5.17.0 still compute the variance and the normalisation in fp32 internally (hidden_states.to(torch.float32)), then "
    "cast back to the input dtype before the weight multiply.")
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear", "bias": "none",
        "task_type": "CAUSAL_LM"}


@dataclass(frozen=True)
class LoadCfg:
    path: str                   # hub id or local snapshot directory
    revision: str | None
    quant: str = "nf4"          # "nf4" on the pod; "none" for CPU tests
    attn: str = "sdpa"          # requested attention implementation
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    upcast_fp32: bool = False   # the original QLoRA fp32 upcast of non-quantised weights; required for every real job


def load_base(cfg: LoadCfg):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = {"revision": cfg.revision, "dtype": getattr(torch, cfg.dtype), "attn_implementation": cfg.attn}
    if cfg.quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    elif cfg.quant != "none":
        raise ValueError(f"unknown quant {cfg.quant!r}")
    return AutoModelForCausalLM.from_pretrained(cfg.path, **kwargs)


def reload_roundtrip(cfg: LoadCfg, adapter_dir: str, out_dir: str) -> tuple[str, str]:
    """Load a saved adapter onto a fresh base, save it again; return (tensor digest before, after)."""
    from peft import PeftModel

    from .receipt import tensor_digest

    import gc

    import torch

    base = load_base(cfg)
    model = PeftModel.from_pretrained(base, adapter_dir, torch_device=str(model_device(base)))
    model.save_pretrained(out_dir)
    del model, base  # the canary parent must not keep a 4-5 GB NF4 base on the card while the next model trains
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (tensor_digest(f"{adapter_dir}/adapter_model.safetensors"),
            tensor_digest(f"{out_dir}/adapter_model.safetensors"))


def load_for_training(cfg: LoadCfg, seed: int):
    """Return (peft_model, info). info["attn_implementation_loaded"] is read back from the loaded model, not echoed."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model = load_base(cfg)
    attn_loaded = model.config._attn_implementation
    for p in model.parameters():
        p.requires_grad = False
        if cfg.upcast_fp32 and p.dtype in (torch.float16, torch.bfloat16) and type(p).__name__ != "Params4bit":
            p.data = p.data.to(torch.float32)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" not in n})
    info = {"attn_implementation_requested": cfg.attn, "attn_implementation_loaded": attn_loaded,
            "quant": cfg.quant, "dtype": cfg.dtype, "trainable_params": trainable, "lora": LORA,
            "base_param_dtypes": base_dtypes, "upcast_fp32": cfg.upcast_fp32,
            "qlora_deviation": None if cfg.upcast_fp32 else QLORA_DEVIATION}
    return model, info


def model_device(model):
    """The device the model's weights actually live on (NF4 loads go to cuda:0, CPU test models stay on CPU)."""
    return next(model.parameters()).device


def flash_probe(model, device, length: int = 256) -> bool | None:
    """True if a forward runs with SDPA restricted to the flash kernel; None for a model that is not on CUDA."""
    import torch

    if model_device(model).type != "cuda":
        return None
    from torch.nn.attention import SDPBackend, sdpa_kernel

    ids = torch.ones((1, length), dtype=torch.long, device=device)
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.no_grad():
            model(input_ids=ids)
        return True
    except RuntimeError:
        return False
