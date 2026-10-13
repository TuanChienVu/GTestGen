import pytest

from tests_gpu.tiny import make_tiny_llama
from v2gpu.modeling import LoadCfg, flash_probe, load_for_training


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_loaded_attention_is_read_back(tmp_path, tokenizers, attn):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    _, info = load_for_training(LoadCfg(path, None, quant="none", attn=attn, dtype="float32"), seed=42)
    assert info["attn_implementation_loaded"] == attn


def test_only_lora_weights_train(tmp_path, tokenizers):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    model, info = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32"), seed=42)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert info["trainable_params"] == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_base_weights_are_not_upcast(tmp_path, tokenizers):
    """No fp32 norms/embeddings: a bf16 load stays bf16, so q/k/v reach SDPA in bf16 (flash-eligible on the card)."""
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    _, info = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="bfloat16"), seed=42)
    assert info["base_param_dtypes"] == ["torch.bfloat16"]


def test_upcast_option_reproduces_the_original_qlora_upcast(tmp_path, tokenizers):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    _, info = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="bfloat16", upcast_fp32=True), seed=42)
    assert info["base_param_dtypes"] == ["torch.float32"] and info["qlora_deviation"] is None


def test_flash_probe_is_not_applicable_without_cuda(tmp_path, tokenizers):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    model, _ = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32"), seed=42)
    assert flash_probe(model, "cpu") is None


def test_unknown_quant_refused(tmp_path, tokenizers):
    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    with pytest.raises(ValueError):
        load_for_training(LoadCfg(path, None, quant="int3", attn="sdpa", dtype="float32"), seed=42)


def test_reload_roundtrip_frees_the_base(tmp_path, tokenizers, monkeypatch):
    """The base must be unreachable when gc.collect (then cuda.empty_cache) runs inside the call, not only after it
    returns: that is what hands the memory back before the next model trains."""
    import gc
    import weakref

    from v2gpu import modeling
    from v2gpu.modeling import reload_roundtrip

    path = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    model, _ = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32"), seed=42)
    model.save_pretrained(tmp_path / "adapter")
    bases, real = [], modeling.load_base
    monkeypatch.setattr(modeling, "load_base", lambda cfg: (lambda m: (bases.append(weakref.ref(m)), m)[1])(real(cfg)))
    freed_at_collect, collect = [], gc.collect
    monkeypatch.setattr(gc, "collect", lambda *a: (collect(*a), freed_at_collect.append(bases[0]() is None))[0])
    a, b = reload_roundtrip(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32"), str(tmp_path / "adapter"),
                            str(tmp_path / "again"))
    assert a == b and freed_at_collect == [True]
