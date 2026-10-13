"""A tiny random Llama sharing CodeLlama's vocabulary, so CPU tests run the real tokenizer + chat template path."""
from pathlib import Path


def make_tiny_llama(path: Path, vocab_size: int) -> str:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return str(path)
