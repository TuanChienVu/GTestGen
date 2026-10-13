"""Write config/special_tokens.json: per model, the control tokens a prompt or an answer must never contain.

python -m v2gpu.special_tokens > config/special_tokens.json   (reads the pinned tokenizers from the HF cache)
compose (v2tools) reads this file to drop and count such records, so the two sides share one definition without v2tools
importing v2gpu. Control = special ids + added tokens written "<...>"; DeepSeek's other added tokens are single accented
letters and are listed under allowed_added_tokens, with the reason.
"""
from __future__ import annotations

import json
import sys

from .models import MODELS, load_tokenizer
from .serialize import control_ids


def export(tokenizers: dict) -> dict:
    out = {"definition": "special ids plus added tokens whose content (without a leading ▁) starts with '<'",
           "models": {}}
    for key, tok in sorted(tokenizers.items()):
        ctrl = control_ids(tok)
        added = {i: str(t.content) for i, t in tok.added_tokens_decoder.items()}
        content = {**added, **{i: tok.convert_ids_to_tokens(i) for i in ctrl if i not in added}}
        out["models"][key] = {
            "hf_id": MODELS[key].hf_id, "revision": MODELS[key].revision,
            "control_tokens": [{"id": i, "content": content[i]} for i in sorted(ctrl)],
            "allowed_added_tokens": [{"id": i, "content": c, "reason": "ordinary letter or symbol, not a control token"}
                                     for i, c in sorted(added.items()) if i not in ctrl]}
    return out


if __name__ == "__main__":
    json.dump(export({k: load_tokenizer(s) for k, s in MODELS.items()}), sys.stdout, ensure_ascii=False, indent=1,
              sort_keys=True)
    sys.stdout.write("\n")
