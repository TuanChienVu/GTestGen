import json
from pathlib import Path

from v2gpu import special_tokens
from v2gpu.serialize import control_ids

CONFIG = Path(__file__).resolve().parents[1] / "config" / "special_tokens.json"


def test_config_file_matches_the_pinned_tokenizers(tokenizers):
    """compose reads config/special_tokens.json; it must equal what serialize blocks, token for token."""
    on_disk = json.loads(CONFIG.read_text())
    assert on_disk == special_tokens.export(tokenizers)
    for key, tok in tokenizers.items():
        assert {t["id"] for t in on_disk["models"][key]["control_tokens"]} == set(control_ids(tok))


def test_deepseek_letters_are_listed_as_allowed_not_blocked(tokenizers):
    ds = special_tokens.export(tokenizers)["models"]["deepseek"]
    allowed = {t["content"] for t in ds["allowed_added_tokens"]}
    assert {"ú", "õ", "Á"} <= allowed and not allowed & {t["content"] for t in ds["control_tokens"]}
