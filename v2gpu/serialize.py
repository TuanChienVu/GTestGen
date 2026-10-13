"""Chat-template serialisation shared by training and inference.

`prompt_ids` is the ONLY way a prompt becomes tokens, at train time and at generation time, so the adapter is trained on
exactly the token sequence it is later prompted with. The answer is whatever the model's own template appends after the
generation prompt, cut right after the first EOS; loss is computed on those tokens only. An answer is never truncated:
a record that does not fit raises `AnswerTooLong` (the context must be shrunk upstream, by v2tools.compose).

A prompt must hold no control token except the single leading BOS, and an answer none except its final EOS: Gherkin or
code that happens to contain "</s>", "<|EOT|>", "<｜fim▁hole｜>" or " <PRE>" would otherwise be tokenised as a control
token. Control tokens = the tokenizer's special ids plus its added tokens written "<...>"; DeepSeek's other added tokens
(ids 32000-32012) are single accented letters and stay allowed. `control_ids` is exported so compose can drop and count
such records with the same definition.

`decode_answer` inverts `answer_ids` up to `normalize_answer`: the templates add a separator before the answer (" " for
CodeLlama and Mistral, "\n" for DeepSeek) and CodeLlama trims the content, so any comparison of generated text with a
gold answer applies `normalize_answer` to the gold side and `decode_answer` to the generated ids.
"""
from __future__ import annotations

from dataclasses import dataclass

IGNORE = -100


class AnswerTooLong(ValueError):
    pass


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_prompt: int
    n_answer: int


def control_ids(tok) -> frozenset[int]:
    added = {i for i, t in tok.added_tokens_decoder.items() if str(t.content).lstrip("▁").startswith("<")}
    return frozenset(set(tok.all_special_ids) | added)


def normalize_answer(text: str) -> str:
    return text.strip()


def _prompt_text(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def prompt_ids(tok, messages: list[dict]) -> list[int]:
    ids = tok(_prompt_text(tok, messages), add_special_tokens=False)["input_ids"]
    ctrl = control_ids(tok)
    found = [(i, t) for i, t in enumerate(ids) if t in ctrl]
    if found != [(0, tok.bos_token_id)]:
        raise ValueError(f"prompt must hold exactly one control token, the leading BOS; found {found[:5]} "
                         "(a BOS/EOS or other control-token string in the messages?)")
    return ids


def answer_ids(tok, messages: list[dict], answer: str) -> list[int]:
    if control_ids(tok) & set(tok(answer, add_special_tokens=False)["input_ids"]):
        raise ValueError("answer text contains a control token (EOS, BOS or another special/added token)")
    prompt = _prompt_text(tok, messages)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the generation prompt is not a prefix of the full conversation")
    ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if tok.eos_token_id not in ids:
        raise ValueError("chat template rendered the answer without EOS")
    return ids[: ids.index(tok.eos_token_id) + 1]


def encode(tok, messages: list[dict], answer: str, max_seq: int) -> Encoded:
    p, a = prompt_ids(tok, messages), answer_ids(tok, messages, answer)
    if len(p) + len(a) > max_seq:
        raise AnswerTooLong(f"{len(p)} prompt + {len(a)} answer tokens > {max_seq}")
    return Encoded(p + a, [IGNORE] * len(p) + a, len(p), len(a))


def decode_answer(tok, ids: list[int]) -> str:
    """Generated (or gold) answer ids -> answer text: stop at EOS, drop the template's separator, strip the ends."""
    if tok.eos_token_id in ids:
        ids = ids[: ids.index(tok.eos_token_id)]
    return normalize_answer(tok.decode(ids, skip_special_tokens=True))


GEN_CAP = 6144


def max_new_tokens(prompt_len: int, window: int, cap: int = GEN_CAP) -> int:
    """decision 23/09: min(6,144, model window - prompt length); records whose prompt leaves no room are refused."""
    room = window - prompt_len
    if room <= 0:
        raise AnswerTooLong(f"prompt of {prompt_len} tokens leaves no room in a {window}-token window")
    return min(cap, room)
