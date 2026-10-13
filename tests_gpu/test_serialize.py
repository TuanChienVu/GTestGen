import pytest

from v2gpu.serialize import IGNORE, AnswerTooLong, answer_ids, decode_answer, encode, prompt_ids

MSGS = [{"role": "user", "content": "Write Behat step definitions for these new steps:\n- I add a product \"PHP T-Shirt\""}]
ANS = "/**\n * @When I add a product :name\n */\npublic function iAddAProduct(string $name): void\n{\n}"
KEYS = ["codellama", "deepseek", "mistral"]


@pytest.mark.parametrize("key", KEYS)
def test_training_ids_equal_the_tokenised_full_conversation(tokenizers, key):
    """Not a tautology: the ids are built as prompt + answer, then compared with tokenising the whole text at once."""
    tok = tokenizers[key]
    full = tok.apply_chat_template(MSGS + [{"role": "assistant", "content": ANS}], tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    assert encode(tok, MSGS, ANS, 8192).input_ids == ids[: ids.index(tok.eos_token_id) + 1]


@pytest.mark.parametrize("key", KEYS)
def test_inference_prompt_is_the_training_prefix(tokenizers, key):
    tok = tokenizers[key]
    enc = encode(tok, MSGS, ANS, 8192)
    assert enc.input_ids[: enc.n_prompt] == prompt_ids(tok, MSGS)


@pytest.mark.parametrize("key", KEYS)
def test_loss_only_on_answer_and_answer_ends_with_eos(tokenizers, key):
    tok = tokenizers[key]
    enc = encode(tok, MSGS, ANS, 8192)
    assert enc.labels[: enc.n_prompt] == [IGNORE] * enc.n_prompt
    assert enc.labels[enc.n_prompt:] == enc.input_ids[enc.n_prompt:]
    assert enc.input_ids[-1] == tok.eos_token_id and enc.input_ids.count(tok.eos_token_id) == 1
    assert enc.input_ids.count(tok.bos_token_id) == 1


@pytest.mark.parametrize("key", KEYS)
def test_decode_answer_inverts_answer_ids(tokenizers, key):
    tok = tokenizers[key]
    assert decode_answer(tok, answer_ids(tok, MSGS, ANS)) == ANS
    assert decode_answer(tok, answer_ids(tok, MSGS, ANS)[:-1]) == ANS   # generation cut before EOS decodes the same


@pytest.mark.parametrize("key,special", [("codellama", "</s>"), ("deepseek", "<|EOT|>"), ("mistral", "<s>")])
def test_special_token_strings_inside_the_prompt_are_refused(tokenizers, key, special):
    msgs = [{"role": "user", "content": f"Given the page shows {special} somewhere"}]
    with pytest.raises(ValueError, match="BOS|EOS"):
        prompt_ids(tokenizers[key], msgs)


def test_answer_is_never_truncated(tokenizers):
    tok = tokenizers["deepseek"]
    n = len(encode(tok, MSGS, ANS, 8192).input_ids)
    assert len(encode(tok, MSGS, ANS, n).input_ids) == n
    with pytest.raises(AnswerTooLong):
        encode(tok, MSGS, ANS, n - 1)


def test_eos_text_inside_answer_is_refused(tokenizers):
    with pytest.raises(ValueError, match="EOS"):
        encode(tokenizers["codellama"], MSGS, ANS + "</s>", 8192)


def test_deepseek_trailing_newline_after_eot_is_dropped(tokenizers):
    assert encode(tokenizers["deepseek"], MSGS, ANS, 8192).input_ids[-1] == 32021  # <|EOT|>


def test_generation_budget_rule():
    from v2gpu.serialize import max_new_tokens

    assert max_new_tokens(3000, 16384) == 6144
    assert max_new_tokens(12801, 16384) == 3583          # the longest DeepSeek test prompt measured on out3
    assert max_new_tokens(14069, 32768) == 6144          # Mistral's 32K window
    with pytest.raises(AnswerTooLong):
        max_new_tokens(16384, 16384)


@pytest.mark.parametrize("key,text", [("deepseek", "<｜fim▁hole｜>"), ("deepseek", "<｜end▁of▁sentence｜>"), ("deepseek", "<pad>"),
                                      ("codellama", " <PRE>"), ("codellama", "<SUF>"), ("codellama", "<EOT>"),
                                      ("codellama", "<FILL_ME>"), ("mistral", "<unk>")])
def test_every_control_token_string_in_the_prompt_is_refused(tokenizers, key, text):
    with pytest.raises(ValueError, match="control token"):
        prompt_ids(tokenizers[key], [{"role": "user", "content": f"Given the page shows {text} somewhere"}])


@pytest.mark.parametrize("key", KEYS)
def test_accented_letters_are_not_control_tokens(tokenizers, key):
    """DeepSeek's added tokens 32000-32012 are single letters (õ, ú, Á...), legitimate in Gherkin and code."""
    msgs = [{"role": "user", "content": "Given the customer \"João Ú Álvarez\" pays 5 ÷ 2"}]
    assert prompt_ids(tokenizers[key], msgs)


@pytest.mark.parametrize("key,text", [("codellama", "<s>"), ("deepseek", "<｜begin▁of▁sentence｜>"), ("mistral", "<unk>")])
def test_control_tokens_inside_the_answer_are_refused(tokenizers, key, text):
    with pytest.raises(ValueError, match="control token"):
        encode(tokenizers[key], MSGS, ANS + f" // {text}", 8192)


@pytest.mark.parametrize("key", KEYS)
def test_decoded_answer_equals_the_normalised_gold(tokenizers, key):
    """Same normalisation on both sides of any comparison: CodeLlama's template trims, so gold is compared stripped."""
    from v2gpu.serialize import normalize_answer

    gold = "    public function x(): void\n    {\n    }\n"
    assert decode_answer(tokenizers[key], answer_ids(tokenizers[key], MSGS, gold)) == normalize_answer(gold)
