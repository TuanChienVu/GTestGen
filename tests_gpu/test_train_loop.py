import json
import shutil

import pytest

from tests_gpu.tiny import make_tiny_llama
from v2gpu.examples import build
from v2gpu.modeling import LoadCfg, load_for_training
from v2gpu.serialize import AnswerTooLong, encode
from v2gpu import train_loop
from v2gpu.train_loop import LoopCfg, continue_after, run, select_epoch, warmup_steps

RECS = [{"id": f"r{i}", "partition": "train", "scenario": f"Given step number {i}",
         "answer": f"public function step{i}(): void {{ }}"} for i in range(6)]
IDS = [r["id"] for r in RECS]


def compose_prompt(record, fmt, tok, pool):
    return [{"role": "user", "content": f"[{fmt}] {record['scenario']}"}], {"examples": [], "knowledge": []}


def setup(tmp_path, tokenizers, seed=42):
    tok = tokenizers["codellama"]
    path = make_tiny_llama(tmp_path / "tiny", len(tok))
    model, _ = load_for_training(LoadCfg(path, None, quant="none", attn="sdpa", dtype="float32",
                                         gradient_checkpointing=False), seed=seed)
    ex = build(RECS, "codellama", tok, RECS, compose_prompt, lambda r: r["answer"])
    examples = {(e.record_id, e.fmt): e for e in ex}
    dev = {}
    for e in ex:
        dev.setdefault(e.fmt, []).append(encode(tok, e.messages, e.answer, 512))
    (tmp_path / "w").mkdir(parents=True, exist_ok=True)
    return tok, model, examples, dev


CFG = LoopCfg(lr=1e-2, warmup_ratio=0.0, grad_accum=2, epochs=3, max_extra_epochs=0, max_seq=512, save_every=100)
IDENTITY = {"model": "codellama", "seed": 42, "subset": "all", "data": {"records_sha256": "d" * 64},
           "train_config": {"lr": 1e-2}, "train_code_sha256": {"v2gpu/train_loop.py": "c" * 64}}


def cfg(**kw):
    return LoopCfg(**{**CFG.__dict__, **kw})


def adapter(path):
    from safetensors.torch import load_file

    return load_file(path / "adapter_model.safetensors")


def same(a, b):
    return a.keys() == b.keys() and all((a[k] == b[k]).all() for k in a)


def test_continue_after_rule():
    c = LoopCfg(epochs=3, max_extra_epochs=2, extra_epoch_min_gain=0.01)
    assert continue_after([3.0, 2.0], c)
    assert continue_after([3.0, 2.0, 1.9], c)            # 5% better than epoch 2: run epoch 4
    assert not continue_after([3.0, 2.0, 1.995], c)      # only 0.25% better: stop
    assert continue_after([3.0, 2.0, 1.9, 1.8], c)
    assert not continue_after([3.0, 2.0, 1.9, 1.8, 1.0], c)  # hard cap: epochs + max_extra_epochs


def test_select_epoch_is_argmin_earliest_on_tie():
    assert select_epoch([2.0, 1.5, 1.7]) == 2
    assert select_epoch([1.5, 1.5]) == 1


def test_three_epochs_save_adapters_learn_and_log_dev_by_format(tmp_path, tokenizers):
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    res = run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)
    assert res["status"] == "DONE" and res["global_step"] == 9 and len(res["dev_loss"]) == 3
    assert res["dev_loss"][2] < res["dev_loss"][0]
    assert sorted(res["dev_loss_by_format"][0]) == ["F0", "F1", "F2", "all"]
    assert res["dev_loss_by_format"][2]["all"] == res["dev_loss"][2]
    for e in (1, 2, 3):
        assert (tmp_path / "w" / f"epoch-{e}" / "adapter_model.safetensors").exists()
    lines = [json.loads(l) for l in (tmp_path / "w" / "metrics.jsonl").read_text().splitlines()]
    assert [l["step"] for l in lines if "step" in l] == list(range(1, 10))
    assert sum(sum(l["formats"].values()) for l in lines if "step" in l) == 18


def test_resume_reproduces_the_uninterrupted_run_without_duplicate_metrics(tmp_path, tokenizers):
    tok, model, examples, dev = setup(tmp_path / "a", tokenizers)
    run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "a" / "w", "cpu", IDENTITY)
    tok, model, examples, dev = setup(tmp_path / "b", tokenizers)
    assert run(model, tok, cfg(max_steps=4), IDS, 42, examples, dev, tmp_path / "b" / "w", "cpu", IDENTITY)["status"] == "MAX_STEPS"
    with (tmp_path / "b" / "w" / "metrics.jsonl").open("a") as f:   # a step logged after the last save, then a crash
        f.write(json.dumps({"step": 5, "epoch": 2, "loss": 0.0}) + "\n")
    tok, model, examples, dev = setup(tmp_path / "c", tokenizers)   # fresh process: new model object
    res = run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "b" / "w", "cpu", IDENTITY, resume=True)
    assert res["global_step"] == 9
    assert same(adapter(tmp_path / "a" / "w" / "epoch-3"), adapter(tmp_path / "b" / "w" / "epoch-3"))
    steps = [json.loads(l)["step"] for l in (tmp_path / "b" / "w" / "metrics.jsonl").read_text().splitlines()
             if "step" in json.loads(l)]
    assert steps == list(range(1, 10))


def test_resuming_a_finished_run_trains_nothing(tmp_path, tokenizers):
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    first = run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)
    tok, model, examples, dev = setup(tmp_path / "again", tokenizers)
    again = run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY, resume=True)
    assert again["global_step"] == first["global_step"] == 9 and again["dev_loss"] == first["dev_loss"]
    assert again["selected_epoch"] == first["selected_epoch"] and not (tmp_path / "w" / "epoch-4").exists()


def test_resume_survives_a_crash_inside_the_state_swap(tmp_path, tokenizers):
    tok, model, examples, dev = setup(tmp_path / "a", tokenizers)
    run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "a" / "w", "cpu", IDENTITY)
    tok, model, examples, dev = setup(tmp_path / "b", tokenizers)
    w = tmp_path / "b" / "w"
    run(model, tok, cfg(max_steps=4), IDS, 42, examples, dev, w, "cpu", IDENTITY)
    (w / "state").rename(w / "state.old")          # crash right after "rename state -> state.old"
    shutil.copytree(w / "state.old", w / "state.tmp")
    tok, model, examples, dev = setup(tmp_path / "c", tokenizers)
    assert run(model, tok, CFG, IDS, 42, examples, dev, w, "cpu", IDENTITY, resume=True)["global_step"] == 9
    assert same(adapter(tmp_path / "a" / "w" / "epoch-3"), adapter(w / "epoch-3"))


def test_answer_that_does_not_fit_is_an_error_not_a_truncation(tmp_path, tokenizers):
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    with pytest.raises(AnswerTooLong):
        run(model, tok, cfg(max_seq=20), IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)


def test_excluded_examples_are_skipped_and_counted(tmp_path, tokenizers):
    """Rule D removes (record, format) pairs; the rotation keeps its order and skips them."""
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    for key in [("r1", "F1"), ("r2", "F2"), ("r4", "F0")]:
        del examples[key]
    res = run(model, tok, CFG, IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)
    assert res["status"] == "DONE" and res["skipped_item_epochs"] == 3      # each pair is reached once in 3 epochs
    lines = [json.loads(l) for l in (tmp_path / "w" / "metrics.jsonl").read_text().splitlines()]
    assert sum(sum(l["formats"].values()) for l in lines if "step" in l) == 18 - 3


def test_state_files_are_fsynced(tmp_path, tokenizers, monkeypatch):
    import os

    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    run(model, tok, cfg(max_steps=1), IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)
    files = [p for p in (tmp_path / "w" / "state").rglob("*") if p.is_file()]
    assert len(synced) >= len(files) + 2      # every file, the new state directory and the workdir


def test_warmup_is_sized_from_the_items_that_remain(tmp_path, tokenizers, monkeypatch):
    c = cfg(warmup_ratio=0.5, epochs=1)
    assert warmup_steps(8, c) == 2 and warmup_steps(4, c) == 1 and warmup_steps(0, c) == 1
    tok, model, examples, dev = setup(tmp_path, tokenizers)
    for key in [k for k in examples if k[1] != "F0"]:
        del examples[key]                              # rule D left only F0: about a third of the epoch-1 items
    seen = []
    monkeypatch.setattr(train_loop, "warmup_steps", lambda n, c: (seen.append(n), 1)[1])
    run(model, tok, cfg(max_steps=1), IDS, 42, examples, dev, tmp_path / "w", "cpu", IDENTITY)
    from v2gpu.schedule import epoch_items
    assert seen == [sum(it.fmt == "F0" for it in epoch_items(IDS, 42, 1))]


def test_truncating_the_metrics_splits_lines_on_newline_only(tmp_path):
    """decision 24/09: the kept lines come back byte for byte, a line holding U+2028 or U+0085 included."""
    rows = [{"step": 1, "loss": 0.5, "note": "a\u2028b\x85c"}, {"step": 2, "loss": 0.4}]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    train_loop._truncate_metrics(path, {"global_step": 1, "epoch": 0})
    assert path.read_text(encoding="utf-8") == json.dumps(rows[0], ensure_ascii=False) + "\n"


# metrics.jsonl is flushed, not fsynced; a host crash can tear its last line


@pytest.mark.parametrize("tail", [b'{"step": 3, "loss": 0.', "{\"step\": 3, \"note\": \"é".encode()[:-1],
                                  b'{"step": 3, "loss": 0.\n', b"\x00\x00\x00\x00", b"7\n"])
def test_a_torn_final_metrics_line_is_dropped_on_resume(tmp_path, tail):
    """The resume run re-logs the steps after its saved position; a torn tail must not make it crash."""
    rows = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}]
    path = tmp_path / "metrics.jsonl"
    path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode() + tail)
    train_loop._truncate_metrics(path, {"global_step": 2, "epoch": 0})
    assert path.read_text(encoding="utf-8") == "".join(json.dumps(r) + "\n" for r in rows)


def test_an_unreadable_metrics_line_before_the_last_still_fails(tmp_path):
    """Only the tail can be torn by a crash; damage in the middle is something else and is not hidden."""
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step": 1, "loss": 0.5}\n{"step": 2, "lo\n{"step": 3, "loss": 0.3}\n')
    with pytest.raises(ValueError, match="metrics.jsonl:2"):
        train_loop._truncate_metrics(path, {"global_step": 3, "epoch": 0})
