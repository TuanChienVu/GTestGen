# tests_gpu/test_outlog.py
import json
import os

import pytest

from v2gpu.outlog import OutputLog, OutputLogError


def row(rid, cond="full_aware", text="x"):
    return {"record_id": rid, "model": "deepseek", "seed": "s42", "condition": cond, "variant": "main", "text": text}


def lines(path):
    return path.read_bytes().split(b"\n")


def test_each_row_is_one_sorted_line_fsynced_before_the_next(tmp_path, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd)))
    with OutputLog(tmp_path / "out.jsonl") as log:
        log.append(row("a", text="Étape"))
        n = len(synced)
        log.append(row("b"))
    assert n >= 1 and len(synced) >= n + 1
    first = json.loads(lines(tmp_path / "out.jsonl")[0])
    assert first["text"] == "Étape" and list(first) == sorted(first)
    assert "Étape".encode() in (tmp_path / "out.jsonl").read_bytes()          # ensure_ascii=False
    assert (tmp_path / "out.jsonl").read_bytes().endswith(b"\n")


def test_resume_knows_every_written_pair_and_refuses_a_second_copy(tmp_path):
    with OutputLog(tmp_path / "out.jsonl") as log:
        log.append(row("a"))
        log.append(row("a", cond="ft_only"))
    log = OutputLog(tmp_path / "out.jsonl")
    assert ("a", "deepseek", "s42", "full_aware", "main") in log and len(log.done) == 2
    with pytest.raises(OutputLogError, match="already"):
        log.append(row("a"))
    log.append(row("b"))
    log.close()
    assert len(lines(tmp_path / "out.jsonl")) == 4                             # three lines and the final newline


def test_a_torn_last_line_is_cut_and_counted(tmp_path):
    good = "".join(json.dumps(row(r), sort_keys=True) + "\n" for r in ("a", "b"))
    (tmp_path / "out.jsonl").write_text(good + '{"record_id": "c", "mod')
    with OutputLog(tmp_path / "out.jsonl") as log:
        assert log.truncated_tail == len('{"record_id": "c", "mod') and len(log.done) == 2
        log.append(row("c"))
    assert [json.loads(x)["record_id"] for x in lines(tmp_path / "out.jsonl") if x] == ["a", "b", "c"]


def test_a_complete_json_without_its_newline_is_still_a_torn_line(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(row("a"), sort_keys=True))
    with OutputLog(tmp_path / "out.jsonl") as log:
        assert log.truncated_tail > 0 and log.done == {}


def test_a_broken_line_before_the_end_is_a_refusal(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(row("a")) + "\n{broken\n" + json.dumps(row("b")) + "\n")
    with pytest.raises(OutputLogError, match="line 2"):
        OutputLog(tmp_path / "out.jsonl")


def test_duplicates_in_the_file_are_a_refusal(tmp_path):
    (tmp_path / "out.jsonl").write_text((json.dumps(row("a")) + "\n") * 2)
    with pytest.raises(OutputLogError, match="twice"):
        OutputLog(tmp_path / "out.jsonl")


def test_the_check_hook_refuses_lines_the_job_cannot_account_for(tmp_path):
    (tmp_path / "out.jsonl").write_text(json.dumps(row("a")) + "\n" + json.dumps(row("z")) + "\n")
    with pytest.raises(OutputLogError, match="line 2: not planned"):
        OutputLog(tmp_path / "out.jsonl", check=lambda r: [] if r["record_id"] == "a" else ["not planned"])


def test_a_new_file_starts_empty(tmp_path):
    with OutputLog(tmp_path / "sub" / "out.jsonl") as log:
        assert log.done == {} and log.truncated_tail == 0
    assert (tmp_path / "sub" / "out.jsonl").exists()


def test_a_second_writer_on_the_same_file_is_refused(tmp_path):
    """The same job started twice (a human and the queue) must never interleave or duplicate lines."""
    with OutputLog(tmp_path / "out.jsonl") as first:
        first.append(row("a"))
        with pytest.raises(OutputLogError, match="another process"):
            OutputLog(tmp_path / "out.jsonl")
        first.append(row("b"))
    with OutputLog(tmp_path / "out.jsonl") as again:                  # free once the first writer closed
        assert len(again.done) == 2


def test_a_text_holding_unicode_line_separators_stays_one_line(tmp_path):
    """ensure_ascii=False writes U+2028, U+2029 and U+0085 raw; the log splits on b"\\n" only."""
    text = "a b c\x85d"
    with OutputLog(tmp_path / "out.jsonl") as log:
        log.append(row("a", text=text))
    assert " " in (tmp_path / "out.jsonl").read_text()
    with OutputLog(tmp_path / "out.jsonl") as log:
        assert json.loads(log.done[("a", "deepseek", "s42", "full_aware", "main")])["text"] == text
