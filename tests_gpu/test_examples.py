import hashlib
import json

import pytest

from v2gpu.examples import (Example, build, compose_prompt_pending, control_strings, coverage, gold_answer_pending,
                            pool_digest, pool_problems, prompt_files, read_excluded, read_infra, read_jsonl, read_pools,
                            retrieval_pool, structure_problems, subset_ids, write_jsonl)

RECS = [{"id": "b", "partition": "train", "scenario": "S2", "answer": "A2", "subsets": []},
        {"id": "a", "partition": "train", "scenario": "S1", "answer": "A1", "subsets": ["lc600", "lc1200"]}]


def compose_prompt(record, fmt, tok, pool):
    return ([{"role": "user", "content": f"{fmt}: {record['scenario']} (pool {len(pool)})"}],
            {"examples": [p["id"] for p in pool[:1]] if fmt != "F0" else [], "knowledge": [], "skeleton": [], "dropped": []})


def gold_answer(record):
    return record["answer"]


def test_build_one_example_per_record_and_format_with_model_pool_and_provenance():
    ex = build(RECS, "deepseek", None, RECS[1:], compose_prompt, gold_answer)
    assert [(e.record_id, e.fmt) for e in ex] == [("a", "F0"), ("a", "F1"), ("a", "F2"), ("b", "F0"), ("b", "F1"), ("b", "F2")]
    assert ex[1].messages[0]["content"] == "F1: S1 (pool 1)" and ex[1].answer == "A1" and ex[1].partition == "train"
    assert {e.model for e in ex} == {"deepseek"} and {e.pool_sha256 for e in ex} == {pool_digest(["a"])}
    assert ex[1].provenance == {"examples": ["a"], "knowledge": [], "skeleton": [], "dropped": []}


def test_test_and_aux_prompts_never_call_gold_answer():
    def boom(record):
        raise AssertionError("gold_answer called for a test record")

    test = [dict(RECS[0], partition="test"), dict(RECS[1], partition="aux_test")]
    assert {e.answer for e in build(test, "mistral", None, RECS, compose_prompt, boom)} == {None}


def test_pool_digest_ignores_order():
    assert pool_digest(["b", "a"]) == pool_digest(["a", "b"]) != pool_digest(["a"])


def test_roundtrip_several_files_and_duplicates(tmp_path):
    ex = build(RECS, "mistral", None, RECS, compose_prompt, gold_answer)
    write_jsonl(tmp_path / "f0.jsonl", [e for e in ex if e.fmt == "F0"])
    write_jsonl(tmp_path / "f12.jsonl", [e for e in ex if e.fmt != "F0"])
    back = read_jsonl([tmp_path / "f0.jsonl", tmp_path / "f12.jsonl"])
    assert len(back) == 6 and isinstance(back[("b", "F2")], Example) and back[("b", "F2")].answer == "A2"
    with pytest.raises(ValueError):
        read_jsonl([tmp_path / "f0.jsonl", tmp_path / "f0.jsonl"])


def test_prompt_files_must_match_their_sums(tmp_path):
    ex = build(RECS, "mistral", None, RECS, compose_prompt, gold_answer)
    d1 = write_jsonl(tmp_path / "train.F0.jsonl", ex[:3])
    meta = json.dumps({"v2tools_commit": "abc", "compose": {"k": 8}})
    (tmp_path / "PROMPTS_META.json").write_text(meta)
    (tmp_path / "train.F1.jsonl").unlink(missing_ok=True)
    (tmp_path / "PROMPTS_SHA256SUMS").write_text(f"{d1}  train.F0.jsonl\n{hashlib.sha256(meta.encode()).hexdigest()}  PROMPTS_META.json\n")
    problems, meta = prompt_files([tmp_path / "train.F0.jsonl"], tmp_path / "PROMPTS_SHA256SUMS")
    assert problems == [] and meta["meta"]["v2tools_commit"] == "abc"
    (tmp_path / "PROMPTS_META.json").write_text(json.dumps({"v2tools_commit": "edited later"}))
    assert any("PROMPTS_META.json: sha256 differs" in p
               for p in prompt_files([tmp_path / "train.F0.jsonl"], tmp_path / "PROMPTS_SHA256SUMS")[0])
    assert meta["sums_sha256"] == hashlib.sha256((tmp_path / "PROMPTS_SHA256SUMS").read_bytes()).hexdigest()
    write_jsonl(tmp_path / "train.F1.jsonl", ex[3:])                      # not listed (and a stray file of the dir)
    (tmp_path / "train.F0.jsonl").write_text("changed\n")                 # listed but altered
    problems, _ = prompt_files([tmp_path / "train.F0.jsonl", tmp_path / "train.F1.jsonl"], tmp_path / "PROMPTS_SHA256SUMS")
    assert any("sha256" in p for p in problems) and any("not listed" in p for p in problems)


def test_compose_is_pending_until_phase3():
    with pytest.raises(NotImplementedError):
        build(RECS, "deepseek", None, RECS, compose_prompt_pending, gold_answer)
    with pytest.raises(NotImplementedError):
        build(RECS, "deepseek", None, RECS, compose_prompt, gold_answer_pending)


def test_subset_ids():
    assert subset_ids(RECS, "all") == ["a", "b"]
    assert subset_ids(RECS, "lc600") == ["a"]
    with pytest.raises(ValueError):
        subset_ids(RECS, "lc9999")


def test_excluded_pairs_are_read_from_a_listed_file(tmp_path):
    from v2gpu.examples import read_excluded

    rows = [{"record_id": "a", "fmt": None, "model": None, "reason": "control_tokens"},
            {"record_id": "b", "fmt": "F2", "model": "mistral", "reason": "too_long"}]
    data = "".join(json.dumps(r) + "\n" for r in rows)
    (tmp_path / "excluded.jsonl").write_text(data)
    (tmp_path / "PROMPTS_SHA256SUMS").write_text(f"{hashlib.sha256(data.encode()).hexdigest()}  excluded.jsonl\n")
    problems, pairs = read_excluded(tmp_path / "PROMPTS_SHA256SUMS", "mistral")
    assert problems == [] and pairs == {("a", "F0"): "control_tokens", ("a", "F1"): "control_tokens",
                                        ("a", "F2"): "control_tokens", ("b", "F2"): "too_long"}
    assert read_excluded(tmp_path / "PROMPTS_SHA256SUMS", "deepseek")[1] == {
        ("a", "F0"): "control_tokens", ("a", "F1"): "control_tokens", ("a", "F2"): "control_tokens"}
    (tmp_path / "PROMPTS_SHA256SUMS").write_text("")
    assert any("excluded.jsonl" in p for p in read_excluded(tmp_path / "PROMPTS_SHA256SUMS", "mistral")[0])


def line(**kw):
    row = {"record_id": "a", "fmt": "F2", "model": "m", "partition": "train", "pool_sha256": "p",
           "messages": [{"role": "user", "content": "x"}], "answer": "A",
           "provenance": {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}}
    return {**row, **kw}


def test_the_reader_wants_exactly_the_eight_keys(tmp_path):
    for bad in ({k: v for k, v in line().items() if k != "provenance"}, dict(line(), extra=1)):
        (tmp_path / "p.jsonl").write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match="p.jsonl:1"):
            read_jsonl(tmp_path / "p.jsonl")


def test_the_reader_refuses_a_test_or_aux_line_with_an_answer(tmp_path):
    for part in ("test", "aux_test"):
        (tmp_path / "p.jsonl").write_text(json.dumps(line(partition=part, answer="GOLD CODE")) + "\n")
        with pytest.raises(ValueError, match="answer"):
            read_jsonl(tmp_path / "p.jsonl")
    (tmp_path / "p.jsonl").write_text(json.dumps(line(partition="test", answer=None)) + "\n")
    assert read_jsonl(tmp_path / "p.jsonl")[("a", "F2")].answer is None


def test_structure_one_user_turn_and_four_provenance_keys_by_format():
    ok = Example(**line())
    assert structure_problems(ok) == []
    system = Example(**line(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]))
    assert any("one user turn" in p for p in structure_problems(system))
    three = Example(**line(provenance={"examples": [], "knowledge": []}))
    assert any("provenance keys" in p for p in structure_problems(three))
    not_list = Example(**line(provenance={"examples": "e", "knowledge": [], "skeleton": [], "dropped": []}))
    assert any("provenance.examples is not a list" in p for p in structure_problems(not_list))
    f0 = Example(**line(fmt="F0", provenance={"examples": ["e"], "knowledge": [], "skeleton": ["s"], "dropped": []}))
    assert {p for p in structure_problems(f0)} == {"F0 must have empty provenance.examples",
                                                   "F0 must have empty provenance.skeleton"}
    f1 = Example(**line(fmt="F1", provenance={"examples": ["e"], "knowledge": ["k"], "skeleton": [], "dropped": []}))
    assert structure_problems(f1) == ["F1 must have empty provenance.knowledge"]


def test_coverage_accepts_a_missing_pair_only_when_listed():
    have = {("a", "F0"): 1, ("b", "F0"): 1}
    wanted = [("a", "F0"), ("b", "F0"), ("c", "F0")]
    problems, counts = coverage("train", wanted, have, {("c", "F0"): "control_tokens"})
    assert problems == [] and counts == {"control_tokens": 1}
    problems, _ = coverage("train", wanted, have, {("a", "F0"): "too_long"})
    assert problems == ["train a F0: listed in excluded.jsonl but has a prompt", "missing train prompt c F0"]


def pool_rec(rid, scenario="Given x", code="function x() {}", steps=()):
    return {"id": rid, "repo": "R", "background": "", "scenario": scenario, "steps": list(steps),
            "create": [{"fn_id": f"{rid}.php:1", "code": code, "preexisting": False}]}


def test_the_retrieval_pool_is_one_pool_for_every_model():
    """decision of 24/09: subset minus uses_test_infra minus records holding a control token of ANY model in their Gherkin
    or gold; the model-specific exclusions of the query do not change it."""
    recs = [pool_rec("a"), pool_rec("b"), pool_rec("c", scenario="Given </s> in a step"),
            pool_rec("d", code="echo '<|EOT|>';"), pool_rec("e", steps=["a <PRE> tag"]), pool_rec("f")]
    listed = {("b", f): "uses_test_infra" for f in ("F0", "F1", "F2")} | {("f", f): "control_tokens" for f in ("F0", "F1", "F2")}
    control = {"</s>", "<|EOT|>", "<PRE>"}
    assert retrieval_pool(recs, [r["id"] for r in recs], listed, control, {}) == ["a", "f"]


def test_control_strings_cover_every_model_without_the_leading_space_mark():
    c = control_strings()
    assert {"</s>", "<s>", "<|EOT|>", "<｜begin▁of▁sentence｜>", "<PRE>", "<FILL_ME>"} <= c
    assert not any(s.startswith("\u2581") for s in c) and "é" not in c


def test_infra_names_come_from_a_listed_file(tmp_path):
    data = json.dumps({"R": [["src/Ctx.php", "lastOrderFor"], ["src/Ctx.php", "orderRepo"]]})
    (tmp_path / "infra.json").write_text(data)
    (tmp_path / "PROMPTS_SHA256SUMS").write_text(f"{hashlib.sha256(data.encode()).hexdigest()}  infra.json\n")
    assert read_infra(tmp_path / "PROMPTS_SHA256SUMS") == ([], {"R": {("src/Ctx.php", "lastOrderFor"),
                                                                     ("src/Ctx.php", "orderRepo")}})
    (tmp_path / "PROMPTS_SHA256SUMS").write_text("")
    assert any("infra.json" in p for p in read_infra(tmp_path / "PROMPTS_SHA256SUMS")[0])


def sums_for(d, *names):
    (d / "PROMPTS_SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
    return d / "PROMPTS_SHA256SUMS"


def test_blank_or_broken_lines_are_problems_not_crashes(tmp_path):
    (tmp_path / "excluded.jsonl").write_text('{"record_id": "a", "fmt": null, "model": null, "reason": "x"}\n\n{broken\n')
    (tmp_path / "infra.json").write_text("{not json")
    (tmp_path / "pools.json").write_text("[]")
    sums = sums_for(tmp_path, "excluded.jsonl", "infra.json", "pools.json")
    problems, pairs = read_excluded(sums, "m")
    assert any("excluded.jsonl:2: blank line" in p for p in problems) and any("excluded.jsonl:3:" in p for p in problems)
    assert pairs == {("a", "F0"): "x", ("a", "F1"): "x", ("a", "F2"): "x"}
    assert any("infra.json:" in p for p in read_infra(sums)[0])
    assert any("pools.json:" in p for p in read_pools(sums)[0])


def test_a_file_of_the_prompt_directory_that_the_sums_do_not_list_is_refused(tmp_path):
    ex = build(RECS, "mistral", None, RECS, compose_prompt, gold_answer)
    write_jsonl(tmp_path / "train.F0.jsonl", ex[:3])
    (tmp_path / "PROMPTS_META.json").write_text("{}")
    sums = sums_for(tmp_path, "train.F0.jsonl", "PROMPTS_META.json")
    assert prompt_files([tmp_path / "train.F0.jsonl"], sums)[0] == []
    (tmp_path / "old.F0.jsonl").write_text("stale\n")
    assert prompt_files([tmp_path / "train.F0.jsonl"], sums)[0] == [
        "old.F0.jsonl: in the prompt directory but not listed in PROMPTS_SHA256SUMS"]


def test_pools_json_must_equal_the_recomputed_pool(tmp_path):
    ids = ["a", "c"]
    (tmp_path / "pools.json").write_text(json.dumps({"all": {"sha256": pool_digest(ids), "ids": ids}}))
    problems, pools = read_pools(sums_for(tmp_path, "pools.json"))
    assert problems == [] and pool_problems(pools, "all", ["a", "c"]) == []
    assert any("missing ['b']" in p for p in pool_problems(pools, "all", ["a", "b", "c"]))
    assert any("no pool 'lc600'" in p for p in pool_problems(pools, "lc600", ["a"]))
    bad = {"all": {"sha256": "0" * 64, "ids": ids}}
    assert any("sha256" in p for p in pool_problems(bad, "all", ids))


def test_pool_problems_refuses_non_string_ids_without_crashing():
    bad = {"all": {"sha256": "0" * 64, "ids": ["a", 1]}}
    assert pool_problems(bad, "all", ["a", "1"]) == ["pools.json pool 'all': ids are not all strings"]


def test_a_non_object_line_in_excluded_is_a_problem_not_a_crash(tmp_path):
    (tmp_path / "excluded.jsonl").write_text("null\n[1]\n5\n")
    sums = sums_for(tmp_path, "excluded.jsonl")
    problems, pairs = read_excluded(sums, "m")
    assert pairs == {}
    assert problems == ["excluded.jsonl:1: not a JSON object", "excluded.jsonl:2: not a JSON object",
                        "excluded.jsonl:3: not a JSON object"]


def test_a_non_object_line_in_a_prompt_file_is_a_refusal_not_a_crash(tmp_path):
    for text in ("null\n", "5\n", "[1]\n"):
        (tmp_path / "p.jsonl").write_text(text)
        with pytest.raises(ValueError, match="p.jsonl:1: not a JSON object"):
            read_jsonl(tmp_path / "p.jsonl")


def test_the_reader_refuses_a_line_outside_the_expected_partitions(tmp_path):
    """A generation pod reads test prompts with partitions=("test",
    "aux_test"), so a line mislabelled "train", which may carry an answer, is refused instead of read."""
    (tmp_path / "p.jsonl").write_text(json.dumps(line(partition="test", answer=None)) + "\n" +
                                      json.dumps(line(record_id="b", partition="train", answer="GOLD CODE")) + "\n")
    with pytest.raises(ValueError, match=r"p.jsonl:2: partition 'train', expected one of \['aux_test', 'test'\]"):
        read_jsonl(tmp_path / "p.jsonl", partitions=("test", "aux_test"))
    assert len(read_jsonl(tmp_path / "p.jsonl")) == 2                  # without partitions: the A2 behaviour


def test_the_reader_splits_lines_on_newline_only(tmp_path):
    """json.dumps(ensure_ascii=False) writes U+2028, U+2029 and U+0085 raw; str.splitlines would cut the
    line there. The reader splits bytes on b"\n" (examples.jsonl_lines, used by every reader of the GPU code)."""
    from v2gpu.examples import jsonl_lines

    text = "Given a\u2028b and c\x85d\u2029"
    row = line(messages=[{"role": "user", "content": text}])
    (tmp_path / "p.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(tmp_path / "p.jsonl")[("a", "F2")].messages[0]["content"] == text
    (tmp_path / "q.jsonl").write_bytes(b'{"x": 1}\n{"y": "\xe2\x80\xa8"}')
    assert [json.loads(x) for x in jsonl_lines(tmp_path / "q.jsonl")] == [{"x": 1}, {"y": "\u2028"}]


def test_the_record_and_exclusion_readers_split_lines_on_newline_only(tmp_path):
    """decision 24/09: A2's own JSONL readers use jsonl_lines too; a Gherkin text may hold U+2028 or U+0085."""
    from v2gpu.examples import read_records

    rec = {"id": "Sylius/Sylius|a.feature|1|s", "scenario": "Given a\u2028b and c\x85d"}
    (tmp_path / "records.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_records(tmp_path / "records.jsonl") == [rec]
    row = {"record_id": rec["id"], "fmt": None, "model": None, "reason": "control_token", "detail": "a\u2028b\x85c"}
    (tmp_path / "excluded.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "excluded.jsonl").read_bytes()).hexdigest()
    (tmp_path / "PROMPTS_SHA256SUMS").write_text(f"{digest}  excluded.jsonl\n")
    problems, pairs = read_excluded(tmp_path / "PROMPTS_SHA256SUMS", "deepseek")
    assert problems == [] and pairs == {(rec["id"], f): "control_token" for f in ("F0", "F1", "F2")}


@pytest.mark.parametrize("data", [
    b"\xef\xbb\xbf" + b'{"id": "a"}\n',                         # a UTF-8 BOM (json.loads(bytes) reads it as utf-8-sig)
    '{"id": "a"}'.encode("utf-16"),                                # UTF-16 (json.loads(bytes) detects it)
    '{"id": "a"}\n'.encode("utf-32-le"),                            # UTF-32 without a BOM
    b'{"id": "a\xed\xa0\x80"}\n',                                  # a lone surrogate (json.loads(bytes): surrogatepass)
], ids=["bom", "utf16", "utf32", "surrogate"])
def test_the_readers_decode_strict_utf8_as_the_a2_readers_did(tmp_path, data):
    """jsonl_lines decodes strict UTF-8 before any JSON parse, so these are
    refused as str.splitlines over read_text(encoding="utf-8") refused them; U+2028 inside a string stays one line
    (test_the_reader_splits_lines_on_newline_only)."""
    from v2gpu.examples import read_records

    (tmp_path / "r.jsonl").write_bytes(data)
    with pytest.raises(ValueError):
        read_records(tmp_path / "r.jsonl")


def test_the_reader_refuses_an_answered_test_line_when_partitions_are_given(tmp_path):
    """The partition filter does not bypass the answer rule."""
    (tmp_path / "p.jsonl").write_text(json.dumps(line(partition="test", answer="GOLD")) + "\n")
    with pytest.raises(ValueError, match="p.jsonl:1: a test line carries an answer"):
        read_jsonl(tmp_path / "p.jsonl", partitions=("test", "aux_test"))


def test_a_json_error_names_the_file_and_line(tmp_path):
    """A broken JSON line is named by its file and line number."""
    (tmp_path / "p.jsonl").write_text(json.dumps(line()) + "\n{broken\n")
    with pytest.raises(ValueError, match=r"p\.jsonl:2: "):
        read_jsonl(tmp_path / "p.jsonl")


def test_a_file_that_is_not_utf8_names_itself(tmp_path):
    """The strict decode error carries the path (a torn multibyte tail of a crashed run)."""
    from v2gpu.examples import jsonl_lines

    (tmp_path / "torn.jsonl").write_bytes("{\"text\": \"é".encode()[:-1])
    with pytest.raises(ValueError, match=r"torn\.jsonl: not UTF-8"):
        jsonl_lines(tmp_path / "torn.jsonl")


@pytest.mark.parametrize("stray", [".DS_Store", "._train.F0.jsonl", "sub/.DS_Store"])
def test_finder_litter_is_ignored_explicitly_and_noted(tmp_path, stray):
    """macOS .DS_Store and ._* files in a prompt directory no longer refuse it opaquely:
    they are ignored and named in the result (the receipts carry it); any other stray file is still refused."""
    ex = build(RECS, "mistral", None, RECS, compose_prompt, gold_answer)
    write_jsonl(tmp_path / "train.F0.jsonl", ex[:3])
    (tmp_path / "PROMPTS_META.json").write_text("{}")
    sums = sums_for(tmp_path, "train.F0.jsonl", "PROMPTS_META.json")
    (tmp_path / stray).parent.mkdir(exist_ok=True)
    (tmp_path / stray).write_bytes(b"\x00\x05\x16\x07")
    problems, meta = prompt_files([tmp_path / "train.F0.jsonl"], sums)
    assert problems == [] and meta["ignored"] == [stray]
    (tmp_path / "DS_Store").write_text("x")
    assert prompt_files([tmp_path / "train.F0.jsonl"], sums)[0] == [
        "DS_Store: in the prompt directory but not listed in PROMPTS_SHA256SUMS"]
