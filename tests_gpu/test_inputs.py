import hashlib
import json

from v2gpu.inputs import SPECIAL_TOKENS, catalog_problems, main, meta_problems, phase2_problems, strip_catalog, \
    unstripped_problems
from v2gpu.models import MODELS

CATALOG = [{"repo": "R", "fn_id": f, "code": f"code of {f}", "first_line": 1, "suites": ["s"],
            "annotations": [{"def_id": f"{f}@1", "line": 1, "pattern": f"p {f}", "time": 1}]}
           for f in ("train.php:1", "dev.php:1", "test.php:1", "aux.php:1")]
FORBIDDEN = {"R": {"test": ["test.php:1"], "dev": ["dev.php:1"], "aux": ["aux.php:1"]}}


def seal(tmp_path):
    """A Phase 2 directory whose PHASE2_SHA256SUMS lists paths as the Phase 2 build writes them (out/phase2/...)."""
    d = tmp_path / "phase2"
    (d / "records").mkdir(parents=True)
    files = {"catalog.jsonl": "".join(json.dumps(r) + "\n" for r in CATALOG), "forbidden.json": json.dumps(FORBIDDEN),
             "records/train.jsonl": '{"id": "t"}\n', "records/dev.jsonl": '{"id": "d"}\n'}
    for name, text in files.items():
        (d / name).write_text(text)
    (d / "PHASE2_SHA256SUMS").write_text("".join(f"{hashlib.sha256(t.encode()).hexdigest()}  out/phase2/{n}\n"
                                                 for n, t in files.items()))
    return d


def test_phase2_inputs_are_checked_against_the_seal(tmp_path):
    d = seal(tmp_path)
    paths = [d / "records/train.jsonl", d / "records/dev.jsonl", d / "catalog.jsonl", d / "forbidden.json"]
    assert phase2_problems(paths, d / "PHASE2_SHA256SUMS") == []
    (d / "forbidden.json").write_text("{}")                             # an older copy
    (d / "extra.jsonl").write_text("x")
    problems = phase2_problems(paths + [d / "extra.jsonl"], d / "PHASE2_SHA256SUMS")
    assert any("forbidden.json: sha256 differs" in p for p in problems)
    assert any("extra.jsonl: not listed" in p for p in problems)


def test_the_training_catalog_drops_the_code_of_test_and_aux_answers(tmp_path):
    d = seal(tmp_path)
    receipt = strip_catalog(d / "catalog.jsonl", d / "forbidden.json", d / "catalog.train.jsonl")
    rows = {r["fn_id"]: r for r in map(json.loads, (d / "catalog.train.jsonl").read_text().splitlines())}
    assert "code" not in rows["test.php:1"] and "code" not in rows["aux.php:1"]
    assert rows["dev.php:1"]["code"] == "code of dev.php:1" and rows["train.php:1"]["code"] == "code of train.php:1"
    assert rows["test.php:1"]["annotations"] and rows["test.php:1"]["suites"] == ["s"]
    assert receipt["stripped"] == 2
    sums = d / "PHASE2_SHA256SUMS"
    assert phase2_problems([d / "records/train.jsonl", d / "forbidden.json"], sums) == []
    assert catalog_problems(d / "catalog.train.jsonl", sums) == []          # derived from the seal through its receipt
    (d / "catalog.train.jsonl").write_text((d / "catalog.train.jsonl").read_text() + "\n")
    assert any("catalog.train.jsonl: sha256 differs from its receipt" in p
               for p in catalog_problems(d / "catalog.train.jsonl", sums))


def test_a_stripped_catalog_from_another_source_is_refused(tmp_path):
    d = seal(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "catalog.jsonl").write_text((d / "catalog.jsonl").read_text().replace("code of", "CODE OF"))
    strip_catalog(other / "catalog.jsonl", d / "forbidden.json", d / "catalog.train.jsonl")
    assert any("not derived from the sealed catalog.jsonl" in p
               for p in catalog_problems(d / "catalog.train.jsonl", d / "PHASE2_SHA256SUMS"))


def test_a_stripped_catalog_saved_under_the_sealed_name_is_accepted_via_its_receipt(tmp_path):
    """The pod may receive the stripped catalog shipped as catalog.jsonl (the sealed name, still listed with the
    unstripped digest); a receipt beside it must still be honoured."""
    d = seal(tmp_path)
    sums = d / "PHASE2_SHA256SUMS"
    tmp_out = tmp_path / "stripped.jsonl"
    strip_catalog(d / "catalog.jsonl", d / "forbidden.json", tmp_out)
    (d / "catalog.jsonl").write_bytes(tmp_out.read_bytes())
    (d / "catalog.jsonl.receipt.json").write_bytes(tmp_out.with_name(tmp_out.name + ".receipt.json").read_bytes())
    assert catalog_problems(d / "catalog.jsonl", sums) == []


def test_a_hand_made_receipt_does_not_help_an_unlisted_non_catalog_file_pass(tmp_path):
    """phase2_problems no longer has a receipt route at all: records, forbidden and any other file can only pass by
    being listed in the seal, whatever receipt sits beside them."""
    d = seal(tmp_path)
    fake = tmp_path / "records_copy.jsonl"
    fake.write_text((d / "records/train.jsonl").read_text())
    fake.with_name(fake.name + ".receipt.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256((d / "catalog.jsonl").read_bytes()).hexdigest(),
        "forbidden_sha256": hashlib.sha256((d / "forbidden.json").read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(fake.read_bytes()).hexdigest()}))
    assert any("records_copy.jsonl: not listed" in p for p in phase2_problems([fake], d / "PHASE2_SHA256SUMS"))


def test_catalog_problems_refuses_a_receipt_missing_a_required_key(tmp_path):
    d = seal(tmp_path)
    out = d / "catalog.train.jsonl"
    strip_catalog(d / "catalog.jsonl", d / "forbidden.json", out)
    receipt_path = out.with_name(out.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    del receipt["source_sha256"]
    receipt_path.write_text(json.dumps(receipt))
    assert any("source_sha256" in p for p in catalog_problems(out, d / "PHASE2_SHA256SUMS"))


def test_unstripped_problems_flags_forbidden_functions_that_still_carry_code(tmp_path):
    d = seal(tmp_path)
    forbidden = json.loads((d / "forbidden.json").read_text())
    assert unstripped_problems(d / "catalog.jsonl", forbidden) == [
        "catalog.jsonl: holds the code of 2 test/aux answer function(s), e.g. test.php:1; ship the output of "
        "python -m v2gpu.inputs strip-catalog"]
    out = d / "catalog.train.jsonl"
    strip_catalog(d / "catalog.jsonl", d / "forbidden.json", out)
    assert unstripped_problems(out, forbidden) == []


def test_strip_catalog_cli_verifies_the_seal_first(tmp_path, capsys):
    d = seal(tmp_path)
    assert main(["strip-catalog", "--phase2-sums", str(d / "PHASE2_SHA256SUMS"), "--out", str(tmp_path / "c.jsonl")]) == 0
    assert json.loads((tmp_path / "c.jsonl.receipt.json").read_text())["stripped"] == 2
    (d / "catalog.jsonl").write_text("tampered\n")
    assert main(["strip-catalog", "--phase2-sums", str(d / "PHASE2_SHA256SUMS"), "--out", str(tmp_path / "e.jsonl")]) == 2
    assert not (tmp_path / "e.jsonl").exists()


def test_meta_names_the_seal_and_the_pinned_tokenizers(tmp_path):
    d = seal(tmp_path)
    sums = d / "PHASE2_SHA256SUMS"
    good = {"v2tools_commit": "37ed1a23c78b0c0baa0b509a944b91dc8ac9f063",
            "phase2_sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
            "symtimes_sha256": "s", "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}, "compose": {},
            "special_tokens_sha256": hashlib.sha256(SPECIAL_TOKENS.read_bytes()).hexdigest()}
    assert meta_problems(good, sums) == [] and meta_problems(good, None) == []
    assert any("phase2_sums_sha256" in p for p in meta_problems(dict(good, phase2_sums_sha256="0" * 64), sums))
    revs = dict(good["tokenizer_revisions"], mistral="main")
    assert any("mistral" in p for p in meta_problems(dict(good, tokenizer_revisions=revs), sums))
    assert any("tokenizer_revisions" in p for p in meta_problems({"phase2_sums_sha256": good["phase2_sums_sha256"]}, sums))
    no_sym = {k: v for k, v in good.items() if k != "symtimes_sha256"}
    assert meta_problems(no_sym, sums) == ["PROMPTS_META.json lacks key(s) ['symtimes_sha256']"]
    assert any("special_tokens_sha256" in p for p in meta_problems(dict(good, special_tokens_sha256="0" * 64), sums))


def test_the_catalog_tools_split_lines_on_newline_only(tmp_path):
    """decision 24/09: strip-catalog and its check read the catalog as bytes split on b"\\n"."""
    from v2gpu.inputs import unstripped_problems

    rows = [{"repo": "R", "fn_id": "t1", "code": "x\u2028y"}, {"repo": "R", "fn_id": "k1", "code": "k\x85z"}]
    cat = tmp_path / "catalog.jsonl"
    cat.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    forbidden = {"R": {"test": ["t1"], "aux": []}}
    (tmp_path / "forbidden.json").write_text(json.dumps(forbidden))
    assert "t1" in unstripped_problems(cat, forbidden)[0]
    strip_catalog(cat, tmp_path / "forbidden.json", tmp_path / "out.jsonl")
    out = [json.loads(x) for x in (tmp_path / "out.jsonl").read_bytes().split(b"\n") if x]
    assert out == [{"repo": "R", "fn_id": "t1"}, rows[1]]
    assert unstripped_problems(tmp_path / "out.jsonl", forbidden) == []


def test_meta_names_the_3a_commit_as_a_full_sha(tmp_path):
    """v2tools_commit is the 40-hex commit 3A built from, never a free label."""
    d = seal(tmp_path)
    sums = d / "PHASE2_SHA256SUMS"
    good = {"v2tools_commit": "37ed1a23c78b0c0baa0b509a944b91dc8ac9f063",
            "phase2_sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(), "symtimes_sha256": "s",
            "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}, "compose": {},
            "special_tokens_sha256": hashlib.sha256(SPECIAL_TOKENS.read_bytes()).hexdigest()}
    assert meta_problems(good, sums) == []
    for bad in ("t", "37ed1a2", "37ED1A23C78B0C0BAA0B509A944B91DC8AC9F063", "37ed1a23c78b0c0baa0b509a944b91dc8ac9f06z",
                None):
        assert any("v2tools_commit" in p for p in meta_problems(dict(good, v2tools_commit=bad), sums)), bad


def test_only_the_training_canarys_synthetic_prompts_may_name_no_3a_commit(tmp_path):
    """v2gpu.canary writes synthetic prompts with v2tools_commit None (no 3A build) and runs
    train_job in canary mode; that None is accepted there only. A free label is refused everywhere."""
    d = seal(tmp_path)
    sums = d / "PHASE2_SHA256SUMS"
    meta = {"v2tools_commit": None, "phase2_sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
            "symtimes_sha256": "s", "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}, "compose": {},
            "special_tokens_sha256": hashlib.sha256(SPECIAL_TOKENS.read_bytes()).hexdigest()}
    assert meta_problems(meta, sums, synthetic=True) == []
    assert any("v2tools_commit" in p for p in meta_problems(meta, sums))
    assert any("v2tools_commit" in p for p in meta_problems(dict(meta, v2tools_commit="t"), sums, synthetic=True))
