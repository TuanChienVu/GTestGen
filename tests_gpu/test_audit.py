import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from v2gpu.audit import main
from v2gpu.examples import Example, pool_digest, write_jsonl
from v2gpu.inputs import SPECIAL_TOKENS
from v2gpu.models import MODELS

REPO = "Sylius/Sylius"
V0 = {"variant": "v0", "k": 8, "n_examples": 3, "ratio": 0.5, "budget_examples": 1536, "budget_knowledge": 1536,
      "mode": "full"}                               # 3A's META compose of the v0 directory


def code_for(rid):
    return f"#[Given('step {rid}')]\npublic function step{rid.capitalize()}(): void {{ $this->shop->make('{rid}'); }}"


def rec(rid, partition, feature=None, neighbours=None):
    r = {"id": rid, "repo": REPO, "language": "php", "partition": partition, "subsets": [],
         "feature_path": feature or f"features/{rid}.feature", "suite": "s", "tags": [], "scoping": None,
         "scenario_time": 100, "t_vis": 150, "background": "", "scenario": f"Given step {rid}", "steps": [f"step {rid}"],
         "create_def_ids": [f"{rid}.php:1@1"],
         "create": [{"fn_id": f"{rid}.php:1", "path": f"{rid}.php", "code": code_for(rid), "preexisting": False}]}
    return {**r, "neighbours": neighbours or []} if partition == "train" else r


TRAIN = [rec("t1", "train"), rec("t2", "train")]
QUERIES = [rec("x1", "test"), rec("x2", "test"), rec("a1", "aux_test")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prov(**kw):
    return {"examples": [], "knowledge": [], "skeleton": [], "dropped": [], **kw}


FORBIDDEN = {REPO: {"test": ["x1.php:1", "x2.php:1"], "aux": ["a1.php:1"]}}


def setup(tmp_path, queries=QUERIES, text=lambda r, fmt: f"[{fmt}] {r['scenario']}", answer=None, skip=(),
          excluded=(), pools=None, train=TRAIN, train_text=None, forbidden=FORBIDDEN, train_prov=None):
    """train_text: also write the train records' prompts (train.deepseek.jsonl, with answers); a dev
    query adds records/dev.jsonl to --records and to the seal. forbidden: the seal's forbidden.json,
    e.g. with dev answers; train_prov(r, fmt): the provenance of a train prompt line (default: empty lists)."""
    TRAIN = train                                  # noqa: N806 - the records of this setup
    p2 = tmp_path / "phase2"
    (p2 / "records").mkdir(parents=True)
    held = [("records/test.jsonl", "test"), ("records/aux_test.jsonl", "aux_test")]
    held += [("records/dev.jsonl", "dev")] if any(q["partition"] == "dev" for q in queries) else []
    for name, rows in (("records/train.jsonl", TRAIN), *((n, [q for q in queries if q["partition"] == part])
                                                         for n, part in held)):
        (p2 / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (p2 / "catalog.jsonl").write_text("".join(json.dumps(
        {"repo": REPO, "fn_id": c["fn_id"], "code": c["code"], "first_line": 1, "suites": ["s"], "catch_all": False,
         "scope_tags": [], "annotations": [{"def_id": f"{c['fn_id']}@1", "line": 1, "pattern": f"step {r['id']}", "time": 150}]})
        + "\n" for r in TRAIN + queries for c in r["create"]))
    (p2 / "forbidden.json").write_text(json.dumps(forbidden))
    (p2 / "PHASE2_SHA256SUMS").write_text("".join(
        f"{sha(p2 / n)}  out/phase2/{n}\n" for n in ("records/train.jsonl", *(n for n, _ in held),
                                                     "catalog.jsonl", "forbidden.json")))
    d = tmp_path / "prompts"
    d.mkdir()
    pool = pool_digest(r["id"] for r in TRAIN if r["id"] not in {e["record_id"] for e in excluded
                                                                  if e["reason"] == "uses_test_infra"})
    exs = [Example(q["id"], f, "deepseek", q["partition"], pool, [{"role": "user", "content": text(q, f)}], answer, prov())
           for q in queries for f in ("F0", "F1", "F2") if (q["id"], f) not in skip]
    lines = [f"{write_jsonl(d / 'test.deepseek.jsonl', exs)}  test.deepseek.jsonl"]
    trains = [] if train_text is None else [
        Example(t["id"], f, "deepseek", "train", pool, [{"role": "user", "content": train_text(t, f)}], "GOLD",
                train_prov(t, f) if train_prov else prov())
        for t in TRAIN for f in ("F0", "F1", "F2")]
    if trains:
        lines.append(f"{write_jsonl(d / 'train.deepseek.jsonl', trains)}  train.deepseek.jsonl")
    (d / "excluded.jsonl").write_text("".join(json.dumps(e) + "\n" for e in excluded))
    (d / "infra.json").write_text("{}")
    infra_users = {e["record_id"] for e in excluded if e["reason"] == "uses_test_infra"}
    ids = sorted(r["id"] for r in TRAIN if r["id"] not in infra_users)
    (d / "pools.json").write_text(json.dumps(pools or {"all": {"sha256": pool_digest(ids), "ids": ids}}))
    (d / "PROMPTS_META.json").write_text(json.dumps({"v2tools_commit": "3" * 40,          # 40-hex
                                                      "phase2_sums_sha256": sha(p2 / "PHASE2_SHA256SUMS"),
                                                      "symtimes_sha256": "t", "compose": V0,
                                                      "files": sorted(x.split("  ")[1] for x in lines),
                                                      "special_tokens_sha256": sha(SPECIAL_TOKENS),
                                                      "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}}))
    lines += [f"{sha(d / n)}  {n}" for n in ("excluded.jsonl", "infra.json", "pools.json", "PROMPTS_META.json")]
    (d / "PROMPTS_SHA256SUMS").write_text("\n".join(lines) + "\n")
    return ["--model", "deepseek", "--subset", "all", "--prompts", str(d / "test.deepseek.jsonl"),
            *([str(d / "train.deepseek.jsonl")] if trains else []),
            "--prompts-sums", str(d / "PROMPTS_SHA256SUMS"), "--records", *(str(p2 / n) for n, _ in held),
            "--train-records", str(p2 / "records/train.jsonl"),
            "--catalog", str(p2 / "catalog.jsonl"), "--forbidden", str(p2 / "forbidden.json"),
            "--phase2-sums", str(p2 / "PHASE2_SHA256SUMS"), "--out", str(tmp_path / "leakcheck-receipt.json")]


INFRA_USER = [{"record_id": "t2", "fmt": None, "model": None, "reason": "uses_test_infra"}]


def test_test_prompts_built_in_their_own_directory_pass(tmp_path, tokenizers):
    """The directory holds only test/aux prompts; its excluded.jsonl still lists every
    uses_test_infra record (a train one here) and pools.json names the cleaned pool."""
    assert main(setup(tmp_path, excluded=INFRA_USER)) == 0
    assert receipt(tmp_path)["status"] == "PASS"


def test_a_directory_whose_excluded_misses_a_train_infra_user_fails_on_pools(tmp_path, tokenizers):
    ids = ["t1"]                                   # compose's pool without t2, but excluded.jsonl forgot t2
    assert main(setup(tmp_path, pools={"all": {"sha256": pool_digest(ids), "ids": ids}})) == 2
    assert any("pools.json pool 'all' differs from the recomputed pool (missing [], extra [])" not in p
               and "pools.json pool 'all' differs" in p for p in receipt(tmp_path)["problems"])


def receipt(tmp_path):
    return json.loads((tmp_path / "leakcheck-receipt.json").read_text())


def test_clean_test_and_aux_prompts_pass_with_a_receipt(tmp_path, tokenizers):
    assert main(setup(tmp_path)) == 0
    r = receipt(tmp_path)
    assert r["status"] == "PASS" and r["n_checked"] == 9 and r["by_partition"] == {"aux_test": 3, "test": 6}
    assert r["prompts"]["sums_sha256"] == sha(tmp_path / "prompts" / "PROMPTS_SHA256SUMS")
    assert r["inputs"][str(tmp_path / "phase2" / "forbidden.json")] == sha(tmp_path / "phase2" / "forbidden.json")


def test_a_test_prompt_showing_its_answer_fails(tmp_path, tokenizers):
    leaky = lambda r, fmt: f"[{fmt}] {r['scenario']}\n{r['create'][0]['code']}" if r["id"] == "x2" else f"[{fmt}] x"
    assert main(setup(tmp_path, text=leaky)) == 2
    r = receipt(tmp_path)
    assert r["status"] == "FAIL" and any("x2: answer code of x2.php:1" in p for p in r["problems"])


def test_a_test_line_with_an_answer_is_refused(tmp_path, tokenizers):
    assert main(setup(tmp_path, answer="GOLD CODE")) == 2
    assert any("carries an answer" in p for p in receipt(tmp_path)["problems"])


def test_missing_pairs_must_be_listed_in_excluded(tmp_path, tokenizers):
    assert main(setup(tmp_path, skip={("x1", "F2")})) == 2
    assert "missing test prompt x1 F2" in receipt(tmp_path)["problems"]
    listed = [{"record_id": "x1", "fmt": None, "model": None, "reason": "too_long_test_rule"}]
    (tmp_path / "phase2").rename(tmp_path / "old")
    (tmp_path / "prompts").rename(tmp_path / "oldp")
    assert main(setup(tmp_path, skip={("x1", f) for f in ("F0", "F1", "F2")}, excluded=listed)) == 0
    assert receipt(tmp_path)["excluded_by_compose"] == {"test": {"too_long_test_rule": 3}}


def test_a_test_record_linked_to_train_fails(tmp_path, tokenizers):
    queries = [rec("x1", "test", feature="features/t1.feature"), *QUERIES[1:]]
    assert main(setup(tmp_path, queries=queries)) == 2
    assert any("test record x1 is a direct neighbour of train record(s) ['t1']" in p for p in receipt(tmp_path)["problems"])


def test_control_tokens_in_a_test_prompt_fail(tmp_path, tokenizers):
    assert main(setup(tmp_path, text=lambda r, fmt: f"[{fmt}] <｜begin▁of▁sentence｜> {r['scenario']}")) == 2
    assert any("control token" in p for p in receipt(tmp_path)["problems"])


def test_python_m_v2gpu_leakcheck_runs_the_audit():
    out = subprocess.run([sys.executable, "-m", "v2gpu.leakcheck", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--train-records" in out.stdout


def test_a_crash_leaves_no_receipt_behind(tmp_path, tokenizers):
    """--out is deleted before any other work, so an audit that crashes never leaves
    an older PASS receipt behind for a generation job to trust."""
    import pytest

    args = setup(tmp_path)
    assert main(args) == 0 and receipt(tmp_path)["status"] == "PASS"
    args[args.index("--prompts-sums") + 1] = str(tmp_path / "missing" / "PROMPTS_SHA256SUMS")   # the FIRST read (F1)
    with pytest.raises(OSError):
        main(args)
    assert not (tmp_path / "leakcheck-receipt.json").exists()


def test_the_receipt_names_the_code_that_decided_it(tmp_path, tokenizers):
    """code_sha256 = {repo-relative path: sha256} of every module whose code decides PASS."""
    from pathlib import Path

    from v2gpu.audit import AUDIT_CODE

    assert main(setup(tmp_path)) == 0
    code = receipt(tmp_path)["code_sha256"]
    root = Path(__file__).resolve().parents[1]
    assert set(code) == set(AUDIT_CODE) and all(code[f] == sha(root / f) for f in code)
    assert {"v2gpu/leakcheck.py", "v2gpu/audit.py", "v2gpu/examples.py", "v2gpu/inputs.py",
            "v2gpu/serialize.py"} <= set(code)


def test_every_v2gpu_module_the_audit_loads_is_in_its_code_digest():
    from pathlib import Path

    from v2gpu.audit import AUDIT_CODE

    probe = "import sys, v2gpu.audit; print(' '.join(sorted(m for m in sys.modules if m.startswith('v2gpu.'))))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                         text=True, check=True)
    loaded = {"v2gpu/" + m.split(".", 1)[1].replace(".", "/") + ".py" for m in out.stdout.split()}
    assert "v2gpu/leakcheck.py" in loaded and loaded <= set(AUDIT_CODE)


def test_the_catalog_reader_splits_lines_on_newline_only(tmp_path):
    """decision 24/09: catalog code may hold U+2028 or U+0085 (json.dumps(ensure_ascii=False) writes them raw)."""
    from v2gpu.audit import _catalog

    fn = {"repo": "R", "fn_id": "f1", "code": "// a\u2028b\x85c"}
    (tmp_path / "catalog.jsonl").write_text(json.dumps(fn, ensure_ascii=False) + "\n", encoding="utf-8")
    assert _catalog(tmp_path / "catalog.jsonl") == {"R": {"f1": fn}}


def test_an_out_that_aliases_an_input_is_refused_before_anything_is_deleted(tmp_path):
    """--out is unlinked first, so an --out that is an input (or lies in the
    prompt directory) would delete it. argparse refuses such an --out before the unlink."""
    import pytest

    args = setup(tmp_path)
    sums, records = tmp_path / "prompts" / "PROMPTS_SHA256SUMS", Path(args[args.index("--records") + 1])
    kept = {p: p.read_bytes() for p in (sums, records)}
    for target in (sums, records, tmp_path / "prompts" / "receipt.json"):
        args[args.index("--out") + 1] = str(target)
        with pytest.raises(SystemExit) as e:
            main(args)
        assert e.value.code == 2
        assert {p: p.read_bytes() for p in kept} == kept and not (tmp_path / "prompts" / "receipt.json").exists()


# Audit scope (25/09): content checks on every prompt line, train included;
# held-out query checks (train neighbour, membership) on dev/test/aux queries only; --records holds held-out records


def clean(r, fmt):
    return f"[{fmt}] {r['scenario']}"


def test_train_prompts_pass_the_content_checks_and_the_receipt_lists_every_file(tmp_path, tokenizers):
    assert main(setup(tmp_path, train_text=clean)) == 0
    r, d = receipt(tmp_path), tmp_path / "prompts"
    assert r["status"] == "PASS" and r["n_checked"] == 15
    assert r["by_partition"] == {"aux_test": 3, "test": 6, "train": 6}
    assert r["files"]["train.deepseek.jsonl"] == {
        "sha256": sha(d / "train.deepseek.jsonl"), "partitions": ["train"], "prompts": 6, "problems": 0,
        "checks": {"content": 6, "forbidden_code": 6, "control_tokens": 6, "held_out_query": 0}}
    held = r["files"]["test.deepseek.jsonl"]
    assert held["partitions"] == ["aux_test", "test"] and held["prompts"] == 9 and held["problems"] == 0
    assert held["checks"] == {"content": 9, "forbidden_code": 9, "control_tokens": 9, "held_out_query": 9}


@pytest.mark.parametrize("partition", ["train", "dev"])
def test_a_prompt_showing_the_code_of_a_hidden_answer_fails(tmp_path, tokenizers, partition):
    """A test answer (the catalog code of a forbidden function) planted in a train or dev prompt: its provenance does
    not list it, so only the text check against the partition's hidden functions can see it."""
    def leaky(r, fmt):
        return clean(r, fmt) + ("\n" + code_for("x1") if r["id"] in ("t1", "d1") and fmt == "F2" else "")

    queries = [*QUERIES, rec("d1", "dev")] if partition == "dev" else QUERIES
    args = setup(tmp_path, queries=queries, text=leaky, train_text=leaky if partition == "train" else None)
    assert main(args) == 2
    r = receipt(tmp_path)
    rid = "t1" if partition == "train" else "d1"
    assert [p for p in r["problems"] if "x1.php:1" in p] == [
        f"{rid} F2: code of x1.php:1, hidden for partition {partition} (forbidden.json), appears in the prompt"]
    name = "train.deepseek.jsonl" if partition == "train" else "test.deepseek.jsonl"
    assert r["files"][name]["problems"] == 1


def test_a_test_prompt_may_show_another_test_answers_code(tmp_path, tokenizers):
    """Nothing is hidden for a test query (V(r) may hold pre-T code of another test record's answer);
    the audit judges labels, not context."""
    shows = lambda r, fmt: clean(r, fmt) + ("\n" + code_for("x2") if r["id"] == "x1" else "")
    assert main(setup(tmp_path, text=shows)) == 0


def test_records_holding_a_train_record_are_refused_at_the_start(tmp_path, tokenizers, capsys):
    args = setup(tmp_path)
    args.insert(args.index("--records") + 1, str(tmp_path / "phase2" / "records" / "train.jsonl"))
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "--records" in err and "held-out" in err and "t1" in err and "--train-records" in err
    assert not (tmp_path / "leakcheck-receipt.json").exists()


def test_the_train_neighbour_rule_holds_for_held_out_queries_only(tmp_path, tokenizers):
    """Train queries are linked to other train records by construction (the neighbours field); a
    held-out query linked to train still fails."""
    shared = "features/shared.feature"
    train = [rec("t1", "train", feature=shared, neighbours=["t2"]),
             rec("t2", "train", feature=shared, neighbours=["t1"])]
    assert main(setup(tmp_path / "a", train=train, train_text=clean)) == 0
    queries = [*QUERIES, rec("d1", "dev", feature=shared)]
    assert main(setup(tmp_path / "b", train=train, queries=queries, train_text=clean)) == 2
    assert [p for p in receipt(tmp_path / "b")["problems"] if "neighbour" in p] == [
        "dev record d1 is a direct neighbour of train record(s) ['t1', 't2']"]


def test_a_train_prompt_of_a_record_that_is_not_a_train_record_fails(tmp_path, tokenizers):
    args = setup(tmp_path, train_text=clean)
    train_file = tmp_path / "prompts" / "train.deepseek.jsonl"
    rows = [json.loads(x) for x in train_file.read_text().splitlines()]
    train_file.write_text("".join(json.dumps({**x, "record_id": "t9"} if x["record_id"] == "t2" else x) + "\n"
                                  for x in rows))
    lines = [x for x in (tmp_path / "prompts" / "PROMPTS_SHA256SUMS").read_text().splitlines()
             if not x.endswith("train.deepseek.jsonl")] + [f"{sha(train_file)}  train.deepseek.jsonl"]
    (tmp_path / "prompts" / "PROMPTS_SHA256SUMS").write_text("\n".join(lines) + "\n")
    assert main(args) == 2
    assert any("t9" in p and "not a train record" in p for p in receipt(tmp_path)["problems"])


# 25/09: pin the train-line checks that mutants once showed unpinned; the dev answers of the seal are hidden for train lines


DEV_ANSWER = {REPO: {**FORBIDDEN[REPO], "dev": ["d1.php:1"]}}          # d1's CREATE function is a dev answer


def test_the_content_checks_run_on_train_lines(tmp_path, tokenizers):
    """A train prompt showing its own record's answer code fails `content` (leakcheck.check); the forbidden_code
    check cannot see it, since a train record's answer is hidden for nobody."""
    own = lambda r, fmt: clean(r, fmt) + ("\n" + r["create"][0]["code"] if r["id"] == "t1" and fmt == "F1" else "")
    assert main(setup(tmp_path, train_text=own)) == 2
    r = receipt(tmp_path)
    mine = [p for p in r["problems"] if "t1" in p]
    assert "[F1] t1: answer code of t1.php:1 appears in the prompt" in mine
    assert all(p.startswith("[F1] t1: ") for p in mine)                        # leakcheck.check, on the F1 line
    assert r["files"]["train.deepseek.jsonl"]["problems"] == len(mine) == len(r["problems"])


def test_the_control_token_check_runs_on_train_lines(tmp_path, tokenizers):
    """A control token (DeepSeek's BOS as text) in a train prompt fails each of that record's train lines."""
    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    token = lambda r, fmt: clean(r, fmt) if r["id"] != "t2" else f"[{fmt}] {bos} {r['scenario']}"
    assert main(setup(tmp_path, train_text=token)) == 2
    r = receipt(tmp_path)
    assert sorted(p.split(":")[0] for p in r["problems"] if "control token" in p) == ["t2 F0", "t2 F1", "t2 F2"]
    assert r["files"]["train.deepseek.jsonl"]["problems"] == 3 and r["files"]["test.deepseek.jsonl"]["problems"] == 0


def test_a_train_line_showing_a_dev_answers_code_fails(tmp_path, tokenizers):
    """The forbidden_code check on a train line uses the TRAIN hidden set, which holds the dev answers (dev's hidden set
    would not)."""
    leaky = lambda r, fmt: clean(r, fmt) + ("\n" + code_for("d1") if r["id"] == "t1" and fmt == "F2" else "")
    args = setup(tmp_path, queries=[*QUERIES, rec("d1", "dev")], train_text=leaky, forbidden=DEV_ANSWER)
    assert main(args) == 2
    assert [p for p in receipt(tmp_path)["problems"] if "d1.php:1" in p] == [
        "t1 F2: code of d1.php:1, hidden for partition train (forbidden.json), appears in the prompt"]


def test_a_train_line_whose_provenance_shows_a_dev_answer_fails(tmp_path, tokenizers):
    """The content check on a train line runs leakcheck.check with the train hidden set (dev answers included), never with
    an empty one: ④(b) listing the dev answer d1.php:1 fails."""
    shows = lambda r, fmt: prov(knowledge=["d1.php:1"]) if r["id"] == "t1" and fmt == "F2" else prov()
    args = setup(tmp_path, queries=[*QUERIES, rec("d1", "dev")], train_text=clean, forbidden=DEV_ANSWER,
                 train_prov=shows)
    assert main(args) == 2
    assert any(p.startswith("[F2]") and "d1.php:1" in p and "hidden for partition train" in p
               for p in receipt(tmp_path)["problems"]), receipt(tmp_path)["problems"]
