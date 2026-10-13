import argparse
import hashlib
import json
import subprocess
import sys

import pytest

from tests_gpu.tiny import make_tiny_llama
from v2gpu.examples import build, pool_digest, write_jsonl
from v2gpu.inputs import SPECIAL_TOKENS
from v2gpu.models import MODELS
from v2gpu.receipt import tensor_digest
from v2gpu.train_job import JOB_LOCK, check_attention, locked_violations, main

REPO = "Sylius/Sylius"


def code_for(i, answer=None):
    return answer or f"#[Given('step {i}')]\npublic function step{i}(): void {{ }}"


def rec(i, partition="train", subsets=(), preexisting=False, answer=None):
    r = {"id": f"{partition[0]}{i}", "repo": REPO, "language": "php", "partition": partition,
         "feature_path": f"features/{partition}{i}.feature", "subsets": list(subsets), "suite": "s", "tags": [],
         "scoping": None, "scenario_time": 100, "t_vis": 150, "background": "", "steps": [f"step number {i}"],
         "scenario": f"Given step number {i}", "create_def_ids": [f"{partition}F.php:{i}@1"],
         "create": [{"fn_id": f"{partition}F.php:{i}", "path": f"{partition}F.php", "code": code_for(i, answer),
                     "preexisting": preexisting}]}
    return {**r, "neighbours": []} if partition == "train" else r


TRAIN = [rec(i, subsets=["lc600"] if i < 2 else []) for i in range(6)]
DEV = [rec(i, "dev") for i in range(2)]


def catalog_rows(records, extra=()):
    rows = [{"repo": REPO, "fn_id": c["fn_id"], "code": c["code"], "first_line": 1, "suites": ["s"], "catch_all": False,
             "scope_tags": [], "annotations": [{"def_id": f"{c['fn_id']}@1", "line": 1, "pattern": f"step {r['id']}",
                                                "time": 150}]}
            for r in records for c in r["create"]]
    return rows + list(extra)


def gold(r):
    return "\n\n".join(c["code"] for c in r["create"])


def prov(**kw):
    return {"examples": [], "knowledge": [], "skeleton": [], "dropped": [], **kw}


def prompt(r, fmt, tok, pool):
    return [{"role": "user", "content": f"[{fmt}] {r['scenario']}"}], prov()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(tmp_path, tokenizers, train=TRAIN, dev=DEV, compose=prompt, pool=None, model="codellama", forbidden=None,
          catalog_extra=(), excluded=(), meta=None, infra=None, answer=None, pools=None):
    """Phase 2 inputs sealed by a PHASE2_SHA256SUMS (paths as the Phase 2 build writes them); the prompt directory
    prompts/ holds the
    prompt files, excluded.jsonl, infra.json, pools.json, META and the SUMS listing all of them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (tmp_path / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dev))
    (tmp_path / "catalog.jsonl").write_text("".join(json.dumps(r) + "\n" for r in catalog_rows(train + dev, catalog_extra)))
    (tmp_path / "forbidden.json").write_text(json.dumps(forbidden or {}))
    (tmp_path / "PHASE2_SHA256SUMS").write_text("".join(f"{sha(tmp_path / n)}  out/phase2/{n}\n" for n in (
        "train.jsonl", "dev.jsonl", "catalog.jsonl", "forbidden.json")))
    pool = train if pool is None else pool
    tok = tokenizers["codellama"]
    P = tmp_path / "prompts"
    P.mkdir(exist_ok=True)
    write_jsonl(P / "train.prompts.jsonl", build(train, model, tok, pool, compose, answer or gold))
    write_jsonl(P / "dev.prompts.jsonl", build(dev, model, tok, pool, compose, answer or gold))
    (P / "excluded.jsonl").write_text("".join(json.dumps(e) + "\n" for e in excluded))
    (P / "infra.json").write_text(json.dumps(infra or {}))
    ids = {"all": sorted(r["id"] for r in pool)} | {s: sorted(r["id"] for r in pool if s in r["subsets"])
                                                    for s in ("lc600", "lc1200")}
    (P / "pools.json").write_text(json.dumps(pools or {k: {"sha256": pool_digest(v), "ids": v} for k, v in ids.items()}))
    (P / "PROMPTS_META.json").write_text(json.dumps(meta or {
        "v2tools_commit": "7e57" * 10, "compose": {"k": 8}, "phase2_sums_sha256": sha(tmp_path / "PHASE2_SHA256SUMS"),
        "symtimes_sha256": "test", "special_tokens_sha256": sha(SPECIAL_TOKENS),
        "tokenizer_revisions": {k: m.revision for k, m in MODELS.items()}}))
    resum(tmp_path)
    tiny = tmp_path / "tiny"
    if not tiny.exists():
        make_tiny_llama(tiny, len(tok))
    return ["--model", "codellama", "--seed", "42", "--records", str(tmp_path / "train.jsonl"),
            "--dev-records", str(tmp_path / "dev.jsonl"), "--catalog", str(tmp_path / "catalog.jsonl"),
            "--forbidden", str(tmp_path / "forbidden.json"), "--train-prompts", str(P / "train.prompts.jsonl"),
            "--dev-prompts", str(P / "dev.prompts.jsonl"), "--prompts-sums", str(P / "PROMPTS_SHA256SUMS"),
            "--phase2-sums", str(tmp_path / "PHASE2_SHA256SUMS"),
            "--mode", "test", "--model-path", str(tiny), "--quant", "none", "--dtype", "float32", "--grad-accum", "2",
            "--max-seq", "512", "--lr", "1e-2", "--no-gradient-checkpointing"]


def resum(tmp_path):
    """PROMPTS_SHA256SUMS over every file of the prompt directory (the builder's last step)."""
    P = tmp_path / "prompts"
    (P / "PROMPTS_SHA256SUMS").write_text("".join(f"{sha(f)}  {f.name}\n" for f in sorted(P.iterdir())
                                                  if f.name != "PROMPTS_SHA256SUMS"))


def rewrite_prompts(tmp_path, name, examples):
    """Replace one prompt file and keep PROMPTS_SHA256SUMS consistent (a builder that made those prompts)."""
    write_jsonl(tmp_path / "prompts" / name, examples)
    resum(tmp_path)


def receipt(tmp_path):
    return json.loads((tmp_path / "run" / "receipt-train.json").read_text())


def job(tmp_path, args, *extra):
    return main(args + ["--subset", "all", "--workdir", str(tmp_path / "run"), *extra])


def test_full_tiny_job_writes_a_complete_receipt(tmp_path, tokenizers):
    assert job(tmp_path, files(tmp_path, tokenizers)) == 0
    r = receipt(tmp_path)
    assert r["status"] == "DONE" and r["data"]["n_records"] == 6 and r["mode"] == "test"
    assert r["model_load"]["attn_implementation_loaded"] == "sdpa" and r["model_load"]["flash_kernel_eligible"] is None
    assert "fp32 internally" in r["model_load"]["qlora_deviation"]
    assert r["data"]["prompts"]["meta"]["v2tools_commit"] == "7e57" * 10 and len(r["data"]["schedule_sha256"]) == 64
    assert r["preflight"]["examples_checked"] == 6 * 3 + 2 * 3
    assert r["preflight"]["excluded_rule_d"] == {"dev": {}, "train": {}}
    assert r["selected_adapter"]["tensor_sha256"] == tensor_digest(tmp_path / "run" / r["selected_adapter"]["path"])


def test_rule_d_excludes_and_counts_instead_of_refusing(tmp_path, tokenizers):
    long = "#[Given('long')]\npublic function x() { // " + "x " * 600 + "}"
    train = [rec(1, answer=long) if i == 1 else rec(i) for i in range(6)]
    assert job(tmp_path, files(tmp_path, tokenizers, train=train)) == 0
    r = receipt(tmp_path)
    assert r["preflight"]["excluded_rule_d"]["train"] == {"F0": 1, "F1": 1, "F2": 1}
    assert r["result"]["skipped_item_epochs"] >= 3


def test_learning_curve_subset_uses_its_own_pool(tmp_path, tokenizers):
    lc = [r for r in TRAIN if "lc600" in r["subsets"]]
    args = files(tmp_path, tokenizers, pool=lc)
    rewrite_prompts(tmp_path, "train.prompts.jsonl", build(lc, "codellama", None, lc, prompt, gold))
    assert main(args + ["--subset", "lc600", "--workdir", str(tmp_path / "run")]) == 0
    assert receipt(tmp_path)["data"]["n_records"] == 2


def test_prompts_built_with_the_wrong_pool_are_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    lc = [r for r in TRAIN if "lc600" in r["subsets"]]
    rewrite_prompts(tmp_path, "train.prompts.jsonl", build(lc, "codellama", None, TRAIN, prompt, gold))
    assert main(args + ["--subset", "lc600", "--workdir", str(tmp_path / "run")]) == 2
    assert any("retrieval pool" in p for p in receipt(tmp_path)["problems"])


def test_prompts_for_another_model_are_refused(tmp_path, tokenizers):
    assert job(tmp_path, files(tmp_path, tokenizers, model="mistral")) == 2
    assert any("built for model 'mistral'" in p for p in receipt(tmp_path)["problems"])


def test_prompt_files_that_do_not_match_their_sums_are_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    with (tmp_path / "prompts" / "train.prompts.jsonl").open("a") as f:
        f.write("")
    (tmp_path / "prompts" / "PROMPTS_SHA256SUMS").write_text("0" * 64 + "  train.prompts.jsonl\n")
    assert job(tmp_path, args) == 2
    p = receipt(tmp_path)["problems"]
    assert any("sha256 differs" in x for x in p) and any("dev.prompts.jsonl: not listed" in x for x in p)


def test_prompt_holding_a_new_create_function_is_refused(tmp_path, tokenizers):
    def leaky(r, fmt, tok, pool):
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']}\nExample:\n{gold(r)}"}], prov()

    assert job(tmp_path, files(tmp_path, tokenizers, compose=leaky)) == 2
    r = receipt(tmp_path)
    assert r["n_problems"] == 8 * 3 * 4          # the code, its annotation line, its signature line, its method name
    for kind in ("answer code", "annotation of CREATE pattern", "signature line", "method name step0"):
        assert any(kind in p for p in r["problems"]), kind


def shows_old(r, fmt, tok, pool):
    """④ holds the record's preexisting function without its CREATE annotation (line 1), as V(r) renders it."""
    body = gold(r).split("\n", 1)[1]
    return [{"role": "user", "content": f"[{fmt}] {r['scenario']}\nKnown:\n{body}"}], prov()


def test_preexisting_function_bodies_may_be_shown(tmp_path, tokenizers):
    train = [rec(i, preexisting=True) for i in range(6)]
    dev = [rec(i, "dev", preexisting=True) for i in range(2)]
    assert job(tmp_path, files(tmp_path, tokenizers, train=train, dev=dev, compose=shows_old)) == 0


def test_a_preexisting_function_that_keeps_its_create_annotation_is_refused(tmp_path, tokenizers):
    def keeps_annotation(r, fmt, tok, pool):
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']}\nKnown:\n{gold(r)}"}], prov()

    train = [rec(i, preexisting=True) for i in range(6)]
    dev = [rec(i, "dev", preexisting=True) for i in range(2)]
    assert job(tmp_path, files(tmp_path, tokenizers, train=train, dev=dev, compose=keeps_annotation)) == 2
    p = receipt(tmp_path)["problems"]
    assert len(p) == 8 * 3 and all("annotation of CREATE pattern" in x for x in p)


def test_hidden_knowledge_and_foreign_examples_are_refused(tmp_path, tokenizers):
    hidden_fn = {"repo": REPO, "fn_id": "Test.php:1", "code": "public function t() { }", "first_line": 1, "suites": ["s"],
                 "annotations": [{"def_id": "Test.php:1@1", "line": 1, "pattern": "t", "time": 50}]}

    def bad(r, fmt, tok, pool):
        leaky = prov(examples=["d0"], knowledge=["Test.php:1"]) if fmt == "F2" else prov()
        return [{"role": "user", "content": r["scenario"]}], leaky

    args = files(tmp_path, tokenizers, compose=bad, forbidden={REPO: {"test": ["Test.php:1"]}}, catalog_extra=[hidden_fn])
    assert job(tmp_path, args) == 2
    p = "\n".join(receipt(tmp_path)["problems"])
    assert "knowledge Test.php:1: hidden for partition train" in p and "example d0: not in the retrieval pool" in p


def test_partition_and_overlap_are_checked(tmp_path, tokenizers):
    train = TRAIN[:5] + [dict(TRAIN[5], partition="test")]
    dev = DEV + [dict(TRAIN[0], partition="dev")]
    assert job(tmp_path, files(tmp_path, tokenizers, train=train, dev=dev)) == 2
    p = receipt(tmp_path)["problems"]
    assert any("partition 'test'" in x for x in p) and any("in both train and dev" in x for x in p)


def test_non_empty_workdir_is_refused_and_left_untouched(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "other.txt").write_text("someone else's run")
    assert job(tmp_path, args) == 2
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["other.txt"]


def test_missing_dev_format_is_refused_before_the_model_loads(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    rewrite_prompts(tmp_path, "dev.prompts.jsonl",
                    [e for e in build(DEV, "codellama", None, TRAIN, prompt, gold) if e.fmt != "F1"])
    assert job(tmp_path, args) == 2
    r = receipt(tmp_path)
    assert "missing dev prompt d0 F1" in r["problems"] and "model_load" not in r


def test_messages_the_template_rejects_are_refused(tmp_path, tokenizers):
    def two_user_turns(r, fmt, tok, pool):
        return [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], prov()

    assert job(tmp_path, files(tmp_path, tokenizers, compose=two_user_turns)) == 2
    r = receipt(tmp_path)
    assert r["n_problems"] == 48 and "model_load" not in r      # the contract, and the chat template, per example
    assert sum("one user turn" in p for p in r["problems"]) == 24


def test_preflight_only_needs_no_workdir(tmp_path, tokenizers, capsys):
    args = files(tmp_path, tokenizers)
    assert main(args + ["--subset", "all", "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out)["examples_checked"] == 24
    rewrite_prompts(tmp_path, "dev.prompts.jsonl", [])
    assert main(args + ["--subset", "all", "--preflight-only"]) == 2


def test_canary_max_steps_exits_4_and_resume_continues(tmp_path, tokenizers):
    canary = [a if a != "test" else "canary" for a in files(tmp_path, tokenizers)]
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    assert receipt(tmp_path)["status"] == "MAX_STEPS" and receipt(tmp_path)["result"]["tokens_per_second"] > 0
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 4
    assert receipt(tmp_path)["result"]["global_step"] == 4
    assert (tmp_path / "run" / "receipt-train.attempt-1.json").exists()
    assert main(canary + ["--subset", "all", "--workdir", str(tmp_path / "x")]) == 2   # canary needs --max-steps


def test_resume_of_a_finished_job_is_refused_and_receipt_kept(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    assert job(tmp_path, args) == 0
    before = (tmp_path / "run" / "receipt-train.json").read_text()
    assert job(tmp_path, args, "--resume") == 2
    assert (tmp_path / "run" / "receipt-train.json").read_text() == before


def test_resume_without_state_is_refused(tmp_path, tokenizers):
    assert job(tmp_path, files(tmp_path, tokenizers), "--resume") == 2


# --resume must compare identity with the
# previous attempt's receipt (model, seed, subset, every data and prompt digest, schedule_sha256, the normalised
# training config), and refuse instead of silently carrying on with different values


def replaced(args, flag, value):
    args = list(args)
    args[args.index(flag) + 1] = value
    return args


def canary_args(tmp_path, tokenizers, **files_kw):
    return [a if a != "test" else "canary" for a in files(tmp_path, tokenizers, **files_kw)]


def test_a_resume_with_another_seed_is_refused(tmp_path, tokenizers):
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    before = (tmp_path / "run" / "receipt-train.json").read_text()
    other_seed = replaced(canary, "--seed", "43")
    assert job(tmp_path, other_seed, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("seed" in m for m in r["mismatches"])
    assert (tmp_path / "run" / "receipt-train.attempt-1.json").read_text() == before   # the state, untouched
    assert json.loads((tmp_path / "run" / "receipt-train.attempt-1.json").read_text())["result"]["global_step"] == 2


def test_a_resume_with_another_lr_is_refused(tmp_path, tokenizers):
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    other_lr = replaced(canary, "--lr", "5e-2")
    assert job(tmp_path, other_lr, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("train_config.lr" in m for m in r["mismatches"])


def test_a_resume_with_another_prompt_file_is_refused(tmp_path, tokenizers):
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    # a byte json.loads ignores (trailing whitespace): every parsed record is identical, so preflight sees no
    # problem of its own; only the prompt file's sha256 (and PROMPTS_SHA256SUMS) changes
    p = tmp_path / "prompts" / "dev.prompts.jsonl"
    p.write_text(p.read_text().rstrip("\n") + " \n")
    resum(tmp_path)
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY"
    assert any(m.startswith("data.prompt_files") or m.startswith("data.prompts") for m in r["mismatches"]), \
        r["mismatches"]


def test_an_identical_resume_proceeds(tmp_path, tokenizers):
    """The other side of the resume identity: an identical resume (same everything but --max-steps, canary's one
    allowed knob) is
    never refused by the identity check."""
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 4
    r = receipt(tmp_path)
    assert r["status"] == "MAX_STEPS" and r["result"]["global_step"] == 4


def test_identity_mismatches_covers_subset_too(tmp_path, tokenizers):
    """Subset was in the receipt compare's list but had no test. A real --subset change on the full CLI
    changes which records are in scope (train prompts built for "all" would then be "outside the subset", a
    REFUSED_PREFLIGHT of its own, before the identity check ever runs), so this pins the identity compare itself,
    the same function train_loop._load_state calls for a real resume."""
    from v2gpu.train_loop import identity_mismatches

    base = {"model": {"key": "codellama"}, "seed": 42, "subset": "all", "data": {}, "train_config": {},
           "train_code_sha256": {"v2gpu/train_job.py": "a" * 64}}
    assert identity_mismatches(base, {**base, "subset": "lc600"}) == ["subset"]
    assert identity_mismatches(base, dict(base)) == []


def test_a_resume_with_another_model_is_refused(tmp_path, tokenizers, monkeypatch):
    """A mutant that dropped 'model' from the compare and survived 9/9 green -- 'model' must have its own test.
    Simulated like the training-code-change test: the receipt's own model dict is not directly settable through
    the CLI without a second model's whole fixture, so `spec` is monkeypatched to look like the wrong model to the
    checkout building the identity, which is exactly what the compare is protecting against."""
    from v2gpu import train_job
    from v2gpu.models import MODELS

    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    real_asdict = train_job.dataclasses.asdict
    monkeypatch.setattr(train_job.dataclasses, "asdict",
                       lambda spec: {**real_asdict(spec), "key": "another-model"} if spec is MODELS["codellama"]
                       else real_asdict(spec))
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("model" in m for m in r["mismatches"])


# Since 27/09 the resume identity is bound to
# the STATE (identity.json in every snapshot), not to a receipt -- a kill (SIGTERM/SIGKILL, a pod loss) leaves state/
# but no receipt, which a receipt-only compare could not reach at all


class Killed(BaseException):
    """Escapes past train_job's `except Exception`, exactly like a signal would: no receipt is ever written."""


def kill_after_saves(monkeypatch, n=1):
    from v2gpu import train_loop

    real, calls = train_loop._save_state, []

    def fake(*a, **kw):
        real(*a, **kw)
        calls.append(1)
        if len(calls) == n:
            raise Killed(f"killed after save #{n}")
    monkeypatch.setattr(train_loop, "_save_state", fake)


def state_shas(workdir):
    d = workdir / "state"
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


def test_a_resume_after_a_kill_with_no_receipt_is_refused(tmp_path, tokenizers, monkeypatch):
    """Attempt 1 killed right after its first state save (state saved, no receipt at all), then
    --resume --seed 43 --lr 5e-2. The receipt compare could not see this (prev_receipt stayed None); the state
    compare does."""
    canary = canary_args(tmp_path, tokenizers)
    kill_after_saves(monkeypatch, n=1)
    with pytest.raises(Killed):
        job(tmp_path, canary, "--max-steps", "2")
    assert not (tmp_path / "run" / "receipt-train.json").exists()
    other = replaced(replaced(canary, "--seed", "43"), "--lr", "5e-2")
    assert job(tmp_path, other, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY"
    assert any("seed" in m for m in r["mismatches"]) and any("train_config.lr" in m for m in r["mismatches"])


def test_a_refused_resume_retried_is_refused_again(tmp_path, tokenizers, monkeypatch):
    """The refused resume above, retried unchanged, must be refused again -- not "fixed" by the
    fact that the refusal itself just wrote a receipt-train.json (REFUSED_RESUME_IDENTITY is never a reference)."""
    canary = canary_args(tmp_path, tokenizers)
    kill_after_saves(monkeypatch, n=1)
    with pytest.raises(Killed):
        job(tmp_path, canary, "--max-steps", "2")
    other = replaced(canary, "--seed", "43")
    before = state_shas(tmp_path / "run")
    assert job(tmp_path, other, "--max-steps", "4", "--resume") == 2
    assert receipt(tmp_path)["status"] == "REFUSED_RESUME_IDENTITY"
    assert job(tmp_path, other, "--max-steps", "4", "--resume") == 2      # retried, unchanged
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("seed" in m for m in r["mismatches"])
    assert state_shas(tmp_path / "run") == before


def test_an_older_attempts_receipt_with_the_latest_attempt_killed_is_refused(tmp_path, tokenizers, monkeypatch):
    """Attempt 1 wrote a receipt (seed 42, MAX_STEPS); attempt 2 (same identity, resumed further)
    was killed with no receipt of its own -- receipt-train.json is archived to attempt-1 and never rewritten. Attempt
    3 (--seed 43) must still be refused: the state's identity (not "the latest receipt file") is the reference."""
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    assert receipt(tmp_path)["status"] == "MAX_STEPS"
    kill_after_saves(monkeypatch, n=1)
    with pytest.raises(Killed):
        job(tmp_path, canary, "--max-steps", "4", "--resume")
    assert not (tmp_path / "run" / "receipt-train.json").exists()
    assert (tmp_path / "run" / "receipt-train.attempt-1.json").exists()
    other = replaced(canary, "--seed", "43")
    assert job(tmp_path, other, "--max-steps", "6", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("seed" in m for m in r["mismatches"])


def test_a_state_with_no_identity_json_is_refused(tmp_path, tokenizers):
    """A state saved before this fix (or one whose identity.json was lost) fails closed instead of resuming blind."""
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    (tmp_path / "run" / "state" / "identity.json").unlink()
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and any("identity" in m for m in r["mismatches"])


def test_kill_window_a_between_identity_and_pos_resume_never_skips_the_compare(tmp_path, tokenizers, monkeypatch):
    """kill-window (a): kill mid-save, between identity.json and pos.json. The half-written state.tmp/ is
    incomplete (no pos.json), so state_dir() falls back to the last COMPLETE snapshot; resuming with the SAME command
    either continues from there or is refused (never silently past the compare)."""
    from v2gpu import train_loop

    canary = canary_args(tmp_path, tokenizers)
    real_save, cut = train_loop._save_state, []

    def cut_after_identity(workdir, model, opt, sched, pos, identity, cuda=False):
        tmp = workdir / "state.tmp"
        import shutil as _shutil
        _shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        (tmp / "identity.json").write_text(json.dumps(identity, sort_keys=True))
        cut.append(1)
        if len(cut) == 1:
            raise Killed("killed mid-save, before pos.json")
        real_save(workdir, model, opt, sched, pos, identity, cuda)

    monkeypatch.setattr(train_loop, "_save_state", cut_after_identity)
    with pytest.raises(Killed):
        job(tmp_path, canary, "--max-steps", "2")
    assert not (tmp_path / "run" / "state").exists() and not (tmp_path / "run" / "receipt-train.json").exists()
    monkeypatch.setattr(train_loop, "_save_state", real_save)
    assert job(tmp_path, canary, "--max-steps", "2", "--resume") == 2   # REFUSED_NO_STATE: no complete state exists
    assert receipt(tmp_path)["status"] == "REFUSED_NO_STATE"


def test_identity_mismatches_monkeypatch_sanity_check(tmp_path, tokenizers, monkeypatch):
    """A sanity check of the monkeypatch only. Patching identity_mismatches() to always return [] only proves the
    patch has an effect on resume (a resume that should be refused now proceeds); it does NOT exercise the real
    source mutants it should catch (one drops the `if mismatches: raise IdentityMismatch(...)` of
    _load_state, the other the whole identity-check block of _load_state) -- under either, this
    monkeypatch changes nothing (identity_mismatches is still called and still returns [] either way, since neither
    mutant touches identity_mismatches itself), so this test would still pass. Those two real mutants are instead
    caught by test_a_resume_after_a_kill_with_no_receipt_is_refused, test_a_refused_resume_retried_is_refused_again,
    test_an_older_attempts_receipt_with_the_latest_attempt_killed_is_refused,
    test_a_state_with_no_identity_json_is_refused,
    test_kill_window_a_between_identity_and_pos_resume_never_skips_the_compare, and every
    test_a_resume_with_another_*_is_refused case: each asserts REFUSED_RESUME_IDENTITY, which either mutant would
    never write (verified manually against real source mutants)."""
    from v2gpu import train_loop

    monkeypatch.setattr(train_loop, "identity_mismatches", lambda saved, current: [])
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    other = replaced(canary, "--seed", "43")
    assert job(tmp_path, other, "--max-steps", "4", "--resume") != 2, \
        "the monkeypatch makes every resume pass its identity check"


# The training receipt gains train_code_sha256 and train_cfg_sha256


def test_the_receipt_names_the_code_and_config_that_decided_it(tmp_path, tokenizers):
    from pathlib import Path

    from v2gpu.train_job import TRAIN_CODE, train_cfg_sha256

    assert job(tmp_path, files(tmp_path, tokenizers)) == 0
    r = receipt(tmp_path)
    root = Path(__file__).resolve().parents[1]
    code = r["train_code_sha256"]
    assert set(code) == set(TRAIN_CODE) and all(code[f] == sha(root / f) for f in code)
    assert {"v2gpu/train_job.py", "v2gpu/train_loop.py", "v2gpu/modeling.py"} <= set(code)
    assert r["train_cfg_sha256"] == train_cfg_sha256(r["train_config"])


def test_every_v2gpu_module_the_training_code_loads_is_in_its_code_digest():
    """The module probe, like audit's: only the modules train_job imports eagerly are provable this way (modeling is
    imported lazily, only when a job actually trains); v2gpu/modeling.py is checked separately, by source, since
    provoking a real load here would mean training a tiny model just for this probe."""
    from pathlib import Path

    from v2gpu.train_job import TRAIN_CODE

    probe = "import sys, v2gpu.train_job; print(' '.join(sorted(m for m in sys.modules if m.startswith('v2gpu.'))))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                         text=True, check=True)
    loaded = {"v2gpu/" + m.split(".", 1)[1].replace(".", "/") + ".py" for m in out.stdout.split()}
    assert "v2gpu/leakcheck.py" in loaded and "v2gpu/train_loop.py" in loaded and loaded <= set(TRAIN_CODE)
    train_job_src = (Path(__file__).resolve().parents[1] / "v2gpu" / "train_job.py").read_text()
    assert "from .modeling import" in train_job_src and "v2gpu/modeling.py" in TRAIN_CODE


def test_a_real_tiny_job_loads_exactly_train_code_plus_engines(tmp_path, tokenizers):
    """Since 27/09: v2gpu/engines.py IS loaded by a real job (receipt.environment() calls
    engines.nvidia_driver), but is deliberately excluded from TRAIN_CODE (it only fills environment.driver, never a
    training decision; freezing it would freeze the generation engine code for a variant's whole training span).
    A real tiny job (not an import-only probe) proves the loaded modules are exactly TRAIN_CODE plus {engines},
    neither more nor less."""
    from pathlib import Path

    from v2gpu.train_job import TRAIN_CODE

    root = Path(__file__).resolve().parents[1]
    args = files(tmp_path, tokenizers) + ["--subset", "all", "--workdir", str(tmp_path / "run")]
    probe = ("import sys\n"
            f"sys.argv = ['train_job'] + {args!r}\n"
            "from v2gpu.train_job import main\n"
            "rc = main(sys.argv[1:])\n"
            "print('RC', rc)\n"
            "print('MODULES', ' '.join(sorted(m for m in sys.modules if m.startswith('v2gpu.'))))\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True, text=True)
    assert "RC 0" in out.stdout.splitlines(), (out.stdout, out.stderr)
    loaded_line = next(l for l in out.stdout.splitlines() if l.startswith("MODULES "))
    loaded = {"v2gpu/" + m.split(".", 1)[1].replace(".", "/") + ".py" for m in loaded_line[len("MODULES "):].split()}
    assert loaded == set(TRAIN_CODE) | {"v2gpu/engines.py"}, loaded


def test_a_resume_after_a_training_code_change_is_refused(tmp_path, tokenizers, monkeypatch):
    """The training-code digest is part of the resume identity too (the receipt compare's list): simulate a code change
    between attempts (real code cannot change mid-test) by making the second call see a different digest."""
    from v2gpu import train_job

    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    monkeypatch.setattr(train_job, "train_code_sha256", lambda root=train_job.ROOT: {"v2gpu/train_job.py": "changed"})
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and "train_code_sha256" in r["mismatches"]


def test_upcast_fp32_is_mandatory_for_a_real_job_but_not_for_canary(tmp_path, tokenizers, capsys):
    """Decision 27/09, after the real A6000 canary: a real job (mode "job", which always
    requires --phase2-sums) without --upcast-fp32 is REFUSED, exit 2, with a clear message; the canary path (mode
    "canary") never checks JOB_LOCK at all, so its own bf16/fp32 upcast comparison (--compare-upcast) is untouched."""
    job_mode = [a if a != "test" else "job" for a in files(tmp_path, tokenizers)]
    assert job(tmp_path, job_mode) == 2
    assert "--upcast-fp32=False (job mode requires True)" in capsys.readouterr().err
    canary = canary_args(tmp_path / "c", tokenizers)
    assert job(tmp_path / "c", canary, "--max-steps", "1") == 4        # canary: unaffected, proceeds to MAX_STEPS


def test_upcast_fp32_is_part_of_the_locked_train_config_and_its_digest():
    """"upcast_fp32 is already part of train_cfg, and therefore of train_cfg_sha256" --
    pinned directly, so a future refactor dropping it from train_config could not silently stop it deciding the one
    digest genqueue/gencoverage require to agree across every adapter of a variant."""
    from v2gpu.train_job import train_cfg_sha256

    base = {"lr": 2e-4, "grad_accum": 16, "max_seq": 8192, "quant": "nf4", "attn": "sdpa", "dtype": "bfloat16",
           "gradient_checkpointing": True, "upcast_fp32": True}
    assert train_cfg_sha256(base) != train_cfg_sha256({**base, "upcast_fp32": False})


def test_gpu_name_joins_the_resume_identity_and_a_card_change_is_refused(tmp_path, tokenizers, monkeypatch):
    """The training GPU name (receipt.environment()["device"]["name"]) joins the resume
    identity; a resume on a different card class is refused like any other identity field (REFUSED_RESUME_IDENTITY).
    Real hardware cannot change mid-test, so environment() is monkeypatched between the two job() calls, exactly as
    the training-code-change test does."""
    from v2gpu import train_job

    real_env = train_job.environment
    canary = canary_args(tmp_path, tokenizers)
    monkeypatch.setattr(train_job, "environment", lambda: {**real_env(), "device": {"name": "NVIDIA A6000"}})
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    monkeypatch.setattr(train_job, "environment", lambda: {**real_env(), "device": {"name": "NVIDIA A100-SXM4-80GB"}})
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 2
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_RESUME_IDENTITY" and "gpu_name" in r["mismatches"]


def test_gpu_name_identical_resume_on_the_same_card_proceeds(tmp_path, tokenizers, monkeypatch):
    """The other side of the card-class check: a resume on the SAME card class is never refused by the identity check."""
    from v2gpu import train_job

    real_env = train_job.environment
    monkeypatch.setattr(train_job, "environment", lambda: {**real_env(), "device": {"name": "NVIDIA A6000"}})
    canary = canary_args(tmp_path, tokenizers)
    assert job(tmp_path, canary, "--max-steps", "2") == 4
    assert job(tmp_path, canary, "--max-steps", "4", "--resume") == 4
    r = receipt(tmp_path)
    assert r["status"] == "MAX_STEPS" and r["result"]["global_step"] == 4


def test_job_mode_locks_the_training_parameters_and_the_seed():
    ns = argparse.Namespace(**JOB_LOCK, seed=42, model="codellama", subset="all")
    assert locked_violations(ns) == []
    ns.max_steps, ns.lr, ns.seed = 3, 1e-3, 7
    assert len(locked_violations(ns)) == 3


def test_job_mode_seed_lock_is_per_model_and_subset():
    ns = argparse.Namespace(**JOB_LOCK, seed=7, model="deepseek", subset="all")
    assert locked_violations(ns) == []
    ns.seed = 99
    assert locked_violations(ns) == []
    ns.subset = "lc600"
    assert len(locked_violations(ns)) == 1
    ns.model, ns.subset, ns.seed = "mistral", "all", 7
    assert len(locked_violations(ns)) == 1
    ns.model, ns.seed = "deepseek", 5
    assert len(locked_violations(ns)) == 1


def test_allowed_seeds_extends_only_for_deepseek_on_the_full_train_set():
    from v2gpu.models import DEEPSEEK_EXTRA_SEEDS, SEEDS, allowed_seeds

    assert allowed_seeds("deepseek", "all") == SEEDS + DEEPSEEK_EXTRA_SEEDS
    assert allowed_seeds("deepseek", "lc600") == SEEDS
    assert allowed_seeds("deepseek", "lc1200") == SEEDS
    assert allowed_seeds("mistral", "all") == SEEDS
    assert allowed_seeds("codellama", "all") == SEEDS


def test_job_mode_refuses_overrides(tmp_path, tokenizers):
    args = [a if a != "test" else "job" for a in files(tmp_path, tokenizers)]
    assert job(tmp_path, args) == 2 and not (tmp_path / "run").exists()


def test_attention_fallback_exits_3_with_a_receipt(tmp_path, tokenizers, monkeypatch):
    from v2gpu import modeling

    real = modeling.load_for_training

    def eager(cfg, seed):
        model, info = real(cfg, seed)
        return model, dict(info, attn_implementation_loaded="eager")

    monkeypatch.setattr(modeling, "load_for_training", eager)
    assert job(tmp_path, files(tmp_path, tokenizers)) == 3
    r = receipt(tmp_path)
    assert r["status"] == "REFUSED_ATTENTION_FALLBACK" and "eager" in r["reason"]


def test_a_crash_before_training_still_leaves_a_receipt(tmp_path, tokenizers, monkeypatch):
    from v2gpu import modeling

    def boom(cfg, seed):
        raise RuntimeError("CUDA error: out of memory (simulated)")

    monkeypatch.setattr(modeling, "load_for_training", boom)
    with pytest.raises(RuntimeError):
        job(tmp_path, files(tmp_path, tokenizers))
    r = receipt(tmp_path)
    assert r["status"] == "CRASHED" and "out of memory" in r["error"]


def test_attention_check():
    assert check_attention({"attn_implementation_requested": "sdpa", "attn_implementation_loaded": "sdpa"}) is None
    assert "eager" in check_attention({"attn_implementation_requested": "sdpa", "attn_implementation_loaded": "eager"})


def test_a_missing_pair_listed_in_excluded_is_accepted_and_counted(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers, excluded=[{"record_id": "t1", "fmt": None, "model": None,
                                                  "reason": "control_tokens"}])
    rewrite_prompts(tmp_path, "train.prompts.jsonl",
                    [e for e in build(TRAIN, "codellama", None, TRAIN, prompt, gold) if e.record_id != "t1"])
    assert job(tmp_path, args) == 0
    assert receipt(tmp_path)["preflight"]["excluded_by_compose"]["train"] == {"control_tokens": 3}


def test_an_excluded_pair_that_still_has_a_prompt_is_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers, excluded=[{"record_id": "t1", "fmt": "F2", "model": "codellama",
                                                  "reason": "too_long_test_rule"}])
    assert job(tmp_path, args) == 2
    assert any("t1 F2: listed in excluded.jsonl but has a prompt" in p for p in receipt(tmp_path)["problems"])


def test_neighbours_field_that_differs_from_the_direct_links_is_refused(tmp_path, tokenizers):
    train = [dict(r, feature_path="features/shared.feature") if r["id"] in ("t2", "t3") else r for r in TRAIN]
    assert job(tmp_path, files(tmp_path, tokenizers, train=train)) == 2        # t2, t3 share a feature, fields say none
    assert any("t2: neighbours field differs" in p for p in receipt(tmp_path)["problems"])


def test_a_dev_record_linked_to_train_is_refused(tmp_path, tokenizers):
    dev = [dict(DEV[0], feature_path=TRAIN[4]["feature_path"]), DEV[1]]
    assert job(tmp_path, files(tmp_path, tokenizers, dev=dev)) == 2
    assert any("dev record d0 is a direct neighbour of train record(s) ['t4']" in p for p in receipt(tmp_path)["problems"])


def test_a_system_turn_is_refused(tmp_path, tokenizers):
    def system(r, fmt, tok, pool):
        return [{"role": "system", "content": "be good"}, {"role": "user", "content": r["scenario"]}], prov()

    assert job(tmp_path, files(tmp_path, tokenizers, compose=system)) == 2
    assert all("one user turn" in p for p in receipt(tmp_path)["problems"])


def test_a_prompt_line_without_provenance_is_refused_before_anything_runs(tmp_path, tokenizers, capsys):
    args = files(tmp_path, tokenizers)
    rows = [json.loads(l) for l in (tmp_path / "prompts" / "train.prompts.jsonl").read_text().splitlines()]
    del rows[0]["provenance"]
    (tmp_path / "prompts" / "train.prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    resum(tmp_path)
    assert job(tmp_path, args) == 2
    assert "train.prompts.jsonl:1: keys" in capsys.readouterr().err


def test_phase2_inputs_that_differ_from_the_seal_are_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    (tmp_path / "forbidden.json").write_text(json.dumps({REPO: {"test": []}}))    # e.g. an older copy
    assert job(tmp_path, args) == 2
    assert any("forbidden.json: sha256 differs from PHASE2_SHA256SUMS" in p for p in receipt(tmp_path)["problems"])


def test_meta_with_another_tokenizer_revision_is_refused(tmp_path, tokenizers):
    meta = {"v2tools_commit": "test", "phase2_sums_sha256": "0" * 64,
            "tokenizer_revisions": {k: "main" for k in MODELS}}
    assert job(tmp_path, files(tmp_path, tokenizers, meta=meta)) == 2
    p = receipt(tmp_path)["problems"]
    assert any("phase2_sums_sha256" in x for x in p) and sum("tokenizer revision" in x for x in p) == 3


def test_job_mode_needs_the_phase2_seal(tmp_path, tokenizers, capsys):
    args = [a for a in files(tmp_path, tokenizers)]
    i = args.index("--phase2-sums")
    del args[i:i + 2]
    args[args.index("test")] = "job"
    assert job(tmp_path, args) == 2
    assert "--phase2-sums" in capsys.readouterr().err


def test_an_answer_without_the_create_code_is_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    exs = build(TRAIN, "codellama", None, TRAIN, prompt, lambda r: "// FILE: x.php\nsomething else")
    rewrite_prompts(tmp_path, "train.prompts.jsonl", exs)
    assert job(tmp_path, args) == 2
    assert any("answer lacks the code of CREATE function(s) ['trainF.php:0']" in p for p in receipt(tmp_path)["problems"])


def test_rule_d_can_exclude_one_format_only(tmp_path, tokenizers):
    def long_f2(r, fmt, tok, pool):
        filler = " ".join(["context"] * 700) if (fmt, r["id"]) == ("F2", "t3") else ""
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']} {filler}"}], prov()

    assert job(tmp_path, files(tmp_path, tokenizers, compose=long_f2)) == 0
    assert receipt(tmp_path)["preflight"]["excluded_rule_d"]["train"] == {"F2": 1}


def test_short_answer_slices_are_refused_not_report_only(tmp_path, tokenizers):
    longer = "#[Given('step 1')]\npublic function stepOne(): void { $this->shop->create('one'); $this->shop->save(); }"
    train = [rec(1, answer=longer) if i == 1 else rec(i) for i in range(6)]

    def shows_body(r, fmt, tok, pool):
        body = gold(r).split("\n", 1)[1]            # a short answer shown without its annotation line
        text = f"[{fmt}] {r['scenario']}\n" + (body if (fmt, r["id"]) == ("F2", "t1") else "")
        return [{"role": "user", "content": text}], prov()

    # refused: both its signature line and its short answer slice appear in the prompt
    assert job(tmp_path, files(tmp_path, tokenizers, train=train, compose=shows_body)) == 2
    r = receipt(tmp_path)
    assert any("[F2] t1: short answer slice of trainF.php:1 appears in the prompt" in p for p in r["problems"])
    assert "short_answer_slice" not in r["preflight"]["report_only"]


def test_items_compose_dropped_are_counted_by_rule(tmp_path, tokenizers):
    def drops(r, fmt, tok, pool):
        dropped = [{"kind": "example", "id": "t9", "rule": "same_group"}] if fmt == "F1" else []
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']}"}], prov(dropped=dropped)

    assert job(tmp_path, files(tmp_path, tokenizers, compose=drops)) == 0
    assert receipt(tmp_path)["preflight"]["dropped_by_compose"] == {"train": {"same_group": 6}, "dev": {"same_group": 2}}


def test_a_record_excluded_in_every_format_leaves_the_retrieval_pool(tmp_path, tokenizers):
    listed = [{"record_id": "t2", "fmt": None, "model": None, "reason": "uses_test_infra"}]
    pool = [r for r in TRAIN if r["id"] != "t2"]
    args = files(tmp_path, tokenizers, pool=pool, excluded=listed)
    rewrite_prompts(tmp_path, "train.prompts.jsonl", build(pool, "codellama", None, pool, prompt, gold))
    assert job(tmp_path, args) == 0
    r = receipt(tmp_path)
    assert r["preflight"]["excluded_by_compose"]["train"] == {"uses_test_infra": 3}
    assert r["data"]["pool_sha256"] == pool_digest(r["id"] for r in pool)


def test_a_preexisting_answer_is_the_rendered_catalog_function(tmp_path, tokenizers):
    code = "#[Given('step 1')]\n#[Given('added later')]\npublic function step1(): void { $this->x = 1; }"
    later = {"repo": REPO, "fn_id": "trainF.php:1", "code": code, "first_line": 1, "suites": ["s"], "catch_all": False,
             "scope_tags": [], "annotations": [{"def_id": "trainF.php:1@1", "line": 1, "pattern": "step t1", "time": 150},
                                               {"def_id": "trainF.php:1@2", "line": 2, "pattern": "later", "time": 900}]}
    train = [rec(1, preexisting=True, answer=code) if i == 1 else rec(i) for i in range(6)]

    def rendered(r):
        return gold(r).replace("#[Given('added later')]\n", "")

    args = files(tmp_path, tokenizers, train=train, catalog_extra=[later], answer=rendered)
    assert job(tmp_path, args) == 0                     # the later annotation is not part of the gold
    args = files(tmp_path / "b", tokenizers, train=train, catalog_extra=[later],
                 answer=lambda r: gold(r).replace("public function step1", "public function other"))
    assert job(tmp_path / "b", args) == 2
    assert any("answer lacks the code of CREATE function(s) ['trainF.php:1']" in p
               for p in receipt(tmp_path / "b")["problems"])


def test_test_infrastructure_names_are_counted_not_refused(tmp_path, tokenizers):
    def uses_infra(r, fmt, tok, pool):
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']}\n$this->lastOrderFor($x);"}], prov()

    infra = {REPO: [["trainF.php", "lastOrderFor"], ["devF.php", "lastOrderFor"], ["elsewhere.php", "lastOrderFor"]]}
    args = files(tmp_path, tokenizers, compose=uses_infra, infra=infra)     # the target file is shown, elsewhere is not
    assert job(tmp_path, args) == 0
    assert receipt(tmp_path)["preflight"]["report_only"]["test_infra_name"] == 8 * 3


def test_a_record_with_another_models_control_token_leaves_the_pool(tmp_path, tokenizers):
    train = [dict(r, scenario=r["scenario"] + " <|EOT|>") if r["id"] == "t3" else r for r in TRAIN]   # DeepSeek's token
    pool = [r for r in train if r["id"] != "t3"]
    args = files(tmp_path, tokenizers, train=train, pool=pool)          # t3 is still a CodeLlama query, not an example
    assert job(tmp_path, args) == 0
    assert receipt(tmp_path)["data"]["pool_sha256"] == pool_digest(r["id"] for r in pool)


def test_pools_json_that_differs_from_the_recomputed_pool_is_refused(tmp_path, tokenizers):
    wrong = [r["id"] for r in TRAIN if r["id"] != "t5"]
    args = files(tmp_path, tokenizers, pools={"all": {"sha256": pool_digest(wrong), "ids": wrong}})
    assert job(tmp_path, args) == 2
    assert any("pools.json pool 'all' differs from the recomputed pool (missing ['t5']" in p
               for p in receipt(tmp_path)["problems"])


def test_a_blank_line_in_excluded_is_refused_with_a_receipt(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    (tmp_path / "prompts" / "excluded.jsonl").write_text("\n")
    resum(tmp_path)
    assert job(tmp_path, args) == 2
    assert "excluded.jsonl:1: blank line" in receipt(tmp_path)["problems"]


def test_a_stray_file_in_the_prompt_directory_is_refused(tmp_path, tokenizers):
    args = files(tmp_path, tokenizers)
    (tmp_path / "prompts" / "train.prompts.old.jsonl").write_text("{}\n")
    assert job(tmp_path, args) == 2
    assert "train.prompts.old.jsonl: in the prompt directory but not listed in PROMPTS_SHA256SUMS" in \
        receipt(tmp_path)["problems"]


def test_the_catalog_must_have_the_sha256_the_job_manifest_names(tmp_path, tokenizers, capsys):
    args = files(tmp_path, tokenizers)
    assert job(tmp_path, args, "--catalog-sha256", sha(tmp_path / "catalog.jsonl")) == 0
    assert job(tmp_path / "b", files(tmp_path / "b", tokenizers), "--catalog-sha256", "0" * 64) == 2
    assert any("catalog.jsonl: sha256 is not the --catalog-sha256 of the job" in p
               for p in receipt(tmp_path / "b")["problems"])
    job_mode = [a if a != "test" else "job" for a in files(tmp_path / "c", tokenizers)]
    assert job(tmp_path / "c", job_mode) == 2 and "--catalog-sha256 is required in job mode" in capsys.readouterr().err


def test_leak_problems_name_the_format(tmp_path, tokenizers):
    def leaky(r, fmt, tok, pool):
        return [{"role": "user", "content": f"[{fmt}] {r['scenario']}\n{gold(r)}"}], prov()

    assert job(tmp_path, files(tmp_path, tokenizers, compose=leaky)) == 2
    p = receipt(tmp_path)["problems"]
    assert any(x.startswith("[F0] ") for x in p) and any(x.startswith("[F2] ") for x in p)


def test_a_catalog_carrying_the_code_of_a_forbidden_test_function_is_refused(tmp_path, tokenizers):
    """The training pod must refuse an unstripped catalog, whatever else about the job is fine."""
    extra_fn = {"repo": REPO, "fn_id": "Test.php:1", "code": "public function t() { }", "first_line": 1,
                "suites": ["s"], "catch_all": False, "scope_tags": [], "annotations": []}
    forbidden = {REPO: {"test": ["Test.php:1"]}}
    args = files(tmp_path, tokenizers, forbidden=forbidden, catalog_extra=[extra_fn])
    assert job(tmp_path, args) == 2
    assert any("holds the code of 1 test/aux answer function(s), e.g. Test.php:1" in p
               for p in receipt(tmp_path)["problems"])


def test_the_catalog_reader_splits_lines_on_newline_only(tmp_path):
    """decision 24/09: the training pod's catalog reader, like every JSONL reader, splits on b"\\n" only."""
    from v2gpu.train_job import read_catalog

    fn = {"repo": "R", "fn_id": "f1", "code": "// a\u2028b\x85c"}
    (tmp_path / "catalog.jsonl").write_text(json.dumps(fn, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_catalog(tmp_path / "catalog.jsonl") == {"R": {"f1": fn}}


def test_the_upcast_help_and_the_loadcfg_comment_say_it_is_required_for_real_jobs(capsys):
    """The --upcast-fp32 help and modeling.LoadCfg's comment
    still said "canary comparison only", although since 27/09 it is required for every real job.
    Both files are in TRAIN_CODE (modeling.py also in generate.GEN_CODE), so the fix changes train_code_sha256:
    fixed before the first real training job (decision 28/09)."""
    import inspect

    from v2gpu import modeling, train_job

    with pytest.raises(SystemExit):
        train_job.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "canary comparison" not in help_text and "required for every real job" in help_text
    src = " ".join(inspect.getsource(modeling.LoadCfg).split())
    assert "canary comparison only" not in src and "required for every real job" in src
