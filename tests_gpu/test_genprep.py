# tests_gpu/test_genprep.py
import hashlib
import json
from dataclasses import asdict

import pytest

from tests_gpu.test_audit import QUERIES, main as audit, setup
from tests_gpu.tiny import make_tiny_llama
from v2gpu.examples import Example
from v2gpu.genprep import preflight, receipt_problems, tokenize
from v2gpu.modeling import LoadCfg, load_for_training
from v2gpu.receipt import tensor_digest
from v2gpu.runs import Run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    """A real peft adapter (tiny Llama) and the tensor digest a train receipt would carry for it."""
    d = tmp_path_factory.mktemp("adapter")
    model, _ = load_for_training(LoadCfg(make_tiny_llama(d / "tiny", 64), None, quant="none", attn="sdpa",
                                         dtype="float32", gradient_checkpointing=False), seed=42)
    model.save_pretrained(d / "ad")
    return d / "ad", tensor_digest(d / "ad" / "adapter_model.safetensors")


def audited(tmp_path, **kw):
    """Prompt files + a real laptop leak-audit receipt (v2gpu.audit), as the laptop hands them to a generation pod."""
    assert audit(setup(tmp_path, **kw)) == 0
    return tmp_path / "prompts", tmp_path / "leakcheck-receipt.json"


def resum(d):
    """Rewrite PROMPTS_SHA256SUMS after a file of the directory changed (someone re-sums after the audit)."""
    names = sorted(p.name for p in d.iterdir() if p.name != "PROMPTS_SHA256SUMS")
    (d / "PROMPTS_SHA256SUMS").write_text("".join(f"{sha(d / n)}  {n}\n" for n in names))


def run(d, receipt, model="deepseek", weights="base", variant="main", partitions=("test", "aux_test"), **kw):
    args = dict(model=model, weights=weights, variant=variant, partitions=partitions,
                prompts=[d / "test.deepseek.jsonl"], sums=d / "PROMPTS_SHA256SUMS", subset="all", receipt=receipt,
                receipt_sha256=sha(receipt) if receipt.exists() else "0" * 64)
    return preflight(**{**args, **kw})


def test_a_clean_audited_directory_gives_every_pair_of_the_job(tmp_path, tokenizers, adapter):
    d, r = audited(tmp_path)
    problems, prep = run(d, r)
    assert problems == [] and prep.receipt_sha256 == sha(r)
    assert prep.files == {"test.deepseek.jsonl": sha(d / "test.deepseek.jsonl")}
    assert [(p.run.condition, p.record_id) for p in prep.pairs] == [
        (c, q) for c in ("zero_shot", "rag_only") for q in ("a1", "x1", "x2")]
    assert {p.partition for p in prep.pairs if p.record_id == "a1"} == {"aux_test"}
    assert all(p.prompts_sha256 == sha(d / "test.deepseek.jsonl") and p.example.fmt == p.run.fmt for p in prep.pairs)
    problems, prep = run(d, r, weights="s7", adapter=adapter[0], adapter_sha256=adapter[1])
    assert problems == [] and len(prep.pairs) == 9
    assert [p.run for p in prep.pairs[:1]] == [Run("deepseek", "s7", "ft_only")]


def test_the_receipt_must_be_the_one_named_in_the_job_command(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    problems, _ = run(d, r, receipt_sha256="f" * 64)
    assert problems == [f"leak-audit receipt sha256 {sha(r)} is not the job's {'f' * 64}"]
    problems, _ = run(d, tmp_path / "nowhere.json")
    assert any("missing" in p for p in problems)


def test_a_failed_or_foreign_receipt_is_refused(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    good = json.loads(r.read_text())
    for change, words in (({"status": "FAIL", "n_problems": 1}, "PASS"), ({"model": "mistral"}, "model"),
                          ({"subset": "lc600"}, "pool"), ({"schema": "gtestgen.v2.other.v1"}, "schema")):
        r.write_text(json.dumps({**good, **change}))
        problems, _ = run(d, r)
        assert any(words in p for p in problems), change


def test_a_receipt_without_code_digests_is_refused(tmp_path, tokenizers):
    """A receipt from an audit older than the code_sha256 field names no code, so it cannot pass."""
    d, r = audited(tmp_path)
    good = json.loads(r.read_text())
    r.write_text(json.dumps({k: v for k, v in good.items() if k != "code_sha256"}))
    problems, _ = run(d, r)
    assert any("code_sha256" in p for p in problems)


def test_a_receipt_made_by_other_leak_rules_is_refused(tmp_path, tokenizers):
    """The receipt's leakcheck.py is not this checkout's (e.g. an audit run before 24/09/2026, when
    short answer slices became a refusal); a receipt that leaves a deciding module out is refused too."""
    d, r = audited(tmp_path)
    good = json.loads(r.read_text())
    r.write_text(json.dumps({**good, "code_sha256": {**good["code_sha256"], "v2gpu/leakcheck.py": "0" * 64}}))
    problems, _ = run(d, r)
    assert any("other code" in p and "v2gpu/leakcheck.py" in p for p in problems)
    fewer = {k: v for k, v in good["code_sha256"].items() if k != "v2gpu/serialize.py"}
    r.write_text(json.dumps({**good, "code_sha256": fewer}))
    assert any("v2gpu/serialize.py" in p for p in run(d, r)[0])


def test_a_prompt_file_changed_and_resummed_after_the_audit_is_caught_by_the_receipt(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    with (d / "test.deepseek.jsonl").open("a") as f:     # one more line after the audit, sums rewritten to match
        f.write(json.dumps(asdict(Example("x9", "F0", "deepseek", "test", "p", [{"role": "user", "content": "new"}],
                                          None, {"examples": [], "knowledge": [], "skeleton": [], "dropped": []})),
                           sort_keys=True) + "\n")
    resum(d)
    problems, _ = run(d, r)
    assert any("not audited by the receipt: ['test.deepseek.jsonl']" in p for p in problems)
    assert any("audited another PROMPTS_SHA256SUMS" in p for p in problems)


def test_every_partition_the_job_reads_must_have_been_audited(tmp_path, tokenizers):
    d, r = audited(tmp_path, queries=QUERIES[:2])                         # the audit saw test only
    problems, _ = run(d, r)
    assert any("partitions ['test']" in p for p in problems)


def test_a_mislabelled_train_line_with_an_answer_is_refused_by_the_reader(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    with (d / "test.deepseek.jsonl").open("a") as f:
        prov = {"examples": [], "knowledge": [], "skeleton": [], "dropped": []}
        f.write(json.dumps(asdict(Example("t9", "F0", "deepseek", "train", "p", [{"role": "user", "content": "q"}],
                                          "GOLD CODE", prov)), sort_keys=True) + "\n")
    resum(d)
    problems, prep = run(d, r)
    assert any("partition 'train', expected one of ['aux_test', 'test']" in p for p in problems) and prep.pairs == []


def test_missing_pairs_need_an_exclusion_and_excluded_pairs_are_counted(tmp_path, tokenizers):
    d, r = audited(tmp_path, skip={("x1", "F1")},
                   excluded=[{"record_id": "x1", "fmt": "F1", "model": None, "reason": "too_long_test_rule"}])
    problems, prep = run(d, r)
    assert problems == [] and prep.excluded == {"test": {"too_long_test_rule": 1}}
    assert ("rag_only", "x1") not in {(p.run.condition, p.record_id) for p in prep.pairs} and len(prep.pairs) == 5
    (tmp_path / "prompts").rename(tmp_path / "p0")
    (tmp_path / "phase2").rename(tmp_path / "q0")
    d, r = audited(tmp_path)
    lines = (d / "test.deepseek.jsonl").read_text().splitlines()
    (d / "test.deepseek.jsonl").write_text("".join(l + "\n" for l in lines if not ('"x1"' in l and '"F1"' in l)))
    resum(d)
    problems, _ = run(d, r)
    assert "missing test prompt x1 F1" in problems


def test_an_exclusion_of_another_model_does_not_skip_this_models_pair(tmp_path, tokenizers):
    """excluded.jsonl lines with a model apply to that model only."""
    d, r = audited(tmp_path, excluded=[{"record_id": "x1", "fmt": "F1", "model": "mistral", "reason": "control_token"}])
    problems, prep = run(d, r)
    assert problems == [] and prep.excluded == {} and len(prep.pairs) == 6
    assert ("rag_only", "x1") in {(p.run.condition, p.record_id) for p in prep.pairs}


def test_every_expected_partition_must_have_prompts(tmp_path, tokenizers):
    """A DeepSeek main job given only its test lines would finish DONE with no aux_test pair."""
    d, r = audited(tmp_path)
    lines = (d / "test.deepseek.jsonl").read_text().splitlines()
    (d / "test.deepseek.jsonl").write_text("".join(x + "\n" for x in lines if '"aux_test"' not in x))
    resum(d)
    problems, _ = run(d, r)
    assert "no aux_test prompt among --prompts (an expected partition)" in problems


def test_a_listed_prompt_file_with_this_jobs_lines_must_be_passed(tmp_path, tokenizers):
    """The prompt files are exactly the files the sums list for (model, partition, formats needed)."""
    d, r = audited(tmp_path)
    lines = (d / "test.deepseek.jsonl").read_text().splitlines()
    (d / "test.deepseek.jsonl").write_text("".join(x + "\n" for x in lines if '"aux_test"' not in x))
    (d / "aux_test.deepseek.jsonl").write_text("".join(x + "\n" for x in lines if '"aux_test"' in x))
    resum(d)
    problems, _ = run(d, r)
    assert any(p.startswith("aux_test.deepseek.jsonl: listed in PROMPTS_SHA256SUMS with deepseek aux_test")
               for p in problems)
    problems, _ = run(d, r, prompts=[d / "test.deepseek.jsonl", d / "aux_test.deepseek.jsonl"])
    assert not any("listed in PROMPTS_SHA256SUMS" in p or "an expected partition" in p for p in problems)


def test_lines_must_be_built_for_the_job_model_and_pool(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    problems, _ = run(d, r, model="codellama", variant="gen_canary")
    assert any("built for model 'deepseek'" in p for p in problems)
    assert any("model" in p and "receipt" in p for p in problems)
    pools = json.loads((d / "pools.json").read_text())
    pools["all"]["sha256"] = "e" * 64
    (d / "pools.json").write_text(json.dumps(pools))
    resum(d)
    problems, _ = run(d, r)
    assert any("retrieval pool" in p for p in problems)


def test_unplanned_runs_and_missing_adapters_are_refused(tmp_path, tokenizers, adapter):
    d, r = audited(tmp_path)
    problems, _ = run(d, r, model="codellama")                       # aux_test is DeepSeek only
    assert any("not planned" in p for p in problems)
    problems, _ = run(d, r, weights="s42", adapter=tmp_path / "no-adapter", adapter_sha256="0" * 64)
    assert any("missing" in p and "adapter" in p for p in problems)
    problems, _ = run(d, r, weights="s42", adapter=adapter[0], adapter_sha256="0" * 64)
    assert any("differs from the train receipt" in p for p in problems)
    problems, _ = run(d, r, weights="s42", adapter=None, adapter_sha256=None)
    assert any("--adapter" in p for p in problems)
    problems, _ = run(d, r, weights="base", adapter=tmp_path, adapter_sha256="0" * 64)
    assert any("base" in p and "adapter" in p for p in problems)


def test_tokenize_gives_ids_budgets_and_a_digest(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    _, prep = run(d, r)
    tok = tokenizers["deepseek"]
    problems, ids, digest = tokenize(tok, prep.pairs, 16384)
    first = prep.pairs[0]
    assert problems == [] and ids[(first.record_id, first.run.fmt)][0][0] == tok.bos_token_id
    assert ids[(first.record_id, first.run.fmt)][1] == min(6144, 16384 - len(ids[(first.record_id, first.run.fmt)][0]))
    assert len(digest) == 64 and tokenize(tok, list(reversed(prep.pairs)), 16384)[2] == digest
    problems, _, _ = tokenize(tok, prep.pairs, 5)
    assert problems and all("no room" in p for p in problems)


def test_receipt_problems_is_usable_alone(tmp_path, tokenizers):
    d, r = audited(tmp_path)
    assert receipt_problems(r, sha(r), "deepseek", "all", {"test.deepseek.jsonl": sha(d / "test.deepseek.jsonl")},
                            sha(d / "PROMPTS_SHA256SUMS"), ("test", "aux_test")) == []


def test_a_dev_job_needs_a_pass_receipt_that_audited_its_dev_prompts(tmp_path, tokenizers):
    """decision 24/09: the laptop leak-audit PASS receipt is required for dev jobs too (dev_pilot_*, dev_grid_*):
    a receipt of other prompts, no receipt, or a FAIL receipt refuses the job before any engine loads."""
    d, r = audited(tmp_path)
    path = d / "test.deepseek.jsonl"
    rows = [json.loads(x) for x in path.read_bytes().split(b"\n") if x]
    path.write_text("".join(json.dumps({**row, "partition": "dev", "answer": "A"}, sort_keys=True) + "\n"
                            for row in rows))
    resum(d)
    dev = {"variant": "dev_pilot_v0", "partitions": ("dev",)}
    problems, _ = run(d, r, **dev)
    assert any("the job reads ['dev']" in p for p in problems) and any("not audited" in p for p in problems)
    problems, _ = run(d, tmp_path / "nowhere.json", **dev)
    assert any("missing" in p for p in problems)
    r.write_text(json.dumps({**json.loads(r.read_text()), "status": "FAIL", "n_problems": 1}))
    problems, _ = run(d, r, **dev)
    assert any("PASS needed" in p for p in problems)


def test_a_job_with_no_partitions_is_a_problem(tmp_path, tokenizers):
    """No partition means no pair, so a job would finish DONE with nothing."""
    d, r = audited(tmp_path)
    problems, prep = run(d, r, partitions=())
    assert any("no partitions" in p for p in problems) and prep.pairs == []


@pytest.mark.parametrize("body", ["[]", "5", '{"prompts": ["x"], "inputs": {}}', '{"prompts": {}, "inputs": ["x"]}'])
def test_a_receipt_of_the_wrong_shape_is_a_problem_not_a_crash(tmp_path, body):
    """A receipt of the wrong shape is reported as a problem."""
    r = tmp_path / "receipt.json"
    r.write_text(body)
    problems = receipt_problems(r, sha(r), "deepseek", "all", {"test.deepseek.jsonl": "0" * 64}, "f" * 64, ("test",))
    assert any("leak-audit receipt" in p and "not" in p for p in problems), problems



K8 = {"variant": "dev_grid_k8_n3_r050", "k": 8, "n_examples": 3, "ratio": 0.5, "budget_examples": 1536,
      "budget_knowledge": 1536, "mode": "full"}          # 3A --variant is the B label for a grid point
V0 = {**K8, "variant": "v0"}                             # the real v0 META compose (dev_pilot_v0, and main's shape)
ROUND1 = {**V0, "variant": "v1", "ratio": 0.6666666666666666}   # round 1's OWN dev_pilot_v1 META compose


@pytest.mark.parametrize("variant, meta, bound", [
    ("dev_grid_k8_n3_r050", {"compose": K8}, True),
    ("dev_grid_v1_k8_n3_r050", {"compose": {**K8, "variant": "dev_grid_v1_k8_n3_r050"}}, True),
    ("dev_grid_k4_n3_r033", {"compose": {**K8, "variant": "dev_grid_k4_n3_r033", "k": 4, "ratio": 1 / 3}}, True),
    # The grid builds r033/r067 with the exact floats; 0.335 is not r033
    ("dev_grid_k4_n1_r033", {"compose": {**K8, "variant": "dev_grid_k4_n1_r033", "k": 4, "n_examples": 1,
                                         "ratio": 0.3333333333333333}}, True),
    ("dev_grid_v1_k12_n2_r067", {"compose": {**K8, "variant": "dev_grid_v1_k12_n2_r067", "k": 12, "n_examples": 2,
                                             "ratio": 0.6666666666666666}}, True),
    ("dev_grid_k4_n1_r033", {"compose": {**K8, "variant": "dev_grid_k4_n1_r033", "k": 4, "n_examples": 1,
                                         "ratio": 0.335}}, False),
    ("dev_grid_v1_k12_n2_r067", {"compose": {**K8, "variant": "dev_grid_v1_k12_n2_r067", "k": 12, "n_examples": 2,
                                             "ratio": 0.675}}, False),
    ("dev_grid_k4_n1_r033", {"compose": {**K8, "variant": "dev_grid_k4_n1_r033"}}, False),
    ("dev_grid_k8_n2_r050", {"compose": {**K8, "variant": "dev_grid_k8_n2_r050"}}, False),
    ("dev_grid_k8_n3_r067", {"compose": {**K8, "variant": "dev_grid_k8_n3_r067"}}, False),
    ("dev_grid_k8_n3_r050", {"compose": {**K8, "k": 8.0}}, False),
    ("dev_grid_k8_n3_r050", {}, False),
    ("dev_grid_k8_n3_r050", {"compose": V0}, False),             # the v0 pilot dir under a grid label
    ("dev_grid_v1_k8_n3_r050", {"compose": K8}, False),          # a round-0 grid dir under a round-1 label
    ("dev_grid_k8_n3_r050", {"compose": {**K8, "mode": "4a"}}, False),
    ("main", {"compose": V0}, True),
    ("main", {}, False),                                         # every real variant needs META compose
    ("dev_pilot_v0", {"compose": V0}, True),
    ("dev_pilot_v1", {"compose": V0}, False),                    # round 1 on the round-0 directory
    ("dev_pilot_v1", {"compose": {**V0, "variant": "v1", "k": 12}}, True),
    ("dev_pilot_v0", {"compose": {**V0, "mode": "placebo"}}, False),
    ("dev_pilot_v0", {"compose": K8}, False),
    ("gen_canary", {}, True),
])
def test_a_run_label_is_bound_to_the_compose_block_of_its_meta(variant, meta, bound):
    """dev_pilot_vN reads a 3A directory built with --variant vN,
    a dev grid label one built with --variant <that label> and its k/n/r; mode full; main needs a compose block."""
    from v2gpu.genprep import compose_problems

    assert (compose_problems(variant, meta) == []) == bound


@pytest.mark.parametrize("variant, train, pilot, bound", [
    ("dev_pilot_v0", V0, None, True),
    ("dev_pilot_v0", {**V0, "k": 12}, None, False),             # a round-1-style adapter on the round-0 prompts
    ("dev_pilot_v0", {**V0, "variant": "v1"}, None, False),
    ("main", V0, None, True),
    ("main", {**V0, "budget_examples": 1024}, None, False),     # every key
    ("dev_grid_k4_n1_r033", V0, V0, True),                      # the round-v0 adapter matches its own pilot compose
    ("dev_grid_k4_n1_r033", {**V0, "variant": "v1"}, V0, False),
    ("dev_grid_v1_k4_n1_r033", ROUND1, ROUND1, True),                   # the round-1 adapter matches its own pilot compose
    ("dev_grid_v1_k4_n1_r033", V0, ROUND1, False),
    ("dev_grid_v1_k4_n1_r033", ROUND1, V0, False),                  # the wrong round's pilot compose given
    ("dev_grid_v1_k4_n1_r033", ROUND1, None, False),                # no round pilot compose given at all
    # Since 27/09/2026 a round-1 adapter trained at ratio 0.5 must be refused
    # once dev_pilot_v1 has been rebuilt at another ratio (the OLD rule only checked compose["variant"])
    ("dev_grid_v1_k12_n2_r067", {**ROUND1, "ratio": 0.5}, ROUND1, False),
    # An earlier version dropped the round-binding check, so a v0 adapter
    # given the v0 pilot META passed under a dev_grid_v1_* label (pilot_compose == train_compose == V0, so the
    # plain equality check alone finds nothing wrong) -- must be refused: v0 is the wrong round for dev_grid_v1_*.
    ("dev_grid_v1_k12_n2_r067", V0, V0, False),
    ("dev_pilot_v0", None, None, False),
])
def test_the_adapter_was_trained_on_the_compose_the_run_label_needs(variant, train, pilot, bound):
    """Since 25/09/2026: dev_pilot_vN and main run an adapter whose train receipt compose
    (data.prompts.meta.compose) equals the generation META compose, every key. A dev grid label's own directory
    legitimately varies k, n_examples and ratio away from its
    round's pilot, so its adapter is checked instead against the round's OWN dev_pilot_vN META compose, given
    separately as `pilot`; no pilot compose given is a refusal, not a silent pass."""
    from v2gpu.genprep import adapter_compose_problems

    meta = {"compose": V0 if "grid" not in variant else {**K8, "variant": variant, "k": 4, "n_examples": 1,
                                                          "ratio": 0.33}}
    assert (adapter_compose_problems(variant, meta, train, pilot) == []) == bound
    if train is None:                                        # an old receipt says so, instead of a list of keys
        assert adapter_compose_problems(variant, meta, train, pilot) == [
            "the adapter's train receipt names no compose (data.prompts.meta.compose)"]


def test_grid_meta_exempt_keys_are_pinned():
    """The keys exempt from the grid-META-vs-pilot equality check are
    ONE named module constant, so nobody widens it later without this test failing."""
    from v2gpu.genprep import GRID_META_EXEMPT_KEYS

    assert GRID_META_EXEMPT_KEYS == frozenset(
        {"variant", "k", "n_examples", "ratio", "budget_examples", "budget_knowledge"})


def test_grid_meta_must_equal_pilot_meta_outside_the_exempt_keys():
    """A dev grid
    label's OWN PROMPTS_META.json compose (never checked against anything, before this) must equal the round's
    pilot compose on every key except GRID_META_EXEMPT_KEYS. k/n_examples/ratio (and the two ratio-derived budgets)
    legitimately vary per grid point; a grid built in a different compose MODE than its round's pilot is a real
    defect the adapter-vs-pilot check alone cannot see, since the adapter can match pilot perfectly while the
    grid's own prompts do not."""
    from v2gpu.genprep import adapter_compose_problems

    ok_meta = {"compose": {**K8, "variant": "dev_grid_v1_k4_n1_r033", "k": 4, "n_examples": 1, "ratio": 1 / 3}}
    assert adapter_compose_problems("dev_grid_v1_k4_n1_r033", ok_meta, ROUND1, ROUND1) == []
    bad_meta = {"compose": {**K8, "variant": "dev_grid_v1_k4_n1_r033", "k": 4, "n_examples": 1, "ratio": 1 / 3,
                            "mode": "4a"}}
    problems = adapter_compose_problems("dev_grid_v1_k4_n1_r033", bad_meta, ROUND1, ROUND1)
    assert problems and any("mode" in p for p in problems), problems


def test_dev_pilot_compose_reads_the_rounds_own_meta_file(tmp_path):
    """A dev grid job's own directory varies k, n_examples and ratio away from its round's
    pilot, so the pilot's compose is read straight from its own PROMPTS_META.json, given to genqueue/gencoverage
    separately (--dev-pilot-meta), never derived from the grid's own directory."""
    from v2gpu.genprep import dev_pilot_compose

    meta = tmp_path / "PROMPTS_META.json"
    meta.write_text(json.dumps({"compose": ROUND1, "files": []}))
    assert dev_pilot_compose(meta) == ROUND1
    assert dev_pilot_compose(tmp_path / "missing.json") is None
    meta.write_text("not json")
    assert dev_pilot_compose(meta) is None
    meta.write_text(json.dumps({"files": []}))                # no compose block
    assert dev_pilot_compose(meta) is None
    meta.write_text(json.dumps({"compose": "not a dict"}))
    assert dev_pilot_compose(meta) is None


@pytest.mark.parametrize("files, bound", [(["test.deepseek.jsonl"], True), ([], False),
                                          (["test.deepseek.jsonl", "other.jsonl"], False), (None, False)])
def test_meta_files_are_the_prompt_files_the_sums_list(tmp_path, tokenizers, files, bound):
    """META files equals the prompt files PROMPTS_SHA256SUMS lists."""
    d, r = audited(tmp_path)
    meta = json.loads((d / "PROMPTS_META.json").read_text())
    meta.pop("files")
    (d / "PROMPTS_META.json").write_text(json.dumps({**meta, **({"files": files} if files is not None else {})}))
    resum(d)
    assert (not any("PROMPTS_META.json files" in p for p in run(d, r)[0])) == bound


def test_preflight_refuses_a_k8_directory_under_a_k4_label(tmp_path, tokenizers):
    """The dev labels all read dev prompts with one run matrix; only META compose tells them apart."""
    d, r = audited(tmp_path)
    meta = json.loads((d / "PROMPTS_META.json").read_text())
    (d / "PROMPTS_META.json").write_text(json.dumps({**meta, "compose": K8}))
    resum(d)
    dev = {"partitions": ("dev",)}
    assert any("PROMPTS_META.json compose" in p for p in run(d, r, variant="dev_grid_k4_n1_r033", **dev)[0])
    assert not any("PROMPTS_META.json compose" in p for p in run(d, r, variant="dev_grid_k8_n3_r050", **dev)[0])
    (d / "PROMPTS_META.json").write_text(json.dumps({**meta, "compose": V0}))       # the v0 directory
    resum(d)
    for label in ("dev_pilot_v1", "dev_grid_v1_k8_n3_r050"):
        assert any("PROMPTS_META.json compose" in p for p in run(d, r, variant=label, **dev)[0]), label
    assert not any("PROMPTS_META.json compose" in p for p in run(d, r, variant="dev_pilot_v0", **dev)[0])


def test_a_listed_prompt_file_that_is_not_utf8_is_a_problem_naming_it(tmp_path, tokenizers):
    """listed_prompt_problems read it strictly and raised, unnamed; now the file is named, and the job refused."""
    d, r = audited(tmp_path)
    (d / "other.deepseek.jsonl").write_bytes("{\"x\": \"é".encode()[:-1])
    resum(d)
    problems, _ = run(d, r)
    assert any("other.deepseek.jsonl" in p and "not UTF-8" in p for p in problems), problems


def test_a_receipt_that_also_covers_train_prompts_serves_a_test_job(tmp_path, tokenizers):
    """The audit receipt may list extra (train) files; the job's own files must be among them."""
    d, r = audited(tmp_path, train_text=lambda rec, fmt: f"[{fmt}] {rec['scenario']}")
    assert "train.deepseek.jsonl" in json.loads(r.read_text())["files"]
    problems, prep = run(d, r)
    assert problems == [] and len(prep.pairs) == 6


def test_preflight_compares_the_control_token_list_with_meta(tmp_path, tokenizers):
    """META special_tokens_sha256 must be the sha256 of config/special_tokens.json."""
    d, r = audited(tmp_path)
    meta = json.loads((d / "PROMPTS_META.json").read_text())
    (d / "PROMPTS_META.json").write_text(json.dumps({**meta, "special_tokens_sha256": "0" * 64}))
    resum(d)
    assert any("special_tokens_sha256" in p for p in run(d, r)[0])
