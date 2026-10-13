# tests_gpu/test_runs.py
import json

import pytest

from v2gpu.models import allowed_seeds
from v2gpu.runs import (CONDITIONS, OUTPUT_FIELDS, RUN_SEEDS, Run, job_runs, planned_runs, prompt_sha256, row_key,
                        row_problems, seed_label, variant_arg, variant_conditions, variant_spec)


def test_main_test_has_39_runs_and_aux_17():
    """Runs per record only; record counts come from the prompt files and excluded.jsonl, never from a constant."""
    test, aux = planned_runs("main", "test"), planned_runs("main", "aux_test")
    assert len(test) == 39 and len(aux) == 17
    assert len(set(test)) == 39 and {r.model for r in aux} == {"deepseek"}


def test_base_conditions_run_once_per_model_and_adapter_conditions_once_per_seed():
    test = planned_runs("main", "test")
    base = [r for r in test if r.seed == "base"]
    assert sorted((r.model, r.condition) for r in base) == sorted(
        (m, c) for m in ("codellama", "deepseek", "mistral") for c in ("zero_shot", "rag_only"))
    assert {r.seed for r in test if r.model == "deepseek" and r.seed != "base"} == {"s42", "s11", "s23", "s7", "s99"}
    assert {r.seed for r in test if r.model == "mistral" and r.seed != "base"} == {"s42", "s11", "s23"}
    assert all(r.weights == "adapter" for r in test if r.seed != "base")


def test_dev_pilot_is_deepseek_seed_42_and_base_five_conditions():
    for variant in ("dev_pilot_v0", "dev_pilot_v1"):
        runs = planned_runs(variant, "dev")
        assert sorted(r.condition for r in runs) == sorted(CONDITIONS)
        assert {(r.model, r.seed) for r in runs} == {("deepseek", "base"), ("deepseek", "s42")}
        assert len(runs) == 5                      # x the dev records of the prompt files, minus excluded.jsonl


def test_unknown_variant_or_partition_is_refused():
    with pytest.raises(ValueError):
        planned_runs("main", "dev")
    with pytest.raises(ValueError):
        planned_runs("lc600", "test")


def test_conditions_pair_weights_with_formats():
    assert {c: (Run("deepseek", "s42", c).weights, Run("deepseek", "s42", c).fmt) for c in CONDITIONS} == {
        "zero_shot": ("base", "F0"), "rag_only": ("base", "F1"), "ft_only": ("adapter", "F0"),
        "full_neutral": ("adapter", "F1"), "full_aware": ("adapter", "F2")}


def test_job_runs_are_the_planned_runs_of_one_engine_load():
    base = job_runs("main", ("test", "aux_test"), "deepseek", "base")
    assert [r.condition for r in base] == ["zero_shot", "rag_only"]
    adapter = job_runs("main", ("test",), "mistral", "s11")
    assert [r.condition for r in adapter] == ["ft_only", "full_neutral", "full_aware"]
    with pytest.raises(ValueError, match="aux_test"):
        job_runs("main", ("test", "aux_test"), "codellama", "base")    # aux_test is DeepSeek only
    with pytest.raises(ValueError, match="s7"):
        job_runs("main", ("test",), "mistral", "s7")                     # Mistral has no seed 7
    assert len(job_runs("gen_canary", ("dev",), "mistral", "s42")) == 3   # the canary may run any trained model


def test_seed_labels():
    assert seed_label(None) == "base" and seed_label(7) == "s7"


def test_run_seeds_are_the_training_seed_lock_of_the_full_train_set():
    """One seed source (models.allowed_seeds); the literal table stays pinned."""
    assert RUN_SEEDS == {"deepseek": (42, 11, 23, 7, 99), "codellama": (42, 11, 23), "mistral": (42, 11, 23)}
    assert RUN_SEEDS == {m: allowed_seeds(m, "all") for m in ("codellama", "deepseek", "mistral")}


GRID = [f"dev_grid_{v}k{k}_n{n}_r{r}" for v in ("", "v1_") for k in (4, 8, 12) for n in (1, 2, 3)
        for r in ("033", "050", "067")]


def test_every_dev_grid_label_plans_only_the_full_aware_run():
    """Labels fixed 24/09, shared with the scoring code: 27 grid points, before and after a compose change. Since
    04/10: the selection rule reads only CA at Full-aware, and a grid label changes only
    ④ (k, ratio) and ③ (examples), so its F0 prompts equal v0's and F1 has three sets: a grid label runs Full-aware
    (adapter, F2) only, on the pilot's partition, model and seed."""
    assert len(GRID) == len(set(GRID)) == 54
    for label in GRID:
        assert variant_spec(label) == variant_spec("dev_pilot_v0") and variant_arg(label) == label
        assert variant_conditions(label) == ("full_aware",)
        assert planned_runs(label, "dev") == [Run("deepseek", "s42", "full_aware")]
        assert job_runs(label, ("dev",), "deepseek", "s42") == [Run("deepseek", "s42", "full_aware")]
        with pytest.raises(ValueError, match="no run"):
            job_runs(label, ("dev",), "deepseek", "base")


def test_the_pilot_and_main_still_plan_every_condition():
    """The 04/10 change touches only the grid labels: dev_pilot_v0 and dev_pilot_v1 keep the five dev conditions, main
    keeps its 39 test runs
    and 17 aux_test runs (zero_shot and rag_only once on base weights, the three adapter conditions per seed)."""
    five = ["zero_shot", "rag_only", "ft_only", "full_neutral", "full_aware"]
    for v in ("dev_pilot_v0", "dev_pilot_v1"):
        assert variant_conditions(v) == tuple(five)
        assert [r.condition for r in planned_runs(v, "dev")] == five
        assert len(job_runs(v, ("dev",), "deepseek", "base")) == 2 and len(job_runs(v, ("dev",), "deepseek", "s42")) == 3
    assert variant_conditions("main") == tuple(five)
    assert len(planned_runs("main", "test")) == (2 + 3 * 5) + (2 + 3 * 3) * 2 == 39
    assert len(planned_runs("main", "aux_test")) == 2 + 3 * 5 == 17


@pytest.mark.parametrize("label", ["dev_grid_k5_n3_r050", "dev_grid_k8_n0_r050", "dev_grid_k8_n3_r50", "dev_grid_k8_n3",
                                   "dev_grid_k8_n3_r050_x", "dev_grid_v2_k8_n3_r050", "DEV_GRID_K8_N3_R050",
                                   "dev_grid_k08_n3_r050", "lc600"])
def test_malformed_variant_labels_are_refused_everywhere(label):
    import argparse

    with pytest.raises(ValueError, match="variant"):
        variant_spec(label)
    with pytest.raises(ValueError):
        planned_runs(label, "dev")
    with pytest.raises(argparse.ArgumentTypeError):
        variant_arg(label)
    assert any("variant" in p for p in row_problems(row(variant=label)))


def test_a_variant_that_is_not_a_string_is_a_problem_not_a_crash():
    assert any("variant" in p for p in row_problems(row(variant=["main"])))
    assert any("variant" in p for p in row_problems(row(variant=None)))
    assert any("model" in p for p in row_problems(row(model=["deepseek"])))


def test_a_row_of_a_grid_variant_is_valid_only_at_full_aware():
    """A grid output row of any other condition is a problem (the grid plans Full-aware only)."""
    assert row_problems(row(variant="dev_grid_k8_n3_r050", condition="full_aware", fmt="F2")) == []
    assert row_problems(row(variant="dev_grid_v1_k12_n1_r067", condition="full_aware", fmt="F2")) == []
    off = row_problems(row(variant="dev_grid_v1_k12_n1_r067", seed="base", condition="zero_shot", fmt="F0",
                           adapter_tensor_sha256=None))
    assert any("full_aware" in p and "dev_grid" in p for p in off), off
    assert any("full_aware" in p for p in row_problems(row(variant="dev_grid_k4_n1_r033", condition="ft_only",
                                                           fmt="F0")))


def test_prompt_sha256_is_the_sorted_key_dump_of_the_messages():
    m = [{"role": "user", "content": "Étape"}]
    import hashlib
    assert prompt_sha256(m) == hashlib.sha256(json.dumps(m, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert prompt_sha256(m) != prompt_sha256([{"content": "Étape ", "role": "user"}])


def row(**kw):
    r = {"record_id": "Sylius/Sylius|f.feature|10|s", "model": "deepseek", "seed": "s42", "condition": "full_aware",
         "variant": "main", "fmt": "F2", "text": "// FILE: a.php", "prompt_tokens": 100, "new_tokens": 12,
         "max_new_tokens": 6144, "stopped_at_eos": True, "prompts_sha256": "a" * 64, "prompt_sha256": "b" * 64,
         "adapter_tensor_sha256": "c" * 64, "leakcheck_receipt_sha256": "d" * 64, "finish_reason": "stop"}
    r.update(kw)
    return r


def test_a_valid_row_has_exactly_the_sixteen_fields():
    assert row_problems(row()) == [] and len(OUTPUT_FIELDS) == 16 and set(row()) == set(OUTPUT_FIELDS)
    assert row_key(row()) == ("Sylius/Sylius|f.feature|10|s", "deepseek", "s42", "full_aware", "main")


@pytest.mark.parametrize("bad, words", [
    ({"fmt": "F1"}, "fmt"),                                       # full_aware is F2
    ({"seed": "base"}, "base"),                                   # an adapter condition on base weights
    ({"condition": "zero_shot", "fmt": "F0"}, "adapter_tensor_sha256"),  # base run with an adapter digest
    ({"finish_reason": "abort"}, "finish_reason"),
    ({"stopped_at_eos": True, "finish_reason": "length"}, "stopped_at_eos"),
    ({"new_tokens": 7000}, "new_tokens"),
    ({"seed": "s7", "model": "mistral"}, "seed"),
    ({"leakcheck_receipt_sha256": None}, "leakcheck_receipt_sha256"),
    ({"variant": "lc600"}, "variant"),
])
def test_row_problems_name_the_broken_field(bad, words):
    assert any(words in p for p in row_problems(row(**bad)))


def test_eos_as_the_last_allowed_token_is_a_valid_stop():
    """new_tokens == max_new_tokens and the last token is EOS -> finish 'stop', stopped_at_eos."""
    assert row_problems(row(new_tokens=6144, max_new_tokens=6144)) == []


@pytest.mark.parametrize("bad, words", [
    ({"finish_reason": "length", "stopped_at_eos": False, "new_tokens": 12}, "length"),     # capped below its cap
    ({"new_tokens": 0}, "stop"),                                                          # a stop needs the EOS id
    ({"max_new_tokens": 4, "new_tokens": 3}, "min(6144"),                                 # not the job rule
    ({"model": "mistral", "seed": "s42", "prompt_tokens": 30000, "max_new_tokens": 6144}, "min(6144"),
])
def test_row_problems_tie_the_finish_to_the_counts_and_the_budget_rule(bad, words):
    """"length" means new_tokens == max_new_tokens, "stop" means at least the EOS id, and a job variant
    line has max_new_tokens = min(6144, window - prompt_tokens) of the model's pinned window."""
    assert any(words in p for p in row_problems(row(**bad)))


def test_the_budget_rule_is_for_job_variants_and_can_be_switched_off_by_the_writer():
    assert row_problems(row(model="mistral", seed="s42", prompt_tokens=30000, max_new_tokens=2768, new_tokens=5)) == []
    assert row_problems(row(variant="gen_canary", max_new_tokens=4, new_tokens=3)) == []
    assert row_problems(row(max_new_tokens=4, new_tokens=3), window_rule=False) == []
    assert row_problems(row(finish_reason="length", stopped_at_eos=False, new_tokens=6144)) == []


def test_row_problems_on_extra_or_missing_keys():
    r = row()
    r["partition"] = "test"
    assert any("keys" in p for p in row_problems(r))
    r = row()
    del r["text"]
    assert any("keys" in p for p in row_problems(r))
    assert row_problems(["not", "a", "dict"]) == ["output line is not a JSON object"]
