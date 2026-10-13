import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from v2tools.sap import (EVIDENCE_RULE, EXTRACT_STATUSES, MANIFEST_KEY, MANIFEST_SHAS, REPO_NOTES, ROBUST_RULE,
                         SapInputError, above_top4, accepted_missing_errors, agreement_ca_trace, answers, ca_prec,
                         ca_prec_denominator_errors, ca_prec_differences, call_stats, calls_by_repo, check_complete,
                         canonical_meta_errors, code_digest, code_files, code_tree_errors, committed_bytes,
                         committed_file_errors,
                         compose_dropped, confirmatory, contrasts, descriptive, direction, len_strata,
                         RQ3_METRICS, rq3_full_aware_by_model, rq3_profile,
                         controls_ca_prec, denominator_rows, history_errors, ledger_errors, ledger_rows,
                         results_differences,
                         design, design_records, docblock_by_repo, dose_response, eligibility, er_complete_errors,
                         er_scope, family, has_evidence, jsonl_rows, kappa_ca_trace, lazy_by_repo, listed_at,
                         listed_sha256, aux_design, lock_commit_errors, lock_precedes_scoring_errors, locked_flags,
                         manifest_design_errors, manifest_overlays,
                         mde_from_dev, overlay_errors, pair, per_run, per_seed_pairs, pooled, pre_t_errors, pre_t_flags,
                         provenance_errors, read_manifests, robust_label, robust_verdicts, robustness, rq_verdict,
                         scenario_values, score_row_errors, scores_meta_commit_errors, scores_meta_errors, seed_values,
                         select_dev_config, semv_sample,
                         subset_flags, type_i_rates, usable, usable_records, verdict_of, with_ca_prec)

CFG = {**json.loads((Path(__file__).resolve().parents[1] / "config" / "sap.json").read_text()),
       "reps_bootstrap": 500, "reps_permutation": 499}
CONDS = ("zero_shot", "rag_only", "ft_only", "full_neutral", "full_aware")
SEEDS = tuple(CFG["seeds"]["deepseek"])
CAL = {"r": {"base": {"tau_0": 0.9, "tau_0.04": 1.05}, "adapter": {"tau_0": 0.7, "tau_0.04": 0.95}},
       "r_sign_flip": {"base": {"tau_0": 0.8, "tau_0.04": 0.8}, "adapter": {"tau_0": 0.45, "tau_0.04": 0.47}}}


def rows(effect: dict[str, float], seed: int = 1, n_features: int = 40, seeds=SEEDS, variant: str = "main",
         seed_effect: dict[tuple[str, str], float] | None = None) -> list[dict]:
    """Two suites per scenario; CA ~ Bernoulli(0.4 + effect[condition] + seed_effect[(condition, seed)] + cluster shift);
    base conditions have seed "base". Every column sap.py reads is present, the three sub_* of 3C included (every
    record post_cutoff_all, none in the other two)."""
    rng, out = random.Random(seed), []
    for f in range(n_features):
        shift = rng.uniform(-0.2, 0.2)
        for sc in range(rng.randint(1, 5)):
            for cond in CONDS:
                for sd in (("base",) if cond in ("zero_shot", "rag_only") else seeds):
                    for suite in ("a", "b"):
                        p = min(1, max(0, 0.4 + effect.get(cond, 0) + (seed_effect or {}).get((cond, sd), 0) + shift))
                        ca = int(rng.random() < p)
                        out.append({"record_id": f"R|f{f}.feature|{sc}|{suite}", "scenario_key": f"R|f{f}.feature|{sc}",
                                    "feature_path": f"f{f}.feature", "repo": "R", "language": "php", "model": "deepseek",
                                    "seed": sd, "condition": cond, "variant": variant, "ca": ca, "ca_defined": True,
                                    "bv": ca, "er": None, "er_trace": None, "trace_action": None, "trace_then": None,
                                    "preexisting": False, "dose": f % 4, "sub_post_cutoff_all": True, "sub_time_coherent_30d": False,
                                    "sub_residual_clean": False, "extract_status": "ok", "excl_bv": False, "excl_ca": False,
                                    "insert_status": "ok", **{f"bv_{c}": 1 for c in ("syntax", "binding", "no_ambiguity", "arity", "grounded", "resolved")},
                                    "cap_hit": False, "open_resolution": False, "hallucinated": 0, "members_used": 1,
                                    "conv_style": 1.0, "conv_pattern": 1.0, "conv_naming": 1.0, "rouge_l": 0.5, "ca_ref_v3m": 0,
                                    "ca_ref_v3names": 0, "n_calls": 2, "call_precision": 0.5, "suite": suite,
                                    "partition": "test", "exec_core": False, "duplicate_reuse": False,
                                    "exact": False, "len_bucket": "le8192"})
    return out


def comps(conf: dict) -> dict:
    return {k: v for f in conf.values() for k, v in f["comparisons"].items()}


def test_scenario_values_average_suites_then_seeds():
    rs = [{"record_id": "a", "scenario_key": "S", "feature_path": "F", "repo": "R", "model": "m", "condition": "c", "variant": "main",
           "seed": sd, "ca": v, "ca_defined": True} for sd, v in (("s1", 1), ("s1", 0), ("s2", 1), ("s2", 1))]
    rs.append({**rs[0], "record_id": "x", "ca": 0})
    rs.append({**rs[0], "record_id": "y", "ca": 0, "ca_defined": False})
    vals = scenario_values(rs, "ca", usable("ca", {"x"}))
    assert vals[("m", "c", "main")]["S"] == ("R|F", pytest.approx(0.75))


def test_pairs_by_seed_then_averages_and_keeps_scenarios_apart():
    rs = [{"record_id": f"{s}|r", "scenario_key": s, "feature_path": "F", "repo": "R", "model": "m", "condition": c, "variant": "main",
           "seed": sd, "er": v, "ca_defined": True}
          for s, c, sd, v in (("A", "x", "s1", 1), ("A", "x", "s2", 0), ("B", "x", "s1", 1), ("B", "x", "s2", 1),
                              ("A", "y", "s1", 0), ("A", "y", "s2", 0), ("B", "y", "s2", 1), ("A", "z", "base", 1), ("B", "z", "base", 0))]
    sv = seed_values(rs, "er", lambda r: True)
    ps = per_seed_pairs(sv, ("m", "x", "main"), ("m", "y", "main"))
    assert ps == {"s1": {"A": ("R|F", 1.0)}, "s2": {"A": ("R|F", 0.0), "B": ("R|F", 0.0)}}  # B has no s1 value for y
    assert pooled(ps) == [("R|F", 0.5), ("R|F", 0.0)]
    vs_base = per_seed_pairs(sv, ("m", "x", "main"), ("m", "z", "main"))
    assert vs_base == {"s1": {"A": ("R|F", 0.0), "B": ("R|F", 1.0)}, "s2": {"A": ("R|F", -1.0), "B": ("R|F", 1.0)}}


def test_verdict_and_rq_rules():
    assert verdict_of(0.1, 0.01, 0.02, 0.05) == "supported" and verdict_of(-0.1, 0.01, 0.02, 0.05) == "contrary"
    assert verdict_of(0.1, 0.01, 0.2, 0.05) == "conditional_only_positive"
    assert verdict_of(-0.1, 0.01, None, 0.05) == "conditional_only_negative"
    assert verdict_of(0.1, 0.2, 0.01, 0.05) == "no_difference" and verdict_of(None, None, None, 0.05) == "not_run"
    assert rq_verdict(["supported"] * 3) == "full" and rq_verdict(["supported", "conditional_only_negative", "no_difference"]) == "partial"
    assert rq_verdict(["conditional_only_positive", "no_difference", "not_run"]) == "not_supported"
    assert rq_verdict(["supported", "supported", "contrary"]) == "contrary"


def test_true_effect_is_supported_and_null_is_not():
    conf = confirmatory(rows({"full_aware": 0.3}), CFG, set(), {})
    assert {v["verdict"] for v in comps(conf).values()} == {"supported"} and conf["RQ1"]["rq_verdict"] == "full"
    c = conf["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]["ca"]
    assert len(c["per_seed"]) == 5 and c["seeds_positive"] == 5 and c["two_level"]["n_seeds"] == 5
    null = confirmatory(rows({}, seed=5), CFG, set(), {})
    assert all(v["verdict"] in ("no_difference", "conditional_only_positive", "conditional_only_negative") for v in comps(null).values())
    for v in comps(conf).values():
        assert v["ca"]["p_holm"] >= v["ca"]["p"] and v["ca"]["two_level"]["p_holm"] >= v["ca"]["two_level"]["p"]


def test_seed_disagreement_gives_conditional_only():
    up = {("full_aware", s): e for s, e in zip(SEEDS, (0.45, 0.4, -0.05, -0.1, 0.0))}
    conf = confirmatory(rows({}, seed=11, n_features=60, seed_effect=up), CFG, set(), {})
    v = conf["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]
    assert v["verdict"] == "conditional_only_positive" and conf["RQ2"]["rq_verdict"] == "not_supported"
    down = {k: -e for k, e in up.items()}
    conf = confirmatory(rows({}, seed=11, n_features=60, seed_effect=down), CFG, set(), {})
    assert conf["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]["verdict"] == "conditional_only_negative"
    text = answers(conf)
    assert conf["RQ2"]["rq_verdict"] == "not_supported" and "Lưu ý" in text[-1] and "full_neutral" in text[-1]


def test_opposite_descriptive_metric_is_flagged():
    rs = rows({"full_aware": 0.3}, seed=2)
    for r in rs:
        r["bv"] = 1 - r["ca"]
    conf = confirmatory(rs, CFG, set(), {})
    assert conf["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]["descriptive"]["bv"]["opposite"] is True
    same = confirmatory(rows({"full_aware": 0.3}, seed=2), CFG, set(), {})
    assert same["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]["descriptive"]["bv"]["opposite"] is False


def test_eligibility_trace_and_self_contained():
    ctl = [{"record_id": "a", "condition": "gold", "er_status": "PASS", "trace_action": 1},
           {"record_id": "a", "condition": "stub", "er_status": "FAIL", "trace_action": 0},
           {"record_id": "b", "condition": "gold", "er_status": "PASS", "trace_action": 1},
           {"record_id": "b", "condition": "stub", "er_status": "PASS", "trace_action": 0},
           {"record_id": "c", "condition": "gold", "er_status": "FAIL", "trace_action": 1},
           {"record_id": "c", "condition": "stub", "er_status": "FAIL", "trace_action": 1}]
    e = eligibility(ctl, {"a": True, "b": False}, {"a", "b", "c"})
    assert e["er"] == {"a": True, "b": False, "c": False} and e["trace"] == {"a": True, "b": True, "c": False}
    assert e["trace@self_contained"] == {"a": True, "b": False, "c": False}
    r = {"record_id": "b", "trace_then": 1}
    assert usable("trace_then", set(), e)(r) and not usable("trace_then@self_contained", set(), e)(r)
    rs = [{"record_id": "a", "model": "m", "condition": "c", "variant": "main", "seed": "s", "ca": 1, "ca_defined": True, "trace_action": 1},
          {"record_id": "b", "model": "m", "condition": "d", "variant": "main", "seed": "s", "ca": 0, "ca_defined": True, "trace_action": 0}]
    assert agreement_ca_trace(rs, set(), e["trace"], "m")["ALL"]["n"] == 2
    assert agreement_ca_trace(rs, set(), e["trace@self_contained"], "m")["ALL"]["n"] == 1
    with pytest.raises(SapInputError, match="er_status"):
        eligibility([{"record_id": "a", "condition": "gold", "status": "PASS", "trace_action": 1}], {}, {"a"})


GOLD = {"record_id": "a", "condition": "gold", "er_status": "PASS", "trace_action": 1}
STUB = {"record_id": "a", "condition": "stub", "er_status": "FAIL", "trace_action": 0}


def test_eligibility_refuses_a_partial_er_control_file():
    """A record with a gold row and no stub row (or the reverse) is not silently ineligible."""
    assert eligibility([GOLD, STUB], {}, {"a"})["er"] == {"a": True}
    for ctl in ([GOLD], [STUB], [GOLD, STUB, {**GOLD, "record_id": "b"}], [GOLD, STUB, {**STUB, "record_id": "b"}]):
        with pytest.raises(SapInputError, match="partial ER control file"):
            eligibility(ctl, {}, {"a", "b"})


def test_eligibility_refuses_an_exec_core_record_without_rows():
    """An exec_core test record with no ER control row at all is not silently ineligible."""
    with pytest.raises(SapInputError, match=r"partial ER control file: 2 exec_core test records .*, e\.g\. b$"):
        eligibility([GOLD, STUB], {}, {"a", "b", "c"})
    with pytest.raises(SapInputError, match=r"partial ER control file: 1 .*, e\.g\. a$"):
        eligibility([], {}, {"a"})
    assert eligibility([], {}, set()) == {"er": {}, "er@self_contained": {}, "trace": {}, "trace@self_contained": {}}


def test_eligibility_refuses_duplicate_rows():
    """A duplicated (record_id, condition) row stops the analysis, whatever it holds (no last-wins)."""
    for extra in (GOLD, {**GOLD, "er_status": "FAIL"}):
        with pytest.raises(SapInputError, match=r"1 duplicated \(record_id, condition\) .*, e\.g\. \('a', 'gold'\)$"):
            eligibility([GOLD, STUB, extra], {}, {"a"})
    with pytest.raises(SapInputError, match=r"2 duplicated .*\('a', 'gold'\)$"):
        eligibility([GOLD, STUB, STUB, GOLD], {}, {"a"})


def test_eligibility_refuses_stray_rows():
    """Rows of a record outside the exec_core test records stop the analysis, complete or not."""
    for ctl in ([GOLD, STUB, {**GOLD, "record_id": "b"}, {**STUB, "record_id": "b"}],
                [GOLD, STUB, {**STUB, "record_id": "b"}]):
        with pytest.raises(SapInputError, match=r"1 records in the ER control results are not exec_core test records, "
                                                r"e\.g\. b$"):
            eligibility(ctl, {}, {"a"})


def test_er_scope_leaves_out_the_dev_rows_of_the_er_controls_run():
    """The ER controls run also covers dev records (write_overlays' default ids: test exec_core +
    dev Sylius api*); their rows are left out of the SAP. A row of a test record that is not exec_core stays
    (eligibility refuses it as stray), and so does a row without record_id (eligibility reports the missing field)."""
    records = [{"id": "a", "exec_core": True}, {"id": "t", "exec_core": False}, {"id": "u", "exec_core": True}]
    dev = [{**GOLD, "record_id": "d"}, {**STUB, "record_id": "d"}]
    t = {**STUB, "record_id": "t"}
    no_id = {k: v for k, v in GOLD.items() if k != "record_id"}
    rows, expected = er_scope([GOLD, dev[0], STUB, t, dev[1], no_id], records)
    assert rows == [GOLD, STUB, t, no_id] and expected == {"a", "u"}
    assert er_scope([], []) == ([], set())
    ok = [{**GOLD, "record_id": "u"}, {**STUB, "record_id": "u"}]
    rows, expected = er_scope([GOLD, STUB, *dev, *ok], records)
    assert eligibility(rows, {}, expected)["er"] == {"a": True, "u": True}
    with pytest.raises(SapInputError, match=r"not exec_core test records, e\.g\. t$"):
        eligibility(er_scope([GOLD, STUB, *dev, *ok, t], records)[0], {}, expected)
    with pytest.raises(SapInputError, match=r"lacks \['record_id'\]"):
        eligibility(er_scope([GOLD, STUB, *dev, *ok, no_id], records)[0], {}, expected)


ER_DONE = {"expected": 2, "rows": 2, "missing": [], "extra": [], "duplicates": [], "suspect": [],
           "scope": {"conditions": ["gold", "stub"], "limit": None}, "scope_missing": [], "complete": True,
           "running": False, "manifest_sha256": "m"}


def test_er_complete_errors_read_the_er_complete_json_next_to_each_file(tmp_path):
    """The ER harness's er_results contract (0c42865): the results are read only when the er_complete.json of their run
    directory says complete, and not running, for the ER manifest given; one message per problem, one check per
    directory."""
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "er_complete.json").write_text(json.dumps(ER_DONE))
    assert er_complete_errors([good / "er_results.jsonl", good / "part2.jsonl"], "m") == []
    assert er_complete_errors([], "m") == []
    status = bad / "er_complete.json"
    assert er_complete_errors([good / "er_results.jsonl", bad / "er_results.jsonl"], "m") == [f"{status} is missing"]
    not_done, running = f"{status} complete is not true", f"{status} says the batch is still running"
    other = f"{status} manifest_sha256 is not the sha256 of the ER manifest given here"
    for st, want in (({**ER_DONE, "complete": False}, [not_done]), ({**ER_DONE, "complete": "true"}, [not_done]),
                     ({k: v for k, v in ER_DONE.items() if k != "complete"}, [not_done]),
                     ({**ER_DONE, "running": True}, [running]), ({**ER_DONE, "complete": False, "running": True},
                                                                 [not_done, running]),
                     ({k: v for k, v in ER_DONE.items() if k != "running"}, []),
                     ({**ER_DONE, "manifest_sha256": "old"}, [other]),
                     ({k: v for k, v in ER_DONE.items() if k != "manifest_sha256"}, [other]),
                     ({**ER_DONE, "complete": False, "manifest_sha256": "old"}, [not_done, other])):
        status.write_text(json.dumps(st))
        assert er_complete_errors([bad / "a.jsonl", bad / "b.jsonl"], "m") == want
    status.write_text(json.dumps({**ER_DONE, "manifest_sha256": None}))
    assert er_complete_errors([bad / "a.jsonl"], None) == [other]  # a missing manifest never matches
    for text in ("{", "[true]", ""):
        status.write_text(text)
        assert er_complete_errors([bad / "er_results.jsonl"], "m") == [f"{status} is not a JSON object"]


def test_manifest_overlays_hash_the_raw_lines_as_the_er_harness_does(tmp_path):
    """overlay_sha256 is the sha256 of the raw manifest line, the file split on "\\n" only, blank lines
    skipped (er/manifest.py: load_manifest, Entry.overlay_sha256); U+2028 inside a JSON string stays."""
    lines = [json.dumps({"record_id": "a", "condition": "gold", "scenario_name": "x y\x85z"}, ensure_ascii=False),
             json.dumps({"record_id": "a", "condition": "stub", "files": {}})]
    p = tmp_path / "manifest.jsonl"
    p.write_text(lines[0] + "\n\n" + lines[1] + "\n", encoding="utf-8")
    h = lambda line: hashlib.sha256(line.encode("utf-8")).hexdigest()
    assert manifest_overlays(p) == {("a", "gold"): h(lines[0]), ("a", "stub"): h(lines[1])}
    assert h(lines[0]) == hashlib.sha256(p.read_bytes().split(b"\n")[0]).hexdigest()
    p.write_text(lines[1] + "\n" + lines[0] + "\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(SapInputError, match=r"ER manifest .* lists \('a', 'stub'\) twice"):
        manifest_overlays(p)
    for bad in ("{", "[1]", '{"record_id": "a"}'):  # refused as bad input, not a crash
        p.write_text(lines[0] + "\n" + bad + "\n", encoding="utf-8")
        with pytest.raises(SapInputError, match=r"ER manifest .* line 2 is not a JSON object with record_id and "
                                                r"condition"):
            manifest_overlays(p)


def test_overlay_errors_tie_each_er_row_to_its_manifest_line():
    """Every ER control row given carries the overlay_sha256 of the manifest line of its (record_id,
    condition); a row with another one (or none), or of a pair the manifest does not list, is reported."""
    ov = {("a", "gold"): "g", ("a", "stub"): "s", ("b", "gold"): "bg"}
    ok = [{**GOLD, "overlay_sha256": "g"}, {**STUB, "overlay_sha256": "s"}]
    assert overlay_errors(ok, ov) == [] and overlay_errors([], ov) == []
    stale = "ER control rows have an overlay_sha256 that is not the sha256 of their manifest line, e.g. "
    assert overlay_errors([ok[0], {**STUB, "overlay_sha256": "g"}], ov) == [f"1 {stale}('a', 'stub')"]
    assert overlay_errors([GOLD, STUB], ov) == [f"2 {stale}('a', 'gold')"]
    neg = {**GOLD, "condition": "neg", "overlay_sha256": "g"}
    unlisted = "1 ER control rows are not in the ER manifest given here, e.g. ('a', 'neg')"
    assert overlay_errors([*ok, neg], ov) == [unlisted]
    assert overlay_errors([neg, STUB], ov) == [unlisted, f"1 {stale}('a', 'stub')"]


A, B, U, Z = (f"R|{x}.feature|1|s" for x in "abuz")
SR_CONTROLS = [{"record_id": r, "control": c, "ca_defined": (r != U) == (c == "gold")} for r in (A, B, U)
               for c in ("gold", "stub-v3")]
SR_EXCLUSIONS = [{"record_id": B, "reason": "gold_fails_bv", "excl_bv": True, "excl_ca": True},
                 {"record_id": U, "reason": "ca_undefined", "excl_bv": False, "excl_ca": True}]


def sr_row(rid: str, **kw) -> dict:
    """A score row of `rid` consistent with SR_CONTROLS and SR_EXCLUSIONS."""
    return {"record_id": rid, "scenario_key": rid.rsplit("|", 1)[0], "repo": "R", "feature_path": rid.split("|")[1],
            "model": "deepseek", "condition": "full_aware", "variant": "main", "seed": "s42", "excl_bv": rid == B,
            "ca_defined": rid != U, "ca": 1, "bv": 1, "er": None, "extract_status": "ok", "n_calls": 2,
            "call_precision": 0.5, "language": "php", "suite": "s", "exec_core": False, "partition": "test",
            "preexisting": False, "dose": 1, "ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "ca_ref_top4": 0,
            "php_tag_stripped": False, "php_docblock_steps": 0, "php_attribute_steps": 2, "unread_step_annotations": 0,
            "cuke_escaped_patterns": 0, "cuke_custom_types": 0, **kw}


SR_GOOD = [sr_row(A, er=1), sr_row(A, er=0, seed="s7"),
           sr_row(A, condition="zero_shot", variant=None, seed="base", er=0), sr_row(B, bv=None), sr_row(U, ca=None),
           sr_row(A, model="mistral"), sr_row(A, variant="k4")]


def sr_check(*extra, full=True) -> list[str]:
    return score_row_errors(SR_GOOD + list(extra), SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, full)


def ex(rid: str, seed: str = "s42") -> str:
    return f"('{rid}', 'deepseek', 'full_aware', 'main', '{seed}')"


def test_score_row_errors_cross_check_the_locked_controls():
    """A score row's excl_bv is the locked exclusions.jsonl's (False when the record is not listed), its ca_defined that
    of the record's gold row in the locked controls.jsonl, and in full mode every anchor-model main row of an
    ER-eligible record has an er value."""
    assert sr_check() == [] and sr_check(full=False) == []
    bv = "score rows have an excl_bv that is not the one of the locked exclusions.jsonl, e.g. "
    ca = ("score rows have a ca_defined that is not the one of their record's gold row in the locked controls.jsonl, "
          "e.g. ")
    er = "score rows of ER-eligible records (anchor model, main) have no er value, e.g. "
    assert sr_check(sr_row(A, er=1, excl_bv=True)) == [f"1 {bv}{ex(A)}"]
    assert sr_check(sr_row(B, excl_bv=False), sr_row(A, er=1, seed="s1", excl_bv=None)) == [f"2 {bv}{ex(A, 's1')}"]
    assert sr_check(sr_row(U, ca_defined=True)) == [f"1 {ca}{ex(U)}"]
    assert sr_check(sr_row(A, er=1, ca_defined=False, seed="s99")) == [f"1 {ca}{ex(A, 's99')}"]
    assert sr_check(sr_row(A, seed="s99")) == [f"1 {er}{ex(A, 's99')}"]
    assert sr_check(sr_row(A, seed="s99"), full=False) == []
    assert sr_check(sr_row(A, seed="s99", excl_bv=True, ca_defined=False)) == \
        [f"1 {bv}{ex(A, 's99')}", f"1 {ca}{ex(A, 's99')}", f"1 {er}{ex(A, 's99')}"]


def test_score_row_errors_refuse_unknown_records_wrong_keys_missing_values_and_filled_flags():
    """A row of a record with no gold row in the locked controls.jsonl, a scenario_key, repo or feature_path that is
    not that of the record_id repo|feature_path|scenario_line|suite, no ca on a row with ca_defined or no bv on a row
    without excl_bv (a crash before), or an input row that already carries filled_missing, whatever its value (such
    rows once skipped every check) stops the run."""
    unknown = "score rows are of records with no gold row in the locked controls.jsonl, e.g. "
    key = ("score rows have a scenario_key, repo or feature_path that is not that of their record_id "
           "(repo|feature_path|scenario_line|suite), e.g. ")
    value = "score rows have no ca value although ca_defined, or no bv value although not excl_bv, e.g. "
    carried = "score rows already carry filled_missing (only check_complete adds it), e.g. "
    assert sr_check(sr_row(Z, er=1)) == [f"1 {unknown}{ex(Z)}"]
    for bad in (sr_row(A, er=1, scenario_key="R|a.feature|2"), sr_row(A, er=1, repo="S"),
                sr_row(A, er=1, feature_path="b.feature"), sr_row(A, er=1, scenario_key=None)):
        assert sr_check(bad) == [f"1 {key}{ex(A)}"]
    odd = {**sr_row(A, er=1), "record_id": "R|a.feature|1", "scenario_key": "R|a.feature"}  # not four parts
    assert sr_check(odd) == [f"1 {unknown}{ex('R|a.feature|1')}", f"1 {key}{ex('R|a.feature|1')}"]
    assert sr_check(sr_row(A, er=1, ca=None), sr_row(A, er=1, bv=None, seed="s7")) == [f"2 {value}{ex(A)}"]
    assert sr_check(sr_row(A, er=1, filled_missing=True)) == [f"1 {carried}{ex(A)}"]
    assert sr_check(sr_row(A, er=1, filled_missing=False)) == [f"1 {carried}{ex(A)}"]
    assert len(sr_check(sr_row(A, excl_bv=True, ca_defined=False, filled_missing=True))) == 4


def test_score_row_errors_refuse_an_extract_status_the_sap_does_not_count():
    """per_run counts every status of EXTRACT_STATUSES (multi_file_fence, the fifth, included) so that the
    counts sum to the rows; a status outside it (or none) would drop out of the sum unseen, so it stops the run."""
    assert EXTRACT_STATUSES == ("ok", "multi_file_fence", "unclosed", "malformed", "none")
    assert sr_check(*(sr_row(A, er=1, extract_status=st) for st in EXTRACT_STATUSES)) == []
    status = "score rows have an extract_status that is not one of EXTRACT_STATUSES, e.g. "
    assert sr_check(sr_row(A, er=1, extract_status="multi_file"), sr_row(A, er=1, extract_status=None, seed="s7")) == \
        [f"2 {status}{ex(A)}"]


def test_score_row_errors_require_call_precision_and_n_calls():
    """D11 (c): call_precision decides CA_prec, so it is a required score column, with n_calls: None exactly when the
    output has no project call (n_calls = 0, 3C's call_overlap), otherwise in [0, 1]."""
    assert sr_check(sr_row(A, er=1, n_calls=0, call_precision=None),
                    sr_row(A, er=1, call_precision=0.0, seed="s7")) == []
    calls = ("score rows lack call_precision or n_calls, or have a call_precision that does not fit n_calls (None "
             "exactly when n_calls is 0, else in [0, 1]), e.g. ")
    missing = {k: v for k, v in sr_row(A, er=1).items() if k != "call_precision"}
    no_call = {k: v for k, v in sr_row(A, er=1, n_calls=0).items() if k != "call_precision"}  # no call, no column
    for bad in (missing, no_call, sr_row(A, er=1, call_precision=None), sr_row(A, er=1, n_calls=0, call_precision=0.5),
                sr_row(A, er=1, call_precision=1.5), sr_row(A, er=1, n_calls=None),
                sr_row(A, er=1, n_calls=True), sr_row(A, er=1, n_calls=-1, call_precision=0.5)):  # bool, < 0
        assert sr_check(bad) == [f"1 {calls}{ex(A)}"]


def test_ca_prec_is_ca_when_call_precision_reaches_tau():
    """D11 (c) (design specification at 3739dd6): CA_prec = CA × [call_precision ≥ τ], τ = 0.25 fixed in config/sap.json
    before any output; call_precision None (an output without project calls) counts as 0. The controls and the taus of
    the descriptive controls table are pre-registered in the config too."""
    assert CFG["ca_prec_tau"] == 0.25 and CFG["ca_prec_control_taus"] == [0.1, 0.25, 0.5]
    assert CFG["ca_prec_controls"] == ["gold", "gold-helpers", "gold-helpers-renamed", "swap", "stub-v3m", "stub-v3all",
                                       "stub-v3names", "stub-v3top4"]  # D11 (d): top4, at most 4 calls
    for ca, cp, want in ((1, 0.25, 1), (1, 0.2499, 0), (1, None, 0), (0, 1.0, 0), (1, 1.0, 1), (None, 0.5, 0)):
        assert ca_prec({"ca": ca, "call_precision": cp}, 0.25) == want
    assert ca_prec({"ca": 1}, 0.25) == 0
    rs = [{"ca": 1, "call_precision": 0.3}, {"ca": 1, "call_precision": 0.1}]
    assert [r["ca_prec"] for r in with_ca_prec(rs, 0.25)] == [1, 0] and "ca_prec" not in rs[0]


def test_confirmatory_on_ca_prec_repeats_the_ca_procedure():
    """D11 (c): every confirmatory comparison of RQ1 and RQ2 runs again on CA_prec with the same procedure (both tests,
    Holm per family and per test, verdicts, one n and one two-level n), without the descriptive metrics. When every
    call_precision reaches τ the results are CA's, and ca_prec_differences says so; otherwise it names every comparison
    and every RQ whose conclusion differs, whatever the direction (no selection)."""
    rs = with_ca_prec(rows({"full_aware": 0.3}), CFG["ca_prec_tau"])
    conf, prec = confirmatory(rs, CFG, set(), {}), confirmatory(rs, CFG, set(), {}, metric="ca_prec")
    assert set(prec) == set(conf)
    for f in conf:
        assert set(prec[f]["comparisons"]) == set(conf[f]["comparisons"]) and prec[f]["rq_verdict"] == conf[f]["rq_verdict"]
        for k, v in conf[f]["comparisons"].items():
            assert prec[f]["comparisons"][k] == {"ca_prec": v["ca"], "verdict": v["verdict"]}
    assert ca_prec_differences(conf, prec) == ["Kết luận theo CA_prec trùng với CA ở mọi phép so và mọi RQ."]
    low = with_ca_prec([{**r, "call_precision": 0.1} if r["condition"] == "full_aware" else r
                        for r in rows({"full_aware": 0.3})], CFG["ca_prec_tau"])
    prec = confirmatory(low, CFG, set(), {}, metric="ca_prec")
    assert {v["verdict"] for f in prec.values() for v in f["comparisons"].values()} == {"contrary"}
    diff = ca_prec_differences(conf, prec)
    assert len(diff) == 6 and diff[0] == "RQ1 · full_aware_vs_zero_shot: kết luận khác nhau giữa CA (supported) và " \
        "CA_prec (contrary)."
    assert "RQ2: kết luận mức RQ khác nhau giữa CA (full) và CA_prec (contrary)." in diff


def test_per_run_reports_every_lazy_reference_and_the_call_spread():
    """D11 (a), (b): per_run reports ca_ref_v3names next to v3m and v3all, each with its "CA above the lazy writer" rate
    (CA = 1 and the reference = 0), over the CA rows whose reference is known (3C writes None where it has no
    stub-v3names row; a filled row has none), and the spread of call_precision (median, Q1, Q3) and n_calls (median,
    p90) over the CA rows with an output and a defined value, by linear interpolation between order statistics."""
    base = {"model": "deepseek", "condition": "full_aware", "variant": "main", "seed": "s42", "ca_defined": True,
            "excl_bv": False, "bv": 1, "ca_ref_v3m": 0, "ca_ref_v3all": 1}
    specs = [(1, 0, 0.5, 2), (1, 1, 1.0, 1), (0, 0, None, 0), (1, None, 0.25, 4), (0, 1, 0.0, 3)]
    rs = [{**base, "record_id": f"R|f.feature|{i}|a", "ca": ca, "ca_ref_v3names": v3n, "call_precision": cp,
           "n_calls": nc} for i, (ca, v3n, cp, nc) in enumerate(specs)]
    out = per_run(rs + [{**base, "record_id": "R|f.feature|9|a", "ca": 0, "ca_ref_v3m": None, "ca_ref_v3all": None,
                         "filled_missing": True}], set())["deepseek|full_aware|main|s42"]
    assert list(out)[:7] == ["ca", "ca_ref_v3m", "ca_above_lazy", "ca_ref_v3all", "ca_above_v3all", "ca_ref_v3names",
                             "ca_above_v3names"]
    assert out["ca"] == {"n": 6, "rate": 3 / 6} and out["ca_ref_v3m"] == {"n": 5, "rate": 0.0}
    assert out["ca_above_lazy"] == {"n": 5, "rate": 3 / 5} and out["ca_above_v3all"] == {"n": 5, "rate": 0.0}
    assert out["ca_ref_v3names"] == {"n": 4, "rate": 0.5} and out["ca_above_v3names"] == {"n": 4, "rate": 0.25}
    assert out["call_precision"] == {"n": 4, "median": pytest.approx(0.375), "q1": pytest.approx(0.1875),
                                     "q3": pytest.approx(0.625)}
    assert out["n_calls"] == {"n": 5, "median": 2, "p90": pytest.approx(3.6)}
    assert "ca_ref_v3names" not in per_run([{k: v for k, v in r.items() if k != "ca_ref_v3names"} for r in rs],
                                           set())["deepseek|full_aware|main|s42"]
    assert call_stats([]) == {"call_precision": {"n": 0, "median": None, "q1": None, "q3": None},
                              "n_calls": {"n": 0, "median": None, "p90": None}}


def test_score_row_errors_require_the_lazy_references_and_php_tag_stripped():
    """D11 (d) (design specification at 4d8bf2a), §1.4 and §2.3 (fixed 26/09/2026): a score row carries the four lazy
    references ca_ref_v3m, ca_ref_v3all, ca_ref_v3names and ca_ref_top4 (0, 1, or None where the record has no such
    control, e.g. Ruby), php_tag_stripped (a bool, or None where there is no PHP text), php_docblock_steps and
    php_attribute_steps (an int >= 0, or None where there is no PHP text), and, by §2.2 of the design specification,
    unread_step_annotations, cuke_escaped_patterns and cuke_custom_types (an int >= 0, or None where there is no text);
    a missing column would drop a pre-registered rate unseen, so it stops the run."""
    assert sr_check(sr_row(A, er=1, ca_ref_top4=None, php_tag_stripped=None, php_docblock_steps=None,
                           php_attribute_steps=None, unread_step_annotations=None, cuke_escaped_patterns=None,
                           cuke_custom_types=None),
                    sr_row(A, er=1, ca_ref_top4=1, ca_ref_v3names=True, php_tag_stripped=True, php_docblock_steps=3,
                           seed="s7")) == []
    columns = ("score rows lack ca_ref_v3m, ca_ref_v3all, ca_ref_v3names, ca_ref_top4, php_tag_stripped, "
               "unread_step_annotations, cuke_escaped_patterns, cuke_custom_types, "
               "php_docblock_steps or php_attribute_steps, or have a lazy reference that is not 0, 1 or None, a "
               "php_tag_stripped that is not a bool or None, or a step count that is not an int >= 0 or None, e.g. ")
    for k in ("ca_ref_v3m", "ca_ref_v3all", "ca_ref_v3names", "ca_ref_top4", "php_tag_stripped", "php_docblock_steps",
              "php_attribute_steps", "unread_step_annotations", "cuke_escaped_patterns", "cuke_custom_types"):
        assert sr_check({x: v for x, v in sr_row(A, er=1).items() if x != k}) == [f"1 {columns}{ex(A)}"], k
    for bad in (sr_row(A, er=1, ca_ref_top4=2), sr_row(A, er=1, ca_ref_top4=0.5), sr_row(A, er=1, ca_ref_v3m="0"),
                sr_row(A, er=1, php_tag_stripped=1), sr_row(A, er=1, php_tag_stripped="yes"),
                sr_row(A, er=1, php_docblock_steps=-1), sr_row(A, er=1, php_docblock_steps=True),
                sr_row(A, er=1, php_attribute_steps=1.0), sr_row(A, er=1, unread_step_annotations=-1),
                sr_row(A, er=1, cuke_escaped_patterns=False), sr_row(A, er=1, cuke_custom_types="0")):
        assert sr_check(bad) == [f"1 {columns}{ex(A)}"]


def test_score_row_errors_hold_the_descriptive_columns_of_p26_to_their_contract():
    """The descriptive columns of P26 (fixed 06/10/2026, before any test output): members_used and hallucinated ints >=
    0 with hallucinated <= members_used, duplicate_reuse and exact bools, conv_* in [0, 1], len_bucket le8192 or
    gt8192; None or absent pass. A malformed value would leave the row out of a rate, or count it, unseen."""
    ok = dict(members_used=3, hallucinated=1, duplicate_reuse=False, exact=True, conv_style=1.0, conv_pattern=0,
              conv_naming=None, len_bucket="gt8192")
    assert sr_check(sr_row(A, er=1, **ok), sr_row(A, er=1, seed="s7", members_used=None, hallucinated=None,
                                                  duplicate_reuse=None, exact=None, len_bucket=None)) == []
    msg = ("score rows have a members_used or hallucinated that is not an int >= 0 (hallucinated <= members_used), a "
           "duplicate_reuse or exact that is not a bool, a conv_* outside [0, 1], or a len_bucket that is not le8192 "
           "or gt8192 (P26; None or absent are allowed), e.g. ")
    for bad in (dict(members_used=2.0), dict(members_used=-1), dict(hallucinated=-1), dict(hallucinated=True),
                dict(hallucinated=4),
                dict(duplicate_reuse=1), dict(exact="False"), dict(conv_style=1.5), dict(conv_pattern="1"),
                dict(conv_naming=-0.1), dict(len_bucket="gt4096")):
        assert sr_check(sr_row(A, er=1, **{**ok, **bad})) == [f"1 {msg}{ex(A)}"], bad


def test_score_row_errors_hold_the_er_fields_to_the_int_0_or_1_or_none():
    """Fixed 07/10/2026: er, er_trace, trace_action and trace_then are the int 0 or 1, or None (absent is None), as in
    the real ER results (dev_pilot_v0_0482a76 7c0237c9…, controls e721ebb0…: ints 0/1 and None only) and 3C's score
    rows, which copy them (describe.ER_FIELDS). Anything else stops the analysis: the SAP averages these values (usable,
    scenario_values) and agreement_ca_trace reads int(trace_action), so a "1" or a 2 would pass unseen into a level
    or the kappa table. A bool or a float is refused too: the real data has none."""
    good = [sr_row(A, er=1, er_trace=0, trace_action=1, trace_then=None),
            sr_row(A, er=0, seed="s7", er_trace=None, trace_action=0, trace_then=1),
            sr_row(A, er=1, seed="s11")]  # no er_trace, trace_action, trace_then key: None
    assert sr_check(*good) == [] and sr_check(*good, full=False) == []
    msg = ("score rows have an er, er_trace, trace_action or trace_then that is not the int 0 or 1, or None, e.g. ")
    for field in ("er", "er_trace", "trace_action", "trace_then"):
        for bad in (2, -1, "1", "0", 0.5, 1.0, 0.0, True, False, float("nan"), [1]):
            assert sr_check(sr_row(A, **{"er": 1, field: bad})) == [f"1 {msg}{ex(A)}"], (field, bad)
            assert sr_check(sr_row(A, model="mistral", **{field: bad}), full=False) == [
                f"1 {msg}('{A}', 'mistral', 'full_aware', 'main', 's42')"], (field, bad)  # every model, aux too
    assert sr_check(sr_row(A, er=1, trace_action="1"), sr_row(A, er=1, seed="s7", trace_action=2)) == [
        f"2 {msg}{ex(A)}"]  # the two values that would enter the kappa table


def test_per_run_reports_top4_against_ca_prec_and_the_php_tag_rate():
    """D11 (d) (design specification at 4d8bf2a): per_run adds ca_ref_top4 after the three other
    lazy references, with "CA_prec vượt mức lười top4" (CA_prec = 1 and ca_ref_top4 = 0) over the
    CA rows whose ca_ref_top4 is known; rows without ca_prec (sap.with_ca_prec) stop it. By §1.4
    (fixed 26/09/2026), also the rate of php_tag_stripped over the PHP rows with a text (not
    None; the scorer strips a leading <?php and a trailing ?> for every condition), per run."""
    base = {"model": "deepseek", "condition": "zero_shot", "variant": "main", "seed": "base", "ca_defined": True,
            "excl_bv": False, "bv": 1, "ca_ref_v3m": 0, "ca_ref_v3all": 0, "ca_ref_v3names": 0, "n_calls": 2,
            "language": "php"}
    specs = [(1, 0.5, 0, True), (1, 0.1, 0, False), (1, 0.5, 1, None), (0, None, 0, True), (1, 0.3, None, False)]
    rs = with_ca_prec([{**base, "record_id": f"R|f.feature|{i}|a", "ca": ca, "call_precision": cp, "ca_ref_top4": t4,
                        "php_tag_stripped": tag} for i, (ca, cp, t4, tag) in enumerate(specs)], CFG["ca_prec_tau"])
    rs.append({**rs[0], "record_id": "R|g.feature|1|a", "language": "java", "php_tag_stripped": False})
    out = per_run(rs, set())["deepseek|zero_shot|main|base"]
    assert list(out)[:9] == ["ca", "ca_ref_v3m", "ca_above_lazy", "ca_ref_v3all", "ca_above_v3all", "ca_ref_v3names",
                             "ca_above_v3names", "ca_ref_top4", "ca_prec_above_top4"]
    assert out["ca_ref_top4"] == {"n": 5, "rate": 1 / 5} and out["ca_prec_above_top4"] == {"n": 5, "rate": 2 / 5}
    assert out["php_tag_stripped"] == {"n": 4, "rate": 2 / 4}  # the java row and the None one are out
    assert "ca_ref_top4" not in per_run([{k: v for k, v in r.items() if k != "ca_ref_top4"} for r in rs], set())[
        "deepseek|zero_shot|main|base"]
    with pytest.raises(SapInputError, match="ca_prec"):
        per_run([{k: v for k, v in r.items() if k != "ca_prec"} for r in rs], set())



def test_per_run_reports_the_reader_counts_and_the_er_error_rate():
    """§2.2 of the design specification at c6999dc (fixed 27/09/2026), descriptive, per run (so per condition): for each
    of unread_step_annotations, cuke_escaped_patterns and cuke_custom_types, over the rows with a text (not None), the
    share with at least one and their sum; and the rate of er_status "ERROR" (the output stopped the ER run while
    loading, e.g. an attribute Behat rejects) over the rows with an er_status."""
    base = {"model": "deepseek", "condition": "rag_only", "variant": "main", "seed": "s42", "ca_defined": True,
            "excl_bv": False, "bv": 1, "ca": 1, "call_precision": 0.5, "ca_ref_v3m": 0, "ca_ref_v3all": 0,
            "ca_ref_v3names": 0, "ca_ref_top4": 0, "n_calls": 2, "language": "java", "php_tag_stripped": None}
    specs = [(0, 0, 0, "PASS"), (2, 1, 0, "ERROR"), (None, None, None, None), (1, 0, 3, "FAIL"), (0, 0, 0, "ERROR")]
    rs = with_ca_prec([{**base, "record_id": f"R|f.feature|{i}|a", "unread_step_annotations": u,
                        "cuke_escaped_patterns": e, "cuke_custom_types": c, "er_status": st}
                       for i, (u, e, c, st) in enumerate(specs)], CFG["ca_prec_tau"])
    out = per_run(rs, set())["deepseek|rag_only|main|s42"]
    assert out["unread_step_annotations"] == {"n": 4, "rate": 2 / 4, "sum": 3}
    assert out["cuke_escaped_patterns"] == {"n": 4, "rate": 1 / 4, "sum": 1}
    assert out["cuke_custom_types"] == {"n": 4, "rate": 1 / 4, "sum": 3}
    assert out["er_error"] == {"n": 4, "rate": 2 / 4}
    none = per_run([{**r, "er_status": None} for r in rs], set())["deepseek|rag_only|main|s42"]
    assert none["er_error"] == {"n": 0, "rate": None}

def test_ca_prec_above_top4_is_given_for_every_condition_of_the_families():
    """D11 (d) (design specification at 4d8bf2a): the rate "CA_prec vượt mức lười top4" next to CA_prec for the runs of
    every RQ1/RQ2 comparison: per condition of the families (anchor model, variant main, the seeds pooled), the rate of
    CA_prec = 1 and ca_ref_top4 = 0 over the usable CA rows (not excluded, not compose-dropped, CA defined) whose
    ca_ref_top4 is known; descriptive, no test."""
    rs = with_ca_prec(rows({"full_aware": 0.3}, n_features=8), CFG["ca_prec_tau"])
    first = rs[0]["record_id"]
    rs = [{**r, "ca_ref_top4": int(r["record_id"] == first)} for r in rs]
    rs += [{**r, "model": "codellama"} for r in rs[:50]] + [{**r, "variant": "k4"} for r in rs[:50]]
    out = above_top4(rs, CFG, {rs[1]["record_id"]}, {rs[2]["record_id"]})
    assert list(out) == ["full_aware", "zero_shot", "rag_only", "ft_only", "full_neutral"]
    for cond, v in out.items():
        keep = [r for r in rs if (r["model"], r["variant"], r["condition"]) == ("deepseek", "main", cond)
                and r["record_id"] not in {rs[1]["record_id"], rs[2]["record_id"]}]
        assert v == {"n": len(keep), "rate": sum(r["ca"] == 1 and r["record_id"] != first for r in keep) / len(keep)}
    assert above_top4([{**r, "ca_ref_top4": None} for r in rs], CFG, set(), set())["full_aware"] == \
        {"n": 0, "rate": None}


def test_docblock_steps_are_given_next_to_er_per_repo_and_run():
    """§2.3 of the design specification (fixed 26/09/2026), descriptive (the ER image registers only PHP 8 attribute
    steps, so a docblock step is undefined in ER while BV and CA take both forms): per repo and run (model, condition,
    variant; the seeds pooled), over the rows not compose-dropped with a php_docblock_steps (PHP outputs with a text):
    n, the rate of outputs with >= 1 docblock step, the docblock and attribute steps summed; next to the ER rate over
    the rows with an er value."""
    base = {"repo": "R", "model": "deepseek", "condition": "full_aware", "variant": "main", "er": None}
    specs = [("s1", 0, 3, 1), ("s1", 2, 1, 0), ("s2", 1, 0, None), ("s2", None, None, None), ("s3", 0, 2, 1)]
    rs = [{**base, "record_id": f"R|f.feature|{i}|a", "seed": sd, "php_docblock_steps": d, "php_attribute_steps": a,
           "er": er} for i, (sd, d, a, er) in enumerate(specs)]
    rs.append({**rs[0], "record_id": "R|gone.feature|1|a", "php_docblock_steps": 5})  # compose-dropped
    rs.append({**rs[0], "repo": "S", "record_id": "S|f.feature|1|a", "condition": "zero_shot", "seed": "base"})
    out = docblock_by_repo(rs, {"R|gone.feature|1|a"})
    assert out == {"R": {"deepseek|full_aware|main": {"n": 4, "docblock_rate": 0.5, "docblock_steps": 3,
                                                     "attribute_steps": 6, "er": {"n": 3, "rate": 2 / 3}}},
                   "S": {"deepseek|zero_shot|main": {"n": 1, "docblock_rate": 0.0, "docblock_steps": 0,
                                                    "attribute_steps": 3, "er": {"n": 1, "rate": 1.0}}}}
    assert docblock_by_repo([{**r, "php_docblock_steps": None, "php_attribute_steps": None, "er": None} for r in rs],
                            set()) == {}


def test_kappa_ca_trace_is_also_given_on_attribute_only_outputs():
    """P16 (fixed 26/09/2026, §2.3 of the design specification): the CA × ER-trace agreement (kappa and raw agreement,
    per run and pooled) of each eligibility, trace and trace@self_contained, is also given, next to it, over the outputs
    with php_docblock_steps == 0 (attribute steps only, which the ER image registers); an output with docblock steps, or
    without a count, is out of that version."""
    elig = {"trace": {"a": True, "b": True, "c": True}, "trace@self_contained": {"a": True, "b": False, "c": True}}
    rs = [{"record_id": rid, "model": "m", "condition": "c", "variant": "main", "seed": "s", "ca": ca,
           "ca_defined": True, "trace_action": ta, "php_docblock_steps": d}
          for rid, ca, ta, d in (("a", 1, 1, 0), ("b", 0, 0, 0), ("c", 1, 0, 2), ("a", 0, 0, None))]
    out = kappa_ca_trace(rs, set(), elig, "m")
    assert list(out) == ["trace", "trace@attribute_only", "trace@self_contained", "trace@self_contained@attribute_only"]
    assert out["trace"]["ALL"]["n"] == 4 and out["trace@attribute_only"]["ALL"]["n"] == 2
    assert out["trace@self_contained"]["ALL"]["n"] == 3 and out["trace@self_contained@attribute_only"]["ALL"]["n"] == 1
    assert out["trace@attribute_only"]["ALL"]["agreement"] == 1.0


def test_ca_prec_keeps_the_denominator_of_ca():
    """CA_prec has CA's denominator: usable("ca_prec") needs ca_defined and drops the excluded and compose-dropped
    records as CA does; confirmatory on CA_prec refuses comparisons of different n as it does on CA; and
    ca_prec_denominator_errors names every comparison whose CA_prec n (scenarios, clusters, two-level) is not CA's,
    which sap_run refuses (a CA_prec analysis run on other records)."""
    keep = usable("ca_prec", {"x"}, dropped={"d"})
    assert keep({"record_id": "a", "ca_defined": True, "ca_prec": 1})
    assert not keep({"record_id": "a", "ca_defined": False, "ca_prec": 1})
    assert not keep({"record_id": "x", "ca_defined": True, "ca_prec": 1})
    assert not keep({"record_id": "d", "ca_defined": True, "ca_prec": 1})
    rs = with_ca_prec(rows({"full_aware": 0.3}, n_features=10), CFG["ca_prec_tau"])
    gone = rs[0]["scenario_key"]
    uneven = [r for r in rs if not (r["scenario_key"] == gone and r["condition"] == "zero_shot")]
    with pytest.raises(SapInputError, match="confirmatory comparisons have different n"):
        confirmatory(uneven, CFG, set(), {}, metric="ca_prec")
    conf, prec = confirmatory(rs, CFG, set(), {}), confirmatory(rs, CFG, set(), {}, metric="ca_prec")
    assert ca_prec_denominator_errors(conf, prec) == []
    other = confirmatory(rs, CFG, {r["record_id"] for r in rs if r["scenario_key"] == gone}, {},
                         metric="ca_prec")  # CA_prec over other records: a scenario fewer
    errs = ca_prec_denominator_errors(conf, other)
    n = lambda c: (c["n_scenarios"], c["n_clusters"], c["two_level"]["n_scenarios"])
    ca, cp = (x["RQ1"]["comparisons"]["full_aware_vs_zero_shot"][m] for x, m in ((conf, "ca"), (other, "ca_prec")))
    assert n(cp)[0] == n(ca)[0] - 1 and len(errs) == 4
    assert errs[0] == f"RQ1 · full_aware_vs_zero_shot: CA_prec n {n(cp)} is not CA's {n(ca)} (scenarios, clusters, " \
                      "two-level)"


def test_ca_prec_differences_name_every_status_that_differs():
    """ca_prec_differences names a comparison or an RQ whose verdict differs, whatever the two
    statuses (not only supported / contrary)."""
    conf = {"RQ1": {"rq_verdict": "partial", "comparisons": {"a_vs_b": {"verdict": "no_difference"},
                                                             "a_vs_c": {"verdict": "conditional_only_negative"},
                                                             "a_vs_d": {"verdict": "not_run"}}}}
    prec = {"RQ1": {"rq_verdict": "not_supported",
                    "comparisons": {"a_vs_b": {"verdict": "conditional_only_positive"},
                                    "a_vs_c": {"verdict": "conditional_only_positive"},
                                    "a_vs_d": {"verdict": "not_run"}}}}
    assert ca_prec_differences(conf, prec) == [
        "RQ1 · a_vs_b: kết luận khác nhau giữa CA (no_difference) và CA_prec (conditional_only_positive).",
        "RQ1 · a_vs_c: kết luận khác nhau giữa CA (conditional_only_negative) và CA_prec (conditional_only_positive).",
        "RQ1: kết luận mức RQ khác nhau giữa CA (partial) và CA_prec (not_supported)."]


def test_call_stats_leave_the_filled_rows_out():
    """call_stats reads the rows with an output only; a filled row (check_complete) never counts, whatever it
    carries."""
    real = [{"call_precision": 0.5, "n_calls": 2}, {"call_precision": 1.0, "n_calls": 4}]
    filled = {"call_precision": 0.0, "n_calls": 9, "filled_missing": True}
    assert call_stats(real + [filled]) == call_stats(real)
    assert call_stats(real)["n_calls"]["n"] == 2


def test_lazy_references_per_repo_and_run():
    """The per-repo CA table carries the lazy references: per repo and run (the seeds pooled),
    each reference of LAZY_REFS the rows carry, over the CA rows whose reference is known, with its "vượt" rate
    (above_lazy)."""
    rs = with_ca_prec(rows({"full_aware": 0.3}, n_features=6), CFG["ca_prec_tau"])
    ps = "PrestaShop/PrestaShop"
    rs += [{**r, "record_id": ps + r["record_id"][1:], "repo": ps, "ca_ref_v3m": 1, "ca_ref_top4": 0} for r in rs[:60]]
    out = lazy_by_repo(rs, {rs[0]["record_id"]}, set())
    assert set(out) == {"R", ps}
    run = "deepseek|full_aware|main"
    xs = [r for r in rs if r["repo"] == "R" and r["condition"] == "full_aware" and r["record_id"] != rs[0]["record_id"]]
    assert out["R"][run]["ca_ref_v3m"] == {"n": len(xs), "rate": 0.0} and "ca_ref_top4" not in out["R"][run]
    assert out["R"][run]["ca_above_lazy"] == {"n": len(xs), "rate": sum(r["ca"] for r in xs) / len(xs)}
    assert set(out[ps]["deepseek|zero_shot|main"]) == {"ca_ref_v3m", "ca_above_lazy", "ca_ref_v3names",
                                                      "ca_above_v3names", "ca_ref_top4", "ca_prec_above_top4"}


FMT = {"zero_shot": "F0", "ft_only": "F0", "rag_only": "F1", "full_neutral": "F1", "full_aware": "F2"}


def attempt(status: str, k: int, last: bool = False) -> dict:
    return {"file": "receipt-generate.json" if last else f"receipt-generate.attempt-{k}.json", "status": status,
            "sha256": f"{k}" * 64}


def line_shas(model: str, cond: str, seed: str) -> dict:
    """The sha fields of a manifest line in the generation run's shapes: 64 lowercase hex, gen_code_sha256 an object
    {relative path: 64 lowercase hex}, as the real writer (ee6bc8c) writes it."""
    h = lambda k: hashlib.sha256(f"{k}:{model}:{cond}:{seed}".encode()).hexdigest()
    return {k: {"v2gpu/generate.py": h(k), "v2gpu/runs.py": h(k + "/runs")} if k == "gen_code_sha256" else h(k)
            for k in MANIFEST_SHAS}


def mline(rid: str, model: str = "deepseek", cond: str = "full_aware", seed: str = "s42", status: str = "ok",
          attempts: list | None = None, **kw) -> dict:
    """A line of the generation run's manifest.<variant>.jsonl, as its manifest contract defines it (fixed 26/09/2026;
    design specification §5 at 7af1673)."""
    x = {"model": model, "condition": cond, "seed": seed, "variant": "main", "record_id": rid, "fmt": FMT[cond],
         **line_shas(model, cond, seed), "status": status,
         "line_sha256": "l" * 64 if status == "ok" else None, "attempts": attempts,
         "receipt_status": "COMPLETE", "receipt_sha256": "r" * 64}
    if status == "missing":
        att = attempts if attempts is not None else [attempt("CRASHED", 1), attempt("CRASHED", 2, last=True)]
        x.update(attempts=att, receipt_status=att[-1]["status"] if att else "absent",
                 receipt_sha256=att[-1]["sha256"] if att else None)
    if status == "excluded":
        x["reason"] = "too_long_union"
    return {**x, **kw}


def write_manifest(tmp: Path, lines: list[dict], name: str = "main", cov: dict | None = None) -> tuple[Path, Path]:
    m, c = tmp / f"manifest.{name}.jsonl", tmp / f"coverage.{name}.json"
    m.write_text("".join(json.dumps(x) + "\n" for x in lines))
    miss = any(x.get("status") == "missing" for x in lines)
    c.write_text(json.dumps(cov if cov is not None else {"manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest(),
                                                        "status": "FAIL" if miss else "PASS",
                                                        "status_except_missing": "PASS"}))
    return m, c


def test_the_evidence_rule_is_one_named_constant():
    """Design specification §5 at 7af1673 (fixed 26/09/2026): a missing pair is accepted only when its job's last
    receipt is CRASHED or INCOMPLETE over at least 2 attempts, each with a sha256; REFUSED_* (operator-made),
    RECHECK_MISMATCH and an absent job (no attempt) never count."""
    assert EVIDENCE_RULE == {"statuses": ("CRASHED", "INCOMPLETE"), "min_attempts": 2}
    ok = lambda *st: has_evidence(mline("R|a|1|s", status="missing",
                                        attempts=[attempt(x, k, k == len(st)) for k, x in enumerate(st, 1)]))
    assert ok("CRASHED", "CRASHED") and ok("REFUSED_POLICY", "INCOMPLETE") and ok("CRASHED", "INCOMPLETE", "CRASHED")
    assert not ok("CRASHED") and not ok() and not ok("CRASHED", "REFUSED_POLICY")
    assert not ok("CRASHED", "RECHECK_MISMATCH")
    no_sha = mline("R|a|1|s", status="missing",
                   attempts=[attempt("CRASHED", 1), {**attempt("CRASHED", 2), "sha256": None}])
    assert not has_evidence(no_sha) and not has_evidence(mline("R|a|1|s"))


def test_read_manifests_checks_the_generation_contract(tmp_path):
    """The generation run's schema (fixed 26/09/2026): every line has the key (model, condition, seed, variant,
    record_id, fmt), the six sha fields, status ok / excluded / missing, a non-empty line_sha256 exactly when ok,
    attempts null when ok or excluded and a list when missing, whose last attempt gives receipt_status and
    receipt_sha256 ("absent" and null without one); evidence_eligible, when present, is never trusted: it must be what
    the attempts give. A pair is listed once over all the manifests. The coverage file hashes its manifest; status PASS
    allows no missing line, and status and status_except_missing both not PASS (RECHECK_MISMATCH) mean an incomplete
    generation."""
    good = [mline("R|a|1|s"), mline("R|b|1|s", status="missing"), mline("R|c|1|s", status="excluded"),
            mline("R|a|1|s", cond="zero_shot", seed="base", evidence_eligible=False),
            mline("R|d|1|s", status="missing", evidence_eligible=True)]
    m, c = write_manifest(tmp_path, good)
    lines, shas, problems = read_manifests([(m, c)])
    assert problems == [] and shas == {hashlib.sha256(m.read_bytes()).hexdigest(): hashlib.sha256(c.read_bytes())
                                       .hexdigest()}
    assert set(lines) == {("R|a|1|s", "deepseek", "full_aware", "main", "s42"), ("R|b|1|s", "deepseek", "full_aware",
                          "main", "s42"), ("R|c|1|s", "deepseek", "full_aware", "main", "s42"),
                          ("R|a|1|s", "deepseek", "zero_shot", "main", "base"),
                          ("R|d|1|s", "deepseek", "full_aware", "main", "s42")}
    e = ("R|z|1|s", "deepseek", "full_aware", "main", "s42")
    cases = {
        "lines lack a field of the manifest contract (the key, the six sha256, status, line_sha256, attempts; "
        "receipt_status and receipt_sha256 when missing, reason when excluded)":
            {k: v for k, v in mline("R|z|1|s").items() if k != "decision_sha256"},
        "lines have a status that is not ok, excluded or missing": mline("R|z|1|s", status="done"),
        "lines have a line_sha256 that does not fit their status (a non-empty one exactly when ok)":
            mline("R|z|1|s", line_sha256=None),
        "lines have attempts that do not fit their status (null when ok or excluded, a list of objects when missing)":
            mline("R|z|1|s", attempts=[]),
        "missing lines have a receipt_status or receipt_sha256 that is not their last attempt's (absent and null "
        "without one)": mline("R|z|1|s", status="missing", receipt_sha256="x" * 64),
        "lines have an evidence_eligible that their attempts do not give (EVIDENCE_RULE)":
            mline("R|z|1|s", status="missing", attempts=[], evidence_eligible=True)}
    for what, bad in cases.items():
        m, c = write_manifest(tmp_path, good + [bad])
        assert read_manifests([(m, c)])[2] == [f"{m}: 1 {what}, e.g. {e}"], (what, read_manifests([(m, c)])[2])
    m, c = write_manifest(tmp_path, good)
    m2, c2 = write_manifest(tmp_path, [mline("R|a|1|s", fmt="F9")], name="other")
    assert read_manifests([(m, c), (m2, c2)])[2] == [
        f"{m2}: 1 lines list a pair already listed (model, condition, seed, variant, record_id), e.g. "
        f"{('R|a|1|s', 'deepseek', 'full_aware', 'main', 's42')}"]
    msha = hashlib.sha256(m.read_bytes()).hexdigest()
    for cov, what in (({"manifest_sha256": "0" * 64, "status": "FAIL", "status_except_missing": "PASS"},
                       f"{c}: manifest_sha256 is not the sha256 of {m}"),
                      ({"manifest_sha256": msha, "status": "PASS", "status_except_missing": "PASS"},
                       f"{c}: status PASS, but {m} has 2 missing lines"),
                      ({"manifest_sha256": msha, "status": "FAIL", "status_except_missing": "FAIL"},
                       f"{c}: status FAIL and status_except_missing FAIL: the generation is not complete (a "
                       "RECHECK_MISMATCH fails both)"),
                      ([], f"{c}: not a JSON object")):
        c.write_text(json.dumps(cov))
        assert read_manifests([(m, c)])[2] == [what]


DESIGN_IDS = {"R|a|1|s", "R|b|1|s"}


def design_lines(ids=DESIGN_IDS) -> list[dict]:
    return [mline(rid, model=rk[0], cond=rk[1], seed=sd) for rk, sd in design(CFG) for rid in sorted(ids)]


def test_manifest_design_errors_tie_the_manifest_to_the_design():
    """Over the records of the test partition (rule fixed 26/09/2026), the ok and missing manifest lines are exactly
    the design (every model × condition × seed × record of design_records): a line of a run or seed outside the design
    (the deferred runs with --with-deferred only), of a test record outside design_records that compose did not drop,
    or of a record of no locked partition, and a design cell with no ok or missing line (none, or excluded while the
    locked compose file keeps the record) are errors. The lines of a compose-dropped (model, record) are free; the
    lines of aux_test records belong to the aux run."""
    ids = {m: set(DESIGN_IDS) for m in CFG["seeds"]}
    lines = {pair(x): x for x in design_lines()}
    known, test = DESIGN_IDS | {"X|a|1|s"}, set(DESIGN_IDS) | {"R|c|1|s"}
    check = lambda ls, deferred=False, dropped=None: manifest_design_errors(
        {pair(x): x for x in ls}, CFG, ids, dropped or {}, test, known | {"R|c|1|s"}, deferred)
    assert check(lines.values()) == []
    aux = mline("X|a|1|s", cond="zero_shot", seed="base")
    assert check([*lines.values(), aux]) == []
    cell = ("R|a|1|s", "mistral", "full_aware", "main", "s23")
    absent = "1 design cells have no ok or missing manifest line (none, or excluded while the locked compose file " \
             f"keeps the record), e.g. {cell}"
    assert check([x for k, x in lines.items() if k != cell]) == [absent]
    assert check([mline("R|a|1|s", model="mistral", seed="s23", status="excluded") if k == cell else x
                  for k, x in lines.items()]) == [absent]
    assert check([mline("R|a|1|s", model="mistral", seed="s23", status="missing") if k == cell else x
                  for k, x in lines.items()]) == []
    outside = "1 manifest lines outside the design (model, condition, variant, seed, record), e.g. "
    for extra in (mline("R|a|1|s", seed="s5"), mline("R|a|1|s", model="gpt"), mline("R|a|1|s", variant="k4"),
                  mline("R|a|1|s", cond="zero_shot", seed="s42"), mline("R|c|1|s")):
        assert check([*lines.values(), extra]) == [f"{outside}{pair(extra)}"], extra
    lc = mline("R|a|1|s", variant="lc600")
    assert check([*lines.values(), lc]) == [f"{outside}{pair(lc)}"] and check([*lines.values(), lc], True) == []
    gone = mline("R|c|1|s", model="mistral", seed="s23", status="excluded")
    assert check([*lines.values(), gone, mline("R|c|1|s", model="mistral", cond="zero_shot", seed="base")],
                 dropped={"mistral": {"R|c|1|s"}}) == []
    stray = mline("Q|a|1|s")
    assert check([*lines.values(), stray]) == [
        f"1 manifest lines of records of no locked partition, e.g. {pair(stray)}"]


def entry(line: dict, **kw) -> dict:
    return {**{k: line[k] for k in MANIFEST_KEY}, "reason": "rerun failed", "receipt_sha256": line["receipt_sha256"],
            **kw}


def test_accepted_missing_errors_apply_the_evidence_rule():
    """Design specification §5 at 7af1673 (fixed 26/09/2026): every accepted_missing entry (the key, reason,
    receipt_sha256) is a `missing` line of the manifests, once, with that line's receipt_sha256 (its last attempt's) and
    with evidence (EVIDENCE_RULE); every missing line is listed (none may pass unseen)."""
    miss = mline("R|b|1|s", status="missing")
    weak = mline("R|c|1|s", status="missing", attempts=[attempt("CRASHED", 1), attempt("REFUSED_POLICY", 2, True)])
    absent = mline("R|d|1|s", status="missing", attempts=[])
    lines = {pair(x): x for x in (mline("R|a|1|s"), miss)}
    assert accepted_missing_errors([entry(miss)], lines) == []
    e = pair(miss)
    assert accepted_missing_errors(None, lines) == [
        f"1 missing manifest lines and no --accepted-missing, e.g. {e}"]
    assert accepted_missing_errors([], lines) == [f"1 missing manifest lines are not in accepted_missing, e.g. {e}"]
    assert accepted_missing_errors([entry(miss), entry(miss)], lines) == [
        f"1 accepted_missing entries list a pair twice, e.g. {e}"]
    assert accepted_missing_errors([{k: v for k, v in entry(miss).items() if k != "reason"}], lines)[0] == (
        f"1 accepted_missing entries lack a field (model, condition, seed, variant, record_id, fmt, reason, "
        f"receipt_sha256), e.g. {e}")
    ok = pair(mline("R|a|1|s"))
    assert accepted_missing_errors([entry(miss), entry(mline("R|a|1|s"))], lines) == [
        f"1 accepted_missing entries are not a missing line of the manifests (same key and fmt), e.g. {ok}"]
    assert accepted_missing_errors([entry(miss, fmt="F0")], lines)[0].startswith(
        "1 accepted_missing entries are not a missing line of the manifests (same key and fmt)")
    assert accepted_missing_errors([entry(miss, receipt_sha256="x" * 64)], lines) == [
        f"1 accepted_missing entries have a receipt_sha256 that is not their manifest line's, e.g. {e}"]
    rule = ("accepted_missing entries have no evidence (last receipt CRASHED or INCOMPLETE over >= 2 attempts, each "
            "with a sha256; REFUSED_* and absent never count), e.g. ")
    for line, status in ((weak, "REFUSED_POLICY"), (absent, "absent")):
        both = {**lines, pair(line): line}
        assert accepted_missing_errors([entry(miss), entry(line)], both) == [f"1 {rule}{pair(line)}: {status}"]


def test_scores_meta_errors_tie_each_score_file_to_its_manifest():
    """Fixed 26/09/2026: each SCORES_META.json of the scored runs is an outputs run of the partition analysed, names a
    given --manifest (manifest_sha256) and its --coverage (coverage_sha256), and, when it names an accepted_missing
    file, that of --accepted-missing and its commit, dated before the scoring (scored_utc); every --manifest and the
    --accepted-missing are named by some SCORES_META. It also names its own scores.jsonl (scores_sha256), so a scores
    file edited after scoring (a tamper the manifest and the accepted_missing file never see, e.g. flipping ca on
    rows while both still pass) is refused too; a SCORES_META missing the field entirely is refused the same way
    (None is not the file's sha256 either)."""
    shas = {"m" * 64: "c" * 64, "n" * 64: "d" * 64}
    meta = {"mode": "outputs", "partition": "test", "manifest_sha256": "m" * 64, "coverage_sha256": "c" * 64,
            "accepted_missing_sha256": "a" * 64, "accepted_missing_commit": "abc1234",
            "scored_utc": "2026-09-27T10:00:00Z", "scores_sha256": "s" * 64}
    other = {**meta, "manifest_sha256": "n" * 64, "coverage_sha256": "d" * 64, "accepted_missing_sha256": None,
             "accepted_missing_commit": None, "scores_sha256": "t" * 64}
    full, when = "abc1234" + "0" * 33, "2026-09-27T16:59:59+07:00"
    scores_shas = ("s" * 64, "t" * 64)
    check = lambda *ms, acc="a" * 64, commit=full, at=when, scores=scores_shas: scores_meta_errors(
        [(Path(f"S{i}.json"), m) for i, m in enumerate(ms)], shas, "test", acc, commit, at, scores)
    assert check(meta, other) == []
    assert check({**meta, "mode": "controls"}, other) == ["S0.json: mode 'controls' is not outputs"]
    assert check({**meta, "partition": "aux_test"}, other) == ["S0.json: partition 'aux_test' is not test"]
    assert check({**meta, "manifest_sha256": "z" * 64}, other) == [
        f"S0.json: manifest_sha256 {'z' * 64} is not the sha256 of a given --manifest",
        f"--manifest {'m' * 64} is named by no SCORES_META"]
    assert check({**meta, "coverage_sha256": "d" * 64}, other) == [
        "S0.json: coverage_sha256 is not the sha256 of the --coverage of its manifest"]
    assert check({**meta, "accepted_missing_sha256": "b" * 64}, other) == [
        f"S0.json: accepted_missing_sha256 {'b' * 64} is not the sha256 of --accepted-missing ({'a' * 64})",
        "--accepted-missing is named by no SCORES_META"]
    assert check({**meta, "accepted_missing_commit": "def5678"}, other) == [
        f"S0.json: accepted_missing_commit def5678 is not --accepted-missing-commit {full}"]
    assert check(meta, other, at="2026-09-27T17:00:00+07:00") == [
        "S0.json: scored_utc 2026-09-27T10:00:00Z is not after the commit date 2026-09-27T17:00:00+07:00 of the "
        "accepted_missing file"]
    assert check({**meta, "scored_utc": "yesterday"}, other) == ["S0.json: scored_utc 'yesterday' is not a UTC time"]
    assert check(meta, other, acc=None, commit=None, at=None) == [
        f"S0.json: accepted_missing_sha256 {'a' * 64} names an accepted_missing file, but no --accepted-missing is "
        "given"]
    # the scores file's sha256 (computed by sap_run over --scores, independently of everything the meta
    # itself claims) must equal scores_sha256; a mismatch and an absent field are refused identically
    assert check({**meta, "scores_sha256": "z" * 64}, other) == [
        f"S0.json: scores_sha256 {'z' * 64} is not the sha256 of its --scores file ({'s' * 64})"]
    assert check({k: v for k, v in meta.items() if k != "scores_sha256"}, other) == [
        f"S0.json: scores_sha256 None is not the sha256 of its --scores file ({'s' * 64})"]


def test_score_row_errors_bind_each_row_to_its_ok_manifest_line():
    """With the manifests (fixed 26/09/2026), every score row is an `ok` line (record, model, condition, variant, seed),
    carries decision_sha256 and gen_code_sha256, and each sha256 it carries (prompts, prompt, adapter, decision, gen
    code, leak-check receipt) is its line's. Its line_sha256 (score_run's own copy of the ok line's line_sha256,
    `mline`'s fixed "l" * 64 here) must be that line's too, so a row cannot be passed off as the output of an ok line it
    was not actually scored from."""
    shas = lambda r: line_shas(r["model"], r["condition"], r["seed"])
    good = [{**r, **shas(r), "line_sha256": "l" * 64} for r in SR_GOOD]
    lines = {pair(r): mline(r["record_id"], model=r["model"], cond=r["condition"], seed=r["seed"],
                            variant=r.get("variant") or "main") for r in good}
    check = lambda *extra, ls=lines: score_row_errors(good + list(extra), SR_CONTROLS, SR_EXCLUSIONS,
                                                      {"er": {A: True, B: False}}, CFG, True, manifest=ls)
    assert check() == []
    not_ok = "score rows are not an ok line of the manifests (record_id, model, condition, variant, seed), e.g. "
    bad_sha = ("score rows lack decision_sha256 (64 lowercase hex) or gen_code_sha256 (a non-empty {relative path: 64 "
               "lowercase hex}), or carry a sha256 (prompts, prompt, adapter, "
               "decision, gen code, leak-check receipt) or a line_sha256 that is not their manifest line's, e.g. ")
    extra = {**good[0], "seed": "s99", "er": 1}
    assert check(extra) == [f"1 {not_ok}{ex(A, 's99')}"]
    k = pair(good[0])
    assert check(ls={**lines, k: {**lines[k], "status": "missing", "line_sha256": None}}) == [f"1 {not_ok}{ex(A)}"]
    for change in ({"decision_sha256": None}, {"gen_code_sha256": ""}, {"prompt_sha256": "x"},
                   {"leakcheck_receipt_sha256": "y"}):
        rows_ = [{**good[0], **change}] + good[1:]
        assert score_row_errors(rows_, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                                manifest=lines) == [f"1 {bad_sha}{ex(A)}"], change
    # strict (decision of 27/09/2026): every sha of MANIFEST_SHAS equals the line's, None included: a row without a sha
    # the line has is refused, and a base run whose line has no adapter tensor (None) must carry None too
    none = [{**good[0], "adapter_tensor_sha256": None}] + good[1:]
    assert score_row_errors(none, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=lines) == [f"1 {bad_sha}{ex(A)}"]
    base_lines = {**lines, k: {**lines[k], "adapter_tensor_sha256": None}}
    assert score_row_errors(none, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=base_lines) == []
    assert score_row_errors(good, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=base_lines) == [f"1 {bad_sha}{ex(A)}"]
    # a row that lacks a MANIFEST_SHAS key ENTIRELY (not just carries it as None) whose line
    # has a non-null value there is refused too; this pins the "shas" check against a `k in r and …` guard, which
    # would let such a row through (r.get(k) on a missing key already reads None, same as a key set to None: no such
    # guard exists to begin with, or should ever be reintroduced).
    absent = [{k_: v for k_, v in good[0].items() if k_ != "adapter_tensor_sha256"}] + good[1:]
    assert "adapter_tensor_sha256" not in absent[0]
    assert score_row_errors(absent, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=lines) == [f"1 {bad_sha}{ex(A)}"]
    # a row whose line_sha256 is not its ok line's is refused too (flipping a value on a row while the SCORES_META
    # and manifest both still pass, since nothing tied the row's own content to the specific manifest line it
    # claims); a row missing the key entirely is refused the same way (None != "l" * 64).
    wrong = [{**good[0], "line_sha256": "w" * 64}] + good[1:]
    assert score_row_errors(wrong, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=lines) == [f"1 {bad_sha}{ex(A)}"]
    no_line_sha = [{k_: v for k_, v in good[0].items() if k_ != "line_sha256"}] + good[1:]
    assert score_row_errors(no_line_sha, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG, True,
                            manifest=lines) == [f"1 {bad_sha}{ex(A)}"]


def test_committed_file_errors_need_origin_for_the_accepted_missing_file(tmp_path):
    """Design specification §5 at 7af1673: accepted_missing.jsonl counts only when committed and pushed to origin: the
    file equals git show <sha>:<path>, the commit is an ancestor of HEAD and on a branch of origin (another remote does
    not do); its fuller dates and commit date (%cI) are returned."""
    other, origin, repo = tmp_path / "other.git", tmp_path / "origin.git", tmp_path / "repo"
    for bare in (other, origin):
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "mirror", str(other))
    git(repo, "remote", "add", "origin", str(origin))
    f = repo / "accepted_missing.jsonl"
    f.write_text("{}\n")
    git(repo, "add", f.name)
    git(repo, "commit", "-q", "-m", "accept")
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "-q", "mirror", "main")
    git(repo, "fetch", "-q", "mirror")
    what = ("the accepted_missing file", "--accepted-missing-commit")
    assert committed_file_errors(f, sha, repo, *what, remote="origin") == ([f"commit {sha} is on no branch of origin"],
                                                                            None)
    assert committed_file_errors(f, sha, repo, *what)[0] == []  # any remote: the lock's rule
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    problems, info = committed_file_errors(f, sha, repo, *what, remote="origin")
    assert problems == [] and info["path"] == "accepted_missing.jsonl" and "CommitDate:" in info["fuller"]
    assert info["committed"] == git(repo, "show", "-s", "--format=%cI", sha)
    f.write_text("{}\n{}\n")
    assert committed_file_errors(f, sha, repo, *what, remote="origin")[0] == [
        f"the accepted_missing file accepted_missing.jsonl is not, byte for byte, the one committed at {sha}"]
    # with `once` (fixed 27/09/2026), the file is written on origin by that one commit, never edited
    assert committed_file_errors(f, sha, repo, *what, remote="origin", once=True)[0] == [
        f"the accepted_missing file accepted_missing.jsonl is not, byte for byte, the one committed at {sha}"]
    git(repo, "commit", "-q", "-am", "edit the accepted file")
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    again = git(repo, "rev-parse", "HEAD")
    assert committed_file_errors(f, again, repo, *what, remote="origin")[0] == []  # without once: the lock's rule
    assert committed_file_errors(f, again, repo, *what, remote="origin", once=True)[0] == [
        f"the accepted_missing file accepted_missing.jsonl is written on origin by {again}, {sha}, not by "
        f"--accepted-missing-commit {again} alone: it is written once and never edited"]


def test_committed_file_errors_once_survives_a_no_ff_merge(tmp_path):
    """Until 28/09/2026 the `once` check used a bare `git log --full-history`, which also lists a --no-ff merge that
    carries the file forward unchanged (TREESAME to the branch it merges in: default history simplification, which
    --full-history turns off, is what normally hides such a merge from a path's log), as a SECOND "write" of the path,
    so a side branch's push followed by a `git merge --no-ff` of it into main (both pushed) was a FALSE refusal though
    nobody edited the file. Fix: a commit counts only when its blob (absent counts as a value) differs from every
    parent's; a root commit counts when it has the file. (1) the file committed on a side branch, merged into main with
    --no-ff, both pushed: accepted (once). (2) a later commit on main that edits the file, pushed: now a genuine
    second write, refused. (3) the accepted_missing file exercises committed_file_errors directly (as
    sap_run.generation does for --accepted-missing); the same two steps are then repeated for a SCORES_META.json
    through scores_meta_commit_errors, the other caller of committed_file_errors with once=True."""
    remote, repo = tmp_path / "origin.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "commit", "-q", "--allow-empty", "-m", "root")
    git(repo, "checkout", "-q", "-b", "side")
    f = repo / "accepted_missing.jsonl"
    f.write_text("{}\n")
    git(repo, "add", f.name)
    git(repo, "commit", "-q", "-m", "accept")
    feat = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "-q", "origin", "HEAD:refs/heads/side")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(repo, "fetch", "-q", "origin")
    what = ("the accepted_missing file", "--accepted-missing-commit")
    # (1): the merge is TREESAME to the side branch for this path, so it is not a second write: accepted
    problems, info = committed_file_errors(f, feat, repo, *what, remote="origin", once=True)
    assert problems == [] and info["sha"] == feat and info["path"] == "accepted_missing.jsonl"
    assert "CommitDate:" in info["fuller"] and info["committed"] == git(repo, "show", "-s", "--format=%cI", feat)
    # (2): a later, real edit on main IS a genuine second write of the path
    f.write_text("{}\n{}\n")
    dated(repo, f"@{int(git(repo, 'show', '-s', '--format=%ct', feat)) + 1}", "commit", "-q", "-am", "edit on main")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(repo, "fetch", "-q", "origin")
    edited = git(repo, "rev-parse", "HEAD")
    assert committed_file_errors(f, edited, repo, *what, remote="origin", once=True)[0] == [
        f"the accepted_missing file accepted_missing.jsonl is written on origin by {edited}, {feat}, not by "
        f"--accepted-missing-commit {edited} alone: it is written once and never edited"]
    # the same, in words, for a SCORES_META.json through scores_meta_commit_errors:
    # (1) the meta committed on a side branch, merged into main with --no-ff, both pushed: accepted
    dated(repo, "2026-09-28T09:00:00+00:00", "commit", "-q", "--allow-empty", "-m", "lock2")
    lock2 = git(repo, "rev-parse", "HEAD")
    lock2_committed = git(repo, "show", "-s", "--format=%cI", lock2)
    git(repo, "checkout", "-q", "-b", "side2")
    meta = repo / "SCORES_META.json"
    meta.write_text(json.dumps({"scored_utc": "2026-09-28T09:30:00Z", "scores_sha256": "a" * 64}) + "\n")
    git(repo, "add", meta.name)
    dated(repo, "2026-09-28T10:00:00+00:00", "commit", "-q", "-m", "scores meta")
    meta_feat = git(repo, "rev-parse", "HEAD")
    git(repo, "push", "-q", "origin", "HEAD:refs/heads/side2")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", "merge side2", "side2")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(repo, "fetch", "-q", "origin")
    read = lambda: [(meta, json.loads(meta.read_text()))]
    problems, info = scores_meta_commit_errors(read(), meta_feat, repo, lock2_committed, lock2)
    assert problems == [] and info["sha"] == meta_feat and info["paths"] == ["SCORES_META.json"]
    # (2) a later commit on main that edits the meta, pushed: now a genuine second write, refused
    meta.write_text(json.dumps({"scored_utc": "2026-09-28T09:30:00Z", "scores_sha256": "b" * 64}) + "\n")
    dated(repo, "2026-09-28T11:00:00+00:00", "commit", "-q", "-am", "edit meta on main")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(repo, "fetch", "-q", "origin")
    meta_edit = git(repo, "rev-parse", "HEAD")
    assert scores_meta_commit_errors(read(), meta_feat, repo, lock2_committed, lock2)[0] == [
        f"the scores meta SCORES_META.json is not, byte for byte, the one committed at {meta_feat}"]
    assert scores_meta_commit_errors(read(), meta_edit, repo, lock2_committed, lock2)[0] == [
        f"the scores meta SCORES_META.json is written on origin by {meta_edit}, {meta_feat}, not by "
        f"--scores-meta-commit {meta_edit} alone: it is written once and never edited"]


def test_meta_anchor_refuses_symlinks_hard_links_and_a_repo_below_its_top_level(tmp_path):
    """Until 28/09/2026, two bypasses left the META anchor void, each letting an edited scores table through at exit 0
    (RQ1 0.2711 -> 0.5482). (a) A symlink at the canonical path to a META written once in another folder (here
    copied_meta), or a symlink in place of the folder scores/test: the path was
    resolved before it was compared and before it became the path given to git. (b) --repo a subfolder of the
    checkout: `git show <sha>:<rel>` read <rel> from the root, `git log -- <rel>` from the subfolder. Fix: the path
    is a fixed string under --repo (repo_rel), no step down to it is a symlink and the file has one link
    (link_errors), and --repo is the top level (top_level_errors; the log reads `:(top)<rel>`). canonical_meta_errors
    and committed_file_errors both refuse (a); committed_file_errors refuses (b) (sap_run's lock check, the first to
    run, meets it first). A symlink ABOVE --repo is no concern (macOS's /tmp): the same checkout is accepted through
    it. Exactly one META per run is pinned here too."""
    remote, repo = tmp_path / "origin.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))
    publish = lambda msg: (git(repo, "add", "-A"), git(repo, "commit", "-q", "-m", msg),
                           git(repo, "push", "-q", "origin", "HEAD:main"), git(repo, "fetch", "-q", "origin"),
                           git(repo, "rev-parse", "HEAD"))[-1]
    rel = "out/phase3/scores/test/SCORES_META.json"
    canon, elsewhere = repo / rel, repo / "copied_meta" / "SCORES_META.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text('{"scores_sha256": "e"}\n')
    once_elsewhere = publish("a META written once, in another folder")
    what = ("the scores meta", "--scores-meta-commit", "origin", True)
    assert committed_file_errors(elsewhere, once_elsewhere, repo, *what)[0] == []  # a valid anchor of ITS own path
    assert "canonical path" in canonical_meta_errors([elsewhere], repo, "test")[0]
    # (a) a symlink AT the canonical path: both checks once followed it to copied_meta and passed
    canon.parent.mkdir(parents=True)
    canon.symlink_to(Path("../../../../copied_meta/SCORES_META.json"))
    assert canon.read_text() == elsewhere.read_text()
    at_file = (f"the scores meta {rel}: {rel} is a symlink under --repo; the file and every folder above it must be "
               "real entries")
    assert canonical_meta_errors([canon], repo, "test") == [at_file]
    assert committed_file_errors(canon, once_elsewhere, repo, *what) == ([at_file], None)
    # (a) the folder scores/test replaced by a symlink to copied_meta
    canon.unlink()
    canon.parent.rmdir()
    canon.parent.symlink_to(Path("../../../copied_meta"), target_is_directory=True)
    assert canon.read_text() == elsewhere.read_text()
    at_folder = (f"the scores meta {rel}: out/phase3/scores/test is a symlink under --repo; the file and every folder "
                 "above it must be real entries")
    assert canonical_meta_errors([canon], repo, "test") == [at_folder]
    assert committed_file_errors(canon, once_elsewhere, repo, *what) == ([at_folder], None)
    # a real file at the canonical path, written once: accepted, also through a symlink ABOVE --repo
    canon.parent.unlink()
    canon.parent.mkdir()
    canon.write_text('{"scores_sha256": "c"}\n')
    real = publish("the META at its canonical path")
    assert canonical_meta_errors([canon], repo, "test") == []
    assert committed_file_errors(canon, real, repo, *what)[0] == []
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    assert committed_file_errors(alias / rel, real, alias, *what)[0] == []
    assert committed_file_errors(canon, real, alias, *what)[0] == []
    # a hard link to it opens no bypass (the bytes at the fixed path are what is compared) but is refused (fail closed)
    os.link(canon, tmp_path / "second_name.json")
    hard = f"the scores meta {rel} has 2 hard links; the anchored file must have one"
    assert canonical_meta_errors([canon], repo, "test") == [hard]
    assert committed_file_errors(canon, real, repo, *what) == ([hard], None)
    (tmp_path / "second_name.json").unlink()
    # exactly one META per run
    assert canonical_meta_errors([canon, canon], repo, "test") == [
        "exactly one SCORES_META.json per run: got 2 for partition test (the canonical path holds one META per "
        "partition, so one --scores file per run)"]
    # (b) --repo a subfolder: the root META rewritten and the same bytes at the subfolder's canonical path, one commit
    sub_meta = repo / "sub" / rel
    sub_meta.parent.mkdir(parents=True)
    canon.write_text('{"scores_sha256": "edited"}\n')
    sub_meta.write_text(canon.read_text())
    attack = publish("the root META rewritten, a copy under sub/")
    assert committed_file_errors(canon, attack, repo, *what)[0] == [  # from the top level: a second write, refused
        f"the scores meta {rel} is written on origin by {attack}, {real}, not by --scores-meta-commit {attack} alone: "
        "it is written once and never edited"]
    problems, info = committed_file_errors(sub_meta, attack, repo / "sub", *what)  # once accepted
    assert info is None and len(problems) == 1
    assert problems[0].startswith(f"--repo {repo / 'sub'} is not the top level of its git checkout ("), problems
    assert problems[0].endswith("); paths are read from the top level")


def test_the_history_git_reads_is_the_committed_one(tmp_path, monkeypatch):
    """Until 28/09/2026, a local `git replace` of the committed META blob, of the lock blob, or `git
    replace --graft` of a META onto the lock changed what the anchor checks read while the commits stayed
    the registered ones. Every git command now ignores replace objects (git_run: GIT_NO_REPLACE_OBJECTS=1,
    no GIT_* of the environment), committed_bytes gives the blob as committed, and history_errors refuses
    a --repo with replace refs, a grafts file or a shallow cut, and an environment that sets
    GIT_REPLACE_REF_BASE or GIT_GRAFT_FILE. Also, a path `x/..` through a symlink `x` lexically names the
    checked file but reads another; the path given must lead to the very file checked."""
    remote, repo = tmp_path / "origin.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))
    rel = "out/phase3/scores/test/SCORES_META.json"
    meta = repo / rel
    meta.parent.mkdir(parents=True)
    meta.write_text('{"scores_sha256": "original"}\n')
    git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", "meta")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(repo, "fetch", "-q", "origin")
    sha = git(repo, "rev-parse", "HEAD")
    what = ("the scores meta", "--scores-meta-commit", "origin", True)
    assert history_errors(repo) == [] and committed_file_errors(meta, sha, repo, *what)[0] == []
    assert committed_bytes(repo, sha, rel) == b'{"scores_sha256": "original"}\n'
    assert committed_bytes(repo, sha, "no/such/file") is None
    other = tmp_path / "other"  # a GIT_DIR of the caller's environment cannot point the checks at another repository
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    assert committed_bytes(repo, sha, rel) == b'{"scores_sha256": "original"}\n'
    monkeypatch.delenv("GIT_DIR")
    # `x/..` through a symlink x lexically IS the canonical path, but reads the file next to x's target
    evil = repo / "copied_meta"
    (evil / "inner").mkdir(parents=True)
    (evil / "SCORES_META.json").write_text('{"scores_sha256": "edited"}\n')
    (meta.parent / "x").symlink_to(evil / "inner", target_is_directory=True)
    dotdot = meta.parent / "x" / ".." / "SCORES_META.json"
    assert dotdot.read_text() != meta.read_text()
    problems, info = committed_file_errors(dotdot, sha, repo, *what)
    assert info is None and problems[0].startswith(f"the scores meta {dotdot} does not lead to {rel} under --repo")
    assert canonical_meta_errors([dotdot], repo, "test")[0].startswith(f"the scores meta {dotdot} does not lead to {rel}")
    (meta.parent / "x").unlink()
    # the META blob replaced locally; the disk file edited to match the replacement
    orig = git(repo, "rev-parse", f"{sha}:{rel}")
    meta.write_text('{"scores_sha256": "edited"}\n')
    git(repo, "replace", orig, git(repo, "hash-object", "-w", str(meta)))
    assert committed_bytes(repo, sha, rel) == b'{"scores_sha256": "original"}\n'  # replace objects ignored
    replaced = ("--repo has replace refs (refs/replace/" + orig + "): a local `git replace` can swap what a commit holds or "
                "its parents")
    assert history_errors(repo) == [replaced] and committed_file_errors(meta, sha, repo, *what) == ([replaced], None)
    git(repo, "replace", "-d", orig)
    meta.write_text('{"scores_sha256": "original"}\n')
    # a graft of parents is a replace ref as well
    git(repo, "commit", "-q", "--allow-empty", "-m", "later")
    git(repo, "replace", "--graft", "HEAD")
    assert history_errors(repo)[0].startswith("--repo has replace refs (refs/replace/")
    git(repo, "replace", "-d", git(repo, "rev-parse", "HEAD"))
    grafts = repo / ".git" / "info" / "grafts"
    grafts.parent.mkdir(exist_ok=True)
    grafts.write_text(git(repo, "rev-parse", "HEAD") + "\n")
    assert history_errors(repo) == ["--repo has a grafts file (.git/info/grafts): it can fake a commit's parents"]
    grafts.unlink()
    for var in ("GIT_REPLACE_REF_BASE", "GIT_GRAFT_FILE"):
        monkeypatch.setenv(var, str(tmp_path / "x"))
        assert history_errors(repo) == [f"{var} is set in the environment: a replace base or graft file is refused"]
        monkeypatch.delenv(var)
    assert history_errors(repo) == []
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{remote}", str(shallow)], check=True)
    assert history_errors(shallow) == ["--repo is a shallow clone: its cut history can hide an earlier write of a file"]


def test_step_8_compares_the_ledger_and_the_results_field_by_field():
    """The SAP results section of PREREG_LEDGER.md (heading "§6. Neo kết quả SAP"; checked since 28/09/2026) is read
    from its table only, one row per part, and equals the run in all six columns, the commit date of META_COMMIT in +07
    included (a date one second off is a mismatch); two SAP_RESULTS.json are equal in every field but RESULTS_IGNORED
    (the code's HEAD, the files' paths): changing an ignored field leaves them equal, changing any other one does not."""
    ledger = ("# PREREG_LEDGER\n\n## §5. Khác\n\n| test | x |\n\n## §6. Neo kết quả SAP\n\n"
              "| phần | LOCK_COMMIT (40 ký tự) | sha256(SAP_LOCK.json) | META_COMMIT | sha256(SCORES_META.json) | "
              "scores_sha256 | ngày commit của META_COMMIT (git %cI, quy về +07) |\n|---|---|---|---|---|---|---|\n"
              "| test | L | l | M | m | s | 2026-10-05T14:03:21+07:00 |\n"
              "| aux_test | L | l | A | a | t | 2026-10-05T15:00:00+07:00 |\n\n## §7. Sau\n\n| test | y |\n")
    rows = ledger_rows(ledger)
    assert rows == {"test": [["L", "l", "M", "m", "s", "2026-10-05T14:03:21+07:00"]],
                    "aux_test": [["L", "l", "A", "a", "t", "2026-10-05T15:00:00+07:00"]]}
    run = ["L", "l", "M", "m", "s", "2026-10-05T14:03:21+07:00"]
    assert ledger_errors(rows, "test", run) == []
    assert ledger_errors(rows, "test", run[:5] + ["2026-10-05T14:03:22+07:00"]) == [
        "§6 test: ngày commit của META_COMMIT (+07) is '2026-10-05T14:03:21+07:00' in the ledger but "
        "'2026-10-05T14:03:22+07:00' in the run from the fresh clone"]
    assert ledger_errors(rows, "test", ["L", "l", "X", "m", "s", run[5]])[0].startswith("§6 test: META_COMMIT is 'M'")
    assert ledger_errors({"test": rows["test"] * 2}, "test", run) == ["§6 of the ledger has 2 rows for test, not one"]
    assert ledger_errors(ledger_rows(ledger.replace("§6", "§8")), "test", run) == [
        "§6 of the ledger has 0 rows for test, not one"]
    ref = {"tool_commit": "c1", "n_rows": 5, "confirmatory": {"RQ1": 0.2711}, "lock": {"code": {"a": "h"}},
           "generation": {"manifests": [{"manifest": "/w/m.jsonl", "sha256": "h1", "coverage": "/w/c.json",
                                         "coverage_sha256": "h2"}],
                          "scores_meta": [{"file": "/w/SCORES_META.json", "scores_sha256": "s"}],
                          "scores_meta_commit": {"sha": "M"}},
           "accepted_missing": {"file": "/w/accepted_missing.jsonl", "sha256": "q"}}
    moved = json.loads(json.dumps(ref))
    moved["tool_commit"] = "c2"
    moved["generation"]["manifests"][0].update(manifest="/clone/m.jsonl", coverage="/clone/c.json")
    moved["generation"]["scores_meta"][0]["file"] = "/clone/SCORES_META.json"
    moved["accepted_missing"]["file"] = "/clone/accepted_missing.jsonl"
    assert results_differences(moved, ref) == []
    for edit, key in ((lambda r: r.update(n_rows=6), "n_rows"), (lambda r: r["confirmatory"].update(RQ1=0.5482), "confirmatory"),
                      (lambda r: r["generation"]["manifests"][0].update(sha256="h9"), "generation"),
                      (lambda r: r["generation"]["scores_meta"][0].update(scores_sha256="s9"), "generation"),
                      (lambda r: r["accepted_missing"].update(sha256="q9"), "accepted_missing"),
                      (lambda r: r.update(rehearsal=True), "rehearsal")):
        changed = json.loads(json.dumps(moved))
        edit(changed)
        assert results_differences(changed, ref) == [key], key


def test_lock_precedes_scoring_errors_refuses_a_lock_committed_after_scoring(tmp_path):
    """The lock's own commit date (committed_file_errors' %cI of --lock-commit; the fixture
    commit below is made right now, so this is "today", the real date, never a hardcoded or inferred one) must be
    strictly before every scored_utc a SCORES_META names; a lock committed at or after a scoring is refused. A meta
    whose scored_utc does not even parse is left for scores_meta_errors, which already refuses that."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    f = repo / "SAP_LOCK.json"
    f.write_text("{}\n")
    git(repo, "add", f.name)
    git(repo, "commit", "-q", "-m", "lock")
    sha = git(repo, "rev-parse", "HEAD")
    committed = git(repo, "show", "-s", "--format=%cI", sha)  # today's real commit date, taken from git, not inferred
    assert lock_precedes_scoring_errors([(Path("S0.json"), {"scored_utc": "2099-01-01T00:00:00Z"})], committed) == []
    assert lock_precedes_scoring_errors([(Path("S0.json"), {"scored_utc": "2000-01-01T00:00:00Z"})], committed) == [
        f"S0.json: scored_utc 2000-01-01T00:00:00Z is not after the lock's commit date {committed}"]
    assert lock_precedes_scoring_errors([(Path("S0.json"), {"scored_utc": committed})], committed) == [
        f"S0.json: scored_utc {committed} is not after the lock's commit date {committed}"]  # equal: not before it
    assert lock_precedes_scoring_errors([(Path("S0.json"), {"scored_utc": "yesterday"})], committed) == []
    assert lock_precedes_scoring_errors([(Path("S0.json"), {"scored_utc": "2099-01-01T00:00:00Z"}),
                                         (Path("S1.json"), {"scored_utc": "2000-01-01T00:00:00Z"})], committed) == [
        f"S1.json: scored_utc 2000-01-01T00:00:00Z is not after the lock's commit date {committed}"]



def dated(repo, date, *args) -> str:
    """git in `repo` with its author and committer dates set to `date` (an ISO time)."""
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                           "commit.gpgsign=false", *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def test_scores_meta_commit_errors_anchor_every_meta_after_the_lock_and_the_scoring(tmp_path):
    """Fixed 27/09/2026: a SCORES_META.json, whose scores_sha256 ties its scores file's bytes, counts only as committed
    right after the scoring and pushed to origin, once: each meta equals, byte for byte, git show
    <--scores-meta-commit>:<its path>, the commit is an ancestor of HEAD and on a branch of origin, it is the only
    commit of origin that writes that path (written once, never edited), and its commit date is strictly after the
    lock's commit date and after every scored_utc. A meta edited after that commit (with its scores file) is refused,
    and so is one edited, committed and pushed again; a message is given once. Since 28/09/2026 the meta commit must
    also be a DESCENDANT of --lock-commit (git merge-base --is-ancestor), so a meta cut from a branch older than the
    lock, then given a late date, is refused too, not only bounded by dates."""
    remote, repo = tmp_path / "origin.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))
    scored = "2099-01-01T00:00:00Z"
    dated(repo, "2020-01-01T00:00:00+00:00", "commit", "-q", "--allow-empty", "-m", "root")
    root = git(repo, "rev-parse", "HEAD")  # a point BEFORE the lock, to branch off later
    dated(repo, "2026-09-27T10:00:00+00:00", "commit", "-q", "--allow-empty", "-m", "lock")
    lock_commit = git(repo, "rev-parse", "HEAD")
    lock_committed = git(repo, "show", "-s", "--format=%cI", lock_commit)

    def metas(d, scored_utc=scored):
        out = [repo / d / "SCORES_META.json", repo / d / "aux" / "SCORES_META.json"]
        for p in out:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"scored_utc": scored_utc, "scores_sha256": "a" * 64}) + "\n")
        return out
    read = lambda ps: [(p, json.loads(p.read_text())) for p in ps]
    errs = lambda metas, sha: scores_meta_commit_errors(metas, sha, repo, lock_committed, lock_commit)

    def commit(date, push=True, branch="main"):
        git(repo, "add", "-A")
        dated(repo, date, "commit", "-q", "-m", "scores meta")
        if push:
            git(repo, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
            git(repo, "fetch", "-q", "origin")
        return git(repo, "rev-parse", "HEAD")
    a = metas("a")
    good = commit("2099-01-01T01:00:00+00:00")
    problems, info = errs(read(a), good[:10])
    assert problems == [] and info["sha"] == good and info["paths"] == ["a/SCORES_META.json", "a/aux/SCORES_META.json"]
    assert "CommitDate:" in info["fuller"] and info["committed"] == git(repo, "show", "-s", "--format=%cI", good)
    a[0].write_text(json.dumps({"scored_utc": scored, "scores_sha256": "b" * 64}) + "\n")  # the tampering edit
    assert errs(read(a), good) == (
        [f"the scores meta a/SCORES_META.json is not, byte for byte, the one committed at {good}"], None)
    again = commit("2099-01-01T02:00:00+00:00")  # the edit committed and pushed again (rule of 27/09/2026): refused
    assert errs(read(a), again)[0] == [
        f"the scores meta a/SCORES_META.json is written on origin by {again}, {good}, not by --scores-meta-commit "
        f"{again} alone: it is written once and never edited",
        f"the scores meta a/aux/SCORES_META.json is written on origin by {good}, not by --scores-meta-commit {again} "
        "alone: it is written once and never edited"]
    c = metas("c")
    unpushed = commit("2099-01-01T03:00:00+00:00", push=False)
    assert errs(read(c), unpushed) == ([f"commit {unpushed} is on no branch of origin"], None)  # once, not per meta
    d = metas("d")
    early = commit(scored.replace("Z", "+00:00"))
    committed = git(repo, "show", "-s", "--format=%cI", early)
    assert errs(read(d), early) == (
        [f"{p}: --scores-meta-commit {early} (committed {committed}) is not after its scored_utc {scored}"
         for p in d], None)
    e = metas("e", "later")  # no readable scored_utc (scores_meta_errors refuses it): the lock's date still bounds it
    old = commit("2000-01-01T00:00:00+00:00")
    committed = git(repo, "show", "-s", "--format=%cI", old)
    assert errs(read(e), old) == (
        [f"--scores-meta-commit {old} (committed {committed}) is not after the lock's commit date {lock_committed}"],
        None)
    assert errs(read(e), "0" * 40)[0] == [f"--scores-meta-commit {'0' * 40} is not a commit of {repo}"]
    # a meta cut from "root" (before the lock), on a branch that never merges the lock in, dated after it and
    # never edited (so every OTHER check here passes) is still refused: it is not a descendant of --lock-commit
    git(repo, "checkout", "-q", "-b", "off_branch", root)
    off_meta = metas("off")
    off = commit("2099-01-01T05:00:00+00:00", branch="off_branch")
    assert errs(read(off_meta), off) == (
        [f"--lock-commit {lock_commit} is not an ancestor of --scores-meta-commit {off}: the lock must precede the "
         "scores meta in this repository's history, not merely by date"], None)

def test_calls_by_repo_pool_the_seeds_of_each_condition():
    """D11 (b): call_precision and n_calls per repo and per condition (model, condition, variant; the seeds pooled),
    over the CA rows (not dropped, not excluded, CA defined) with an output; the report adds REPO_NOTES to
    PrestaShop."""
    rs = [{**r, "call_precision": 0.1 * (i % 5), "n_calls": i % 7} for i, r in enumerate(rows({}, n_features=6))]
    ps = "PrestaShop/PrestaShop"
    rs += [{**r, "record_id": ps + r["record_id"][1:], "repo": ps} for r in rs[:40]]
    out = calls_by_repo(rs, set(), set())
    assert set(out) == {"R", ps}
    want = call_stats([r for r in rs if r["repo"] == "R" and r["condition"] == "full_aware"])
    assert out["R"]["deepseek|full_aware|main"] == want and want["n_calls"]["n"] == 5 * len({r["record_id"] for r in rs
                                                                                               if r["repo"] == "R"})
    gone = rs[0]["record_id"]
    assert calls_by_repo(rs, {gone}, set())["R"]["deepseek|zero_shot|main"]["n_calls"]["n"] == \
        out["R"]["deepseek|zero_shot|main"]["n_calls"]["n"] - 1
    assert REPO_NOTES == {ps: "Ở PrestaShop, CA dễ đạt chỉ bằng các lời gọi hạ tầng (D11): đọc CA ở repo này cùng "
                              "call_precision và CA_prec."}


def test_controls_ca_prec_rates_at_each_pre_registered_tau():
    """D11 (decision of 25/09/2026): the CA_prec rate of each pre-registered control of controls.jsonl (which carries
    n_calls and call_precision) at each τ of ca_prec_control_taus, over the usable records with CA defined; a control
    row without call_precision stops the analysis; a control absent from the run is reported with n = 0."""
    ctl = [{"record_id": r, "control": c, "ca": ca, "ca_defined": d, "n_calls": 1, "call_precision": cp}
           for r, c, ca, d, cp in (("a", "gold", 1, True, 1.0), ("b", "gold", 1, True, 0.2), ("u", "gold", 1, True, 0.05),
                                   ("z", "gold", 1, True, 1.0), ("v", "gold", 0, False, None),
                                   ("a", "stub-v3names", 1, True, 0.3), ("b", "stub-v3names", 0, True, 0.9))]
    out = controls_ca_prec(ctl, {"a", "b", "u", "v"}, CFG)
    assert list(out) == CFG["ca_prec_controls"]
    assert out["gold"] == {"n": 3, "ca": 1.0, "ca_prec@0.1": 2 / 3, "ca_prec@0.25": 1 / 3, "ca_prec@0.5": 1 / 3}
    assert out["stub-v3names"] == {"n": 2, "ca": 0.5, "ca_prec@0.1": 0.5, "ca_prec@0.25": 0.5, "ca_prec@0.5": 0.0}
    assert out["swap"] == {"n": 0, "ca": None, "ca_prec@0.1": None, "ca_prec@0.25": None, "ca_prec@0.5": None}
    with pytest.raises(SapInputError, match=r"controls.jsonl rows of stub-v3names lack call_precision"):
        controls_ca_prec([{k: v for k, v in x.items() if k != "call_precision"} for x in ctl[-1:]], {"a", "b"}, CFG)


def test_score_row_errors_compare_the_record_fields_and_dose_with_the_locked_inputs():
    """A row's language, suite, exec_core and partition are its record's in the locked records, and its preexisting
    is any(create.preexisting) of that record (as 3C's describe.row writes them; they decide the strata); a
    row whose record the locked records lack is refused too. With a locked 3C dose table (fixed 25/09/2026), a row's
    dose is the table's for its (model, record), and a row absent from the table is refused."""
    recs = {rid: {"id": rid, "language": "php", "suite": "s", "exec_core": False, "partition": "test",
                  "create": [{"preexisting": False}]} for rid in (A, B, U)}
    dose = {(r["model"], r["record_id"]): 1 for r in SR_GOOD}
    check = lambda *extra, records=recs, table=dose: score_row_errors(SR_GOOD + list(extra), SR_CONTROLS, SR_EXCLUSIONS,
                                                                        {"er": {A: True, B: False}}, CFG, True, records,
                                                                        table)
    assert check() == [] and check(table=None) == []
    rec = ("score rows have a language, suite, exec_core, partition or preexisting that is not their record's in the "
           "locked records, e.g. ")
    for kw in ({"language": "java"}, {"suite": "t"}, {"exec_core": True}, {"partition": "aux_test"},
               {"preexisting": True}, {"suite": None}):
        assert check(sr_row(A, er=1, **kw)) == [f"1 {rec}{ex(A)}"], kw
    k4 = f"('{A}', 'deepseek', 'full_aware', 'k4', 's42')"  # the smallest key of SR_GOOD's five rows of A
    assert check(records={**recs, A: {**recs[A], "create": [{"preexisting": False}, {"preexisting": True}]}}) == \
        [f"5 {rec}{k4}"]
    assert check(records={k: v for k, v in recs.items() if k != A}) == [f"5 {rec}{k4}"]
    dosed = ("score rows have a dose that is not the one of the locked dose table for their (model, record), or no "
             "entry there, e.g. ")
    assert check(sr_row(A, er=1, dose=2)) == [f"1 {dosed}{ex(A)}"]
    assert check(sr_row(A, er=1, model="codellama")) == [f"1 {dosed}('{A}', 'codellama', 'full_aware', 'main', 's42')"]
    assert check(table={**dose, ("deepseek", A): None}) == [f"4 {dosed}{k4}"]  # SR_GOOD's DeepSeek rows of A


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                           "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout.strip()


def test_lock_commit_errors_authenticate_the_lock_by_a_published_commit(tmp_path):
    """SAP_LOCK.json counts only as committed: it lies inside --repo and equals, byte for byte, git show
    <sha>:<its path>; the commit is an ancestor of HEAD and on a branch of origin (published, fixed 27/09/2026: a push
    to another remote, a fork, does not do); the commit's git show -s
    --format=fuller (author and commit dates) is returned for SAP_RESULTS and SAP_REPORT."""
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "remote", "add", "origin", str(remote))
    lock = repo / "out" / "SAP_LOCK.json"
    lock.parent.mkdir()
    lock.write_text('{"inputs": {}}\n')
    git(repo, "add", "out/SAP_LOCK.json")
    git(repo, "commit", "-q", "-m", "lock")
    sha = git(repo, "rev-parse", "HEAD")
    assert lock_commit_errors(lock, sha, repo) == ([f"commit {sha} is on no branch of origin"], None)
    fork = tmp_path / "fork.git"
    subprocess.run(["git", "init", "-q", "--bare", str(fork)], check=True)
    git(repo, "remote", "add", "fork", str(fork))
    git(repo, "push", "-q", "fork", "main")
    git(repo, "fetch", "-q", "fork")
    assert lock_commit_errors(lock, sha, repo) == ([f"commit {sha} is on no branch of origin"], None)  # a fork only
    git(repo, "push", "-q", "origin", "main")
    git(repo, "fetch", "-q", "origin")
    problems, anchor = lock_commit_errors(lock, sha[:10], repo)
    assert problems == [] and anchor["sha"] == sha and anchor["path"] == "out/SAP_LOCK.json"
    assert "AuthorDate:" in anchor["fuller"] and "CommitDate:" in anchor["fuller"] and sha in anchor["fuller"]
    lock.write_text('{"inputs": {"config": "forged"}}\n')
    assert lock_commit_errors(lock, sha, repo)[0] == [f"the lock out/SAP_LOCK.json is not, byte for byte, the one "
                                                      f"committed at {sha}"]
    git(repo, "checkout", "-q", "-b", "side")
    git(repo, "commit", "-q", "-am", "forged lock")
    git(repo, "push", "-q", "origin", "side")
    git(repo, "fetch", "-q", "origin")
    side = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    lock.write_text('{"inputs": {"config": "forged"}}\n')  # the side commit's bytes, a commit HEAD does not contain
    assert lock_commit_errors(lock, side, repo)[0] == [f"commit {side} is not an ancestor of HEAD"]
    assert lock_commit_errors(lock, "0" * 40, repo)[0] == [f"--lock-commit {'0' * 40} is not a commit of {repo}"]
    assert lock_commit_errors(tmp_path / "SAP_LOCK.json", sha, repo)[0] == \
        [f"the lock {tmp_path / 'SAP_LOCK.json'} is not inside --repo {repo}"]
    assert lock_commit_errors(repo / "none.json", sha, repo)[0] == [f"the lock {repo / 'none.json'} does not exist"]


def test_robustness_views_carry_no_test_no_holm_and_no_verdict():
    """The exploratory robustness section ("độ vững") runs every comparison of RQ1 and RQ2,
    without its family label, on each view: the four subsets, the strata by repo and by language, the preexisting
    groups, leaving each repo out, and without the pre-T records (the frozen list); each gives n,
    the estimate and the 95% cluster-bootstrap CI only: no p, no Holm, no verdict."""
    rs = rows({"full_aware": 0.3}, n_features=12)
    q = [{**r, "record_id": "Q" + r["record_id"][1:], "scenario_key": "Q" + r["scenario_key"][1:], "repo": "Q",
          "language": "java"} for r in rs if r["feature_path"] in ("f0.feature", "f1.feature")]
    rs += q
    first = {r["record_id"] for r in rs if r["scenario_key"] == rs[0]["scenario_key"]}
    out = robustness(rs, CFG, set(), pre_t=first)
    assert list(out) == ["subset:post_cutoff_all", "subset:time_coherent_30d", "subset:residual_clean",
                         "subset:self_contained", "repo=Q", "repo=R", "language=java", "language=php",
                         "preexisting=True", "preexisting=False", "without_repo=Q", "without_repo=R", "without_pre_t",
                         "len_bucket=le8192", "len_bucket=gt8192"]
    comps = [f"{a}_vs_{b}" for pairs in CFG["families"].values() for a, b in pairs]
    for view in out.values():
        assert list(view) == comps and all(set(v) == {"n_scenarios", "n_clusters", "estimate", "ci95"}
                                           for v in view.values())
    n = lambda view: out[view]["full_aware_vs_full_neutral"]["n_scenarios"]
    total = len({r["scenario_key"] for r in rs})
    assert n("subset:post_cutoff_all") == total and n("subset:time_coherent_30d") == 0
    assert n("repo=Q") + n("repo=R") == total == n("language=java") + n("language=php") == n("preexisting=False")
    assert n("without_repo=R") == n("repo=Q") and n("without_pre_t") == total - 1
    none = robustness(rs, CFG, set())
    assert all((none["without_pre_t"][c]["n_scenarios"], none["without_pre_t"][c]["estimate"]) ==
               (none["subset:post_cutoff_all"][c]["n_scenarios"], none["subset:post_cutoff_all"][c]["estimate"])
               for c in comps)
    assert CFG["pre_t_flags"] == {"path": "evidence/pret_flags/FLAGGED_PRE_T.tsv",  # the frozen list
                                  "sha256": "dbf347d85f8b5092e382e4c9e3fdd014b5cf961697e44d51c929d3253ae81987"}


def test_pre_t_flags_read_the_frozen_tsv(tmp_path):
    """The frozen pre-T list (30706f2) is a TSV without header, one "<partition>\\t<record_id>" per
    line; it is read by partition. A line that is not two non-empty tab-separated fields without surrounding whitespace
    (a blank one inside, or a "\\r\\n" ending, included) stops the analysis."""
    p = tmp_path / "FLAGGED_PRE_T.tsv"
    p.write_text("test\tR|a.feature|1|s\ndev\tR|b.feature|2|s\ntest\tR|c.feature|3|s\naux_test\tX|d.feature|4|s\n")
    assert pre_t_flags(p) == {"test": {"R|a.feature|1|s", "R|c.feature|3|s"}, "dev": {"R|b.feature|2|s"},
                              "aux_test": {"X|d.feature|4|s"}}
    p.write_text("")
    assert pre_t_flags(p) == {}
    for bad in ("test R|a.feature|1|s\n", "test\tR|a\n\ntest\tR|b\n", "test\ta\tb\n", "\tR|a\n", "test\t\n",
                "test\tR|a\r\n", " test\tR|a\n"):
        p.write_text(bad)
        with pytest.raises(SapInputError, match=r"FLAGGED_PRE_T.tsv line \d+ is not <partition>\\t<record_id>"):
            pre_t_flags(p)


def test_pre_t_errors_check_the_config_sha_the_partitions_and_the_records():
    """The pre-T list (30706f2) is used only when its sha256 is the one config/sap.json names, its
    partitions are test, dev and aux_test, and every record it lists for test or aux_test is one of that partition's
    locked records (a record the robustness view could not drop would leave it silently unchanged)."""
    cfg = {"pre_t_flags": {"path": "evidence/pret_flags/FLAGGED_PRE_T.tsv", "sha256": "a" * 64}}
    known = {"test": {"R|a|1|s", "R|b|1|s"}, "aux_test": {"X|a|1|s"}}
    flags = {"test": {"R|a|1|s"}, "dev": {"R|d|1|s"}, "aux_test": {"X|a|1|s"}}
    assert pre_t_errors(cfg, Path("f.tsv"), "a" * 64, flags, known) == []
    assert pre_t_errors(cfg, Path("f.tsv"), "b" * 64, flags, known) == [
        f"f.tsv (sha256 {'b' * 64}) is not the sha256 config/sap.json names for the pre-T flags ({'a' * 64})"]
    assert pre_t_errors({}, Path("f.tsv"), "a" * 64, flags, known) == [
        f"f.tsv (sha256 {'a' * 64}) is not the sha256 config/sap.json names for the pre-T flags (None)"]
    bad = {"test": {"R|a|1|s", "R|z|1|s"}, "Test": {"R|b|1|s"}, "aux_test": {"R|a|1|s"}}
    assert pre_t_errors(cfg, Path("f.tsv"), "a" * 64, bad, known) == [
        "f.tsv lists partitions outside test, dev and aux_test: ['Test']",
        "f.tsv lists 1 test records that are not in the locked test records, e.g. R|z|1|s",
        "f.tsv lists 1 aux_test records that are not in the locked aux_test records, e.g. R|a|1|s"]


def test_a_conclusion_is_robust_only_when_ca_and_ca_prec_agree():
    """A confirmatory conclusion is "vững" (robust) only when CA and CA_prec give the same verdict (a verdict
    carries its direction), per comparison and per RQ; otherwise "chưa vững", with no choosing. Since 28/09/2026 each
    entry also carries "label" (robust_label of itself), so SAP_RESULTS.json states the same text the report prints,
    next to the boolean."""
    conf = {"RQ1": {"rq_verdict": "full", "comparisons": {"a_vs_b": {"verdict": "supported"},
                                                          "a_vs_c": {"verdict": "supported"}}},
            "RQ2": {"rq_verdict": "full", "comparisons": {"a_vs_d": {"verdict": "supported"}}}}
    prec = {"RQ1": {"rq_verdict": "partial", "comparisons": {"a_vs_b": {"verdict": "supported"},
                                                             "a_vs_c": {"verdict": "no_difference"}}},
            "RQ2": {"rq_verdict": "full", "comparisons": {"a_vs_d": {"verdict": "supported"}}}}
    out = robust_verdicts(conf, prec)
    assert out["rule"] == ROBUST_RULE and "vững" in ROBUST_RULE and "không chọn" in ROBUST_RULE
    no_est = {"ca_estimate": None, "ca_prec_estimate": None, "opposite_estimates": False}
    same = {"ca": "supported", "ca_prec": "supported", "robust": True, **no_est, "label": "vững"}
    assert out["comparisons"] == {"RQ1": {"a_vs_b": same, "a_vs_c": {"ca": "supported", "ca_prec": "no_difference",
                                                                     "robust": False, **no_est,
                                                                     "label": "chưa vững"}},
                                  "RQ2": {"a_vs_d": same}}
    assert out["rq"] == {"RQ1": {"ca": "full", "ca_prec": "partial", "robust": False, "label": "chưa vững"},
                         "RQ2": {"ca": "full", "ca_prec": "full", "robust": True, "label": "vững"}}
    assert robust_verdicts({}, {}) == {"rule": ROBUST_RULE, "comparisons": {}, "rq": {}}
    # every entry's "label" is exactly robust_label of itself, for every comparison and every RQ
    for fam in out["comparisons"].values():
        for e in fam.values():
            assert e["label"] == robust_label(e)
    for e in out["rq"].values():
        assert e["label"] == robust_label(e)



def test_no_difference_with_opposite_estimates_stays_robust_with_a_note():
    """P22 (fixed 27/09/2026, design specification §5): robust ("vững") means the same P9 verdict label; supported,
    contrary and conditional_only_± carry their direction, so the same label is the same direction; for no_difference
    the sign of an estimate that is not significant is noise and does not remove "vững": both estimates are kept
    (ca_estimate, ca_prec_estimate) and, when their signs differ, robust_label adds "(ước lượng trái dấu)" next to the
    label. The rule itself is unchanged."""
    nd = lambda e, m: {"verdict": "no_difference", m: {"estimate": e}}
    conf = {"RQ2": {"rq_verdict": "not_supported", "comparisons": {"a_vs_d": nd(0.03, "ca")}}}
    prec = {"RQ2": {"rq_verdict": "not_supported", "comparisons": {"a_vs_d": nd(-0.04, "ca_prec")}}}
    out = robust_verdicts(conf, prec)
    c = out["comparisons"]["RQ2"]["a_vs_d"]
    assert c == {"ca": "no_difference", "ca_prec": "no_difference", "robust": True, "ca_estimate": 0.03,
                 "ca_prec_estimate": -0.04, "opposite_estimates": True, "label": "vững (ước lượng trái dấu)"}
    assert robust_label(c) == "vững (ước lượng trái dấu)" and robust_label(out["rq"]["RQ2"]) == "vững"
    assert out["rq"]["RQ2"]["label"] == "vững"  # the RQ entry carries its own label too
    same = robust_verdicts(conf, {"RQ2": {**prec["RQ2"], "comparisons": {"a_vs_d": nd(0.01, "ca_prec")}}})
    assert robust_label(same["comparisons"]["RQ2"]["a_vs_d"]) == "vững"
    zero = robust_verdicts(conf, {"RQ2": {**prec["RQ2"], "comparisons": {"a_vs_d": nd(0.0, "ca_prec")}}})
    assert zero["comparisons"]["RQ2"]["a_vs_d"]["opposite_estimates"] is False  # zero has no sign
    assert robust_label({"robust": False, "opposite_estimates": False}) == "chưa vững"
    assert robust_label({"robust": False, "opposite_estimates": True}) == "chưa vững"  # the labels already differ
    assert "(ước lượng trái dấu)" in ROBUST_RULE and "no_difference" in ROBUST_RULE


def test_answers_carry_the_robustness_label():
    """P22 (fixed 27/09/2026): every pre-written answer, per comparison and per
    RQ, ends with its label of robust_verdicts (robust_label, "(ước lượng trái
    dấu)" included where it applies); without robust_verdicts nothing is added."""
    conf = {"RQ2": {"rq_verdict": "not_supported", "comparisons": {"full_aware_vs_full_neutral": {
        "verdict": "no_difference", "ca": {"estimate": 0.03, "ci95": [-0.03, 0.09], "p_holm": 0.41,
                                           "two_level": {"ci95": [-0.04, 0.1], "p_holm": 0.52}}}}}}
    prec = {"RQ2": {"rq_verdict": "not_supported", "comparisons": {"full_aware_vs_full_neutral": {
        "verdict": "no_difference", "ca_prec": {"estimate": -0.04}}}}}
    text = answers(conf, None, robust_verdicts(conf, prec))
    assert text[0].endswith(" Độ vững (CA và CA_prec): vững (ước lượng trái dấu).")
    assert text[1].endswith(" Độ vững (CA và CA_prec): vững.")
    assert answers(conf) == [t.rsplit(" Độ vững", 1)[0] for t in text]
    worse = {"RQ2": {"rq_verdict": "contrary", "comparisons": {"full_aware_vs_full_neutral": {
        "verdict": "contrary", "ca_prec": {"estimate": -0.2}}}}}
    assert all(t.endswith(" Độ vững (CA và CA_prec): chưa vững.") for t in answers(conf, None,
                                                                                   robust_verdicts(conf, worse)))

def test_denominators_per_repo_and_partition():
    """Records and scenarios per repo and partition of the locked records: total, usable for CA
    (the controls' gold records minus the mechanical exclusions minus those compose dropped for the anchor),
    mechanically excluded, compose-dropped for the anchor."""
    rec = lambda repo, f, line, suite, part="test": {"id": f"{repo}|f{f}.feature|{line}|{suite}", "repo": repo,
                                                     "feature_path": f"f{f}.feature", "scenario_line": line,
                                                     "partition": part}
    records = [rec("R", 0, 0, "a"), rec("R", 0, 0, "b"), rec("R", 0, 1, "a"), rec("Q", 0, 0, "a")]
    controls = [{"record_id": r["id"], "control": "gold"} for r in records]
    exclusions = [{"record_id": "R|f0.feature|1|a"}]
    compose = [{"record_id": "R|f0.feature|0|b", "model": None}, {"record_id": "Q|f0.feature|0|a", "model": "mistral"}]
    assert denominator_rows(records, controls, exclusions, compose, "deepseek") == [
        {"repo": "Q", "partition": "test", "records": 1, "scenarios": 1, "ca_records": 1, "ca_scenarios": 1,
         "excluded": 0, "dropped": 0},
        {"repo": "R", "partition": "test", "records": 3, "scenarios": 2, "ca_records": 1, "ca_scenarios": 1,
         "excluded": 1, "dropped": 1}]


def test_code_digest_hashes_the_fixed_files_and_every_sap_script_present(tmp_path):
    """Since 05/10/2026 the lock hashes v2tools/stats.py, v2tools/sap.py and every scripts/sap_*.py present when it is
    written (sap_verify.py, sap_swrap.py and any later one), no other script."""
    for p in ("v2tools/stats.py", "v2tools/sap.py", "scripts/sap_run.py", "scripts/sap_verify.py", "scripts/sap_new.py",
              "scripts/score_run.py", "scripts/sap_notes.txt", "v2tools/describe.py"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(p)
    assert code_files(tmp_path) == ("v2tools/stats.py", "v2tools/sap.py", "scripts/sap_new.py", "scripts/sap_run.py",
                                    "scripts/sap_verify.py")
    digest = code_digest(tmp_path)
    assert list(digest) == list(code_files(tmp_path))
    assert digest["scripts/sap_verify.py"] == hashlib.sha256(b"scripts/sap_verify.py").hexdigest()
    (tmp_path / "v2tools" / "stats.py").unlink()
    assert code_digest(tmp_path)["v2tools/stats.py"] is None  # a fixed file missing: sap_lock refuses (None)


def test_code_tree_errors_want_a_clean_git_checkout_of_the_locked_code(tmp_path):
    """Since 05/10/2026: git status of v2tools/stats.py, v2tools/sap.py and scripts/sap_*.py, untracked files included,
    must be empty, git must answer, and no hashed file may be a symlink. Other files do not count."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                        "commit.gpgsign=false", *args], check=True, capture_output=True)
    for p in ("v2tools/stats.py", "v2tools/sap.py", "scripts/sap_run.py", "scripts/score_run.py"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(p)
    assert any("git" in e for e in code_tree_errors(tmp_path))  # not a checkout
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "code")
    assert code_tree_errors(tmp_path) == []
    (tmp_path / "scripts" / "other.py").write_text("x")  # not a 3D code file
    (tmp_path / "scripts" / "score_run.py").write_text("changed")
    assert code_tree_errors(tmp_path) == []
    (tmp_path / "scripts" / "sap_new.py").write_text("x")
    (tmp_path / "v2tools" / "sap.py").write_text("changed")
    bad = code_tree_errors(tmp_path)
    assert len(bad) == 2 and any("scripts/sap_new.py" in e for e in bad) and any("v2tools/sap.py" in e for e in bad)
    (tmp_path / "scripts" / "sap_new.py").unlink()
    git("checkout", "--", "v2tools/sap.py")
    (tmp_path / "scripts" / "sap_link.py").symlink_to(tmp_path / "scripts" / "sap_run.py")
    git("add", "-A")
    git("commit", "-q", "-m", "a committed symlink")
    assert code_tree_errors(tmp_path) == ["scripts/sap_link.py is a symlink, not a regular file"]


def test_code_tree_errors_name_an_ignored_sap_script(tmp_path):
    """Added 05/10/2026: `git status` does not list an ignored file unless asked, yet code_files hashes it;
    a scripts/sap_*.py ignored through .git/info/exclude (or any .gitignore or global excludes) is an error too."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                        "commit.gpgsign=false", *args], check=True, capture_output=True)
    for p in ("v2tools/stats.py", "v2tools/sap.py", "scripts/sap_run.py"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(p)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "code")
    (tmp_path / ".git" / "info").mkdir(exist_ok=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("scripts/sap_ignored.py\n")
    (tmp_path / "scripts" / "sap_ignored.py").write_text("x")
    assert "scripts/sap_ignored.py" in code_files(tmp_path)
    bad = code_tree_errors(tmp_path)
    assert len(bad) == 1 and "scripts/sap_ignored.py" in bad[0], bad


def test_type_i_rates_are_read_from_the_locked_calibration():
    """The measured rejection rates under H0 (delta = 0) of test (1), test (2) and the "supported" rule, per comparison
    type and tau, at each alpha of CALIBRATION.json, with their Monte Carlo SE, for SAP_REPORT."""
    rate = lambda x, se: {"rate": x, "mc_se": se}
    at = lambda a, b, c: {"sign_flip": rate(a, 0.004), "two_level": rate(b, 0.003), "supported": rate(c, 0.002),
                          "contrary": rate(0.0, 0.0)}
    cal = {"results": {"base|tau=0|delta=0": {"n": 2000, "alpha=0.05": at(0.047, 0.0245, 0.0115),
                                              "alpha=0.01667": at(0.015, 0.0085, 0.005)},
                       "base|tau=0|delta=0.06": {"n": 2000, "alpha=0.05": at(0.7, 0.6, 0.5)},
                       "adapter|tau=0.04|delta=0": {"n": 2000, "alpha=0.05": at(0.34, 0.0525, 0.03)}}}
    assert type_i_rates(cal) == [
        {"type": "base", "tau": "0", "n": 2000,
         "alphas": {"alpha=0.05": {k: v for k, v in at(0.047, 0.0245, 0.0115).items() if k != "contrary"},
                    "alpha=0.01667": {k: v for k, v in at(0.015, 0.0085, 0.005).items() if k != "contrary"}}},
        {"type": "adapter", "tau": "0.04", "n": 2000,
         "alphas": {"alpha=0.05": {k: v for k, v in at(0.34, 0.0525, 0.03).items() if k != "contrary"}}}]


def test_bv_denominator_leaves_out_only_excl_bv_rows():
    r = {"record_id": "x", "bv": 1, "ca": 0, "ca_defined": False, "excl_bv": False, "excl_ca": True}
    assert usable("bv", {"x"})(r) and not usable("ca", {"x"})(r) and not usable("bv", set())({**r, "excl_bv": True})


def test_compose_dropped_records_are_out_of_every_metric():
    """Record d is compose-excluded for the anchor (every metric), b is a mechanical exclusion flagged excl_bv (out
    of CA and BV, in ER when eligible), u a mechanical exclusion with CA undefined (in BV, out of CA)."""
    row = lambda rid, **kw: {"record_id": rid, "ca": 1, "ca_defined": True, "bv": 1, "excl_bv": False, "excl_ca": False,
                             "er": 1, "er_trace": 1, **kw}
    rs = [row("d"), row("b", ca_defined=False, excl_bv=True, excl_ca=True), row("u", ca_defined=False, excl_ca=True), row("ok")]
    elig = {"er": {"d": True, "b": True, "u": True}, "trace": {"d": True, "b": True}}
    dropped = compose_dropped([{"record_id": "d", "model": "deepseek"}, {"record_id": "ok", "model": "mistral"}], "deepseek")
    keep = lambda m: [r["record_id"] for r in rs if usable(m, {"b", "u"}, elig, dropped)(r)]
    assert dropped == {"d"} and keep("ca") == ["ok"] and keep("bv") == ["u", "ok"]
    assert keep("er") == ["b", "u"] and keep("er_trace") == ["b"]
    full = rows({}, n_features=4)
    rid, g = full[0]["record_id"], "deepseek|full_aware|main|s42"
    a, b = per_run(full, set())[g], per_run(full, {rid}, {rid})[g]
    assert (b["bv"]["n"], b["ca"]["n"], b["syntax"]["n"]) == (a["bv"]["n"] - 1, a["ca"]["n"] - 1, a["syntax"]["n"] - 1)


def test_completeness_stops_on_missing_or_duplicate_and_fills_only_accepted():
    rs = rows({}, n_features=6)
    ids = {r["record_id"] for r in rs}
    ids, one = {r["record_id"] for r in rs}, {"deepseek": {r["record_id"] for r in rs}}
    assert check_complete(rs, one, CFG)[1] == {"records_covered": {"deepseek": len(ids)}, "missing_filled": 0}
    gone = next(r for r in rs if r["condition"] == "full_aware" and r["seed"] == "s7")
    partial = [r for r in rs if r is not gone]
    with pytest.raises(SapInputError, match="missing"):
        check_complete(partial, one, CFG)
    with pytest.raises(SapInputError, match="duplicated"):
        check_complete(rs + [dict(rs[0])], one, CFG)
    filled, info = check_complete(partial, one, CFG, {(gone["record_id"], "deepseek", "full_aware", "s7")})
    new = filled[-1]
    assert info["missing_filled"] == 1 and new["ca"] == 0 and new["filled_missing"] and new["sub_post_cutoff_all"] is True
    assert new["dose"] == gone["dose"] and new["scenario_key"] == gone["scenario_key"]
    assert per_run(filled, set())["deepseek|full_aware|main|s7"]["filled_missing"] == 1
    assert usable_records([{"record_id": i, "control": "gold"} for i in ids] + [{"record_id": "z", "control": "stub-v3"}],
                          [{"record_id": gone["record_id"]}], [{"record_id": rs[0]["record_id"], "model": None}], "deepseek") \
        == ids - {gone["record_id"], rs[0]["record_id"]}


def test_completeness_refuses_design_runs_under_seeds_the_design_does_not_have():
    """A row of a design run (a model of the config, a design condition, variant main) has a design seed, "base" for
    the base conditions and one of the model's config seeds for the adapter ones; an extra adapter seed or a leftover
    seeded zero-shot run would otherwise enter the pairing. A row of a run outside the design (another variant, a
    model the config lacks) is refused too."""
    rs = rows({}, n_features=6)
    ids = {"deepseek": {r["record_id"] for r in rs}}
    moved = lambda cond, seed, to: [{**r, "seed": to} for r in rs if r["condition"] == cond and r["seed"] == seed]
    for extra, n in ((moved("full_aware", "s42", "s5")[:3], 3), (moved("zero_shot", "base", "s42")[:1], 1),
                     (moved("ft_only", "s42", "base")[:2], 2)):
        with pytest.raises(SapInputError, match=rf"^{n} rows of a design run under a seed the design does not have"):
            check_complete(rs + extra, ids, CFG)
    fa = [r for r in rs if r["condition"] == "full_aware" and r["seed"] == "s42"]
    with pytest.raises(SapInputError, match=rf"^{len(fa)} rows of a design run under a seed the design does not have"):
        check_complete(rs + [{**r, "model": "mistral", "seed": "s7"} for r in fa], ids, CFG)  # mistral has 3 seeds
    outside = r"^{} rows of runs outside the design \(model, condition, variant\), e.g. "
    for other in ({"variant": "k4"}, {"model": "gpt"}, {"condition": "full_aware_2"}):
        with pytest.raises(SapInputError, match=outside.format(len(fa))):
            check_complete(rs + [{**r, **other} for r in fa], ids, CFG)
    lc = [{**r, "variant": "lc600"} for r in fa]  # a deferred variant: outside the design, allowed with deferred
    assert check_complete(rs + lc, ids, CFG, deferred=True)[1]["missing_filled"] == 0
    with pytest.raises(SapInputError, match=outside.format(len(fa))):
        check_complete(rs + lc, ids, CFG)



def test_the_aux_design_is_the_anchor_models_runs_only():
    """--descriptive-only (aux_test) goes through check_complete too (since 27/09/2026), over aux_design(cfg): the
    anchor model's runs only, the aux_test runs of the main generation (fixed 24/09/2026: DeepSeek, the two base
    conditions once and the three adapter ones × 5 seeds, 17 runs); a
    duplicated, missing or other-model aux row stops it. The config itself is not changed."""
    aux = aux_design(CFG)
    assert set(aux["seeds"]) == {CFG["anchor_model"]} and len(design(aux)) == 2 + 3 * 5
    assert set(CFG["seeds"]) == {"deepseek", "codellama", "mistral"}
    rs = rows({}, n_features=4)
    ids = {"deepseek": {r["record_id"] for r in rs}}
    assert check_complete(rs, ids, aux)[1] == {"records_covered": {"deepseek": len(ids["deepseek"])},
                                               "missing_filled": 0}
    with pytest.raises(SapInputError, match="^1 duplicated"):
        check_complete(rs + [dict(rs[0])], ids, aux)
    with pytest.raises(SapInputError, match="^1 missing rows"):
        check_complete(rs[1:], ids, aux)
    other = [{**r, "model": "codellama"} for r in rs if r["seed"] in ("base", "s42")][:3]
    with pytest.raises(SapInputError, match=r"^3 rows of runs outside the design"):
        check_complete(rs + other, ids, aux)

def test_completeness_covers_every_model_of_the_design():
    """Every model of cfg["seeds"] is in the design (the base conditions once, the adapter conditions × its own seeds,
    variant main), each over its own records (the gold records minus those compose dropped for that model); a missing
    row of any model stops the analysis unless accepted for that model (an --accept-missing entry without "model" is the
    anchor's)."""
    assert {rk[0] for rk, _ in design(CFG)} == set(CFG["seeds"]) and len(design(CFG)) == 2 * 3 + 3 * (5 + 3 + 3)
    assert ((("mistral", "ft_only", "main"), "s23") in design(CFG)
            and (("mistral", "ft_only", "main"), "s7") not in design(CFG))
    rs = rows({}, n_features=4)
    ids = sorted({r["record_id"] for r in rs})
    others = [{**r, "model": m} for r in rs if r["seed"] in ("base", "s42", "s11", "s23")
              for m in ("codellama", "mistral") if not (m == "mistral" and r["record_id"] == ids[0])]
    by = {"deepseek": set(ids), "codellama": set(ids), "mistral": set(ids) - {ids[0]}}
    rows_all, info = check_complete(rs + others, by, CFG)
    assert info == {"records_covered": {"deepseek": len(ids), "codellama": len(ids), "mistral": len(ids) - 1},
                    "missing_filled": 0}
    gone = next(r for r in others if (r["model"], r["condition"], r["seed"]) == ("mistral", "full_aware", "s23"))
    rest = [r for r in rs + others if r is not gone]
    with pytest.raises(SapInputError,
                       match=r"^1 missing rows, e.g. \('.*', \('mistral', 'full_aware', 'main'\), 's23'\)"):
        check_complete(rest, by, CFG)
    with pytest.raises(SapInputError, match="missing rows"):  # accepted for the anchor only
        check_complete(rest, by, CFG, {(gone["record_id"], "deepseek", "full_aware", "s23")})
    filled, info = check_complete(rest, by, CFG, {(gone["record_id"], "mistral", "full_aware", "s23")})
    assert info["missing_filled"] == 1 and filled[-1]["model"] == "mistral" and filled[-1]["filled_missing"]


def test_design_records_are_every_controls_record_but_the_compose_dropped():
    """Every design run covers every record with a gold row in the locked controls run, mechanical exclusions included
    (they count in BV and ER), except the records compose dropped for the anchor."""
    controls = [{"record_id": i, "control": c} for i in "abcd" for c in ("gold", "stub-v3")] + \
        [{"record_id": "s", "control": "stub-v3"}]
    compose = [{"record_id": "c", "model": None}, {"record_id": "d", "model": "mistral"},
               {"record_id": "a", "fmt": "F2"}]
    assert design_records(controls, compose, "deepseek") == {"b", "d"}


def test_filled_rows_take_the_locked_flags_and_count_only_in_ca_and_bv():
    """A row accepted as missing scores CA = 0 and BV = 0 with the flags of its record in the locked
    controls run (locked_flags: ca_defined of the gold row, excl_bv and excl_ca of exclusions.jsonl), not those of a
    template row, so a record with no row left keeps its BV exclusion; it has no ER value, so it never counts in ER or
    ER-trace."""
    rs = rows({}, n_features=6)
    ids = {r["record_id"] for r in rs}
    rid = sorted(ids)[0]
    controls = [{"record_id": i, "control": "gold", "ca_defined": True} for i in ids]
    exclusions = [{"record_id": rid, "reason": "gold_fails_bv", "excl_bv": True, "excl_ca": True}]
    flags = locked_flags(controls, exclusions)
    assert flags[rid] == {"ca_defined": True, "excl_bv": True, "excl_ca": True}
    assert flags[sorted(ids)[1]] == {"ca_defined": True, "excl_bv": False, "excl_ca": False}
    runs = {(c, s) for c in CFG["base_conditions"] for s in ("base",)} | \
        {(c, s) for c in CFG["adapter_conditions"] for s in CFG["seeds"]["deepseek"]}
    rest = [r for r in rs if r["record_id"] != rid]  # the record has no row left
    accepted = {(rid, "deepseek", c, s) for c, s in runs}
    filled, info = check_complete(rest, {"deepseek": ids}, CFG, accepted, flags)
    new = [r for r in filled if r.get("filled_missing")]
    assert info["missing_filled"] == len(runs) == len(new)
    assert all((r["ca"], r["bv"], r["excl_bv"], r["excl_ca"], r["ca_defined"]) == (0, 0, True, True, True)
               and "er" not in r for r in new)
    elig = {"er": {rid: True}, "trace": {rid: True}}
    assert not any(usable(m, set(), elig)(r) for r in new for m in ("bv", "er", "er_trace"))
    assert all(not r["excl_bv"] for r in check_complete(rest, {"deepseek": ids}, CFG, accepted)[0]
               if r.get("filled_missing"))


def test_unequal_two_level_n_between_comparisons_stops():
    """Every comparison with data has one two-level n, as it has one test (1) n: an extra full_aware seed
    on a few scenarios leaves test (1)'s n alone but shrinks the two-level n of the base comparisons."""
    rs = rows({"full_aware": 0.3}, n_features=20)
    some = sorted({r["scenario_key"] for r in rs})[:5]
    extra = [{**r, "seed": "s5"} for r in rs if r["condition"] == "full_aware" and r["seed"] == "s42"
             and r["scenario_key"] in some]
    with pytest.raises(SapInputError, match=r"confirmatory comparisons have different two-level n: \[5, "):
        confirmatory(rs + extra, CFG, set(), {})


def test_unequal_n_between_comparisons_stops():
    rs = rows({"full_aware": 0.3}, n_features=10)
    drop = rs[0]["scenario_key"]
    with pytest.raises(SapInputError, match="different n"):
        confirmatory([r for r in rs if not (r["condition"] == "rag_only" and r["scenario_key"] == drop)], CFG, set(), {})


def test_mde_from_dev_uses_locked_variant_and_calibration():
    dev = [r for r in rows({}, seed=3, variant="dev_pilot_v0") if r["seed"] in ("s42", "base")]
    m = mde_from_dev(dev, CFG, set(), clusters_target=175, scenarios_target=343, calibration=CAL)
    other = [r for r in rows({"full_aware": 0.4}, seed=4, variant="k4") if r["seed"] in ("s42", "base")]
    assert mde_from_dev(other + dev, CFG, set(), 175, 343, CAL) == m  # another variant's rows are left out
    assert set(m) == {"RQ1:full_aware_vs_zero_shot", "RQ1:full_aware_vs_rag_only", "RQ1:full_aware_vs_ft_only",
                      "RQ2:full_aware_vs_full_neutral"}
    z, n = m["RQ1:full_aware_vs_zero_shot"], m["RQ2:full_aware_vs_full_neutral"]
    assert z["type"] == "base" and z["mde_two_level@tau_0"] == pytest.approx(z["mde"] * 0.9)
    assert n["type"] == "adapter" and n["mde_two_level@tau_0.04"] == pytest.approx(n["mde"] * 0.95) and n["mde_k_seeds@tau_0"] > 0
    with pytest.raises(SapInputError, match="locked variant"):
        mde_from_dev(other + [{**r, "variant": "main"} for r in dev], CFG, set(), 175, 343)
    with pytest.raises(SapInputError, match="locked variant"):
        mde_from_dev([{**r, "model": "mistral"} for r in dev], CFG, set(), 175, 343)
    null = confirmatory(rows({}, seed=5), CFG, set(), {})
    text = answers(null, m)
    n = m["RQ2:full_aware_vs_full_neutral"]
    both = f"MDE là {n['mde_k_seeds@tau_0']:.3f} với phép (1) và {n['mde_two_level@tau_0']:.3f} với phép (2)"
    assert any("không phát hiện khác biệt" in t and both in t for t in text if t.startswith("RQ2"))


def test_no_difference_answer_quotes_only_k_seed_and_two_level_mde():
    conf = {"RQ2": {"rq_verdict": "not_supported", "comparisons": {"full_aware_vs_full_neutral": {
        "verdict": "no_difference", "ca": {"estimate": 0.012, "ci95": [-0.03, 0.05], "p_holm": 0.41,
                                           "two_level": {"ci95": [-0.04, 0.061], "p_holm": 0.52}}}}}}
    mdes = {"RQ2:full_aware_vs_full_neutral": {"mde": 0.08, "mde_k_seeds@tau_0": 0.036, "mde_two_level@tau_0": 0.057,
                                               "degenerate": False}}
    text = answers(conf, mdes)[0]
    assert text == ("RQ2: không phát hiện khác biệt về CA giữa Full-aware và full_neutral (ước lượng +0.012, KTC 95% "
                    "[-0.030, +0.050]; p Holm đảo dấu = 0.4100; p Holm hai tầng = 0.5200). Trên cùng mặt bằng 5 seed, MDE là "
                    "0.036 với phép (1) và 0.057 với phép (2); đề án không loại trừ được một hiệu ứng nhỏ hơn các MDE đó.")
    assert "0.080" not in text
    assert "MDE là (chưa công bố) với phép (1)" in answers(conf)[0]
    degenerate = {k: {**v, "degenerate": True} for k, v in mdes.items()}
    assert "0.036 (suy biến) với phép (1) và 0.057 (suy biến)" in answers(conf, degenerate)[0]


def test_subset_flags_come_from_the_locked_file():
    rs = [{"record_id": i, "sub_a": i in ("x", "y"), "sub_b": False} for i in ("x", "y", "z")]
    assert subset_flags(rs, {"a": ["x", "y"], "b": []}, ["a", "b"]) == rs
    assert subset_flags(rs, {"a": {"test": ["x"], "aux_test": ["y"]}, "b": {"test": []}}, ["a", "b"]) == rs
    with pytest.raises(SapInputError, match="differ"):
        subset_flags(rs, {"a": ["x"], "b": []}, ["a", "b"])
    with pytest.raises(SapInputError, match="differ"):  # an absent column is not False
        subset_flags([{"record_id": "z", "sub_a": False}], {"a": [], "b": []}, ["a", "b"])
    with pytest.raises(SapInputError, match="lacks"):
        subset_flags(rs, {"a": ["x", "y"]}, ["a", "b"])
    filled = subset_flags(rs + [{"record_id": "y", "filled_missing": True}], {"a": ["x", "y"], "b": []}, ["a", "b"])[-1]
    assert filled["sub_a"] is True and filled["sub_b"] is False


def test_self_contained_subset_comes_from_the_locked_file():
    """The fourth sensitivity subset of CA, self_contained, is the records flagged True in the
    locked self_contained.jsonl (not the rows, not sensitivity_subsets.json); a row whose self_contained column is not
    its record's flag in that file (an absent column, or a record the file lacks, included) stops the analysis, like a
    sub_* flag against sensitivity_subsets.json; a row filled by check_complete takes the file's membership."""
    assert CFG["subsets"] == ["post_cutoff_all", "time_coherent_30d", "residual_clean", "self_contained"]
    sc = {"x": True, "y": False, "z": False}
    rs = [{"record_id": i, "sub_a": i == "x", "self_contained": sc[i]} for i in ("x", "y", "z")]
    names, sealed = ["a", "self_contained"], {"a": ["x"]}
    out = subset_flags(rs, sealed, names, sc)
    assert [(r["sub_a"], r["sub_self_contained"]) for r in out] == [(True, True), (False, False), (False, False)]
    assert subset_flags(rs, {**sealed, "self_contained": ["y", "z"]}, names, sc) == out  # not read from that file
    assert subset_flags([{**rs[1], "sub_self_contained": True}], sealed, names, sc)[0]["sub_self_contained"] is False
    differs = r"^1 rows have a self_contained that differs from self_contained\.jsonl"
    for bad in ({**rs[1], "self_contained": True}, {**rs[0], "self_contained": None},
                {k: v for k, v in rs[0].items() if k != "self_contained"}, {**rs[2], "record_id": "w", "sub_a": False}):
        with pytest.raises(SapInputError, match=differs):
            subset_flags([bad], sealed, names, sc)
    filled = subset_flags(rs + [{"record_id": "x", "filled_missing": True}], sealed, names, sc)[-1]
    assert filled["sub_self_contained"] is True
    with pytest.raises(SapInputError, match="the self_contained subset needs the locked self_contained.jsonl"):
        subset_flags(rs, sealed, names)
    assert subset_flags(rs, sealed, ["a"]) == [{**r, "sub_a": r["record_id"] == "x"} for r in rs]


PROV_DIGESTS ={"config": "c", "calibration": "k", "dev_scores[0]": "d0", "dev_scores[1]": "d1", "dev_exclusions": "e",
                "mde": "m", "controls": "t1", "exclusions": "t2", "compose_excluded": "t3", "records": "r",
                "subsets": "s"}
PROV_MDE = {"config_sha256": "c", "calibration_sha256": "k",
            "dev_inputs": {"b.jsonl": "d1", "a.jsonl": "d0", "dc/exclusions.jsonl": "e"},
            "test_inputs": {"tc/controls.jsonl": "t1", "tc/exclusions.jsonl": "t2", "excluded.jsonl": "t3"},
            "dev_variant": "dev_pilot_v0"}
PROV_SEL = {"choice": "dev_pilot_v0",
            "inputs": {"a.jsonl": "d0", "b.jsonl": "d1", "k12.jsonl": "d2", "dc/exclusions.jsonl": "e"}}
PROV_CAL = {"config_sha256": "c",
            "inputs": {"records": "r", "controls": "t1", "exclusions": "t2", "compose_excluded": "t3"}}
PHASE2_LISTED = {"out/phase2/records/dev.jsonl": "x", "out/phase2/records/test.jsonl": "r",
                 "out/phase2/sensitivity_subsets.json": "s"}
NOT_TEST_RECORDS = "the records file given here is not PHASE2_SHA256SUMS's records/test.jsonl"
NOT_CAL_RECORDS = "CALIBRATION.json inputs records is not the records file given here"
NOT_SUBSETS = "the subsets file given here is not PHASE2_SHA256SUMS's sensitivity_subsets.json"


def provenance(m=PROV_MDE, c=PROV_CAL, dd=PROV_DIGESTS, s=PROV_SEL, ch=("dev_scores[0]",),
               ls=PHASE2_LISTED) -> list[str]:
    """provenance_errors on consistent inputs (the good case gives []), with some of them replaced."""
    return provenance_errors(m, c, dd, s, CFG, list(ch), dict(ls))


def test_lock_provenance_errors():
    d, mde, sel, cal, check = PROV_DIGESTS, PROV_MDE, PROV_SEL, PROV_CAL, provenance
    assert check() == []
    assert check(m={**mde, "config_sha256": "old"}) == ["MDE.json config_sha256 is not the current config"]
    assert check(c={**cal, "config_sha256": "old"}) == ["CALIBRATION.json config_sha256 is not the current config"]
    assert check(m={**mde, "calibration_sha256": "old"}) == ["MDE.json calibration_sha256 is not the current calibration"]
    assert len(check(dd={**d, "dev_scores[1]": "changed"})) == 1 and "dev_inputs" in check(dd={**d, "dev_scores[1]": "changed"})[0]
    assert "dev_inputs" in check(m={**mde, "dev_inputs": {}})[0]
    assert "test_inputs" in check(dd={**d, "exclusions": "changed"})[0]
    assert "differ" in check(s={**sel, "choice": "dev_grid_k4_n3_r050"})[0]
    # CALIBRATION.json: computed on the locked test controls, exclusions and compose file, and on the locked records
    assert check(dd={**d, "exclusions": "changed"})[1:] == ["CALIBRATION.json inputs ['exclusions'] are not the test files given here"]
    assert check(c={**cal, "inputs": {**cal["inputs"], "compose_excluded": "old"}}) == \
        ["CALIBRATION.json inputs ['compose_excluded'] are not the test files given here"]
    assert check(c={"config_sha256": "c"}) == \
        ["CALIBRATION.json inputs ['controls', 'exclusions', 'compose_excluded'] are not the test files given here",
         NOT_CAL_RECORDS]
    # DEV_SELECTION.json: it read the chosen variant's dev score files and the dev exclusions given here
    assert check(ch=("dev_scores[0]", "dev_scores[1]")) == []
    changed = check(dd={**d, "dev_scores[0]": "changed"})
    assert len(changed) == 2 and "dev_inputs" in changed[0]
    assert changed[1] == "DEV_SELECTION.json inputs do not hold the chosen variant's dev score files given here"
    assert check(s={**sel, "inputs": {"b.jsonl": "d1", "dc/exclusions.jsonl": "e"}}) == [changed[1]]
    assert check(ch=()) == [changed[1]]
    assert check(s={**sel, "inputs": {"a.jsonl": "d0", "dc/exclusions.jsonl": "other"}}) == \
        ["DEV_SELECTION.json inputs do not hold the dev exclusions given here"]
    assert check(dd={**d, "dev_exclusions": "changed"})[1:] == ["DEV_SELECTION.json inputs do not hold the dev exclusions given here"]
    assert len(check(s={"choice": "dev_pilot_v0"})) == 2


def test_lock_provenance_pins_records_and_subsets_to_phase2():
    """The records file given to the lock is PHASE2_SHA256SUMS's records/test.jsonl and CALIBRATION.json
    was computed on it (records/dev.jsonl, listed too, does not pass); the locked sensitivity_subsets.json is the one
    PHASE2_SHA256SUMS lists."""
    assert provenance() == []
    cal_dev = {**PROV_CAL, "inputs": {**PROV_CAL["inputs"], "records": "x"}}  # a calibration on records/dev.jsonl
    assert provenance(c=cal_dev) == [NOT_CAL_RECORDS]
    assert provenance(c=cal_dev, dd={**PROV_DIGESTS, "records": "x"}) == [NOT_TEST_RECORDS]
    assert provenance(dd={**PROV_DIGESTS, "records": "x"}) == [NOT_TEST_RECORDS, NOT_CAL_RECORDS]
    assert provenance(dd={**PROV_DIGESTS, "subsets": "other"}) == [NOT_SUBSETS]
    assert provenance(ls={**PHASE2_LISTED, "out/phase2/sensitivity_subsets.json": "other"}) == [NOT_SUBSETS]
    # no records/test.jsonl entry (a look-alike path is not one), or two: nothing to pin the records to
    no_test = {k: v for k, v in PHASE2_LISTED.items() if not k.endswith("test.jsonl")}
    for ls in (no_test, {**no_test, "out/phase2/myrecords/test.jsonl": "r"},
               {**PHASE2_LISTED, "old/records/test.jsonl": "r"}):
        assert provenance(ls=ls) == [NOT_TEST_RECORDS]
    assert provenance(ls={}) == [NOT_TEST_RECORDS, NOT_SUBSETS]
    # a missing digest never matches a missing entry
    no_rec = {k: v for k, v in PROV_DIGESTS.items() if k not in ("records", "subsets")}
    cal_no_rec = {**PROV_CAL, "inputs": {k: v for k, v in PROV_CAL["inputs"].items() if k != "records"}}
    assert provenance(dd=no_rec, c=cal_no_rec, ls={}) == [NOT_TEST_RECORDS, NOT_CAL_RECORDS, NOT_SUBSETS]


def test_jsonl_rows_reads_exactly_as_symcache_jsonl_rows(tmp_path):
    """The one 3D JSONL reader, sap.jsonl_rows, keeps and refuses exactly what 3A's
    symcache.jsonl_rows does: split on "\\n" only (a raw U+2028 or U+0085 in a string stays in its row), only a final
    "" dropped, so a blank line inside a file, or a lone "\\r" taken for a separator, raises. 3D raises
    SapInputError (which its CLIs turn into their input refusal) where 3A raises json's error."""
    from v2tools.symcache import jsonl_rows as symcache_rows
    rows = [{"record_id": "a", "text": "x\u2028y\x85z"}, {"record_id": "b", "text": "é\u2029"}]
    body = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    p = tmp_path / "rows.jsonl"
    for data, want in ((body, rows), (body[:-1], rows), ('{"a": 1}\r\n{"a": 2}\r\n', [{"a": 1}, {"a": 2}]), ("", []),
                       ("\n", None), (body + "\n", None), (body.replace("\n", "\n\n", 1), None),
                       ('{"a": 1}\r{"a": 2}\r', None)):
        p.write_bytes(data.encode("utf-8"))
        if want is None:
            with pytest.raises(ValueError):
                symcache_rows(p)
            with pytest.raises(SapInputError, match=r"rows\.jsonl line \d+ is not a JSON row"):
                jsonl_rows(p)
        else:
            assert jsonl_rows(p) == symcache_rows(p) == want
    assert len(body.splitlines()) > len(rows)  # what str.splitlines would have made of it
    p.write_bytes(b'{"a": "\xff"}\n')
    with pytest.raises(ValueError):
        symcache_rows(p)
    with pytest.raises(SapInputError, match="is not UTF-8"):
        jsonl_rows(p)
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))  # a BOM is not skipped, as in 3A
    with pytest.raises(SapInputError, match=r"rows\.jsonl line 1 is not a JSON row"):
        jsonl_rows(p)
    for line in ("[1, 2]", "3", '"x"', "null"):  # a row is an object, or the analysis stops cleanly
        p.write_text(json.dumps(rows[0]) + "\n" + line + "\n", encoding="utf-8")
        with pytest.raises(SapInputError, match=r"rows\.jsonl line 2 is not a JSON object"):
            jsonl_rows(p)


def test_listed_sha256_reads_a_sums_file(tmp_path):
    p = tmp_path / "PHASE2_SHA256SUMS"
    p.write_text("A" * 64 + "  out/phase2/records/test.jsonl\n\n" + "b" * 64 + "  out/phase2/records/dev.jsonl\n"
                 + "c" * 64 + "  out/phase2/sensitivity_subsets.json\n")
    listed = listed_sha256(p)
    assert listed == {"out/phase2/records/test.jsonl": "a" * 64, "out/phase2/records/dev.jsonl": "b" * 64,
                      "out/phase2/sensitivity_subsets.json": "c" * 64}
    assert listed_at(listed, "records/test.jsonl") == "a" * 64
    assert listed_at(listed, "sensitivity_subsets.json") == "c" * 64
    assert listed_at(listed, "cords/test.jsonl") is None and listed_at(listed, "records/aux_test.jsonl") is None
    assert listed_at({**listed, "copy/records/test.jsonl": "a" * 64}, "records/test.jsonl") is None


def test_dev_selection_keeps_baseline_unless_three_scenarios_better():
    def dev(variant: str, extra: int) -> list[dict]:
        out = []
        for i in range(10):
            ca = int(i < 4 + extra)
            out.append({"record_id": f"R|f{i}.feature|1|a", "scenario_key": f"R|f{i}.feature|1", "feature_path": f"f{i}.feature", "repo": "R",
                        "model": "deepseek", "seed": "s42", "condition": "full_aware", "variant": variant, "ca": ca, "ca_defined": True})
        return out
    base = dev("dev_pilot_v0", 0)
    assert select_dev_config(base + dev("k4", 2), CFG, set())["choice"] == "dev_pilot_v0"
    assert select_dev_config(base + dev("k4", 3) + dev("k12", 3), CFG, set())["choice"] == "k12"
    assert select_dev_config(base + dev("k4", 3) + dev("k12", 4), CFG, set())["choice"] == "k12"
    with pytest.raises(SapInputError):
        select_dev_config(dev("k4", 3), CFG, set())


def test_semv_sample_is_blind_distinct_and_deterministic():
    rs = rows({})
    texts = {(r["record_id"], r["condition"], r["seed"]): f"code {r['record_id']} {r['condition']}" for r in rs}
    blind, key = semv_sample(rs, CFG, set(), texts)
    again = semv_sample(list(reversed(rs)), CFG, set(), texts)
    assert (blind, key) == again and len(blind) == 100 and len({x["record_id"] for x in blind}) == 50
    assert all(set(x) == {"item_id", "record_id", "text"} and x["text"] for x in blind)
    by = {x["item_id"]: x for x in key}
    assert {x["condition"] for x in key} == {"full_aware", "ft_only"} and all(by[b["item_id"]]["condition"] in b["text"] for b in blind)
    per_rec = {}
    for x in key:
        per_rec.setdefault(x["record_id"], set()).add(x["seed"])
    assert all(len(s) == 1 for s in per_rec.values())


def test_per_run_counts_every_extract_status_and_the_fence_truncated_rate():
    """per_run counts the five extract statuses (multi_file_fence included) over the rows of the run that
    have an output, with that n, so the counts sum to it; fence_truncated is a rate over the rows with a text (not
    None), like every rate of per_run."""
    g, rs = "deepseek|full_aware|main|s42", rows({}, n_features=4)
    mine = [i for i, r in enumerate(rs) if (r["condition"], r["seed"]) == ("full_aware", "s42")]
    for k, i in enumerate(mine):
        st = EXTRACT_STATUSES[k % 5]
        rs[i] = {**rs[i], "extract_status": st, "fence_truncated": None if st == "none" else k % 3 == 0}
    rs[mine[0]] = {k: v for k, v in rs[mine[0]].items() if k != "fence_truncated"}  # an older row: no column
    e = per_run(rs, set())[g]["extract"]
    assert list(e) == [*EXTRACT_STATUSES, "n"] and sum(e[s] for s in EXTRACT_STATUSES) == e["n"] == len(mine)
    assert all(e[st] == len(range(k, len(mine), 5)) for k, st in enumerate(EXTRACT_STATUSES))
    text = [k for k in range(1, len(mine)) if EXTRACT_STATUSES[k % 5] != "none"]
    assert per_run(rs, set())[g]["fence_truncated"] == {"n": len(text),
                                                       "rate": sum(k % 3 == 0 for k in text) / len(text)}
    filled, _ = check_complete([r for i, r in enumerate(rs) if i != mine[1]],
                               {"deepseek": {r["record_id"] for r in rs}}, CFG,
                               {(rs[mine[1]]["record_id"], "deepseek", "full_aware", "s42")})
    after = per_run(filled, set())[g]
    assert after["extract"]["n"] == len(mine) - 1 and after["filled_missing"] == 1  # a filled row has no output
    assert after["fence_truncated"]["n"] == len(text) - 1  # the removed row (multi_file_fence) had a text


def test_per_run_reports_v3all_only_when_scored():
    rs = rows({}, n_features=4)
    assert "ca_ref_v3all" not in per_run(rs, set())["deepseek|full_aware|main|s42"]
    out = per_run([{**r, "ca_ref_v3all": 1} for r in rs], set())["deepseek|full_aware|main|s42"]
    assert out["ca_ref_v3all"]["rate"] == 1.0 and out["ca_above_v3all"]["rate"] == 0.0


def test_dose_response_buckets():
    """P26 (e) (fixed 06/10/2026): each bucket's F2 − F1 carries the direction label of design specification §2.4
    (expected > 0); the rise with the dose gets no label."""
    out = dose_response(rows({"full_aware": 0.2}), CFG, set())
    assert set(out) == {"0-0", "1-2", "3-"} and all(v["n_scenarios"] > 0 for v in out.values())
    assert all(v["direction"] == direction(v["ci95"], 1, v["n_clusters"]) and v["direction"] for v in out.values())
    few = [{**r, "ca": int(r["condition"] == "full_aware")} for r in rows({}, n_features=4)]  # doses 0, 1, 2, 3
    one = dose_response(few, CFG, set())  # buckets of one, two and one cluster, F2 − F1 = 1 everywhere
    assert [(v["n_clusters"], v["estimate"], v["direction"]) for v in one.values()] == [
        (1, 1.0, "không rõ chiều"), (2, 1.0, "theo chiều kỳ vọng"), (1, 1.0, "không rõ chiều")]
    empty = dose_response([{**r, "dose": None} for r in rows({})], CFG, set())
    assert all(v == {"n_scenarios": 0, "n_clusters": 0, "estimate": None, "ci95": [None, None], "p": None,
                     "direction": None} for v in empty.values())


def test_a_missing_condition_is_not_run_in_both_tests_and_holm_keeps_its_place():
    rs = [r for r in rows({"full_aware": 0.3}) if r["condition"] != "rag_only"]
    conf = confirmatory(rs, CFG, set(), {})
    v = conf["RQ1"]["comparisons"]
    assert v["full_aware_vs_rag_only"]["verdict"] == "not_run" and v["full_aware_vs_ft_only"]["verdict"] == "supported"
    assert conf["RQ1"]["rq_verdict"] == "partial" and any("không có kịch bản nào" in a for a in answers(conf))
    sub = family(seed_values(rs, "ca", usable("ca", set())), CFG, CFG["families"]["RQ1"], CFG["seed"])
    assert sub["full_aware_vs_rag_only"]["verdict"] == "not_run" and sub["full_aware_vs_rag_only"]["p_holm"] is None
    ft = sub["full_aware_vs_ft_only"]
    assert ft["p_holm"] >= min(1.0, 2 * ft["p"]) - 1e-12 and ft["two_level"]["p_holm"] >= min(1.0, 2 * ft["two_level"]["p"]) - 1e-12


# ----- the generation run's gen_code_sha256 is an object (fixed 04/10/2026) ------------------------------------------

def test_score_rows_take_gen_code_sha256_as_the_generation_object_and_nothing_else():
    """The real writer (ee6bc8c) puts gen_code_sha256 = {relative path: sha256} in every ok line and score_run
    copies it; before, every such row was refused (1,020/1,020 dev rows on the real writer). Accepted: a
    non-empty object whose keys are relative POSIX paths and whose values are 64 lowercase hex; decision_sha256 is 64
    lowercase hex. Refused even when the line carries the same value: the old string form, an empty object, a bad
    path or value; and, as before, a row whose object differs from its line's (tampered)."""
    good = [{**r, **line_shas(r["model"], r["condition"], r["seed"]), "line_sha256": "l" * 64} for r in SR_GOOD]
    lines = {pair(r): mline(r["record_id"], model=r["model"], cond=r["condition"], seed=r["seed"],
                            variant=r.get("variant") or "main") for r in good}
    check = lambda rows_, ls=lines: score_row_errors(rows_, SR_CONTROLS, SR_EXCLUSIONS, {"er": {A: True, B: False}}, CFG,
                                                     True, manifest=ls)
    bad_sha = ("1 score rows lack decision_sha256 (64 lowercase hex) or gen_code_sha256 (a non-empty {relative path: 64 "
               "lowercase hex}), or carry a sha256 (prompts, prompt, adapter, decision, gen code, leak-check receipt) or "
               f"a line_sha256 that is not their manifest line's, e.g. {ex(A)}")
    assert check(good) == []
    k, code = pair(good[0]), good[0]["gen_code_sha256"]
    tampered = {**code, "v2gpu/runs.py": "0" * 64}
    assert check([{**good[0], "gen_code_sha256": tampered}] + good[1:]) == [bad_sha]  # equality with the line, kept
    assert check([{**good[0], "gen_code_sha256": {}}] + good[1:]) == [bad_sha]
    hx = "a" * 64
    for value in (hx, {}, {"/abs/generate.py": hx}, {"../generate.py": hx}, {"v2gpu/../x.py": hx}, {"v2gpu//x.py": hx},
                  {"v2gpu\\x.py": hx}, {"v2gpu/x.py": "A" * 64}, {"v2gpu/x.py": hx[:63]}, {"v2gpu/x.py": None},
                  [["v2gpu/x.py", hx]]):
        same = {**lines, k: {**lines[k], "gen_code_sha256": value}}  # the line carries it too: the shape alone refuses
        assert check([{**good[0], "gen_code_sha256": value}] + good[1:], ls=same) == [bad_sha], value
    for value in ("A" * 64, "a" * 63, "deci:deepseek"):
        same = {**lines, k: {**lines[k], "decision_sha256": value}}
        assert check([{**good[0], "decision_sha256": value}] + good[1:], ls=same) == [bad_sha], value
    ok = {"v2gpu/engines.py": "1" * 64, "v2gpu/generate.py": "2" * 64}  # the real writer's two keys (dev pilot manifest)
    same = {**lines, k: {**lines[k], "gen_code_sha256": ok}}
    assert check([{**good[0], "gen_code_sha256": ok}] + good[1:], ls=same) == []


def test_has_evidence_counts_the_attempts_that_read_as_a_receipt_as_generation_evidence_eligible_does():
    """The generation run's gencoverage.evidence_eligible counts the attempts whose receipt parsed (a status); a
    torn attempt receipt has a sha256 and no status. With [torn, CRASHED] the writer says
    False and the old rule said True, so the manifest's own evidence_eligible disagreed and the whole manifest was
    refused. Same rule as score_run.evidence."""
    line = lambda *st: mline("R|a|1|s", status="missing", attempts=[
        {"file": f"f{i}", "status": s, "sha256": f"{i}" * 64} for i, s in enumerate(st)])
    assert [has_evidence(line(*st)) for st in ((None, "CRASHED"), (None, "CRASHED", "CRASHED"), ("INCOMPLETE", "CRASHED"),
                                                ("CRASHED",), ("CRASHED", "REFUSED_PREFLIGHT"), ("CRASHED", None))] == [
        False, True, True, False, False, False]


# ----- P26 (added 06/10/2026, before any test output): descriptive measures of design specification §2.4 ------

def test_per_run_reports_the_duplicate_and_exact_rates_over_the_rows_with_an_output():
    """P26 (design specification §2.4): per_run gives the rate of duplicate_reuse (the output rewrites a step REUSE
    already defines) and of exact (a verbatim match of the gold answer after whitespace normalisation) over the rows
    with an output, the denominator of `extract`: an excl_bv row, a filled row and a row without the column are out."""
    g, rs = "deepseek|full_aware|main|s42", rows({}, n_features=4)
    mine = [i for i, r in enumerate(rs) if (r["condition"], r["seed"]) == ("full_aware", "s42")]
    for k, i in enumerate(mine):
        rs[i] = {**rs[i], "duplicate_reuse": k % 3 == 0, "exact": k % 4 == 0}
    rs[mine[1]] = {**rs[mine[1]], "excl_bv": True}
    rs[mine[2]] = {k: v for k, v in rs[mine[2]].items() if k not in ("duplicate_reuse", "exact")}
    filled, _ = check_complete([r for i, r in enumerate(rs) if i != mine[3]],
                               {"deepseek": {r["record_id"] for r in rs}}, CFG,
                               {(rs[mine[3]]["record_id"], "deepseek", "full_aware", "s42")})
    out = per_run(filled, set())[g]
    kept = [k for k in range(len(mine)) if k not in (1, 2, 3)]
    assert out["duplicate_reuse"] == {"n": len(kept), "rate": sum(k % 3 == 0 for k in kept) / len(kept)}
    assert out["exact"] == {"n": len(kept), "rate": sum(k % 4 == 0 for k in kept) / len(kept)}
    assert out["extract"]["n"] == len(kept) + 1 and out["filled_missing"] == 1  # mine[2] has an output, no column


def test_direction_is_the_presentation_label_of_the_ci():
    """The presentation rule for a direction (design specification §2.4, fixed 06/10/2026): "không rõ chiều" when the
    95% CI holds 0 (an end at 0 included), else "theo chiều kỳ vọng" or "ngược chiều" by the side of the CI against the
    expected sign; None without a CI."""
    assert direction([0.1, 0.3], 1) == direction([-0.3, -0.1], -1) == "theo chiều kỳ vọng"
    assert direction([-0.3, -0.1], 1) == direction([0.1, 0.3], -1) == "ngược chiều"
    assert all(direction(c, s) == "không rõ chiều" for c in ([-0.1, 0.2], [0.0, 0.2], [-0.2, 0.0], [0.0, 0.0])
               for s in (1, -1))
    assert direction([None, None], 1) is None
    # a CI over fewer than 2 clusters (degenerate, [x, x]) says nothing of the direction (fixed 06/10/2026)
    assert direction([0.8, 0.8], 1, 1) == direction([-0.8, -0.8], 1, 0) == "không rõ chiều"
    assert direction([0.1, 0.3], 1, 2) == "theo chiều kỳ vọng" and direction([0.1, 0.3], 1) == "theo chiều kỳ vọng"


def test_contrasts_give_full_aware_minus_full_neutral_with_the_direction_label():
    """P26 (design specification §2.4): Full-aware minus Full-neutral (anchor, main), with the expected directions
    declared in advance, paired by seed and pooled per scenario, with the 95% cluster-bootstrap CI, no p, and the label
    of `direction` against the expected sign (made-up symbols < 0, the conventions > 0). The rows are per_run's: a
    compose-dropped record, an excl_bv row, a row without a value are out; the made-up symbol rate is the record's
    hallucinated / members_used, over the rows with members_used > 0."""
    rs = rows({}, n_features=12)
    for r in rs:
        aware = r["condition"] == "full_aware"
        r.update(hallucinated=0 if aware else 1, members_used=2, conv_style=1.0 if aware else 0.5,
                 conv_pattern=0.0 if aware else 1.0, conv_naming=0.5)
        if r["condition"] == "ft_only":  # another condition: never read
            r.update(hallucinated=2, conv_style=0.0, conv_pattern=1.0, conv_naming=0.0)
    out = contrasts(rs, CFG)
    total = len({r["scenario_key"] for r in rs})
    assert list(out) == ["hallucinated_rate", "conv_style", "conv_pattern", "conv_naming"]
    assert all(set(v) == {"expected", "n_scenarios", "n_clusters", "estimate", "ci95", "direction"}
               and v["n_scenarios"] == total for v in out.values())
    h, s, p, n = (out[k] for k in out)
    assert (h["expected"], h["estimate"], h["direction"]) == ("< 0", pytest.approx(-0.5), "theo chiều kỳ vọng")
    assert (s["expected"], s["estimate"], s["direction"]) == ("> 0", pytest.approx(0.5), "theo chiều kỳ vọng")
    assert (p["estimate"], p["ci95"], p["direction"]) == (pytest.approx(-1.0), [-1.0, -1.0], "ngược chiều")
    assert (n["estimate"], n["ci95"], n["direction"]) == (0.0, [0.0, 0.0], "không rõ chiều")
    x, y, z, w = sorted({r["scenario_key"] for r in rs})[:4]
    rs = [{**r, "excl_bv": True} if r["scenario_key"] == x else
          {**r, "members_used": 0, "hallucinated": 0} if r["scenario_key"] == z else
          {**r, "conv_style": None} if r["scenario_key"] == w else r for r in rs]
    dropped = {r["record_id"] for r in rs if r["scenario_key"] == y}
    got = contrasts(rs, CFG, dropped)
    assert {k: v["n_scenarios"] for k, v in got.items()} == {
        "hallucinated_rate": total - 3, "conv_style": total - 3, "conv_pattern": total - 2, "conv_naming": total - 2}
    assert descriptive(rs, CFG, set(), {}, dropped=dropped)["contrasts"] == got
    # the rows of the other models and of another variant are never read
    other = [{**r, "model": "codellama", "conv_style": 0.0} for r in rs] + [
        {**r, "variant": "lc600", "conv_pattern": 1.0, "hallucinated": 2} for r in rs if r["condition"] == "full_aware"]
    assert contrasts(rs + other, CFG, dropped) == got


def test_contrasts_pair_on_the_common_seeds_with_their_own_seed():
    """P26: Full-neutral is never taken for a base run, so a value Full-neutral keeps at one seed
    only is paired with Full-aware's at that seed alone; each measure's CI uses its own bootstrap seed (8000 + i)."""
    from v2tools.sap import as_dict
    from v2tools.stats import compare
    rs = rows({}, n_features=12)
    for r in rs:
        k = SEEDS.index(r["seed"]) if r["seed"] in SEEDS else 0
        r["conv_style"] = 0.2 * k if r["condition"] == "full_aware" else (0.0 if r["seed"] == SEEDS[0] else None)
        r["conv_naming"] = float(r["ca"])
    c = contrasts(rs, CFG)
    assert c["conv_style"]["estimate"] == 0.0  # seed s42 alone: 0.0 − 0.0 (as a base run it would be 0.4)
    keep = lambda r: not r["excl_bv"] and r.get("conv_naming") is not None
    a, b = ("deepseek", "full_aware", "main"), ("deepseek", "full_neutral", "main")
    ref = as_dict(compare(pooled(per_seed_pairs(seed_values(rs, "conv_naming", keep), a, b)), CFG["reps_bootstrap"], 0,
                          CFG["seed"] + 8000 + 3, with_p=False))
    assert (c["conv_naming"]["estimate"], c["conv_naming"]["ci95"]) == (ref["estimate"], ref["ci95"])
    assert ref["ci95"][0] < ref["ci95"][1]
    f0 = [{**r, "conv_style": 1.0 if r["condition"] == "full_aware" else 0.5} for r in rs
          if r["feature_path"] == "f0.feature"]
    one = contrasts(f0, CFG)["conv_style"]  # one cluster: [+0.5, +0.5] says nothing of the direction
    assert (one["n_clusters"], one["estimate"], one["ci95"], one["direction"]) == (1, 0.5, [0.5, 0.5], "không rõ chiều")


def test_the_direction_label_sits_beside_the_p11_flag():
    """P11 and design specification §2.4 (fixed 06/10/2026): each descriptive metric beside a comparison keeps P11's
    `opposite` flag, its logic unchanged, and also carries the §2.4 direction label (expected: A − B > 0). A supported
    CA with a negative BV estimate whose CI holds 0 is flagged by P11 (`opposite`) and labelled "không rõ chiều", never
    "ngược chiều"."""
    rs = rows({"full_aware": 0.3}, seed=2)
    first = sorted({r["scenario_key"] for r in rs})[0]
    for r in rs:
        r["bv"] = 0 if (r["condition"], r["scenario_key"]) == ("full_aware", first) else 1
    rq2 = lambda xs: confirmatory(xs, CFG, set(), {})["RQ2"]["comparisons"]["full_aware_vs_full_neutral"]
    v = rq2(rs)
    d = v["descriptive"]["bv"]
    assert v["verdict"] == "supported" and d["estimate"] < 0 and d["ci95"][0] < 0 <= d["ci95"][1]
    assert d["opposite"] is True and d["direction"] == "không rõ chiều"
    neg = rq2([{**r, "bv": 1 - r["ca"]} for r in rs])["descriptive"]["bv"]
    assert neg["ci95"][1] < 0 and neg["opposite"] is True and neg["direction"] == "ngược chiều"
    pos = rq2([{**r, "bv": r["ca"]} for r in rs])["descriptive"]["bv"]
    assert pos["ci95"][0] > 0 and pos["opposite"] is False and pos["direction"] == "theo chiều kỳ vọng"
    assert all("direction" in m for m in v["descriptive"].values())
    one = [{**r, "bv": int(r["condition"] == "full_aware") if r["feature_path"] == "f0.feature" else None}
           for r in rs]  # BV of one cluster only: a degenerate CI
    d1 = rq2(one)["descriptive"]["bv"]
    assert d1["n_clusters"] == 1 and d1["ci95"][0] == d1["ci95"][1] > 0 and d1["direction"] == "không rõ chiều"


def test_robustness_groups_by_prompt_length_follow_the_full_aware_row():
    """Prompt-length groups (P26, design specification §2.4; fixed 06/10/2026), by the anchor model's Full-aware prompt:
    two views after every other one (whose seeds stay), len_bucket=le8192 and len_bucket=gt8192, take each record by the
    len_bucket of its real Full-aware rows (anchor, main), whatever the bucket of its other rows, so every comparison
    keeps its pairs; a record without such a row is in neither group and is counted; Full-aware rows of one record in
    two buckets, or in a bucket 3C does not write, stop the analysis."""
    from v2tools.sap import as_dict
    from v2tools.stats import compare
    rs = rows({"full_aware": 0.3}, n_features=12)
    gt = {r["record_id"] for r in rs if r["feature_path"] in ("f0.feature", "f1.feature", "f2.feature")}
    for r in rs:
        long = (r["record_id"] in gt) == (r["condition"] == "full_aware")
        r["len_bucket"] = "gt8192" if long else "le8192"
    out = robustness(rs, CFG, set())
    assert list(out)[-3:] == ["without_pre_t", "len_bucket=le8192", "len_bucket=gt8192"]
    total = len({r["scenario_key"] for r in rs})
    n = lambda view, c: out[view][c]["n_scenarios"]
    assert all(n("len_bucket=le8192", c) + n("len_bucket=gt8192", c) == total for c in out["without_pre_t"])
    assert n("len_bucket=gt8192", "full_aware_vs_zero_shot") == len({r["scenario_key"] for r in rs
                                                                     if r["record_id"] in gt})
    sv = seed_values(rs, "ca", usable("ca", set()))
    for j, c in enumerate(out["without_pre_t"]):  # the views before keep their seeds
        a, b = c.split("_vs_")
        ref = as_dict(compare(pooled(per_seed_pairs(sv, ("deepseek", a, "main"), ("deepseek", b, "main"))),
                              CFG["reps_bootstrap"], 0, CFG["seed"] + 20000 + 100 * (len(out) - 3) + j, with_p=False))
        assert out["without_pre_t"][c] == {k: ref[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")}
    strata, unplaced = len_strata(rs, CFG)
    assert unplaced == 0 and {k for k, v in strata.items() if v == "gt8192"} == gt
    flip = lambda r: {**r, "len_bucket": "le8192" if r["len_bucket"] == "gt8192" else "gt8192"}
    other = [{**flip(r), "model": "codellama"} for r in rs] + [{**flip(r), "variant": "lc600"} for r in rs]
    assert len_strata(rs + other, CFG) == (strata, 0)  # only the anchor's main Full-aware rows
    assert descriptive(rs, CFG, set(), {})["len_strata"] == {"records": {"le8192": len(strata) - len(gt),
                                                                         "gt8192": len(gt)}, "unplaced": 0}
    gone = sorted(gt)[0]
    no_fa = [r for r in rs if (r["record_id"], r["condition"]) != (gone, "full_aware")]
    assert len_strata(no_fa, CFG)[1] == 1 and gone not in len_strata(no_fa, CFG)[0]
    assert descriptive(no_fa, CFG, set(), {})["len_strata"]["unplaced"] == 1
    assert len_strata(no_fa, CFG, {gone})[1] == 0  # a compose-dropped record is out of every metric already
    filled, _ = check_complete(no_fa, {"deepseek": {r["record_id"] for r in rs}}, CFG,
                               {(gone, "deepseek", "full_aware", sd) for sd in SEEDS})
    assert any(r.get("filled_missing") and r["record_id"] == gone and r["len_bucket"] for r in filled)
    assert len_strata(filled, CFG)[1] == 1  # a filled row is no Full-aware prompt
    assert len_strata([{**r, "len_bucket": None} if r["record_id"] == gone else r for r in rs], CFG)[1] == 1
    two = [{**r, "len_bucket": "le8192"}
           if (r["record_id"], r["condition"], r["seed"]) == (gone, "full_aware", SEEDS[0]) else r for r in rs]
    with pytest.raises(SapInputError, match="len_bucket"):
        len_strata(two, CFG)
    with pytest.raises(SapInputError, match="len_bucket"):
        len_strata([{**r, "len_bucket": "gt4096"} if r["record_id"] == gone else r for r in rs], CFG)


# ----- the RQ3 profile per model (P14, added 07/10/2026, before any test output) ------------------------------

def test_rq3_profile_gives_each_model_its_full_aware_levels_with_null_where_it_has_none():
    """P14 (fixed 07/10/2026): RQ3 in its original wording, a quality profile that includes Executable Rate: per model
    of cfg["seeds"], the level at Full-aware (variant main) of CA, BV, ER, ER@self_contained, ER-trace and
    ER-trace@self_contained, each the estimate and 95% cluster-bootstrap CI of the scenario means pooled over the
    model's seeds, no p, no label (CA: the very level of rq3_full_aware_by_model); denominators as usable() and the P11
    eligibility tables. score_row_errors asks an ER value of the anchor model only, so an ER metric of another model is
    never a level over part of its eligible rows: some missing (incomplete) or all missing (the ER run does not cover
    it) gives n 0, a null estimate and a reason, never 0; so does a model with no Full-aware row."""
    from v2tools.sap import level, scenario_values, usable as use
    rs = rows({"full_aware": 0.3}, n_features=12)
    ids = sorted({r["record_id"] for r in rs})
    elig = {"er": {i: True for i in ids[:10]}, "er@self_contained": {i: True for i in ids[:4]},
            "trace": {i: True for i in ids[:8]}, "trace@self_contained": {i: True for i in ids[:3]}}
    for r in rs:
        if r["record_id"] in ids[:10]:
            r.update(er=r["ca"], er_trace=r["ca"])
    copy = lambda model, **kw: [{**r, "model": model, "ca": 1 - r["ca"], **kw} for r in rs
                                if r["seed"] in ("base", "s42", "s11", "s23")]
    cl = copy("codellama")  # the ER run covers it: every eligible Full-aware row has its er
    gap = ids[5]  # eligible for ER and ER-trace, not for their self_contained tables
    mi = [{**r, "er": None, "er_trace": None} if r["record_id"] == gap else r for r in copy("mistral")]  # one gap
    other = [{**r, "variant": "lc600", "ca": 1, "er": None, "er_trace": None} for r in rs
             if r["condition"] == "full_aware"]  # another variant: never read (its missing ER would show)
    excluded = {i for i in ids if i.rsplit("|", 1)[0] == ids[0].rsplit("|", 1)[0]}  # the first scenario, out of CA
    prof_of = lambda xs, dropped=frozenset(): rq3_profile(xs, CFG, excluded, elig, dropped, ca_levels=
                                                          rq3_full_aware_by_model(xs, CFG, excluded, dropped))
    prof = prof_of(rs + cl + mi + other)
    fa = rq3_full_aware_by_model(rs + cl + mi + other, CFG, excluded)
    assert RQ3_METRICS == ("ca", "bv", "er", "er@self_contained", "er_trace", "er_trace@self_contained")
    assert list(prof) == sorted(CFG["seeds"]) and all(list(v) == list(RQ3_METRICS) for v in prof.values())
    for i, model in enumerate(sorted(CFG["seeds"])):
        key = (model, "full_aware", "main")
        mine = [r for r in rs + cl + mi if (r["model"], r["condition"]) == key[:2]]
        for j, m in enumerate(RQ3_METRICS):
            got = prof[model][m]
            if model == "mistral" and m in ("er", "er_trace"):
                continue
            want = fa[model] if m == "ca" else level(scenario_values(mine, m, use(m, excluded, elig))[key], CFG,
                                                       CFG["seed"] + 9000 + 10 * i + j)
            assert {k: got[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")} == {
                k: want[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")}, (model, m)
            assert got["reason"] is None, (model, m)
    assert prof["deepseek"]["ca"]["n_scenarios"] < prof["deepseek"]["bv"]["n_scenarios"]  # CA leaves the excluded out
    assert prof["deepseek"]["er@self_contained"]["n_scenarios"] < prof["deepseek"]["er"]["n_scenarios"]
    assert prof["codellama"]["ca"]["estimate"] != prof["deepseek"]["ca"]["estimate"]
    for m in ("er", "er_trace"):
        e = prof["mistral"][m]  # the gap has no er: no level over the rest (its self_contained tables lack it: levels)
        assert (e["n_scenarios"], e["estimate"], e["ci95"]) == (0, None, [None, None]), m
        assert "incomplete" in e["reason"], m
    gone = prof_of(rs + cl + mi, {gap})  # mistral's gap is a compose-dropped record
    assert all(gone["mistral"][m]["reason"] is None for m in RQ3_METRICS)
    assert descriptive(rs + cl + mi, CFG, excluded, elig, dropped={gap})["rq3_profile"] == gone
    none = prof_of(rs + copy("codellama", er=None, er_trace=None))
    for m in ("er", "er@self_contained", "er_trace", "er_trace@self_contained"):
        e = none["codellama"][m]  # rows, but no ER value at all: the ER run does not cover the model
        assert (e["n_scenarios"], e["estimate"]) == (0, None) and "does not cover" in e["reason"], m
    assert all(v["estimate"] is None and "Full-aware" in v["reason"] for v in none["mistral"].values())
    d = descriptive(rs + cl, CFG, excluded, elig)
    assert d["rq3_profile"] == prof_of(rs + cl)
    assert set(d["rq3_full_aware_by_model"]) == {"deepseek", "codellama"}  # the key of before, unchanged


def er_fixture(er_ids, trace_ids, er_sc, trace_sc):
    """Anchor rows with er = er_trace = ca on the records of either P11 eligibility table, and that eligibility."""
    rs = rows({"full_aware": 0.3}, n_features=12)
    ids = sorted({r["record_id"] for r in rs})
    pick = lambda sl: {i: True for i in ids[sl]}
    elig = {"er": pick(er_ids), "er@self_contained": pick(er_sc), "trace": pick(trace_ids),
            "trace@self_contained": pick(trace_sc)}
    for r in rs:
        if elig["er"].get(r["record_id"]) or elig["trace"].get(r["record_id"]):
            r.update(er=r["ca"], er_trace=r["ca"])
    return rs, ids, elig


def profile(xs, elig, excluded=frozenset(), dropped=frozenset()):
    return rq3_profile(xs, CFG, set(excluded), elig, dropped,
                       ca_levels=rq3_full_aware_by_model(xs, CFG, set(excluded), dropped))


def test_rq3_profile_leaves_an_accepted_missing_pair_out_of_the_er_denominator():
    """A pair accepted missing under P4 (rule fixed 07/10/2026), which check_complete fills with CA = BV = 0, never has
    an ER value, so it is out of the ER denominators of rq3_profile as of confirmatory: one filled Full-aware s7 row of
    the anchor on an ER-eligible record (1 of the 50 eligible rows) leaves each ER level that of the model's other rows,
    with no reason, where it read "ER incomplete ... 1 of 50" before; CA counts it as 0 (P4), through
    rq3_full_aware_by_model."""
    rs, ids, elig = er_fixture(slice(0, 10), slice(0, 10), slice(0, 4), slice(0, 4))
    gap = ids[3]  # ER-eligible, in both self_contained tables
    kept = [r for r in rs if (r["record_id"], r["condition"], r["seed"]) != (gap, "full_aware", "s7")]
    filled, info = check_complete(kept, {"deepseek": set(ids)}, CFG, {(gap, "deepseek", "full_aware", "s7")})
    assert info["missing_filled"] == 1 and sum(len(elig["er"]) for sd in SEEDS) == 50
    prof, base = profile(filled, elig)["deepseek"], profile(kept, elig)["deepseek"]
    for m in ("er", "er@self_contained", "er_trace", "er_trace@self_contained"):
        assert prof[m]["reason"] is None and prof[m]["estimate"] is not None and prof[m] == base[m], m
    lv = rq3_full_aware_by_model(filled, CFG, set())["deepseek"]
    assert prof["ca"] == {**{k: lv[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")}, "reason": None}
    real = [{**r, "er": None} if (r["record_id"], r["condition"], r["seed"]) == (gap, "full_aware", "s7") else r
            for r in rs]  # a real output with no ER value is still a gap
    assert profile(real, elig)["deepseek"]["er"]["reason"] == ("ER incomplete for this model: 1 of 50 eligible "
                                                               "Full-aware rows have no value")


def test_rq3_profile_takes_the_ca_level_of_rq3_full_aware_by_model():
    """The CA entry of rq3_profile (fixed 07/10/2026) is the very level of rq3_full_aware_by_model, the same estimate
    and the same CI, with no second bootstrap, so that the project report never prints two CIs for one number (a figure
    reads rq3_full_aware_by_model, a table rq3_profile). A model with Full-aware rows but none in CA's denominator has
    no CA level: "no row in the denominator"."""
    rs, ids, elig = er_fixture(slice(0, 10), slice(0, 10), slice(0, 4), slice(0, 4))
    excluded = {i for i in ids if i.rsplit("|", 1)[0] == ids[0].rsplit("|", 1)[0]}
    copy = lambda model, **kw: [{**r, "model": model, "ca": 1 - r["ca"], **kw} for r in rs
                                if r["seed"] in ("base", "s42", "s11", "s23")]
    mi = [{**r, "ca_defined": False} if r["condition"] == "full_aware" else r for r in copy("mistral")]
    xs = rs + copy("codellama") + mi
    d = descriptive(xs, CFG, excluded, elig)
    fa = d["rq3_full_aware_by_model"]
    assert fa == rq3_full_aware_by_model(xs, CFG, excluded) and set(fa) == {"codellama", "deepseek"}
    for model, lv in fa.items():
        assert d["rq3_profile"][model]["ca"] == {**{k: lv[k] for k in ("n_scenarios", "n_clusters", "estimate", "ci95")},
                                                 "reason": None}, model
    assert d["rq3_profile"]["mistral"]["ca"] == {"n_scenarios": 0, "n_clusters": 0, "estimate": None,
                                                 "ci95": [None, None], "reason": "no row in the denominator"}
    assert d["rq3_profile"]["mistral"]["bv"]["reason"] is None  # BV's denominator does not ask ca_defined
    other = {m: {**lv, "ci95": [0.0, 1.0]} for m, lv in fa.items()}  # read from ca_levels, never bootstrapped again
    assert rq3_profile(xs, CFG, excluded, elig, ca_levels=other)["deepseek"]["ca"]["ci95"] == [0.0, 1.0]


def test_rq3_profile_checks_each_er_metric_on_its_own_table_and_value():
    """Each ER metric checks for gaps on its own P11 eligibility table and its own value (fixed 07/10/2026). The trace
    table is the wider one in the real data (gold's trace_action 1 and stub's 0, against gold PASS and stub not PASS),
    and an output may have its er and lack its er_trace. A model the ER run covers (some er or er_trace value on its
    real Full-aware rows) is never said not covered: a table whose eligible rows all lack a value reads "incomplete"
    with its counts. A table with no eligible row has "no row in the denominator"."""
    rs, ids, elig = er_fixture(slice(0, 4), slice(0, 8), slice(0, 2), slice(6, 8))
    hole = ids[6]  # trace-eligible (and trace@self_contained), not ER-eligible: its er is there, its er_trace is not
    rs = [{**r, "er_trace": None} if r["record_id"] == hole else r for r in rs]
    ds = profile(rs, elig)["deepseek"]
    assert ds["er"]["reason"] is None and ds["er@self_contained"]["reason"] is None
    assert ds["er_trace"]["reason"] == "ER incomplete for this model: 5 of 40 eligible Full-aware rows have no value"
    assert ds["er_trace@self_contained"]["reason"] == ("ER incomplete for this model: 5 of 10 eligible Full-aware rows "
                                                       "have no value")
    three = lambda model, **kw: [{**r, "model": model, **kw} for r in rs if r["seed"] in ("base", "s42", "s11", "s23")]
    mi = [{**r, "er": None, "er_trace": None} if r["record_id"] in ids[:2] else r for r in three("mistral")]
    cl = three("codellama", er_trace=None)  # its er is there, no er_trace at all
    p = profile(rs + mi + cl, elig)
    assert p["mistral"]["er"]["reason"] == "ER incomplete for this model: 6 of 12 eligible Full-aware rows have no value"
    assert p["mistral"]["er@self_contained"]["reason"] == ("ER incomplete for this model: 6 of 6 eligible Full-aware "
                                                           "rows have no value")
    assert p["mistral"]["er_trace"]["reason"] == "ER incomplete for this model: 9 of 24 eligible Full-aware rows have no value"
    assert p["mistral"]["er_trace@self_contained"]["reason"] == ("ER incomplete for this model: 3 of 6 eligible "
                                                                 "Full-aware rows have no value")
    assert p["codellama"]["er"]["reason"] is None
    assert p["codellama"]["er_trace"]["reason"] == ("ER incomplete for this model: 24 of 24 eligible Full-aware rows "
                                                    "have no value")
    assert not any("does not cover" in (v["reason"] or "") for m in ("mistral", "codellama") for v in p[m].values())
    tr = three("codellama", er=None)  # its er_trace is there, no er at all: the ER run covers it all the same
    assert profile(rs + tr, elig)["codellama"]["er"]["reason"] == ("ER incomplete for this model: 12 of 12 eligible "
                                                                     "Full-aware rows have no value")
    gone = {k: v for k, v in elig.items() if k != "trace@self_contained"}
    assert profile(rs, gone)["deepseek"]["er_trace@self_contained"] == {
        "n_scenarios": 0, "n_clusters": 0, "estimate": None, "ci95": [None, None], "reason": "no row in the denominator"}


def test_kappa_all_row_pools_the_anchor_model_only():
    """P16 (fixed 07/10/2026, before any test output): the ER run of the test partition also covers CodeLlama and
    Mistral at Full-aware, for the RQ3 profile (P14). Their runs get their own rows in the CA x ER-trace agreement,
    while the pooled row "ALL" stays over the anchor model's runs, as P16 was designed: a registered measure keeps its
    meaning."""
    mk = lambda model, rid, ca, ta, d=0: {"record_id": rid, "model": model, "condition": "full_aware", "variant": "main",
                                          "seed": "s42", "ca": ca, "ca_defined": True, "trace_action": ta,
                                          "php_docblock_steps": d}
    elig = {"trace": {"a": True, "b": True}, "trace@self_contained": {"a": True}}
    anchor = [mk("deepseek", "a", 1, 1), mk("deepseek", "b", 0, 0)]
    more = anchor + [mk("codellama", "a", 1, 0), mk("mistral", "b", 0, 1), mk("mistral", "a", 0, 1)]
    one, two = kappa_ca_trace(anchor, set(), elig, "deepseek"), kappa_ca_trace(more, set(), elig, "deepseek")
    for k in one:
        assert two[k]["ALL"] == one[k]["ALL"], k
    assert two["trace"]["ALL"] == {**two["trace"]["ALL"], "n": 2, "agreement": 1.0}
    assert two["trace"]["codellama|full_aware|main|s42"]["n"] == 1 and two["trace"]["mistral|full_aware|main|s42"]["n"] == 2
    assert agreement_ca_trace(more, set(), elig["trace"], "deepseek") == two["trace"]
    assert "ALL" not in agreement_ca_trace(more[2:], set(), elig["trace"], "deepseek")
