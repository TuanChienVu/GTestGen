import json

import pytest

from tests_gpu.tiny import make_tiny_llama
from v2gpu import canary
from v2gpu.serialize import encode

RECS = [{"id": f"r{i:02d}", "background": "", "scenario": f"Scenario: s{i}\n  Given step {i}",
         "create": [{"fn_id": f"F.php:{i}", "code": f"#[Given('step {i}')]\npublic function step{i}(): void\n{{\n"
                                                    f"    $this->x = {i};\n}}"}]} for i in range(40)]


@pytest.mark.parametrize("target", [256, 512])
def test_synthetic_examples_land_just_under_the_target(tokenizers, target):
    tok = tokenizers["codellama"]
    items = canary.synthetic(RECS, tok, target, 5)
    assert len({cr["id"] for cr, _, _ in items}) == 5
    for cr, messages, answer in items:
        assert target - 64 < len(encode(tok, messages, answer, target).input_ids) <= target
        assert cr["partition"] == "train" and not canary._leaks(cr, messages)


def test_synthetic_dev_items_come_from_other_records(tokenizers):
    tok = tokenizers["codellama"]
    train = canary.synthetic(RECS, tok, 256, 5)
    dev = canary.synthetic(RECS, tok, 256, 2, "d", offset=5)
    assert {a for _, _, a in train}.isdisjoint({a for _, _, a in dev})
    assert {cr["partition"] for cr, _, _ in dev} == {"dev"}


def test_a_filler_holding_the_method_name_is_never_used(tokenizers):
    calls = [dict(r, create=[dict(r["create"][0], code=r["create"][0]["code"].replace("$this->x = 1;", "$this->step0();"))])
             if r["id"] == "r01" else r for r in RECS]
    items = canary.synthetic(calls, tokenizers["codellama"], 256, 5)
    assert not any("s0\n" in cr["scenario"] for cr, _, _ in items)       # r00's filler would hold r01's call to step0
    assert not any(canary._leaks(cr, m) for cr, m, _ in items)


def test_synthetic_refuses_when_too_few_records_fit(tokenizers):
    with pytest.raises(ValueError):
        canary.synthetic(RECS[:2], tokenizers["codellama"], 256, 5)


def test_projection_on_length_buckets():
    rates = {1024: 1000.0, 2048: 1000.0, 4096: 500.0, 8192: 250.0}
    buckets = {"1024": 1.8, "2048": 1.8, "4096": 1.8, "8192": 1.8}   # 0.5 h + 0.5 h + 1 h + 2 h = 4 h
    p = canary.project(rates, buckets, 2.0)
    assert p["hours_expected"] == 4.6 and p["usd_expected"] == 9.2            # +15% overhead
    assert p["hours_conservative"] == 5.75 and p["usd_ceiling_5_epochs"] == pytest.approx(19.17, abs=0.01)


def good(total=80.0, peak=10.0, flash=True, attn="sdpa", status="MAX_STEPS"):
    return {"status": status, "environment": {"device": {"total_memory_gb": total}},
            "model_load": {"attn_implementation_loaded": attn, "flash_kernel_eligible": flash},
            "result": {"peak_reserved_gb": peak, "tokens_per_second": 1000.0, "global_step": 24}}


RESUME = dict(good(), _n=12)
SMOKE = {"pass": True}


def test_verdict_pass_and_each_failure():
    ok = {1024: good(), 8192: good()}
    assert canary.verdict(ok, RESUME, ("a", "a"), SMOKE, "sdpa")["pass"]
    for runs, resume, reload, smoke in [
        ({1024: good(), 8192: good(attn="eager")}, RESUME, ("a", "a"), SMOKE),
        ({1024: good(), 8192: good(flash=False)}, RESUME, ("a", "a"), SMOKE),
        ({1024: good(), 8192: good(peak=75.0)}, RESUME, ("a", "a"), SMOKE),
        ({1024: good(), 8192: good(status="CRASHED")}, RESUME, ("a", "a"), SMOKE),
        (ok, dict(RESUME, _n=10), ("a", "a"), SMOKE),
        (ok, None, ("a", "a"), SMOKE),
        (ok, RESUME, ("a", "b"), SMOKE),
        (ok, RESUME, ("a", "a"), {"pass": False}),
    ]:
        assert not canary.verdict(runs, resume, reload, smoke, "sdpa")["pass"]


def test_canary_end_to_end_on_cpu(tmp_path, tokenizers):
    tiny = make_tiny_llama(tmp_path / "tiny", len(tokenizers["codellama"]))
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECS))
    plan = {"epochs_expected": 3, "epochs_max": 5, "models": {"codellama": {"192": 45.0, "320": 45.0}}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    code = canary.main(["--records", str(tmp_path / "train.jsonl"), "--out", str(tmp_path / "c"),
                        "--models", "codellama", "--lengths", "192", "320", "--resume-length", "192", "--steps", "2",
                        "--compare-upcast", "codellama", "--upcast-steps", "12", "--smoke-cap", "4",
                        "--model-path", tiny, "--quant", "none", "--dtype", "float32",
                        "--plan-tokens", str(tmp_path / "plan.json"), "--price", "1.61"])
    r = json.loads((tmp_path / "c" / "canary.json").read_text())
    assert code == 0 and r["pass"], r
    m = r["models"]["codellama"]
    assert m["criteria"]["gpu_resume_reaches_2n"] is True and m["criteria"]["smoke_codepath"] is True
    assert m["criteria"]["flash_kernel_eligible"] == "n/a"   # CPU
    assert r["upcast_comparison"]["steps"] == [12, 12, 12] and r["upcast_comparison"]["flag_for_review"] is False
    assert r["upcast_comparison"]["step0"]["numeric_difference"] == 0.0     # CPU fp32 run: the upcast is a no-op
    assert m["projection"]["usd_ceiling_5_epochs"] > m["projection"]["usd_conservative"] > m["projection"]["usd_expected"]
    assert r["total"]["usd_expected"] == m["projection"]["usd_expected"]


def run_canary(tmp_path, tokenizers, extra=()):
    tiny = tmp_path / "tiny"
    if not tiny.exists():
        make_tiny_llama(tiny, len(tokenizers["codellama"]))
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in RECS))
    return canary.main(["--records", str(tmp_path / "train.jsonl"), "--out", str(tmp_path / "c"), "--models",
                        "codellama", "--lengths", "192", "--resume-length", "192", "--steps", "2", "--smoke-cap", "4",
                        "--model-path", str(tiny), "--quant", "none", "--dtype", "float32", *extra])


def test_a_failed_upcast_comparison_is_reported_not_fatal(tmp_path, tokenizers, monkeypatch):
    real = canary._train

    def fp32_dies(d, key, L, steps, a, workdir="run", extra=()):
        return {"status": "CRASHED"} if workdir == "fp32" else real(d, key, L, steps, a, workdir, extra)

    monkeypatch.setattr(canary, "_train", fp32_dies)
    code = run_canary(tmp_path, tokenizers, ["--compare-upcast", "codellama", "--upcast-steps", "12"])
    r = json.loads((tmp_path / "c" / "canary.json").read_text())
    assert code == 0 and r["pass"] and r["models"]["codellama"]["pass"]
    assert "did not finish" in r["upcast_comparison"]["error"] and r["upcast_comparison"]["flag_for_review"] is True


def test_a_model_that_crashes_still_leaves_a_report(tmp_path, tokenizers, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr("v2gpu.modeling.reload_roundtrip", boom)
    assert run_canary(tmp_path, tokenizers) == 1
    r = json.loads((tmp_path / "c" / "canary.json").read_text())
    assert r["pass"] is False and "out of memory" in r["models"]["codellama"]["error"]


def test_upcast_rule_uses_paired_steps_twice_the_noise_and_the_margin():
    flat = [2.0] * 12
    assert canary.upcast_verdict(flat, flat, [2.0] * 11 + [2.001])["flag_for_review"] is False   # under the 0.5% margin
    assert canary.upcast_verdict(flat, flat, [1.9] * 12)["flag_for_review"] is True
    noisy = [2.0] * 11 + [2.5]                            # paired noise over the last 10 steps: mean |diff| = 0.05
    v = canary.upcast_verdict(flat, noisy, [2.0] + [2.08] * 11)      # step 0: identical weights, identical loss
    assert v["noise_floor"] == pytest.approx(0.05) and v["flag_for_review"] is False            # within 2 x noise
    assert canary.upcast_verdict(flat, [], flat)["flag_for_review"] is True                      # a missing run


def test_upcast_rule_flags_a_step0_difference_too():
    base = [2.0] * 12
    v = canary.upcast_verdict(base, base, [2.1] + [2.0] * 11)       # identical weights, 5% apart at step 0
    assert v["step0"]["flag"] is True and v["flag_for_review"] is True
    v = canary.upcast_verdict(base, [2.3] + [2.0] * 11, [2.1] + [2.0] * 11)   # step 0 is that noisy anyway
    assert v["step0"]["flag"] is False and v["flag_for_review"] is False


def test_upcast_rule_flags_an_incomplete_curve_when_steps_given():
    full = [2.0] * 12
    truncated = full[:10]                    # a run cut short (e.g. by --max-steps mismatch, or a partial crash)
    v = canary.upcast_verdict(full, full, truncated, steps=12)
    assert v["flag_for_review"] is True and "incomplete loss curve" in v["error"]
    assert canary.upcast_verdict(full, full, full, steps=12)["flag_for_review"] is False
    assert canary.upcast_verdict(full, full, full)["flag_for_review"] is False   # steps=None: no length check at all


def test_paired_compares_the_common_prefix_when_curves_differ_in_length():
    """canary.py:239-243 (pre-fix): _paired compared a[-n:] with b[-n:] independently, pairing different steps when the
    two curves have different lengths. It must instead pair on the shared prefix."""
    a = [1.0] * 5 + [100.0]      # length 6: an extra step that must not leak into the comparison
    b = [1.0] * 5                # length 5
    assert canary._paired(a, b) == 0.0


def test_the_loss_reader_splits_lines_on_newline_only(tmp_path):
    """decision 24/09: every JSONL reader of A2 splits on b"\\n" only, the metrics readers too."""
    rows = [{"step": 1, "loss": 0.5, "note": "a\u2028b\x85c"}, {"epoch_end": 0}]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
    assert canary._losses(tmp_path) == [0.5]
